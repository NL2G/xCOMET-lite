# coding: utf-8

import argparse
import time
from typing import Dict, Union
import numpy as np

import torch
import onnx
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import neural_compressor
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

from .utils import logger
from .xcomet import OnnxModel


class SQuantizer:

    def __init__(
        self,
        quantize_params: Dict,
        loader: torch.utils.data.dataloader.DataLoader,
        shuffling: bool = False
    ):
        self.sq_config = PostTrainingQuantConfig(
            approach='dynamic', domain='nlp',
            calibration_sampling_size=[self.calibration_sampling_size],
            quant_level='auto', diagnosis=False,
            recipes={
                'smooth_quant': True,
                'smooth_quant_args': {
                    'alpha': quantize_params['smooth_alpha'],
                    'folding': quantize_params['folding'],
                }
            }
        )
        self.calibration_sampling_size = quantize_params['calibration_sampling_size']
        self.loader = loader
        self.shuffling = shuffling

    def quantize(self, onnx_input: str, onnx_output: str):
        sq_model = quantization.fit(
            onnx.load(onnx_input),
            self.sq_config,
            calib_dataloader=self.loader,
            eval_func=None,
        )
        sq_model.save(onnx_output)


class DynamicQuantizer:

    def __init__(self, quantize_params: Dict):
        self.quantize_params = quantize_params

    def quantize(self, onnx_input: str, onnx_output: str):

        quantize_dynamic(
            model_input=onnx_input,
            model_output=onnx_output,
            per_channel=self.quantize_params['per_channel'],
            reduce_range=self.quantize_params['reduce_range'],
            weight_type=self.quantize_params['weight_type'],
            extra_options=self.quantize_params['extra_options']
        )


class OnnxQuantizer:

    def __init__(
        self,
        model: str,
        quantized_model: OnnxModel,
        quantizer: Union[SQuantizer, DynamicQuantizer],
        dataset: str
    ):

        self.model = model
        self.quantized_model = quantized_model
        self.quantizer = quantizer

    def quantize(self):
        self.quantizer.quantize(self.model, self.quantized_model)
        onnx.checker.check_model(self.quantized_model)

    def calculate_metrics(self):
        pass


def onnx_optimize(
        onnx_input: str,
        onnx_output: str,
        use_gpu: bool = False,
        model_type: str = 'bert'
    ):
    opt_options = FusionOptions(model_type)
    opt_options.enable_embed_layer_norm = False
    opt_options.enable_gemm_fast_gelu = True
    start = time.perf_counter()
    model_optimizer = optimizer.optimize_model(
        input=onnx_input,
        model_type=model_type,
        optimization_options=opt_options,
        use_gpu=use_gpu
    )
    logger.info(f'ONNX optimization took {time.perf_counter()-start} s')
    model_optimizer.save_model_to_file(onnx_output, use_external_data_format=True)
    onnx.checker.check_model(onnx_output)

def onnx_preprocess(
        onnx_input: str,
        onnx_output: str,
    ):
    quant_pre_process(onnx_input, onnx_output, skip_symbolic_shape=True, skip_onnx_shape=True)
    onnx.checker.check_model(onnx_output)
