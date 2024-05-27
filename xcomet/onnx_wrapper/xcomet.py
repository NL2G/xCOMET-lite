# coding: utf-8

import os
from tqdm import tqdm
import time
from collections import namedtuple
from itertools import chain

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import onnxruntime as ort

import comet
from comet.models.utils import Prediction
from comet import download_model, load_from_checkpoint

from .utils import logger


class OnnxModel:
    """
    Base class for wrapping onnx model
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool,
        use_trt: bool,
        one_thread: bool
    ):
        sess_options = ort.SessionOptions()
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        if one_thread:
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
        if use_gpu:
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            if use_trt:
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers=['CPUExecutionProvider']

        logger.info('Load ONNX model...')
        self.ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
    def forward(
        self,
        batch: Dict[str, Union[np.ndarray, torch.Tensor, List[Tuple[int]]]]
    ) -> List:
        batch = {input_field.name: batch[input_field.name] for input_field in self.ort_session.get_inputs()}
        if isinstance(batch[next(iter(batch))], (torch.Tensor, list)):
            batch  = self.to_onnx_format(batch)
        result = self.ort_session.run(None, batch)
        return result

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def to_onnx_format(self, batch):
        return {key: val if isinstance(val, list) else self.to_numpy(val) for key, val in batch.items()}

    def collate_fn(self):
        pass


class OnnxXCOMETModel(OnnxModel):
    """
    XCOMET onnx model with support of torch functions
    """

    def __init__(
        self,
        onnx_model_path: str,
        xcomet_model: Union[comet.models.multitask.xcomet_metric.XCOMETMetric, str] = None,
        use_gpu: bool = False,
        one_thread: bool = True
    ):
        super().__init__(onnx_model_path, use_gpu, one_thread)

        logger.info('Load xCOMET model')
        self.xcomet_model = xcomet_model
        if isinstance(xcomet_model, str):
            self.xcomet_model = load_from_checkpoint(
                download_model(xcomet_model)
            )

    def collate_fn(self, batch):
        return self.xcomet_model.prepare_sample(batch, stage='predict')


class OnnxXCOMETMetric:
    """
    Onnx version of XCOMET metric
    """

    OnnxPrediction = namedtuple('OnnxPrediction', ('score', 'logits'))

    def __init__(
        self,
        model: OnnxXCOMETModel,
        reference_free: bool = False
    ):
        self.model = model
        self.reference_free = reference_free

    def predict(
            self,
            dataset: List[Dict[str, str]],
            batch_size: int,
            num_workers: int = 2,
            **kwards
    ):
        if self.reference_free:
            dataset.pop('ref', None)

        loader = self.get_dataloader(dataset, batch_size, num_workers)

        latency = []
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader):
                start = time.perf_counter()
                predictions.append(self.batch_predict(batch))
                latency.append(time.perf_counter()-start)
        logger.info(f'{self.__class__.__name__} inference took {np.sum(latency).round(2)}s')
        scores = torch.cat([p.scores for p in predictions])
        if self.reference_free:
            metadata = Prediction(
                src_scores=torch.cat([p.metadata.src_scores for p in predictions]),
                mqm_scores=torch.cat([p.metadata.mqm_scores for p in predictions]),
                error_spans=list(chain.from_iterable([p.metadata.error_spans for p in predictions])),
            )
        else:
            metadata = Prediction(
                src_scores=torch.cat([p.metadata.src_scores for p in predictions]),
                ref_scores=torch.cat([p.metadata.ref_scores for p in predictions]),
                unified_scores=torch.cat([p.metadata.unified_scores for p in predictions]),
                mqm_scores=torch.cat([p.metadata.mqm_scores for p in predictions]),
                error_spans=list(chain.from_iterable([p.metadata.error_spans for p in predictions])),
            )
        return Prediction(
            scores=scores,
            system_score=scores.mean().item(),
            metadata=metadata
        )

    def get_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.model.collate_fn, num_workers=num_workers
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        result = self.model.forward({'input_ids': input_ids, 'attention_mask': attention_mask})
        result = self.OnnxPrediction(*map(torch.from_numpy, result))
        return result

    def batch_predict(self, batch):

        def _compute_mqm_from_spans(error_spans):
            scores = []
            for sentence_spans in error_spans:
                sentence_score = 0
                for annotation in sentence_spans:
                    if annotation["severity"] == "minor":
                        sentence_score += 1
                    elif annotation["severity"] == "major":
                        sentence_score += 5
                    elif annotation["severity"] == "critical":
                        sentence_score += 10

                if sentence_score > 25:
                    sentence_score = 25

                scores.append(sentence_score)

            # Rescale between 0 and 1
            scores = (torch.tensor(scores) * -1 + 25) / 25
            return scores

        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Regression scores are weighted with self.score_weights
            regression_scores = torch.stack(
                [
                    torch.where(pred.score > 1.0, 1.0, pred.score) * w
                    for pred, w in zip(predictions, self.model.xcomet_model.score_weights[:3])
                ],
                dim=0,
            ).sum(dim=0)
            mt_mask = batch[0]['label_ids'] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()

            # Weighted average of the softmax probs along the different inputs.
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                for w, o in zip(self.model.xcomet_model.input_weights_spans, predictions)
            ]
            subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
            error_spans = self.model.xcomet_model.decode(
                subword_probs, batch[0]['input_ids'], batch[0]['mt_offsets']
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_scores
                + mqm_scores.to(regression_scores.device) * self.model.xcomet_model.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                ),
            )

        # XCOMET if reference is not available we fall back to QE model.
        else:
            model_output = self.forward(batch[0])
            regression_score = torch.where(
                model_output.score > 1.0, 1.0, model_output.score
            )
            mt_mask = batch[0]['label_ids'] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                :, :seq_len, :
            ]
            error_spans = self.model.xcomet_model.decode(
                subword_probs, batch[0]['input_ids'], batch[0]['mt_offsets']
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_score * sum(self.model.xcomet_model.score_weights[:3])
                + mqm_scores.to(regression_score.device) * self.model.xcomet_model.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=regression_score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                ),
            )
        return batch_prediction
