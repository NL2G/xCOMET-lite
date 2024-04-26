import os
from argparse import ArgumentParser

import onnx
from onnxruntime.quantization.quantize import QuantType
from onnxconverter_common import auto_mixed_precision, float16

from comet import download_model, load_from_checkpoint

from onnx_wrapper.utils import xcomet_to_onnx, logger
from onnx_wrapper.optimizer import (
    onnx_optimize, onnx_preprocess,
    DynamicQuantizer, SQuantizer,
    OnnxQuantizer
)


def make_parser():
    parser = ArgumentParser(description="ONNX xCOMET optimizer")
    parser.add_argument("--onnx_path", help="Directory with ONNX .onnx file. If doesn't exist, create it", required=True)
    parser.add_argument("--model", help="Which model to use (name on huggingface)")
    parser.add_argument("--lp", help="On which language pair to compute metrics")
    parser.add_argument("--dataset", help="Which dataset to use (huggingface dataset/path to tsv file)")
    parser.add_argument("--domain", default="news", help="On which domain to compute metrics")
    parser.add_argument("--year", type=int, default=2022, help="In which year to compute metrics")
    parser.add_argument(
        "--opt", action="store_true", default=False,
        help=(
            "Optimizations are basically of three kinds:\n"
            "- Constant Folding: Convert static variables to constants in the graph\n"
            "- Deadcode Elimination: Remove nodes never accessed in the graph\n"
            "- Operator Fusing: Merge multiple instruction into one (Linear -> ReLU can be fused to be LinearReLU)"
        )
    )
    parser.add_argument(
        "--dint8", action="store_true", default=False,
        help="Whether dynamically quantize to INT8 data-freely"
    )
    parser.add_argument(
        "--sq", action="store_true", default=False,
        help="Whether use SmoothQuant approach"
    )
    parser.add_argument(
        "--half", action="store_true", default=False,
        help="To fp16"
    )
    parser.add_argument(
        "--amp", action="store_true", default=False,
        help="Automatic mixed precision conversion"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="Whether use GPU for operations"
    )
    return parser

def create_model(args):
    logger.info("Creating model...")
    if os.path.exists(args.onnx_path):
        return
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    xcomet_to_onnx(model, args.onnx_path)


def main():
    parser = make_parser()
    args = parser.parse_args()
    print(args)

    model_path = args.onnx_path
    create_model(args)
    if args.opt:
        logger.info("Optimize model...")
        onnx_optimize(model_path, model_path.replace(".onnx", ".opt.onnx"), args.gpu)
    if args.half:
        logger.info("Convert to mixed precision...")
        model_fp16 = float16.convert_float_to_float16(
            onnx.load(model_path),
            disable_shape_infer=True
        )
        onnx.save(
            model_fp16, model_path.replace(".onnx", ".half.onnx"),
            save_as_external_data=True, all_tensors_to_one_file=True,
            size_threshold=1024, convert_attribute=False, location='external_data'
        )
        model_path = model_path.replace(".onnx", ".half.onnx")
    if args.amp:
        logger.info("Use automatic mixed precision...")
        pass
    if args.dint8 or args.sq:
        logger.info("Preprocess model...")
        onnx_preprocess(model_path, model_path.replace(".onnx", ".preproc.onnx"))
        model_path = model_path.replace(".onnx", ".preproc.onnx")
        if args.dint8:
            params = {
                "per_channel": True,
                "reduce_range": False,
                "weight_type": QuantType.QInt8,
                "extra_options": {"ActivationSymmetric": False, "WeightSymmetric": True}
            }
            logger.info(f"Quantize: DQ ({params})")
            quantizer = DynamicQuantizer(params)
            suffix = ".dint8.per_channel.QInt8.onnx"
        elif args.sq:
            params = {
                "smooth_alpha": 0.5,
                "folding": False,
                "calibration_sampling_size": 100,
            }
            logger.info(f"Quantize: SmoothQuant ({params})")
            quantizer = SQuantizer(params, None)
            suffix = ".sq.alpha_0.5.calib_size_100.onnx"
        OnnxQuantizer(model_path, model_path).quantize(
            model_path,
            model_path.replace(".onnx", suffix),
            quantizer,
            None
        )


if __name__ == "__main__":
    main()
