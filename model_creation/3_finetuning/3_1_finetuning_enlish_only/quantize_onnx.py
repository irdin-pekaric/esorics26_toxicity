import os
import argparse
import sys
import shutil

INPUT_ONNX: str = "output/onnx_ft/toxic_bert_finetuned.onnx"

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception as e:
    print("Error importing onnxruntime.quantization:", e)
    print("Make sure onnxruntime (>=1.8.0) is installed:")
    print("  pip install onnxruntime onnx")
    sys.exit(2)


WEIGHT_TYPE_MAP = {
    "qint8": QuantType.QInt8,
    "quint8": QuantType.QUInt8,
    "qint16": QuantType.QInt16,
}


def quantize_model(input_onnx: str, output_dir: str, weight_type: str):
    if not os.path.isfile(input_onnx):
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx}")

    os.makedirs(output_dir, exist_ok=True)
    model_dst = os.path.join(output_dir, "model.onnx")
    quant_dst = os.path.join(output_dir, "model_quantized.onnx")

    shutil.copy2(input_onnx, model_dst)

    qt = WEIGHT_TYPE_MAP.get(weight_type.lower())
    if qt is None:
        raise ValueError(
            f"Invalid weight_type: {weight_type}. Valid options: {', '.join(WEIGHT_TYPE_MAP.keys())}"
        )

    print(f"Quantizing {model_dst} -> {quant_dst} with weight_type={weight_type}")
    quantize_dynamic(model_input=model_dst, model_output=quant_dst, weight_type=qt)
    print("Done. Quantized model saved to:", quant_dst)
    return model_dst, quant_dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamically quantize an ONNX model and save it into 'for_extension' by default."
    )
    parser.add_argument("input", nargs="?", help="Path to the ONNX file (optional if INPUT_ONNX is set in the file)")
    parser.add_argument("--output-dir", default="for_extension", help="Target folder (default: for_extension)")
    parser.add_argument(
        "--weight-type",
        default="qint8",
        choices=list(WEIGHT_TYPE_MAP.keys()),
        help="Weight type for quantization",
    )
    return parser.parse_args()


def main():
    if INPUT_ONNX:
        input_path = INPUT_ONNX
        output_dir = os.environ.get("QUANT_OUTPUT_DIR", "output/for_extension")
        weight_type = os.environ.get("QUANT_WEIGHT_TYPE", "qint8")
    else:
        args = parse_args()
        if not args.input:
            print("No input file provided. Either set INPUT_ONNX inside the file or provide the path as a CLI argument.")
            sys.exit(2)
        input_path = args.input
        output_dir = args.output_dir
        weight_type = args.weight_type

    try:
        model_path, quant_path = quantize_model(input_path, output_dir, weight_type)
        in_size = os.path.getsize(input_path) if os.path.exists(input_path) else None
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else None
        quant_size = os.path.getsize(quant_path) if os.path.exists(quant_path) else None
        if in_size and model_size and quant_size:
            saved = in_size - quant_size
            pct = (saved / in_size) * 100 if in_size > 0 else 0
            print(f"Input size: {in_size} bytes, Copied model: {model_size} bytes, Quantized: {quant_size} bytes, Saved: {saved} bytes ({pct:.2f}%)")
    except Exception as e:
        print("Error during quantization:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
