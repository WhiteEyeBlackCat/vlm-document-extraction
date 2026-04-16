from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser(
        description="Run LLaMA Vision on your preprocessed local document images."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "Qwen2B", "Qwen8B"],
        required=True,
        help="Choice backbone VLM model. Only can enter llama, Qwen2B or Qwen8B"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder that contains preprocessed JPGs such as page-1.jpg.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic generation.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional bitsandbytes quantization mode for inference.",
    )
    return parser.parse_args()
