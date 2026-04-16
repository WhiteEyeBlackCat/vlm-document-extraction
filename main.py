from model import model_function, resolve_model_id
from parse_args import parse_args


def main():
    args = parse_args()
    model_id = resolve_model_id(args.model)
    runner = model_function(model_id)
    result = runner.run_inference(
        args.folder,
        args.max_tokens,
        args.seed,
        quantization=args.quantization,
    )

    print("Input images:")
    for path in result["image_paths"]:
        print(f"- {path}")

    print(f"\nModel: {result['model_id']}")
    print(f"Device: {result['device']}")
    print(f"Quantization: {result['quantization']}")
    print("\nModel output:")
    print(result["response"])


if __name__ == "__main__":
    main()
