from model import model_function
from parse_args import parse_args

def main():
    args = parse_args()

    if args.model == "llama":
        MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif args.model == "Qwen2B":
        MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    elif args.model == "Qwen8B":
        MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    else:
        raise KeyError("Only can enter llama, Qwen2B or Qwen8B")

    func = model_function(MODEL_ID)
    
    func.set_seed(args.seed)

    image_paths, images = func.load_images(args.folder)

    model, processor = func.build_model()

    messages = func.build_messages(len(images))
    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=images,
        text=input_text,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=False,
        num_beams=1,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output[0][prompt_length:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    print("Input images:")
    for path in image_paths:
        print(f"- {path}")

    print("\nModel output:")
    print(response.strip())


if __name__ == "__main__":
    main()
