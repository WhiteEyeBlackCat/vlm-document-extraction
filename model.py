from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from prompt import Prompt


MODEL_ID_MAP = {
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen2B": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen8B": "Qwen/Qwen3-VL-8B-Instruct",
}


class model_function:
    def __init__(self, model_id):
        self.model_id = model_id

    def build_model(self):
        if self.model_id.startswith("meta-llama"):
            model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        elif self.model_id.startswith("Qwen"):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise ValueError("Only Qwen and llama are available")

        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_images(self, folder: Path):
        preferred_order = [
            "page-1.jpg",
        ]

        image_paths = [folder / name for name in preferred_order if (folder / name).exists()]
        if not image_paths:
            image_paths = sorted(folder.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(f"No JPG files found in {folder}")

        images = [Image.open(path).convert("RGB") for path in image_paths]
        return image_paths, images

    def build_messages(self, image_count: int):
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": Prompt})
        return [{"role": "user", "content": content}]

    def run_inference(self, folder: Path, max_tokens: int, seed: int):
        self.set_seed(seed)
        image_paths, images = self.load_images(folder)

        model, processor = self.build_model()
        messages = self.build_messages(len(images))
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
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
        )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output[0][prompt_length:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "image_paths": image_paths,
            "response": response,
            "device": str(model.device),
            "model_id": self.model_id,
        }


def resolve_model_id(model_name: str) -> str:
    try:
        return MODEL_ID_MAP[model_name]
    except KeyError as exc:
        raise KeyError("Only can enter llama, Qwen2B or Qwen8B") from exc
