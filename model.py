from pathlib import Path
import importlib.util
from io import BytesIO
import subprocess

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from prompt import build_extraction_prompt


MODEL_ID_MAP = {
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen2B": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen8B": "Qwen/Qwen3-VL-8B-Instruct",
}


class model_function:
    def __init__(self, model_id):
        self.model_id = model_id

    def _build_quantization_config(self, quantization: str):
        if quantization == "none":
            return None

        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "bitsandbytes is not installed. Install it first to use 4bit or 8bit quantization."
            )

        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)

        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        raise ValueError("Only none, 8bit, and 4bit quantization are available")

    def build_model(self, quantization: str = "none", gpu: str = "auto"):
        device_map = "auto" if gpu == "auto" else f"cuda:{gpu.lstrip('cuda:')}"
        model_kwargs = {
            "device_map": device_map,
        }

        if self.model_id.startswith("meta-llama") and quantization != "none":
            raise ValueError(
                "bitsandbytes quantization is currently disabled for "
                "meta-llama/Llama-3.2-Vision models in this app because "
                "the combination can fail inside bitsandbytes. "
                "Use --quantization none, or switch to a Qwen model."
            )

        quantization_config = self._build_quantization_config(quantization)
        if quantization_config is None:
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["quantization_config"] = quantization_config

        if self.model_id.startswith("meta-llama"):
            model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        elif self.model_id.startswith("Qwen"):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        else:
            raise ValueError("Only Qwen and llama are available")

        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_pdf_page_count(self, pdf_path: Path):
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"{pdf_path} does not exist.")

        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())

        raise ValueError(f"Could not determine page count for {pdf_path}")

    def _load_pdf_images(self, pdf_path: Path):
        page_count = self._get_pdf_page_count(pdf_path)
        image_labels = []
        images = []
        preview_images = []

        for page_index in range(1, page_count + 1):
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page_index),
                    "-l",
                    str(page_index),
                    str(pdf_path),
                ],
                check=True,
                capture_output=True,
            )

            if not result.stdout:
                raise ValueError(
                    f"pdftoppm returned empty output for {pdf_path} page {page_index}"
                )

            image_labels.append(f"{pdf_path.name}#page-{page_index}")
            with Image.open(BytesIO(result.stdout)) as image:
                rgb_image = image.convert("RGB")
                images.append(rgb_image.copy())

                preview_buffer = BytesIO()
                rgb_image.save(preview_buffer, format="JPEG", quality=90)
                preview_images.append(preview_buffer.getvalue())

        return image_labels, images, preview_images

    @staticmethod
    def _resize_for_model(image: Image.Image, max_size: int = 1024) -> Image.Image:
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        return image

    def load_images(self, input_path: Path):
        if input_path.suffix.lower() == ".pdf":
            image_labels, images, preview_images = self._load_pdf_images(input_path)
            images = [self._resize_for_model(img) for img in images]
            return image_labels, images, preview_images

        # Direct image file (.jpg / .jpeg / .png / .tiff / .tif / .webp)
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
        if input_path.suffix.lower() in IMAGE_EXTS and input_path.is_file():
            img = self._resize_for_model(ImageOps.exif_transpose(Image.open(input_path)).convert("RGB"))
            return [input_path], [img], None

        # Legacy: folder containing jpg files
        folder = input_path
        preferred_order = ["page-1.jpg"]
        image_paths = [folder / name for name in preferred_order if (folder / name).exists()]
        if not image_paths:
            image_paths = sorted(folder.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(p for ext in IMAGE_EXTS for p in folder.glob(f"*{ext}"))
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {folder}")

        images = [self._resize_for_model(ImageOps.exif_transpose(Image.open(path)).convert("RGB")) for path in image_paths]
        return image_paths, images, None

    def build_messages(self, image_count: int, document_context: str | None = None):
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": build_extraction_prompt(document_context)})
        return [{"role": "user", "content": content}]

    def run_inference(
        self,
        input_path: Path,
        max_tokens: int,
        seed: int,
        quantization: str = "none",
        gpu: str = "auto",
    ):
        self.set_seed(seed)
        image_paths, images, preview_images = self.load_images(input_path)

        model, processor = self.build_model(quantization=quantization, gpu=gpu)
        messages = self.build_messages(len(images))
        input_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = processor(
            images=images,
            text=input_text,
            return_tensors="pt",
        )
        inputs = {
            key: value.contiguous().to(model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.15,
        )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output[0][prompt_length:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "image_paths": image_paths,
            "preview_images": preview_images,
            "response": response,
            "device": str(model.device),
            "model_id": self.model_id,
            "quantization": quantization,
        }


def resolve_model_id(model_name: str) -> str:
    try:
        return MODEL_ID_MAP[model_name]
    except KeyError as exc:
        raise KeyError("Only can enter llama, Qwen2B or Qwen8B") from exc
