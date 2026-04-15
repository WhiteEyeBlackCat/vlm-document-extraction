from pathlib import Path
from prompt import Prompt
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration, Qwen3VLForConditionalGeneration


class model_function():
    def __init__(self, model_id):
        self.model_id = model_id

    def build_model(self):
        if self.model_id.startswith("meta-llama"):
            model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(self.model_id)
        elif self.model_id.startswith("Qwen"):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, 
                dtype=torch.bfloat16, 
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(self.model_id)
        else:
            raise ValueError("Only Qwen and llama is available")
        return model, processor

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Ask PyTorch to favor deterministic kernels when possible.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def load_images(self, folder: Path):
        preferred_order = [
            #"page-1_head.jpg",
            #"page-1_ship.jpg",
            #"page-1_general_info.jpg",
            #"page-1_lower_half.jpg",
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
        prompt = Prompt

        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]







