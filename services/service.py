from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import torch
from PIL import Image

from model import MODEL_ID_MAP, model_function, resolve_model_id


MODEL_CHOICES = list(MODEL_ID_MAP.keys())
QUANT_CHOICES = ["none", "8bit", "4bit"]
GPU_CHOICES = ["auto"] + [str(i) for i in range(torch.cuda.device_count())]

_model_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def get_model(model_id: str, quantization: str, gpu: str = "auto"):
    key = (model_id, quantization, gpu)
    if key not in _model_cache:
        runner = model_function(model_id)
        model, processor = runner.build_model(quantization=quantization, gpu=gpu)
        _model_cache.clear()
        _model_cache[key] = (model, processor)
    return _model_cache[key]


def build_gallery_items(image_paths, preview_images):
    if preview_images:
        return [
            {
                "name": f"page-{i + 1}",
                "image_bytes": img_bytes,
                "media_type": "image/jpeg",
            }
            for i, img_bytes in enumerate(preview_images)
        ]

    items = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        items.append(
            {
                "name": Path(str(path)).name,
                "image_bytes": buffer.getvalue(),
                "media_type": "image/jpeg",
            }
        )
    return items


def encode_gallery_data_urls(gallery_items):
    import base64

    return [
        {
            "name": item["name"],
            "src": f"data:{item['media_type']};base64,{base64.b64encode(item['image_bytes']).decode('ascii')}",
            "kind": "image",
        }
        for item in gallery_items
    ]


def get_preview_from_path(input_path: Path) -> dict[str, Any]:
    runner = model_function(MODEL_ID_MAP["Qwen2B"])
    image_paths, _, preview_images = runner.load_images(input_path)
    gallery_items = build_gallery_items(image_paths, preview_images)
    return {
        "input_labels": [str(path) for path in image_paths],
        "gallery": encode_gallery_data_urls(gallery_items),
    }


def run_extraction_from_path(
    input_path: Path,
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
) -> dict[str, Any]:
    t_start = time.time()
    runner = model_function(resolve_model_id(model_name))
    image_paths, images, preview_images = runner.load_images(input_path)

    model_id = resolve_model_id(model_name)
    cache_miss = (model_id, quantization, gpu) not in _model_cache
    model, processor = get_model(model_id, quantization, gpu)
    messages = runner.build_messages(len(images))
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=images, text=input_text, return_tensors="pt")
    inputs = {
        key: value.contiguous().to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    output = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=False,
        num_beams=1,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    response = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
    elapsed = time.time() - t_start

    parsed_json = None
    json_error = None
    try:
        parsed_json = json.loads(response)
    except json.JSONDecodeError as exc:
        json_error = str(exc)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "device": str(model.device),
        "quantization": quantization,
        "elapsed_seconds": round(elapsed, 2),
        "cache_miss": cache_miss,
        "input_labels": [str(path) for path in image_paths],
        "gallery_items": build_gallery_items(image_paths, preview_images),
        "raw_response": response,
        "parsed_json": parsed_json,
        "json_error": json_error,
    }


def run_extraction_from_upload(
    upload_bytes: bytes,
    filename: str,
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
    ) -> dict[str, Any]:
    suffix = Path(filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload_bytes)
        temp_path = Path(temp_file.name)

    try:
        return run_extraction_from_path(
            input_path=temp_path,
            model_name=model_name,
            max_tokens=max_tokens,
            quantization=quantization,
            gpu=gpu,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def run_batch_from_paths(
    file_paths: list[Path],
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
    output_dir: str,
) -> dict[str, Any]:
    if not file_paths:
        raise ValueError("No input files provided.")

    out_dir = Path(output_dir.strip()) if output_dir and output_dir.strip() else Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(model_name)
    cache_miss = (model_id, quantization, gpu) not in _model_cache
    runner = model_function(model_id)
    model, processor = get_model(model_id, quantization, gpu)

    results = []
    t_start = time.time()

    for file_path in file_paths:
        path = Path(file_path)
        try:
            _, images, _ = runner.load_images(path)
            messages = runner.build_messages(len(images))
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=images, text=input_text, return_tensors="pt")
            inputs = {
                key: value.contiguous().to(model.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=False,
                num_beams=1,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            response = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

            out_path = out_dir / f"{path.stem}.json"
            json_valid = True
            try:
                parsed = json.loads(response.strip())
                out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                message = f"✅ {path.name} → {out_path.name}"
            except json.JSONDecodeError as exc:
                json_valid = False
                out_path.write_text(response, encoding="utf-8")
                message = f"⚠️ {path.name} → JSON 解析失敗，原始輸出已儲存 ({exc})"

            results.append(
                {
                    "file_name": path.name,
                    "output_path": str(out_path.absolute()),
                    "json_valid": json_valid,
                    "message": message,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "file_name": path.name,
                    "output_path": None,
                    "json_valid": False,
                    "message": f"❌ {path.name} → {exc}",
                }
            )

    elapsed = time.time() - t_start
    return {
        "model_name": model_name,
        "model_id": model_id,
        "device": str(model.device),
        "quantization": quantization,
        "cache_miss": cache_miss,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(out_dir.absolute()),
        "results": results,
    }


def run_batch_from_uploads(
    uploads: list[tuple[str, bytes]],
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
    output_dir: str,
) -> dict[str, Any]:
    temp_paths: list[Path] = []
    try:
        for filename, upload_bytes in uploads:
            suffix = Path(filename or "upload.pdf").suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(upload_bytes)
                temp_paths.append(Path(temp_file.name))

        return run_batch_from_paths(
            file_paths=temp_paths,
            model_name=model_name,
            max_tokens=max_tokens,
            quantization=quantization,
            gpu=gpu,
            output_dir=output_dir,
        )
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
