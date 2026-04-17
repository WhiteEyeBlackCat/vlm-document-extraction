from __future__ import annotations

from io import BytesIO
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import re

import torch
from PIL import Image

from model import MODEL_ID_MAP, model_function, resolve_model_id
from services.annotate import build_bbox_annotations
from services.layout import (
    LAYOUT_ENGINE_CHOICES,
    detect_available_engines as detect_available_layout_engines,
    layout_engine_issue,
    layout_engine_ready,
    run_layout,
)
from services.ocr import OCR_ENGINE_CHOICES, detect_available_engines, ocr_engine_ready, run_ocr
from services.parse import build_document_context, build_layout_view, normalize_ocr_blocks


CACHE_DIR = Path("output/cache")


def extraction_cache_key(
    file_bytes: bytes,
    page_number: int | None,
    model_name: str,
    quantization: str,
    ocr_engine: str,
    layout_engine: str,
) -> str:
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    page = page_number if page_number is not None else 0
    return f"{file_hash}_p{page}_{model_name}_{quantization}_{ocr_engine}_{layout_engine}"


def load_extraction_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_extraction_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def get_cache_stats() -> dict:
    if not CACHE_DIR.exists():
        return {"count": 0, "size_mb": 0.0}
    files = list(CACHE_DIR.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"count": len(files), "size_mb": round(total_bytes / 1e6, 1)}


def clear_extraction_cache() -> int:
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)


MODEL_CHOICES = list(MODEL_ID_MAP.keys())
QUANT_CHOICES = ["none", "8bit", "4bit"]
GPU_CHOICES = ["auto"] + [str(i) for i in range(torch.cuda.device_count())]
AVAILABLE_OCR_ENGINES = detect_available_engines()
AVAILABLE_LAYOUT_ENGINES = [
    engine for engine in detect_available_layout_engines() if layout_engine_ready(engine)
]

_model_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def _split_comma_string(s: str) -> list[str] | None:
    """If s looks like 'val1, val2, ...' with ≥2 parts, return the parts; else None."""
    if not isinstance(s, str):
        return None
    parts = [p.strip() for p in s.split(",")]
    # Require ≥2 non-empty parts; reject if any part is empty or the string has no comma
    if len(parts) >= 2 and all(parts):
        return parts
    return None


def _normalize_tables(node: Any) -> Any:
    """Post-process: convert parallel column dicts into arrays of row objects.

    Handles two model output styles:
      1. Parallel array columns:  {"a": [v1, v2], "b": [v3, v4]}
      2. Comma-merged strings:    {"a": "v1, v2", "b": "v3, v4"}
    Both are converted to: [{"a": v1, "b": v3}, {"a": v2, "b": v4}]
    """
    from collections import Counter

    if isinstance(node, list):
        return [_normalize_tables(item) for item in node]
    if not isinstance(node, dict):
        return node

    # Recurse children first
    node = {k: _normalize_tables(v) for k, v in node.items()}

    # ── Style 1: parallel array columns ──────────────────────────────────────
    array_items = {k: v for k, v in node.items() if isinstance(v, list) and len(v) >= 2}
    if len(array_items) >= 2:
        length_counts = Counter(len(v) for v in array_items.values())
        dominant_len = length_counts.most_common(1)[0][0]
        matching = {k: v for k, v in array_items.items() if len(v) == dominant_len}
        if dominant_len >= 2 and len(matching) >= 2:
            rows = []
            for i in range(dominant_len):
                row = {}
                for k, v in node.items():
                    if k in matching:
                        row[k] = v[i]
                    elif isinstance(v, list) and len(v) == 1:
                        row[k] = v[0]
                    elif not isinstance(v, list):
                        row[k] = v
                rows.append(row)
            leftover = {k: v for k, v in node.items()
                        if k not in matching and isinstance(v, list) and len(v) not in (1, dominant_len)}
            if leftover:
                result = {k: v for k, v in node.items() if k not in matching and k not in leftover}
                result.update(leftover)
                result["rows"] = rows
                return result
            return rows

    # ── Style 2: comma-merged strings ────────────────────────────────────────
    splittable = {}
    for k, v in node.items():
        if isinstance(v, str):
            parts = _split_comma_string(v)
            if parts:
                splittable[k] = parts

    if len(splittable) >= 2:
        length_counts = Counter(len(v) for v in splittable.values())
        dominant_len = length_counts.most_common(1)[0][0]
        matching_str = {k: v for k, v in splittable.items() if len(v) == dominant_len}
        if dominant_len >= 2 and len(matching_str) >= 2:
            rows = []
            for i in range(dominant_len):
                row = {}
                for k, v in node.items():
                    if k in matching_str:
                        row[k] = matching_str[k][i]
                    else:
                        row[k] = v  # scalar or non-splittable — broadcast
                rows.append(row)
            return rows

    return node


def _extract_json(text: str) -> Any:
    """Parse JSON from model output, stripping markdown fences and repairing truncation."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # First try direct parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open structures
        parsed = _repair_truncated_json(text)
        if parsed is None:
            return json.loads(text)  # re-raise original error

    return _normalize_tables(parsed)


def _repair_truncated_json(text: str) -> Any | None:
    """Best-effort repair of JSON truncated mid-generation."""
    # Remove the last incomplete token (unterminated string or trailing comma)
    # Walk back from the end to find the last valid boundary
    for end in range(len(text), max(len(text) - 200, 0), -1):
        candidate = text[:end].rstrip().rstrip(",").rstrip()
        # Count open brackets/braces to close
        stack = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not in_string:
                in_string = True
            elif ch == '"' and in_string:
                in_string = False
            elif not in_string:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and stack[-1] == ch:
                        stack.pop()
        if in_string:
            continue  # still inside a string, try shorter
        closing = "".join(reversed(stack))
        try:
            return json.loads(candidate + closing)
        except json.JSONDecodeError:
            continue
    return None


def get_model(model_id: str, quantization: str, gpu: str = "auto"):
    key = (model_id, quantization, gpu)
    if key not in _model_cache:
        if _model_cache:
            # Explicitly release previous model weights from GPU before loading a new one
            old_model, old_processor = next(iter(_model_cache.values()))
            old_model.cpu()  # move tensors off GPU first so CUDA frees them immediately
            del old_model, old_processor
            _model_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
        runner = model_function(model_id)
        model, processor = runner.build_model(quantization=quantization, gpu=gpu)
        _model_cache[key] = (model, processor)
    return _model_cache[key]


def build_gallery_items(image_paths, preview_images, rendered_images):
    if preview_images:
        items = []
        for i, img_bytes in enumerate(preview_images):
            image = Image.open(BytesIO(img_bytes))
            rendered = rendered_images[i]
            items.append(
                {
                    "name": f"page-{i + 1}",
                    "image_bytes": img_bytes,
                    "media_type": "image/jpeg",
                    "width": rendered.width,
                    "height": rendered.height,
                    "preview_width": image.width,
                    "preview_height": image.height,
                }
            )
        return items

    items = []
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        rendered = rendered_images[index]
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        items.append(
            {
                "name": Path(str(path)).name,
                "image_bytes": buffer.getvalue(),
                "media_type": "image/jpeg",
                "width": rendered.width,
                "height": rendered.height,
                "preview_width": image.width,
                "preview_height": image.height,
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
            "width": item.get("width"),
            "height": item.get("height"),
            "preview_width": item.get("preview_width"),
            "preview_height": item.get("preview_height"),
        }
        for item in gallery_items
    ]


def get_preview_from_path(input_path: Path) -> dict[str, Any]:
    runner = model_function(MODEL_ID_MAP["Qwen2B"])
    image_paths, images, preview_images = runner.load_images(input_path)
    gallery_items = build_gallery_items(image_paths, preview_images, images)
    return {
        "input_labels": [str(path) for path in image_paths],
        "gallery": encode_gallery_data_urls(gallery_items),
    }


def get_preview_from_upload(
    upload_bytes: bytes,
    filename: str,
    page_number: int | None = None,
) -> dict[str, Any]:
    suffix = Path(filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload_bytes)
        temp_path = Path(temp_file.name)
    try:
        runner = model_function(MODEL_ID_MAP["Qwen2B"])
        image_paths, images, preview_images = runner.load_images(temp_path)
        total_pages = len(images)
        if page_number is not None:
            idx = page_number - 1
            if 0 <= idx < total_pages:
                images = [images[idx]]
                image_paths = [image_paths[idx]]
                preview_images = [preview_images[idx]] if preview_images else None
        gallery_items = build_gallery_items(image_paths, preview_images, images)
        return {
            "total_pages": total_pages,
            "current_page": page_number,
            "input_labels": [str(path) for path in image_paths],
            "gallery": encode_gallery_data_urls(gallery_items),
        }
    finally:
        temp_path.unlink(missing_ok=True)


def run_extraction_from_path(
    input_path: Path,
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
    ocr_engine: str = "none",
    layout_engine: str = "none",
    page_number: int | None = None,
) -> dict[str, Any]:
    t_start = time.time()
    runner = model_function(resolve_model_id(model_name))
    image_paths, images, preview_images = runner.load_images(input_path)

    total_pages = len(images)
    if page_number is not None:
        idx = page_number - 1
        if not (0 <= idx < total_pages):
            raise ValueError(f"Page {page_number} out of range (1–{total_pages})")
        images = [images[idx]]
        image_paths = [image_paths[idx]]
        preview_images = [preview_images[idx]] if preview_images else None

    layout_regions: list[dict[str, Any]] = []
    layout_error = None
    if layout_engine != "none":
        if not layout_engine_ready(layout_engine):
            layout_error = layout_engine_issue(layout_engine) or (
                f"Layout engine '{layout_engine}' is not available."
            )
            layout_regions = run_layout(images, engine="none")
        else:
            try:
                layout_regions = run_layout(images, engine=layout_engine)
            except Exception as exc:
                layout_error = str(exc)
                layout_regions = run_layout(images, engine="none")
    else:
        layout_regions = run_layout(images, engine="none")

    ocr_blocks: list[dict[str, Any]] = []
    ocr_error = None
    if ocr_engine != "none":
        if not ocr_engine_ready(ocr_engine):
            ocr_error = f"OCR engine '{ocr_engine}' is not installed."
        else:
            try:
                ocr_blocks = run_ocr(images, engine=ocr_engine, regions=layout_regions)
            except Exception as exc:
                ocr_error = str(exc)

    ocr_blocks = normalize_ocr_blocks(ocr_blocks)
    parsed_layout = build_layout_view(layout_regions, ocr_blocks)
    document_context = build_document_context(parsed_layout)

    model_id = resolve_model_id(model_name)
    cache_miss = (model_id, quantization, gpu) not in _model_cache
    model, processor = get_model(model_id, quantization, gpu)
    messages = runner.build_messages(len(images), document_context=document_context)
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

    inputs = processor(text=input_text, images=images, return_tensors="pt", processor_kwargs={})
    inputs = {
        key: value.contiguous().to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    output = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.15,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    response = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
    elapsed = time.time() - t_start

    parsed_json = None
    json_error = None
    try:
        parsed_json = _extract_json(response)
    except json.JSONDecodeError as exc:
        json_error = str(exc)

    bbox_annotations = build_bbox_annotations(parsed_json, ocr_blocks)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "device": str(model.device),
        "quantization": quantization,
        "elapsed_seconds": round(elapsed, 2),
        "cache_miss": cache_miss,
        "total_pages": total_pages,
        "current_page": page_number,
        "input_labels": [str(path) for path in image_paths],
        "gallery_items": build_gallery_items(image_paths, preview_images, images),
        "raw_response": response,
        "parsed_json": parsed_json,
        "json_error": json_error,
        "ocr_engine": ocr_engine,
        "ocr_available_engines": AVAILABLE_OCR_ENGINES,
        "layout_engine": layout_engine,
        "layout_available_engines": AVAILABLE_LAYOUT_ENGINES,
        "layout_regions": layout_regions,
        "layout_error": layout_error,
        "parsed_layout": parsed_layout,
        "document_context": document_context,
        "ocr_blocks": ocr_blocks,
        "ocr_error": ocr_error,
        "bbox_annotations": bbox_annotations,
    }


def run_extraction_from_upload(
    upload_bytes: bytes,
    filename: str,
    model_name: str,
    max_tokens: int,
    quantization: str,
    gpu: str,
    ocr_engine: str = "none",
    layout_engine: str = "none",
    page_number: int | None = None,
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
            ocr_engine=ocr_engine,
            layout_engine=layout_engine,
            page_number=page_number,
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
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=input_text, images=images, return_tensors="pt", processor_kwargs={})
            inputs = {
                key: value.contiguous().to(model.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            response = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

            out_path = out_dir / f"{path.stem}.json"
            json_valid = True
            try:
                parsed = _extract_json(response)
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
