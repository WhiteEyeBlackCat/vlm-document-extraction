from __future__ import annotations

import importlib.util
import shutil
from typing import Any

import numpy as np
from PIL import Image


OCR_ENGINE_CHOICES = ["none", "tesseract", "easyocr", "paddleocr"]


def detect_available_engines() -> list[str]:
    available = ["none"]
    if shutil.which("tesseract") and importlib.util.find_spec("pytesseract"):
        available.append("tesseract")
    if importlib.util.find_spec("easyocr"):
        available.append("easyocr")
    if importlib.util.find_spec("paddleocr"):
        available.append("paddleocr")
    return available


def ocr_engine_ready(engine: str) -> bool:
    return engine in detect_available_engines()


def run_ocr(
    images: list[Image.Image],
    engine: str = "none",
    regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if engine == "none":
        return []
    if engine == "tesseract":
        return _run_tesseract(images, regions)
    if engine == "easyocr":
        return _run_easyocr(images, regions)
    if engine == "paddleocr":
        return _run_paddleocr(images, regions)
    raise ValueError(f"Unsupported OCR engine: {engine}")


def _iter_ocr_targets(
    images: list[Image.Image],
    regions: list[dict[str, Any]] | None,
) -> list[tuple[dict[str, Any], Image.Image]]:
    if not regions:
        targets = []
        for page_index, image in enumerate(images, start=1):
            width, height = image.size
            targets.append(
                (
                    {
                        "page": page_index,
                        "label": "full_page",
                        "bbox": [0, 0, width, height],
                    },
                    image,
                )
            )
        return targets

    page_images = {index: image for index, image in enumerate(images, start=1)}
    targets = []
    for region in regions:
        page = int(region.get("page", 0))
        image = page_images.get(page)
        bbox = region.get("bbox") or [0, 0, 0, 0]
        if image is None or len(bbox) != 4:
            continue
        left, top, right, bottom = [int(v) for v in bbox]
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)
        if right <= left or bottom <= top:
            continue
        cropped = image.crop((left, top, right, bottom))
        targets.append(
            (
                {
                    "page": page,
                    "label": region.get("label", "region"),
                    "bbox": [left, top, right, bottom],
                },
                cropped,
            )
        )
    return targets


def _make_block(
    region: dict[str, Any],
    text: str,
    confidence: float | None,
    local_bbox: list[int],
) -> dict[str, Any]:
    region_bbox = region["bbox"]
    return {
        "page": region["page"],
        "region_label": region.get("label", "region"),
        "region_bbox": region_bbox,
        "text": text,
        "confidence": confidence,
        "bbox": [
            region_bbox[0] + local_bbox[0],
            region_bbox[1] + local_bbox[1],
            region_bbox[0] + local_bbox[2],
            region_bbox[1] + local_bbox[3],
        ],
    }


def _run_tesseract(
    images: list[Image.Image],
    regions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    import pytesseract

    blocks: list[dict[str, Any]] = []
    for region, image in _iter_ocr_targets(images, regions):
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        total = len(data["text"])
        for idx in range(total):
            text = (data["text"][idx] or "").strip()
            if not text:
                continue
            left = int(data["left"][idx])
            top = int(data["top"][idx])
            width = int(data["width"][idx])
            height = int(data["height"][idx])
            conf_raw = data["conf"][idx]
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = None
            blocks.append(_make_block(region, text, confidence, [left, top, left + width, top + height]))
    return blocks


def _run_easyocr(
    images: list[Image.Image],
    regions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    blocks: list[dict[str, Any]] = []
    for region, image in _iter_ocr_targets(images, regions):
        results = reader.readtext(image)
        for bbox_points, text, confidence in results:
            if not text or not text.strip():
                continue
            xs = [point[0] for point in bbox_points]
            ys = [point[1] for point in bbox_points]
            blocks.append(
                _make_block(
                    region,
                    text.strip(),
                    float(confidence) if confidence is not None else None,
                    [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                )
            )
    return blocks


def _run_paddleocr(
    images: list[Image.Image],
    regions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    from paddleocr import PaddleOCR

    reader = PaddleOCR(use_angle_cls=True, lang="en")
    blocks: list[dict[str, Any]] = []
    for region, image in _iter_ocr_targets(images, regions):
        results = reader.ocr(np.array(image))
        for line in results or []:
            for item in line or []:
                bbox_points, text_info = item
                text = (text_info[0] or "").strip()
                confidence = text_info[1] if len(text_info) > 1 else None
                if not text:
                    continue
                xs = [point[0] for point in bbox_points]
                ys = [point[1] for point in bbox_points]
                blocks.append(
                    _make_block(
                        region,
                        text,
                        float(confidence) if confidence is not None else None,
                        [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    )
                )
    return blocks
