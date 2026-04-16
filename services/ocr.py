from __future__ import annotations

import importlib.util
import shutil
from typing import Any

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


def run_ocr(images: list[Image.Image], engine: str = "none") -> list[dict[str, Any]]:
    if engine == "none":
        return []
    if engine == "tesseract":
        return _run_tesseract(images)
    if engine == "easyocr":
        return _run_easyocr(images)
    if engine == "paddleocr":
        return _run_paddleocr(images)
    raise ValueError(f"Unsupported OCR engine: {engine}")


def _run_tesseract(images: list[Image.Image]) -> list[dict[str, Any]]:
    import pytesseract

    blocks: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
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
            blocks.append(
                {
                    "page": page_index,
                    "text": text,
                    "confidence": confidence,
                    "bbox": [left, top, left + width, top + height],
                }
            )
    return blocks


def _run_easyocr(images: list[Image.Image]) -> list[dict[str, Any]]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    blocks: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
        results = reader.readtext(image)
        for bbox_points, text, confidence in results:
            if not text or not text.strip():
                continue
            xs = [point[0] for point in bbox_points]
            ys = [point[1] for point in bbox_points]
            blocks.append(
                {
                    "page": page_index,
                    "text": text.strip(),
                    "confidence": float(confidence) if confidence is not None else None,
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                }
            )
    return blocks


def _run_paddleocr(images: list[Image.Image]) -> list[dict[str, Any]]:
    from paddleocr import PaddleOCR

    reader = PaddleOCR(use_angle_cls=True, lang="en")
    blocks: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
        results = reader.ocr(image)
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
                    {
                        "page": page_index,
                        "text": text,
                        "confidence": float(confidence) if confidence is not None else None,
                        "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    }
                )
    return blocks
