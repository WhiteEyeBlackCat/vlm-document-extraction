from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
from PIL import Image


OCR_ENGINE_CHOICES = ["none", "paddleocr"]

_ocr_engine_cache: dict[str, Any] = {}


def detect_available_engines() -> list[str]:
    available = ["none"]
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


def _get_paddleocr():
    if "paddleocr" not in _ocr_engine_cache:
        from paddleocr import PaddleOCR
        _ocr_engine_cache["paddleocr"] = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _ocr_engine_cache["paddleocr"]


def _run_paddleocr(
    images: list[Image.Image],
    regions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    reader = _get_paddleocr()
    blocks: list[dict[str, Any]] = []
    for region, image in _iter_ocr_targets(images, regions):
        results = reader.ocr(np.array(image))
        for page_result in results or []:
            # PaddleOCR 3.x: each result is a dict with rec_texts, rec_scores, dt_polys
            texts = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            polys = page_result.get("dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                text = (text or "").strip()
                if not text:
                    continue
                pts = np.array(poly)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                blocks.append(
                    _make_block(
                        region,
                        text,
                        float(score) if score is not None else None,
                        [x1, y1, x2, y2],
                    )
                )
    return blocks
