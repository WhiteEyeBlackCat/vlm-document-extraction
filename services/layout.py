from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LAYOUT_ENGINE_CHOICES = ["none", "doclayout_yolo"]
DEFAULT_MODEL_CANDIDATES = [
    Path("models/doclayout_yolo.pt"),
    Path("models/doclayout-yolo.pt"),
    Path("weights/doclayout_yolo.pt"),
    Path("weights/doclayout-yolo.pt"),
]
DEFAULT_HF_REPO_ID = os.getenv(
    "DOCLAYOUT_YOLO_REPO_ID",
    "juliozhao/DocLayout-YOLO-DocStructBench",
)
DEFAULT_HF_FILENAME = os.getenv(
    "DOCLAYOUT_YOLO_FILENAME",
    "doclayout_yolo_docstructbench_imgsz1024.pt",
)
DEFAULT_CONFIDENCE = float(os.getenv("DOCLAYOUT_YOLO_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("DOCLAYOUT_YOLO_IOU", "0.45"))

_layout_model_cache: dict[str, Any] = {}


def detect_available_engines() -> list[str]:
    available = ["none"]
    if importlib.util.find_spec("ultralytics") or importlib.util.find_spec("doclayout_yolo"):
        available.append("doclayout_yolo")
    return available


def layout_engine_ready(engine: str) -> bool:
    if engine == "none":
        return True
    if engine != "doclayout_yolo":
        return False
    if resolve_doclayout_model_path(allow_missing=True) is not None:
        return importlib.util.find_spec("ultralytics") is not None
    return importlib.util.find_spec("doclayout_yolo") is not None


def layout_engine_issue(engine: str) -> str | None:
    if engine == "none":
        return None
    if engine != "doclayout_yolo":
        return f"Unsupported layout engine: {engine}"
    model_path = resolve_doclayout_model_path(allow_missing=True)
    if model_path is not None and importlib.util.find_spec("ultralytics") is None:
        return "Local DocLayout-YOLO weights require the 'ultralytics' package."
    if model_path is None and importlib.util.find_spec("doclayout_yolo") is None:
        searched = ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
        return (
            "DocLayout-YOLO requires either the 'doclayout-yolo' package for "
            f"Hugging Face download ('{DEFAULT_HF_REPO_ID}/{DEFAULT_HF_FILENAME}') "
            f"or a local weight file in: {searched}"
        )
    return None


def run_layout(images: list[Image.Image], engine: str = "none") -> list[dict[str, Any]]:
    if engine == "none":
        return _fallback_regions(images)
    if engine == "doclayout_yolo":
        return _run_doclayout_yolo(images)
    raise ValueError(f"Unsupported layout engine: {engine}")


def resolve_doclayout_model_path(allow_missing: bool = False) -> Path | None:
    env_path = os.getenv("DOCLAYOUT_YOLO_MODEL")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"DocLayout-YOLO model not found at DOCLAYOUT_YOLO_MODEL={path}"
        )

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    if allow_missing:
        return None
    searched = ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "DocLayout-YOLO model weights not found. "
        f"Set DOCLAYOUT_YOLO_MODEL or place weights in one of: {searched}"
    )


def _load_doclayout_model():
    model_path = resolve_doclayout_model_path(allow_missing=True)
    if model_path is not None:
        cache_key = f"local::{model_path}"
        if cache_key not in _layout_model_cache:
            from ultralytics import YOLO

            _layout_model_cache.clear()
            _layout_model_cache[cache_key] = YOLO(str(model_path))
        return _layout_model_cache[cache_key]

    cache_key = f"hf::{DEFAULT_HF_REPO_ID}"
    if cache_key not in _layout_model_cache:
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10

        downloaded_path = hf_hub_download(
            repo_id=DEFAULT_HF_REPO_ID,
            filename=DEFAULT_HF_FILENAME,
        )
        _layout_model_cache.clear()
        _layout_model_cache[cache_key] = YOLOv10(downloaded_path)
    return _layout_model_cache[cache_key]


def _fallback_regions(images: list[Image.Image]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
        width, height = image.size
        regions.append(
            {
                "page": page_index,
                "label": "full_page",
                "bbox": [0, 0, width, height],
                "score": 1.0,
            }
        )
    return regions


def _run_doclayout_yolo(images: list[Image.Image]) -> list[dict[str, Any]]:
    model = _load_doclayout_model()
    regions: list[dict[str, Any]] = []

    for page_index, image in enumerate(images, start=1):
        width, height = image.size
        results = model.predict(
            source=np.array(image),
            conf=DEFAULT_CONFIDENCE,
            iou=DEFAULT_IOU,
            verbose=False,
        )
        page_regions = _extract_regions_from_results(results, page_index)
        if not page_regions:
            page_regions = [
                {
                    "page": page_index,
                    "label": "full_page",
                    "bbox": [0, 0, width, height],
                    "score": 1.0,
                }
            ]
        regions.extend(page_regions)

    return regions


def _extract_regions_from_results(results: list[Any], page_index: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        names = getattr(result, "names", {})
        xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        confs = boxes.conf.tolist() if boxes.conf is not None else []
        classes = boxes.cls.tolist() if boxes.cls is not None else []

        for coords, score, cls_id in zip(xyxy, confs, classes):
            left, top, right, bottom = [int(round(value)) for value in coords]
            label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
            regions.append(
                {
                    "page": page_index,
                    "label": label,
                    "bbox": [left, top, right, bottom],
                    "score": round(float(score), 4),
                }
            )
    return regions
