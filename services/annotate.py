from __future__ import annotations

import re
from typing import Any


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _iter_leaf_values(node: Any, prefix: str = "content") -> list[tuple[str, str]]:
    leaves: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            leaves.extend(_iter_leaf_values(value, next_prefix))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            leaves.extend(_iter_leaf_values(value, f"{prefix}[{index}]"))
    elif isinstance(node, str):
        if node.strip():
            leaves.append((prefix, node.strip()))
    return leaves


def build_bbox_annotations(parsed_json: dict[str, Any] | None, ocr_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parsed_json or not ocr_blocks:
        return []

    normalized_blocks = []
    for block in ocr_blocks:
        normalized_text = _normalize_text(block.get("text", ""))
        if normalized_text:
            normalized_blocks.append((normalized_text, block))

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int], int]] = set()

    for path, raw_value in _iter_leaf_values(parsed_json):
        normalized_value = _normalize_text(raw_value)
        if not normalized_value:
            continue
        matches = []
        for block_text, block in normalized_blocks:
            if normalized_value in block_text or block_text in normalized_value:
                bbox = block.get("bbox")
                page = block.get("page")
                key = (path, tuple(bbox), page)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "page": page,
                        "text": block.get("text"),
                        "bbox": bbox,
                        "confidence": block.get("confidence"),
                    }
                )
        if matches:
            annotations.append(
                {
                    "path": path,
                    "value": raw_value,
                    "matches": matches,
                }
            )
    return annotations
