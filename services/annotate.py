from __future__ import annotations

import re
from typing import Any


def _normalize_text(value: str) -> str:
    s = value.lower().strip()
    # Normalize trailing zeros in decimals: 0.600 → 0.6, 50.0 → 50
    s = re.sub(r'(\d+\.\d*?)0+\b', lambda m: m.group(1).rstrip('.'), s)
    # Remove all non-alphanumeric characters
    return re.sub(r"[^a-z0-9]+", "", s)


def _normalize_variants(value: str) -> list[str]:
    """Return multiple normalized forms to widen matching."""
    base = _normalize_text(value)
    variants = {base}
    # Also try converting numeric strings: "25,000" and "25000" both → "25000"
    stripped = re.sub(r"[^0-9.]", "", value)
    if stripped:
        try:
            variants.add(re.sub(r"[^a-z0-9]+", "", f"{float(stripped):.10g}".lower()))
        except ValueError:
            pass
    return variants


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
        block_variants = _normalize_variants(block.get("text", ""))
        if any(block_variants):
            normalized_blocks.append((block_variants, block))

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int], int]] = set()

    for path, raw_value in _iter_leaf_values(parsed_json):
        value_variants = _normalize_variants(raw_value)
        if not any(v for v in value_variants if len(v) >= 3):
            continue

        # Exact matches first, then partial — avoids short-string false positives
        exact: list[tuple[str, dict]] = []
        partial: list[tuple[str, dict]] = []
        for block_variants, block in normalized_blocks:
            # Check any combination of value variant vs block variant
            if value_variants & block_variants:
                exact.append((next(iter(block_variants)), block))
            elif any(
                len(v) >= 4 and (v in bv or bv in v)
                for v in value_variants for bv in block_variants
            ):
                partial.append((next(iter(block_variants)), block))

        candidates = exact if exact else partial
        matches = []
        for _, block in candidates:
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
