from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


def build_layout_view(
    layout_regions: list[dict[str, Any]],
    ocr_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regions_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for region in layout_regions:
        item = dict(region)
        item["blocks"] = []
        item["text"] = ""
        regions_by_page[region["page"]].append(item)

    for block in ocr_blocks:
        page = block.get("page")
        bbox = block.get("bbox") or [0, 0, 0, 0]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        matched = None
        for region in regions_by_page.get(page, []):
            left, top, right, bottom = region["bbox"]
            if left <= center_x <= right and top <= center_y <= bottom:
                matched = region
                break

        if matched is None:
            continue
        matched["blocks"].append(block)

    parsed_pages = []
    for page in sorted(regions_by_page):
        page_regions = regions_by_page[page]
        for region in page_regions:
            region["text"] = "\n".join(block["text"] for block in region["blocks"])
        parsed_pages.append(
            {
                "page": page,
                "regions": page_regions,
            }
        )
    return parsed_pages


def build_document_context(parsed_layout: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for page_entry in parsed_layout:
        page = page_entry.get("page")
        lines.append(f"Page {page}:")
        regions = page_entry.get("regions") or []
        if not regions:
            lines.append("- No regions detected")
            continue

        for region_index, region in enumerate(regions, start=1):
            label = region.get("label", "region")
            bbox = region.get("bbox") or []
            blocks = _sort_blocks(region.get("blocks") or [])
            lines.append(f"- Region {region_index}: {label} bbox={bbox}")
            if not blocks:
                lines.append("  text: <empty>")
                continue

            grouped_rows = _group_blocks_into_rows(blocks)
            for row_index, row in enumerate(grouped_rows, start=1):
                row_text = " | ".join(block["text"] for block in row if block.get("text"))
                if row_text:
                    lines.append(f"  row {row_index}: {row_text}")
    return "\n".join(lines)


def _sort_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        blocks,
        key=lambda block: (
            block.get("bbox", [0, 0, 0, 0])[1],
            block.get("bbox", [0, 0, 0, 0])[0],
        ),
    )


def _group_blocks_into_rows(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        block_height = max(1, bbox[3] - bbox[1])
        center_y = (bbox[1] + bbox[3]) / 2

        matched_row = None
        for row in rows:
            sample_bbox = row[0].get("bbox") or [0, 0, 0, 0]
            sample_center_y = (sample_bbox[1] + sample_bbox[3]) / 2
            sample_height = max(1, sample_bbox[3] - sample_bbox[1])
            tolerance = max(block_height, sample_height) * 0.6
            if abs(center_y - sample_center_y) <= tolerance:
                matched_row = row
                break

        if matched_row is None:
            rows.append([block])
        else:
            matched_row.append(block)

    for row in rows:
        row.sort(key=lambda block: block.get("bbox", [0, 0, 0, 0])[0])
    return rows


def normalize_ocr_blocks(ocr_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for block in ocr_blocks:
        text = re.sub(r"\s+", " ", str(block.get("text", "")).strip())
        if not text:
            continue

        normalized_block = dict(block)
        normalized_block["text"] = text
        normalized.append(normalized_block)
    return normalized
