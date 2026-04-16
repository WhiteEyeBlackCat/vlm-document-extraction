from __future__ import annotations

from collections import defaultdict
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
