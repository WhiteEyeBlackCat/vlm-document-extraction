from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def find_content_bbox(
    image: Image.Image,
    white_threshold: int = 245,
    min_content_ratio: float = 0.005,
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-white content."""
    gray = image.convert("L")
    arr = np.array(gray)

    content_mask = arr < white_threshold
    row_ratio = content_mask.mean(axis=1)
    col_ratio = content_mask.mean(axis=0)

    row_idx = np.where(row_ratio > min_content_ratio)[0]
    col_idx = np.where(col_ratio > min_content_ratio)[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        return None

    top = int(row_idx[0])
    bottom = int(row_idx[-1]) + 1
    left = int(col_idx[0])
    right = int(col_idx[-1]) + 1
    return left, top, right, bottom


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def crop_image(
    src_path: Path,
    dst_path: Path,
    white_threshold: int,
    min_content_ratio: float,
    padding: int,
) -> tuple[int, int, int, int] | None:
    image = Image.open(src_path)
    bbox = find_content_bbox(
        image,
        white_threshold=white_threshold,
        min_content_ratio=min_content_ratio,
    )
    if bbox is None:
        return None

    bbox = expand_bbox(bbox, image.size, padding)
    cropped = image.crop(bbox)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst_path, quality=95)
    return bbox


def build_output_path(src_path: Path, src_root: Path, dst_root: Path) -> Path:
    return dst_root / src_path.relative_to(src_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-crop document JPGs by detecting non-white content."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/jpg/Sunrisetek-2"),
        help="Folder containing source JPGs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/jpg_cropped/Sunrisetek-2"),
        help="Folder to save cropped JPGs.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Pixels darker than this are treated as content.",
    )
    parser.add_argument(
        "--min-content-ratio",
        type=float,
        default=0.005,
        help="Minimum dark-pixel ratio per row/column to count as content.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=30,
        help="Extra pixels kept around the detected content box.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N JPGs for quick testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jpg_paths = sorted(args.input_root.rglob("*.jpg"))
    if args.limit is not None:
        jpg_paths = jpg_paths[: args.limit]

    if not jpg_paths:
        raise SystemExit(f"No JPG files found under: {args.input_root}")

    cropped_count = 0
    skipped_count = 0

    for src_path in jpg_paths:
        dst_path = build_output_path(src_path, args.input_root, args.output_root)
        bbox = crop_image(
            src_path=src_path,
            dst_path=dst_path,
            white_threshold=args.white_threshold,
            min_content_ratio=args.min_content_ratio,
            padding=args.padding,
        )
        if bbox is None:
            skipped_count += 1
            print(f"SKIP {src_path}")
            continue

        cropped_count += 1
        print(f"CROP {src_path} -> {dst_path} bbox={bbox}")

    print(
        f"Done. cropped={cropped_count}, skipped={skipped_count}, "
        f"output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
