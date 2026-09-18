"""Run the trained Pitt HexAI segmentation backend on local test images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pelvis_segmentation import (  # noqa: E402
    PITT_HEXAI_HIGHRES_DEFAULT_CHECKPOINT,
    PelvisSegmentationBackend,
)


DEFAULT_OUTPUT_DIR = APP_DIR / "data" / "pitt_hexai_segmentation_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the local Pitt HexAI HighRes segmentation backend."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PITT_HEXAI_HIGHRES_DEFAULT_CHECKPOINT,
        help="Training checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for masks, overlays, and contact sheet.",
    )
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Image path to segment. Can be repeated. Defaults to app tutorial/test images.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    polarity = parser.add_mutually_exclusive_group()
    polarity.add_argument(
        "--invert",
        action="store_true",
        help="Force inverted grayscale preprocessing.",
    )
    polarity.add_argument(
        "--no-invert",
        action="store_true",
        help="Force normal grayscale preprocessing.",
    )
    parser.add_argument(
        "--intersect-prior",
        action="store_true",
        help="Intersect prediction with the app pelvis-shaped prior.",
    )
    return parser.parse_args()


def default_images() -> list[Path]:
    images = []
    for pattern in ("data/tutorial_image_*.tif", "test_images/*.tif", "test_images/*.tiff"):
        images.extend(sorted(APP_DIR.glob(pattern)))
    return images


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = rgb.copy()
    overlay[mask > 0] = np.array([255, 70, 40], dtype=np.uint8)
    blended = (0.62 * rgb + 0.38 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(path)


def make_contact_sheet(overlay_paths: list[Path], output_path: Path) -> None:
    if not overlay_paths:
        return
    tile_w, tile_h = 320, 260
    label_h = 34
    cols = min(4, len(overlay_paths))
    rows = int(np.ceil(len(overlay_paths) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(overlay_paths):
        col = idx % cols
        row = idx // cols
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tile_w - 20, tile_h - 16))
        x = col * tile_w + (tile_w - image.width) // 2
        y = row * (tile_h + label_h) + label_h + (tile_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((col * tile_w + 10, row * (tile_h + label_h) + 10), path.stem, fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PELVIS_PITT_HEXAI_CHECKPOINT"] = str(checkpoint)
    os.environ["PELVIS_PITT_HEXAI_THRESHOLD"] = str(args.threshold)
    if args.invert:
        os.environ["PELVIS_PITT_HEXAI_INVERT_GRAYSCALE"] = "1"
    elif args.no_invert:
        os.environ["PELVIS_PITT_HEXAI_INVERT_GRAYSCALE"] = "0"
    else:
        os.environ["PELVIS_PITT_HEXAI_INVERT_GRAYSCALE"] = "auto"
    os.environ["PELVIS_PITT_HEXAI_INTERSECT_PRIOR"] = "1" if args.intersect_prior else "0"

    image_paths = args.image or default_images()
    if not image_paths:
        raise SystemExit("No input images found.")

    backend = PelvisSegmentationBackend(mode="pitt_hexai_highres")
    overlay_paths: list[Path] = []
    manifest: list[str] = []
    for image_path in image_paths:
        image_path = image_path.expanduser().resolve()
        image = Image.open(image_path)
        result = backend.segment(image)
        stem = safe_stem(image_path)
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        Image.fromarray((result.mask.astype(np.uint8) * 255)).save(mask_path)
        save_overlay(image, result.mask, overlay_path)
        overlay_paths.append(overlay_path)
        manifest.append(
            f"{image_path}\n  mask={mask_path}\n  overlay={overlay_path}\n"
            f"  pixels={int(result.mask.sum())}\n  detail={result.detail}\n"
        )
        print(
            f"{image_path.name}: {int(result.mask.sum())} mask pixels, "
            f"{result.detail} -> {overlay_path}"
        )

    make_contact_sheet(overlay_paths, output_dir / "contact_sheet.png")
    (output_dir / "manifest.txt").write_text("\n".join(manifest))
    print(f"Wrote previews to {output_dir}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
