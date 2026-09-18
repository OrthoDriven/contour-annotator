"""Download small pelvis X-ray segmentation dataset samples.

The public datasets used here are large or API-gated. This script keeps the
sample folder small by reading ZIP central directories over HTTP range requests
and extracting only a few named files.
"""

from __future__ import annotations

import io
import json
import os
import re
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib
import pydicom
import requests
from PIL import Image, ImageDraw, ImageOps


OUT_DIR = Path("data/pelvis_segmentation_dataset_samples")
PENGWIN_ZIP_URL = "https://zenodo.org/records/10913196/files/train.zip?download=1"
PITT_HIP_ZIP_URL = (
    "https://media.githubusercontent.com/media/"
    "pitthexai/AI_Fairness_in_Hip_and_Knee_Bony_Anatomy_Segmentation/"
    "main/Sample_Dataset/hip_sample.zip"
)

ROBOFLOW_SOURCES = [
    {
        "slug": "roboflow_linas_pelvis_ap_xray",
        "title": "Roboflow Universe: Pelvis AP X-ray by Linas WS",
        "url": "https://universe.roboflow.com/linas-ws/pelvis-ap-x-ray-ys9pz",
        "classes": [
            "FEMUR",
            "ILIAK",
            "OBTURATOR",
            "PUBIS",
            "SAKRO-ILIAK-EKLEM",
            "SUORCIL",
            "TEARDROP",
        ],
        "license": "CC BY 4.0",
        "note": "Public project page reports 458 images but 0 exported dataset versions.",
    },
    {
        "slug": "roboflow_yolov8_pelvis_xray",
        "title": "Roboflow Universe: pelvis_xray by YoloV8",
        "url": "https://universe.roboflow.com/yolov8-jl4qm/pelvis_xray",
        "classes": [
            "AcetabulumL",
            "AcetabulumR",
            "FemurL",
            "FemurR",
            "GreaterTrochanterL",
            "GreaterTrochanterR",
            "IliumL",
            "IliumR",
            "IschiumL",
            "IschiumR",
            "LesserTrochanterL",
            "LesserTrochanterR",
            "PelvicBrim",
            "PubisL",
            "PubisR",
            "Sacrum&Coccyx",
            "TearDropL",
            "TearDropR",
        ],
        "license": "Public Domain",
        "note": "Public project page reports 191 images and 5 dataset versions.",
    },
    {
        "slug": "roboflow_xray_pelvis",
        "title": "Roboflow Universe: Pelvis by xray",
        "url": "https://universe.roboflow.com/xray-3l69l/pelvis-rh5xb-gluu8",
        "classes": ["FEMUR", "ILIAK", "OBTURATOR", "PUBIS", "SUORCIL", "TEARDROP"],
        "license": "CC BY 4.0",
        "note": "Public model page reports 458 images and 1 dataset version.",
    },
]


class RemoteHTTPFile(io.RawIOBase):
    """Seekable, read-only HTTP byte-range adapter for zipfile.ZipFile."""

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self.url = url
        self.session = session or requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.url = response.url
        length = response.headers.get("content-length")
        if not length:
            raise RuntimeError(f"Could not determine remote file size for {url}")
        self.length = int(length)
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.position + offset
        elif whence == os.SEEK_END:
            new_position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence {whence}")
        if new_position < 0:
            raise ValueError("Cannot seek before beginning of file")
        self.position = min(new_position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        if size == 0:
            return b""
        start = self.position
        end = min(self.position + size, self.length) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=120,
        )
        response.raise_for_status()
        if response.status_code != 206 and response.headers.get("content-length"):
            expected = end - start + 1
            actual = int(response.headers["content-length"])
            if actual != expected:
                raise RuntimeError(
                    f"Server ignored range request for {self.url}: "
                    f"wanted {expected} bytes, got {actual}"
                )
        data = response.content
        self.position += len(data)
        return data


@dataclass(frozen=True)
class SavedSample:
    dataset: str
    sample_id: str
    files: list[str]
    note: str = ""


def normalize_to_uint8(array: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = arr[finite].reshape(-1) if arr.ndim == 1 else np.where(finite, arr, np.nan)
    low, high = np.nanpercentile(arr, [1.0, 99.5])
    if high <= low:
        high = float(np.nanmax(arr))
        low = float(np.nanmin(arr))
    scaled = (array.astype(np.float32) - low) / max(high - low, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    if invert:
        scaled = 1.0 - scaled
    return (scaled * 255).astype(np.uint8)


def save_overlay(image_u8: np.ndarray, mask: np.ndarray, path: Path) -> None:
    base = np.stack([image_u8, image_u8, image_u8], axis=-1)
    overlay = base.copy()
    overlay[mask] = np.array([255, 60, 40], dtype=np.uint8)
    blended = (0.65 * base + 0.35 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(path)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_member(zip_file: zipfile.ZipFile, name: str) -> bytes:
    try:
        with zip_file.open(name) as handle:
            return handle.read()
    except KeyError as exc:
        raise RuntimeError(f"Missing expected ZIP member: {name}") from exc


def download_pengwin_samples(out_dir: Path) -> list[SavedSample]:
    sample_ids = ["001_0000", "001_0250", "002_0000"]
    saved: list[SavedSample] = []
    with zipfile.ZipFile(RemoteHTTPFile(PENGWIN_ZIP_URL)) as zf:
        for sample_id in sample_ids:
            image_name = f"train/input/images/x-ray/{sample_id}.tif"
            seg_name = f"train/output/images/x-ray/{sample_id}.tif"
            sample_dir = out_dir / "pengwin_task2_synthetic_xray" / sample_id
            image_bytes = extract_member(zf, image_name)
            seg_bytes = extract_member(zf, seg_name)

            raw_image_path = sample_dir / "image_raw.tif"
            raw_seg_path = sample_dir / "segmentation_encoded_uint32.tif"
            write_bytes(raw_image_path, image_bytes)
            write_bytes(raw_seg_path, seg_bytes)

            image = Image.open(io.BytesIO(image_bytes))
            seg = Image.open(io.BytesIO(seg_bytes))
            image_arr = np.asarray(image)
            seg_arr = np.asarray(seg)
            image_u8 = normalize_to_uint8(image_arr)
            mask = seg_arr != 0

            image_png = sample_dir / "image_display.png"
            mask_png = sample_dir / "mask_binary_any_fragment.png"
            overlay_png = sample_dir / "overlay_binary_any_fragment.png"
            Image.fromarray(image_u8).save(image_png)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_png)
            save_overlay(image_u8, mask, overlay_png)

            saved.append(
                SavedSample(
                    dataset="PENGWIN Task 2 synthetic X-ray",
                    sample_id=sample_id,
                    files=[
                        str(raw_image_path),
                        str(raw_seg_path),
                        str(image_png),
                        str(mask_png),
                        str(overlay_png),
                    ],
                    note="Mask is a binary union of all encoded pelvic fragments; raw uint32 bitmask is also saved.",
                )
            )
    return saved


def file_kind(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".dcm") or "/images/" in lower:
        return "image"
    if lower.endswith((".nii", ".nii.gz")) or "/annotations/" in lower:
        return "mask"
    if not lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        return "other"
    if any(token in lower for token in ("mask", "label", "seg", "manual", "ground", "gt")):
        return "mask"
    return "image"


def file_stem(path: str) -> str:
    name = Path(path).name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def find_pitt_pairs(names: Iterable[str]) -> list[tuple[str, str]]:
    image_names = [name for name in names if file_kind(name) == "image"]
    mask_names = [name for name in names if file_kind(name) == "mask"]
    pairs: list[tuple[str, str]] = []

    def stem_key(path: str) -> str:
        stem = file_stem(path).lower()
        stem = re.sub(r"(_|-)?(image|img|xray|radiograph|mask|label|seg|manual|ground|gt)s?$", "", stem)
        return stem

    masks_by_key: dict[str, list[str]] = {}
    for name in mask_names:
        masks_by_key.setdefault(stem_key(name), []).append(name)

    for image_name in image_names:
        candidates = masks_by_key.get(stem_key(image_name), [])
        if candidates:
            pairs.append((image_name, candidates[0]))
        if len(pairs) >= 3:
            break

    if pairs:
        return pairs

    for image_name, mask_name in zip(image_names, mask_names, strict=False):
        pairs.append((image_name, mask_name))
        if len(pairs) >= 3:
            break
    return pairs


def raw_suffix(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(path).suffix.lower()


def load_medical_image(path: Path, source_name: str, raw_bytes: bytes) -> np.ndarray:
    lower = source_name.lower()
    if lower.endswith(".dcm"):
        dataset = pydicom.dcmread(io.BytesIO(raw_bytes))
        array = dataset.pixel_array.astype(np.float32)
        slope_value = getattr(dataset, "RescaleSlope", 1.0)
        intercept_value = getattr(dataset, "RescaleIntercept", 0.0)
        slope = float(slope_value) if slope_value is not None else 1.0
        intercept = float(intercept_value) if intercept_value is not None else 0.0
        array = array * slope + intercept
        invert = getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1"
        return normalize_to_uint8(array, invert=invert)
    if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        array = np.asarray(Image.open(io.BytesIO(raw_bytes)))
        if array.ndim == 3:
            return array[:, :, :3].mean(axis=2).astype(np.uint8)
        return normalize_to_uint8(array)
    raise RuntimeError(f"Unsupported image format: {source_name}")


def load_medical_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        data = np.asarray(nib.load(str(path)).get_fdata())
        data = np.squeeze(data)
        while data.ndim > 2:
            data = data.max(axis=-1)
        mask = data > 0
    else:
        mask_array = np.asarray(Image.open(path))
        if mask_array.ndim == 3:
            mask = mask_array[:, :, :3].max(axis=2) > 0
        else:
            mask = mask_array > 0

    if mask.shape != image_shape and mask.T.shape == image_shape:
        mask = mask.T
    if mask.shape != image_shape:
        raise RuntimeError(
            f"Mask shape {mask.shape} does not match image shape {image_shape} for {path}"
        )
    return mask


def download_pitt_samples(out_dir: Path) -> list[SavedSample]:
    saved: list[SavedSample] = []
    dataset_dir = out_dir / "pitt_hexai_oai_hip_sample"
    with zipfile.ZipFile(RemoteHTTPFile(PITT_HIP_ZIP_URL)) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        (dataset_dir / "zip_listing.txt").parent.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "zip_listing.txt").write_text("\n".join(names) + "\n")
        pairs = find_pitt_pairs(names)
        if not pairs:
            raise RuntimeError("Could not infer image/mask pairs in Pitt hip sample ZIP")

        for index, (image_name, mask_name) in enumerate(pairs[:3], start=1):
            sample_id = f"sample_{index:02d}"
            sample_dir = dataset_dir / sample_id
            image_bytes = extract_member(zf, image_name)
            mask_bytes = extract_member(zf, mask_name)
            raw_image_path = sample_dir / f"image_raw{raw_suffix(image_name)}"
            raw_mask_path = sample_dir / f"mask_raw{raw_suffix(mask_name)}"
            write_bytes(raw_image_path, image_bytes)
            write_bytes(raw_mask_path, mask_bytes)

            image_u8 = load_medical_image(raw_image_path, image_name, image_bytes)
            mask = load_medical_mask(raw_mask_path, image_u8.shape)

            image_png = sample_dir / "image_display.png"
            mask_png = sample_dir / "mask_binary.png"
            overlay_png = sample_dir / "overlay_binary.png"
            Image.fromarray(image_u8).save(image_png)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_png)
            save_overlay(image_u8, mask, overlay_png)

            saved.append(
                SavedSample(
                    dataset="Pitt HexAI OAI hip sample",
                    sample_id=sample_id,
                    files=[
                        str(raw_image_path),
                        str(raw_mask_path),
                        str(image_png),
                        str(mask_png),
                        str(overlay_png),
                    ],
                    note=f"ZIP members: image={image_name}, mask={mask_name}",
                )
            )
    return saved


def write_roboflow_notes(out_dir: Path) -> list[dict[str, object]]:
    notes = []
    for source in ROBOFLOW_SOURCES:
        dataset_dir = out_dir / source["slug"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        note = {
            **source,
            "download_status": "not_downloaded",
            "reason": (
                "Raw image/mask export requires a Roboflow API key or interactive "
                "Universe download session. No ROBOFLOW_API_KEY was present locally."
            ),
            "how_to_populate": (
                "Set ROBOFLOW_API_KEY and export the project version in COCO "
                "segmentation or YOLO segmentation format, then copy a few "
                "image/label pairs into this folder."
            ),
        }
        (dataset_dir / "README.json").write_text(json.dumps(note, indent=2) + "\n")
        notes.append(note)
    return notes


def write_manifest(out_dir: Path, saved: list[SavedSample], roboflow_notes: list[dict[str, object]]) -> None:
    manifest = {
        "created_by": "app/scripts/download_pelvis_dataset_samples.py",
        "purpose": "Small visual sample set for pelvis/hip X-ray segmentation datasets.",
        "samples": [sample.__dict__ for sample in saved],
        "roboflow_sources": roboflow_notes,
        "sources": {
            "pengwin_task2": {
                "url": "https://zenodo.org/records/10913196",
                "license": "CC BY 4.0",
                "access": "Direct Zenodo byte-range extraction from train.zip",
            },
            "pitt_hexai_oai_hip_sample": {
                "url": "https://github.com/pitthexai/AI_Fairness_in_Hip_and_Knee_Bony_Anatomy_Segmentation",
                "access": "Direct GitHub LFS media byte-range extraction from Sample_Dataset/hip_sample.zip",
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = f"""\
    # Pelvis Segmentation Dataset Samples

    This folder contains small visual samples pulled from public pelvis/hip X-ray
    segmentation datasets. For each extracted sample, the folder contains the raw
    downloaded image, the raw downloaded segmentation file where available, a
    display-normalized PNG, a binary mask PNG, and an overlay PNG.

    ## Downloaded

    - `pengwin_task2_synthetic_xray/`: 3 synthetic C-arm X-ray samples from
      PENGWIN Task 2. The official masks are uint32 multi-label TIFF files, so
      `mask_binary_any_fragment.png` is a display helper formed by unioning all
      encoded pelvic fragments.
    - `pitt_hexai_oai_hip_sample/`: 3 hip radiograph samples from the Pitt HexAI
      sample dataset ZIP, with their manual segmentation masks.

    ## Not Downloaded Here

    The Roboflow Universe pelvis datasets are documented in per-source
    `README.json` files, but raw image/mask exports were not downloaded because
    Roboflow rejected anonymous API/export requests and no `ROBOFLOW_API_KEY`
    was configured locally. Those datasets can still be exported from Universe
    with a free Roboflow account/API key.

    ## Sources

    - PENGWIN Task 2: https://zenodo.org/records/10913196
    - Pitt HexAI sample dataset:
      https://github.com/pitthexai/AI_Fairness_in_Hip_and_Knee_Bony_Anatomy_Segmentation
    - Roboflow Linas WS Pelvis AP X-ray:
      https://universe.roboflow.com/linas-ws/pelvis-ap-x-ray-ys9pz
    - Roboflow YoloV8 pelvis_xray:
      https://universe.roboflow.com/yolov8-jl4qm/pelvis_xray
    - Roboflow xray Pelvis:
      https://universe.roboflow.com/xray-3l69l/pelvis-rh5xb-gluu8
    """
    (out_dir / "README.md").write_text(textwrap.dedent(readme))


def make_contact_sheet(paths: list[Path], output_path: Path, title: str) -> None:
    if not paths:
        return
    tile_w, tile_h = 360, 320
    label_h = 42
    sheet = Image.new("RGB", (tile_w * len(paths), tile_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(20, 20, 20))
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tile_w - 24, tile_h - 24))
        x0 = index * tile_w + (tile_w - image.width) // 2
        y0 = label_h + (tile_h - image.height) // 2
        sheet.paste(image, (x0, y0))
        draw.text((index * tile_w + 12, label_h - 18), path.parent.name, fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_contact_sheets(out_dir: Path) -> None:
    pengwin_dir = out_dir / "pengwin_task2_synthetic_xray"
    pitt_dir = out_dir / "pitt_hexai_oai_hip_sample"
    make_contact_sheet(
        sorted(pengwin_dir.glob("*/overlay_binary_any_fragment.png")),
        pengwin_dir / "contact_sheet.png",
        "PENGWIN Task 2 synthetic X-ray: image + pelvic fragment union mask",
    )
    make_contact_sheet(
        sorted(pitt_dir.glob("*/overlay_binary.png")),
        pitt_dir / "contact_sheet.png",
        "Pitt HexAI OAI hip sample: image + manual segmentation mask",
    )
    make_contact_sheet(
        sorted(pengwin_dir.glob("*/overlay_binary_any_fragment.png"))
        + sorted(pitt_dir.glob("*/overlay_binary.png")),
        out_dir / "overview_contact_sheet.png",
        "Pelvis segmentation dataset samples",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[SavedSample] = []
    saved.extend(download_pengwin_samples(OUT_DIR))
    saved.extend(download_pitt_samples(OUT_DIR))
    roboflow_notes = write_roboflow_notes(OUT_DIR)
    write_manifest(OUT_DIR, saved, roboflow_notes)
    write_contact_sheets(OUT_DIR)
    print(f"Wrote {len(saved)} image/mask samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
