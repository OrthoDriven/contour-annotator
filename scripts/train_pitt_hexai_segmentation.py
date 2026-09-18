"""Train a binary hip/pelvis segmentation model on Pitt HexAI-style data.

This script is intentionally focused on local training. It supports both:

- the small sample folder created by download_pelvis_dataset_samples.py:
  data/pelvis_segmentation_dataset_samples/pitt_hexai_oai_hip_sample/sample_*/...
- an extracted Pitt/OAI-style folder:
  HipSample/Images/*.dcm and HipSample/Annotations/*.nii.gz

The model uses the landmark backbones in the workspace-root src/ folder and
trains them as a single-channel binary segmentation network.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent
ROOT_SRC = WORKSPACE_DIR / "src"
if ROOT_SRC.exists():
    sys.path.insert(0, str(ROOT_SRC))

try:
    from high_resolution_landmark_network import HighResolutionLandmarkModel
    from hailo_landmark_network import HeatmapModelHailo
    from original_landmark_network import HeatmapModel
except ImportError as exc:  # pragma: no cover - gives a clearer CLI error.
    raise SystemExit(
        f"Could not import landmark backbones from {ROOT_SRC}. "
        "Run this script from the app Pixi environment inside the workspace."
    ) from exc


DEFAULT_DATA_ROOT = APP_DIR / "data" / "pelvis_segmentation_dataset_samples" / "pitt_hexai_oai_hip_sample"
DEFAULT_OUTPUT_DIR = APP_DIR / "models" / "pitt_hexai_segmentation"


@dataclass(frozen=True)
class ImageMaskPair:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    model: str
    image_size: int
    base_ch: int
    task_head_channels: int | None
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_fraction: float
    seed: int
    num_workers: int
    amp: bool
    threshold: float
    max_samples: int | None
    export_torchscript: bool
    positive_labels: list[int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary hip/pelvis segmentation on Pitt HexAI-style DICOM/NIfTI pairs."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("highres", "hailo", "original"),
        default="highres",
        help="Workspace-root src backbone to train as a binary segmentation model.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--task-head-channels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--positive-labels",
        default="1,2,3,4",
        help=(
            "Comma-separated NIfTI labels to treat as pelvis. "
            "Default 1,2,3,4 = acetabula plus ilium/ischium/pubis; "
            "use 'all' to train on every nonzero label, including femurs."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit discovered pairs for smoke tests.",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Also save best.torchscript.pt for app inference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover pairs and instantiate one batch; do not train.",
    )
    return parser.parse_args()


def parse_positive_labels(value: str) -> list[int] | None:
    normalized = value.strip().lower()
    if normalized in {"", "all", "nonzero", "*"}:
        return None
    labels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not labels:
        return None
    return sorted(set(labels))


def discover_pairs(data_root: Path) -> list[ImageMaskPair]:
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    pairs = discover_sample_folder_pairs(data_root)
    if not pairs:
        pairs = discover_hipsample_pairs(data_root)
    if not pairs:
        pairs = discover_generic_pairs(data_root)
    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found under {data_root}. Expected either "
            "sample_*/image_raw.dcm + mask_raw.nii.gz, Images/*.dcm + "
            "Annotations/*.nii.gz, or image_display.png + mask_binary.png."
        )
    return sorted(pairs, key=lambda p: p.sample_id)


def discover_sample_folder_pairs(data_root: Path) -> list[ImageMaskPair]:
    pairs: list[ImageMaskPair] = []
    for sample_dir in sorted(data_root.glob("sample_*")):
        image = first_existing(sample_dir, ["image_raw.dcm", "image_display.png"])
        mask = first_existing(sample_dir, ["mask_raw.nii.gz", "mask_binary.png"])
        if image and mask:
            pairs.append(ImageMaskPair(sample_dir.name, image, mask))
    return pairs


def discover_hipsample_pairs(data_root: Path) -> list[ImageMaskPair]:
    roots = [data_root]
    if (data_root / "data" / "HipSample").exists():
        roots.insert(0, data_root / "data" / "HipSample")
    if (data_root / "HipSample").exists():
        roots.insert(0, data_root / "HipSample")

    pairs: list[ImageMaskPair] = []
    for root in roots:
        images_dir = root / "Images"
        annotations_dir = root / "Annotations"
        if not images_dir.exists() or not annotations_dir.exists():
            continue
        masks_by_stem = {strip_compound_suffix(path): path for path in annotations_dir.glob("*.nii*")}
        for image_path in sorted(images_dir.glob("*.dcm")):
            sample_id = image_path.stem
            mask_path = masks_by_stem.get(sample_id)
            if mask_path:
                pairs.append(ImageMaskPair(sample_id, image_path, mask_path))
        if pairs:
            return pairs
    return pairs


def discover_generic_pairs(data_root: Path) -> list[ImageMaskPair]:
    pairs: list[ImageMaskPair] = []
    for image_path in sorted(data_root.rglob("image_display.png")):
        mask_path = image_path.parent / "mask_binary.png"
        if not mask_path.exists():
            mask_path = image_path.parent / "mask_binary_any_fragment.png"
        if mask_path.exists():
            pairs.append(ImageMaskPair(image_path.parent.name, image_path, mask_path))
    return pairs


def first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def strip_compound_suffix(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def split_pairs(
    pairs: list[ImageMaskPair],
    *,
    val_fraction: float,
    seed: int,
    max_samples: int | None,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    if max_samples is not None:
        pairs = pairs[: max(1, int(max_samples))]
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


class PittHipSegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        pairs: list[ImageMaskPair],
        *,
        image_size: int,
        augment: bool,
        seed: int,
        positive_labels: list[int] | None,
    ) -> None:
        self.pairs = pairs
        self.image_size = int(image_size)
        self.augment = augment
        self.rng = random.Random(seed)
        self.positive_labels = positive_labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        image = load_image(pair.image_path)
        mask = load_mask(pair.mask_path, image.shape, positive_labels=self.positive_labels)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255))

        if self.augment:
            image_pil, mask_pil = apply_augmentation(image_pil, mask_pil, self.rng)

        image_pil = ImageOps.contain(image_pil, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        mask_pil = ImageOps.contain(mask_pil, (self.image_size, self.image_size), method=Image.Resampling.NEAREST)
        image_pil = pad_to_square(image_pil, self.image_size, fill=0)
        mask_pil = pad_to_square(mask_pil, self.image_size, fill=0)

        image_arr = np.asarray(image_pil, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask_pil, dtype=np.uint8) > 0).astype(np.float32)
        image_arr = normalize_image(image_arr)

        return {
            "image": torch.from_numpy(image_arr[None, ...]),
            "mask": torch.from_numpy(mask_arr[None, ...]),
            "sample_id": pair.sample_id,
        }


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope_value = getattr(ds, "RescaleSlope", 1.0)
        intercept_value = getattr(ds, "RescaleIntercept", 0.0)
        slope = float(slope_value) if slope_value is not None else 1.0
        intercept = float(intercept_value) if intercept_value is not None else 0.0
        arr = arr * slope + intercept
        invert = getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1"
        return percentile_uint8(arr, invert=invert)

    with Image.open(path) as image:
        gray = image.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def load_mask(
    path: Path,
    image_shape: tuple[int, int],
    *,
    positive_labels: list[int] | None,
) -> np.ndarray:
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        data = np.asarray(nib.load(str(path)).get_fdata())
        data = np.squeeze(data)
        while data.ndim > 2:
            data = data.max(axis=-1)
        if positive_labels is None:
            mask = data > 0
        else:
            mask = np.isin(data.astype(np.int16), np.asarray(positive_labels, dtype=np.int16))
    else:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0

    if mask.shape != image_shape and mask.T.shape == image_shape:
        mask = mask.T
    if mask.shape != image_shape:
        resized = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
            (image_shape[1], image_shape[0]),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(resized, dtype=np.uint8) > 0
    return mask


def percentile_uint8(arr: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite]
    low, high = np.percentile(valid, [1.0, 99.5])
    if high <= low:
        low = float(valid.min())
        high = float(valid.max())
    scaled = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
    if invert:
        scaled = 1.0 - scaled
    return (scaled * 255).astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        return image - mean
    return (image - mean) / std


def pad_to_square(image: Image.Image, size: int, *, fill: int) -> Image.Image:
    out = Image.new(image.mode, (size, size), color=fill)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    out.paste(image, (x, y))
    return out


def apply_augmentation(
    image: Image.Image,
    mask: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if rng.random() < 0.25:
        angle = rng.uniform(-8.0, 8.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    if rng.random() < 0.5:
        arr = np.asarray(image, dtype=np.float32)
        contrast = rng.uniform(0.85, 1.2)
        brightness = rng.uniform(-0.05, 0.05) * 255.0
        arr = np.clip((arr - arr.mean()) * contrast + arr.mean() + brightness, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))
    return image, mask


def build_model(name: str, *, base_ch: int, task_head_channels: int | None) -> nn.Module:
    if name == "highres":
        return HighResolutionLandmarkModel(
            in_ch=1,
            base_ch=base_ch,
            out_ch=1,
            task_head_channels=task_head_channels,
        )
    if name == "hailo":
        return HeatmapModelHailo(in_ch=1, base_ch=base_ch, out_ch=1)
    if name == "original":
        return HeatmapModel(in_ch=1, base_ch=base_ch, out_ch=1)
    raise ValueError(f"Unknown model: {name}")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
) -> tuple[float, float]:
    preds = torch.sigmoid(logits) >= threshold
    labels = targets >= 0.5
    dims = tuple(range(1, preds.ndim))
    intersection = (preds & labels).sum(dim=dims).float()
    union = (preds | labels).sum(dim=dims).float()
    pred_sum = preds.sum(dim=dims).float()
    label_sum = labels.sum(dim=dims).float()
    dice = (2.0 * intersection + 1e-6) / (pred_sum + label_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice.mean().item()), float(iou.mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    autocast_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks)
        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    return total / max(1, count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    dice_total = 0.0
    iou_total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = segmentation_loss(logits, masks)
        dice, iou = segmentation_metrics(logits, masks, threshold=threshold)
        batch_size = images.shape[0]
        loss_total += float(loss.item()) * batch_size
        dice_total += dice * batch_size
        iou_total += iou * batch_size
        count += batch_size
    count = max(1, count)
    return {
        "loss": loss_total / count,
        "dice": dice_total / count,
        "iou": iou_total / count,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    config: TrainConfig,
    train_pairs: list[ImageMaskPair],
    val_pairs: list[ImageMaskPair],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "train_sample_ids": [p.sample_id for p in train_pairs],
            "val_sample_ids": [p.sample_id for p in val_pairs],
        },
        path,
    )


def export_torchscript(model: nn.Module, path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    example = torch.zeros(1, 1, image_size, image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(path))


@torch.no_grad()
def write_preview(
    model: nn.Module,
    dataset: PittHipSegmentationDataset,
    output_path: Path,
    device: torch.device,
    *,
    threshold: float,
) -> None:
    if len(dataset) == 0:
        return
    item = dataset[0]
    image_tensor = item["image"].unsqueeze(0).to(device)
    logits = model(image_tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = prob >= threshold
    image = item["image"][0].numpy()
    image = percentile_uint8(image)
    target = item["mask"][0].numpy() > 0.5
    rgb = np.stack([image, image, image], axis=-1)
    overlay = rgb.copy()
    overlay[target] = np.array([20, 170, 60], dtype=np.uint8)
    overlay[pred] = np.array([255, 70, 40], dtype=np.uint8)
    blended = (0.55 * rgb + 0.45 * overlay).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)


def write_manifest(
    output_dir: Path,
    config: TrainConfig,
    train_pairs: list[ImageMaskPair],
    val_pairs: list[ImageMaskPair],
) -> None:
    manifest = {
        "config": asdict(config),
        "root_src": str(ROOT_SRC),
        "train_pairs": [
            {"sample_id": p.sample_id, "image_path": str(p.image_path), "mask_path": str(p.mask_path)}
            for p in train_pairs
        ],
        "val_pairs": [
            {"sample_id": p.sample_id, "image_path": str(p.image_path), "mask_path": str(p.mask_path)}
            for p in val_pairs
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def append_history(output_dir: Path, row: dict[str, float | int]) -> None:
    path = output_dir / "history.csv"
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> int:
    args = parse_args()
    set_reproducible(args.seed)
    positive_labels = parse_positive_labels(args.positive_labels)

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pairs = discover_pairs(data_root)
    train_pairs, val_pairs = split_pairs(
        pairs,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    config = TrainConfig(
        data_root=str(data_root),
        output_dir=str(output_dir),
        model=args.model,
        image_size=args.image_size,
        base_ch=args.base_ch,
        task_head_channels=args.task_head_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        threshold=args.threshold,
        max_samples=args.max_samples,
        export_torchscript=args.export_torchscript,
        positive_labels=positive_labels,
    )
    write_manifest(output_dir, config, train_pairs, val_pairs)

    print(f"Discovered {len(pairs)} pairs under {data_root}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Output: {output_dir}")

    train_dataset = PittHipSegmentationDataset(
        train_pairs,
        image_size=args.image_size,
        augment=True,
        seed=args.seed + 1,
        positive_labels=positive_labels,
    )
    val_dataset = PittHipSegmentationDataset(
        val_pairs,
        image_size=args.image_size,
        augment=False,
        seed=args.seed + 2,
        positive_labels=positive_labels,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, base_ch=args.base_ch, task_head_channels=args.task_head_channels).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Model: {args.model} ({parameter_count:,} trainable parameters)")

    first_batch = next(iter(train_loader))
    print(
        "First batch:",
        tuple(first_batch["image"].shape),
        tuple(first_batch["mask"].shape),
        list(first_batch["sample_id"]),
    )
    if args.dry_run:
        print("Dry run complete; no training performed.")
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if (not args.no_amp and device.type == "cuda") else None

    best_dice = -math.inf
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_amp=not args.no_amp,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        append_history(output_dir, row)
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dice,
            config=config,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
            )
            write_preview(
                model,
                val_dataset,
                output_dir / "best_preview.png",
                device,
                threshold=args.threshold,
            )
            if args.export_torchscript:
                export_torchscript(model, output_dir / "best.torchscript.pt", args.image_size, device)

        elapsed_min = (time.time() - start_time) / 60.0
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"lr={current_lr:.2e} "
            f"elapsed={elapsed_min:.1f}m"
        )

    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved best checkpoint: {output_dir / 'best.pt'}")
    if args.export_torchscript:
        print(f"Saved TorchScript model: {output_dir / 'best.torchscript.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
