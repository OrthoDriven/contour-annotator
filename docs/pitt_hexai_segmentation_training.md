# Pitt HexAI Segmentation Training

Use these commands from the app directory:

```bash
cd <repo-root>
```

## Check GPU

```bash
pixi run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Smoke Test On The 3 Downloaded Samples

This only proves the pipeline works. It is too small to make a useful model.

```bash
pixi run python scripts/train_pitt_hexai_segmentation.py --dry-run --data-root data/pelvis_segmentation_dataset_samples/pitt_hexai_oai_hip_sample --output-dir models/pitt_hexai_segmentation/dryrun --model highres --image-size 256 --base-ch 16 --batch-size 1 --num-workers 0
```

```bash
pixi run python scripts/train_pitt_hexai_segmentation.py --data-root data/pelvis_segmentation_dataset_samples/pitt_hexai_oai_hip_sample --output-dir models/pitt_hexai_segmentation/highres_3_sample_smoke --model highres --image-size 256 --base-ch 16 --batch-size 1 --epochs 20 --lr 2e-4 --num-workers 0 --export-torchscript
```

## Download The Full Pitt HexAI Hip Sample ZIP

The public GitHub LFS hip sample is about 424 MB and contains many more DICOM/NIfTI pairs than the 3-case visual sample folder.

```bash
mkdir -p data/pitt_hexai_hip_sample_full
curl -L "https://media.githubusercontent.com/media/pitthexai/AI_Fairness_in_Hip_and_Knee_Bony_Anatomy_Segmentation/main/Sample_Dataset/hip_sample.zip" -o data/pitt_hexai_hip_sample_full/hip_sample.zip
pixi run python -m zipfile -e data/pitt_hexai_hip_sample_full/hip_sample.zip data/pitt_hexai_hip_sample_full
```

## Train On The Full Pitt HexAI Hip Sample

Start here on the GPU:

```bash
pixi run python scripts/train_pitt_hexai_segmentation.py --data-root data/pitt_hexai_hip_sample_full/data/HipSample --output-dir models/pitt_hexai_segmentation/highres_pitt_512 --model highres --image-size 512 --base-ch 32 --batch-size 2 --epochs 120 --lr 2e-4 --weight-decay 1e-4 --num-workers 4 --export-torchscript
```

By default the trainer uses Pitt hip labels `1,2,3,4` as positive pelvis labels:
right/left acetabulum plus right/left ilium/ischium/pubis. Pitt labels `5,6`
are femurs and are excluded. To reproduce the older all-bone behavior, add
`--positive-labels all`.

If GPU memory allows it:

```bash
pixi run python scripts/train_pitt_hexai_segmentation.py --data-root data/pitt_hexai_hip_sample_full/data/HipSample --output-dir models/pitt_hexai_segmentation/highres_pitt_768 --model highres --image-size 768 --base-ch 32 --batch-size 2 --epochs 120 --lr 2e-4 --weight-decay 1e-4 --num-workers 4 --export-torchscript
```

If you hit CUDA out-of-memory:

```bash
pixi run python scripts/train_pitt_hexai_segmentation.py --data-root data/pitt_hexai_hip_sample_full/data/HipSample --output-dir models/pitt_hexai_segmentation/highres_pitt_512_base16 --model highres --image-size 512 --base-ch 16 --batch-size 1 --epochs 120 --lr 2e-4 --weight-decay 1e-4 --num-workers 2 --export-torchscript
```

## Outputs

Each run writes:

- `best.pt`: best validation Dice checkpoint.
- `last.pt`: latest checkpoint.
- `best.torchscript.pt`: traced model if `--export-torchscript` was passed.
- `history.csv`: epoch metrics.
- `best_preview.png`: validation image overlay, with green ground truth and red prediction.
- `manifest.json`: exact data split and training config.

## Preview A Checkpoint

```bash
python scripts/preview_pitt_hexai_segmentation.py --checkpoint models/pitt_hexai_segmentation/highres_pitt_512/best.pt --output-dir data/pitt_hexai_segmentation_previews
```

The app now has a `Local Pitt HighRes` pelvis segmentation backend option. By
default it loads:

```text
models/pitt_hexai_segmentation/highres_pitt_512/best.pt
```

To point it at another checkpoint:

```bash
export PELVIS_PITT_HEXAI_CHECKPOINT=/absolute/path/to/best.pt
```

The app backend uses automatic grayscale polarity by default for this Pitt
checkpoint: it tries normal and inverted grayscale and keeps the more
pelvis-plausible mask. To force one polarity for a run:

```bash
export PELVIS_PITT_HEXAI_INVERT_GRAYSCALE=1
```

or:

```bash
export PELVIS_PITT_HEXAI_INVERT_GRAYSCALE=0
```

For fluoroscopy, the backend also removes frame-like edge components by default
before saving the mask. To disable that:

```bash
export PELVIS_PITT_HEXAI_REMOVE_FRAME=0
```

## Notes

The model choices are:

- `--model highres`: uses `../src/high_resolution_landmark_network.py`; default and recommended.
- `--model hailo`: uses `../src/hailo_landmark_network.py`.
- `--model original`: uses `../src/original_landmark_network.py`.

For a real backend model, train on the full available Pitt/OAI cases, not the 3-case visual sample folder.
