# 2D Point Annotator App

To install the windows version:

``` powershell
powershell -ExecutionPolicy Bypass -c "irm https://is.gd/mS0yFr | iex"
```

And the unix version:

``` shell
curl -fsSL https://is.gd/XsFkW8 | bash
```

## Pelvis segmentation

The `Segment Pelvis` button uses the local segmentation backend in
`src/pelvis_segmentation.py`. The radio buttons are model choices, not runtime
formats. Available pelvis/hip radiograph model choices include:

- `HF YOLO11s pelvis`: downloads and caches the public pelvis-trained
  `Pelvis X-ray Segmentation (YOLO11s)` weights from
  `mbar0075/YOLO-Application-Toolkit` on Hugging Face.
- `Replicate U-Net++ pelvis`: calls `medicapture/seg-model`, a U-Net++ pelvis
  X-ray segmentation model.
- `Roboflow ericyum AP`, `Roboflow sooyeon AP`, and `Roboflow YOLOv8 pelvis`:
  call separate public Roboflow Universe pelvis AP X-ray instance-segmentation
  models.
- `OpenCV fallback`: non-ML fallback retained only as an editable last resort.

Install/update the Pixi environment before using the ML backends:

``` shell
pixi install
```

Optional overrides:

``` shell
# Use a specific local pelvis-trained YOLO model instead of the default download.
export PELVIS_SEGMENTATION_BACKEND=hf_yolo11s_pelvis_xray
export PELVIS_HF_YOLO11S_MODEL=/path/to/yolo11s_seg_pelvis_xray.pt

# Enable hosted model choices.
export ROBOFLOW_API_KEY=...
export REPLICATE_API_TOKEN=...

# Optional external HipPelvisAnnotator-style command, used by auto when set.
export PELVIS_SEGMENTATION_BACKEND=hip_pelvis_annotator
export HIP_PELVIS_ANNOTATOR_CMD='python /path/to/infer.py --input {input} --output {output}'
```

`Auto pelvis models` tries the local Hugging Face pelvis YOLO model first, then
the Replicate and Roboflow pelvis models when their API tokens are configured,
and finally the OpenCV option.
