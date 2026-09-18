from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

cv2 = pytest.importorskip("cv2")

from pelvis_segmentation import PelvisSegmentationBackend


def test_yolo_result_to_mask_prefers_named_pelvis_class():
    masks = np.zeros((2, 80, 90), dtype=np.float32)
    masks[0, 62:74, 32:44] = 1.0
    masks[1, 10:58, 8:82] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(
            cls=np.array([0, 1], dtype=np.float32),
            conf=np.array([0.95, 0.72], dtype=np.float32),
        ),
        names={0: "femur", 1: "pelvis"},
    )

    backend = PelvisSegmentationBackend(mode="pelvis_yolo")
    mask = backend._ultralytics_result_to_mask(result, (80, 90), None)

    assert mask is not None
    assert mask.shape == (80, 90)
    assert mask[24, 44] == 1
    assert mask[68, 38] == 0


def test_roboflow_payload_to_mask_rasterizes_pelvis_polygons_only():
    payload = {
        "predictions": [
            {
                "class": "FemurL",
                "confidence": 0.99,
                "points": [
                    {"x": 4, "y": 44},
                    {"x": 20, "y": 44},
                    {"x": 20, "y": 58},
                    {"x": 4, "y": 58},
                ],
            },
            {
                "class": "IliumL",
                "confidence": 0.88,
                "points": [
                    {"x": 10, "y": 10},
                    {"x": 70, "y": 10},
                    {"x": 70, "y": 42},
                    {"x": 10, "y": 42},
                ],
            },
        ]
    }

    backend = PelvisSegmentationBackend(mode="rf_yolov8_pelvis_xray")
    mask = backend._roboflow_payload_to_mask(payload, (64, 80))

    assert mask is not None
    assert mask.shape == (64, 80)
    assert mask[22, 30] == 1
    assert mask[50, 12] == 0


def test_replicate_output_to_mask_resizes_probability_image():
    output = np.zeros((20, 30), dtype=np.uint8)
    output[4:16, 6:24] = 220

    backend = PelvisSegmentationBackend(mode="replicate_unetpp_pelvis")
    mask = backend._replicate_output_to_mask(Image.fromarray(output), (40, 60))

    assert mask is not None
    assert mask.shape == (40, 60)
    assert int(mask.sum()) > 200


def test_opencv_backend_segments_synthetic_pelvis_shape():
    img = np.zeros((160, 180), dtype=np.uint8)
    cv2.ellipse(img, (90, 76), (62, 34), 0, 0, 360, 150, -1)
    cv2.ellipse(img, (58, 84), (24, 34), -18, 0, 360, 225, -1)
    cv2.ellipse(img, (122, 84), (24, 34), 18, 0, 360, 225, -1)
    cv2.rectangle(img, (84, 72), (96, 132), 190, -1)
    cv2.line(img, (132, 88), (152, 152), 255, 4)

    backend = PelvisSegmentationBackend(mode="opencv")
    result = backend.segment(Image.fromarray(img))

    assert result.backend == "OpenCV heuristic"
    assert result.mask.shape == img.shape
    assert result.mask.dtype == np.uint8
    assert int(result.mask.sum()) > 1200
    assert result.mask[80, 90] == 1
    assert result.mask[5, 5] == 0


def test_pitt_hexai_backend_loads_training_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(root_src))
    from high_resolution_landmark_network import HighResolutionLandmarkModel

    model = HighResolutionLandmarkModel(
        in_ch=1,
        base_ch=4,
        out_ch=1,
        task_head_channels=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.hm_head[-1].bias.fill_(6.0)

    checkpoint_path = tmp_path / "pitt_highres.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": "highres",
                "image_size": 64,
                "base_ch": 4,
                "task_head_channels": 4,
                "threshold": 0.5,
            },
        },
        checkpoint_path,
    )
    monkeypatch.setenv("PELVIS_PITT_HEXAI_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("PELVIS_PITT_HEXAI_DEVICE", "cpu")

    img = np.zeros((48, 56), dtype=np.uint8)
    img[8:42, 10:48] = 180

    backend = PelvisSegmentationBackend(mode="pitt_hexai_highres")
    result = backend.segment(Image.fromarray(img))

    assert result.backend == "Local Pitt HexAI HighRes"
    assert result.mask.shape == img.shape
    assert result.mask.dtype == np.uint8
    assert int(result.mask.sum()) > 0
