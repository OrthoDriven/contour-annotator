from __future__ import annotations

"""Pelvis segmentation backends for AP/PA hip radiographs.

The visible ML choices are actual trained pelvis/hip X-ray segmentation models,
not runtime formats. Some run locally from cached weights and some call hosted
model endpoints. The OpenCV option is intentionally kept as a non-ML fallback
for offline environments or broken model installations.
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - exercised by app runtime environments
    cv2 = None


Box = Tuple[int, int, int, int]

HF_YOLO11S_PELVIS_MODEL_URL = (
    "https://huggingface.co/mbar0075/YOLO-Application-Toolkit/resolve/main/"
    "yolo11s_seg_pelvis_xray.pt"
)
HF_YOLO11S_PELVIS_MODEL_NAME = "yolo11s_seg_pelvis_xray.pt"
REPLICATE_MEDICAPTURE_VERSION = (
    "d51c0e7cb948983e4f1a74c1a0467e0a6805e797fb2e4055eabba2cca4b71a6b"
)
PELVIS_YOLO_EXCLUDE_CLASS_TERMS = ("femur", "trochanter")
PELVIS_YOLO_INCLUDE_CLASS_TERMS = (
    "acetabulum",
    "ilium",
    "iliac",
    "iliak",
    "ischium",
    "obturator",
    "pelvic",
    "pelvis",
    "pubis",
    "sacro",
    "sacrum",
    "sakro",
    "coccyx",
    "suorcil",
    "teardrop",
    "tear drop",
)

PELVIS_LOCAL_YOLO_MODELS = {
    "hf_yolo11s_pelvis_xray": {
        "label": "HF YOLO11s pelvis X-ray",
        "url_env": "PELVIS_HF_YOLO11S_URL",
        "path_env": "PELVIS_HF_YOLO11S_MODEL",
        "url": HF_YOLO11S_PELVIS_MODEL_URL,
        "filename": HF_YOLO11S_PELVIS_MODEL_NAME,
    },
    # Backward-compatible alias for older env/config values.
    "pelvis_yolo": {
        "label": "HF YOLO11s pelvis X-ray",
        "url_env": "PELVIS_YOLO_MODEL_URL",
        "path_env": "PELVIS_YOLO_MODEL",
        "url": HF_YOLO11S_PELVIS_MODEL_URL,
        "filename": HF_YOLO11S_PELVIS_MODEL_NAME,
    },
}

PELVIS_ROBOFLOW_MODELS = {
    "rf_ericyum_pelvis_ap": {
        "label": "Roboflow ericyum pelvis AP",
        "model_id": "pelvis-ap-x-ray-ys9pz-uviss/1",
        "model_id_env": "PELVIS_ROBOFLOW_ERICYUM_MODEL_ID",
    },
    "rf_sooyeon_pelvis_ap": {
        "label": "Roboflow sooyeon pelvis AP",
        "model_id": "pelvis-ap-x-ray-ys9pz-bhoxd/1",
        "model_id_env": "PELVIS_ROBOFLOW_SOOYEON_MODEL_ID",
    },
    "rf_yolov8_pelvis_xray": {
        "label": "Roboflow YOLOv8 pelvis_xray",
        "model_id": "pelvis_xray/1",
        "model_id_env": "PELVIS_ROBOFLOW_YOLOV8_MODEL_ID",
    },
}

PITT_HEXAI_HIGHRES_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pitt_hexai_segmentation"
    / "highres_pitt_512"
    / "best.pt"
)


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    backend: str
    detail: str


class PelvisSegmentationError(RuntimeError):
    pass


class PelvisSegmentationBackend:
    """Segment pelvic bone with an optional MedSAM/SAM model and OpenCV fallback."""

    def __init__(self, mode: str | None = None) -> None:
        configured = (mode or os.environ.get("PELVIS_SEGMENTATION_BACKEND") or "auto")
        self.mode = configured.strip().lower()
        self.model_path = (
            os.environ.get("PELVIS_SEGMENTATION_MODEL")
            or os.environ.get("MEDSAM_CHECKPOINT")
            or ""
        ).strip()
        self.model_type = (
            os.environ.get("PELVIS_SEGMENTATION_MODEL_TYPE") or "vit_b"
        ).strip()
        self.device = (os.environ.get("PELVIS_SEGMENTATION_DEVICE") or "cpu").strip()
        self.sam2_checkpoint = (
            os.environ.get("PELVIS_SEGMENTATION_SAM2_CHECKPOINT")
            or os.environ.get("SAM2_CHECKPOINT")
            or ""
        ).strip()
        self.sam2_config = (
            os.environ.get("PELVIS_SEGMENTATION_SAM2_CONFIG")
            or os.environ.get("SAM2_CONFIG")
            or "configs/sam2.1/sam2.1_hiera_l.yaml"
        ).strip()
        self.sam2_hf_id = (
            os.environ.get("PELVIS_SEGMENTATION_SAM2_HF_ID")
            or os.environ.get("SAM2_HF_ID")
            or ""
        ).strip()
        self.torchscript_model_path = (
            os.environ.get("PELVIS_TORCHSCRIPT_MODEL")
            or os.environ.get("PELVIS_SEGMENTATION_TORCHSCRIPT")
            or ""
        ).strip()
        self.onnx_model_path = (
            os.environ.get("PELVIS_ONNX_MODEL")
            or os.environ.get("PELVIS_SEGMENTATION_ONNX")
            or ""
        ).strip()
        self.hip_pelvis_annotator_cmd = (
            os.environ.get("HIP_PELVIS_ANNOTATOR_CMD") or ""
        ).strip()
        self.yolo_model_path = (
            os.environ.get("PELVIS_YOLO_MODEL")
            or os.environ.get("PELVIS_SEGMENTATION_YOLO_MODEL")
            or ""
        ).strip()
        self.yolo_model_url = (
            os.environ.get("PELVIS_YOLO_MODEL_URL") or HF_YOLO11S_PELVIS_MODEL_URL
        ).strip()
        self.pitt_hexai_checkpoint = (
            os.environ.get("PELVIS_PITT_HEXAI_CHECKPOINT")
            or str(PITT_HEXAI_HIGHRES_DEFAULT_CHECKPOINT)
        ).strip()
        self._sam_predictor = None
        self._sam2_predictor = None
        self._torchscript_model = None
        self._onnx_session = None
        self._onnx_dnn_net = None
        self._ultralytics_models: dict[str, Any] = {}
        self._pitt_hexai_model = None
        self._pitt_hexai_model_config: dict[str, Any] = {}

    def segment(self, image: Image.Image) -> SegmentationResult:
        valid_modes = {
            "auto",
            "hip_pelvis_annotator",
            "pelvis_onnx",
            "pelvis_torchscript",
            "pitt_hexai_highres",
            "replicate_unetpp_pelvis",
            "medsam",
            "sam",
            "sam2",
            "torchscript",
            "onnx",
            "opencv",
        } | set(PELVIS_LOCAL_YOLO_MODELS) | set(PELVIS_ROBOFLOW_MODELS)
        if self.mode not in valid_modes:
            raise PelvisSegmentationError(
                "PELVIS_SEGMENTATION_BACKEND must be auto, one of the named pelvis "
                "ML model modes, or opencv."
            )

        if self.mode == "auto":
            model_order = [
                "pitt_hexai_highres",
                "hf_yolo11s_pelvis_xray",
                "replicate_unetpp_pelvis",
                "rf_ericyum_pelvis_ap",
                "rf_sooyeon_pelvis_ap",
                "rf_yolov8_pelvis_xray",
                "hip_pelvis_annotator",
            ]
            for mode in model_order:
                try:
                    result = self._segment_named_model(image, mode)
                    if result is not None:
                        return result
                except PelvisSegmentationError:
                    continue
                except Exception:
                    continue
            return self._segment_with_opencv(image)

        if self.mode in PELVIS_LOCAL_YOLO_MODELS:
            try:
                result = self._segment_named_model(image, self.mode)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                raise
            except Exception as exc:
                raise PelvisSegmentationError(str(exc)) from exc

        if self.mode == "replicate_unetpp_pelvis":
            return self._segment_with_replicate_unetpp(image)

        if self.mode == "pitt_hexai_highres":
            return self._segment_with_pitt_hexai_highres(image)

        if self.mode in PELVIS_ROBOFLOW_MODELS:
            return self._segment_with_roboflow_model(image, self.mode)

        if self.mode == "hip_pelvis_annotator":
            try:
                result = self._segment_with_hip_pelvis_annotator(image)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                raise
            except Exception as exc:
                raise PelvisSegmentationError(str(exc)) from exc

        if self.mode in {"pelvis_onnx", "onnx"}:
            try:
                result = self._segment_with_onnx(image)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                raise
            except Exception as exc:
                raise PelvisSegmentationError(str(exc)) from exc

        if self.mode in {"pelvis_torchscript", "torchscript"}:
            try:
                result = self._segment_with_torchscript(image)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                raise
            except Exception as exc:
                raise PelvisSegmentationError(str(exc)) from exc

        if self.mode == "sam2":
            try:
                result = self._segment_with_sam2(image)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                if self.mode == "sam2":
                    raise
            except Exception as exc:
                if self.mode == "sam2":
                    raise PelvisSegmentationError(str(exc)) from exc

        if self.mode in {"medsam", "sam"}:
            try:
                result = self._segment_with_medsam(image)
                if result is not None:
                    return result
            except PelvisSegmentationError:
                if self.mode in {"medsam", "sam"}:
                    raise
            except Exception as exc:
                if self.mode in {"medsam", "sam"}:
                    raise PelvisSegmentationError(str(exc)) from exc

        if self.mode in {"auto", "opencv"}:
            return self._segment_with_opencv(image)

        raise PelvisSegmentationError("No pelvis segmentation backend is available.")

    def _segment_named_model(
        self, image: Image.Image, mode: str
    ) -> SegmentationResult | None:
        if mode in PELVIS_LOCAL_YOLO_MODELS:
            return self._segment_with_local_yolo_model(image, mode)
        if mode == "pitt_hexai_highres":
            return self._segment_with_pitt_hexai_highres(image)
        if mode == "replicate_unetpp_pelvis":
            return self._segment_with_replicate_unetpp(image)
        if mode in PELVIS_ROBOFLOW_MODELS:
            return self._segment_with_roboflow_model(image, mode)
        if mode == "hip_pelvis_annotator":
            return self._segment_with_hip_pelvis_annotator(image)
        return None

    def _segment_with_local_yolo_model(
        self, image: Image.Image, mode: str
    ) -> SegmentationResult:
        spec = PELVIS_LOCAL_YOLO_MODELS[mode]
        model_path = self._ensure_local_yolo_model_path(mode)
        return self._segment_with_ultralytics_model(
            image,
            model_path,
            backend=str(spec["label"]),
            detail=model_path.name,
        )

    def _segment_with_pitt_hexai_highres(self, image: Image.Image) -> SegmentationResult:
        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise PelvisSegmentationError(
                'Install PyTorch to use the local Pitt HexAI backend.'
            ) from exc

        checkpoint_path = Path(self.pitt_hexai_checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise PelvisSegmentationError(
                f"Pitt HexAI checkpoint not found: {checkpoint_path}"
            )

        model, config = self._load_pitt_hexai_model(checkpoint_path, torch)
        input_size = int(
            os.environ.get(
                "PELVIS_PITT_HEXAI_IMAGE_SIZE",
                str(config.get("image_size") or 512),
            )
        )
        input_size = max(128, min(2048, input_size))
        threshold = float(
            os.environ.get(
                "PELVIS_PITT_HEXAI_THRESHOLD",
                str(config.get("threshold") or 0.5),
            )
        )
        threshold = max(0.05, min(0.95, threshold))
        device = next(model.parameters()).device
        polarity = (
            os.environ.get("PELVIS_PITT_HEXAI_INVERT_GRAYSCALE") or "auto"
        ).strip().lower()
        if polarity in {"", "auto"}:
            polarity_options = [False, True]
            polarity_label = "auto"
        else:
            polarity_options = [_env_flag("PELVIS_PITT_HEXAI_INVERT_GRAYSCALE")]
            polarity_label = "inverted" if polarity_options[0] else "normal"

        best_mask: np.ndarray | None = None
        best_score = -1e9
        best_inverted = False
        original_w, original_h = image.size
        for invert_grayscale in polarity_options:
            tensor, content_box, original_size = _pitt_hexai_image_to_tensor(
                image, input_size, invert_grayscale=invert_grayscale
            )
            original_w, original_h = original_size
            tensor = tensor.to(device)
            with torch.inference_mode():
                logits = model(tensor)
                if logits.shape[-2:] != (input_size, input_size):
                    logits = F.interpolate(
                        logits,
                        size=(input_size, input_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

            x0, y0, x1, y1 = content_box
            prob = prob[y0:y1, x0:x1]
            prob_img = Image.fromarray(
                (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
            )
            prob_img = prob_img.resize(
                (original_w, original_h), Image.Resampling.BILINEAR
            )
            raw_mask = (
                np.asarray(prob_img) >= int(round(threshold * 255))
            ).astype(np.uint8)

            if cv2 is not None and _env_flag(
                "PELVIS_PITT_HEXAI_REMOVE_FRAME", default=True
            ):
                raw_mask = self._remove_frame_like_edge_components(raw_mask)

            if _env_flag("PELVIS_PITT_HEXAI_INTERSECT_PRIOR", default=False):
                raw_mask = (
                    (raw_mask > 0) & (_pelvis_prior_mask((original_h, original_w)) > 0)
                ).astype(np.uint8)

            if cv2 is not None:
                mask = self._postprocess_binary_mask(
                    raw_mask, (original_h, original_w), allow_prior_on_small=False
                )
            else:
                mask = raw_mask if bool(np.any(raw_mask)) else None
            if mask is None:
                continue

            score = self._score_pelvis_mask(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                best_inverted = invert_grayscale

        if best_mask is None:
            raise PelvisSegmentationError(
                "Pitt HexAI HighRes returned an empty pelvis mask."
            )

        return SegmentationResult(
            mask=best_mask.astype(np.uint8),
            backend="Local Pitt HexAI HighRes",
            detail=(
                f"{checkpoint_path.name}, threshold={threshold:.2f}, "
                f"polarity={polarity_label}->{('inverted' if best_inverted else 'normal')}"
            ),
        )

    def _load_pitt_hexai_model(self, checkpoint_path: Path, torch: Any) -> tuple[Any, dict[str, Any]]:
        if self._pitt_hexai_model is not None:
            return self._pitt_hexai_model, self._pitt_hexai_model_config

        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise PelvisSegmentationError(
                f"Pitt HexAI checkpoint is not a supported training checkpoint: {checkpoint_path}"
            )

        config = dict(checkpoint.get("config") or {})
        model_name = str(config.get("model") or "highres")
        base_ch = int(config.get("base_ch") or 32)
        task_head_channels = config.get("task_head_channels")
        if task_head_channels is not None:
            task_head_channels = int(task_head_channels)

        model = _build_pitt_hexai_model(
            model_name=model_name,
            base_ch=base_ch,
            task_head_channels=task_head_channels,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        requested_device = (
            os.environ.get("PELVIS_PITT_HEXAI_DEVICE")
            or os.environ.get("PELVIS_SEGMENTATION_DEVICE")
            or self.device
            or "cpu"
        ).strip()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        device = torch.device(requested_device)
        model.to(device)
        model.eval()
        self._pitt_hexai_model = model
        self._pitt_hexai_model_config = config
        return model, config

    def _segment_with_ultralytics_model(
        self,
        image: Image.Image,
        model_path: Path,
        backend: str,
        detail: str,
    ) -> SegmentationResult:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PelvisSegmentationError(
                'Install the ML backend first: "pixi install" or '
                '"pixi add --pypi ultralytics onnx onnxruntime".'
            ) from exc

        cache_key = str(model_path)
        model = self._ultralytics_models.get(cache_key)
        if model is None:
            model = YOLO(cache_key, task="segment")
            self._ultralytics_models[cache_key] = model

        rgb = np.array(image.convert("RGB"))
        image_size = int(os.environ.get("PELVIS_YOLO_IMAGE_SIZE", "1024"))
        image_size = max(256, min(2048, image_size))
        conf = float(os.environ.get("PELVIS_YOLO_CONF", "0.01"))
        predict_kwargs = {
            "source": rgb,
            "imgsz": image_size,
            "conf": conf,
            "verbose": False,
            "retina_masks": True,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        try:
            results = model.predict(**predict_kwargs)
        except Exception as exc:
            raise PelvisSegmentationError(
                f"{backend} prediction failed: {exc}"
            ) from exc

        if not results:
            raise PelvisSegmentationError(f"{backend} did not return a result.")

        mask = self._ultralytics_result_to_mask(
            results[0],
            rgb.shape[:2],
            getattr(model, "names", None),
        )
        if mask is None:
            raise PelvisSegmentationError(f"{backend} did not return a pelvis mask.")

        return SegmentationResult(mask=mask, backend=backend, detail=detail)

    def _segment_with_roboflow_model(
        self, image: Image.Image, mode: str
    ) -> SegmentationResult:
        api_key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip()
        if not api_key:
            raise PelvisSegmentationError(
                "Set ROBOFLOW_API_KEY to use Roboflow pelvis segmentation models."
            )
        spec = PELVIS_ROBOFLOW_MODELS[mode]
        model_id = (
            os.environ.get(str(spec["model_id_env"])) or str(spec["model_id"])
        ).strip()
        api_url = (
            os.environ.get("PELVIS_ROBOFLOW_API_URL")
            or os.environ.get("ROBOFLOW_API_URL")
            or "https://outline.roboflow.com"
        ).strip().rstrip("/")
        confidence = int(float(os.environ.get("PELVIS_ROBOFLOW_CONFIDENCE", "10")))
        confidence = max(1, min(100, confidence))

        try:
            import base64
            import io
            import requests
        except Exception as exc:
            raise PelvisSegmentationError(
                "The requests package is required to call Roboflow models."
            ) from exc

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue())
        url = f"{api_url}/{model_id}"
        try:
            response = requests.post(
                url,
                params={"api_key": api_key, "confidence": confidence},
                data=encoded,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=(10, float(os.environ.get("PELVIS_ROBOFLOW_TIMEOUT", "120"))),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise PelvisSegmentationError(
                f"{spec['label']} inference failed: {exc}"
            ) from exc

        rgb = np.array(image.convert("RGB"))
        mask = self._roboflow_payload_to_mask(payload, rgb.shape[:2])
        if mask is None:
            raise PelvisSegmentationError(
                f"{spec['label']} did not return a pelvis polygon mask."
            )
        return SegmentationResult(
            mask=mask,
            backend=str(spec["label"]),
            detail=model_id,
        )

    def _roboflow_payload_to_mask(
        self, payload: dict[str, Any], shape: tuple[int, int]
    ) -> np.ndarray | None:
        if cv2 is None:
            raise PelvisSegmentationError("OpenCV is required to rasterize polygons.")

        h, w = shape
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            return None

        masks: list[np.ndarray] = []
        scored: list[tuple[float, np.ndarray]] = []
        has_named_pelvis_class = False
        has_known_class_names = False
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            class_name = str(prediction.get("class") or "").lower()
            if class_name:
                has_known_class_names = True
            if any(term in class_name for term in PELVIS_YOLO_EXCLUDE_CLASS_TERMS):
                continue
            points = prediction.get("points")
            if not isinstance(points, list) or len(points) < 3:
                continue
            polygon = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    px = int(round(float(point["x"])))
                    py = int(round(float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    continue
                polygon.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))
            if len(polygon) < 3:
                continue

            component = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(component, [np.asarray(polygon, dtype=np.int32)], 1)
            if not bool(np.any(component)):
                continue

            is_pelvis_name = any(
                term in class_name for term in PELVIS_YOLO_INCLUDE_CLASS_TERMS
            )
            if is_pelvis_name:
                has_named_pelvis_class = True
                masks.append(component)
                continue
            confidence = float(prediction.get("confidence") or 0.0)
            scored.append((self._score_pelvis_mask(component) + confidence, component))

        if has_named_pelvis_class:
            raw_mask = np.any(np.stack(masks, axis=0) > 0, axis=0)
        elif not has_known_class_names and len(scored) == 1:
            raw_mask = scored[0][1]
        elif scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            raw_mask = np.any(
                np.stack([component for _score, component in scored[:4]], axis=0) > 0,
                axis=0,
            )
        else:
            return None

        return self._postprocess_binary_mask(
            raw_mask.astype(np.uint8),
            shape,
            allow_prior_on_small=False,
        )

    def _segment_with_replicate_unetpp(self, image: Image.Image) -> SegmentationResult:
        api_token = (os.environ.get("REPLICATE_API_TOKEN") or "").strip()
        if not api_token:
            raise PelvisSegmentationError(
                "Set REPLICATE_API_TOKEN to use the Replicate U-Net++ pelvis model."
            )

        try:
            import base64
            import io
            import requests
        except Exception as exc:
            raise PelvisSegmentationError(
                "The requests package is required to call Replicate models."
            ) from exc

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
        }
        payload = {
            "version": os.environ.get(
                "PELVIS_REPLICATE_MEDICAPTURE_VERSION",
                REPLICATE_MEDICAPTURE_VERSION,
            ),
            "input": {"image": data_uri},
        }

        try:
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
                timeout=(10, float(os.environ.get("PELVIS_REPLICATE_TIMEOUT", "120"))),
            )
            response.raise_for_status()
            prediction = response.json()
            prediction = self._wait_for_replicate_prediction(
                prediction,
                headers,
                requests,
            )
            output = prediction.get("output")
        except Exception as exc:
            raise PelvisSegmentationError(
                f"Replicate U-Net++ pelvis inference failed: {exc}"
            ) from exc

        output_url = _replicate_output_url(output)
        if not output_url:
            raise PelvisSegmentationError(
                "Replicate U-Net++ pelvis did not return an output image."
            )

        try:
            output_response = requests.get(
                output_url,
                timeout=(10, float(os.environ.get("PELVIS_REPLICATE_TIMEOUT", "120"))),
            )
            output_response.raise_for_status()
            output_img = Image.open(io.BytesIO(output_response.content))
        except Exception as exc:
            raise PelvisSegmentationError(
                f"Could not download Replicate U-Net++ pelvis output: {exc}"
            ) from exc

        rgb = np.array(image.convert("RGB"))
        mask = self._replicate_output_to_mask(output_img, rgb.shape[:2])
        if mask is None:
            raise PelvisSegmentationError(
                "Replicate U-Net++ pelvis returned an empty mask."
            )
        return SegmentationResult(
            mask=mask,
            backend="Replicate U-Net++ pelvis",
            detail="medicapture/seg-model",
        )

    def _wait_for_replicate_prediction(
        self,
        prediction: dict[str, Any],
        headers: dict[str, str],
        requests_module: Any,
    ) -> dict[str, Any]:
        import time

        timeout = float(os.environ.get("PELVIS_REPLICATE_TIMEOUT", "120"))
        deadline = time.monotonic() + timeout
        while str(prediction.get("status") or "").lower() in {
            "starting",
            "processing",
            "queued",
        }:
            if time.monotonic() >= deadline:
                raise PelvisSegmentationError("Replicate prediction timed out.")
            poll_url = str(prediction.get("urls", {}).get("get") or "")
            if not poll_url:
                raise PelvisSegmentationError("Replicate did not return a poll URL.")
            time.sleep(1.0)
            response = requests_module.get(poll_url, headers=headers, timeout=(10, 30))
            response.raise_for_status()
            prediction = response.json()

        status = str(prediction.get("status") or "").lower()
        if status != "succeeded":
            error = prediction.get("error") or status or "unknown status"
            raise PelvisSegmentationError(f"Replicate prediction failed: {error}")
        return prediction

    def _replicate_output_to_mask(
        self, output_img: Image.Image, shape: tuple[int, int]
    ) -> np.ndarray | None:
        if cv2 is None:
            raise PelvisSegmentationError("OpenCV is required for output resizing.")
        h, w = shape
        rgba = np.array(output_img.convert("RGBA"))
        alpha = rgba[..., 3]
        gray = np.array(output_img.convert("L"))
        if bool(np.any((alpha > 0) & (alpha < 255))) or int(alpha.min()) == 0:
            raw = (alpha > 0).astype(np.uint8)
        else:
            threshold = float(os.environ.get("PELVIS_REPLICATE_THRESHOLD", "32"))
            raw = (gray > threshold).astype(np.uint8)
        return self._postprocess_binary_mask(raw, (h, w), allow_prior_on_small=False)

    def _ultralytics_result_to_mask(
        self,
        result: Any,
        shape: tuple[int, int],
        model_names: Any,
    ) -> np.ndarray | None:
        masks_obj = getattr(result, "masks", None)
        mask_data = getattr(masks_obj, "data", None)
        masks = _to_numpy(mask_data)
        if masks is None:
            return None
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None

        boxes = getattr(result, "boxes", None)
        classes = _to_numpy(getattr(boxes, "cls", None))
        confidences = _to_numpy(getattr(boxes, "conf", None))
        result_names = getattr(result, "names", None)
        names = result_names if result_names is not None else model_names
        threshold = float(os.environ.get("PELVIS_YOLO_MASK_THRESHOLD", "0.5"))

        selectable: list[tuple[float, np.ndarray]] = []
        name_matched: list[np.ndarray] = []
        has_named_pelvis_class = False
        has_known_class_names = False

        for idx, mask in enumerate(masks):
            cls_idx = _class_index_at(classes, idx)
            class_name = _class_name_for_index(names, cls_idx)
            normalized_name = class_name.lower()
            if normalized_name:
                has_known_class_names = True
            if any(term in normalized_name for term in PELVIS_YOLO_EXCLUDE_CLASS_TERMS):
                continue

            binary = (mask >= threshold).astype(np.uint8)
            if not bool(np.any(binary)):
                continue

            is_pelvis_name = any(
                term in normalized_name for term in PELVIS_YOLO_INCLUDE_CLASS_TERMS
            )
            if is_pelvis_name:
                has_named_pelvis_class = True
                name_matched.append(binary)
                continue

            score = self._score_pelvis_mask(binary)
            confidence = _float_at(confidences, idx, default=0.0)
            selectable.append((score + confidence, binary))

        if has_named_pelvis_class:
            raw_mask = np.any(np.stack(name_matched, axis=0) > 0, axis=0)
        elif not has_known_class_names and len(selectable) == 1:
            raw_mask = selectable[0][1]
        elif selectable:
            selectable.sort(key=lambda item: item[0], reverse=True)
            best_score = selectable[0][0]
            keep = [
                mask
                for score, mask in selectable
                if score >= max(0.0, best_score - 0.4)
            ]
            if not keep:
                keep = [selectable[0][1]]
            raw_mask = np.any(np.stack(keep[:4], axis=0) > 0, axis=0)
        else:
            return None

        return self._postprocess_binary_mask(
            raw_mask.astype(np.uint8),
            shape,
            allow_prior_on_small=False,
        )

    def _ensure_local_yolo_model_path(self, mode: str) -> Path:
        spec = PELVIS_LOCAL_YOLO_MODELS[mode]
        configured_path = (
            os.environ.get(str(spec["path_env"]))
            or (self.yolo_model_path if mode == "pelvis_yolo" else "")
            or ""
        ).strip()
        if configured_path:
            configured = configured_path
            if configured.startswith(("http://", "https://")):
                return self._download_cached_file(configured, Path(configured).name)
            model_path = Path(configured).expanduser()
            if not model_path.exists():
                raise PelvisSegmentationError(
                    f"Pelvis YOLO model not found: {model_path}"
                )
            return model_path

        model_url = (
            os.environ.get(str(spec["url_env"]))
            or (self.yolo_model_url if mode == "pelvis_yolo" else "")
            or str(spec["url"])
        ).strip()
        return self._download_cached_file(
            model_url,
            str(spec["filename"]),
        )

    def _ensure_pelvis_yolo_model_path(self) -> Path:
        return self._ensure_local_yolo_model_path("pelvis_yolo")

    def _ensure_yolo_export(self, format_name: str) -> Path:
        if format_name not in {"onnx", "torchscript"}:
            raise PelvisSegmentationError(f"Unsupported YOLO export: {format_name}")

        model_path = self._ensure_pelvis_yolo_model_path()
        export_suffix = ".onnx" if format_name == "onnx" else ".torchscript"
        expected_path = model_path.with_suffix(export_suffix)
        if expected_path.exists() and expected_path.stat().st_size > 0:
            return expected_path

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PelvisSegmentationError(
                "Ultralytics is required to export the pelvis YOLO model."
            ) from exc

        image_size = int(os.environ.get("PELVIS_YOLO_EXPORT_IMAGE_SIZE", "1024"))
        image_size = max(256, min(2048, image_size))
        try:
            model = YOLO(str(model_path), task="segment")
            exported = model.export(
                format=format_name,
                imgsz=image_size,
                device=self.device or "cpu",
                half=False,
                dynamic=False,
                simplify=False,
                verbose=False,
            )
        except Exception as exc:
            raise PelvisSegmentationError(
                f"Failed to export pelvis YOLO model to {format_name}: {exc}"
            ) from exc

        exported_path = Path(str(exported)).expanduser()
        if not exported_path.is_absolute():
            exported_path = model_path.parent / exported_path
        if exported_path.exists() and exported_path.stat().st_size > 0:
            return exported_path
        if expected_path.exists() and expected_path.stat().st_size > 0:
            return expected_path
        raise PelvisSegmentationError(
            f"Ultralytics did not create a usable {format_name} export."
        )

    def _download_cached_file(self, url: str, filename: str) -> Path:
        cache_dir = _pelvis_model_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / filename
        if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
            return cache_path

        try:
            import requests
        except Exception as exc:
            raise PelvisSegmentationError(
                "The requests package is required to download the pelvis model."
            ) from exc

        temp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temp_path.stat().st_size <= 1_000_000:
                raise PelvisSegmentationError(
                    "Downloaded pelvis model was too small to be a valid weight file."
                )
            temp_path.replace(cache_path)
        except PelvisSegmentationError:
            raise
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise PelvisSegmentationError(
                f"Could not download pelvis YOLO model from {url}: {exc}"
            ) from exc
        return cache_path

    def _segment_with_medsam(self, image: Image.Image) -> SegmentationResult | None:
        if not self.model_path:
            if self.mode in {"medsam", "sam"}:
                raise PelvisSegmentationError(
                    "Set MEDSAM_CHECKPOINT or PELVIS_SEGMENTATION_MODEL to use MedSAM."
                )
            return None

        checkpoint = Path(self.model_path).expanduser()
        if not checkpoint.exists():
            raise PelvisSegmentationError(f"Model checkpoint not found: {checkpoint}")

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise PelvisSegmentationError(
                'Install PyTorch and MedSAM/SAM\'s "segment_anything" package first.'
            ) from exc

        predictor = self._sam_predictor
        if predictor is None:
            if self.model_type not in sam_model_registry:
                available = ", ".join(sorted(sam_model_registry.keys()))
                raise PelvisSegmentationError(
                    f"Unknown SAM model type {self.model_type!r}. Available: {available}"
                )
            sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
            sam.to(device=self.device)
            sam.eval()
            predictor = SamPredictor(sam)
            self._sam_predictor = predictor

        rgb = np.array(image.convert("RGB"))
        box = self._estimate_pelvis_box(rgb)
        with torch.no_grad():
            predictor.set_image(rgb)
            masks, scores, _logits = predictor.predict(
                box=np.array(box, dtype=np.float32),
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise PelvisSegmentationError("MedSAM did not return a mask.")

        best_idx = self._best_prompted_mask_index(masks, scores)
        mask = self._postprocess_binary_mask(masks[best_idx], rgb.shape[:2])
        if mask is None:
            raise PelvisSegmentationError("MedSAM returned an implausible mask.")

        return SegmentationResult(
            mask=mask,
            backend="MedSAM/SAM",
            detail=f"box={box}",
        )

    def _segment_with_sam2(self, image: Image.Image) -> SegmentationResult | None:
        if not self.sam2_hf_id and not self.sam2_checkpoint:
            if self.mode == "sam2":
                raise PelvisSegmentationError(
                    "Set PELVIS_SEGMENTATION_SAM2_HF_ID or "
                    "PELVIS_SEGMENTATION_SAM2_CHECKPOINT to use SAM 2."
                )
            return None

        if self.sam2_checkpoint and not Path(self.sam2_checkpoint).expanduser().exists():
            raise PelvisSegmentationError(
                f"SAM 2 checkpoint not found: {self.sam2_checkpoint}"
            )

        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise PelvisSegmentationError(
                'Install PyTorch and Meta SAM 2 first, for example "pip install sam2".'
            ) from exc

        predictor = self._sam2_predictor
        if predictor is None:
            if self.sam2_hf_id:
                try:
                    predictor = SAM2ImagePredictor.from_pretrained(self.sam2_hf_id)
                except AttributeError as exc:
                    raise PelvisSegmentationError(
                        "This SAM 2 install does not support from_pretrained; "
                        "use a local checkpoint and config instead."
                    ) from exc
            else:
                try:
                    from sam2.build_sam import build_sam2
                except Exception as exc:
                    raise PelvisSegmentationError(
                        "SAM 2 local checkpoint mode needs sam2.build_sam.build_sam2."
                    ) from exc
                model = build_sam2(
                    self.sam2_config,
                    str(Path(self.sam2_checkpoint).expanduser()),
                    device=self.device,
                )
                predictor = SAM2ImagePredictor(model)
            self._sam2_predictor = predictor

        rgb = np.array(image.convert("RGB"))
        box = self._estimate_pelvis_box(rgb)
        with torch.inference_mode():
            predictor.set_image(rgb)
            masks, scores, _logits = predictor.predict(
                box=np.array(box, dtype=np.float32),
                multimask_output=True,
            )

        masks_np = _normalize_predictor_masks(masks)
        if masks_np is None or masks_np.shape[0] == 0:
            raise PelvisSegmentationError("SAM 2 did not return a mask.")

        scores_np = _to_numpy(scores)
        best_idx = self._best_prompted_mask_index(masks_np, scores_np)
        mask = self._postprocess_binary_mask(masks_np[best_idx], rgb.shape[:2])
        if mask is None:
            raise PelvisSegmentationError("SAM 2 returned an implausible mask.")

        return SegmentationResult(mask=mask, backend="SAM 2", detail=f"box={box}")

    def _segment_with_torchscript(self, image: Image.Image) -> SegmentationResult | None:
        if not self.torchscript_model_path:
            export_path = self._ensure_yolo_export("torchscript")
            return self._segment_with_ultralytics_model(
                image,
                export_path,
                backend="YOLO TorchScript pelvis",
                detail=export_path.name,
            )

        model_path = Path(self.torchscript_model_path).expanduser()
        if not model_path.exists():
            raise PelvisSegmentationError(f"TorchScript model not found: {model_path}")

        try:
            return self._segment_with_ultralytics_model(
                image,
                model_path,
                backend="YOLO TorchScript pelvis",
                detail=model_path.name,
            )
        except PelvisSegmentationError:
            pass

        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise PelvisSegmentationError("Install PyTorch to use TorchScript.") from exc

        model = self._torchscript_model
        if model is None:
            model = torch.jit.load(str(model_path), map_location=self.device)
            model.eval()
            self._torchscript_model = model

        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        input_size = int(os.environ.get("PELVIS_TORCHSCRIPT_INPUT_SIZE", "512"))
        input_size = max(128, min(2048, input_size))
        arr = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = F.interpolate(
            tensor,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.inference_mode():
            output = model(tensor)

        logits = _extract_tensor_output(output)
        if logits is None:
            raise PelvisSegmentationError(
                "TorchScript model must return a tensor, tuple/list, or dict with 'out'."
            )

        logits = logits.detach()
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim == 3 and logits.shape[0] > 1:
            class_idx = int(os.environ.get("PELVIS_TORCHSCRIPT_CLASS_INDEX", "1"))
            class_idx = max(0, min(int(logits.shape[0]) - 1, class_idx))
            prob = torch.softmax(logits, dim=0)[class_idx]
        elif logits.ndim == 3:
            prob = torch.sigmoid(logits[0])
        elif logits.ndim == 2:
            prob = torch.sigmoid(logits)
        else:
            raise PelvisSegmentationError(
                f"Unsupported TorchScript output shape: {tuple(logits.shape)}"
            )

        prob = F.interpolate(
            prob.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        threshold = float(os.environ.get("PELVIS_TORCHSCRIPT_THRESHOLD", "0.5"))
        mask = (prob.cpu().numpy() >= threshold).astype(np.uint8)
        mask = self._postprocess_binary_mask(mask, (h, w), allow_prior_on_small=False)
        if mask is None:
            raise PelvisSegmentationError("TorchScript model returned an empty mask.")

        return SegmentationResult(
            mask=mask,
            backend="Pelvis TorchScript",
            detail=str(model_path.name),
        )

    def _segment_with_onnx(self, image: Image.Image) -> SegmentationResult | None:
        if not self.onnx_model_path:
            export_path = self._ensure_yolo_export("onnx")
            return self._segment_with_ultralytics_model(
                image,
                export_path,
                backend="YOLO ONNX pelvis",
                detail=export_path.name,
            )

        model_path = Path(self.onnx_model_path).expanduser()
        if not model_path.exists():
            raise PelvisSegmentationError(f"ONNX model not found: {model_path}")

        try:
            return self._segment_with_ultralytics_model(
                image,
                model_path,
                backend="YOLO ONNX pelvis",
                detail=model_path.name,
            )
        except PelvisSegmentationError:
            pass

        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        input_size = (int(os.environ.get("PELVIS_ONNX_INPUT_SIZE", "512")),) * 2
        input_size = (
            max(128, min(2048, input_size[0])),
            max(128, min(2048, input_size[1])),
        )
        arr = rgb.astype(np.float32) / 255.0
        if cv2 is None:
            raise PelvisSegmentationError("OpenCV is required for ONNX preprocessing.")
        output = None

        try:
            import onnxruntime as ort

            session = self._onnx_session
            if session is None:
                providers = ["CPUExecutionProvider"]
                session = ort.InferenceSession(str(model_path), providers=providers)
                self._onnx_session = session
            input_meta = session.get_inputs()[0]
            input_size = _onnx_input_size(input_meta.shape)
            resized = cv2.resize(arr, input_size, interpolation=cv2.INTER_AREA)
            chw = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
                1, 3, 1, 1
            )
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            chw = (chw - mean) / std
            output = session.run(None, {input_meta.name: chw})[0]
        except ImportError:
            net = self._onnx_dnn_net
            if net is None:
                net = cv2.dnn.readNetFromONNX(str(model_path))
                self._onnx_dnn_net = net
            resized = cv2.resize(arr, input_size, interpolation=cv2.INTER_AREA)
            chw = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
                1, 3, 1, 1
            )
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            chw = (chw - mean) / std
            net.setInput(chw)
            output = net.forward()

        if output is None:
            raise PelvisSegmentationError("ONNX model did not return output.")
        prob = _prediction_to_probability(output)
        if prob is None:
            raise PelvisSegmentationError(
                "ONNX model output must be a 2D/3D/4D mask or class-logit tensor."
            )

        prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        threshold = float(os.environ.get("PELVIS_ONNX_THRESHOLD", "0.5"))
        mask = (prob >= threshold).astype(np.uint8)
        mask = self._postprocess_binary_mask(mask, (h, w), allow_prior_on_small=False)
        if mask is None:
            raise PelvisSegmentationError("ONNX model returned an empty mask.")

        return SegmentationResult(mask=mask, backend="Pelvis ONNX", detail=model_path.name)

    def _segment_with_hip_pelvis_annotator(
        self, image: Image.Image
    ) -> SegmentationResult | None:
        if not self.hip_pelvis_annotator_cmd:
            if self.mode == "hip_pelvis_annotator":
                raise PelvisSegmentationError(
                    "Set HIP_PELVIS_ANNOTATOR_CMD to a command that runs the "
                    "pelvis-trained HipPelvisAnnotator model."
                )
            return None

        with tempfile.TemporaryDirectory(prefix="pelvis_segment_") as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.png"
            mask_path = temp_path / "mask.png"
            image.convert("RGB").save(image_path)
            command = self.hip_pelvis_annotator_cmd.format(
                input=str(image_path),
                output=str(mask_path),
            )
            completed = subprocess.run(
                command,
                shell=True,
                cwd=temp_dir,
                check=False,
                text=True,
                capture_output=True,
                timeout=float(os.environ.get("HIP_PELVIS_ANNOTATOR_TIMEOUT", "120")),
            )
            if completed.returncode != 0:
                raise PelvisSegmentationError(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"HipPelvisAnnotator command failed with {completed.returncode}."
                )
            if not mask_path.exists():
                raise PelvisSegmentationError(
                    f"HipPelvisAnnotator did not write the expected mask: {mask_path}"
                )

            mask_img = Image.open(mask_path).convert("L")
            mask = (np.array(mask_img) > 0).astype(np.uint8)
            h, w = image.size[1], image.size[0]
            mask = self._postprocess_binary_mask(mask, (h, w))
            if mask is None:
                raise PelvisSegmentationError(
                    "HipPelvisAnnotator returned an empty mask."
                )
            return SegmentationResult(
                mask=mask,
                backend="HipPelvisAnnotator",
                detail="pelvis-trained AP radiograph CNN",
            )

    def _segment_with_opencv(self, image: Image.Image) -> SegmentationResult:
        if cv2 is None:
            raise PelvisSegmentationError(
                'OpenCV is not available. Install with "pip install opencv-python".'
            )

        gray = _image_to_gray_uint8(image)
        candidates: list[tuple[float, np.ndarray, str]] = []
        for invert in (False, True):
            mask = self._opencv_candidate(gray, invert=invert)
            if mask is None:
                continue
            score = self._score_pelvis_mask(mask) + self._foreground_contrast_score(
                mask, gray
            )
            candidates.append((score, mask, "inverted" if invert else "normal"))

        if not candidates:
            prior = _pelvis_prior_mask(gray.shape)
            return SegmentationResult(
                mask=prior,
                backend="OpenCV heuristic",
                detail="anatomical prior fallback",
            )

        candidates.sort(key=lambda item: (item[0], int(item[1].sum())), reverse=True)
        score, mask, polarity = candidates[0]
        if score <= 0:
            prior = _pelvis_prior_mask(gray.shape)
            merged = ((mask > 0) | (prior > 0)).astype(np.uint8)
            cleaned = self._postprocess_binary_mask(merged, gray.shape)
            mask = cleaned if cleaned is not None else prior
            polarity = f"{polarity}, low-confidence prior blend"

        return SegmentationResult(
            mask=mask,
            backend="OpenCV heuristic",
            detail=f"{polarity}, score={score:.2f}",
        )

    def _opencv_candidate(self, gray: np.ndarray, invert: bool) -> np.ndarray | None:
        proc = 255 - gray if invert else gray
        proc = _percentile_uint8(proc, 1.0, 99.7)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = clahe.apply(proc)
        proc = cv2.GaussianBlur(proc, (5, 5), 0)

        h, w = proc.shape[:2]
        if h < 32 or w < 32:
            return None

        otsu_val, otsu_mask = cv2.threshold(
            proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        p55 = float(np.percentile(proc, 55.0))
        p68 = float(np.percentile(proc, 68.0))
        high_mask = proc >= min(max(p55, float(otsu_val) * 0.82), p68)

        block = _odd_at_least(min(max(31, min(h, w) // 9), 151), 31)
        adaptive = cv2.adaptiveThreshold(
            proc,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            -2,
        )
        prior = _pelvis_prior_mask((h, w))
        candidate = (
            ((otsu_mask > 0) | high_mask | (adaptive > 0)) & (prior > 0)
        ).astype(np.uint8)

        candidate[: max(1, int(0.02 * h)), :] = 0
        candidate[int(0.94 * h) :, :] = 0

        metal = _metal_like_mask(proc)
        if metal is not None:
            candidate[metal > 0] = 0

        open_k = _ellipse_kernel(max(3, min(h, w) // 180))
        close_k = _ellipse_kernel(max(5, min(h, w) // 55))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_k, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_k, iterations=2)

        candidate = self._keep_plausible_components(candidate, proc)
        if candidate is None:
            relaxed = ((proc >= p55) & (prior > 0)).astype(np.uint8)
            candidate = cv2.morphologyEx(
                relaxed, cv2.MORPH_CLOSE, close_k, iterations=2
            )
            candidate = self._keep_largest_components(candidate, max_components=8)
            if candidate is None:
                candidate = prior

        fill_k = _ellipse_kernel(max(5, min(h, w) // 42))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, fill_k, iterations=1)
        return self._postprocess_binary_mask(candidate, (h, w))

    def _estimate_pelvis_box(self, rgb: np.ndarray) -> Box:
        gray = _image_to_gray_uint8(Image.fromarray(rgb))
        mask = self._opencv_candidate(gray, invert=False)
        if mask is None:
            h, w = gray.shape[:2]
            return (int(0.04 * w), int(0.03 * h), int(0.96 * w), int(0.92 * h))

        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            h, w = gray.shape[:2]
            return (int(0.04 * w), int(0.03 * h), int(0.96 * w), int(0.92 * h))

        h, w = gray.shape[:2]
        pad_x = max(8, int(0.04 * w))
        pad_y = max(8, int(0.04 * h))
        x0 = max(0, int(xs.min()) - pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        x1 = min(w - 1, int(xs.max()) + pad_x)
        y1 = min(h - 1, int(ys.max()) + pad_y)
        return (x0, y0, x1, y1)

    def _keep_plausible_components(
        self, candidate: np.ndarray, gray: np.ndarray
    ) -> np.ndarray | None:
        h, w = candidate.shape[:2]
        min_area = max(20, int(0.0005 * h * w))
        _count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )

        kept = np.zeros_like(candidate, dtype=np.uint8)
        scored: list[tuple[float, int]] = []
        for label in range(1, stats.shape[0]):
            x, y, bw, bh, area = stats[label]
            if area < min_area:
                continue
            cx, cy = centroids[label]
            if y > 0.9 * h or cy < 0.04 * h:
                continue
            aspect = max(float(bw) / max(1.0, float(bh)), float(bh) / max(1.0, bw))
            lower_slender = cy > 0.48 * h and bw < 0.16 * w and aspect > 3.8
            if lower_slender:
                continue
            component = labels == label
            mean_intensity = float(gray[component].mean())
            center_weight = 1.0 - min(0.85, abs((cx / max(1, w)) - 0.5) * 1.4)
            y_weight = 1.0 - min(0.75, abs((cy / max(1, h)) - 0.48) * 1.25)
            score = float(area) * center_weight * y_weight * (0.35 + mean_intensity / 255)
            scored.append((score, label))

        if not scored:
            return self._keep_largest_components(candidate, max_components=8)

        scored.sort(reverse=True)
        max_area = int(0.7 * h * w)
        for _score, label in scored[:8]:
            next_mask = kept | (labels == label).astype(np.uint8)
            if int(next_mask.sum()) > max_area:
                continue
            kept = next_mask

        if int(kept.sum()) >= min_area:
            return kept
        return self._keep_largest_components(candidate, max_components=8)

    def _keep_largest_components(
        self, candidate: np.ndarray, max_components: int = 8
    ) -> np.ndarray | None:
        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )
        if stats.shape[0] <= 1:
            return None
        h, w = candidate.shape[:2]
        min_area = max(10, int(0.00025 * h * w))
        ranked = sorted(
            (
                (int(stats[label, cv2.CC_STAT_AREA]), label)
                for label in range(1, stats.shape[0])
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
            ),
            reverse=True,
        )
        if not ranked:
            return None
        kept = np.zeros_like(candidate, dtype=np.uint8)
        for _area, label in ranked[:max_components]:
            kept[labels == label] = 1
        return kept if bool(np.any(kept)) else None

    def _remove_frame_like_edge_components(self, candidate: np.ndarray) -> np.ndarray:
        h, w = candidate.shape[:2]
        if cv2 is None or h < 32 or w < 32:
            return candidate.astype(np.uint8)

        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )
        if stats.shape[0] <= 1:
            return candidate.astype(np.uint8)

        margin = max(2, int(0.01 * min(h, w)))
        kept = np.zeros_like(candidate, dtype=np.uint8)
        removed_any = False
        for label in range(1, stats.shape[0]):
            x, y, bw, bh, area = (int(v) for v in stats[label])
            bbox_area = max(1, bw * bh)
            fill_ratio = float(area) / float(bbox_area)
            touches_top = y <= margin
            touches_left = x <= margin
            touches_right = x + bw >= w - margin

            wide_top_bar = (
                touches_top
                and bw >= int(0.35 * w)
                and bh <= int(0.16 * h)
            )
            tall_side_bar = (
                (touches_left or touches_right)
                and bh >= int(0.32 * h)
                and bw <= int(0.08 * w)
            )
            open_frame = (
                touches_top
                and (touches_left or touches_right)
                and bw >= int(0.58 * w)
                and bh >= int(0.32 * h)
                and fill_ratio <= 0.30
            )
            if wide_top_bar or tall_side_bar or open_frame:
                removed_any = True
                continue
            kept[labels == label] = 1

        if removed_any:
            return kept
        return candidate.astype(np.uint8)

    def _postprocess_binary_mask(
        self,
        raw_mask: np.ndarray,
        shape: tuple[int, int],
        allow_prior_on_small: bool = True,
    ) -> np.ndarray | None:
        h, w = shape
        mask = np.asarray(raw_mask).astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        mask = mask.astype(np.uint8)
        close_k = _ellipse_kernel(max(3, min(h, w) // 90))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

        area = int(mask.sum())
        if area < max(40, int(0.001 * h * w)):
            if allow_prior_on_small:
                return _pelvis_prior_mask((h, w))
            return None
        if area > int(0.82 * h * w):
            prior = _pelvis_prior_mask((h, w))
            mask = (mask > 0) & (prior > 0)
        return (mask > 0).astype(np.uint8)

    def _score_pelvis_mask(self, mask: np.ndarray) -> float:
        h, w = mask.shape[:2]
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return -1.0

        area_ratio = xs.size / float(h * w)
        if area_ratio < 0.04 or area_ratio > 0.76:
            return -1.0

        cx = float(xs.mean()) / max(1.0, w)
        cy = float(ys.mean()) / max(1.0, h)
        if cy < 0.06 or cy > 0.9:
            return -1.0

        x_span = (float(xs.max()) - float(xs.min()) + 1.0) / max(1.0, w)
        y_span = (float(ys.max()) - float(ys.min()) + 1.0) / max(1.0, h)
        area_score = 1.0 - min(1.0, abs(area_ratio - 0.22) / 0.35)
        center_score = 1.0 - min(0.85, abs(cx - 0.5) * 1.25)
        y_score = 1.0 - min(0.8, abs(cy - 0.48) * 1.5)
        span_score = min(1.0, max(x_span, y_span) * 1.4)
        border_touch = float(mask[:, :2].mean() + mask[:, -2:].mean()) * 0.5
        return area_score + center_score + y_score + span_score - border_touch

    def _foreground_contrast_score(self, mask: np.ndarray, gray: np.ndarray) -> float:
        foreground = mask > 0
        if not bool(np.any(foreground)) or not bool(np.any(~foreground)):
            return 0.0
        fg = float(gray[foreground].mean())
        bg = float(gray[~foreground].mean())
        return 3.0 * ((fg - bg) / 255.0)

    def _best_prompted_mask_index(self, masks: np.ndarray, scores: Any) -> int:
        best_idx = 0
        best_score = -1e9
        for idx, mask in enumerate(masks):
            prior_score = self._score_pelvis_mask(mask.astype(np.uint8))
            model_score = 0.0
            if scores is not None and idx < len(scores):
                try:
                    model_score = float(scores[idx])
                except (TypeError, ValueError):
                    model_score = 0.0
            score = prior_score + model_score
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


def _image_to_gray_uint8(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("L"))
    if arr.dtype == np.uint8:
        return arr
    return _percentile_uint8(arr, 1.0, 99.5)


def _pitt_hexai_image_to_tensor(
    image: Image.Image, input_size: int, *, invert_grayscale: bool
) -> tuple[Any, tuple[int, int, int, int], tuple[int, int]]:
    import torch

    gray = _image_to_gray_uint8(image)
    if invert_grayscale:
        gray = 255 - gray
    original_h, original_w = gray.shape[:2]
    source = Image.fromarray(gray)
    contained = ImageOps.contain(
        source,
        (input_size, input_size),
        method=Image.Resampling.BILINEAR,
    )
    x0 = (input_size - contained.width) // 2
    y0 = (input_size - contained.height) // 2
    square = Image.new("L", (input_size, input_size), color=0)
    square.paste(contained, (x0, y0))

    arr = np.asarray(square, dtype=np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        arr = arr - mean
    else:
        arr = (arr - mean) / std
    tensor = torch.from_numpy(arr[None, None, :, :])
    return tensor, (x0, y0, x0 + contained.width, y0 + contained.height), (
        original_w,
        original_h,
    )


def _build_pitt_hexai_model(
    *,
    model_name: str,
    base_ch: int,
    task_head_channels: int | None,
) -> Any:
    root_src = Path(__file__).resolve().parent
    if root_src.exists() and str(root_src) not in sys.path:
        sys.path.insert(0, str(root_src))

    try:
        if model_name == "highres":
            from high_resolution_landmark_network import HighResolutionLandmarkModel

            return HighResolutionLandmarkModel(
                in_ch=1,
                base_ch=base_ch,
                out_ch=1,
                task_head_channels=task_head_channels,
            )
        if model_name == "hailo":
            from hailo_landmark_network import HeatmapModelHailo

            return HeatmapModelHailo(in_ch=1, base_ch=base_ch, out_ch=1)
        if model_name == "original":
            from original_landmark_network import HeatmapModel

            return HeatmapModel(in_ch=1, base_ch=base_ch, out_ch=1)
    except Exception as exc:
        raise PelvisSegmentationError(
            f"Could not import Pitt HexAI model backbone from {root_src}: {exc}"
        ) from exc

    raise PelvisSegmentationError(f"Unsupported Pitt HexAI model type: {model_name}")


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pelvis_model_cache_dir() -> Path:
    configured = os.environ.get("PELVIS_SEGMENTATION_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("2d-point-annotator", "orthodriven")) / (
            "pelvis-segmentation"
        )
    except Exception:
        return Path.home() / ".cache" / "2d-point-annotator" / "pelvis-segmentation"


def _class_index_at(classes: Any, idx: int) -> int | None:
    if classes is None:
        return None
    try:
        arr = np.asarray(classes).reshape(-1)
        if idx < arr.size:
            return int(arr[idx])
    except Exception:
        return None
    return None


def _float_at(values: Any, idx: int, default: float = 0.0) -> float:
    if values is None:
        return default
    try:
        arr = np.asarray(values).reshape(-1)
        if idx < arr.size:
            return float(arr[idx])
    except Exception:
        return default
    return default


def _class_name_for_index(names: Any, class_idx: int | None) -> str:
    if class_idx is None:
        return ""
    if isinstance(names, dict):
        value = names.get(class_idx, names.get(str(class_idx), ""))
        return str(value) if value is not None else ""
    if isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
        return str(names[class_idx])
    return ""


def _replicate_output_url(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return _replicate_output_url(output[0])
    if isinstance(output, dict):
        for key in ("url", "image", "output"):
            value = output.get(key)
            if isinstance(value, str):
                return value
            nested = _replicate_output_url(value)
            if nested:
                return nested
    return ""


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _normalize_predictor_masks(value: Any) -> np.ndarray | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        return None
    return arr.astype(bool)


def _extract_tensor_output(output: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        for value in output.values():
            if torch is None or isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch is None or isinstance(first, torch.Tensor):
            return first
    return None


def _onnx_input_size(shape: list[Any] | tuple[Any, ...]) -> tuple[int, int]:
    default = int(os.environ.get("PELVIS_ONNX_INPUT_SIZE", "512"))
    default = max(128, min(2048, default))
    if len(shape) >= 4:
        dim_h, dim_w = shape[-2], shape[-1]
        if isinstance(dim_h, int) and isinstance(dim_w, int) and dim_h > 0 and dim_w > 0:
            return int(dim_w), int(dim_h)
    return default, default


def _prediction_to_probability(output: Any) -> np.ndarray | None:
    arr = _to_numpy(output)
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] <= 8:
        if arr.shape[0] > 1:
            class_idx = int(os.environ.get("PELVIS_ONNX_CLASS_INDEX", "1"))
            class_idx = max(0, min(int(arr.shape[0]) - 1, class_idx))
            arr = _softmax_numpy(arr, axis=0)[class_idx]
        else:
            arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] <= 8:
        if arr.shape[-1] > 1:
            class_idx = int(os.environ.get("PELVIS_ONNX_CLASS_INDEX", "1"))
            class_idx = max(0, min(int(arr.shape[-1]) - 1, class_idx))
            arr = _softmax_numpy(arr, axis=-1)[..., class_idx]
        else:
            arr = arr[..., 0]
    elif arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    if float(np.nanmin(arr)) < 0.0 or float(np.nanmax(arr)) > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _softmax_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    return exp / denom


def _percentile_uint8(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    lo, hi = np.percentile(arr_f, (low, high))
    if hi <= lo:
        return np.zeros(arr_f.shape, dtype=np.uint8)
    stretched = (arr_f - lo) / (hi - lo)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value


def _ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return (mask > 0).astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood == 0
    return (binary | holes.astype(np.uint8)).astype(np.uint8)


def _pelvis_prior_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    x = xx.astype(np.float32) / max(1.0, float(w - 1))
    y = yy.astype(np.float32) / max(1.0, float(h - 1))

    basin = ((x - 0.5) / 0.47) ** 2 + ((y - 0.47) / 0.35) ** 2 <= 1.0
    left_wing = ((x - 0.32) / 0.23) ** 2 + ((y - 0.45) / 0.3) ** 2 <= 1.0
    right_wing = ((x - 0.68) / 0.23) ** 2 + ((y - 0.45) / 0.3) ** 2 <= 1.0
    lower_ring = ((x - 0.5) / 0.28) ** 2 + ((y - 0.63) / 0.28) ** 2 <= 1.0
    upper_sacrum = ((x - 0.5) / 0.17) ** 2 + ((y - 0.34) / 0.28) ** 2 <= 1.0
    mask = basin | left_wing | right_wing | lower_ring | upper_sacrum
    mask &= y >= 0.05
    mask &= y <= 0.88

    if cv2 is not None:
        kernel = _ellipse_kernel(max(3, min(h, w) // 80))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.uint8)
    return mask.astype(np.uint8)


def _metal_like_mask(gray: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return None

    threshold = max(245.0, float(np.percentile(gray, 99.65)))
    bright = (gray >= threshold).astype(np.uint8)
    if int(bright.sum()) < max(10, int(0.0002 * h * w)):
        return None

    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        bright, connectivity=8
    )
    metal = np.zeros_like(bright, dtype=np.uint8)
    for label in range(1, stats.shape[0]):
        _x, _y, bw, bh, area = stats[label]
        if area < max(10, int(0.0001 * h * w)):
            continue
        aspect = max(float(bw) / max(1.0, bh), float(bh) / max(1.0, bw))
        if aspect >= 4.0 or area > 0.003 * h * w:
            metal[labels == label] = 1

    if not bool(np.any(metal)):
        return None
    kernel = _ellipse_kernel(max(1, min(h, w) // 220))
    return cv2.dilate(metal, kernel, iterations=1)
