from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportUninitializedInstanceVariable=false, reportOperatorIssue=false
import ast
import asyncio
import base64
import binascii
import datetime
import getpass
import io
import json
import logging
import math
import os
import sqlite3
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import zlib
from datetime import datetime
from pathlib import Path, PurePath
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging - writes to file for debugging without disrupting users
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "annotator.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from auth import (  # pyright: ignore[reportImplicitRelativeImport]
    SHAREPOINT_DRIVE_ID,
    OneDriveBackup,
)
from dataset_config import get_data_dir  # pyright: ignore[reportImplicitRelativeImport]
from dirs import BASE_DIR, PLATFORM  # pyright: ignore[reportImplicitRelativeImport]
from landmark_reference import (  # pyright: ignore[reportImplicitRelativeImport]
    LandmarkReference,  # pyright: ignore[reportImplicitRelativeImport]
)
from landmark_reference_dialog import (  # pyright: ignore[reportImplicitRelativeImport]
    LandmarkReferenceDialog,  # pyright: ignore[reportImplicitRelativeImport]
)
from path_utils import extract_filename  # pyright: ignore[reportImplicitRelativeImport]
from pelvis_segmentation import (  # pyright: ignore[reportImplicitRelativeImport]
    PelvisSegmentationBackend,
)

AnnotationPoint = Tuple[float, float]
EllipseAnnotation = Dict[str, object]
AnnotationValue = Union[AnnotationPoint, List[AnnotationPoint], EllipseAnnotation]

# Display order for the landmark panel and keyboard navigation.
# Landmarks not listed here are appended at the end in their original JSON order.
# Changing this list does NOT affect how annotations are stored or read.
# Extra landmarks injected into specific views regardless of what the project JSON says.
# This lets us extend view definitions for all users (including those with old JSONs) without
# touching their annotation data.
VIEW_LANDMARK_EXTENSIONS: Dict[str, List[str]] = {}

PROTOCOL_LEFT_DIRECT_LANDMARKS: List[str] = [
    "L-C-ILI",
    "L-C-PS",
    "L-DSI",
    "L-C-PO",
    "L-C-OF",
    "L-C-IT",
    "L-PT",
    "L-C-AC",
    "L-C-FAC",
    "L-C-FHC",
    "L-C-GT",
    "L-C-LT",
    "L-C-PMF",
    "L-C-DMF",
]
PROTOCOL_RIGHT_DIRECT_LANDMARKS: List[str] = [
    "R-C-ILI",
    "R-C-PS",
    "R-DSI",
    "R-C-PO",
    "R-C-OF",
    "R-C-IT",
    "R-PT",
    "R-C-AC",
    "R-C-FAC",
    "R-C-FHC",
    "R-C-GT",
    "R-C-LT",
    "R-C-PMF",
    "R-C-DMF",
]
PROTOCOL_DIRECT_LANDMARKS: List[str] = (
    PROTOCOL_LEFT_DIRECT_LANDMARKS + PROTOCOL_RIGHT_DIRECT_LANDMARKS
)

PROTOCOL_FREEHAND_CONTOUR_LANDMARKS: Set[str] = {
    "L-C-ILI",
    "R-C-ILI",
    "L-C-PO",
    "R-C-PO",
    "L-C-IT",
    "R-C-IT",
    "L-C-PS",
    "R-C-PS",
    "L-C-GT",
    "R-C-GT",
    "L-C-PMF",
    "R-C-PMF",
    "L-C-LT",
    "R-C-LT",
    "L-C-DMF",
    "R-C-DMF",
    "L-C-OF",
    "R-C-OF",
}
PROTOCOL_CIRCLE_CONTOUR_LANDMARKS: Set[str] = {
    "L-C-AC",
    "R-C-AC",
    "L-C-FHC",
    "R-C-FHC",
}
PROTOCOL_ELLIPSE_CONTOUR_LANDMARKS: Set[str] = {"L-C-FAC", "R-C-FAC"}
PROTOCOL_POINT_LANDMARKS: Set[str] = {"L-PT", "R-PT", "L-DSI", "R-DSI"}
PROTOCOL_LINE_LANDMARKS: Set[str] = {"L-D-FA", "R-D-FA", "POD"}
LEGACY_LINE_LANDMARKS: Set[str] = {"L-FA", "R-FA"}
PROTOCOL_CONTOUR_LANDMARKS: Set[str] = (
    PROTOCOL_FREEHAND_CONTOUR_LANDMARKS
    | PROTOCOL_CIRCLE_CONTOUR_LANDMARKS
    | PROTOCOL_ELLIPSE_CONTOUR_LANDMARKS
)

LEGACY_IT_EDGE_LANDMARKS: Set[str] = {"L-E-IT", "R-E-IT", "L-IT-E", "R-IT-E"}
LEGACY_ELLIPSE_LANDMARKS: Set[str] = {"L-FAC", "R-FAC"}
LEGACY_AC_CIRCLE_LANDMARKS: Set[str] = {"L-AC", "R-AC"}
LEGACY_FHC_CIRCLE_LANDMARKS: Set[str] = {"L-FHC", "R-FHC"}

ELLIPSE_LANDMARKS: Set[str] = (
    PROTOCOL_ELLIPSE_CONTOUR_LANDMARKS | LEGACY_ELLIPSE_LANDMARKS
)

ELLIPSE_LANDMARK_EXTENSIONS: Dict[str, List[str]] = {
    "AP Bilateral": ["L-FAC", "R-FAC"],
    "AP Unilateral (Left)": ["L-FAC"],
    "AP Unilateral (Right)": ["R-FAC"],
}

AUTO_SEGMENTATION_LANDMARKS: Set[str] = {"LOB", "ROB"}
IT_EDGE_LANDMARKS: Set[str] = PROTOCOL_FREEHAND_CONTOUR_LANDMARKS | LEGACY_IT_EDGE_LANDMARKS
PELVIS_SEGMENTATION_LANDMARK = "PELVIS"
SEGMENTATION_LANDMARKS: Set[str] = (
    AUTO_SEGMENTATION_LANDMARKS
    | IT_EDGE_LANDMARKS
    | {PELVIS_SEGMENTATION_LANDMARK}
)
SEGMENTATION_MASK_ENCODING = "zlib+base64+packbits"
MANUAL_SEGMENTATION_MASK_SIZE = 512
MANUAL_SEGMENTATION_MASK_SHAPE = (
    MANUAL_SEGMENTATION_MASK_SIZE,
    MANUAL_SEGMENTATION_MASK_SIZE,
)
PELVIS_SEGMENTATION_BACKEND_OPTIONS: List[Tuple[str, str]] = [
    ("auto", "Auto pelvis models"),
    ("pitt_hexai_highres", "Local Pitt HighRes"),
    ("hf_yolo11s_pelvis_xray", "HF YOLO11s pelvis"),
    ("replicate_unetpp_pelvis", "Replicate U-Net++ pelvis"),
    ("rf_ericyum_pelvis_ap", "Roboflow ericyum AP"),
    ("rf_sooyeon_pelvis_ap", "Roboflow sooyeon AP"),
    ("rf_yolov8_pelvis_xray", "Roboflow YOLOv8 pelvis"),
    ("opencv", "OpenCV fallback"),
]

SEGMENTATION_LANDMARK_EXTENSIONS: Dict[str, List[str]] = {
    "AP Bilateral": ["L-E-IT", "R-E-IT"],
    "AP Unilateral (Left)": ["L-E-IT"],
    "AP Unilateral (Right)": ["R-E-IT"],
}

DERIVED_ACETABULAR_LANDMARKS: Set[str] = {
    "L-A-DAB",
    "R-A-DAB",
    "L-A-SAB",
    "R-A-SAB",
    "L-DAB",
    "R-DAB",
    "L-SAB",
    "R-SAB",
}

INTERMEDIATE_ACETABULAR_LANDMARKS: Set[str] = {
    "L-A-IDAB",
    "R-A-IDAB",
    "L-A-ISAB",
    "R-A-ISAB",
}

DERIVED_ACETABULAR_LANDMARK_EXTENSIONS: Dict[str, List[str]] = {
    "AP Bilateral": ["L-A-SAB", "R-A-SAB", "L-A-DAB", "R-A-DAB"],
    "AP Unilateral (Left)": ["L-A-SAB", "L-A-DAB"],
    "AP Unilateral (Right)": ["R-A-SAB", "R-A-DAB"],
}

INTERMEDIATE_ACETABULAR_LANDMARK_EXTENSIONS: Dict[str, List[str]] = {
    "AP Bilateral": ["L-A-ISAB", "R-A-ISAB", "L-A-IDAB", "R-A-IDAB"],
    "AP Unilateral (Left)": ["L-A-ISAB", "L-A-IDAB"],
    "AP Unilateral (Right)": ["R-A-ISAB", "R-A-IDAB"],
}

PROTOCOL_DERIVED_FEATURES_BY_SOURCE: Dict[str, List[str]] = {
    "L-C-ILI": ["L-SIP", "L-LIP"],
    "R-C-ILI": ["R-SIP", "R-LIP"],
    "L-C-PO": ["L-LPO"],
    "R-C-PO": ["R-LPO"],
    "L-C-IT": ["L-IT"],
    "R-C-IT": ["R-IT"],
    "L-C-PS": ["L-SPS", "L-IPS"],
    "R-C-PS": ["R-SPS", "R-IPS"],
    "L-C-AC": ["L-AC"],
    "R-C-AC": ["R-AC"],
    "L-C-FAC": ["L-DAB", "L-SAB"],
    "R-C-FAC": ["R-DAB", "R-SAB"],
    "L-C-FHC": ["L-FHC"],
    "R-C-FHC": ["R-FHC"],
    "L-C-GT": ["L-SGT", "L-LGT"],
    "R-C-GT": ["R-SGT", "R-LGT"],
    "L-C-LT": ["L-PLT", "L-MLT", "L-DLT"],
    "R-C-LT": ["R-PLT", "R-MLT", "R-DLT"],
    "L-C-DMF": ["L-D-FA"],
    "R-C-DMF": ["R-D-FA"],
}
PROTOCOL_DERIVED_FEATURE_SOURCES: Dict[str, List[str]] = {
    derived: [source]
    for source, derived_ids in PROTOCOL_DERIVED_FEATURES_BY_SOURCE.items()
    for derived in derived_ids
}
PROTOCOL_DERIVED_FEATURE_SOURCES["POD"] = ["L-LPO", "R-LPO"]
PROTOCOL_DERIVED_FEATURE_SOURCES["L-D-FA"] = ["L-C-DMF", "L-C-GT"]
PROTOCOL_DERIVED_FEATURE_SOURCES["R-D-FA"] = ["R-C-DMF", "R-C-GT"]
PROTOCOL_DERIVED_FEATURE_LANDMARKS: Set[str] = set(
    PROTOCOL_DERIVED_FEATURE_SOURCES
)
HIP_CATEGORY_VALUES: Set[str] = {"", "native", "prosthetic", "other"}
HIP_CATEGORY_DISPLAY_VALUES: List[str] = ["", "Native", "Prosthetic", "Other"]
DERIVED_FEATURE_STATUS_VALUES: Set[str] = {"", "correct", "incorrect", "missing"}
DERIVED_FEATURE_STATUS_DISPLAY_VALUES: List[str] = [
    "",
    "Correct",
    "Incorrect",
    "Missing",
]
DIRECT_ANNOTATION_STATUS_VALUES: Set[str] = {"", "missing", "not_applicable"}
DIRECT_ANNOTATION_STATUS_DISPLAY_VALUES: List[str] = [
    "",
    "Missing",
    "Not Applicable",
]

PROTOCOL_LEFT_DISPLAY_ORDER: List[str] = [
    "L-C-ILI",
    "L-SIP",
    "L-LIP",
    "L-C-PS",
    "L-SPS",
    "L-IPS",
    "L-DSI",
    "L-C-PO",
    "L-LPO",
    "L-C-OF",
    "L-C-IT",
    "L-IT",
    "L-PT",
    "L-C-AC",
    "L-AC",
    "L-C-FAC",
    "L-DAB",
    "L-SAB",
    "L-C-FHC",
    "L-FHC",
    "L-C-GT",
    "L-SGT",
    "L-LGT",
    "L-C-LT",
    "L-PLT",
    "L-MLT",
    "L-DLT",
    "L-C-PMF",
    "L-C-DMF",
    "L-D-FA",
]
PROTOCOL_RIGHT_DISPLAY_ORDER: List[str] = [
    "R-C-ILI",
    "R-SIP",
    "R-LIP",
    "R-C-PS",
    "R-SPS",
    "R-IPS",
    "R-DSI",
    "R-C-PO",
    "R-LPO",
    "POD",
    "R-C-OF",
    "R-C-IT",
    "R-IT",
    "R-PT",
    "R-C-AC",
    "R-AC",
    "R-C-FAC",
    "R-DAB",
    "R-SAB",
    "R-C-FHC",
    "R-FHC",
    "R-C-GT",
    "R-SGT",
    "R-LGT",
    "R-C-LT",
    "R-PLT",
    "R-MLT",
    "R-DLT",
    "R-C-PMF",
    "R-C-DMF",
    "R-D-FA",
]

FAC_CONSTRUCTION_SNAPPED = "snapped_ac_circle"
FAC_CONSTRUCTION_HEMISPHERE = "projected_hemisphere"
FAC_CONSTRUCTION_MODES: Set[str] = {
    FAC_CONSTRUCTION_SNAPPED,
    FAC_CONSTRUCTION_HEMISPHERE,
}

LEGACY_LANDMARK_DISPLAY_ORDER: List[str] = [
    # Bilateral symmetry pairs (assessed together)
    "L-LIP",
    "R-LIP",
    "L-DSI",
    "R-DSI",
    "L-POD",
    "R-POD",
    "L-IT",
    "R-IT",
    "L-E-IT",
    "R-E-IT",
    # Symphysis (four points, bilateral)
    "L-SPS",
    "R-SPS",
    "L-IPS",
    "R-IPS",
    # Left side-specific
    "L-PT",
    "L-AC",
    "L-SAB",
    "L-A-SAB",
    "L-A-ISAB",
    "L-DAB",
    "L-A-DAB",
    "L-A-IDAB",
    "L-FHC",
    "L-FAC",
    "L-SGT",
    "L-LGT",
    "L-PLT",
    "L-MLT",
    "L-DLT",
    "L-FA",
    # Right side-specific
    "R-PT",
    "R-AC",
    "R-SAB",
    "R-A-SAB",
    "R-A-ISAB",
    "R-DAB",
    "R-A-DAB",
    "R-A-IDAB",
    "R-FHC",
    "R-FAC",
    "R-SGT",
    "R-LGT",
    "R-PLT",
    "R-MLT",
    "R-DLT",
    "R-FA",
]
LANDMARK_DISPLAY_ORDER: List[str] = (
    PROTOCOL_LEFT_DISPLAY_ORDER
    + PROTOCOL_RIGHT_DISPLAY_ORDER
    + [
        lm
        for lm in LEGACY_LANDMARK_DISPLAY_ORDER
        if lm not in set(PROTOCOL_LEFT_DISPLAY_ORDER + PROTOCOL_RIGHT_DISPLAY_ORDER)
    ]
)

MULTI_REVIEW_COLORS: List[str] = [
    "#E53935",
    "#1E88E5",
    "#43A047",
    "#FB8C00",
    "#8E24AA",
    "#00ACC1",
    "#D81B60",
    "#6D4C41",
]

SEGMENTATION_OVERLAY_RGBA: Tuple[int, int, int, int] = (0, 255, 255, 90)
PELVIS_SEGMENTATION_OVERLAY_RGBA: Tuple[int, int, int, int] = (255, 176, 0, 86)
ACETABULAR_LOWER_ARC_HIGHLIGHT = "#FF4DA6"
PROJECTED_HEMISPHERE_BOUNDARY_COLOR = "orange"
PROTOCOL_CONTOUR_MARKER_COLOR = "#00FFFF"
PROTOCOL_CONTOUR_PANEL_COLOR = "#007C89"
PROTOCOL_DERIVED_UNCLASSIFIED_COLOR = "red"
PROTOCOL_DERIVED_CLASSIFIED_COLOR = "#39D353"
PROTOCOL_STATUS_MISSING_COLOR = "#D6A700"
PROTOCOL_STATUS_NOT_APPLICABLE_COLOR = "red"
PROTOCOL_POD_MEDIAL_AXIS_COLOR = "#B875FF"
PROTOCOL_POD_HELPER_COLOR = "#FFFFFF"

ONEDRIVE_ENABLED = os.environ.get("ANNOTATOR_ENABLE_ONEDRIVE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class AnnotationGUI(tk.Tk):
    HOVER_CIRCLE_LANDMARKS: Set[str] = (
        PROTOCOL_CIRCLE_CONTOUR_LANDMARKS
        | LEGACY_AC_CIRCLE_LANDMARKS
        | LEGACY_FHC_CIRCLE_LANDMARKS
    )

    def __init__(self) -> None:
        super().__init__()

        # Force Tk named fonts (for classic tk widgets)
        import tkinter.font as tkfont

        self.heading_font = tkfont.nametofont("TkDefaultFont").copy()
        self.heading_font.configure(weight="bold")  # keep size default

        self.dialogue_font = tkfont.nametofont("TkDefaultFont").copy()
        self.landmark_font = tkfont.nametofont("TkDefaultFont").copy()
        self.window_close_flag = False
        self._onedrive_backup_timer: str | None = None
        self._onedrive_upload_in_flight: bool = False
        self._upload_flag_lock = threading.Lock()
        self._pending_autosave_id: str | None = None
        self._pending_hover_radius_sync_id: str | None = None
        self._pending_hover_radius_sync_landmark = ""
        self._pending_hover_radius_sync_key = ""
        if PLATFORM == "Linux":
            self._configure_linux_fonts()
        self._configure_annotation_label_fonts()
        self._configure_landmark_table_header_font()

        self.title("2D Point Annotation")
        self.possible_image_suffix = [
            ".png",
            ".PNG",
            ".jpg",
            ".jpeg",
            ".JPEG",
            ".JPG",
            ".bmp",
            ".BMP",
            ".tif",
            ".tiff",
            ".TIFF",
            ".TIF",
        ]
        self._panel_width = 450
        self._scrollbar_width = 18
        self._start_min_w = 0
        self._start_min_h = 0
        self.use_ff = tk.BooleanVar(value=True)
        self.use_adap_cc = tk.BooleanVar(value=False)
        self.autosave_var = tk.BooleanVar(value=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.csv_path_column = "image_path"
        self.dirty = False
        self.path_var = tk.StringVar(value="No image loaded")
        self.quality_var = tk.StringVar(value="N/A")
        self.selected_landmark = tk.StringVar(value="")
        self._landmark_ref: LandmarkReference | None = None
        self._landmark_ref_dialog: LandmarkReferenceDialog | None = None
        _lm_json = Path(__file__).parent.parent / "docs" / "landmarks.json"
        if _lm_json.exists():
            try:
                self._landmark_ref = LandmarkReference(_lm_json)
            except Exception:
                logger.warning(
                    "Failed to load landmark reference: %s", _lm_json, exc_info=True
                )
        self.current_view_var = tk.StringVar(value="")
        self.view_dropdown: ttk.Combobox | None = None
        self.image_flag_var = tk.BooleanVar(value=False)
        self.landmark_visibility: Dict[str, tk.BooleanVar] = {}
        self.landmark_found = {}
        self.landmark_flagged: Dict[str, tk.BooleanVar] = {}
        self.landmark_flag_widgets: Dict[str, tk.Checkbutton] = {}
        self.landmark_found_widgets: Dict[str, tk.Widget] = {}
        self.annotations: Dict[str, Dict[str, AnnotationValue]] = {}
        self.landmarks: List[str] = []
        self.images: List[Path] = []
        self.image_index_map: Dict[str, int] = {}
        self.current_image_index: int = -1
        self.json_path: Optional[Path] = None
        self.json_dir: Optional[Path] = None
        self.json_data: Dict = {"landmarks": [], "views": {}, "images": []}
        self._json_metadata: Dict = {}  # Metadata tracking for traceability
        self._original_json_hash: Optional[str] = (
            None  # SHA-256 of loaded JSON for backup dedup
        )
        self._original_json_backed_up: bool = (
            False  # Track if original has been backed up
        )
        self.allowed_views: Dict[str, List[str]] = {}
        self.landmark_meta: Dict[str, Dict[str, Dict[str, Union[bool, str]]]] = {}
        self.hover_radii: Dict[str, Dict[str, int]] = {}
        self.categorical_annotations: Dict[str, Dict[str, Any]] = {}
        self.saved_image_snapshots: Dict[str, str] = {}
        self.queue_status_var = tk.StringVar(value="")
        self.multi_review_status_var = tk.StringVar(value="")
        self.exit_queue_btn: tk.Button | None = None
        self.exit_csv_check_btn: tk.Button | None = None
        self.exit_multi_review_btn: tk.Button | None = None
        self.image_tree: ttk.Treeview | None = None
        self._suspend_image_tree_select = False
        self._navigation_in_progress = False
        self.current_image_flag = False
        self.image_flag_check: tk.Checkbutton | None = None
        self.current_image_direction: str = "AP"
        self.image_direction_var = tk.StringVar(value="AP")
        self.line_landmarks: Set[str] = (
            set(PROTOCOL_LINE_LANDMARKS) | LEGACY_LINE_LANDMARKS
        )
        self.ellipse_landmarks: Set[str] = set(ELLIPSE_LANDMARKS)
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.current_image_quality: int = 0
        self.img_obj: ImageTk.PhotoImage | None = None
        self.seg_img_objs: Dict[str, ImageTk.PhotoImage] = {}
        self.seg_item_ids: Dict[str, int] = {}
        self.seg_masks: Dict[str, Dict[str, np.ndarray]] = {}
        self.pelvis_segmenter: PelvisSegmentationBackend | None = None
        self.pelvis_segmenters: Dict[str, PelvisSegmentationBackend] = {}
        configured_pelvis_backend = (
            os.environ.get("PELVIS_SEGMENTATION_BACKEND") or "auto"
        ).strip().lower()
        valid_pelvis_backends = {mode for mode, _label in PELVIS_SEGMENTATION_BACKEND_OPTIONS}
        if configured_pelvis_backend not in valid_pelvis_backends:
            configured_pelvis_backend = "auto"
        self.pelvis_segmentation_backend_var = tk.StringVar(
            value=configured_pelvis_backend
        )
        self.pelvis_segmentation_running: bool = False
        self.pelvis_segmentation_undo_stack: Dict[str, List[np.ndarray | None]] = {}
        self.pelvis_segmentation_status_var = tk.StringVar(value="")
        self.segment_pelvis_btn: tk.Button | None = None
        self.undo_pelvis_btn: tk.Button | None = None
        self.pelvis_segmentation_status_label: tk.Label | None = None
        self.it_edge_border_img_objs: Dict[str, ImageTk.PhotoImage] = {}
        self.it_edge_border_item_ids: Dict[str, int] = {}
        self.it_edge_border_masks: Dict[str, Dict[str, np.ndarray]] = {}
        self.protocol_pod_constructions: Dict[str, Dict[str, Any]] = {}
        self.it_edge_border_live_item_id: Optional[int] = None
        self.it_edge_border_live_dot_id: Optional[int] = None
        self.it_edge_border_live_landmark: Optional[str] = None
        self.it_edge_border_live_coords: List[float] = []
        self.it_edge_border_last_overlay_update_at: float = 0.0
        self.it_edge_border_last_zoom_update_at: float = 0.0
        self.it_edge_border_drag_update_interval_sec: float = 1.0 / 30.0
        self.pair_line_ids: Dict[str, int] = {}
        self.last_seed: Dict[str, Optional[Tuple[int, int]]] = {}
        self.hover_enabled = tk.BooleanVar(value=False)
        self.hover_radius = tk.IntVar(value=25)
        self.hover_circle_id: Optional[int] = None
        self.fac_construction_mode = tk.StringVar(value=FAC_CONSTRUCTION_SNAPPED)
        self.projected_hemisphere_z = tk.IntVar(value=70)
        self.projected_hemisphere_z_scale: tk.Scale | None = None
        self.ellipse_snap_check: tk.Checkbutton | None = None
        self.ellipse_lock_check: tk.Checkbutton | None = None
        self.fac_apply_button: tk.Button | None = None
        self.ellipse_snap_to_ac_enabled = tk.BooleanVar(value=True)
        self.ellipse_lock_major_when_snapped_enabled = tk.BooleanVar(value=True)
        self.femoral_axis_enabled = tk.BooleanVar(value=False)
        self.femoral_axis_count = tk.IntVar(value=5)
        self.femoral_axis_proj_length = tk.IntVar(value=60)
        self.femoral_axis_whisker_tip_length = tk.IntVar(value=10)
        self.femoral_axis_item_ids: list[int] = []
        self.femoral_axis_count_scale: tk.Scale | None = None
        self.femoral_axis_whisker_tip_length_scale: tk.Scale | None = None
        self.extended_crosshair_enabled = tk.BooleanVar(value=False)
        self.extended_crosshair_length = tk.IntVar(value=50)
        self.extended_crosshair_ids: list[int] = []
        self.method = tk.StringVar(value="Flood Fill")
        self.fill_sensitivity = tk.IntVar(value=18)
        self.edge_lock = tk.BooleanVar(value=True)
        self.edge_lock_width = tk.IntVar(value=2)
        self.use_clahe = tk.BooleanVar(value=True)
        self.grow_shrink = tk.IntVar(value=0)
        self.it_edge_border_mode = tk.StringVar(value="draw")
        self.it_edge_border_brush_size = tk.IntVar(value=1)
        self.it_edge_border_brush_scale: tk.Scale | None = None
        self.disp_scale: float = 1.0
        self.disp_off: Tuple[int, int] = (0, 0)  # (offset_x, offset_y)
        self.disp_size: Tuple[int, int] = (0, 0)  # (disp_w, disp_h)
        self.base_img_item: Optional[int] = None
        self.mouse_crosshair_ids: list[int] = []
        self.last_mouse_canvas_pos: tuple[int, int] | None = None
        self.right_mouse_held: bool = False
        self.zoom_canvas: tk.Canvas | None = None
        self.zoom_percent = tk.IntVar(value=8)
        self.show_selected_landmark_in_zoom: tk.BooleanVar = tk.BooleanVar(value=True)
        self.enable_zoom_contrast: tk.BooleanVar = tk.BooleanVar(value=False)
        self.enable_zoom_pixel_art: tk.BooleanVar = tk.BooleanVar(value=False)
        self.enable_zoom_percentile_stretch: tk.BooleanVar = tk.BooleanVar(value=False)
        self.zoom_percentile_low: tk.IntVar = tk.IntVar(value=1)
        self.zoom_percentile_high: tk.IntVar = tk.IntVar(value=99)
        self.zoom_img_obj: ImageTk.PhotoImage | None = None
        self.zoom_seg_img_obj: ImageTk.PhotoImage | None = None
        self.zoom_base_item: Optional[int] = None
        self.zoom_src_rect: Tuple[float, float, float, float] | None = None
        self.line_preview_id: Optional[int] = None
        self.dragging_landmark: Optional[str] = None
        self.dragging_point_index: Optional[int] = None
        self.dragging_line_whole: bool = False
        self.dragging_line_last_img_pos: Optional[AnnotationPoint] = None
        self.dragging_ellipse_handle: Optional[str] = None
        self.dragging_ellipse_whole: bool = False
        self.dragging_ellipse_last_img_pos: Optional[AnnotationPoint] = None
        self.drawing_it_edge_border: bool = False
        self.it_edge_border_last_img_pos: Optional[AnnotationPoint] = None
        self.drag_tolerance_px = 10
        self.drag_line_tolerance_px = 8
        self.zoom_crosshair_ids: list[int] = []
        self.zoom_extended_crosshair_ids: list[int] = []
        self.zoom_landmark_overlay_ids: list[int] = []
        self.zoom_hover_circle_id: Optional[int] = None
        self.lm_settings: Dict[str, Dict[str, Dict]] = {}
        self.note_text: Optional[tk.Text] = None
        self.note_text_internal_update: bool = False
        self.landmark_status_var = tk.StringVar(value="")
        self.landmark_status_combo: ttk.Combobox | None = None
        self.left_hip_category_var = tk.StringVar(value="")
        self.right_hip_category_var = tk.StringVar(value="")
        self.derived_feature_review_var = tk.StringVar(value="")
        self.derived_feature_status_var = tk.StringVar(value="")
        self.derived_feature_combo: ttk.Combobox | None = None
        self.derived_feature_status_combo: ttk.Combobox | None = None
        self.derived_feature_status_vars: Dict[str, tk.StringVar] = {}
        self.derived_feature_status_widgets: Dict[str, ttk.Combobox] = {}
        self.derived_feature_status_frame: tk.Frame | None = None
        self._categorical_shortcut_focus_key = ""
        self.landmark_correct_labels: Dict[str, tk.Label] = {}
        self.landmark_delete_widgets: Dict[str, tk.Button] = {}
        self.landmark_group_frames: Dict[str, tk.Frame] = {}
        self.landmark_group_children: Dict[str, List[tk.Widget]] = {}
        self._active_landmark_group: str = ""
        self.multi_review_mode = False
        self.multi_review_reviewer_order: List[str] = []
        self.multi_review_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.multi_review_annotations: Dict[
            str, Dict[str, Dict[str, AnnotationValue]]
        ] = {}
        self.multi_review_meta: Dict[
            str, Dict[str, Dict[str, Dict[str, Union[bool, str]]]]
        ] = {}
        self.multi_review_image_key_by_path: Dict[str, str] = {}
        self.multi_review_temp_annotations: Dict[
            str, Dict[str, AnnotationValue]
        ] = {}
        self.multi_review_visibility: Dict[str, tk.BooleanVar] = {}
        self.multi_review_visibility_widgets: Dict[str, tk.Checkbutton] = {}
        self.multi_review_frame: ttk.LabelFrame | None = None
        self.multi_review_summary_var = tk.StringVar(value="")
        self.csv_loaded = False
        # Also adjust these if you want everything to match:
        # Already configured above to match heading_font
        self._setup_ui()
        self.after(0, self._fit_window_and_set_min)
        self.after(50, self._lock_initial_minsize)
        self.unbind("<Up>")
        self.unbind("<Down>")
        self._bind_shortcut("<Up>", self._on_arrow_up)
        self._bind_shortcut("<Down>", self._on_arrow_down)
        self._bind_shortcut("<f>", self._on_arrow_down)
        self._bind_shortcut("<d>", self._on_arrow_up)
        self._bind_shortcut("<Left>", self._on_arrow_left)
        self._bind_shortcut("<Right>", self._on_arrow_right)
        self._bind_shortcut("<Control-b>", self._on_pg_up)
        self._bind_shortcut("<Control-n>", self._on_pg_down)
        self._bind_shortcut("<n>", self._on_pg_down)
        self._bind_shortcut("<g>", self._on_pg_down)
        self._bind_shortcut("<b>", self._on_pg_up)
        self._bind_shortcut("<BackSpace>", self._on_backspace)
        self._bind_shortcut("<Delete>", self._on_backspace)
        self._bind_shortcut("<space>", self._on_space)
        for index in range(1, 10):
            self._bind_shortcut(str(index), self._make_categorical_shortcut_handler(index))
        self._bind_shortcut("<Tab>", self._on_tab_press)
        self._bind_shortcut("<h>", self._on_h_press)
        self._bind_shortcut("<question>", self._open_landmark_reference)
        self.queue_mode = False
        self.unannotated_queue: List[Path] = []
        self.queue_index = 0
        self.check_csv_mode = False
        self.csv_path_queue: List[Path] = []
        self.csv_index = 0
        self.db_path: Optional[Path] = None
        self.last_update = datetime.now()
        # Initialize path attributes to prevent AttributeError if accessed before assignment
        self.abs_csv_path: Optional[str] = None
        self.absolute_current_image_path: Optional[Path] = None
        self.csv_local_image_directory_path: Optional[str] = None

        self.onedrive_backup: OneDriveBackup | None = None
        if ONEDRIVE_ENABLED:
            self.onedrive_backup = OneDriveBackup()
            self.after(100, self._init_onedrive_credentials)

    # Builds the left image canvas, right control panel, and tool widgets.
    def focus_widget(self, event):
        event.widget.focus_set()

    def _on_space(self, event) -> None:
        """
        Here, we are marking the current image as "verified" in the table
        """
        if getattr(self, "multi_review_mode", False):
            return

        if self.db_path is not None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.cursor()
                    if self.absolute_current_image_path is None:
                        return
                    image_filename = extract_filename(self.absolute_current_image_path)

                    cur.execute(
                        """
                        INSERT INTO annotations (image_filename, verified)
                        VALUES (?, 1)
                        ON CONFLICT(image_filename) DO UPDATE
                        SET verified = 1-annotations.verified
                        """,
                        (image_filename,),
                    )
                    conn.commit()
                self._draw_points()
            except sqlite3.Error as e:
                messagebox.showerror(
                    "Database Error",
                    f"Failed to toggle verification status:\n{e}",
                )
        return

    def _is_current_image_verified(self) -> bool:
        if self.db_path is not None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.cursor()
                    if self.absolute_current_image_path is None:
                        return False

                    query = """
                        SELECT verified FROM annotations WHERE image_filename = ?
                    """
                    image_fname = extract_filename(self.absolute_current_image_path)

                    cur.execute(
                        query,
                        (image_fname,),
                    )
                    row = cur.fetchone()
                return bool(row[0] if row else False)
            except sqlite3.Error:
                # Silently return False if we can't check - non-critical operation
                return False
        return False

    def _resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()
        if self.json_dir is not None:
            standard = (self.json_dir / path).resolve()
            if standard.exists():
                return standard
            # Fallback: if the JSON lives inside the directory referenced by
            # the first component of image_path, strip that component.
            # e.g. image_path="group1/img.tif", json is in .../group1/
            #   → standard tries .../group1/group1/img.tif (wrong)
            #   → fallback strips "group1/" → .../group1/img.tif (correct)
            parts = path.parts
            if len(parts) > 1 and parts[0] == self.json_dir.name:
                return (self.json_dir / Path(*parts[1:])).resolve()
            return standard
        return path.resolve()

    def _path_key(self, path: Union[str, Path]) -> str:
        return str(Path(path).resolve())

    def _get_current_image_record(self) -> Optional[Dict]:
        if self.current_image_path is None:
            return None
        idx = self.image_index_map.get(self._path_key(self.current_image_path))
        if idx is None:
            return None
        images = self.json_data.get("images", [])
        if not isinstance(images, list) or idx >= len(images):
            return None
        record = images[idx]
        return record if isinstance(record, dict) else None

    def _sync_current_state_to_json_record(self) -> None:
        record = self._get_current_image_record()
        if record is None or self.current_image_path is None:
            return

        self._update_protocol_derived_features_for_current_image(mark_dirty=False)
        record["image_flag"] = bool(self.current_image_flag)
        record["image_direction"] = self.current_image_direction
        record["view"] = self.current_view_var.get().strip() or None
        record["annotations"] = self._prepare_landmark_data(for_json=True)
        record["categories"] = self._prepare_categorical_data()
        record["app_version"] = self._get_app_version()
        record["protocol_version"] = self._get_protocol_version()

        key = self._path_key(self.current_image_path)
        record["resolved_image_path"] = key
        self.annotations[key] = self._parse_annotations_for_record(record)
        self.categorical_annotations[key] = (
            self._parse_categorical_annotations_for_record(record)
        )

    def _autosave_annotation_change(self) -> bool:
        if self.json_path is not None and self._get_current_image_record() is not None:
            return self._maybe_autosave_current_image()
        return self._auto_save_to_db()

    def _display_ordered(self, landmarks: List[str]) -> List[str]:
        """Return landmarks sorted by LANDMARK_DISPLAY_ORDER; unknowns appended in original order."""
        order_index = {lm: i for i, lm in enumerate(LANDMARK_DISPLAY_ORDER)}
        known = [lm for lm in LANDMARK_DISPLAY_ORDER if lm in set(landmarks)]
        unknown = [lm for lm in landmarks if lm not in order_index]
        return known + unknown

    def _check_landmark_display_order(self) -> None:
        json_set = set(self.landmarks)
        display_set = set(LANDMARK_DISPLAY_ORDER)
        missing_from_display = json_set - display_set
        problems = []
        if missing_from_display:
            problems.append(
                f"In JSON but missing from LANDMARK_DISPLAY_ORDER: {sorted(missing_from_display)}"
            )
        if problems:
            msg = "Landmark display order mismatch:\n\n" + "\n\n".join(problems)
            logger.warning(msg)
            messagebox.showwarning("Landmark Order Mismatch", msg)

    def _json_uses_protocol_landmarks(
        self, landmarks: List[str], views: Dict[str, List[str]]
    ) -> bool:
        names = set(landmarks)
        for view_landmarks in views.values():
            names.update(lm for lm in view_landmarks if isinstance(lm, str))
        return bool(names & PROTOCOL_CONTOUR_LANDMARKS)

    def _protocol_landmarks_for_view(self, view_name: str) -> List[str]:
        view_lower = str(view_name or "").lower()
        if "left" in view_lower and "right" not in view_lower:
            return [lm for lm in PROTOCOL_LEFT_DISPLAY_ORDER if lm != "POD"]
        if "right" in view_lower and "left" not in view_lower:
            return [lm for lm in PROTOCOL_RIGHT_DISPLAY_ORDER if lm != "POD"]
        return list(PROTOCOL_LEFT_DISPLAY_ORDER + PROTOCOL_RIGHT_DISPLAY_ORDER)

    def _inject_protocol_landmarks(
        self, landmarks: List[str], views: Dict[str, List[str]]
    ) -> List[str]:
        _ = landmarks
        protocol_order = list(PROTOCOL_LEFT_DISPLAY_ORDER + PROTOCOL_RIGHT_DISPLAY_ORDER)
        seen: Set[str] = set()
        protocol_landmarks = [lm for lm in protocol_order if not (lm in seen or seen.add(lm))]

        for view_name, view_landmarks in views.items():
            if not isinstance(view_landmarks, list):
                views[view_name] = self._protocol_landmarks_for_view(view_name)
                continue
            view_landmarks[:] = self._protocol_landmarks_for_view(view_name)
        return protocol_landmarks

    def _inject_ellipse_landmarks(
        self, landmarks: List[str], views: Dict[str, List[str]]
    ) -> List[str]:
        return self._inject_protocol_landmarks(landmarks, views)

    def _insert_landmark_after(
        self, landmarks: List[str], landmark: str, anchor: str
    ) -> None:
        if landmark in landmarks:
            return
        try:
            idx = landmarks.index(anchor)
        except ValueError:
            landmarks.append(landmark)
            return
        landmarks.insert(idx + 1, landmark)

    def _get_allowed_landmarks_for_current_view(self) -> set[str]:
        if getattr(self, "multi_review_mode", False):
            return set(self.landmarks)
        view = self.current_view_var.get().strip()
        if not view:
            return set()
        return set(self.allowed_views.get(view, []))

    def _get_current_view(self) -> str:
        record = self._get_current_image_record()
        if record is None:
            return ""
        val = record.get("view")
        return "" if val is None else str(val)

    def _set_current_view(self, view_name: str) -> None:
        record = self._get_current_image_record()
        if record is None:
            return
        old_view = record.get("view")
        record["view"] = view_name
        self.current_view_var.set(view_name)
        if not self._prune_annotations_for_current_view():
            # User cancelled — revert view
            record["view"] = old_view
            self.current_view_var.set(old_view or "")
            return
        self._rebuild_landmark_panel_for_view()
        self._load_note_for_selected_landmark()
        self.dirty = True
        self._refresh_image_listbox()

    def _prune_annotations_for_current_view(self) -> bool:
        """Remove annotations not in the current view. Returns False if user cancelled."""
        if getattr(self, "multi_review_mode", False):
            return True
        if self.current_image_path is None:
            return True

        allowed = self._get_allowed_landmarks_for_current_view()
        key = self._path_key(self.current_image_path)
        pts = self.annotations.setdefault(key, {})
        meta = self.landmark_meta.setdefault(key, {})
        radii = getattr(self, "hover_radii", {}).get(key, {})

        to_delete = [lm for lm in pts if lm not in allowed]
        if not to_delete:
            return True
        names = ", ".join(sorted(to_delete))
        if not messagebox.askyesno(
            "Change View",
            f"Switching view will remove {len(to_delete)} annotation(s) "
            f"not in the new view:\n{names}\n\nContinue?",
        ):
            return False
        for lm in to_delete:
            if lm not in pts:
                continue
            was_projected_fac = self._is_ellipse_landmark(
                lm
            ) and self._ellipse_is_projected_hemisphere(lm, pts.get(lm))
            was_projected_fac = self._is_ellipse_landmark(
                lm
            ) and self._ellipse_is_projected_hemisphere(lm, pts.get(lm))
            del pts[lm]
            meta.pop(lm, None)
            radii.pop(lm, None)
            if self._is_ellipse_landmark(lm):
                self._clear_auto_acetabular_points_for_ellipse(lm)
                if was_projected_fac:
                    self._clear_projected_hemisphere_ac_for_ellipse(lm)
            if self._matching_ellipse_landmark_for_ac(lm) is not None:
                self._clear_projected_hemisphere_for_ac(lm)
                self._clear_auto_acetabular_points_for_ac(lm)
            if self._is_segmentation_landmark(lm):
                self.last_seed.pop(lm, None)
                getattr(self, "seg_masks", {}).get(key, {}).pop(lm, None)
                self._remove_overlay_for(lm)
            if self._is_border_segmentation_landmark(lm):
                self._clear_it_edge_border_state(lm, clear_annotation=False)
        return True

    def _rebuild_landmark_panel_for_view(self) -> None:
        allowed = self._get_allowed_landmarks_for_current_view()
        current = self.selected_landmark.get()

        self._build_landmark_panel()

        if current in allowed:
            self.selected_landmark.set(current)
        elif self.landmarks:
            for lm in self.landmarks:
                if lm in allowed:
                    self.selected_landmark.set(lm)
                    break
            else:
                self.selected_landmark.set("")
        else:
            self.selected_landmark.set("")

        self._set_active_landmark_group_from_selection()
        self._draw_points()
        self._refresh_categorical_controls()

    def _prompt_for_view_if_needed(self) -> None:
        if self.current_image_path is None:
            return

        current_view = self._get_current_view()
        if current_view in self.allowed_views:
            self.current_view_var.set(current_view)
            self._prune_annotations_for_current_view()
            self._rebuild_landmark_panel_for_view()
            return

        if not self.allowed_views:
            return

        popup = tk.Toplevel(self)
        popup.title("Select View")
        popup.transient(self)
        popup.resizable(False, False)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="This image needs a valid view selection.",
            font=self.dialogue_font,
        ).pack(anchor="w", pady=(0, 8))

        view_names = list(self.allowed_views.keys())
        choice_var = tk.StringVar(value=view_names[0])

        combo = ttk.Combobox(
            frame,
            textvariable=choice_var,
            values=view_names,
            state="readonly",
            font=self.dialogue_font,
            width=28,
        )
        combo.pack(fill="x", pady=(0, 10))
        combo.current(0)
        combo.focus_set()

        result = {"done": False}

        def confirm() -> None:
            if result["done"]:
                return
            result["done"] = True
            self._set_current_view(choice_var.get())
            self.current_view_var.set(choice_var.get())
            popup.destroy()
            self._maybe_autosave_current_image()

        popup.protocol("WM_DELETE_WINDOW", confirm)

        ttk.Button(frame, text="OK", command=confirm).pack(anchor="e")

        popup.update_idletasks()
        popup.deiconify()
        popup.lift()
        popup.grab_set()
        popup.wait_window()

    def _on_view_selected(self, _event=None) -> None:
        if getattr(self, "multi_review_mode", False):
            return
        new_view = self.current_view_var.get().strip()
        if new_view not in self.allowed_views:
            return
        self._set_current_view(new_view)
        self._maybe_autosave_current_image()
        self.focus_set()

    def _on_image_flag_widget_changed(self) -> None:
        if getattr(self, "multi_review_mode", False):
            return
        self.current_image_flag = bool(self.image_flag_var.get())
        record = self._get_current_image_record()
        if record is not None:
            record["image_flag"] = self.current_image_flag

        self._refresh_image_flag_checkbox_style()

        self.dirty = True
        self._maybe_autosave_current_image()

    def _refresh_image_flag_checkbox_style(self) -> None:
        if self.image_flag_check is None:
            return

        is_flagged = bool(self.image_flag_var.get())

        try:
            self.image_flag_check.configure(
                fg="black",
                activeforeground="black",
                disabledforeground="black",
                selectcolor="#FFB6B6" if is_flagged else self.cget("bg"),
            )
        except Exception:
            pass

    def _set_multi_review_controls_state(self) -> None:
        readonly = bool(getattr(self, "multi_review_mode", False))

        normal_or_disabled = "disabled" if readonly else "normal"
        readonly_combo_state = "disabled" if readonly else "readonly"

        for widget_name, state in (
            ("image_flag_check", normal_or_disabled),
            ("autosave_check", normal_or_disabled),
            ("direction_dropdown", readonly_combo_state),
            ("view_dropdown", readonly_combo_state),
        ):
            widget = self.__dict__.get(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except Exception:
                pass

        if self.exit_multi_review_btn is not None:
            try:
                self.exit_multi_review_btn.configure(
                    state="normal" if readonly else "disabled"
                )
            except Exception:
                pass

        if readonly:
            self._set_note_editor_enabled(False)

        self._refresh_pelvis_segmentation_buttons()
        self._refresh_categorical_controls()

    def _current_multi_review_image_key(self) -> str | None:
        if not getattr(self, "multi_review_mode", False):
            return None
        if self.current_image_path is None:
            return None
        return self.multi_review_image_key_by_path.get(
            self._path_key(self.current_image_path)
        )

    def _current_multi_review_temp_annotations(self) -> Dict[str, AnnotationValue]:
        image_key = self._current_multi_review_image_key()
        if image_key is None:
            return {}
        return self.multi_review_temp_annotations.setdefault(image_key, {})

    def _is_multi_review_reviewer_visible(self, reviewer: str) -> bool:
        var = self.multi_review_visibility.get(reviewer)
        return True if var is None else bool(var.get())

    def _is_ellipse_landmark(self, lm: str) -> bool:
        return lm in self.__dict__.get("ellipse_landmarks", ELLIPSE_LANDMARKS)

    def _is_segmentation_landmark(self, lm: str) -> bool:
        return lm in SEGMENTATION_LANDMARKS

    def _is_pelvis_segmentation_landmark(self, lm: str) -> bool:
        return lm == PELVIS_SEGMENTATION_LANDMARK

    def _is_auto_segmentation_landmark(self, lm: str) -> bool:
        return lm in AUTO_SEGMENTATION_LANDMARKS

    def _is_border_segmentation_landmark(self, lm: str) -> bool:
        return lm in IT_EDGE_LANDMARKS

    def _uses_manual_segmentation_grid(self, lm: str) -> bool:
        return self._is_border_segmentation_landmark(lm)

    def _is_protocol_derived_feature_name(self, lm: str) -> bool:
        return lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS

    def _protocol_sources_available_for_derived_feature(self, lm: str) -> bool:
        sources = PROTOCOL_DERIVED_FEATURE_SOURCES.get(lm, [])
        if not sources:
            return lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS
        landmark_set = set(getattr(self, "landmarks", []))
        return all(source in landmark_set for source in sources)

    def _is_protocol_derived_feature(self, lm: str) -> bool:
        return self._is_protocol_derived_feature_name(
            lm
        ) and self._protocol_sources_available_for_derived_feature(lm)

    def _is_protocol_contour_source(self, lm: str) -> bool:
        return lm in PROTOCOL_CONTOUR_LANDMARKS

    def _protocol_direct_order_for_current_view(self) -> List[str]:
        allowed = self._get_allowed_landmarks_for_current_view()
        view = self.current_view_var.get().strip()
        view_order = self._protocol_landmarks_for_view(view)
        return [
            lm
            for lm in view_order
            if lm in PROTOCOL_DIRECT_LANDMARKS and (not allowed or lm in allowed)
        ]

    def _landmark_annotation_geometry_complete(self, lm: str) -> bool:
        pts, _quality = self._get_annotations()
        if self._is_border_segmentation_landmark(lm):
            mask = self._manual_contour_mask_for_protocol_source(lm)
            return mask is not None and bool(np.any(mask > 0))
        if lm not in pts:
            return False
        if self._is_line_landmark(lm):
            return len(self._get_line_points(lm)) >= 2
        if self._is_ellipse_landmark(lm):
            return self._normalize_ellipse_value(pts.get(lm)) is not None
        return True

    def _direct_sources_for_derived_feature(
        self, feature: str, visited: Set[str] | None = None
    ) -> Set[str]:
        if feature in PROTOCOL_DIRECT_LANDMARKS:
            return {feature}
        if visited is None:
            visited = set()
        if feature in visited:
            return set()
        visited.add(feature)

        direct_sources: Set[str] = set()
        for source in PROTOCOL_DERIVED_FEATURE_SOURCES.get(feature, []):
            if source in PROTOCOL_DIRECT_LANDMARKS:
                direct_sources.add(source)
            elif source in PROTOCOL_DERIVED_FEATURE_LANDMARKS:
                direct_sources.update(
                    self._direct_sources_for_derived_feature(source, visited)
                )
        return direct_sources

    def _derived_feature_owner_landmark(self, feature: str) -> str:
        direct_sources = self._direct_sources_for_derived_feature(feature)
        if not direct_sources:
            return ""

        for lm in reversed(self._protocol_direct_order_for_current_view()):
            if lm in direct_sources:
                return lm
        for lm in reversed(PROTOCOL_DIRECT_LANDMARKS):
            if lm in direct_sources:
                return lm
        return ""

    def _relevant_derived_features_for_landmark(self, lm: str) -> List[str]:
        if not lm or lm not in PROTOCOL_DIRECT_LANDMARKS:
            return []
        allowed_features = set(self._allowed_derived_features_for_current_view())
        candidates: Set[str] = set()
        visited_sources: Set[str] = set()
        frontier: List[str] = [lm]
        while frontier:
            source = frontier.pop(0)
            if source in visited_sources:
                continue
            visited_sources.add(source)
            derived_from_source = list(
                PROTOCOL_DERIVED_FEATURES_BY_SOURCE.get(source, [])
            )
            derived_from_source.extend(
                feature
                for feature, sources in PROTOCOL_DERIVED_FEATURE_SOURCES.items()
                if source in sources
            )
            for feature in derived_from_source:
                if feature not in allowed_features or feature in candidates:
                    continue
                owner = self._derived_feature_owner_landmark(feature)
                if owner and owner != lm:
                    continue
                candidates.add(feature)
                frontier.append(feature)
        return [feature for feature in LANDMARK_DISPLAY_ORDER if feature in candidates]

    def _relevant_derived_feature_statuses_complete(self, lm: str) -> bool:
        features = self._relevant_derived_features_for_landmark(lm)
        if not features:
            return True
        categories = self._current_categorical_annotations()
        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            return False
        return all(
            bool(str(derived.get(feature, "") or "").strip().lower())
            and str(derived.get(feature, "") or "").strip().lower()
            in DERIVED_FEATURE_STATUS_VALUES
            for feature in features
        )

    def _direct_landmark_complete(self, lm: str) -> bool:
        if lm not in PROTOCOL_DIRECT_LANDMARKS:
            return False
        if self._get_landmark_status(lm):
            return True
        if not self._landmark_annotation_geometry_complete(lm):
            return False
        return self._relevant_derived_feature_statuses_complete(lm)

    def _next_incomplete_protocol_direct_landmark(self) -> str:
        for lm in self._protocol_direct_order_for_current_view():
            if not self._direct_landmark_complete(lm):
                return lm
        return ""

    def _landmark_order_index(self, lm: str) -> int | None:
        order = self._protocol_direct_order_for_current_view()
        try:
            return order.index(lm)
        except ValueError:
            return None

    def _landmark_unlocked_by_protocol_order(self, lm: str) -> bool:
        if lm not in PROTOCOL_DIRECT_LANDMARKS or self.current_image_path is None:
            return True
        order = self._protocol_direct_order_for_current_view()
        if lm not in order:
            return True
        next_lm = self._next_incomplete_protocol_direct_landmark()
        if not next_lm:
            return True
        return order.index(lm) <= order.index(next_lm)

    def _advance_to_next_protocol_landmark_if_current_complete(self, lm: str) -> None:
        if lm not in PROTOCOL_DIRECT_LANDMARKS:
            return
        if not self._direct_landmark_complete(lm):
            return
        next_lm = self._next_incomplete_protocol_direct_landmark()
        if not next_lm or next_lm == lm:
            self._refresh_landmark_selectability()
            self._refresh_landmark_label_styles()
            return
        self.selected_landmark.set(next_lm)
        self._on_landmark_selected()
        self._set_status(f"{lm} complete. Next: {next_lm}.")

    def _locked_by_protocol_order_message(self, lm: str) -> str:
        if not self._image_categories_complete():
            return "Select Left Hip and Right Hip image categories before landmarking."
        next_lm = self._next_incomplete_protocol_direct_landmark()
        if next_lm:
            return f"Complete or mark {next_lm} before annotating {lm}."
        return f"{lm} is locked by the protocol annotation order."

    def _derived_feature_is_classified(self, lm: str) -> bool:
        if not self._is_protocol_derived_feature_name(lm):
            return False
        categories = self._current_categorical_annotations()
        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            return False
        status = str(derived.get(lm, "") or "").strip().lower()
        return status in DERIVED_FEATURE_STATUS_VALUES and bool(status)

    @staticmethod
    def _normalize_direct_annotation_status(status: object) -> str:
        value = str(status or "").strip().lower()
        if value in {"not applicable", "not-applicable", "not_applicable", "n/a", "na"}:
            value = "not_applicable"
        return value if value in DIRECT_ANNOTATION_STATUS_VALUES else ""

    @staticmethod
    def _choice_display(value: object) -> str:
        text = str(value or "").strip().replace("_", " ")
        if not text:
            return ""
        return " ".join(part[:1].upper() + part[1:].lower() for part in text.split())

    @staticmethod
    def _normalize_hip_category(value: object) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in HIP_CATEGORY_VALUES else ""

    @staticmethod
    def _hip_category_display(value: object) -> str:
        return AnnotationGUI._choice_display(AnnotationGUI._normalize_hip_category(value))

    @staticmethod
    def _normalize_derived_feature_status(value: object) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in DERIVED_FEATURE_STATUS_VALUES else ""

    @staticmethod
    def _derived_feature_status_display(value: object) -> str:
        return AnnotationGUI._choice_display(
            AnnotationGUI._normalize_derived_feature_status(value)
        )

    @staticmethod
    def _direct_annotation_status_display(status: object) -> str:
        value = AnnotationGUI._normalize_direct_annotation_status(status)
        return AnnotationGUI._choice_display(value)

    def _landmark_has_annotation_value(self, lm: str) -> bool:
        if self._is_border_segmentation_landmark(lm):
            mask = self._manual_contour_mask_for_protocol_source(lm)
            return mask is not None and bool(np.any(mask > 0))
        pts, _quality = self._get_annotations()
        return lm in pts

    def _landmark_allows_direct_status(self, lm: str) -> bool:
        if not lm or not self._landmark_is_selectable(lm):
            return False
        return not (
            self._is_protocol_derived_feature(lm)
            or self._is_derived_acetabular_landmark(lm)
            or self._is_intermediate_acetabular_landmark(lm)
        )

    def _canvas_colors_for_landmark(
        self,
        lm: str,
        *,
        selected: bool,
        flagged: bool,
        image_verified: bool,
    ) -> Tuple[str, str]:
        if selected:
            return "blue", "orange"
        if self._is_protocol_derived_feature(lm):
            status = self._derived_feature_status_for_landmark(lm)
            if status == "correct":
                return PROTOCOL_DERIVED_CLASSIFIED_COLOR, PROTOCOL_DERIVED_CLASSIFIED_COLOR
            if status == "incorrect":
                return PROTOCOL_DERIVED_UNCLASSIFIED_COLOR, PROTOCOL_DERIVED_UNCLASSIFIED_COLOR
            return "black", "black"
        if self._is_protocol_contour_source(lm):
            return PROTOCOL_CONTOUR_MARKER_COLOR, PROTOCOL_CONTOUR_MARKER_COLOR
        return (
            "red" if not image_verified else "green",
            "red" if flagged else "yellow",
        )

    def _landmark_should_render_on_canvas(self, lm: str) -> bool:
        if not self._is_protocol_derived_feature(lm):
            return True
        return self._derived_feature_status_for_landmark(lm) != "missing"

    def _label_font_for_landmark(self, lm: str, *, selected: bool):
        if selected:
            return self.__dict__.get("landmark_font", self.__dict__.get("dialogue_font"))
        if self._is_protocol_contour_source(lm):
            return self.__dict__.get(
                "contour_label_font", self.__dict__.get("dialogue_font")
            )
        if self._is_protocol_derived_feature(lm):
            return self.__dict__.get(
                "derived_label_font", self.__dict__.get("dialogue_font")
            )
        return self.__dict__.get("dialogue_font")

    def _panel_label_color_for_landmark(self, lm: str) -> str:
        if lm in PROTOCOL_DIRECT_LANDMARKS and not self._landmark_unlocked_by_protocol_order(lm):
            return "#777777"
        if self._is_protocol_derived_feature(lm):
            status = self._derived_feature_status_for_landmark(lm)
            if status == "correct":
                return PROTOCOL_DERIVED_CLASSIFIED_COLOR
            if status == "incorrect":
                return PROTOCOL_DERIVED_UNCLASSIFIED_COLOR
            return "black"
        if self._is_protocol_contour_source(lm):
            return PROTOCOL_CONTOUR_PANEL_COLOR
        return "black"

    def _zoom_overlay_color_for_landmark(self, lm: str) -> str:
        if self._is_protocol_derived_feature(lm):
            status = self._derived_feature_status_for_landmark(lm)
            if status == "correct":
                return PROTOCOL_DERIVED_CLASSIFIED_COLOR
            if status == "incorrect":
                return PROTOCOL_DERIVED_UNCLASSIFIED_COLOR
            return "black"
        if self._is_protocol_contour_source(lm):
            return PROTOCOL_CONTOUR_MARKER_COLOR
        return "cyan"

    def _refresh_landmark_label_styles(self) -> None:
        for lm, widget in self.__dict__.get("landmark_radio_widgets", {}).items():
            try:
                color = self._panel_label_color_for_landmark(lm)
                widget.configure(
                    fg=color,
                    activeforeground=color,
                    disabledforeground=color,
                    font=self._label_font_for_landmark(lm, selected=False),
                )
            except Exception:
                pass
        self._refresh_landmark_categorical_count_labels()
        self._refresh_selected_landmark_group_style()

    def _landmark_is_primary_for_panel(self, lm: str) -> bool:
        return not (
            self._is_protocol_derived_feature(lm)
            or self._is_derived_acetabular_landmark(lm)
            or self._is_intermediate_acetabular_landmark(lm)
        )

    def _categorical_correct_count_for_landmark(self, lm: str) -> Tuple[int, int]:
        if self._landmark_is_primary_for_panel(lm):
            features = self._relevant_derived_features_for_landmark(lm)
        elif self._is_protocol_derived_feature(lm):
            features = [lm]
        else:
            features = []
        if not features:
            return 0, 0
        categories = self._current_categorical_annotations()
        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            derived = {}
        correct = sum(
            1
            for feature in features
            if str(derived.get(feature, "") or "").strip().lower() == "correct"
        )
        return correct, len(features)

    def _categorical_correct_text_for_landmark(self, lm: str) -> str:
        correct, total = self._categorical_correct_count_for_landmark(lm)
        return "-" if total == 0 else f"{correct}/{total}"

    def _primary_landmark_status_text(self, lm: str) -> str:
        if self._landmark_annotation_geometry_complete(lm):
            return "A"
        status = self._normalize_direct_annotation_status(self._get_landmark_status(lm))
        if status == "missing":
            return "M"
        if status == "not_applicable":
            return "N/A"
        return "-"

    def _primary_landmark_status_color(self, lm: str) -> str:
        status_text = self._primary_landmark_status_text(lm)
        if status_text == "A":
            return PROTOCOL_DERIVED_CLASSIFIED_COLOR
        if status_text == "M":
            return PROTOCOL_STATUS_MISSING_COLOR
        if status_text == "N/A":
            return PROTOCOL_STATUS_NOT_APPLICABLE_COLOR
        return "#666666"

    def _derived_feature_status_for_landmark(self, lm: str) -> str:
        categories = self._current_categorical_annotations()
        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            return ""
        value = str(derived.get(lm, "") or "").strip().lower()
        return value if value in DERIVED_FEATURE_STATUS_VALUES else ""

    def _derived_feature_status_text_for_landmark(self, lm: str) -> str:
        status = self._derived_feature_status_for_landmark(lm)
        return {"correct": "C", "incorrect": "I", "missing": "M"}.get(status, "-")

    def _status_color_for_derived_feature(self, lm: str) -> str:
        status = self._derived_feature_status_for_landmark(lm)
        if status == "correct":
            return PROTOCOL_DERIVED_CLASSIFIED_COLOR
        if status == "missing":
            return PROTOCOL_STATUS_MISSING_COLOR
        if status == "incorrect":
            return PROTOCOL_DERIVED_UNCLASSIFIED_COLOR
        return "#666666"

    def _image_categories_complete(self) -> bool:
        categories = self._current_categorical_annotations()
        hip_type = categories.get("hip_type", {})
        if not isinstance(hip_type, dict):
            return False
        left = str(hip_type.get("left", "") or "").strip().lower()
        right = str(hip_type.get("right", "") or "").strip().lower()
        return (
            bool(left)
            and bool(right)
            and left in HIP_CATEGORY_VALUES
            and right in HIP_CATEGORY_VALUES
        )

    def _landmark_panel_groups(
        self, visible_landmarks: List[str]
    ) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
        primary_landmarks = [
            lm for lm in visible_landmarks if self._landmark_is_primary_for_panel(lm)
        ]
        derived_landmarks = [
            lm for lm in visible_landmarks if not self._landmark_is_primary_for_panel(lm)
        ]
        derived_set = set(derived_landmarks)
        assigned: Set[str] = set()
        groups: List[Tuple[str, List[str]]] = []
        for primary in primary_landmarks:
            children = [
                child
                for child in self._relevant_derived_features_for_landmark(primary)
                if child in derived_set and child not in assigned
            ]
            assigned.update(children)
            groups.append((primary, children))
        orphans = [lm for lm in derived_landmarks if lm not in assigned]
        return groups, orphans

    def _refresh_landmark_categorical_count_labels(self) -> None:
        for lm, label in self.__dict__.get("landmark_correct_labels", {}).items():
            try:
                if self._is_protocol_derived_feature(lm):
                    text = self._derived_feature_status_text_for_landmark(lm)
                    fg = self._status_color_for_derived_feature(lm)
                else:
                    text = self._primary_landmark_status_text(lm)
                    fg = self._primary_landmark_status_color(lm)
                label.configure(text=text, fg=fg)
            except Exception:
                pass
        self._refresh_selected_landmark_group_style()

    def _primary_group_for_landmark(self, lm: str) -> str:
        if lm in self.__dict__.get("landmark_group_frames", {}):
            return lm
        for primary, children in self.__dict__.get("landmark_group_members", {}).items():
            if lm in children:
                return primary
        return lm

    def _set_active_landmark_group_from_selection(self) -> None:
        selected = self.selected_landmark.get().strip()
        self._active_landmark_group = (
            self._primary_group_for_landmark(selected) if selected else ""
        )

    def _handle_landmark_group_transition(self, lm: str) -> bool:
        selected = lm.strip()
        new_group = self._primary_group_for_landmark(selected) if selected else ""
        old_group = getattr(self, "_active_landmark_group", "")
        if not old_group or new_group == old_group:
            self._active_landmark_group = new_group or old_group
            return True
        if not new_group:
            self._active_landmark_group = ""
            return True
        if not self._maybe_save_before_destructive_action("switch landmarks"):
            self.selected_landmark.set(old_group)
            return False
        self._active_landmark_group = new_group
        return True

    def _landmark_group_has_status(self, primary: str) -> bool:
        if self._primary_landmark_status_text(primary) == "-":
            return False
        derived_children = [
            child
            for child in self.__dict__.get("landmark_group_members", {}).get(primary, [])[1:]
            if self._is_protocol_derived_feature(child)
        ]
        return all(
            bool(self._derived_feature_status_for_landmark(child))
            for child in derived_children
        )

    def _refresh_selected_landmark_group_style(self) -> None:
        selected = self.selected_landmark.get().strip()
        selected_primary = self._primary_group_for_landmark(selected) if selected else ""
        for primary, frame in self.__dict__.get("landmark_group_frames", {}).items():
            active = bool(selected_primary and primary == selected_primary)
            completed = self._landmark_group_has_status(primary)
            if active:
                color = "#1E88E5"
                thickness = 2
                relief = "solid"
            elif completed:
                color = PROTOCOL_DERIVED_CLASSIFIED_COLOR
                thickness = 2
                relief = "solid"
            else:
                color = "#BFCACA"
                thickness = 1
                relief = "groove"
            try:
                frame.configure(
                    highlightthickness=thickness,
                    highlightbackground=color,
                    highlightcolor=color,
                    bd=1,
                    relief=relief,
                )
            except Exception:
                pass

    def _noneditable_derived_message(self, lm: str) -> str:
        if self._is_protocol_derived_feature(lm):
            return f"{lm} is generated automatically from protocol source annotations."
        return f"{lm} is set automatically from FAC/AC geometry."

    def _is_derived_acetabular_landmark(self, lm: str) -> bool:
        if lm in {"L-A-DAB", "R-A-DAB", "L-A-SAB", "R-A-SAB"}:
            return True
        return lm in {"L-DAB", "R-DAB", "L-SAB", "R-SAB"} and self._is_protocol_derived_feature(lm)

    def _is_intermediate_acetabular_landmark(self, lm: str) -> bool:
        return lm in INTERMEDIATE_ACETABULAR_LANDMARKS

    def _landmark_is_selectable(self, lm: str) -> bool:
        if (
            self.current_image_path is not None
            and self._landmark_is_primary_for_panel(lm)
            and not self._image_categories_complete()
        ):
            return False
        if self._is_protocol_derived_feature(lm):
            return False
        if self._is_derived_acetabular_landmark(lm):
            return False
        if self._is_intermediate_acetabular_landmark(lm):
            return self._intermediate_acetabular_landmark_ready(lm)
        if lm in PROTOCOL_DIRECT_LANDMARKS:
            return self._landmark_unlocked_by_protocol_order(lm)
        return True

    def _acetabular_geometry_required_message(self, lm: str) -> str:
        return f"{lm}: place a snapped FAC/AC pair or projected hemisphere first."

    def _current_annotation_key(self) -> str | None:
        if self.current_image_path is None:
            return None
        if self.json_path is not None:
            return self._path_key(self.current_image_path)
        return str(self.current_image_path)

    def _point_from_sequence(self, value: object) -> AnnotationPoint | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        return None

    def _segmentation_mask_key(self) -> str | None:
        if self.current_image_path is None:
            return None
        if self.json_path is not None:
            return self._path_key(self.current_image_path)
        return str(self.current_image_path)

    def _annotation_landmark_keys(self, annotations: object | None = None) -> List[str]:
        keys: List[str] = []
        seen: Set[str] = set()
        for lm in self.landmarks:
            if lm not in seen:
                keys.append(lm)
                seen.add(lm)

        if isinstance(annotations, dict):
            extras = annotations.keys()
        else:
            extras = []

        for lm in extras:
            if (
                isinstance(lm, str)
                and lm not in seen
                and (
                    self._is_segmentation_landmark(lm)
                    or lm in PROTOCOL_DIRECT_LANDMARKS
                    or lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS
                    or self._is_ellipse_landmark(lm)
                    or self._is_line_landmark(lm)
                )
            ):
                keys.append(lm)
                seen.add(lm)
        return keys

    def _encode_segmentation_mask(self, mask: np.ndarray) -> Dict[str, object]:
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = mask_u8.shape[:2]
        packed = np.packbits(mask_u8.reshape(-1))
        compressed = zlib.compress(packed.tobytes())
        return {
            "type": "mask",
            "encoding": SEGMENTATION_MASK_ENCODING,
            "shape": [int(h), int(w)],
            "data": base64.b64encode(compressed).decode("ascii"),
        }

    def _decode_segmentation_mask(self, raw: object) -> np.ndarray | None:
        if not isinstance(raw, dict):
            return None
        if raw.get("type") != "mask":
            return None
        if raw.get("encoding") != SEGMENTATION_MASK_ENCODING:
            return None
        shape = raw.get("shape")
        data = raw.get("data")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            return None
        if not isinstance(data, str):
            return None
        try:
            h, w = int(shape[0]), int(shape[1])
            if h <= 0 or w <= 0:
                return None
            packed_bytes = zlib.decompress(base64.b64decode(data.encode("ascii")))
            packed = np.frombuffer(packed_bytes, dtype=np.uint8)
            bits = np.unpackbits(packed, count=h * w)
            if bits.size != h * w:
                return None
            return bits.reshape((h, w)).astype(np.uint8)
        except (ValueError, TypeError, zlib.error, binascii.Error):
            return None

    def _mask_shape_for_landmark(self, lm: str) -> Tuple[int, int] | None:
        if self._uses_manual_segmentation_grid(lm):
            return MANUAL_SEGMENTATION_MASK_SHAPE
        current_image = self.__dict__.get("current_image")
        if current_image is None:
            return None
        width, height = current_image.size
        return int(height), int(width)

    def _resize_binary_mask(
        self, mask: np.ndarray, shape: Tuple[int, int]
    ) -> np.ndarray:
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_u8.shape[:2] == shape:
            return mask_u8
        mask_img = Image.fromarray(mask_u8 * 255, mode="L").resize(
            (int(shape[1]), int(shape[0])),
            Image.Resampling.NEAREST,
        )
        return (np.array(mask_img) > 0).astype(np.uint8)

    def _normalize_segmentation_mask_for_landmark(
        self, lm: str, mask: np.ndarray
    ) -> np.ndarray:
        expected_shape = self._mask_shape_for_landmark(lm)
        if self._uses_manual_segmentation_grid(lm) and expected_shape is not None:
            return self._resize_binary_mask(mask, expected_shape)
        return (np.asarray(mask) > 0).astype(np.uint8)

    def _image_point_to_mask_point(
        self, point: AnnotationPoint, mask: np.ndarray
    ) -> AnnotationPoint:
        current_image = self.__dict__.get("current_image")
        if current_image is None:
            return point
        image_w, image_h = current_image.size
        mask_h, mask_w = mask.shape[:2]
        sx = (mask_w - 1) / max(1, image_w - 1)
        sy = (mask_h - 1) / max(1, image_h - 1)
        return float(point[0]) * sx, float(point[1]) * sy

    def _mask_point_to_image_point(
        self, point: AnnotationPoint, mask: np.ndarray
    ) -> AnnotationPoint:
        current_image = self.__dict__.get("current_image")
        if current_image is None:
            return point
        image_w, image_h = current_image.size
        mask_h, mask_w = mask.shape[:2]
        sx = (image_w - 1) / max(1, mask_w - 1)
        sy = (image_h - 1) / max(1, mask_h - 1)
        return float(point[0]) * sx, float(point[1]) * sy

    def _image_rect_to_mask_rect(
        self, rect: Tuple[float, float, float, float], mask: np.ndarray
    ) -> Tuple[float, float, float, float] | None:
        current_image = self.__dict__.get("current_image")
        if current_image is None:
            return rect
        image_w, image_h = current_image.size
        mask_h, mask_w = mask.shape[:2]
        if image_w <= 0 or image_h <= 0 or mask_w <= 0 or mask_h <= 0:
            return None
        sx = (mask_w - 1) / max(1, image_w - 1)
        sy = (mask_h - 1) / max(1, image_h - 1)
        left, top, right, bottom = rect
        return left * sx, top * sy, right * sx, bottom * sy

    def _mask_centroid(self, mask: np.ndarray | None) -> AnnotationPoint | None:
        if mask is None:
            return None
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return None
        return round(float(xs.mean()), 1), round(float(ys.mean()), 1)

    def _mask_centroid_in_image_space(
        self, mask: np.ndarray | None
    ) -> AnnotationPoint | None:
        centroid = self._mask_centroid(mask)
        if centroid is None or mask is None:
            return None
        x, y = self._mask_point_to_image_point(centroid, mask)
        return round(float(x), 1), round(float(y), 1)

    def _protocol_side_for_landmark(self, lm: str) -> str | None:
        if lm.startswith("L-"):
            return "L"
        if lm.startswith("R-"):
            return "R"
        return None

    def _side_lateral_score(self, side: str, x: float) -> float:
        # Protocol annotations are patient-side labels: annotation left is shown
        # on image right, and annotation right is shown on image left.
        if side == "L":
            return float(x)
        if side == "R":
            return -float(x)
        return float(x)

    def _manual_contour_mask_for_protocol_source(self, lm: str) -> np.ndarray | None:
        if not self._is_border_segmentation_landmark(lm):
            return None
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return None

        border_mask = (
            self.__dict__.get("it_edge_border_masks", {})
            .get(mask_key, {})
            .get(lm)
        )
        if border_mask is not None and bool(np.any(border_mask > 0)):
            return self._normalize_segmentation_mask_for_landmark(lm, border_mask)

        saved_mask = self.__dict__.get("seg_masks", {}).get(mask_key, {}).get(lm)
        if saved_mask is not None and bool(np.any(saved_mask > 0)):
            return self._normalize_segmentation_mask_for_landmark(lm, saved_mask)
        return None

    def _mask_points_in_image_space(
        self, mask: np.ndarray | None
    ) -> np.ndarray | None:
        if mask is None or self.current_image is None:
            return None
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return None
        image_w, image_h = self.current_image.size
        mask_h, mask_w = mask.shape[:2]
        sx = (image_w - 1) / max(1, mask_w - 1)
        sy = (image_h - 1) / max(1, mask_h - 1)
        return np.column_stack((xs.astype(float) * sx, ys.astype(float) * sy))

    def _select_protocol_extreme_point(
        self,
        points: np.ndarray | None,
        side: str,
        primary: str,
        tie: str,
    ) -> AnnotationPoint | None:
        if points is None or points.size == 0:
            return None

        x_values = points[:, 0]
        y_values = points[:, 1]
        lateral_scores = np.array(
            [self._side_lateral_score(side, float(x)) for x in x_values],
            dtype=float,
        )

        if primary == "superior":
            primary_values = y_values
            target = float(np.min(primary_values))
        elif primary == "inferior":
            primary_values = y_values
            target = float(np.max(primary_values))
        elif primary == "lateral":
            primary_values = lateral_scores
            target = float(np.max(primary_values))
        elif primary == "medial":
            primary_values = lateral_scores
            target = float(np.min(primary_values))
        else:
            return None

        candidates = points[np.isclose(primary_values, target)]
        if candidates.size == 0:
            return None

        candidate_scores = np.array(
            [self._side_lateral_score(side, float(x)) for x in candidates[:, 0]],
            dtype=float,
        )
        if tie == "lateral":
            idx = int(np.argmax(candidate_scores))
        elif tie == "medial":
            idx = int(np.argmin(candidate_scores))
        elif tie == "superior":
            idx = int(np.argmin(candidates[:, 1]))
        elif tie == "median_y":
            order = np.argsort(candidates[:, 1])
            idx = int(order[len(order) // 2])
        else:
            idx = 0

        point = candidates[idx]
        return round(float(point[0]), 1), round(float(point[1]), 1)

    def _mask_pixel_to_image_point(
        self, point: Tuple[int, int], mask: np.ndarray
    ) -> AnnotationPoint:
        x, y = self._mask_point_to_image_point((float(point[0]), float(point[1])), mask)
        return round(float(x), 1), round(float(y), 1)

    def _hull_candidate_pixels(self, mask: np.ndarray, limit: int = 256) -> List[Tuple[int, int]]:
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return []
        coords = np.column_stack((xs, ys)).astype(np.int32)
        if cv2 is not None and len(coords) >= 3:
            try:
                coords = cv2.convexHull(coords.reshape((-1, 1, 2))).reshape((-1, 2))
            except cv2.error:
                pass
        if len(coords) > limit:
            stride = int(math.ceil(len(coords) / float(limit)))
            coords = coords[::stride]
        return [(int(x), int(y)) for x, y in coords]

    def _bresenham_line_pixels(
        self, p0: Tuple[int, int], p1: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        xs: List[int] = []
        ys: List[int] = []
        while True:
            xs.append(x0)
            ys.append(y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)

    def _set_protocol_pod_construction(
        self, geometry: Dict[str, object] | None
    ) -> None:
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return
        store = self.__dict__.setdefault("protocol_pod_constructions", {})
        if geometry is None:
            store.pop(ann_key, None)
        else:
            store[ann_key] = geometry

    def _current_protocol_pod_construction(self) -> Dict[str, object] | None:
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return None
        geometry = self.__dict__.get("protocol_pod_constructions", {}).get(ann_key)
        return geometry if isinstance(geometry, dict) else None

    def _mask_float_point_to_image_point(
        self, point: np.ndarray, mask: np.ndarray
    ) -> AnnotationPoint:
        x, y = self._mask_point_to_image_point(
            (float(point[0]), float(point[1])), mask
        )
        return round(float(x), 1), round(float(y), 1)

    def _protocol_pod_medial_axis_samples(
        self, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> np.ndarray | None:
        left = left_mask > 0
        right = right_mask > 0
        rows = np.flatnonzero(np.any(left, axis=1) & np.any(right, axis=1))
        samples: List[Tuple[float, float]] = []

        for y in rows:
            left_x = np.flatnonzero(left[int(y)])
            right_x = np.flatnonzero(right[int(y)])
            if left_x.size == 0 or right_x.size == 0:
                continue
            left_medial_x = float(np.min(left_x))
            right_medial_x = float(np.max(right_x))
            if left_medial_x <= right_medial_x:
                continue
            samples.append(((left_medial_x + right_medial_x) / 2.0, float(y)))

        if len(samples) >= 2:
            return np.asarray(samples, dtype=float)

        left_rows = np.flatnonzero(np.any(left, axis=1))
        right_rows = np.flatnonzero(np.any(right, axis=1))
        if left_rows.size == 0 or right_rows.size == 0:
            return None

        tolerance = max(8, int(round(left_mask.shape[0] * 0.05)))
        for y in left_rows:
            nearest_idx = int(np.argmin(np.abs(right_rows - y)))
            right_y = int(right_rows[nearest_idx])
            if abs(int(y) - right_y) > tolerance:
                continue
            left_x = np.flatnonzero(left[int(y)])
            right_x = np.flatnonzero(right[right_y])
            if left_x.size == 0 or right_x.size == 0:
                continue
            left_medial_x = float(np.min(left_x))
            right_medial_x = float(np.max(right_x))
            if left_medial_x <= right_medial_x:
                continue
            samples.append(
                (
                    (left_medial_x + right_medial_x) / 2.0,
                    (float(y) + float(right_y)) / 2.0,
                )
            )

        if len(samples) < 2:
            return None
        return np.asarray(samples, dtype=float)

    def _fit_protocol_pod_medial_axis(
        self, samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float] | None:
        if samples.size == 0 or samples.shape[0] < 2:
            return None
        origin = samples.mean(axis=0)
        centered = samples - origin
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        direction = vh[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return None
        direction = direction / norm
        if direction[1] < 0:
            direction = -direction
        projections = centered @ direction
        t_min = float(np.min(projections))
        t_max = float(np.max(projections))
        if abs(t_max - t_min) < 1e-6:
            return None
        return origin, direction, t_min, t_max

    def _protocol_pod_side_sign(
        self,
        mask: np.ndarray,
        axis_point: np.ndarray,
        normal: np.ndarray,
        preferred_point: np.ndarray,
    ) -> float:
        preferred_sign = float(np.dot(preferred_point - axis_point, normal))
        if abs(preferred_sign) > 1e-6:
            return 1.0 if preferred_sign > 0 else -1.0

        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return 1.0
        points = np.column_stack((xs.astype(float), ys.astype(float)))
        mean_sign = float(np.mean((points - axis_point) @ normal))
        if abs(mean_sign) > 1e-6:
            return 1.0 if mean_sign > 0 else -1.0
        return 1.0

    def _protocol_pod_perpendicular_endpoint(
        self,
        mask: np.ndarray,
        axis_point: np.ndarray,
        axis_direction: np.ndarray,
        normal: np.ndarray,
        side_sign: float,
        fallback_point: np.ndarray,
    ) -> np.ndarray | None:
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return None

        points = np.column_stack((xs.astype(float), ys.astype(float)))
        rel = points - axis_point
        axial_delta = rel @ axis_direction
        normal_delta = rel @ normal
        side_delta = side_sign * normal_delta
        side_mask = side_delta >= -0.25
        if not bool(np.any(side_mask)):
            side_mask = np.ones(points.shape[0], dtype=bool)

        for band in (0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, float("inf")):
            candidates = np.flatnonzero(side_mask & (np.abs(axial_delta) <= band))
            if candidates.size == 0:
                continue
            candidate_side = side_delta[candidates]
            outward = float(np.max(candidate_side))
            outward_candidates = candidates[np.isclose(candidate_side, outward)]
            best = outward_candidates[
                int(np.argmin(np.abs(axial_delta[outward_candidates])))
            ]
            return points[int(best)]

        normal_offset = float(np.dot(fallback_point - axis_point, normal))
        return axis_point + normal * normal_offset

    def _derive_protocol_pod(
        self, generated: Dict[str, AnnotationValue]
    ) -> List[AnnotationPoint] | None:
        self._set_protocol_pod_construction(None)

        left = self._landmark_center_from_value("L-LPO", generated.get("L-LPO"))
        right = self._landmark_center_from_value("R-LPO", generated.get("R-LPO"))
        if left is None or right is None:
            return None

        left_mask = self._manual_contour_mask_for_protocol_source("L-C-PO")
        right_mask = self._manual_contour_mask_for_protocol_source("R-C-PO")
        if left_mask is None or right_mask is None:
            return None
        if left_mask.shape != right_mask.shape:
            right_mask = self._resize_binary_mask(right_mask, left_mask.shape)

        sample_groups: List[Tuple[str, np.ndarray]] = []
        po_samples = self._protocol_pod_medial_axis_samples(left_mask, right_mask)
        if po_samples is not None:
            sample_groups.append(("PO", po_samples))

        left_ps_mask = self._manual_contour_mask_for_protocol_source("L-C-PS")
        right_ps_mask = self._manual_contour_mask_for_protocol_source("R-C-PS")
        if left_ps_mask is not None and right_ps_mask is not None:
            if left_ps_mask.shape != left_mask.shape:
                left_ps_mask = self._resize_binary_mask(left_ps_mask, left_mask.shape)
            if right_ps_mask.shape != left_mask.shape:
                right_ps_mask = self._resize_binary_mask(right_ps_mask, left_mask.shape)
            ps_samples = self._protocol_pod_medial_axis_samples(
                left_ps_mask, right_ps_mask
            )
            if ps_samples is not None:
                sample_groups.append(("PS", ps_samples))

        if not sample_groups:
            return None
        samples = np.vstack([group_samples for _name, group_samples in sample_groups])
        axis = self._fit_protocol_pod_medial_axis(samples)
        if axis is None:
            return None

        origin, axis_direction, t_min, t_max = axis
        normal = np.array([-axis_direction[1], axis_direction[0]], dtype=float)

        left_mask_point = np.asarray(
            self._image_point_to_mask_point(left, left_mask), dtype=float
        )
        right_mask_point = np.asarray(
            self._image_point_to_mask_point(right, left_mask), dtype=float
        )
        left_t = float(np.dot(left_mask_point - origin, axis_direction))
        right_t = float(np.dot(right_mask_point - origin, axis_direction))
        mid_t = (left_t + right_t) / 2.0
        mid_t = max(t_min, min(t_max, mid_t))
        axis_mid = origin + axis_direction * mid_t

        left_axis_point = origin + axis_direction * left_t
        right_axis_point = origin + axis_direction * right_t
        left_sign = self._protocol_pod_side_sign(
            left_mask, axis_mid, normal, left_mask_point
        )
        right_sign = self._protocol_pod_side_sign(
            right_mask, axis_mid, normal, right_mask_point
        )
        if left_sign == right_sign:
            right_sign = -left_sign

        left_endpoint = self._protocol_pod_perpendicular_endpoint(
            left_mask,
            axis_mid,
            axis_direction,
            normal,
            left_sign,
            left_mask_point,
        )
        right_endpoint = self._protocol_pod_perpendicular_endpoint(
            right_mask,
            axis_mid,
            axis_direction,
            normal,
            right_sign,
            right_mask_point,
        )
        if left_endpoint is None or right_endpoint is None:
            return None

        pod = [
            self._mask_float_point_to_image_point(left_endpoint, left_mask),
            self._mask_float_point_to_image_point(right_endpoint, left_mask),
        ]
        self._set_protocol_pod_construction(
            {
                "axis": [
                    self._mask_float_point_to_image_point(
                        origin + axis_direction * t_min, left_mask
                    ),
                    self._mask_float_point_to_image_point(
                        origin + axis_direction * t_max, left_mask
                    ),
                ],
                "left_lpo": left,
                "right_lpo": right,
                "left_axis_point": self._mask_float_point_to_image_point(
                    left_axis_point, left_mask
                ),
                "right_axis_point": self._mask_float_point_to_image_point(
                    right_axis_point, left_mask
                ),
                "midpoint": self._mask_float_point_to_image_point(
                    axis_mid, left_mask
                ),
                "pod": pod,
                "axis_sample_sources": [name for name, _samples in sample_groups],
            }
        )
        return pod

    def _derive_protocol_femoral_axis(self, side: str) -> List[AnnotationPoint] | None:
        dmf_mask = self._manual_contour_mask_for_protocol_source(f"{side}-C-DMF")
        gt_mask = self._manual_contour_mask_for_protocol_source(f"{side}-C-GT")
        if dmf_mask is None or gt_mask is None:
            return None
        if dmf_mask.shape != gt_mask.shape:
            gt_mask = self._resize_binary_mask(gt_mask, dmf_mask.shape)

        dmf_y, dmf_x = np.nonzero(dmf_mask > 0)
        gt_y, gt_x = np.nonzero(gt_mask > 0)
        if dmf_x.size < 2 or gt_x.size < 2:
            return None

        y_min = int(np.min(dmf_y))
        y_max = int(np.max(dmf_y))
        if y_max <= y_min:
            return None
        y_cut = int(round(y_min + (y_max - y_min) / 3.0))
        overlap_min = max(y_min, int(np.min(gt_y)))
        overlap_max = min(y_cut, int(np.max(gt_y)))
        if overlap_max <= overlap_min:
            return None

        centers: List[Tuple[float, float]] = []
        for y in range(overlap_min, overlap_max + 1):
            dmf_idx = np.abs(dmf_y - y) <= 2
            gt_idx = np.abs(gt_y - y) <= 2
            if not bool(np.any(dmf_idx)) or not bool(np.any(gt_idx)):
                continue
            dmf_candidates = dmf_x[dmf_idx].astype(float)
            gt_candidates = gt_x[gt_idx].astype(float)
            dmf_scores = np.array(
                [self._side_lateral_score(side, float(x)) for x in dmf_candidates]
            )
            gt_scores = np.array(
                [self._side_lateral_score(side, float(x)) for x in gt_candidates]
            )
            medial_x = float(dmf_candidates[int(np.argmin(dmf_scores))])
            lateral_x = float(gt_candidates[int(np.argmax(gt_scores))])
            centers.append(((medial_x + lateral_x) / 2.0, float(y)))

        if len(centers) < 2:
            return None

        centers_arr = np.asarray(centers, dtype=float)
        mean = centers_arr.mean(axis=0)
        centered = centers_arr - mean
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        except np.linalg.LinAlgError:
            return None
        if float(np.linalg.norm(direction)) < 1e-12:
            return None
        projections = centered @ direction
        p0 = mean + direction * float(np.min(projections))
        p1 = mean + direction * float(np.max(projections))
        return [
            self._mask_pixel_to_image_point((int(round(p0[0])), int(round(p0[1]))), dmf_mask),
            self._mask_pixel_to_image_point((int(round(p1[0])), int(round(p1[1]))), dmf_mask),
        ]

    def _set_protocol_derived_value(
        self,
        pts: Dict[str, AnnotationValue],
        lm: str,
        value: AnnotationValue | None,
    ) -> bool:
        if value is None:
            if lm in pts:
                del pts[lm]
                found_var = self.landmark_found.get(lm)
                if found_var is not None:
                    found_var.set(False)
                return True
            return False

        if pts.get(lm) != value:
            pts[lm] = value
            changed = True
        else:
            changed = False
        found_var = self.landmark_found.get(lm)
        if found_var is not None:
            found_var.set(True)
        return changed

    def _update_protocol_derived_features_for_current_image(
        self, *, mark_dirty: bool = False
    ) -> bool:
        if getattr(self, "multi_review_mode", False):
            return False
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return False
        if not any(lm in set(self.landmarks) for lm in PROTOCOL_CONTOUR_LANDMARKS):
            return False

        pts = self.annotations.setdefault(ann_key, {})
        allowed = self._get_allowed_landmarks_for_current_view()

        def feature_allowed(lm: str) -> bool:
            return self._is_protocol_derived_feature(lm) and (not allowed or lm in allowed)

        generated: Dict[str, AnnotationValue] = {}
        for side in ("L", "R"):
            for source_lm, features in PROTOCOL_DERIVED_FEATURES_BY_SOURCE.items():
                if not source_lm.startswith(f"{side}-"):
                    continue
                if not all(feature_allowed(feature) for feature in features):
                    continue
                mask = self._manual_contour_mask_for_protocol_source(source_lm)
                points = self._mask_points_in_image_space(mask)

                if source_lm.endswith("-C-ILI"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "superior", "lateral"
                    )
                    generated[features[1]] = self._select_protocol_extreme_point(
                        points, side, "lateral", "superior"
                    )
                elif source_lm.endswith("-C-PO"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "lateral", "median_y"
                    )
                elif source_lm.endswith("-C-IT"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "inferior", "lateral"
                    )
                elif source_lm.endswith("-C-PS"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "superior", "medial"
                    )
                    generated[features[1]] = self._select_protocol_extreme_point(
                        points, side, "inferior", "medial"
                    )
                elif source_lm.endswith("-C-GT"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "superior", "lateral"
                    )
                    generated[features[1]] = self._select_protocol_extreme_point(
                        points, side, "lateral", "superior"
                    )
                elif source_lm.endswith("-C-LT"):
                    generated[features[0]] = self._select_protocol_extreme_point(
                        points, side, "superior", "lateral"
                    )
                    generated[features[1]] = self._select_protocol_extreme_point(
                        points, side, "medial", "superior"
                    )
                    generated[features[2]] = self._select_protocol_extreme_point(
                        points, side, "inferior", "lateral"
                    )

            ac_source = f"{side}-C-AC"
            ac_feature = f"{side}-AC"
            if feature_allowed(ac_feature):
                center = self._landmark_center_from_value(ac_source, pts.get(ac_source))
                if center is not None:
                    generated[ac_feature] = (
                        round(float(center[0]), 1),
                        round(float(center[1]), 1),
                    )

            fhc_source = f"{side}-C-FHC"
            fhc_feature = f"{side}-FHC"
            if feature_allowed(fhc_feature):
                center = self._landmark_center_from_value(fhc_source, pts.get(fhc_source))
                if center is not None:
                    generated[fhc_feature] = (
                        round(float(center[0]), 1),
                        round(float(center[1]), 1),
                    )

            fac_source = f"{side}-C-FAC"
            dab_feature = f"{side}-DAB"
            sab_feature = f"{side}-SAB"
            if feature_allowed(dab_feature) and feature_allowed(sab_feature):
                axes = self._ellipse_axis_points(self._get_ellipse(fac_source))
                if axes is not None and self._ellipse_has_acetabular_endpoint_geometry(
                    fac_source, axes
                ):
                    _center, major0, major1, _minor0, _minor1 = axes
                    endpoints = [major0, major1]
                    lateral = max(
                        endpoints,
                        key=lambda point: self._side_lateral_score(side, point[0]),
                    )
                    medial = min(
                        endpoints,
                        key=lambda point: self._side_lateral_score(side, point[0]),
                    )
                    generated[dab_feature] = (
                        round(float(medial[0]), 1),
                        round(float(medial[1]), 1),
                    )
                    generated[sab_feature] = (
                        round(float(lateral[0]), 1),
                        round(float(lateral[1]), 1),
                    )

            fa_feature = f"{side}-D-FA"
            if feature_allowed(fa_feature):
                line = self._derive_protocol_femoral_axis(side)
                if line is not None:
                    generated[fa_feature] = line

        if feature_allowed("POD"):
            pod = self._derive_protocol_pod(generated)
            if pod is not None:
                generated["POD"] = pod

        changed = False
        for lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS:
            if not feature_allowed(lm):
                continue
            if self._set_protocol_derived_value(pts, lm, generated.get(lm)):
                changed = True

        if changed and mark_dirty:
            self.dirty = True
        return changed

    def _settings_from_segmentation_value(self, value: object) -> Dict | None:
        if not isinstance(value, (list, tuple)) or len(value) < 8:
            return None
        try:
            method_code = str(value[2])
            return {
                "method": "Flood Fill"
                if method_code in ("FF", "Flood Fill")
                else "Adaptive CC",
                "sens": int(value[3]),
                "edge_lock": int(value[4]),
                "edge_width": int(value[5]),
                "clahe": int(value[6]),
                "grow": int(value[7]),
            }
        except (TypeError, ValueError):
            return None

    def _normalize_ellipse_value(self, value: object) -> EllipseAnnotation | None:
        center: AnnotationPoint | None = None
        major: List[AnnotationPoint] = []
        minor: List[AnnotationPoint] = []
        construction: str | None = None

        if isinstance(value, dict):
            center = self._point_from_sequence(value.get("center"))
            raw_construction = value.get("construction")
            if isinstance(raw_construction, str) and raw_construction in FAC_CONSTRUCTION_MODES:
                construction = raw_construction

            raw_major = value.get("major")
            if isinstance(raw_major, (list, tuple)):
                for point in raw_major[:2]:
                    parsed = self._point_from_sequence(point)
                    if parsed is not None:
                        major.append(parsed)

            raw_minor = value.get("minor")
            if isinstance(raw_minor, (list, tuple)):
                for point in raw_minor[:2]:
                    parsed = self._point_from_sequence(point)
                    if parsed is not None:
                        minor.append(parsed)

        elif isinstance(value, (list, tuple)) and len(value) >= 4:
            parsed_points = [self._point_from_sequence(point) for point in value[:4]]
            if all(point is not None for point in parsed_points):
                major = [parsed_points[0], parsed_points[1]]  # type: ignore[list-item]
                minor = [parsed_points[2], parsed_points[3]]  # type: ignore[list-item]

        if len(major) != 2 or len(minor) != 2:
            return None

        if center is None:
            center = (
                (major[0][0] + major[1][0]) / 2.0,
                (major[0][1] + major[1][1]) / 2.0,
            )

        normalized: EllipseAnnotation = {
            "type": "ellipse",
            "center": [float(center[0]), float(center[1])],
            "major": [
                [float(major[0][0]), float(major[0][1])],
                [float(major[1][0]), float(major[1][1])],
            ],
            "minor": [
                [float(minor[0][0]), float(minor[0][1])],
                [float(minor[1][0]), float(minor[1][1])],
            ],
        }
        if construction is not None:
            normalized["construction"] = construction
        return normalized

    def _ellipse_center_from_value(self, value: object) -> AnnotationPoint | None:
        ellipse = self._normalize_ellipse_value(value)
        if ellipse is None:
            return None
        return self._point_from_sequence(ellipse.get("center"))

    def _landmark_center_from_value(
        self, lm: str, value: AnnotationValue
    ) -> AnnotationPoint | None:
        if self._is_ellipse_landmark(lm):
            return self._ellipse_center_from_value(value)

        if self._is_line_landmark(lm):
            if not isinstance(value, list) or not value:
                return None
            pts: List[AnnotationPoint] = []
            for point in value[:2]:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        pts.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        pass
            if not pts:
                return None
            return (
                sum(x for x, _y in pts) / len(pts),
                sum(y for _x, y in pts) / len(pts),
            )

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return (float(value[0]), float(value[1]))
        return None

    def _set_all_multi_reviewers_visible(self, visible: bool) -> None:
        for var in self.multi_review_visibility.values():
            var.set(visible)
        self._on_multi_review_visibility_changed()

    def _delete_selected_multi_review_temp_annotation(self) -> bool:
        if not getattr(self, "multi_review_mode", False):
            return False

        image_key = self._current_multi_review_image_key()
        lm = self.selected_landmark.get().strip()
        if image_key is None or not lm:
            return False

        temp_pts = self.multi_review_temp_annotations.get(image_key, {})
        if lm not in temp_pts:
            return False

        del temp_pts[lm]
        if not temp_pts:
            self.multi_review_temp_annotations.pop(image_key, None)

        self._clear_line_preview()
        self._refresh_multi_review_panel()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        return True

    def _on_multi_review_visibility_changed(self) -> None:
        self._refresh_multi_review_panel()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()

    def _refresh_multi_review_panel(self) -> None:
        frame = getattr(self, "multi_review_reviewers_frame", None)
        if frame is None:
            return

        for widget in frame.winfo_children():
            widget.destroy()
        self.multi_review_visibility_widgets = {}

        if not getattr(self, "multi_review_mode", False):
            self.multi_review_summary_var.set("No multi-reviewer data loaded.")
            return

        image_key = self._current_multi_review_image_key()
        annotated_count = 0
        total = len(self.multi_review_reviewer_order)

        for reviewer in self.multi_review_reviewer_order:
            var = self.multi_review_visibility.setdefault(
                reviewer, tk.BooleanVar(value=True)
            )
            has_content = (
                self._record_has_multi_review_content(image_key, reviewer)
                if image_key is not None
                else False
            )
            if has_content:
                annotated_count += 1
            color = self._multi_review_color(reviewer)
            status = "annotated" if has_content else "empty"
            cb = tk.Checkbutton(
                frame,
                text=f"{reviewer}: {status}",
                variable=var,
                command=self._on_multi_review_visibility_changed,
                font=self.dialogue_font,
                fg=color if has_content else "grey45",
                activeforeground=color,
                selectcolor=self.cget("bg"),
                anchor="w",
            )
            cb.pack(anchor="w", fill="x")
            self.multi_review_visibility_widgets[reviewer] = cb

        button_row = tk.Frame(frame)
        button_row.pack(fill="x", pady=(4, 0))
        tk.Button(
            button_row,
            text="View All",
            command=lambda: self._set_all_multi_reviewers_visible(True),
            font=self.dialogue_font,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(
            button_row,
            text="View None",
            command=lambda: self._set_all_multi_reviewers_visible(False),
            font=self.dialogue_font,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        selected_lm = self.selected_landmark.get().strip()
        has_selected_temp = (
            image_key is not None
            and bool(selected_lm)
            and selected_lm in self.multi_review_temp_annotations.get(image_key, {})
        )
        tk.Button(
            button_row,
            text="Delete Temp",
            command=self._delete_selected_multi_review_temp_annotation,
            font=self.dialogue_font,
            state="normal" if has_selected_temp else "disabled",
        ).pack(side="left", expand=True, fill="x")

        self.multi_review_summary_var.set(
            f"Current image: {annotated_count}/{total} reviewers annotated"
        )

    def _on_image_direction_changed(self, _event=None) -> None:
        if getattr(self, "multi_review_mode", False):
            return
        self.current_image_direction = self.image_direction_var.get()
        record = self._get_current_image_record()
        if record is not None:
            record["image_direction"] = self.current_image_direction
        self._maybe_autosave_current_image()
        self.focus_set()

    def _save_json_file(self, show_success: bool = False) -> bool:
        if self.json_path is None:
            messagebox.showwarning("Save", "No JSON data file loaded.")
            return False

        try:
            self._sync_current_state_to_json_record()

            images_to_save: List[Dict] = []

            # Build metadata for traceability
            now_iso = datetime.now().isoformat()
            metadata = dict(self._json_metadata)
            if "created" not in metadata:
                metadata["created"] = now_iso
            metadata["last_modified"] = now_iso
            raw_mask_landmarks = metadata.get("mask_landmarks", [])
            mask_landmarks = (
                [lm for lm in raw_mask_landmarks if isinstance(lm, str)]
                if isinstance(raw_mask_landmarks, list)
                else []
            )
            for lm in self.json_data.get("landmarks", []):
                if isinstance(lm, str) and self._is_segmentation_landmark(lm):
                    if lm not in mask_landmarks:
                        mask_landmarks.append(lm)
            for record in self.json_data.get("images", []):
                if not isinstance(record, dict):
                    continue
                annotations = record.get("annotations", {})
                if not isinstance(annotations, dict):
                    continue
                for lm in annotations.keys():
                    if isinstance(lm, str) and self._is_segmentation_landmark(lm):
                        if lm not in mask_landmarks:
                            mask_landmarks.append(lm)
            if mask_landmarks:
                metadata["mask_landmarks"] = mask_landmarks
                metadata["mask_encoding"] = SEGMENTATION_MASK_ENCODING

            save_data = {
                "landmarks": list(self.json_data.get("landmarks", [])),
                "views": dict(self.allowed_views),
                "images": images_to_save,
                "metadata": metadata,
            }

            for record in self.json_data.get("images", []):
                if not isinstance(record, dict):
                    continue
                clean = dict(record)
                clean.pop("resolved_image_path", None)
                images_to_save.append(clean)

            tmp_path = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(save_data, handle, indent=2)

            # Capture original content BEFORE atomic replace for backup
            original_content = None
            if (
                self._original_json_hash is not None
                and not self._original_json_backed_up
                and self.json_path is not None
                and self.json_path.exists()
            ):
                import hashlib

                new_hash = hashlib.sha256(
                    json.dumps(save_data, indent=2, sort_keys=False).encode("utf-8")
                ).hexdigest()
                if new_hash != self._original_json_hash:
                    original_content = self.json_path.read_bytes()

            tmp_path.replace(self.json_path)

            # Update in-memory metadata after successful save
            self._json_metadata = metadata

            # Backup original JSON to OneDrive (using captured content)
            if original_content is not None and self._onedrive_enabled():
                self._backup_original_json_content(original_content)
                self._original_json_backed_up = True

            self._refresh_saved_snapshot_for_current_image()
            self.dirty = False
            self._refresh_image_listbox()
            if self._onedrive_enabled():
                self._schedule_onedrive_backup()

            if show_success:
                messagebox.showinfo("Saved", "Annotations saved to JSON.")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save JSON:\n{e}")
            return False

    def _parse_annotations_for_record(self, record: Dict) -> Dict[str, AnnotationValue]:
        pts: Dict[str, AnnotationValue] = {}
        annotations = record.get("annotations", {}) or {}
        per_img_settings: Dict[str, Dict] = {}
        per_img_meta: Dict[str, Dict[str, Union[bool, str]]] = {}
        per_img_radius: Dict[str, int] = {}
        per_img_masks: Dict[str, np.ndarray] = {}
        per_img_border_masks: Dict[str, np.ndarray] = {}

        for lm in self._annotation_landmark_keys(annotations):
            raw = annotations.get(lm)
            if raw is None:
                continue

            if isinstance(raw, dict):
                if self._is_ellipse_landmark(lm) and raw.get("type") == "ellipse":
                    val = raw
                    per_img_meta[lm] = {
                        "flag": bool(raw.get("flag", False)),
                        "note": str(raw.get("note", "")),
                    }
                else:
                    val = raw.get("value")
                    per_img_meta[lm] = {
                        "flag": bool(raw.get("flag", False)),
                        "note": str(raw.get("note", "")),
                    }
                status = self._normalize_direct_annotation_status(raw.get("status", ""))
                if status:
                    per_img_meta[lm]["status"] = status
                if lm in self.HOVER_CIRCLE_LANDMARKS and "radius" in raw:
                    try:
                        per_img_radius[lm] = int(raw["radius"])
                    except (TypeError, ValueError):
                        pass
                if self._is_segmentation_landmark(lm):
                    decoded_mask = self._decode_segmentation_mask(
                        raw.get("segmentation")
                    )
                    if decoded_mask is not None:
                        per_img_masks[lm] = (
                            self._normalize_segmentation_mask_for_landmark(
                                lm, decoded_mask
                            )
                        )
                if self._is_border_segmentation_landmark(lm):
                    decoded_border = self._decode_segmentation_mask(
                        raw.get("edge_border")
                    )
                    if decoded_border is not None:
                        per_img_border_masks[lm] = (
                            self._normalize_segmentation_mask_for_landmark(
                                lm, decoded_border
                            )
                        )
            else:
                val = raw
                per_img_meta[lm] = {"flag": False, "note": ""}

            if val is None and self._is_border_segmentation_landmark(lm):
                centroid_mask = per_img_masks.get(lm)
                if centroid_mask is None:
                    centroid_mask = per_img_border_masks.get(lm)
                centroid = self._mask_centroid_in_image_space(centroid_mask)
                if centroid is not None:
                    val = [centroid[0], centroid[1]]

            if val is None:
                continue

            if self._is_ellipse_landmark(lm):
                ellipse = self._normalize_ellipse_value(val)
                if ellipse is not None:
                    pts[lm] = ellipse
                continue

            if self._is_line_landmark(lm):
                if isinstance(val, list):
                    line_pts: List[Tuple[float, float]] = []
                    for point in val:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            try:
                                line_pts.append((float(point[0]), float(point[1])))
                            except (TypeError, ValueError):
                                continue
                    if line_pts:
                        pts[lm] = line_pts[:2]
                continue

            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    pts[lm] = (float(val[0]), float(val[1]))
                except (TypeError, ValueError):
                    continue

                if self._is_auto_segmentation_landmark(lm):
                    settings = self._settings_from_segmentation_value(val)
                    if settings is not None:
                        per_img_settings[lm] = settings

        if self.current_image_path is not None:
            key = self._path_key(self.current_image_path)
            self.lm_settings[key] = per_img_settings
            self.landmark_meta[key] = per_img_meta
            getattr(self, "hover_radii", {})[key] = per_img_radius
            seg_masks = getattr(self, "seg_masks", None)
            if isinstance(seg_masks, dict):
                if per_img_masks:
                    seg_masks[key] = per_img_masks
                else:
                    seg_masks.pop(key, None)
            border_masks = getattr(self, "it_edge_border_masks", None)
            if isinstance(border_masks, dict):
                if per_img_border_masks:
                    border_masks[key] = per_img_border_masks
                else:
                    border_masks.pop(key, None)

        return pts

    def _show_filtered_file_dialog(
        self, initial_dir: Path, round_num: str
    ) -> Optional[Path]:
        dialog = tk.Toplevel(self)
        dialog.title(f"Select Data File - {round_num}")
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=f"Showing files for {round_num}").pack(anchor="w")

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=self.dialogue_font,
            selectmode="single",
            height=15,
            width=65,  # chars
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        json_files = sorted(initial_dir.glob("*.json"))
        round_tag = f"_round{round_num[-1]}_"
        json_files = []
        for root, dirs, files in initial_dir.walk():
            json_files.extend(
                (root / f).relative_to(initial_dir)
                for f in files
                if round_tag in f.lower() and f.endswith(".json")
            )
        json_files.sort()

        for f in json_files:
            listbox.insert(tk.END, str(f))

        selected = [None]

        def on_select():
            if listbox.curselection():
                idx = listbox.curselection()[0]
                filename = listbox.get(idx)
                selected[0] = initial_dir / filename
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(btn_frame, text="Load", command=on_select).pack(
            side="right", padx=(5, 0)
        )
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")

        listbox.bind("<Double-1>", lambda e: on_select())

        dialog.update_idletasks()
        width = 600  # or whatever
        height = dialog.winfo_height()
        parent_x = self.winfo_x() + (self.winfo_width() // 2)
        parent_y = self.winfo_y() + (self.winfo_height() // 2)
        dialog_x = parent_x - (width // 2)
        dialog_y = parent_y - (height // 2)
        # dialog.geometry(f"{width}x{height}+{dialog_x}+{dialog_y}")

        self.wait_window(dialog)
        return selected[0]

    def _default_multi_review_dir(self) -> Path:
        base = BASE_DIR / "data"
        candidates = sorted(base.glob("round2_multi_reviewer_review*"))
        if candidates:
            return candidates[-1]
        data_dir = get_data_dir()
        return data_dir if data_dir.is_dir() else BASE_DIR

    def _normalize_multi_review_image_key(self, raw_path: Union[str, Path]) -> str:
        normalized = str(raw_path).replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _unique_reviewer_name(self, base_name: str, used: Set[str]) -> str:
        reviewer = base_name.strip() or "reviewer"
        reviewer = reviewer.lower()
        if reviewer not in used:
            used.add(reviewer)
            return reviewer

        i = 2
        while f"{reviewer}_{i}" in used:
            i += 1
        unique = f"{reviewer}_{i}"
        used.add(unique)
        return unique

    def _reviewer_name_from_json_path(self, json_path: Path, used: Set[str]) -> str:
        stem = json_path.stem
        if "_" in stem:
            stem = stem.rsplit("_", 1)[-1]
        return self._unique_reviewer_name(stem, used)

    def _resolve_multi_review_image_path(self, raw_path: str, json_dir: Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()

        candidates: List[Path] = []
        candidates.append((json_dir / path).resolve())

        parts = path.parts
        if len(parts) > 1 and parts[0] == json_dir.name:
            candidates.append((json_dir / Path(*parts[1:])).resolve())

        candidates.append((get_data_dir() / path).resolve())
        candidates.append((BASE_DIR / "data" / path).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _parse_multi_review_record_annotations(
        self, record: Dict[str, Any]
    ) -> Tuple[Dict[str, AnnotationValue], Dict[str, Dict[str, Union[bool, str]]]]:
        annotations = record.get("annotations", {}) or {}
        pts: Dict[str, AnnotationValue] = {}
        meta: Dict[str, Dict[str, Union[bool, str]]] = {}

        if not isinstance(annotations, dict):
            return pts, meta

        for lm in self.landmarks:
            raw = annotations.get(lm)
            if raw is None:
                continue

            if isinstance(raw, dict):
                if self._is_ellipse_landmark(lm) and raw.get("type") == "ellipse":
                    val = raw
                    meta[lm] = {
                        "flag": bool(raw.get("flag", False)),
                        "note": str(raw.get("note", "")),
                    }
                else:
                    val = raw.get("value")
                    meta[lm] = {
                        "flag": bool(raw.get("flag", False)),
                        "note": str(raw.get("note", "")),
                    }
                status = self._normalize_direct_annotation_status(raw.get("status", ""))
                if status:
                    meta[lm]["status"] = status
            else:
                val = raw
                meta[lm] = {"flag": False, "note": ""}

            if val is None:
                continue

            if self._is_ellipse_landmark(lm):
                ellipse = self._normalize_ellipse_value(val)
                if ellipse is not None:
                    pts[lm] = ellipse
                continue

            if self._is_line_landmark(lm):
                if isinstance(val, list):
                    line_pts: List[Tuple[float, float]] = []
                    for point in val:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            try:
                                line_pts.append((float(point[0]), float(point[1])))
                            except (TypeError, ValueError):
                                continue
                    if line_pts:
                        pts[lm] = line_pts[:2]
                continue

            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    pts[lm] = (float(val[0]), float(val[1]))
                except (TypeError, ValueError):
                    continue

        return pts, meta

    def _record_has_multi_review_content(self, image_key: str, reviewer: str) -> bool:
        pts = self.multi_review_annotations.get(image_key, {}).get(reviewer, {})
        if pts:
            return True

        meta = self.multi_review_meta.get(image_key, {}).get(reviewer, {})
        return any(
            bool(entry.get("flag", False))
            or bool(str(entry.get("note", "")).strip())
            or bool(self._normalize_direct_annotation_status(entry.get("status", "")))
            for entry in meta.values()
        )

    def _multi_review_counts_for_image_path(self, image_path: Path) -> Tuple[int, int]:
        path_key = self._path_key(image_path)
        image_key = self.multi_review_image_key_by_path.get(path_key)
        if not image_key:
            return 0, len(self.multi_review_reviewer_order)

        total = len(self.multi_review_reviewer_order)
        done = sum(
            1
            for reviewer in self.multi_review_reviewer_order
            if self._record_has_multi_review_content(image_key, reviewer)
        )
        return done, total

    def _clear_multi_review_state(self, clear_loaded: bool = False) -> None:
        self.multi_review_mode = False
        self.multi_review_reviewer_order = []
        self.multi_review_records = {}
        self.multi_review_annotations = {}
        self.multi_review_meta = {}
        self.multi_review_image_key_by_path = {}
        self.multi_review_temp_annotations = {}
        self.multi_review_visibility = {}
        self.multi_review_visibility_widgets = {}
        self.multi_review_status_var.set("")
        self.multi_review_summary_var.set("")
        self._set_multi_review_controls_state()
        self._refresh_multi_review_panel()

        if not clear_loaded:
            return

        self.json_path = None
        self.json_dir = None
        self.json_data = {"landmarks": [], "views": {}, "images": []}
        self.allowed_views = {}
        self.landmarks = []
        self.images = []
        self.image_index_map = {}
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = None
        self.absolute_current_image_path = None
        self.current_image_flag = False
        self.image_flag_var.set(False)
        self.current_view_var.set("")
        self.path_var.set("No image loaded")
        self.quality_var.set("N/A")
        self.annotations.clear()
        self.categorical_annotations.clear()
        self.landmark_meta.clear()
        self.hover_radii.clear()
        self.lm_settings.clear()
        self.seg_masks.clear()
        self.it_edge_border_masks.clear()
        self.last_seed.clear()
        self.pelvis_segmentation_undo_stack.clear()
        self.pelvis_segmentation_status_var.set("")
        self.pelvis_segmentation_running = False
        self.canvas.delete("all")
        self._render_black_zoom_view()
        self._build_landmark_panel()
        self._refresh_image_listbox()
        self._refresh_pelvis_segmentation_buttons()
        self._refresh_categorical_controls()

    def load_multi_reviewer_data(self) -> None:
        return

    def load_data(self) -> None:
        if not self._maybe_save_before_destructive_action("load another data file"):
            return

        data_dir = get_data_dir()
        initial = data_dir if data_dir.is_dir() else BASE_DIR

        json_file = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[("JSON File", "*.json")],
            title="Load annotation data JSON",
        )
        if not json_file:
            return
        json_path = Path(json_file)

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                raw_content = handle.read()
                data = json.loads(raw_content)
        except Exception as e:
            messagebox.showerror("Load Data", f"Failed to read JSON:\n{e}")
            return

        # Compute hash of original file for backup dedup
        import hashlib

        self._original_json_hash = hashlib.sha256(
            raw_content.encode("utf-8")
        ).hexdigest()
        self._original_json_backed_up = False

        landmarks = data.get("landmarks")
        views = data.get("views")
        images = data.get("images")

        if not isinstance(landmarks, list) or not all(
            isinstance(name, str) and name.strip() for name in landmarks
        ):
            messagebox.showerror(
                "Load Data", 'JSON must contain a "landmarks" list of names.'
            )
            return

        if not isinstance(views, dict) or not views:
            messagebox.showerror(
                "Load Data", 'JSON must contain a non-empty "views" mapping.'
            )
            return

        if not isinstance(images, list):
            messagebox.showerror("Load Data", 'JSON must contain an "images" list.')
            return

        self._clear_multi_review_state(clear_loaded=False)

        allowed_views: Dict[str, List[str]] = {}
        for view_name, landmark_list in views.items():
            if not isinstance(view_name, str) or not view_name.strip():
                messagebox.showerror(
                    "Load Data", "All view names must be non-empty strings."
                )
                return
            if not isinstance(landmark_list, list) or not all(
                isinstance(name, str) for name in landmark_list
            ):
                messagebox.showerror(
                    "Load Data",
                    f'View "{view_name}" must map to a list of landmark names.',
                )
                return
            allowed_views[view_name] = list(landmark_list)

        landmarks = self._inject_ellipse_landmarks(list(landmarks), allowed_views)

        self.json_path = json_path
        self.json_dir = json_path.parent
        self.allowed_views = allowed_views
        self._json_metadata = data.get("metadata", {})  # Preserve existing metadata

        # Auto-merge: check if repo has a fresher version of this JSON
        try:
            from auto_merge import auto_merge_on_load

            data = auto_merge_on_load(json_path)
            self._json_metadata = data.get("metadata", {})
            landmarks = data.get("landmarks", landmarks)
            views = data.get("views", views)
            images = data.get("images", images)
            self.allowed_views = {}
            for view_name, landmark_list in views.items():
                if isinstance(view_name, str) and isinstance(landmark_list, list):
                    self.allowed_views[view_name] = list(landmark_list)
            landmarks = self._inject_ellipse_landmarks(
                list(landmarks), self.allowed_views
            )
        except Exception as e:
            logger.warning(f"Failed to auto-merge: {e}")

        self.json_data = {
            "landmarks": list(landmarks),
            "views": dict(self.allowed_views),
            "images": [],
        }
        self.landmarks = list(landmarks)
        self._check_landmark_display_order()
        self.images = []
        self.image_index_map = {}
        self.current_image_index = -1
        self.saved_image_snapshots.clear()
        self.annotations.clear()
        self.categorical_annotations.clear()
        self.lm_settings.clear()
        self.landmark_meta.clear()
        self.seg_masks.clear()
        self.it_edge_border_masks.clear()
        self.last_seed.clear()
        self.pelvis_segmentation_undo_stack.clear()
        self.pelvis_segmentation_status_var.set("")
        self.pelvis_segmentation_running = False
        self._refresh_pelvis_segmentation_buttons()
        self._refresh_categorical_controls()
        self.queue_mode = False
        self.check_csv_mode = False
        try:
            self.hover_radii.clear()
        except AttributeError:
            pass
        self.current_view_var.set("")

        if self.view_dropdown is not None:
            self.view_dropdown["values"] = list(self.allowed_views.keys())

        missing: List[str] = []
        for idx, raw_record in enumerate(images):
            if not isinstance(raw_record, dict):
                missing.append(f"images[{idx}] is not an object")
                continue
            if "image_path" not in raw_record:
                missing.append(f"images[{idx}] is missing image_path")
                continue

            resolved = self._resolve_image_path(str(raw_record["image_path"]))
            record = dict(raw_record)
            record.setdefault("image_flag", False)
            record.setdefault("view", None)
            record.setdefault("annotations", {})
            record["resolved_image_path"] = str(resolved)
            self.json_data["images"].append(record)

            if not resolved.exists():
                missing.append(str(resolved))
                continue

            key = self._path_key(resolved)
            self.images.append(resolved)
            self.image_index_map[key] = len(self.json_data["images"]) - 1

        self._build_landmark_panel()

        if missing:
            preview = "\n".join(missing[:15])
            more = "" if len(missing) <= 15 else f"\n... and {len(missing) - 15} more"
            messagebox.showwarning(
                "Missing image paths",
                "Some image paths from the JSON could not be found:\n\n"
                f"{preview}{more}",
            )

        if not self.images:
            self.current_image = None
            self.current_image_path = None
            self.absolute_current_image_path = None
            self.current_image_index = -1
            self.current_image_flag = False
            self.path_var.set("No valid images found in JSON")
            self.quality_var.set("0")
            self.current_view_var.set("")
            self.canvas.delete("all")
            self._render_black_zoom_view()
            self.dirty = False
            self._refresh_image_listbox()
            return

        self.current_image_index = 0
        self.load_image_from_path(self.images[0])

        if self.selected_landmark.get():
            self._on_landmark_selected()

        for image_path in self.images:
            self.saved_image_snapshots[self._path_key(image_path)] = (
                self._canonical_image_state_for_path(image_path)
            )

        self._refresh_image_listbox()

    @staticmethod
    def _get_app_version() -> str:
        import platform as _platform

        try:
            manifest_path = (
                Path(__file__).parent.parent / ".release-please-manifest.json"
            )
            with manifest_path.open() as f:
                base_version = json.load(f).get(".", "dev")
        except Exception:
            base_version = "dev"

        try:
            if _platform.system() == "Windows":
                try:
                    import platformdirs

                    state_path = (
                        Path(platformdirs.user_documents_dir())
                        / "2D-Point-Annotator"
                        / "update_state.json"
                    )
                except Exception:
                    state_path = (
                        Path.home() / "2D-Point-Annotator" / "update_state.json"
                    )
            else:
                state_path = Path.home() / "2d-point-annotator" / "update_state.json"

            with state_path.open() as f:
                state = json.load(f)

            if state.get("channel") == "nightly":
                sha = state.get("sha", "")
                if sha:
                    return f"{base_version}-{sha[:7]}"
        except Exception:
            pass

        return base_version

    @staticmethod
    def _get_protocol_version() -> str:
        path = Path("docs/landmarks.json")
        try:
            with path.open() as f:
                return json.load(f).get("metadata")["version"]
        except Exception:
            return "N/A"

    def _compute_panel_width(self) -> int:
        screen_w = self.winfo_screenwidth()
        # Target: a wide-but-balanced side panel so the center can stay square
        desired = int(screen_w * 0.27)
        # Ensure both panels + minimum canvas (320px) + padding (30px) fit
        min_canvas = 320
        panel_padding = 30
        max_combined = screen_w - min_canvas - panel_padding

        if max_combined <= 0:
            # Screen too small for the layout; return a minimal workable width
            return 100

        max_each = max_combined // 2
        # Use the smaller of desired and max_each, clamp to a table-friendly range.
        optimal = min(desired, max_each)
        if screen_w >= 1280:
            return max(460, min(520, optimal))
        return max(320, min(460, optimal))

    def _setup_ui(self) -> None:
        self._panel_width = self._compute_panel_width()
        IMAGE_LIST_HEIGHT = 180
        CANVAS_HEIGHT = 380

        self.option_add("*Scale.takeFocus", "0")
        self.option_add("*Checkbutton.takeFocus", "0")
        self.option_add("*Button.takeFocus", "0")
        self.option_add("*Entry.takeFocus", "0")
        self.option_add("*TCombobox.takeFocus", "0")
        ttk.Style(self).configure("PanelHeading.TLabelframe.Label", font=self.heading_font)

        main = tk.Frame(self)
        main.pack(fill="both", expand=True)

        # LEFT PANEL (scrollable)
        left_outer = tk.Frame(main)
        left_outer.pack(side=tk.LEFT, fill="y", padx=(10, 5), pady=10)

        self._left_canvas = tk.Canvas(
            left_outer, width=self._panel_width, highlightthickness=0
        )
        self._left_scrollbar = ttk.Scrollbar(
            left_outer, orient="vertical", command=self._left_canvas.yview
        )
        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)

        self._left_canvas.pack(side=tk.LEFT, fill="y")
        self._left_scrollbar.pack(side=tk.RIGHT, fill="y")

        left_tools = tk.Frame(self._left_canvas)
        self._left_win_id = self._left_canvas.create_window(
            (0, 0), window=left_tools, anchor="nw"
        )
        self._left_tools = left_tools

        # Width sync: canvas Configure → set inner frame width
        self._left_canvas.bind(
            "<Configure>",
            lambda e: self._left_canvas.itemconfigure(self._left_win_id, width=e.width),
        )
        # Scrollregion: inner frame Configure → update scrollregion
        left_tools.bind(
            "<Configure>",
            lambda e: self._left_canvas.configure(
                scrollregion=self._left_canvas.bbox("all")
            ),
        )

        # Mousewheel support for left panel
        left_tools.bind("<Enter>", lambda e: self._bind_left_panel_scroll(True))
        left_tools.bind("<Leave>", lambda e: self._bind_left_panel_scroll(False))

        self.canvas = tk.Canvas(main, bg="grey", width=320, height=320, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True)
        # recompute transform & redraw on canvas size changes
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<ButtonPress-3>", self._on_right_button_press)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_button_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_scroll_linux(1))
        self.canvas.bind("<Button-5>", lambda e: self._on_scroll_linux(-1))
        self.bind("<Configure>", self._on_panel_resize)

        self.landmark_font = tkfont.Font(
            family="Liberation Sans", size=18, weight="bold"
        )
        self.shadow_font = tkfont.Font(family="Liberation Sans", size=20, weight="bold")

        # RIGHT PANEL (scrollable)
        right_outer = tk.Frame(main)
        right_outer.pack(side=tk.RIGHT, fill="y", padx=(5, 10), pady=10)

        self._ctrl_canvas = tk.Canvas(
            right_outer, width=self._panel_width, highlightthickness=0
        )
        self._ctrl_scrollbar = ttk.Scrollbar(
            right_outer, orient="vertical", command=self._ctrl_canvas.yview
        )
        self._ctrl_canvas.configure(yscrollcommand=self._ctrl_scrollbar.set)

        self._ctrl_canvas.pack(side=tk.LEFT, fill="y")
        self._ctrl_scrollbar.pack(side=tk.RIGHT, fill="y")

        ctrl = tk.Frame(self._ctrl_canvas)
        self._ctrl_win_id = self._ctrl_canvas.create_window(
            (0, 0), window=ctrl, anchor="nw"
        )
        self._ctrl = ctrl

        # Width sync: canvas Configure → set inner frame width
        self._ctrl_canvas.bind(
            "<Configure>",
            lambda e: self._ctrl_canvas.itemconfigure(self._ctrl_win_id, width=e.width),
        )
        # Scrollregion: inner frame Configure → update scrollregion
        ctrl.bind(
            "<Configure>",
            lambda e: self._ctrl_canvas.configure(
                scrollregion=self._ctrl_canvas.bbox("all")
            ),
        )

        # Mousewheel support for right panel
        ctrl.bind("<Enter>", lambda e: self._bind_right_panel_scroll(True))
        ctrl.bind("<Leave>", lambda e: self._bind_right_panel_scroll(False))

        zoom_wrap = ttk.LabelFrame(left_tools, text="Zoom View")
        zoom_wrap.pack(fill="x", pady=(0, 8))
        self.zoom_canvas = tk.Canvas(
            zoom_wrap,
            width=self._panel_width,
            height=450,
            bg="black",
            highlightthickness=0,
        )
        self.zoom_canvas.pack(fill="x", padx=0, pady=0)
        tk.Scale(
            zoom_wrap,
            from_=2,
            to=40,
            orient="horizontal",
            label="Zoom (x)",
            variable=self.zoom_percent,
            command=self._on_zoom_change,
            font=self.dialogue_font,
        ).pack(fill="x", padx=6, pady=(6, 6))
        tk.Checkbutton(
            zoom_wrap,
            text="Show Selected Landmark",
            variable=self.show_selected_landmark_in_zoom,
            command=self._refresh_zoom_landmark_overlay,
            font=self.dialogue_font,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        tk.Checkbutton(
            zoom_wrap,
            text="Enhance Contrast (CLAHE)",
            variable=self.enable_zoom_contrast,
            command=lambda: self._update_zoom_view(
                self.last_mouse_canvas_pos[0] if self.last_mouse_canvas_pos else None,
                self.last_mouse_canvas_pos[1] if self.last_mouse_canvas_pos else None,
            ),
            font=self.dialogue_font,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        tk.Checkbutton(
            zoom_wrap,
            text="Pixel-Art Scaling (Scale2x)",
            variable=self.enable_zoom_pixel_art,
            command=lambda: self._update_zoom_view(
                self.last_mouse_canvas_pos[0] if self.last_mouse_canvas_pos else None,
                self.last_mouse_canvas_pos[1] if self.last_mouse_canvas_pos else None,
            ),
            font=self.dialogue_font,
            takefocus=False,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        tk.Checkbutton(
            zoom_wrap,
            text="Percentile Stretch (ignore outliers)",
            variable=self.enable_zoom_percentile_stretch,
            command=lambda: self._update_zoom_view(
                self.last_mouse_canvas_pos[0] if self.last_mouse_canvas_pos else None,
                self.last_mouse_canvas_pos[1] if self.last_mouse_canvas_pos else None,
            ),
            font=self.dialogue_font,
        ).pack(anchor="w", padx=6, pady=(6, 0))
        tk.Scale(
            zoom_wrap,
            from_=0,
            to=49,
            orient="horizontal",
            label="Low Percentile",
            variable=self.zoom_percentile_low,
            command=lambda v: self._update_zoom_view(
                self.last_mouse_canvas_pos[0] if self.last_mouse_canvas_pos else None,
                self.last_mouse_canvas_pos[1] if self.last_mouse_canvas_pos else None,
            ),
            font=self.dialogue_font,
            takefocus=False,
        ).pack(fill="x", padx=6, pady=(0, 0))
        tk.Scale(
            zoom_wrap,
            from_=51,
            to=100,
            orient="horizontal",
            label="High Percentile",
            variable=self.zoom_percentile_high,
            command=lambda v: self._update_zoom_view(
                self.last_mouse_canvas_pos[0] if self.last_mouse_canvas_pos else None,
                self.last_mouse_canvas_pos[1] if self.last_mouse_canvas_pos else None,
            ),
            font=self.dialogue_font,
            takefocus=False,
        ).pack(fill="x", padx=6, pady=(0, 6))
        self.after(0, self._render_black_zoom_view)

        hover_wrap = ttk.LabelFrame(left_tools, text="Hover Circle Tool")
        hover_wrap.pack(fill="x")
        tk.Checkbutton(
            hover_wrap,
            text="Show Hover Circle",
            variable=self.hover_enabled,
            command=self._toggle_hover,
            font=self.dialogue_font,
        ).pack(anchor="w", padx=6, pady=(6, 0))
        self.radius_scale = tk.Scale(
            hover_wrap,
            from_=1,
            to=300,
            orient="horizontal",
            label="Hover Radius",
            variable=self.hover_radius,
            command=self._on_radius_change,
            font=self.dialogue_font,
        )
        self.radius_scale.config(state="disabled")
        self.radius_scale.pack(fill="x", padx=6, pady=6)

        self._refresh_fac_construction_controls()

        cross_wrap = ttk.LabelFrame(left_tools, text="Extended Crosshair Tool")
        cross_wrap.pack(fill="x", pady=(8, 0))

        tk.Checkbutton(
            cross_wrap,
            text="Show Extended Crosshair",
            variable=self.extended_crosshair_enabled,
            command=self._toggle_extended_crosshair,
            font=self.dialogue_font,
        ).pack(anchor="w", padx=6, pady=(6, 0))

        self.crosshair_length_scale = tk.Scale(
            cross_wrap,
            from_=5,
            to=400,
            orient="horizontal",
            label="Crosshair Length",
            variable=self.extended_crosshair_length,
            command=self._on_extended_crosshair_length_change,
            font=self.dialogue_font,
        )
        self.crosshair_length_scale.config(state="disabled")
        self.crosshair_length_scale.pack(fill="x", padx=6, pady=6)

        tk.Label(
            left_tools,
            text=f"App: v{self._get_app_version()}\nProtocol: v{self._get_protocol_version()}\nNOT FDA APPROVED",
            font=self.dialogue_font,
            fg="grey50",
            anchor="w",
            justify="left",
        ).pack(side="bottom", fill="x", padx=6, pady=(0, 4))

        tk.Button(
            ctrl, text="Load Data", command=self.load_data, font=self.heading_font
        ).pack(fill="x", pady=5)
        tk.Button(
            ctrl, text="Next Image", command=self._next_image, font=self.heading_font
        ).pack(fill="x", pady=5)
        tk.Button(
            ctrl,
            text="Previous Image",
            command=self._prev_image,
            font=self.heading_font,
        ).pack(fill="x", pady=5)
        tk.Button(
            ctrl,
            text="Save Annotations",
            command=self.save_annotations,
            font=self.heading_font,
        ).pack(fill="x", pady=5)

        self.autosave_check = tk.Checkbutton(
            ctrl,
            text="Autosave",
            variable=self.autosave_var,
            font=self.dialogue_font,
        )
        self.autosave_check.pack(anchor="w", pady=(0, 6))

        image_cat_wrap = ttk.LabelFrame(
            ctrl, text="Hip Information", style="PanelHeading.TLabelframe"
        )
        image_cat_wrap.pack(fill="x", pady=(10, 10))
        image_cat_wrap.grid_columnconfigure(1, weight=1)
        category_values = HIP_CATEGORY_DISPLAY_VALUES
        tk.Label(image_cat_wrap, text="Left Hip", font=self.dialogue_font).grid(
            row=0, column=0, sticky="w", padx=(6, 4), pady=(6, 2)
        )
        self.left_hip_category_combo = ttk.Combobox(
            image_cat_wrap,
            textvariable=self.left_hip_category_var,
            values=category_values,
            state="readonly",
            font=self.dialogue_font,
            width=16,
            takefocus=False,
        )
        self.left_hip_category_combo.grid(
            row=0, column=1, sticky="ew", padx=(0, 6), pady=(6, 2)
        )
        self.left_hip_category_combo.bind(
            "<<ComboboxSelected>>", self._on_categorical_annotation_changed
        )
        tk.Label(image_cat_wrap, text="Right Hip", font=self.dialogue_font).grid(
            row=1, column=0, sticky="w", padx=(6, 4), pady=(2, 6)
        )
        self.right_hip_category_combo = ttk.Combobox(
            image_cat_wrap,
            textvariable=self.right_hip_category_var,
            values=category_values,
            state="readonly",
            font=self.dialogue_font,
            width=16,
            takefocus=False,
        )
        self.right_hip_category_combo.grid(
            row=1, column=1, sticky="ew", padx=(0, 6), pady=(2, 6)
        )
        self.right_hip_category_combo.bind(
            "<<ComboboxSelected>>", self._on_categorical_annotation_changed
        )

        tk.Label(ctrl, text="Images in JSON", font=self.heading_font).pack(anchor="w")

        self._image_container = tk.Frame(
            ctrl,
            bd=1,
            relief="sunken",
            width=self._panel_width,
            height=IMAGE_LIST_HEIGHT,
        )
        self._image_container.pack(fill="x", pady=(2, 8))
        self._image_container.pack_propagate(False)

        self.image_tree = ttk.Treeview(
            self._image_container,
            columns=("image", "progress"),
            show="headings",
            height=8,
        )
        self.image_tree.heading("image", text="Image")
        self.image_tree.heading("progress", text="Done")
        self.image_tree.column("image", width=290, anchor="w")
        self.image_tree.column("progress", width=90, anchor="center")
        self.image_tree.pack(side=tk.LEFT, fill="both", expand=True)

        image_scrollbar = tk.Scrollbar(
            self._image_container,
            orient="vertical",
            command=self.image_tree.yview,
        )
        image_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.image_tree.configure(yscrollcommand=image_scrollbar.set)

        self.image_tree.bind("<<TreeviewSelect>>", self._on_image_list_select)
        self.image_tree.bind("<Enter>", lambda _e: self._bind_image_list_scroll(True))
        self.image_tree.bind("<Leave>", lambda _e: self._bind_image_list_scroll(False))

        lm_heading_row = tk.Frame(ctrl)
        lm_heading_row.pack(fill="x")
        tk.Label(lm_heading_row, text="Landmarks", font=self.heading_font).pack(
            side="left"
        )
        if self._landmark_ref is not None:
            tk.Button(
                lm_heading_row,
                text="?",
                command=self._open_landmark_reference,
                font=self.dialogue_font,
                width=2,
                padx=0,
                pady=0,
            ).pack(side="left", padx=(4, 0))
        self._landmark_panel_container = tk.Frame(
            ctrl, bd=1, relief="sunken", width=self._panel_width, height=CANVAS_HEIGHT
        )
        self._landmark_panel_container.pack(fill="x", pady=(0, 0))
        self._landmark_panel_container.pack_propagate(False)

        lp_header = tk.Frame(self._landmark_panel_container)
        lp_header.pack(fill="x", padx=(4, self._scrollbar_width + 4), pady=(2, 0))
        lp_header.grid_columnconfigure(0, minsize=220)
        lp_header.grid_columnconfigure(1, minsize=58)
        lp_header.grid_columnconfigure(2, minsize=82)
        lp_header.grid_columnconfigure(3, minsize=74)
        lp_header.grid_anchor("w")
        for col, title in enumerate(("Acronym", "Show", "Status", "Delete")):
            tk.Label(
                lp_header,
                text=title,
                anchor="w",
                font=self.landmark_table_header_font,
            ).grid(row=0, column=col, sticky="ew", padx=(2, 4))

        self.lp_canvas = tk.Canvas(
            self._landmark_panel_container,
            height=CANVAS_HEIGHT,
            width=self._panel_width - self._scrollbar_width,
            highlightthickness=0,
        )
        self.lp_canvas.pack(side=tk.LEFT, fill="both")
        self.lp_scrollbar = tk.Scrollbar(
            self._landmark_panel_container,
            orient="vertical",
            command=self.lp_canvas.yview,
        )
        self.lp_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.lp_canvas.configure(yscrollcommand=self.lp_scrollbar.set)
        self.lp_inner = tk.Frame(self.lp_canvas)
        self.lp_inner_win_id = self.lp_canvas.create_window(
            (0, 0), window=self.lp_inner, anchor="nw"
        )
        self.lp_canvas.bind(
            "<Configure>",
            lambda e: self.lp_canvas.itemconfigure(self.lp_inner_win_id, width=e.width),
        )
        self.lp_inner.bind(
            "<Configure>",
            lambda e: self.lp_canvas.configure(scrollregion=self.lp_canvas.bbox("all")),
        )
        self.lp_inner.bind("<Enter>", lambda e: self._bind_landmark_scroll(True))
        self.lp_inner.bind("<Leave>", lambda e: self._bind_landmark_scroll(False))
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=(6, 6))
        buttons_row = tk.Frame(ctrl)
        buttons_row.pack(fill="x", pady=(0, 6))
        tk.Button(
            buttons_row,
            text="View All",
            command=lambda: self._set_all_visibility(True),
            font=self.dialogue_font,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(
            buttons_row,
            text="View None",
            command=lambda: self._set_all_visibility(False),
            font=self.dialogue_font,
        ).pack(side="left", expand=True, fill="x")

        status_wrap = ttk.LabelFrame(
            ctrl, text="Landmark Status", style="PanelHeading.TLabelframe"
        )
        status_wrap.pack(fill="x", pady=(8, 0))
        status_wrap.grid_columnconfigure(0, weight=1)

        primary_status_row = tk.Frame(status_wrap)
        primary_status_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 6))
        primary_status_row.grid_columnconfigure(1, weight=1)
        tk.Label(
            primary_status_row,
            text="Primary Landmark",
            font=self.dialogue_font,
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.landmark_status_combo = ttk.Combobox(
            primary_status_row,
            textvariable=self.landmark_status_var,
            values=DIRECT_ANNOTATION_STATUS_DISPLAY_VALUES,
            state="disabled",
            font=self.dialogue_font,
            width=18,
            takefocus=False,
        )
        self.landmark_status_combo.grid(row=0, column=1, sticky="ew")
        self.landmark_status_combo.bind(
            "<<ComboboxSelected>>", self._on_landmark_status_changed
        )

        ttk.Separator(status_wrap, orient="horizontal").grid(
            row=1, column=0, sticky="ew", padx=6, pady=(0, 6)
        )
        self.derived_feature_status_frame = tk.Frame(status_wrap)
        self.derived_feature_status_frame.grid(
            row=2, column=0, sticky="ew", padx=6, pady=(0, 6)
        )

        ttk.Separator(left_tools, orient="horizontal").pack(fill="x", pady=(8, 6))
        it_edge_wrap = ttk.LabelFrame(left_tools, text="Contour Highlight Tool")
        it_edge_wrap.pack(fill="x", pady=(0, 8))
        it_mode_row = tk.Frame(it_edge_wrap)
        it_mode_row.pack(fill="x", padx=6, pady=(6, 2))
        tk.Radiobutton(
            it_mode_row,
            text="Draw",
            variable=self.it_edge_border_mode,
            value="draw",
            command=self._on_it_edge_border_mode_changed,
            font=self.dialogue_font,
        ).pack(side="left")
        tk.Radiobutton(
            it_mode_row,
            text="Erase",
            variable=self.it_edge_border_mode,
            value="erase",
            command=self._on_it_edge_border_mode_changed,
            font=self.dialogue_font,
        ).pack(side="left", padx=(10, 0))
        self.it_edge_border_brush_scale = tk.Scale(
            it_edge_wrap,
            from_=1,
            to=40,
            orient="horizontal",
            label="Grid Pixel Size",
            variable=self.it_edge_border_brush_size,
            font=self.dialogue_font,
        )
        self.it_edge_border_brush_scale.pack(fill="x", padx=6, pady=(2, 6))
        self._refresh_it_edge_border_brush_control()
        it_button_row = tk.Frame(it_edge_wrap)
        it_button_row.pack(fill="x", padx=6, pady=(0, 8))
        tk.Button(
            it_button_row,
            text="Clear Selected",
            command=self._clear_selected_it_edge_border,
            font=self.dialogue_font,
        ).pack(fill="x")

    def _detect_path_column(self, df: pd.DataFrame) -> str:
        """Detect which column contains image paths."""
        # Try known column names in order of preference
        candidates = ["image_path", "Dataset", "dataset", "path", "file", "filename"]

        for col in candidates:
            if col in df.columns:
                return col

        # Fallback: use first column if none match
        if len(df.columns) > 0:
            return df.columns[0]

        # Last resort default
        return "image_path"

    def _recompute_transform(self) -> None:
        if not self.current_image:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.current_image.size
        scale = min(cw / iw, ch / ih)
        disp_w, disp_h = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2
        self.disp_scale = scale
        self.disp_off = (off_x, off_y)
        self.disp_size = (disp_w, disp_h)

    def _img_to_screen(self, xi: float, yi: float) -> Tuple[float, float]:
        off_x, off_y = self.disp_off
        s = self.disp_scale
        return off_x + xi * s, off_y + yi * s

    def _screen_to_img(self, xs: float, ys: float) -> Tuple[float, float]:
        off_x, off_y = self.disp_off
        s = self.disp_scale or 1.0
        return (xs - off_x) / s, (ys - off_y) / s

    def _display_rect(self) -> Tuple[int, int, int, int]:
        off_x, off_y = self.disp_off
        disp_w, disp_h = self.disp_size
        return off_x, off_y, off_x + disp_w, off_y + disp_h

    def _get_zoom_canvas_size(self) -> int:
        if self.zoom_canvas is None:
            return 1

        w = self.zoom_canvas.winfo_width()
        h = self.zoom_canvas.winfo_height()

        if w <= 1 or h <= 1:
            req_w = self.zoom_canvas.winfo_reqwidth()
            req_h = self.zoom_canvas.winfo_reqheight()
            w = max(w, req_w, 1)
            h = max(h, req_h, 1)

        return max(1, min(w, h))

    def _render_black_zoom_view(self) -> None:
        if self.zoom_canvas is None:
            return

        size = self._get_zoom_canvas_size()
        black_img = Image.new("RGB", (size, size), "black")
        self.zoom_img_obj = ImageTk.PhotoImage(black_img)

        if self.zoom_base_item is None:
            self.zoom_base_item = self.zoom_canvas.create_image(
                0, 0, anchor="nw", image=self.zoom_img_obj
            )
        else:
            self.zoom_canvas.itemconfigure(self.zoom_base_item, image=self.zoom_img_obj)
            self.zoom_canvas.coords(self.zoom_base_item, 0, 0)

        self.zoom_src_rect = None
        self._update_zoom_crosshair()
        self._clear_zoom_landmark_overlay()

    def _scale2x_numpy(self, img_array):
        """
        Scale2x (EPX) algorithm using vectorized NumPy.
        Scales a 2D (grayscale) or 3D (RGB) image array by 2x.
        """
        h, w = img_array.shape[:2]
        is_rgb = img_array.ndim == 3

        if is_rgb:
            out = np.zeros((h * 2, w * 2, img_array.shape[2]), dtype=img_array.dtype)
            padded = np.pad(img_array, ((1, 1), (1, 1), (0, 0)), mode="edge")
        else:
            out = np.zeros((h * 2, w * 2), dtype=img_array.dtype)
            padded = np.pad(img_array, ((1, 1), (1, 1)), mode="edge")

        # Neighbor extraction (A=top, B=right, C=bottom, D=left, P=center)
        A = padded[0:-2, 1:-1]
        D = padded[1:-1, 0:-2]
        P = padded[1:-1, 1:-1]
        B = padded[1:-1, 2:]
        C = padded[2:, 1:-1]

        # Scale2x / EPX rules:
        #   IF C==A AND C!=D AND A!=B => top-left=A
        #   IF A==B AND A!=C AND B!=D => top-right=B
        #   IF D==C AND D!=B AND C!=A => bottom-left=C
        #   IF B==D AND B!=A AND D!=C => bottom-right=D
        if is_rgb:
            eq = lambda x, y: np.all(x == y, axis=-1)
        else:
            eq = lambda x, y: x == y

        # Default all output pixels to P
        out[0::2, 0::2] = P
        out[0::2, 1::2] = P
        out[1::2, 0::2] = P
        out[1::2, 1::2] = P

        cond = eq(C, A) & ~eq(C, D) & ~eq(A, B)
        mask = cond & eq(C, A)
        out[0::2, 0::2] = np.where(mask[..., None] if is_rgb else mask, A, P)

        cond = eq(A, B) & ~eq(A, C) & ~eq(B, D)
        out[0::2, 1::2] = np.where(cond[..., None] if is_rgb else cond, B, P)

        cond = eq(D, C) & ~eq(D, B) & ~eq(C, A)
        out[1::2, 0::2] = np.where(cond[..., None] if is_rgb else cond, C, P)

        cond = eq(B, D) & ~eq(B, A) & ~eq(D, C)
        out[1::2, 1::2] = np.where(cond[..., None] if is_rgb else cond, D, P)

        return out

    def _update_zoom_view(
        self, mouse_x: Optional[float] = None, mouse_y: Optional[float] = None
    ) -> None:
        if self.zoom_canvas is None:
            return

        size = self._get_zoom_canvas_size()

        if self.current_image is None or mouse_x is None or mouse_y is None:
            self._render_black_zoom_view()
            return

        x0, y0, x1, y1 = self._display_rect()
        if not (x0 <= mouse_x < x1 and y0 <= mouse_y < y1):
            self._render_black_zoom_view()
            return

        xi, yi = self._screen_to_img(mouse_x, mouse_y)
        iw, ih = self.current_image.size

        if not (0 <= xi < iw and 0 <= yi < ih):
            self._render_black_zoom_view()
            return

        zoom_lev = max(2, min(40, float(self.zoom_percent.get())))
        half_w = max(1.0, iw / (zoom_lev * 2.0))
        half_h = max(1.0, ih / (zoom_lev * 2.0))

        src_left = float(xi) - half_w
        src_top = float(yi) - half_h
        src_right = float(xi) + half_w
        src_bottom = float(yi) + half_h

        self.zoom_src_rect = (src_left, src_top, src_right, src_bottom)

        if self.enable_zoom_pixel_art.get():
            crop_box = (
                int(np.floor(src_left)),
                int(np.floor(src_top)),
                int(np.ceil(src_right)),
                int(np.ceil(src_bottom)),
            )
            patch = self.current_image.crop(crop_box).convert("L")
            patch_np = np.array(patch)

            patch_np = self._scale2x_numpy(patch_np)
            patch_np = self._scale2x_numpy(patch_np)

            out = Image.fromarray(patch_np)
            out = out.resize((size, size), Image.Resampling.NEAREST)
        else:
            try:
                out = self.current_image.transform(
                    (size, size),
                    Image.Transform.EXTENT,
                    self.zoom_src_rect,
                    resample=Image.Resampling.BICUBIC,
                    fill=0,
                )
            except TypeError:
                out = self.current_image.transform(
                    (size, size),
                    Image.Transform.EXTENT,
                    self.zoom_src_rect,
                    resample=Image.Resampling.BICUBIC,
                )

        if self.enable_zoom_contrast.get():
            out_gray = out.convert("L")
            img_np = np.array(out_gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            out = Image.fromarray(img_np)

        if self.enable_zoom_percentile_stretch.get():
            # Convert to numpy array if not already from CLAHE
            if not isinstance(out, Image.Image):
                img_np = out
            else:
                img_np = np.array(out)

            low = self.zoom_percentile_low.get()
            high = self.zoom_percentile_high.get()
            img_np = self._percentile_contrast_stretch(img_np, low, high)
            out = Image.fromarray(img_np)

        self.zoom_img_obj = ImageTk.PhotoImage(out)

        if self.zoom_base_item is None:
            self.zoom_base_item = self.zoom_canvas.create_image(
                0, 0, anchor="nw", image=self.zoom_img_obj
            )
        else:
            self.zoom_canvas.itemconfigure(self.zoom_base_item, image=self.zoom_img_obj)
            self.zoom_canvas.coords(self.zoom_base_item, 0, 0)

        self._update_zoom_crosshair()
        self._refresh_zoom_landmark_overlay()
        self._update_zoom_hover_circle()

    def _on_zoom_change(self, _value) -> None:
        if self.last_mouse_canvas_pos is None:
            self._update_zoom_view(None, None)
            return

        mouse_x, mouse_y = self.last_mouse_canvas_pos
        self._update_zoom_view(mouse_x, mouse_y)

    def _change_zoom_percent(self, delta) -> None:
        new_zoom = max(2, min(40, self.zoom_percent.get() + delta))
        if new_zoom != self.zoom_percent.get():
            self.zoom_percent.set(new_zoom)
            self._on_zoom_change(str(new_zoom))

    def _update_zoom_crosshair(self) -> None:
        if self.zoom_canvas is None:
            return

        size = self._get_zoom_canvas_size()
        x = size / 2
        y = size / 2

        circle_r = 16
        cross_r = circle_r

        circle_color = "blue"
        crosshair_color = "orange"

        if not self.zoom_crosshair_ids:
            circle_id = self.zoom_canvas.create_oval(
                x - circle_r,
                y - circle_r,
                x + circle_r,
                y + circle_r,
                outline=circle_color,
                width=1,
                tags="zoom_crosshair",
            )
            hline_id = self.zoom_canvas.create_line(
                x - cross_r,
                y,
                x + cross_r,
                y,
                fill=crosshair_color,
                width=1,
                tags="zoom_crosshair",
            )
            vline_id = self.zoom_canvas.create_line(
                x,
                y - cross_r,
                x,
                y + cross_r,
                fill=crosshair_color,
                width=1,
                tags="zoom_crosshair",
            )
            self.zoom_crosshair_ids = [circle_id, hline_id, vline_id]
        else:
            circle_id, hline_id, vline_id = self.zoom_crosshair_ids
            self.zoom_canvas.coords(
                circle_id,
                x - circle_r,
                y - circle_r,
                x + circle_r,
                y + circle_r,
            )
            self.zoom_canvas.coords(hline_id, x - cross_r, y, x + cross_r, y)
            self.zoom_canvas.coords(vline_id, x, y - cross_r, x, y + cross_r)
            self.zoom_canvas.itemconfigure(circle_id, outline=circle_color)
            self.zoom_canvas.itemconfigure(hline_id, fill=crosshair_color)
            self.zoom_canvas.itemconfigure(vline_id, fill=crosshair_color)

        for item_id in self.zoom_crosshair_ids:
            self.zoom_canvas.tag_raise(item_id)

        self._update_zoom_extended_crosshair()

    def _hide_zoom_crosshair(self) -> None:
        if self.zoom_canvas is None:
            return

        for item_id in self.zoom_crosshair_ids:
            self.zoom_canvas.delete(item_id)
        self.zoom_crosshair_ids = []

    def _update_zoom_extended_crosshair(self) -> None:
        if self.zoom_canvas is None:
            return

        if not self.extended_crosshair_enabled.get():
            self._hide_zoom_extended_crosshair()
            return

        size = self._get_zoom_canvas_size()
        x = size / 2
        y = size / 2

        length = max(5, min(400, int(self.extended_crosshair_length.get())))

        # Keep it inside the zoom canvas
        max_len = max(1, int(size / 2) - 2)
        length = min(length, max_len)

        if not self.zoom_extended_crosshair_ids:
            hline_id = self.zoom_canvas.create_line(
                x - length,
                y,
                x + length,
                y,
                fill="lime",
                width=1,
                tags="zoom_extended_crosshair",
            )
            vline_id = self.zoom_canvas.create_line(
                x,
                y - length,
                x,
                y + length,
                fill="lime",
                width=1,
                tags="zoom_extended_crosshair",
            )
            self.zoom_extended_crosshair_ids = [hline_id, vline_id]
        else:
            hline_id, vline_id = self.zoom_extended_crosshair_ids
            self.zoom_canvas.coords(hline_id, x - length, y, x + length, y)
            self.zoom_canvas.coords(vline_id, x, y - length, x, y + length)

        for item_id in self.zoom_extended_crosshair_ids:
            self.zoom_canvas.tag_raise(item_id)

    def _hide_zoom_extended_crosshair(self) -> None:
        if self.zoom_canvas is None:
            return

        for item_id in self.zoom_extended_crosshair_ids:
            try:
                self.zoom_canvas.delete(item_id)
            except Exception as e:
                logger.warning(f"Failed to delete zoom extended crosshair item: {e}")
        self.zoom_extended_crosshair_ids = []

    def _change_femoral_axis_length(self, delta: int) -> None:
        new_len = max(2, min(300, int(self.femoral_axis_proj_length.get()) + delta))
        if new_len != self.femoral_axis_proj_length.get():
            self.femoral_axis_proj_length.set(new_len)
            self._update_femoral_axis_overlay()

    def _toggle_femoral_axis(self) -> None:
        enabled = self.femoral_axis_enabled.get()
        count_scale = self.__dict__.get("femoral_axis_count_scale")
        tip_scale = self.__dict__.get("femoral_axis_whisker_tip_length_scale")
        if count_scale is not None:
            count_scale.config(state="normal" if enabled else "disabled")
        if tip_scale is not None:
            tip_scale.config(state="normal" if enabled else "disabled")

        if enabled and self.hover_enabled.get():
            self.hover_enabled.set(False)
            radius_scale = self.__dict__.get("radius_scale")
            if radius_scale is not None:
                radius_scale.config(state="disabled")
            self._hide_hover_circle()

        if not enabled:
            self._clear_femoral_axis_overlay()
        else:
            self._update_femoral_axis_overlay()

    def _on_femoral_axis_whisker_tip_length_change(self, _value: str) -> None:
        if not self.femoral_axis_enabled.get():
            return
        new_len = max(1, min(80, int(self.femoral_axis_whisker_tip_length.get())))
        self.femoral_axis_whisker_tip_length.set(new_len)
        self._update_femoral_axis_overlay()

    def _on_femoral_axis_count_change(self, _value: str) -> None:
        if not self.femoral_axis_enabled.get():
            return
        count = max(1, min(20, int(self.femoral_axis_count.get())))
        self.femoral_axis_count.set(count)
        self._update_femoral_axis_overlay()

    def _clear_femoral_axis_overlay(self) -> None:
        for item_id in self.femoral_axis_item_ids:
            try:
                self.canvas.delete(item_id)
            except Exception as e:
                logger.warning(f"Failed to delete femoral axis item: {e}")
        self.femoral_axis_item_ids = []

    def _change_femoral_axis_whisker_tip_length(self, delta: int) -> None:
        new_len = max(
            1,
            min(80, int(self.femoral_axis_whisker_tip_length.get()) + delta),
        )
        if new_len != self.femoral_axis_whisker_tip_length.get():
            self.femoral_axis_whisker_tip_length.set(new_len)
            self._update_femoral_axis_overlay()

    def _get_active_femoral_axis_line_screen(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        if not self.current_image:
            return None
        if not self.femoral_axis_enabled.get():
            return None

        landmark = self.selected_landmark.get()
        if landmark not in ("L-FA", "R-FA"):
            return None

        points = self._get_line_points(landmark)
        if len(points) >= 2:
            x1, y1 = self._img_to_screen(*points[0])
            x2, y2 = self._img_to_screen(*points[1])
            return x1, y1, x2, y2

        if len(points) == 1 and self.last_mouse_canvas_pos is not None:
            mouse_x, mouse_y = self.last_mouse_canvas_pos
            x0, y0, x1, y1 = self._display_rect()
            if x0 <= mouse_x < x1 and y0 <= mouse_y < y1:
                sx, sy = self._img_to_screen(*points[0])
                return sx, sy, mouse_x, mouse_y

        return None

    def _update_femoral_axis_overlay(self) -> None:
        self._clear_femoral_axis_overlay()

        line = self._get_active_femoral_axis_line_screen()
        if line is None:
            return

        x1, y1, x2, y2 = line
        vx = x2 - x1
        vy = y2 - y1
        mag = float((vx * vx + vy * vy) ** 0.5)
        if mag < 1e-6:
            return

        tx = vx / mag
        ty = vy / mag
        nx = -ty
        ny = tx

        n_proj = max(1, int(self.femoral_axis_count.get()))
        proj_len = float(self.femoral_axis_proj_length.get())
        cap_half = float(self.femoral_axis_whisker_tip_length.get())

        for i in range(1, n_proj + 1):
            frac = i / (n_proj + 1.0)
            cx = x1 + frac * vx
            cy = y1 + frac * vy

            ax = cx - proj_len * nx
            ay = cy - proj_len * ny
            bx = cx + proj_len * nx
            by = cy + proj_len * ny

            main_id = self.canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill="magenta",
                width=2,
                tags="femoral_axis",
            )
            cap1_id = self.canvas.create_line(
                ax - cap_half * tx,
                ay - cap_half * ty,
                ax + cap_half * tx,
                ay + cap_half * ty,
                fill="magenta",
                width=2,
                tags="femoral_axis",
            )
            cap2_id = self.canvas.create_line(
                bx - cap_half * tx,
                by - cap_half * ty,
                bx + cap_half * tx,
                by + cap_half * ty,
                fill="magenta",
                width=2,
                tags="femoral_axis",
            )
            self.femoral_axis_item_ids.extend([main_id, cap1_id, cap2_id])

        for item_id in self.femoral_axis_item_ids:
            try:
                self.canvas.tag_raise(item_id)
            except Exception:
                pass
        self.canvas.tag_raise("marker")

    def _clear_zoom_landmark_overlay(self) -> None:
        if self.zoom_canvas is None:
            return

        for item_id in getattr(self, "zoom_landmark_overlay_ids", []):
            try:
                self.zoom_canvas.delete(item_id)
            except Exception as e:
                logger.warning(f"Failed to delete zoom landmark overlay item: {e}")
        self.zoom_landmark_overlay_ids = []
        self.zoom_seg_img_obj = None

    def _is_line_landmark(self, lm: str) -> bool:
        return lm in self.line_landmarks

    def _get_ellipse(self, lm: str) -> EllipseAnnotation | None:
        pts, _quality = self._get_annotations()
        return self._normalize_ellipse_value(pts.get(lm))

    def _ellipse_axis_points(
        self, ellipse: object
    ) -> Tuple[
        AnnotationPoint,
        AnnotationPoint,
        AnnotationPoint,
        AnnotationPoint,
        AnnotationPoint,
    ] | None:
        norm = self._normalize_ellipse_value(ellipse)
        if norm is None:
            return None

        center = self._point_from_sequence(norm.get("center"))
        major_raw = norm.get("major")
        minor_raw = norm.get("minor")
        if (
            center is None
            or not isinstance(major_raw, list)
            or not isinstance(minor_raw, list)
            or len(major_raw) != 2
            or len(minor_raw) != 2
        ):
            return None

        major0 = self._point_from_sequence(major_raw[0])
        major1 = self._point_from_sequence(major_raw[1])
        minor0 = self._point_from_sequence(minor_raw[0])
        minor1 = self._point_from_sequence(minor_raw[1])
        if major0 is None or major1 is None or minor0 is None or minor1 is None:
            return None

        return center, major0, major1, minor0, minor1

    def _ellipse_from_center_vectors(
        self,
        center: AnnotationPoint,
        major_vec: AnnotationPoint,
        minor_vec: AnnotationPoint,
        construction: str | None = None,
    ) -> EllipseAnnotation:
        cx, cy = center
        ux, uy = major_vec
        vx, vy = minor_vec

        def point(px: float, py: float) -> List[float]:
            return [round(float(px), 1), round(float(py), 1)]

        ellipse: EllipseAnnotation = {
            "type": "ellipse",
            "center": point(cx, cy),
            "major": [point(cx - ux, cy - uy), point(cx + ux, cy + uy)],
            "minor": [point(cx - vx, cy - vy), point(cx + vx, cy + vy)],
        }
        if construction in FAC_CONSTRUCTION_MODES:
            ellipse["construction"] = construction
        return ellipse

    def _limit_axis_vector_to_image(
        self, center: AnnotationPoint, vec: AnnotationPoint
    ) -> AnnotationPoint:
        if not self.current_image:
            return vec

        cx, cy = center
        vx, vy = vec
        w, h = self.current_image.size
        max_x = float(max(0, w - 1))
        max_y = float(max(0, h - 1))
        limit = 1.0

        for dx, dy in ((vx, vy), (-vx, -vy)):
            if abs(dx) > 1e-12:
                bound = (max_x - cx) / dx if dx > 0 else (0.0 - cx) / dx
                limit = min(limit, bound)
            if abs(dy) > 1e-12:
                bound = (max_y - cy) / dy if dy > 0 else (0.0 - cy) / dy
                limit = min(limit, bound)

        limit = max(0.0, min(1.0, limit))
        return vx * limit, vy * limit

    def _default_ellipse_for_center(
        self, center: AnnotationPoint, construction: str | None = None
    ) -> EllipseAnnotation:
        cx, cy = self._clamp_img_point(center[0], center[1])
        scale = self.disp_scale or 1.0
        major_len = max(8.0, 70.0 / scale)
        minor_len = max(5.0, 35.0 / scale)
        major_vec = self._limit_axis_vector_to_image((cx, cy), (major_len, 0.0))
        minor_vec = self._limit_axis_vector_to_image((cx, cy), (0.0, minor_len))
        return self._ellipse_from_center_vectors(
            (cx, cy), major_vec, minor_vec, construction
        )

    def _set_ellipse(
        self,
        lm: str,
        ellipse: object,
        construction: str | None = None,
        *,
        sync_dependents: bool = True,
    ) -> None:
        normalized = self._normalize_ellipse_value(ellipse)
        if normalized is None:
            return
        if construction in FAC_CONSTRUCTION_MODES:
            normalized["construction"] = construction
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return
        self.annotations.setdefault(ann_key, {})[lm] = normalized
        self._set_landmark_status(lm, "")
        if sync_dependents and self._ellipse_is_projected_hemisphere(lm, normalized):
            self._sync_projected_hemisphere_ac_for_ellipse(lm, normalized)
        if lm in self.landmark_found:
            self.landmark_found[lm].set(True)
        if sync_dependents:
            self._update_auto_acetabular_points_for_ellipse(lm)

    def _ellipse_polyline_points(
        self, ellipse: object, steps: int = 96
    ) -> List[AnnotationPoint]:
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return []

        center, major0, major1, minor0, minor1 = axes
        cx, cy = center
        ux = (major1[0] - major0[0]) / 2.0
        uy = (major1[1] - major0[1]) / 2.0
        vx = (minor1[0] - minor0[0]) / 2.0
        vy = (minor1[1] - minor0[1]) / 2.0

        points: List[AnnotationPoint] = []
        for i in range(steps + 1):
            t = (2.0 * math.pi * i) / steps
            ct = math.cos(t)
            st = math.sin(t)
            points.append((cx + ux * ct + vx * st, cy + uy * ct + vy * st))
        return points

    def _fac_angle_measurements(
        self, ellipse: object
    ) -> Tuple[float, float] | None:
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return None

        _center, major0, major1, minor0, minor1 = axes
        major_dx = major1[0] - major0[0]
        major_dy = major1[1] - major0[1]
        minor_dx = minor1[0] - minor0[0]
        minor_dy = minor1[1] - minor0[1]
        major_len = math.hypot(major_dx, major_dy)
        minor_len = math.hypot(minor_dx, minor_dy)

        if major_len < 1e-12 or minor_len < 1e-12:
            return None

        inclination = math.degrees(math.atan2(abs(major_dy), abs(major_dx)))
        ratio = max(0.0, min(1.0, minor_len / major_len))
        anteversion = math.degrees(math.asin(ratio))
        return inclination, anteversion

    def _draw_fac_angle_readout(
        self,
        canvas,
        ellipse: object,
        img_to_canvas,
        *,
        font,
        tags: str = "marker",
        fill: str = "#FFCC66",
    ) -> List[int]:
        measurements = self._fac_angle_measurements(ellipse)
        if measurements is None:
            return []

        polyline = self._ellipse_polyline_points(ellipse)
        if not polyline:
            return []

        screen_points = [img_to_canvas(px, py) for px, py in polyline[:-1]]
        if not screen_points:
            return []

        top = min(point[1] for point in screen_points)
        right = max(point[0] for point in screen_points)
        label_x = right + 8
        label_y = top + 4

        try:
            canvas_w = int(canvas.winfo_width())
        except Exception:
            canvas_w = 0
        if canvas_w > 120:
            label_x = min(label_x, canvas_w - 96)
        label_x = max(4, label_x)
        label_y = max(4, label_y)

        inclination, anteversion = measurements
        text = f"I {inclination:.1f} deg\nA {anteversion:.1f} deg"
        item_ids: List[int] = []
        item_ids.append(
            canvas.create_text(
                label_x + 1,
                label_y + 1,
                text=text,
                fill="black",
                font=font,
                anchor="nw",
                justify="left",
                tags=tags,
            )
        )
        item_ids.append(
            canvas.create_text(
                label_x,
                label_y,
                text=text,
                fill=fill,
                font=font,
                anchor="nw",
                justify="left",
                tags=tags,
            )
        )

        for item_id in item_ids:
            try:
                canvas.tag_raise(item_id)
            except Exception:
                pass

        return item_ids

    def _draw_ellipse_annotation(
        self,
        canvas,
        ellipse: object,
        img_to_canvas,
        *,
        outline: str,
        label_fill: str,
        label: str,
        font,
        shadow_font=None,
        width: int = 2,
        handle_radius: int = 5,
        tags: str = "marker",
        label_offset: Tuple[float, float] = (0.0, 0.0),
        show_handles: bool = True,
        dash: Tuple[int, int] = (2, 4),
    ) -> List[int]:
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return []

        center, major0, major1, minor0, minor1 = axes
        polyline = self._ellipse_polyline_points(ellipse)
        if not polyline:
            return []

        item_ids: List[int] = []
        screen_polyline = [img_to_canvas(px, py) for px, py in polyline]
        flat_polyline = [
            coord for point in screen_polyline for coord in (point[0], point[1])
        ]
        item_ids.append(
            canvas.create_line(
                *flat_polyline,
                fill=outline,
                width=width,
                smooth=True,
                tags=tags,
            )
        )

        major0_s = img_to_canvas(*major0)
        major1_s = img_to_canvas(*major1)
        minor0_s = img_to_canvas(*minor0)
        minor1_s = img_to_canvas(*minor1)

        item_ids.append(
            canvas.create_line(
                major0_s[0],
                major0_s[1],
                major1_s[0],
                major1_s[1],
                fill=outline,
                width=width,
                dash=dash,
                tags=tags,
            )
        )
        item_ids.append(
            canvas.create_line(
                minor0_s[0],
                minor0_s[1],
                minor1_s[0],
                minor1_s[1],
                fill=outline,
                width=width,
                dash=dash,
                tags=tags,
            )
        )

        if show_handles:
            for hx, hy in (major0_s, major1_s, minor0_s, minor1_s):
                item_ids.append(
                    canvas.create_oval(
                        hx - handle_radius,
                        hy - handle_radius,
                        hx + handle_radius,
                        hy + handle_radius,
                        outline=outline,
                        width=width,
                        tags=tags,
                    )
                )

        if label:
            label_x = sum(point[0] for point in screen_polyline[:-1]) / max(
                1, len(screen_polyline) - 1
            )
            label_y = min(point[1] for point in screen_polyline[:-1]) - 14
            label_x += label_offset[0]
            label_y += label_offset[1]
            if shadow_font is not None:
                item_ids.append(
                    canvas.create_text(
                        label_x - 1,
                        label_y - 1,
                        text=label,
                        fill="black",
                        font=shadow_font,
                        tags=tags,
                    )
                )
            item_ids.append(
                canvas.create_text(
                    label_x,
                    label_y,
                    text=label,
                    fill=label_fill,
                    font=font,
                    tags=tags,
                )
            )

        for item_id in item_ids:
            try:
                canvas.tag_raise(item_id)
            except Exception:
                pass

        return item_ids

    def _lower_arc_segment_points(
        self, ellipse_lm: str, ratio0: float, ratio1: float, steps: int = 48
    ) -> List[AnnotationPoint]:
        ratio0 = max(0.0, min(1.0, float(ratio0)))
        ratio1 = max(0.0, min(1.0, float(ratio1)))
        if abs(ratio1 - ratio0) < 1e-6:
            return []

        count = max(2, int(steps))
        return [
            point
            for idx in range(count + 1)
            if (
                point := self._point_on_lower_arc_at_ratio(
                    ellipse_lm, ratio0 + (ratio1 - ratio0) * idx / count
                )
            )
            is not None
        ]

    def _draw_acetabular_lower_arc_highlights(
        self,
        canvas,
        ellipse_lm: str,
        img_to_canvas,
        *,
        width: int,
        tags: str = "marker",
    ) -> List[int]:
        intermediate = self._intermediate_acetabular_landmarks_for_ellipse(ellipse_lm)
        if intermediate is None:
            return []

        idab_lm, isab_lm = intermediate
        if not (
            self._intermediate_acetabular_landmark_ready(idab_lm)
            and self._intermediate_acetabular_landmark_ready(isab_lm)
        ):
            return []

        pts, _quality = self._get_annotations()
        item_ids: List[int] = []
        segments: List[Tuple[float, float]] = []

        isab = self._landmark_center_from_value(isab_lm, pts.get(isab_lm))
        if isab is not None:
            isab_ratio = self._lower_arc_ratio_for_point(ellipse_lm, isab)
            if isab_ratio is not None:
                segments.append((0.0, isab_ratio))

        idab = self._landmark_center_from_value(idab_lm, pts.get(idab_lm))
        if idab is not None:
            idab_ratio = self._lower_arc_ratio_for_point(ellipse_lm, idab)
            if idab_ratio is not None:
                segments.append((idab_ratio, 1.0))

        for start_ratio, end_ratio in segments:
            arc_points = self._lower_arc_segment_points(
                ellipse_lm, start_ratio, end_ratio
            )
            if len(arc_points) < 2:
                continue
            screen_points = [img_to_canvas(px, py) for px, py in arc_points]
            flat = [coord for point in screen_points for coord in (point[0], point[1])]
            item_ids.append(
                canvas.create_line(
                    *flat,
                    fill=ACETABULAR_LOWER_ARC_HIGHLIGHT,
                    width=max(3, int(width) + 2),
                    smooth=True,
                    tags=tags,
                )
            )

        for item_id in item_ids:
            try:
                canvas.tag_raise(item_id)
            except Exception:
                pass
        return item_ids

    def _find_ellipse_handle_hit(
        self, lm: str, screen_x: float, screen_y: float
    ) -> Optional[str]:
        axes = self._ellipse_axis_points(self._get_ellipse(lm))
        if axes is None:
            return None

        _center, major0, major1, minor0, minor1 = axes
        handles = {
            "major_start": major0,
            "major_end": major1,
            "minor_start": minor0,
            "minor_end": minor1,
        }
        best_name: Optional[str] = None
        best_dist2 = float("inf")
        tol2 = float(self.drag_tolerance_px * self.drag_tolerance_px)

        for name, point in handles.items():
            xs, ys = self._img_to_screen(*point)
            d2 = (xs - screen_x) ** 2 + (ys - screen_y) ** 2
            if d2 <= tol2 and d2 < best_dist2:
                best_name = name
                best_dist2 = d2

        return best_name

    def _is_ellipse_hit(self, lm: str, screen_x: float, screen_y: float) -> bool:
        ellipse = self._get_ellipse(lm)
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return False

        center, major0, major1, minor0, minor1 = axes
        cx, cy = self._img_to_screen(*center)
        if (cx - screen_x) ** 2 + (cy - screen_y) ** 2 <= float(
            self.drag_tolerance_px * self.drag_tolerance_px
        ):
            return True

        if self._projected_hemisphere_body_hit(lm, screen_x, screen_y):
            return True

        for start, end in ((major0, major1), (minor0, minor1)):
            x1, y1 = self._img_to_screen(*start)
            x2, y2 = self._img_to_screen(*end)
            if (
                self._point_to_segment_distance_px(screen_x, screen_y, x1, y1, x2, y2)
                <= float(self.drag_line_tolerance_px)
            ):
                return True

        screen_points = [
            self._img_to_screen(x, y)
            for x, y in self._ellipse_polyline_points(ellipse)
        ]
        for idx in range(len(screen_points) - 1):
            x1, y1 = screen_points[idx]
            x2, y2 = screen_points[idx + 1]
            if (
                self._point_to_segment_distance_px(screen_x, screen_y, x1, y1, x2, y2)
                <= float(self.drag_line_tolerance_px)
            ):
                return True

        return False

    def _projected_hemisphere_body_hit(
        self, lm: str, screen_x: float, screen_y: float
    ) -> bool:
        if not self._ellipse_is_projected_hemisphere(lm):
            return False
        circle = self._projected_hemisphere_circle_for_ellipse(lm)
        if circle is None:
            return False
        center, radius = circle
        try:
            px, py = self._screen_to_img(screen_x, screen_y)
        except Exception:
            return False
        return math.hypot(px - center[0], py - center[1]) <= radius

    def _fac_construction_mode(self) -> str:
        var = self.__dict__.get("fac_construction_mode")
        mode = str(var.get()).strip() if var is not None else ""
        if mode in FAC_CONSTRUCTION_MODES:
            return mode
        return FAC_CONSTRUCTION_SNAPPED

    def _ellipse_construction(
        self, lm: str, ellipse: object | None = None
    ) -> str | None:
        value = ellipse if ellipse is not None else self._get_ellipse(lm)
        normalized = self._normalize_ellipse_value(value)
        if normalized is None:
            return None
        construction = normalized.get("construction")
        return construction if isinstance(construction, str) else None

    def _ellipse_is_projected_hemisphere(
        self, lm: str, ellipse: object | None = None
    ) -> bool:
        return self._ellipse_construction(lm, ellipse) == FAC_CONSTRUCTION_HEMISPHERE

    def _projected_hemisphere_circle_for_ellipse(
        self, lm: str, ellipse: object | None = None
    ) -> Tuple[AnnotationPoint, float] | None:
        value = ellipse if ellipse is not None else self._get_ellipse(lm)
        axes = self._ellipse_axis_points(value)
        if axes is None:
            return None

        center, major0, major1, _minor0, _minor1 = axes
        radius = math.hypot(major1[0] - major0[0], major1[1] - major0[1]) / 2.0
        if radius <= 0:
            return None
        return center, radius

    def _sync_projected_hemisphere_ac_for_ellipse(
        self, lm: str, ellipse: object | None = None
    ) -> bool:
        ac_lm = self._matching_ac_landmark_for_ellipse(lm)
        circle = self._projected_hemisphere_circle_for_ellipse(lm, ellipse)
        ann_key = self._current_annotation_key()
        if ac_lm is None or circle is None or ann_key is None:
            return False
        if self.landmarks and ac_lm not in self.landmarks:
            return False

        center, radius = circle
        center_point = (round(float(center[0]), 1), round(float(center[1]), 1))
        radius_value = round(float(radius), 1)
        pts = self.annotations.setdefault(ann_key, {})
        changed = False
        if pts.get(ac_lm) != center_point:
            pts[ac_lm] = center_point
            changed = True
        radii = getattr(self, "hover_radii", {}).setdefault(ann_key, {})
        if radii.get(ac_lm) != radius_value:
            radii[ac_lm] = radius_value
            changed = True
        found_var = self.landmark_found.get(ac_lm)
        if found_var is not None:
            found_var.set(True)
        return changed

    def _projected_hemisphere_radius_from_control(self) -> float:
        radius_var = self.__dict__.get("projected_hemisphere_z")
        try:
            screen_radius = float(radius_var.get()) if radius_var is not None else 70.0
        except (TypeError, ValueError):
            screen_radius = 70.0
        scale = self.disp_scale or 1.0
        return max(2.0, screen_radius / scale)

    def _set_projected_hemisphere_from_center(
        self, lm: str, center: AnnotationPoint
    ) -> bool:
        cx, cy = self._clamp_img_point(center[0], center[1])
        radius = self._projected_hemisphere_radius_from_control()
        major_vec = self._limit_axis_vector_to_image((cx, cy), (radius, 0.0))
        radius = math.hypot(*major_vec)
        if radius < 1e-12:
            return False
        minor_vec = self._limit_axis_vector_to_image((cx, cy), (0.0, radius * 0.5))
        if math.hypot(*minor_vec) < 1e-12:
            return False
        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                (cx, cy),
                major_vec,
                minor_vec,
                FAC_CONSTRUCTION_HEMISPHERE,
            ),
            FAC_CONSTRUCTION_HEMISPHERE,
        )
        return True

    def _set_projected_hemisphere_radius_for_ellipse(
        self, lm: str, radius: float
    ) -> bool:
        axes = self._ellipse_axis_points(self._get_ellipse(lm))
        if axes is None:
            return False

        center, major0, major1, minor0, minor1 = axes
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        major_len = math.hypot(*major_vec)
        if major_len < 1e-12:
            major_vec = (float(radius), 0.0)
        else:
            major_vec = (
                major_vec[0] / major_len * float(radius),
                major_vec[1] / major_len * float(radius),
            )
        major_vec = self._limit_axis_vector_to_image(center, major_vec)
        new_major_len = math.hypot(*major_vec)
        if new_major_len < 1e-12:
            return False

        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        minor_len = math.hypot(*minor_vec)
        ratio = 0.5 if major_len < 1e-12 else minor_len / major_len
        ratio = max(0.05, min(0.98, ratio))
        reference_vec = minor_vec if minor_len >= 1e-12 else (-major_vec[1], major_vec[0])
        scaled_minor = self._perpendicular_vector_with_length(
            major_vec,
            new_major_len * ratio,
            reference_vec,
        )
        if scaled_minor is None:
            return False
        scaled_minor = self._limit_axis_vector_to_image(center, scaled_minor)
        if math.hypot(*scaled_minor) < 1e-12:
            return False

        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                center,
                major_vec,
                scaled_minor,
                FAC_CONSTRUCTION_HEMISPHERE,
            ),
            FAC_CONSTRUCTION_HEMISPHERE,
        )
        return True

    def _set_projected_hemisphere_axes(
        self,
        lm: str,
        center: AnnotationPoint,
        major_vec: AnnotationPoint,
        minor_len: float,
        reference_minor_vec: AnnotationPoint,
        *,
        sync_dependents: bool = True,
    ) -> bool:
        major_len = math.hypot(*major_vec)
        if major_len < 1e-12:
            return False

        minor_len = max(2.0, min(major_len * 0.98, float(minor_len)))
        minor_vec = self._perpendicular_vector_with_length(
            major_vec,
            minor_len,
            reference_minor_vec,
        )
        if minor_vec is None:
            return False
        minor_vec = self._limit_axis_vector_to_image(center, minor_vec)
        if math.hypot(*minor_vec) < 1e-12:
            return False

        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                center,
                major_vec,
                minor_vec,
                FAC_CONSTRUCTION_HEMISPHERE,
            ),
            FAC_CONSTRUCTION_HEMISPHERE,
            sync_dependents=sync_dependents,
        )
        return True

    def _move_projected_hemisphere_handle(
        self,
        lm: str,
        handle: str,
        new_point: AnnotationPoint,
        *,
        sync_dependents: bool = True,
    ) -> bool:
        axes = self._ellipse_axis_points(self._get_ellipse(lm))
        if axes is None:
            return False

        center, major0, major1, minor0, minor1 = axes
        cx, cy = center
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        major_len = math.hypot(*major_vec)
        minor_len = math.hypot(*minor_vec)
        min_axis = max(2.0, 4.0 / (self.disp_scale or 1.0))
        if major_len < min_axis:
            return False

        if handle in {"major_start", "major_end"}:
            dx = new_point[0] - cx
            dy = new_point[1] - cy
            pointer_len = math.hypot(dx, dy)
            if pointer_len < 1e-12:
                return False
            if handle == "major_start":
                dx = -dx
                dy = -dy
            rotated_major = (
                dx / pointer_len * major_len,
                dy / pointer_len * major_len,
            )
            return self._set_projected_hemisphere_axes(
                lm,
                center,
                rotated_major,
                max(min_axis, minor_len),
                minor_vec,
                sync_dependents=sync_dependents,
            )

        if handle not in {"minor_start", "minor_end"}:
            return False

        projected_minor = self._minor_vector_with_locked_major_axis(
            handle,
            new_point,
            center,
            major_vec,
            minor_vec,
            min_axis,
        )
        if projected_minor is None:
            return False
        return self._set_projected_hemisphere_axes(
            lm,
            center,
            major_vec,
            math.hypot(*projected_minor),
            projected_minor,
            sync_dependents=sync_dependents,
        )

    def _change_projected_hemisphere_inclination(self, delta_degrees: float) -> bool:
        ellipse_lm = self._active_projected_hemisphere_ellipse_for_size()
        if ellipse_lm is None:
            return False
        axes = self._ellipse_axis_points(self._get_ellipse(ellipse_lm))
        if axes is None:
            return False

        center, major0, major1, minor0, minor1 = axes
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        radians = math.radians(float(delta_degrees))
        cos_t = math.cos(radians)
        sin_t = math.sin(radians)
        rotated_major = (
            major_vec[0] * cos_t - major_vec[1] * sin_t,
            major_vec[0] * sin_t + major_vec[1] * cos_t,
        )
        return self._set_projected_hemisphere_axes(
            ellipse_lm,
            center,
            rotated_major,
            math.hypot(*minor_vec),
            minor_vec,
        )

    def _change_projected_hemisphere_anteversion(self, delta_degrees: float) -> bool:
        ellipse_lm = self._active_projected_hemisphere_ellipse_for_size()
        if ellipse_lm is None:
            return False
        axes = self._ellipse_axis_points(self._get_ellipse(ellipse_lm))
        if axes is None:
            return False

        center, major0, major1, minor0, minor1 = axes
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        major_len = math.hypot(*major_vec)
        minor_len = math.hypot(*minor_vec)
        if major_len < 1e-12:
            return False
        current_degrees = math.degrees(
            math.asin(max(0.0, min(0.98, minor_len / major_len)))
        )
        next_degrees = max(1.0, min(78.0, current_degrees + delta_degrees))
        next_minor_len = major_len * math.sin(math.radians(next_degrees))
        return self._set_projected_hemisphere_axes(
            ellipse_lm,
            center,
            major_vec,
            next_minor_len,
            minor_vec,
        )

    def _apply_projected_hemisphere_adjustment(
        self, kind: str, delta_degrees: float
    ) -> bool:
        if kind == "inclination":
            changed = self._change_projected_hemisphere_inclination(delta_degrees)
        elif kind == "anteversion":
            changed = self._change_projected_hemisphere_anteversion(delta_degrees)
        else:
            changed = False
        if not changed:
            return False
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._autosave_annotation_change()
        return True

    def _set_projected_hemisphere_from_ac(self, ac_lm: str) -> bool:
        ellipse_lm = self._matching_ellipse_landmark_for_ac(ac_lm)
        ann_key = self._current_annotation_key()
        if ellipse_lm is None or ann_key is None:
            return False
        if not self._ellipse_is_projected_hemisphere(ellipse_lm):
            return False

        pts, _quality = self._get_annotations()
        center = self._landmark_center_from_value(ac_lm, pts.get(ac_lm))
        if center is None:
            return False
        radius = getattr(self, "hover_radii", {}).get(ann_key, {}).get(ac_lm)
        try:
            radius_float = float(radius)
        except (TypeError, ValueError):
            circle = self._projected_hemisphere_circle_for_ellipse(ellipse_lm)
            if circle is None:
                return False
            radius_float = circle[1]
        if radius_float <= 0:
            return False

        axes = self._ellipse_axis_points(self._get_ellipse(ellipse_lm))
        if axes is None:
            return False
        _old_center, major0, major1, minor0, minor1 = axes
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        major_len = math.hypot(*major_vec)
        if major_len < 1e-12:
            major_vec = (radius_float, 0.0)
        else:
            major_vec = (
                major_vec[0] / major_len * radius_float,
                major_vec[1] / major_len * radius_float,
            )

        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        minor_len = math.hypot(*minor_vec)
        minor_ratio = 0.5 if major_len < 1e-12 else max(0.05, min(0.98, minor_len / major_len))
        minor_vec = self._perpendicular_vector_with_length(
            major_vec,
            radius_float * minor_ratio,
            minor_vec,
        )
        if minor_vec is None:
            return False
        self._set_ellipse(
            ellipse_lm,
            self._ellipse_from_center_vectors(
                center,
                major_vec,
                minor_vec,
                FAC_CONSTRUCTION_HEMISPHERE,
            ),
            FAC_CONSTRUCTION_HEMISPHERE,
        )
        return True

    def _fac_landmark_candidates_for_current_view(self) -> List[str]:
        allowed = self._get_allowed_landmarks_for_current_view()
        source = self.landmarks or list(ELLIPSE_LANDMARKS)
        return [
            lm
            for lm in ("L-C-FAC", "R-C-FAC", "L-FAC", "R-FAC")
            if lm in source and (not allowed or lm in allowed)
        ]

    def _projected_hemisphere_placement_landmark(
        self, *, prefer_unset: bool = True
    ) -> str | None:
        candidates = self._fac_landmark_candidates_for_current_view()
        if not candidates:
            return None

        selected_var = self.__dict__.get("selected_landmark")
        current = str(selected_var.get()).strip() if selected_var is not None else ""
        if current in candidates:
            return current

        pts, _quality = self._get_annotations()
        if prefer_unset:
            for lm in candidates:
                if self._get_ellipse(lm) is None and lm not in pts:
                    return lm
        return candidates[0]

    def _select_projected_hemisphere_placement_landmark(self) -> str | None:
        lm = self._projected_hemisphere_placement_landmark(prefer_unset=True)
        selected_var = self.__dict__.get("selected_landmark")
        if lm is not None and selected_var is not None:
            selected_var.set(lm)
        return lm

    def _sync_projected_hemisphere_z_control(self, lm: str) -> None:
        if not self._ellipse_is_projected_hemisphere(lm):
            return
        circle = self._projected_hemisphere_circle_for_ellipse(lm)
        z_var = self.__dict__.get("projected_hemisphere_z")
        if circle is None or z_var is None:
            return
        _center, radius = circle
        z_var.set(max(5, min(300, int(round(radius * (self.disp_scale or 1.0))))))

    def _projected_hemisphere_default_center(self) -> AnnotationPoint | None:
        if self.current_image is None:
            return None

        mouse_pos = self.__dict__.get("last_mouse_canvas_pos")
        if mouse_pos is not None:
            try:
                mouse_x, mouse_y = mouse_pos
                x0, y0, x1, y1 = self._display_rect()
                if x0 <= mouse_x < x1 and y0 <= mouse_y < y1:
                    xi, yi = self._screen_to_img(mouse_x, mouse_y)
                    return self._clamp_img_point(xi, yi)
            except Exception:
                pass

        try:
            width, height = self.current_image.size
        except Exception:
            return None
        return self._clamp_img_point(float(width) / 2.0, float(height) / 2.0)

    def _project_selected_fac_as_hemisphere(self) -> bool:
        if getattr(self, "multi_review_mode", False):
            return False

        lm = self._projected_hemisphere_placement_landmark(prefer_unset=False)
        if lm is None:
            self._set_status("Select an acetabular face contour before projecting.")
            return False
        self.selected_landmark.set(lm)

        ellipse = self._get_ellipse(lm)
        if ellipse is None:
            circle = self._get_ac_hover_circle_for_ellipse(lm)
            if circle is None:
                center = self._projected_hemisphere_default_center()
                if center is None:
                    self._set_status(f"{lm}: load an image before projecting.")
                    return False
                if not self._set_projected_hemisphere_from_center(lm, center):
                    self._set_status(f"{lm}: projected hemisphere could not be placed.")
                    return False
                self._draw_points()
                self._refresh_zoom_landmark_overlay()
                self.dirty = True
                self._autosave_annotation_change()
                self._set_status(f"{lm}: projected hemisphere construction applied.")
                return True
            center, radius = circle
            major_vec = (radius, 0.0)
            minor_vec = (0.0, radius * 0.5)
        else:
            axes = self._ellipse_axis_points(ellipse)
            if axes is None:
                return False
            center, major0, major1, minor0, minor1 = axes
            major_vec = (
                (major1[0] - major0[0]) / 2.0,
                (major1[1] - major0[1]) / 2.0,
            )
            minor_vec = (
                (minor1[0] - minor0[0]) / 2.0,
                (minor1[1] - minor0[1]) / 2.0,
            )

        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                center,
                major_vec,
                minor_vec,
                FAC_CONSTRUCTION_HEMISPHERE,
            ),
            FAC_CONSTRUCTION_HEMISPHERE,
        )
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._autosave_annotation_change()
        self._set_status(f"{lm}: projected hemisphere construction applied.")
        return True

    def _projected_hemisphere_arc_points(
        self,
        lm: str,
        *,
        visible: bool,
        ellipse: object | None = None,
        steps: int = 72,
    ) -> List[AnnotationPoint]:
        value = ellipse if ellipse is not None else self._get_ellipse(lm)
        axes = self._ellipse_axis_points(value)
        if axes is None:
            return []

        center, major0, major1, minor0, minor1 = axes
        cx, cy = center
        ux = (major1[0] - major0[0]) / 2.0
        uy = (major1[1] - major0[1]) / 2.0
        radius = math.hypot(ux, uy)
        if radius < 1e-12:
            return []
        ux /= radius
        uy /= radius

        vx = (minor1[0] - minor0[0]) / 2.0
        vy = (minor1[1] - minor0[1]) / 2.0
        perp_x = -uy
        perp_y = ux
        if (perp_x * vx + perp_y * vy) < 0:
            perp_x = -perp_x
            perp_y = -perp_y
        if (cy + perp_y * radius) > (cy - perp_y * radius):
            perp_x = -perp_x
            perp_y = -perp_y
        if not visible:
            perp_x = -perp_x
            perp_y = -perp_y

        count = max(8, int(steps))
        return [
            (
                cx + radius * (ux * math.cos(t) + perp_x * math.sin(t)),
                cy + radius * (uy * math.cos(t) + perp_y * math.sin(t)),
            )
            for t in (math.pi * idx / count for idx in range(count + 1))
        ]

    def _draw_projected_hemisphere_boundary(
        self,
        canvas,
        ellipse_lm: str,
        img_to_canvas,
        *,
        ellipse: object | None = None,
        width: int,
        tags: str = "marker",
    ) -> List[int]:
        value = ellipse if ellipse is not None else self._get_ellipse(ellipse_lm)
        if not self._ellipse_is_projected_hemisphere(ellipse_lm, value):
            return []

        item_ids: List[int] = []
        for visible, dash in ((True, None), (False, (5, 4))):
            arc_points = self._projected_hemisphere_arc_points(
                ellipse_lm,
                visible=visible,
                ellipse=value,
            )
            if len(arc_points) < 2:
                continue
            screen_points = [img_to_canvas(px, py) for px, py in arc_points]
            flat = [coord for point in screen_points for coord in (point[0], point[1])]
            options: Dict[str, object] = {
                "fill": PROJECTED_HEMISPHERE_BOUNDARY_COLOR,
                "width": max(2, int(width)),
                "smooth": True,
                "tags": tags,
            }
            if dash is not None:
                options["dash"] = dash
            item_ids.append(canvas.create_line(*flat, **options))

        for item_id in item_ids:
            try:
                canvas.tag_raise(item_id)
            except Exception:
                pass
        return item_ids

    def _ellipse_snap_enabled(self) -> bool:
        var = self.__dict__.get("ellipse_snap_to_ac_enabled")
        return bool(var.get()) if var is not None else False

    def _ellipse_lock_major_when_snapped_enabled(self) -> bool:
        var = self.__dict__.get("ellipse_lock_major_when_snapped_enabled")
        return bool(var.get()) if var is not None else False

    def _matching_ac_landmark_for_ellipse(self, lm: str) -> str | None:
        if lm == "L-C-FAC":
            return "L-C-AC"
        if lm == "R-C-FAC":
            return "R-C-AC"
        if lm == "L-FAC":
            return "L-AC"
        if lm == "R-FAC":
            return "R-AC"
        return None

    def _matching_ellipse_landmark_for_ac(self, lm: str) -> str | None:
        if lm == "L-C-AC":
            return "L-C-FAC"
        if lm == "R-C-AC":
            return "R-C-FAC"
        if lm == "L-AC":
            return "L-FAC"
        if lm == "R-AC":
            return "R-FAC"
        return None

    def _derived_acetabular_landmarks_for_ellipse(
        self, lm: str
    ) -> Tuple[str, str] | None:
        if lm == "L-C-FAC":
            return "L-DAB", "L-SAB"
        if lm == "R-C-FAC":
            return "R-DAB", "R-SAB"
        if lm == "L-FAC":
            return "L-A-DAB", "L-A-SAB"
        if lm == "R-FAC":
            return "R-A-DAB", "R-A-SAB"
        return None

    def _intermediate_acetabular_context_for_landmark(
        self, lm: str
    ) -> Dict[str, str] | None:
        if lm.startswith("L-A-") and lm in INTERMEDIATE_ACETABULAR_LANDMARKS:
            return {
                "side": "L",
                "ellipse": "L-FAC",
                "dab": "L-A-DAB",
                "sab": "L-A-SAB",
                "idab": "L-A-IDAB",
                "isab": "L-A-ISAB",
            }
        if lm.startswith("R-A-") and lm in INTERMEDIATE_ACETABULAR_LANDMARKS:
            return {
                "side": "R",
                "ellipse": "R-FAC",
                "dab": "R-A-DAB",
                "sab": "R-A-SAB",
                "idab": "R-A-IDAB",
                "isab": "R-A-ISAB",
            }
        return None

    def _intermediate_acetabular_landmarks_for_ellipse(
        self, lm: str
    ) -> Tuple[str, str] | None:
        if lm == "L-FAC":
            return "L-A-IDAB", "L-A-ISAB"
        if lm == "R-FAC":
            return "R-A-IDAB", "R-A-ISAB"
        return None

    def _clear_intermediate_acetabular_points_for_ellipse(self, lm: str) -> bool:
        intermediate = self._intermediate_acetabular_landmarks_for_ellipse(lm)
        ann_key = self._current_annotation_key()
        if intermediate is None or ann_key is None:
            return False
        pts = self.annotations.get(ann_key)
        if pts is None:
            return False

        changed = False
        for inter_lm in intermediate:
            if inter_lm in pts:
                del pts[inter_lm]
                changed = True
            found_var = self.landmark_found.get(inter_lm)
            if found_var is not None:
                found_var.set(False)
        return changed

    def _intermediate_acetabular_landmark_ready(self, lm: str) -> bool:
        ctx = self._intermediate_acetabular_context_for_landmark(lm)
        if ctx is None:
            return False
        if lm not in self.landmarks or not self._current_view_allows_landmark(lm):
            return False

        pts, _quality = self._get_annotations()
        if ctx["dab"] not in pts or ctx["sab"] not in pts:
            return False

        axes = self._ellipse_axis_points(self._get_ellipse(ctx["ellipse"]))
        if axes is None:
            return False
        return self._ellipse_has_acetabular_endpoint_geometry(ctx["ellipse"], axes)

    def _ellipse_lower_arc_samples(
        self, ellipse_lm: str, steps: int = 160
    ) -> List[AnnotationPoint]:
        axes = self._ellipse_axis_points(self._get_ellipse(ellipse_lm))
        if axes is None:
            return []

        center, major0, major1, minor0, minor1 = axes
        cx, cy = center
        ux = (major1[0] - major0[0]) / 2.0
        uy = (major1[1] - major0[1]) / 2.0
        vx = (minor1[0] - minor0[0]) / 2.0
        vy = (minor1[1] - minor0[1]) / 2.0

        plus_v_y = cy + vy
        minus_v_y = cy - vy
        if plus_v_y >= minus_v_y:
            angles = [math.pi * i / steps for i in range(steps + 1)]
        else:
            angles = [math.pi + math.pi * i / steps for i in range(steps + 1)]

        samples = [
            (
                cx + ux * math.cos(t) + vx * math.sin(t),
                cy + uy * math.cos(t) + vy * math.sin(t),
            )
            for t in angles
        ]

        pts, _quality = self._get_annotations()
        derived = self._derived_acetabular_landmarks_for_ellipse(ellipse_lm)
        if derived is None:
            return samples
        _dab_lm, sab_lm = derived
        sab = self._landmark_center_from_value(sab_lm, pts.get(sab_lm))
        if sab is not None and samples:
            first_dist = math.hypot(samples[0][0] - sab[0], samples[0][1] - sab[1])
            last_dist = math.hypot(samples[-1][0] - sab[0], samples[-1][1] - sab[1])
            if last_dist < first_dist:
                samples.reverse()
        return samples

    def _point_on_lower_arc_at_ratio(
        self, ellipse_lm: str, ratio: float
    ) -> AnnotationPoint | None:
        samples = self._ellipse_lower_arc_samples(ellipse_lm)
        if not samples:
            return None
        ratio = max(0.0, min(1.0, float(ratio)))
        idx_float = ratio * (len(samples) - 1)
        lo = int(math.floor(idx_float))
        hi = min(len(samples) - 1, lo + 1)
        frac = idx_float - lo
        x = samples[lo][0] + (samples[hi][0] - samples[lo][0]) * frac
        y = samples[lo][1] + (samples[hi][1] - samples[lo][1]) * frac
        return round(float(x), 1), round(float(y), 1)

    def _lower_arc_ratio_for_point(
        self, ellipse_lm: str, point: AnnotationPoint
    ) -> float | None:
        samples = self._ellipse_lower_arc_samples(ellipse_lm)
        if not samples:
            return None
        best_idx = 0
        best_dist2 = float("inf")
        px, py = point
        for idx, (sx, sy) in enumerate(samples):
            dist2 = (sx - px) ** 2 + (sy - py) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = idx
        return best_idx / max(1, len(samples) - 1)

    def _constrained_intermediate_acetabular_point(
        self, lm: str, point: AnnotationPoint
    ) -> AnnotationPoint | None:
        ctx = self._intermediate_acetabular_context_for_landmark(lm)
        if ctx is None or not self._intermediate_acetabular_landmark_ready(lm):
            return None

        ratio = self._lower_arc_ratio_for_point(ctx["ellipse"], point)
        if ratio is None:
            return None

        margin = 0.02
        min_gap = 0.03
        ratio = max(margin, min(1.0 - margin, ratio))

        pts, _quality = self._get_annotations()
        if lm == ctx["isab"]:
            idab = self._landmark_center_from_value(ctx["idab"], pts.get(ctx["idab"]))
            if idab is not None:
                idab_ratio = self._lower_arc_ratio_for_point(ctx["ellipse"], idab)
                if idab_ratio is not None:
                    ratio = min(ratio, idab_ratio - min_gap)
        elif lm == ctx["idab"]:
            isab = self._landmark_center_from_value(ctx["isab"], pts.get(ctx["isab"]))
            if isab is not None:
                isab_ratio = self._lower_arc_ratio_for_point(ctx["ellipse"], isab)
                if isab_ratio is not None:
                    ratio = max(ratio, isab_ratio + min_gap)

        ratio = max(margin, min(1.0 - margin, ratio))
        return self._point_on_lower_arc_at_ratio(ctx["ellipse"], ratio)

    def _set_intermediate_acetabular_point(
        self, lm: str, point: AnnotationPoint
    ) -> bool:
        constrained = self._constrained_intermediate_acetabular_point(lm, point)
        ann_key = self._current_annotation_key()
        if constrained is None or ann_key is None:
            self._set_status(self._acetabular_geometry_required_message(lm))
            return False
        pts = self.annotations.setdefault(ann_key, {})
        pts[lm] = constrained
        found_var = self.landmark_found.get(lm)
        if found_var is not None:
            found_var.set(True)
        return True

    def _reproject_intermediate_acetabular_points_for_ellipse(self, lm: str) -> bool:
        intermediate = self._intermediate_acetabular_landmarks_for_ellipse(lm)
        ann_key = self._current_annotation_key()
        if intermediate is None or ann_key is None:
            return False
        pts = self.annotations.get(ann_key)
        if pts is None:
            return False

        if not all(
            self._intermediate_acetabular_landmark_ready(inter_lm)
            for inter_lm in intermediate
        ):
            return self._clear_intermediate_acetabular_points_for_ellipse(lm)

        changed = False
        # Reproject lateral point first, then medial point so the order clamp is stable.
        for inter_lm in (intermediate[1], intermediate[0]):
            existing = self._landmark_center_from_value(inter_lm, pts.get(inter_lm))
            if existing is None:
                continue
            constrained = self._constrained_intermediate_acetabular_point(
                inter_lm, existing
            )
            if constrained is None:
                continue
            if pts.get(inter_lm) != constrained:
                pts[inter_lm] = constrained
                changed = True
            found_var = self.landmark_found.get(inter_lm)
            if found_var is not None:
                found_var.set(True)
        return changed

    def _current_view_allows_landmark(self, lm: str) -> bool:
        allowed = self._get_allowed_landmarks_for_current_view()
        return not allowed or lm in allowed

    def _clear_auto_acetabular_points_for_ellipse(self, lm: str) -> bool:
        derived = self._derived_acetabular_landmarks_for_ellipse(lm)
        ann_key = self._current_annotation_key()
        if derived is None or ann_key is None:
            return False
        pts = self.annotations.get(ann_key)
        if pts is None:
            return False

        changed = False
        for derived_lm in derived:
            if derived_lm in pts:
                del pts[derived_lm]
                changed = True
            found_var = self.landmark_found.get(derived_lm)
            if found_var is not None:
                found_var.set(False)
        if self._clear_intermediate_acetabular_points_for_ellipse(lm):
            changed = True
        return changed

    def _clear_auto_acetabular_points_for_ac(self, lm: str) -> bool:
        ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
        if ellipse_lm is None:
            return False
        return self._clear_auto_acetabular_points_for_ellipse(ellipse_lm)

    def _clear_projected_hemisphere_for_ac(self, lm: str) -> bool:
        ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
        ann_key = self._current_annotation_key()
        if ellipse_lm is None or ann_key is None:
            return False
        if not self._ellipse_is_projected_hemisphere(ellipse_lm):
            return False
        pts = self.annotations.get(ann_key)
        if pts is None:
            return False

        changed = False
        if ellipse_lm in pts:
            del pts[ellipse_lm]
            changed = True
        self.landmark_meta.get(ann_key, {}).pop(ellipse_lm, None)
        found_var = self.landmark_found.get(ellipse_lm)
        if found_var is not None:
            found_var.set(False)
        if self._clear_auto_acetabular_points_for_ellipse(ellipse_lm):
            changed = True
        return changed

    def _clear_projected_hemisphere_ac_for_ellipse(self, lm: str) -> bool:
        ac_lm = self._matching_ac_landmark_for_ellipse(lm)
        ann_key = self._current_annotation_key()
        if ac_lm is None or ann_key is None:
            return False
        pts = self.annotations.get(ann_key)
        if pts is None:
            return False

        changed = False
        if ac_lm in pts:
            del pts[ac_lm]
            changed = True
        self.landmark_meta.get(ann_key, {}).pop(ac_lm, None)
        radii = getattr(self, "hover_radii", {}).get(ann_key, {})
        if ac_lm in radii:
            del radii[ac_lm]
            changed = True
        found_var = self.landmark_found.get(ac_lm)
        if found_var is not None:
            found_var.set(False)
        return changed

    def _update_auto_acetabular_points_for_ac(self, lm: str) -> bool:
        ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
        if ellipse_lm is None:
            return False
        return self._update_auto_acetabular_points_for_ellipse(ellipse_lm)

    def _update_auto_acetabular_points_for_current_image(
        self, *, snap_if_enabled: bool = False
    ) -> bool:
        changed = False
        for lm in ("L-C-FAC", "R-C-FAC", "L-FAC", "R-FAC"):
            if lm not in self.landmarks:
                continue
            if (
                snap_if_enabled
                and self._ellipse_snap_enabled()
                and not self._ellipse_is_projected_hemisphere(lm)
            ):
                before = self._get_ellipse(lm)
                snapped = self._snap_ellipse_major_axis_to_ac_circle(lm)
                after = self._get_ellipse(lm)
                if snapped and before != after:
                    changed = True
            if self._update_auto_acetabular_points_for_ellipse(lm):
                changed = True
        return changed

    def _update_auto_acetabular_points_for_ellipse(self, lm: str) -> bool:
        derived = self._derived_acetabular_landmarks_for_ellipse(lm)
        axes = self._ellipse_axis_points(self._get_ellipse(lm))
        ann_key = self._current_annotation_key()
        if derived is None or ann_key is None:
            return False

        dab_lm, sab_lm = derived
        derived_available = all(
            derived_lm in self.landmarks
            and self._current_view_allows_landmark(derived_lm)
            for derived_lm in derived
        )
        if not derived_available:
            return self._clear_auto_acetabular_points_for_ellipse(lm)

        if axes is None or not self._ellipse_has_acetabular_endpoint_geometry(lm, axes):
            return self._clear_auto_acetabular_points_for_ellipse(lm)

        changed = False
        if self._ellipse_is_projected_hemisphere(lm):
            if self._sync_projected_hemisphere_ac_for_ellipse(lm):
                changed = True

        _center, major0, major1, _minor0, _minor1 = axes
        endpoints = [major0, major1]
        side = self._protocol_side_for_landmark(lm) or "L"
        medial = min(
            endpoints,
            key=lambda point: self._side_lateral_score(side, point[0]),
        )
        lateral = max(
            endpoints,
            key=lambda point: self._side_lateral_score(side, point[0]),
        )

        updates = {
            dab_lm: (round(float(medial[0]), 1), round(float(medial[1]), 1)),
            sab_lm: (round(float(lateral[0]), 1), round(float(lateral[1]), 1)),
        }
        pts = self.annotations.setdefault(ann_key, {})
        for derived_lm, point in updates.items():
            if pts.get(derived_lm) != point:
                pts[derived_lm] = point
                changed = True
            found_var = self.landmark_found.get(derived_lm)
            if found_var is not None:
                found_var.set(True)
        if self._reproject_intermediate_acetabular_points_for_ellipse(lm):
            changed = True
        return changed

    def _get_ac_hover_circle_for_ellipse(
        self, lm: str
    ) -> Tuple[AnnotationPoint, float] | None:
        if self.current_image_path is None:
            return None

        ac_lm = self._matching_ac_landmark_for_ellipse(lm)
        if ac_lm is None:
            return None

        pts, _quality = self._get_annotations()
        ac_value = pts.get(ac_lm)
        center = self._landmark_center_from_value(ac_lm, ac_value)
        if center is None:
            return None

        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None
            else str(self.current_image_path)
        )
        radius = getattr(self, "hover_radii", {}).get(key, {}).get(ac_lm)
        if radius is None:
            return None

        try:
            radius_float = float(radius)
        except (TypeError, ValueError):
            return None

        if radius_float <= 0:
            return None

        return center, radius_float

    def _nearest_point_on_circle(
        self,
        point: AnnotationPoint,
        center: AnnotationPoint,
        radius: float,
        fallback_direction: AnnotationPoint = (1.0, 0.0),
    ) -> AnnotationPoint:
        cx, cy = center
        dx = point[0] - cx
        dy = point[1] - cy
        dist = math.hypot(dx, dy)

        if dist < 1e-12:
            dx, dy = fallback_direction
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                dx, dy = 1.0, 0.0
                dist = 1.0

        return cx + radius * dx / dist, cy + radius * dy / dist

    def _snap_point_to_ac_circle(
        self,
        lm: str,
        point: AnnotationPoint,
        fallback_direction: AnnotationPoint = (1.0, 0.0),
    ) -> AnnotationPoint | None:
        circle = self._get_ac_hover_circle_for_ellipse(lm)
        if circle is None:
            return None
        center, radius = circle
        return self._nearest_point_on_circle(point, center, radius, fallback_direction)

    def _ellipse_major_axis_is_snapped_to_ac_circle(
        self, lm: str, axes=None
    ) -> bool:
        if axes is None:
            axes = self._ellipse_axis_points(self._get_ellipse(lm))
        circle = self._get_ac_hover_circle_for_ellipse(lm)
        if axes is None or circle is None:
            return False

        _center, major0, major1, _minor0, _minor1 = axes
        circle_center, radius = circle
        tol = max(1.5, 3.0 / (self.disp_scale or 1.0))

        for point in (major0, major1):
            dist = math.hypot(point[0] - circle_center[0], point[1] - circle_center[1])
            if abs(dist - radius) > tol:
                return False

        return True

    def _ellipse_has_acetabular_endpoint_geometry(self, lm: str, axes=None) -> bool:
        if axes is None:
            axes = self._ellipse_axis_points(self._get_ellipse(lm))
        if axes is None:
            return False
        if self._ellipse_is_projected_hemisphere(lm):
            return True
        return self._ellipse_major_axis_is_snapped_to_ac_circle(lm, axes)

    def _opposite_ac_circle_point(
        self, lm: str, fixed_point: AnnotationPoint
    ) -> AnnotationPoint | None:
        circle = self._get_ac_hover_circle_for_ellipse(lm)
        if circle is None:
            return None

        center, radius = circle
        target = (
            center[0] - (fixed_point[0] - center[0]),
            center[1] - (fixed_point[1] - center[1]),
        )
        return self._nearest_point_on_circle(
            target,
            center,
            radius,
            (
                target[0] - center[0],
                target[1] - center[1],
            ),
        )

    def _axis_vector_with_fixed_endpoint(
        self,
        fixed: AnnotationPoint,
        moved: AnnotationPoint,
        fallback_vec: AnnotationPoint,
        min_axis: float,
    ) -> Tuple[AnnotationPoint, AnnotationPoint]:
        fx, fy = fixed
        mx, my = moved
        dx = mx - fx
        dy = my - fy
        full_len = math.hypot(dx, dy)
        min_full_len = max(min_axis * 2.0, 1e-6)

        if full_len < min_full_len:
            fdx, fdy = fallback_vec
            fallback_len = math.hypot(fdx, fdy)
            if fallback_len < 1e-12:
                fdx, fdy = 1.0, 0.0
                fallback_len = 1.0
            dx = fdx / fallback_len * min_full_len
            dy = fdy / fallback_len * min_full_len
            moved = (fx + dx, fy + dy)
        else:
            moved = (mx, my)

        return moved, (dx / 2.0, dy / 2.0)

    def _perpendicular_vector_with_length(
        self,
        axis_vec: AnnotationPoint,
        length: float,
        reference_vec: AnnotationPoint,
    ) -> AnnotationPoint | None:
        ax, ay = axis_vec
        axis_len = math.hypot(ax, ay)
        if axis_len < 1e-12:
            return None

        px = -ay / axis_len
        py = ax / axis_len
        if px * reference_vec[0] + py * reference_vec[1] < 0:
            px = -px
            py = -py

        return px * length, py * length

    def _minor_vector_with_locked_major_axis(
        self,
        handle: str,
        new_point: AnnotationPoint,
        center: AnnotationPoint,
        major_vec: AnnotationPoint,
        previous_minor_vec: AnnotationPoint,
        min_axis: float,
    ) -> AnnotationPoint | None:
        major_len = math.hypot(*major_vec)
        if major_len < 1e-12:
            return None

        px = -major_vec[1] / major_len
        py = major_vec[0] / major_len

        previous_signed = previous_minor_vec[0] * px + previous_minor_vec[1] * py
        if abs(previous_signed) < 1e-12:
            previous_signed = min_axis

        signed = (new_point[0] - center[0]) * px + (new_point[1] - center[1]) * py

        if handle == "minor_start":
            signed = -signed
            if abs(signed) < min_axis:
                signed = math.copysign(min_axis, previous_signed)
        else:
            if abs(signed) < min_axis:
                signed = math.copysign(min_axis, previous_signed)

        return px * signed, py * signed

    def _move_ellipse_handle(
        self,
        lm: str,
        handle: str,
        new_point: AnnotationPoint,
        *,
        sync_dependents: bool = True,
    ) -> None:
        ellipse = self._get_ellipse(lm)
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return
        construction = self._ellipse_construction(lm, ellipse)
        projected_hemisphere = construction == FAC_CONSTRUCTION_HEMISPHERE

        center, major0, major1, minor0, minor1 = axes
        previous_major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        previous_minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        previous_major_len = math.hypot(*previous_major_vec)
        previous_minor_len = math.hypot(*previous_minor_vec)
        min_axis = max(2.0, 4.0 / (self.disp_scale or 1.0))

        if projected_hemisphere:
            self._move_projected_hemisphere_handle(
                lm, handle, new_point, sync_dependents=sync_dependents
            )
            return

        if (
            handle in {"minor_start", "minor_end"}
            and (
                projected_hemisphere
                or (
                    self._ellipse_lock_major_when_snapped_enabled()
                    and self._ellipse_major_axis_is_snapped_to_ac_circle(lm, axes)
                )
            )
        ):
            locked_center = (
                (major0[0] + major1[0]) / 2.0,
                (major0[1] + major1[1]) / 2.0,
            )
            minor_vec = self._minor_vector_with_locked_major_axis(
                handle,
                new_point,
                locked_center,
                previous_major_vec,
                previous_minor_vec,
                min_axis,
            )
            if minor_vec is None:
                return
            minor_vec = self._limit_axis_vector_to_image(locked_center, minor_vec)
            if math.hypot(*minor_vec) < 1e-12:
                return
            self._set_ellipse(
                lm,
                self._ellipse_from_center_vectors(
                    locked_center,
                    previous_major_vec,
                    minor_vec,
                    construction,
                ),
                construction,
                sync_dependents=sync_dependents,
            )
            return

        if handle == "major_start":
            moved = new_point
            if self._ellipse_snap_enabled() and not projected_hemisphere:
                snapped = self._snap_point_to_ac_circle(
                    lm,
                    moved,
                    (
                        major0[0] - major1[0],
                        major0[1] - major1[1],
                    ),
                )
                if snapped is not None:
                    if math.hypot(snapped[0] - major1[0], snapped[1] - major1[1]) < (
                        min_axis * 2.0
                    ):
                        snapped = self._opposite_ac_circle_point(lm, major1) or snapped
                    moved = snapped

            moved, axis_vec = self._axis_vector_with_fixed_endpoint(
                major1,
                moved,
                (
                    major0[0] - major1[0],
                    major0[1] - major1[1],
                ),
                min_axis,
            )
            new_center = (
                (major1[0] + moved[0]) / 2.0,
                (major1[1] + moved[1]) / 2.0,
            )
            major_vec = (-axis_vec[0], -axis_vec[1])
            minor_vec = self._perpendicular_vector_with_length(
                major_vec,
                max(min_axis, previous_minor_len),
                previous_minor_vec,
            )
        elif handle == "major_end":
            moved = new_point
            if self._ellipse_snap_enabled() and not projected_hemisphere:
                snapped = self._snap_point_to_ac_circle(
                    lm,
                    moved,
                    (
                        major1[0] - major0[0],
                        major1[1] - major0[1],
                    ),
                )
                if snapped is not None:
                    if math.hypot(snapped[0] - major0[0], snapped[1] - major0[1]) < (
                        min_axis * 2.0
                    ):
                        snapped = self._opposite_ac_circle_point(lm, major0) or snapped
                    moved = snapped

            moved, major_vec = self._axis_vector_with_fixed_endpoint(
                major0,
                moved,
                (
                    major1[0] - major0[0],
                    major1[1] - major0[1],
                ),
                min_axis,
            )
            new_center = (
                (major0[0] + moved[0]) / 2.0,
                (major0[1] + moved[1]) / 2.0,
            )
            minor_vec = self._perpendicular_vector_with_length(
                major_vec,
                max(min_axis, previous_minor_len),
                previous_minor_vec,
            )
        elif handle == "minor_start":
            moved, axis_vec = self._axis_vector_with_fixed_endpoint(
                minor1,
                new_point,
                (
                    minor0[0] - minor1[0],
                    minor0[1] - minor1[1],
                ),
                min_axis,
            )
            new_center = (
                (minor1[0] + moved[0]) / 2.0,
                (minor1[1] + moved[1]) / 2.0,
            )
            minor_vec = (-axis_vec[0], -axis_vec[1])
            major_vec = self._perpendicular_vector_with_length(
                minor_vec,
                max(min_axis, previous_major_len),
                previous_major_vec,
            )
        elif handle == "minor_end":
            moved, minor_vec = self._axis_vector_with_fixed_endpoint(
                minor0,
                new_point,
                (
                    minor1[0] - minor0[0],
                    minor1[1] - minor0[1],
                ),
                min_axis,
            )
            new_center = (
                (minor0[0] + moved[0]) / 2.0,
                (minor0[1] + moved[1]) / 2.0,
            )
            major_vec = self._perpendicular_vector_with_length(
                minor_vec,
                max(min_axis, previous_major_len),
                previous_major_vec,
            )
        else:
            return

        if major_vec is None or minor_vec is None:
            return

        if handle in {"major_start", "major_end"}:
            minor_vec = self._limit_axis_vector_to_image(new_center, minor_vec)
            if math.hypot(*minor_vec) < 1e-12:
                return
        else:
            major_vec = self._limit_axis_vector_to_image(new_center, major_vec)
            if math.hypot(*major_vec) < 1e-12:
                return

        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                new_center,
                major_vec,
                minor_vec,
                construction,
            ),
            construction,
            sync_dependents=sync_dependents,
        )

    def _snap_ellipse_major_axis_to_ac_circle(self, lm: str) -> bool:
        axes = self._ellipse_axis_points(self._get_ellipse(lm))
        circle = self._get_ac_hover_circle_for_ellipse(lm)
        if axes is None or circle is None:
            return False

        _center, major0, major1, minor0, minor1 = axes
        circle_center, radius = circle
        previous_minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        previous_minor_len = math.hypot(*previous_minor_vec)
        min_axis = max(2.0, 4.0 / (self.disp_scale or 1.0))

        snapped0 = self._nearest_point_on_circle(
            major0,
            circle_center,
            radius,
            (
                major0[0] - circle_center[0],
                major0[1] - circle_center[1],
            ),
        )
        snapped1 = self._nearest_point_on_circle(
            major1,
            circle_center,
            radius,
            (
                major1[0] - circle_center[0],
                major1[1] - circle_center[1],
            ),
        )

        if math.hypot(snapped1[0] - snapped0[0], snapped1[1] - snapped0[1]) < (
            min_axis * 2.0
        ):
            snapped1 = (
                circle_center[0] - (snapped0[0] - circle_center[0]),
                circle_center[1] - (snapped0[1] - circle_center[1]),
            )

        new_center = (
            (snapped0[0] + snapped1[0]) / 2.0,
            (snapped0[1] + snapped1[1]) / 2.0,
        )
        major_vec = (
            (snapped1[0] - snapped0[0]) / 2.0,
            (snapped1[1] - snapped0[1]) / 2.0,
        )
        minor_vec = self._perpendicular_vector_with_length(
            major_vec,
            max(min_axis, previous_minor_len),
            previous_minor_vec,
        )
        if minor_vec is None:
            return False
        minor_vec = self._limit_axis_vector_to_image(new_center, minor_vec)
        if math.hypot(*minor_vec) < 1e-12:
            return False

        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                new_center,
                major_vec,
                minor_vec,
                FAC_CONSTRUCTION_SNAPPED,
            ),
            FAC_CONSTRUCTION_SNAPPED,
        )
        return True

    def _snap_selected_fac_axis_to_ac_circle(self) -> bool:
        if getattr(self, "multi_review_mode", False):
            return False

        lm = self.selected_landmark.get().strip()
        if not self._is_ellipse_landmark(lm):
            self._set_status("Select an acetabular face contour before snapping.")
            return False

        if not self._snap_ellipse_major_axis_to_ac_circle(lm):
            ac_lm = self._matching_ac_landmark_for_ellipse(lm) or "AC"
            self._set_status(
                f"{lm}: place {ac_lm} with a hover circle before snapping."
            )
            return False

        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._autosave_annotation_change()
        return True

    def _apply_selected_fac_construction_mode(self) -> bool:
        return self._snap_selected_fac_axis_to_ac_circle()

    def _refresh_fac_construction_controls(self) -> None:
        var = self.__dict__.get("fac_construction_mode")
        if var is not None:
            var.set(FAC_CONSTRUCTION_SNAPPED)
        z_scale = self.__dict__.get("projected_hemisphere_z_scale")
        if z_scale is not None:
            try:
                z_scale.configure(state="disabled")
            except Exception:
                pass
        for widget in (
            self.__dict__.get("ellipse_snap_check"),
            self.__dict__.get("ellipse_lock_check"),
        ):
            if widget is None:
                continue
            try:
                widget.configure(state="normal")
            except Exception:
                pass

        button = self.__dict__.get("fac_apply_button")
        if button is not None:
            try:
                button.configure(text="Snap Selected FAC")
            except Exception:
                pass

    def _sync_fac_construction_mode_for_selected_landmark(self, lm: str) -> None:
        var = self.__dict__.get("fac_construction_mode")
        if var is not None:
            var.set(FAC_CONSTRUCTION_SNAPPED)
        self._refresh_fac_construction_controls()

    def _on_fac_construction_mode_changed(self) -> None:
        self._refresh_fac_construction_controls()
        if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE:
            self._select_projected_hemisphere_placement_landmark()
        lm = self.selected_landmark.get().strip()
        if self._is_ellipse_landmark(lm):
            mode_label = (
                "projected hemisphere"
                if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE
                else "AC circle + snapped ellipse"
            )
            self._set_status(f"{lm}: new placements use {mode_label}.")

    def _on_ellipse_snap_toggle(self) -> None:
        if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE:
            return
        if not self._ellipse_snap_enabled():
            return

        lm = self.selected_landmark.get().strip()
        if self._is_ellipse_landmark(lm):
            self._snap_selected_fac_axis_to_ac_circle()

    def _shift_ellipse(
        self, lm: str, delta: AnnotationPoint, *, sync_dependents: bool = True
    ) -> None:
        ellipse = self._get_ellipse(lm)
        axes = self._ellipse_axis_points(ellipse)
        if axes is None:
            return
        construction = self._ellipse_construction(lm, ellipse)

        dx, dy = delta
        center, major0, major1, minor0, minor1 = axes
        points = [center, major0, major1, minor0, minor1]

        if self.current_image:
            w, h = self.current_image.size
            max_x = float(max(0, w - 1))
            max_y = float(max(0, h - 1))
            min_dx = max(0.0 - px for px, _py in points)
            max_dx = min(max_x - px for px, _py in points)
            min_dy = max(0.0 - py for _px, py in points)
            max_dy = min(max_y - py for _px, py in points)
            dx = max(min_dx, min(max_dx, dx))
            dy = max(min_dy, min(max_dy, dy))

        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return

        new_center = (center[0] + dx, center[1] + dy)
        major_vec = (
            (major1[0] - major0[0]) / 2.0,
            (major1[1] - major0[1]) / 2.0,
        )
        minor_vec = (
            (minor1[0] - minor0[0]) / 2.0,
            (minor1[1] - minor0[1]) / 2.0,
        )
        self._set_ellipse(
            lm,
            self._ellipse_from_center_vectors(
                new_center, major_vec, minor_vec, construction
            ),
            construction,
            sync_dependents=sync_dependents,
        )

    def _get_line_points(self, lm: str) -> List[Tuple[float, float]]:
        pts, _quality = self._get_annotations()
        val = pts.get(lm)
        if val is None:
            return []

        out: List[Tuple[float, float]] = []

        if (
            isinstance(val, tuple)
            and len(val) == 2
            and all(isinstance(v, (int, float)) for v in val)
        ):
            return [(float(val[0]), float(val[1]))]

        if isinstance(val, list):
            for p in val:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        out.append((float(p[0]), float(p[1])))
                    except (TypeError, ValueError):
                        pass

        return out[:2]

    def _set_line_points(self, lm: str, pts_list: List[Tuple[float, float]]) -> None:
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return
        self.annotations.setdefault(ann_key, {})[lm] = [
            (float(x), float(y)) for x, y in pts_list[:2]
        ]
        if pts_list:
            self._set_landmark_status(lm, "")
        if lm in self.landmark_found:
            self.landmark_found[lm].set(len(pts_list) > 0)

    def _clamp_img_point(self, xi: float, yi: float) -> Tuple[float, float]:
        if not self.current_image:
            return xi, yi
        w, h = self.current_image.size
        xi = min(max(xi, 0.0), float(w - 1))
        yi = min(max(yi, 0.0), float(h - 1))
        return round(xi, 1), round(yi, 1)

    def _find_line_point_hit(
        self, lm: str, screen_x: float, screen_y: float
    ) -> Optional[int]:
        pts = self._get_line_points(lm)
        if not pts:
            return None

        best_idx = None
        best_dist2 = float("inf")
        tol2 = float(self.drag_tolerance_px * self.drag_tolerance_px)

        for i, (xi, yi) in enumerate(pts):
            xs, ys = self._img_to_screen(xi, yi)
            d2 = (xs - screen_x) ** 2 + (ys - screen_y) ** 2
            if d2 <= tol2 and d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        return best_idx

    def _point_to_segment_distance_px(
        self,
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        vx = x2 - x1
        vy = y2 - y1
        seg_len2 = vx * vx + vy * vy

        if seg_len2 <= 1e-12:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        t = ((px - x1) * vx + (py - y1) * vy) / seg_len2
        t = max(0.0, min(1.0, t))

        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def _is_line_hit(self, lm: str, screen_x: float, screen_y: float) -> bool:
        pts = self._get_line_points(lm)
        if len(pts) != 2:
            return False

        (x1i, y1i), (x2i, y2i) = pts
        x1s, y1s = self._img_to_screen(x1i, y1i)
        x2s, y2s = self._img_to_screen(x2i, y2i)

        dist = self._point_to_segment_distance_px(
            screen_x, screen_y, x1s, y1s, x2s, y2s
        )
        return dist <= float(self.drag_line_tolerance_px)

    def _clear_line_preview(self) -> None:
        if self.line_preview_id is not None:
            try:
                self.canvas.delete(self.line_preview_id)
            except Exception as e:
                logger.warning(f"Failed to delete line preview: {e}")
            self.line_preview_id = None

    def _update_line_preview(self, mouse_x: float, mouse_y: float) -> None:
        lm = self.selected_landmark.get()
        if not lm or not self._is_line_landmark(lm):
            self._clear_line_preview()
            return

        if getattr(self, "multi_review_mode", False):
            temp_value = self._current_multi_review_temp_annotations().get(lm)
            pts: List[AnnotationPoint] = []
            if isinstance(temp_value, list):
                for point in temp_value[:2]:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            pts.append((float(point[0]), float(point[1])))
                        except (TypeError, ValueError):
                            pass
        else:
            pts = self._get_line_points(lm)
        if len(pts) != 1:
            self._clear_line_preview()
            return

        x0, y0 = self._img_to_screen(*pts[0])

        if self.line_preview_id is None:
            self.line_preview_id = self.canvas.create_line(
                x0,
                y0,
                mouse_x,
                mouse_y,
                fill="cyan",
                width=2,
                dash=(4, 2),
                tags="line_preview",
            )
        else:
            self.canvas.coords(self.line_preview_id, x0, y0, mouse_x, mouse_y)

        try:
            self.canvas.tag_lower(self.line_preview_id, "marker")
        except Exception:
            pass

    def _draw_multi_review_zoom_overlay(
        self,
        lm: str,
        img_to_zoom,
        overlay_ids: list[int],
    ) -> None:
        if self.zoom_canvas is None or self.current_image_path is None:
            return

        image_key = self._current_multi_review_image_key()
        if image_key is None:
            return

        offsets = [
            (-7, -7),
            (7, -7),
            (-7, 7),
            (7, 7),
            (0, -12),
            (0, 12),
            (12, 0),
            (-12, 0),
        ]

        def landmark_visible(landmark: str) -> bool:
            vis_var = self.landmark_visibility.get(landmark)
            return True if vis_var is None else bool(vis_var.get())

        def draw_value(
            landmark: str,
            value: AnnotationValue,
            color: str,
            label: str,
            dx: float,
            dy: float,
            *,
            selected: bool = False,
            temp: bool = False,
        ) -> None:
            width = 3 if selected or temp else 2
            if self._is_ellipse_landmark(landmark):
                if self._normalize_ellipse_value(value) is None:
                    return
                overlay_ids.extend(
                    self._draw_ellipse_annotation(
                        self.zoom_canvas,
                        value,
                        img_to_zoom,
                        outline=color,
                        label_fill=color,
                        label=label,
                        font=self.dialogue_font,
                        width=width,
                        handle_radius=8 if temp or selected else 6,
                        tags="zoom_landmark_overlay",
                        label_offset=(dx, dy),
                        show_handles=temp or selected,
                    )
                )
                overlay_ids.extend(
                    self._draw_projected_hemisphere_boundary(
                        self.zoom_canvas,
                        landmark,
                        img_to_zoom,
                        ellipse=value,
                        width=width,
                        tags="zoom_landmark_overlay",
                    )
                )
                return

            if self._is_line_landmark(landmark):
                if not isinstance(value, list) or not value:
                    return

                zoom_pts = [img_to_zoom(px, py) for px, py in value]
                if len(zoom_pts) == 2:
                    line_options = {
                        "fill": color,
                        "width": width,
                        "tags": "zoom_landmark_overlay",
                    }
                    if temp:
                        line_options["dash"] = (5, 3)
                    overlay_ids.append(
                        self.zoom_canvas.create_line(
                            zoom_pts[0][0],
                            zoom_pts[0][1],
                            zoom_pts[1][0],
                            zoom_pts[1][1],
                            **line_options,
                        )
                    )

                for zx, zy in zoom_pts:
                    r = 8 if temp else 7
                    overlay_ids.append(
                        self.zoom_canvas.create_rectangle(
                            zx - r,
                            zy - r,
                            zx + r,
                            zy + r,
                            outline=color,
                            width=width,
                            tags="zoom_landmark_overlay",
                        )
                    )

                if zoom_pts and label:
                    overlay_ids.append(
                        self.zoom_canvas.create_text(
                            zoom_pts[0][0] + dx,
                            zoom_pts[0][1] + dy - 10,
                            text=label,
                            fill=color,
                            font=self.dialogue_font,
                            anchor="w",
                            tags="zoom_landmark_overlay",
                        )
                    )
                return

            center = self._landmark_center_from_value(landmark, value)
            if center is None:
                return

            zx, zy = img_to_zoom(center[0], center[1])
            r = 8 if temp else 7
            overlay_ids.extend(
                [
                    self.zoom_canvas.create_oval(
                        zx - r,
                        zy - r,
                        zx + r,
                        zy + r,
                        outline=color,
                        width=width,
                        tags="zoom_landmark_overlay",
                    ),
                    self.zoom_canvas.create_line(
                        zx - r,
                        zy,
                        zx + r,
                        zy,
                        fill=color,
                        width=width,
                        tags="zoom_landmark_overlay",
                    ),
                    self.zoom_canvas.create_line(
                        zx,
                        zy - r,
                        zx,
                        zy + r,
                        fill=color,
                        width=width,
                        tags="zoom_landmark_overlay",
                    ),
                ]
            )
            if label:
                overlay_ids.append(
                    self.zoom_canvas.create_text(
                        zx + dx + 8,
                        zy + dy - 8,
                        text=label,
                        fill=color,
                        font=self.dialogue_font,
                        anchor="w",
                        tags="zoom_landmark_overlay",
                    )
                )

        for reviewer_idx, reviewer in enumerate(self.multi_review_reviewer_order):
            if not self._is_multi_review_reviewer_visible(reviewer):
                continue

            reviewer_pts = self.multi_review_annotations.get(image_key, {}).get(
                reviewer, {}
            )
            color = self._multi_review_color(reviewer)
            dx, dy = offsets[reviewer_idx % len(offsets)]
            for draw_lm, value in reviewer_pts.items():
                if not landmark_visible(draw_lm):
                    continue
                is_selected = draw_lm == lm
                label = draw_lm if is_selected else ""
                draw_value(
                    draw_lm,
                    value,
                    color,
                    label,
                    dx,
                    dy,
                    selected=is_selected,
                )

        temp_pts = self.multi_review_temp_annotations.get(image_key, {})
        for draw_lm, value in temp_pts.items():
            if not landmark_visible(draw_lm):
                continue
            draw_value(
                draw_lm,
                value,
                "white",
                f"temp:{draw_lm}",
                8,
                -8,
                selected=draw_lm == lm,
                temp=True,
            )

    def _segmentation_mask_for_display(self, lm: str) -> np.ndarray | None:
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return None

        mask = self.__dict__.get("seg_masks", {}).get(mask_key, {}).get(lm)
        if mask is not None:
            return mask

        if self._is_border_segmentation_landmark(lm):
            return (
                self.__dict__.get("it_edge_border_masks", {})
                .get(mask_key, {})
                .get(lm)
            )
        return None

    def _draw_segmentation_zoom_overlay(
        self, lm: str, overlay_ids: list[int]
    ) -> bool:
        if self.zoom_canvas is None or self.zoom_src_rect is None:
            return False
        mask = self._segmentation_mask_for_display(lm)
        if mask is None:
            return False

        size = self._get_zoom_canvas_size()
        if size <= 1:
            return False

        mask_src_rect = self._image_rect_to_mask_rect(self.zoom_src_rect, mask)
        if mask_src_rect is None:
            return False

        try:
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
            mask_zoom = mask_img.transform(
                (size, size),
                Image.Transform.EXTENT,
                mask_src_rect,
                resample=Image.Resampling.NEAREST,
                fill=0,
            )
        except TypeError:
            mask_zoom = mask_img.transform(
                (size, size),
                Image.Transform.EXTENT,
                mask_src_rect,
                resample=Image.Resampling.NEAREST,
            )
        except Exception as e:
            logger.warning(f"Failed to render zoom segmentation overlay for {lm}: {e}")
            return False

        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        color_img = Image.new("RGBA", (size, size), SEGMENTATION_OVERLAY_RGBA)
        overlay.paste(color_img, (0, 0), mask_zoom)
        self.zoom_seg_img_obj = ImageTk.PhotoImage(overlay)
        item_id = self.zoom_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.zoom_seg_img_obj,
            tags="zoom_landmark_overlay",
        )
        overlay_ids.append(item_id)
        return True

    def _zoom_landmarks_for_current_set(self, lm: str) -> List[str]:
        if not lm:
            return []
        primary = self._primary_group_for_landmark(lm)
        members = list(
            self.__dict__.get("landmark_group_members", {}).get(primary, [])
        )
        if not members:
            members = [primary]
            members.extend(self._relevant_derived_features_for_landmark(primary))
        if lm not in members:
            members.insert(0, lm)
        landmark_set = set(getattr(self, "landmarks", []))
        out: List[str] = []
        seen: Set[str] = set()
        for candidate in members:
            if not candidate or candidate in seen:
                continue
            if landmark_set and candidate not in landmark_set:
                continue
            out.append(candidate)
            seen.add(candidate)
        return out

    def _draw_single_zoom_landmark_overlay(
        self,
        lm: str,
        pts: Dict[str, AnnotationValue],
        img_to_zoom,
        overlay_ids: List[int],
        *,
        size: int,
        src_w: float,
    ) -> None:
        if self.zoom_canvas is None:
            return
        self._draw_segmentation_zoom_overlay(lm, overlay_ids)
        if not self._landmark_should_render_on_canvas(lm):
            return

        if self._is_intermediate_acetabular_landmark(lm):
            ctx = self._intermediate_acetabular_context_for_landmark(lm)
            ellipse_lm = ctx["ellipse"] if ctx is not None else ""
            ellipse = self._get_ellipse(ellipse_lm) if ellipse_lm else None
            if ellipse is not None:
                overlay_ids.extend(
                    self._draw_ellipse_annotation(
                        self.zoom_canvas,
                        ellipse,
                        img_to_zoom,
                        outline="#88E0FF",
                        label_fill="#88E0FF",
                        label="",
                        font=self.dialogue_font,
                        width=1,
                        handle_radius=5,
                        tags="zoom_landmark_overlay",
                        show_handles=False,
                    )
                )
                overlay_ids.extend(
                    self._draw_projected_hemisphere_boundary(
                        self.zoom_canvas,
                        ellipse_lm,
                        img_to_zoom,
                        ellipse=ellipse,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                )
                overlay_ids.extend(
                    self._draw_acetabular_lower_arc_highlights(
                        self.zoom_canvas,
                        ellipse_lm,
                        img_to_zoom,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                )

        if getattr(self, "multi_review_mode", False):
            self._draw_multi_review_zoom_overlay(lm, img_to_zoom, overlay_ids)
        elif self._is_ellipse_landmark(lm):
            ellipse = self._get_ellipse(lm)
            if ellipse is not None:
                zoom_color = self._zoom_overlay_color_for_landmark(lm)
                overlay_ids.extend(
                    self._draw_ellipse_annotation(
                        self.zoom_canvas,
                        ellipse,
                        img_to_zoom,
                        outline=zoom_color,
                        label_fill=zoom_color,
                        label="",
                        font=self.dialogue_font,
                        width=2,
                        handle_radius=7,
                        tags="zoom_landmark_overlay",
                        show_handles=True,
                    )
                )
                overlay_ids.extend(
                    self._draw_projected_hemisphere_boundary(
                        self.zoom_canvas,
                        lm,
                        img_to_zoom,
                        ellipse=ellipse,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                )
                overlay_ids.extend(
                    self._draw_acetabular_lower_arc_highlights(
                        self.zoom_canvas,
                        lm,
                        img_to_zoom,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                )
        elif self._is_line_landmark(lm):
            line_pts = self._get_line_points(lm)
            if line_pts:
                zoom_pts = [img_to_zoom(px, py) for px, py in line_pts]
                zoom_color = self._zoom_overlay_color_for_landmark(lm)
                if len(zoom_pts) == 2:
                    line_id = self.zoom_canvas.create_line(
                        zoom_pts[0][0],
                        zoom_pts[0][1],
                        zoom_pts[1][0],
                        zoom_pts[1][1],
                        fill=zoom_color,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                    overlay_ids.append(line_id)

                r = 7
                for zx, zy in zoom_pts:
                    circle_id = self.zoom_canvas.create_oval(
                        zx - r,
                        zy - r,
                        zx + r,
                        zy + r,
                        outline=zoom_color,
                        width=2,
                        tags="zoom_landmark_overlay",
                    )
                    hline_id = self.zoom_canvas.create_line(
                        zx - r,
                        zy,
                        zx + r,
                        zy,
                        fill=zoom_color,
                        width=1,
                        tags="zoom_landmark_overlay",
                    )
                    vline_id = self.zoom_canvas.create_line(
                        zx,
                        zy - r,
                        zx,
                        zy + r,
                        fill=zoom_color,
                        width=1,
                        tags="zoom_landmark_overlay",
                    )
                    overlay_ids.extend([circle_id, hline_id, vline_id])
        else:
            point = pts.get(lm)
            if (
                isinstance(point, tuple)
                and len(point) == 2
                and all(isinstance(v, (int, float)) for v in point)
            ):
                px, py = point
                zx, zy = img_to_zoom(px, py)

                r = 7
                zoom_color = self._zoom_overlay_color_for_landmark(lm)
                circle_id = self.zoom_canvas.create_oval(
                    zx - r,
                    zy - r,
                    zx + r,
                    zy + r,
                    outline=zoom_color,
                    width=2,
                    tags="zoom_landmark_overlay",
                )
                hline_id = self.zoom_canvas.create_line(
                    zx - r,
                    zy,
                    zx + r,
                    zy,
                    fill=zoom_color,
                    width=1,
                    tags="zoom_landmark_overlay",
                )
                vline_id = self.zoom_canvas.create_line(
                    zx,
                    zy - r,
                    zx,
                    zy + r,
                    fill=zoom_color,
                    width=1,
                    tags="zoom_landmark_overlay",
                )
                overlay_ids.extend([circle_id, hline_id, vline_id])
                if lm in self.HOVER_CIRCLE_LANDMARKS and abs(self.disp_scale) > 1e-12:
                    key = (
                        self._path_key(self.current_image_path)
                        if self.json_path is not None
                        else str(self.current_image_path)
                    )
                    saved_radius = getattr(self, "hover_radii", {}).get(key, {}).get(lm)
                    ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
                    if ellipse_lm is not None and self._ellipse_is_projected_hemisphere(
                        ellipse_lm
                    ):
                        overlay_ids.extend(
                            self._draw_projected_hemisphere_boundary(
                                self.zoom_canvas,
                                ellipse_lm,
                                img_to_zoom,
                                width=2,
                                tags="zoom_landmark_overlay",
                            )
                        )
                    elif saved_radius is not None:
                        zoom_r = saved_radius * (size / src_w)
                        overlay_ids.append(
                            self.zoom_canvas.create_oval(
                                zx - zoom_r,
                                zy - zoom_r,
                                zx + zoom_r,
                                zy + zoom_r,
                                outline="orange",
                                width=2,
                                tags="zoom_landmark_overlay",
                            )
                        )

    def _refresh_zoom_landmark_overlay(self) -> None:
        self._clear_zoom_landmark_overlay()

        if self.zoom_canvas is None:
            return
        if not self.show_selected_landmark_in_zoom.get():
            return
        if self.current_image is None or self.current_image_path is None:
            return
        if self.zoom_src_rect is None:
            return

        lm = self.selected_landmark.get().strip()
        if not lm:
            return

        pts, _quality = self._get_annotations()
        size = self._get_zoom_canvas_size()
        if size <= 1:
            return

        src_left, src_top, src_right, src_bottom = self.zoom_src_rect
        src_w = src_right - src_left
        src_h = src_bottom - src_top
        if abs(src_w) < 1e-12 or abs(src_h) < 1e-12:
            return

        def img_to_zoom(px: float, py: float) -> Tuple[float, float]:
            zx = ((float(px) - src_left) / src_w) * size
            zy = ((float(py) - src_top) / src_h) * size
            return zx, zy

        overlay_ids: list[int] = []
        zoom_landmarks = self._zoom_landmarks_for_current_set(lm)
        pod_landmarks = {
            "L-C-PO",
            "R-C-PO",
            "L-C-PS",
            "R-C-PS",
            "L-LPO",
            "R-LPO",
            "POD",
        }
        if any(candidate in pod_landmarks for candidate in zoom_landmarks):
            overlay_ids.extend(
                self._draw_protocol_pod_construction_overlay(
                    self.zoom_canvas,
                    img_to_zoom,
                    tags="zoom_landmark_overlay",
                    draw_pod_line=False,
                    width=2,
                )
            )

        for zoom_lm in zoom_landmarks:
            self._draw_single_zoom_landmark_overlay(
                zoom_lm,
                pts,
                img_to_zoom,
                overlay_ids,
                size=size,
                src_w=src_w,
            )

        self.zoom_landmark_overlay_ids = overlay_ids

        for item_id in self.zoom_landmark_overlay_ids:
            try:
                self.zoom_canvas.tag_raise(item_id)
            except Exception:
                pass

        for item_id in getattr(self, "zoom_crosshair_ids", []):
            try:
                self.zoom_canvas.tag_raise(item_id)
            except Exception:
                pass

        for item_id in getattr(self, "zoom_extended_crosshair_ids", []):
            try:
                self.zoom_canvas.tag_raise(item_id)
            except Exception:
                pass

    # Builds the landmark selection table with visibility and status controls.

    def _build_landmark_panel(self) -> None:
        for w in self.lp_inner.winfo_children():
            w.destroy()
        self.landmark_visibility.clear()
        self.landmark_found.clear()
        self.landmark_flagged.clear()
        self.landmark_radio_widgets = {}
        self.landmark_found_widgets = {}
        self.landmark_flag_widgets = {}
        self.landmark_correct_labels = {}
        self.landmark_delete_widgets = {}
        self.landmark_group_frames = {}
        self.landmark_group_children = {}
        self.landmark_group_members = {}

        self.landmark_table = tk.Frame(self.lp_inner)
        self.landmark_table.pack(fill="x", padx=2, pady=2)

        allowed = self._get_allowed_landmarks_for_current_view()
        visible_landmarks = self._display_ordered(
            [lm for lm in getattr(self, "landmarks", []) if lm in allowed]
        )
        primary_groups, orphan_derived = self._landmark_panel_groups(visible_landmarks)
        primary_landmarks = [primary for primary, _children in primary_groups]
        selectable_landmarks = [
            lm for lm in primary_landmarks if self._landmark_is_selectable(lm)
        ]

        def make_label(parent, text: str, *, bg: str, fg: str = "black", font=None):
            return tk.Label(
                parent,
                text=text,
                anchor="w",
                justify="left",
                font=font or self.dialogue_font,
                fg=fg,
                bg=bg,
            )

        def configure_columns(parent) -> None:
            parent.grid_columnconfigure(0, minsize=220)
            parent.grid_columnconfigure(1, minsize=58)
            parent.grid_columnconfigure(2, minsize=82)
            parent.grid_columnconfigure(3, minsize=74)
            parent.grid_anchor("w")

        def add_landmark_row(parent, lm: str, *, primary: bool, row: int) -> None:
            is_selectable = self._landmark_is_selectable(lm)
            bg = "#F4F8F8" if primary else "#FFFFFF"
            vis_var = tk.BooleanVar(value=True)
            found_var = tk.BooleanVar(value=False)
            self.landmark_visibility[lm] = vis_var
            self.landmark_found[lm] = found_var
            tk.Checkbutton(
                parent,
                variable=vis_var,
                command=self._draw_points,
                font=self.dialogue_font,
                bg=bg,
                activebackground=bg,
                selectcolor=bg,
                takefocus=0,
            ).grid(row=row, column=1, sticky="w", padx=(2, 4), pady=(3 if primary else 1))

            if primary:
                rb = tk.Radiobutton(
                    parent,
                    text=lm,
                    variable=self.selected_landmark,
                    value=lm,
                    anchor="w",
                    justify="left",
                    command=self._on_landmark_selected,
                    font=self.__dict__.get("primary_label_font", self.heading_font),
                    fg=self._panel_label_color_for_landmark(lm),
                    activeforeground=self._panel_label_color_for_landmark(lm),
                    disabledforeground=self._panel_label_color_for_landmark(lm),
                    bg=bg,
                    activebackground=bg,
                    selectcolor=bg,
                    takefocus=0,
                )
                if not is_selectable:
                    rb.configure(state="disabled")
                rb.grid(row=row, column=0, sticky="ew", padx=(2, 4), pady=3)
                self.landmark_radio_widgets[lm] = rb
            else:
                make_label(
                    parent,
                    lm,
                    bg=bg,
                    fg=self._panel_label_color_for_landmark(lm),
                    font=self._label_font_for_landmark(lm, selected=False),
                ).grid(row=row, column=0, sticky="ew", padx=(18, 4), pady=1)

            status_label = make_label(
                parent,
                self._derived_feature_status_text_for_landmark(lm)
                if self._is_protocol_derived_feature(lm)
                else self._primary_landmark_status_text(lm),
                bg=bg,
                fg=(
                    self._status_color_for_derived_feature(lm)
                    if self._is_protocol_derived_feature(lm)
                    else self._primary_landmark_status_color(lm)
                ),
                font=self.__dict__.get("primary_label_font", self.dialogue_font)
                if primary
                else self.dialogue_font,
            )
            status_label.grid(
                row=row,
                column=2,
                sticky="ew",
                padx=(2, 4),
                pady=(3 if primary else 1),
            )
            self.landmark_correct_labels[lm] = status_label

            if primary:
                delete_btn = tk.Button(
                    parent,
                    text="Delete",
                    command=lambda lm=lm: self._delete_landmark_from_button(lm),
                    font=self.dialogue_font,
                    width=7,
                    padx=0,
                    pady=0,
                    takefocus=False,
                )
                delete_btn.grid(row=row, column=3, sticky="w", padx=(2, 4), pady=3)
                self.landmark_delete_widgets[lm] = delete_btn
            else:
                make_label(parent, "", bg=bg).grid(row=row, column=3, sticky="ew", padx=(2, 4), pady=1)

        def add_group(primary: str, children: List[str]) -> None:
            group = tk.Frame(
                self.landmark_table,
                bd=1,
                relief="groove",
                highlightthickness=1,
                highlightbackground="#BFCACA",
                bg="#FFFFFF",
            )
            group.pack(fill="x", padx=(0, 2), pady=(4, 5))
            configure_columns(group)
            self.landmark_group_frames[primary] = group
            self.landmark_group_children[primary] = [group]
            self.landmark_group_members[primary] = [primary] + list(children)
            add_landmark_row(group, primary, primary=True, row=0)
            for idx, child in enumerate(children, start=1):
                add_landmark_row(group, child, primary=False, row=idx)

        for primary, children in primary_groups:
            add_group(primary, children)

        if orphan_derived:
            section = tk.Label(
                self.landmark_table,
                text="Other Derived / Categorical Landmarks",
                anchor="w",
                font=self.heading_font,
                fg="#333333",
                bg="#E6ECEC",
            )
            section.pack(fill="x", padx=(0, 2), pady=(8, 2))
            orphan_frame = tk.Frame(self.landmark_table, bg="#FFFFFF")
            orphan_frame.pack(fill="x", padx=(0, 2), pady=(0, 4))
            configure_columns(orphan_frame)
            for idx, lm in enumerate(orphan_derived):
                add_landmark_row(orphan_frame, lm, primary=False, row=idx)

        current = self.selected_landmark.get()
        if selectable_landmarks and current not in selectable_landmarks:
            self.selected_landmark.set(selectable_landmarks[0])
        elif not selectable_landmarks:
            self.selected_landmark.set("")

        self._sync_fac_construction_mode_for_selected_landmark(
            self.selected_landmark.get()
        )
        self._load_note_for_selected_landmark()
        self._bind_landmark_scroll(True)
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._refresh_landmark_selectability()
        self._refresh_selected_landmark_group_style()

    def _refresh_landmark_selectability(self) -> None:
        for lm, rb in self.__dict__.get("landmark_radio_widgets", {}).items():
            is_selectable = self._landmark_is_selectable(lm)
            try:
                rb.configure(state="normal" if is_selectable else "disabled")
            except Exception:
                pass

        for lm, button in self.__dict__.get("landmark_delete_widgets", {}).items():
            has_stored = self._landmark_has_annotation_value(lm) or bool(
                self._get_landmark_status(lm)
            )
            state = (
                "normal"
                if self.current_image_path is not None
                and not getattr(self, "multi_review_mode", False)
                and self._landmark_is_primary_for_panel(lm)
                and has_stored
                else "disabled"
            )
            try:
                button.configure(state=state)
            except Exception:
                pass
        self._refresh_landmark_categorical_count_labels()

    # (4) Render base image (add to class)
    def _render_base_image(self) -> None:
        if not self.current_image:
            return

        self._recompute_transform()
        disp_w, disp_h = self.disp_size
        off_x, off_y = self.disp_off

        resized = self.current_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.img_obj = ImageTk.PhotoImage(resized)

        if self.base_img_item is None:
            self.base_img_item = self.canvas.create_image(
                off_x, off_y, anchor="nw", image=self.img_obj, tags="base"
            )
        else:
            self.canvas.itemconfigure(self.base_img_item, image=self.img_obj)
            self.canvas.coords(self.base_img_item, off_x, off_y)

    def _on_canvas_resize(self, _event=None) -> None:
        self._clear_it_edge_border_live_overlay()
        if not self.current_image:
            self._clear_line_preview()
            self._update_zoom_view(None, None)
            self._clear_femoral_axis_overlay()
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()
            return
        self._render_base_image()
        self._draw_points()
        for lm in SEGMENTATION_LANDMARKS:
            self._update_overlay_for(lm)
        for lm in IT_EDGE_LANDMARKS:
            self._update_it_edge_border_overlay_for(lm)
        if self.last_mouse_canvas_pos is None:
            self._clear_line_preview()
            self._update_zoom_view(None, None)
            self._update_femoral_axis_overlay()
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()
            return

        mouse_x, mouse_y = self.last_mouse_canvas_pos
        x0, y0, x1, y1 = self._display_rect()
        if x0 <= mouse_x < x1 and y0 <= mouse_y < y1:
            self._update_zoom_view(mouse_x, mouse_y)
            self._update_line_preview(mouse_x, mouse_y)
            self._update_femoral_axis_overlay()
            if self.extended_crosshair_enabled.get():
                self._update_extended_crosshair(mouse_x, mouse_y)
            else:
                self._hide_extended_crosshair()
        else:
            self._clear_line_preview()
            self._update_zoom_view(None, None)
            self._clear_femoral_axis_overlay()
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()

    def _on_panel_resize(self, event=None) -> None:
        if event is None:
            return
        if event.widget is not self:
            return
        new_w = self._compute_panel_width()
        if new_w == self._panel_width:
            return

        self._panel_width = new_w

        # Update panel canvases
        self._left_canvas.config(width=new_w)
        self._ctrl_canvas.config(width=new_w)

        # Update zoom canvas (stay square)
        self.zoom_canvas.config(width=new_w, height=new_w)

        # Update containers
        self._image_container.config(width=new_w)
        self._landmark_panel_container.config(width=new_w)

        # Update landmark canvas
        self.lp_canvas.config(width=new_w - self._scrollbar_width)

        # Update status label wraplength
        if getattr(self, "pelvis_segmentation_status_label", None) is not None:
            self.pelvis_segmentation_status_label.config(
                wraplength=max(1, new_w - 30)
            )

        # Update minsize and re-layout
        self._fit_window_and_set_min()

    # Locks the initial window min-size after the first layout pass.
    def _lock_initial_minsize(self) -> None:
        self.update_idletasks()
        self._start_min_w = self.winfo_width()
        self._start_min_h = self.winfo_height()
        self.minsize(self._start_min_w, self._start_min_h)

    # Fits the window to required size and updates the minimum size.
    def _fit_window_and_set_min(self) -> None:
        self.minsize(0, 0)
        self.update_idletasks()
        req_w = self.winfo_reqwidth()
        side_content_h = max(
            getattr(self, "_left_tools", self).winfo_reqheight(),
            getattr(self, "_ctrl", self).winfo_reqheight(),
        )
        req_h = max(self.winfo_reqheight(), side_content_h + 24)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = screen_w
        max_h = int(screen_h * 0.92)
        target_h = min(max(req_h, 760), max_h)
        side_w = (self._panel_width * 2) + 50
        target_w = min(max(req_w, side_w + target_h), max_w)
        # Center window on screen
        pos_x = (screen_w - target_w) // 2
        pos_y = (screen_h - target_h) // 2
        self.geometry(f"{target_w}x{target_h}+{pos_x}+{pos_y}")
        self.update_idletasks()
        self.minsize(target_w, target_h)

    # Enables or disables mousewheel scrolling for the left panel.
    def _bind_left_panel_scroll(self, bind: bool) -> None:
        if bind:
            self._left_canvas.bind_all("<MouseWheel>", self._left_panel_mousewheel)
        else:
            self._left_canvas.unbind_all("<MouseWheel>")

    # Enables or disables mousewheel scrolling for the right panel.
    def _bind_right_panel_scroll(self, bind: bool) -> None:
        if bind:
            self._ctrl_canvas.bind_all("<MouseWheel>", self._right_panel_mousewheel)
        else:
            self._ctrl_canvas.unbind_all("<MouseWheel>")

    def _bind_image_list_scroll(self, bind: bool) -> None:
        if self.image_tree is None:
            return

        widgets: list[tk.Misc] = [self.image_tree]
        try:
            widgets.extend(self.image_tree.winfo_children())
        except Exception:
            pass

        if bind:
            for widget in widgets:
                try:
                    widget.bind("<MouseWheel>", self._image_list_mousewheel)
                    widget.bind("<Button-4>", self._image_list_mousewheel_linux_up)
                    widget.bind("<Button-5>", self._image_list_mousewheel_linux_down)
                except Exception:
                    pass
        else:
            for widget in widgets:
                try:
                    widget.unbind("<MouseWheel>")
                    widget.unbind("<Button-4>")
                    widget.unbind("<Button-5>")
                except Exception:
                    pass

    def _image_list_mousewheel(self, event) -> None:
        if self.image_tree is None:
            return
        if event.delta > 0:
            self.image_tree.yview_scroll(-1, "units")
        else:
            self.image_tree.yview_scroll(1, "units")

    def _image_list_mousewheel_linux_up(self, _event) -> None:
        if self.image_tree is not None:
            self.image_tree.yview_scroll(-1, "units")

    def _image_list_mousewheel_linux_down(self, _event) -> None:
        if self.image_tree is not None:
            self.image_tree.yview_scroll(1, "units")

    # Scrolls the landmark list in response to mouse wheel events.
    def _landmark_mousewheel(self, event) -> str:
        delta = -1 if event.delta > 0 else 1
        self.lp_canvas.yview_scroll(delta, "units")
        return "break"

    def _landmark_mousewheel_linux_up(self, _event) -> str:
        self.lp_canvas.yview_scroll(-1, "units")
        return "break"

    def _landmark_mousewheel_linux_down(self, _event) -> str:
        self.lp_canvas.yview_scroll(1, "units")
        return "break"

    # Scrolls the left panel in response to mouse wheel events.
    def _left_panel_mousewheel(self, event) -> None:
        delta = -1 if event.delta > 0 else 1
        self._left_canvas.yview_scroll(delta, "units")

    # Scrolls the right panel in response to mouse wheel events.
    def _right_panel_mousewheel(self, event) -> None:
        delta = -1 if event.delta > 0 else 1
        self._ctrl_canvas.yview_scroll(delta, "units")

    def _landmark_scroll_widgets(self) -> List[tk.Misc]:
        widgets: List[tk.Misc] = [self.lp_canvas, self.lp_inner]
        stack = [self.lp_inner]
        while stack:
            widget = stack.pop()
            try:
                children = list(widget.winfo_children())
            except Exception:
                children = []
            widgets.extend(children)
            stack.extend(children)
        return widgets

    # Enables or disables mousewheel scrolling for the landmark list.
    def _bind_landmark_scroll(self, bind: bool) -> None:
        for widget in self._landmark_scroll_widgets():
            try:
                if bind:
                    widget.bind("<MouseWheel>", self._landmark_mousewheel)
                    widget.bind("<Button-4>", self._landmark_mousewheel_linux_up)
                    widget.bind("<Button-5>", self._landmark_mousewheel_linux_down)
                else:
                    widget.unbind("<MouseWheel>")
                    widget.unbind("<Button-4>")
                    widget.unbind("<Button-5>")
            except Exception:
                pass

    def _scroll_landmark_into_view(self, lm: str) -> None:
        rb = getattr(self, "landmark_radio_widgets", {}).get(lm)
        if rb is None:
            return
        self.lp_canvas.update_idletasks()
        bbox = self.lp_canvas.bbox("all")
        if not bbox:
            return
        y1, y2 = bbox[1], bbox[3]
        total = max(1, y2 - y1)
        canvas_h = self.lp_canvas.winfo_height()
        vis_top = self.lp_canvas.canvasy(0)
        vis_bottom = vis_top + canvas_h
        item_top = self._widget_y_in_inner(rb)
        item_bottom = item_top + rb.winfo_height()
        pad = 6
        if item_top < vis_top:
            new_top = max(y1, item_top - pad)
            self.lp_canvas.yview_moveto((new_top - y1) / total)
        elif item_bottom > vis_bottom:
            new_top = min(item_bottom + pad - canvas_h, y2 - canvas_h)
            new_top = max(y1, new_top)
            self.lp_canvas.yview_moveto((new_top - y1) / total)

    # Applies stored per-landmark settings when a landmark is selected.
    def _on_landmark_selected(self) -> None:
        lm = self.selected_landmark.get()
        if not self._handle_landmark_group_transition(lm):
            return
        self._apply_settings_to_ui_for(lm)
        self._sync_fac_construction_mode_for_selected_landmark(lm)
        self._sync_auto_tools_for_selected_landmark()
        self._load_note_for_selected_landmark()
        self._refresh_multi_review_panel()
        self._refresh_categorical_controls()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        if self.last_mouse_canvas_pos is not None:
            self._update_line_preview(*self.last_mouse_canvas_pos)
        else:
            self._clear_line_preview()
        self._update_femoral_axis_overlay()
        self.after_idle(self._scroll_landmark_into_view, lm)
        if self._landmark_ref_dialog is not None:
            self._landmark_ref_dialog.update_landmark(lm)

    def _get_landmark_meta(self, lm: str) -> Dict[str, Union[bool, str]]:
        if self.current_image_path is None:
            return {"flag": False, "note": ""}

        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None
            else str(self.current_image_path)
        )
        per_img = self.landmark_meta.setdefault(key, {})
        return per_img.setdefault(lm, {"flag": False, "note": ""})

    def _set_landmark_flag(self, lm: str, value: bool) -> None:
        meta = self._get_landmark_meta(lm)
        meta["flag"] = bool(value)

    def _get_landmark_flag(self, lm: str) -> bool:
        return bool(self._get_landmark_meta(lm).get("flag", False))

    def _set_landmark_note(self, lm: str, note: str) -> None:
        meta = self._get_landmark_meta(lm)
        meta["note"] = note

    def _get_landmark_note(self, lm: str) -> str:
        return str(self._get_landmark_meta(lm).get("note", ""))

    def _set_landmark_status(self, lm: str, status: object) -> None:
        meta = self._get_landmark_meta(lm)
        normalized = self._normalize_direct_annotation_status(status)
        if normalized:
            meta["status"] = normalized
        else:
            meta.pop("status", None)

    def _get_landmark_status(self, lm: str) -> str:
        if self._landmark_has_annotation_value(lm):
            return ""
        return self._normalize_direct_annotation_status(
            self._get_landmark_meta(lm).get("status", "")
        )

    def _on_flag_checkbox_toggled(self, lm: str) -> None:
        if getattr(self, "multi_review_mode", False):
            return
        if self.current_image_path is None:
            return
        if self._is_protocol_derived_feature(lm):
            flag_var = self.landmark_flagged.get(lm)
            if flag_var is not None:
                flag_var.set(False)
            self._set_status(self._noneditable_derived_message(lm))
            return
        if self._is_derived_acetabular_landmark(lm):
            flag_var = self.landmark_flagged.get(lm)
            if flag_var is not None:
                flag_var.set(False)
            self._set_status(self._noneditable_derived_message(lm))
            return
        if self._is_intermediate_acetabular_landmark(
            lm
        ) and not self._intermediate_acetabular_landmark_ready(lm):
            flag_var = self.landmark_flagged.get(lm)
            if flag_var is not None:
                flag_var.set(False)
            self._set_status(self._acetabular_geometry_required_message(lm))
            return

        flag_var = self.landmark_flagged.get(lm)
        is_flagged = flag_var.get() if flag_var is not None else False
        self._set_landmark_flag(lm, is_flagged)

        if not is_flagged:
            self._set_landmark_note(lm, "")

        if self.selected_landmark.get() == lm:
            self._load_note_for_selected_landmark()

        self.dirty = True
        self._maybe_autosave_current_image()

        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._draw_points()

    def _subsequent_protocol_landmarks_with_data(self, lm: str) -> List[str]:
        order = self._protocol_direct_order_for_current_view()
        if lm not in order:
            return []
        later = order[order.index(lm) + 1 :]
        return [
            candidate
            for candidate in later
            if self._landmark_has_annotation_value(candidate)
            or bool(self._get_landmark_status(candidate))
        ]

    def _prompt_review_subsequent_landmarks(
        self, revised_lm: str, later_landmarks: List[str]
    ) -> None:
        if not later_landmarks:
            return
        preview = ", ".join(later_landmarks[:8])
        if len(later_landmarks) > 8:
            preview += f", and {len(later_landmarks) - 8} more"
        messagebox.showinfo(
            "Review Later Landmarks",
            f"{revised_lm} was deleted or revised. Review subsequent "
            "landmarks that were already annotated or marked before this change:\n\n"
            f"{preview}",
        )

    def _delete_landmark_from_button(self, lm: str) -> None:
        self.selected_landmark.set(lm)
        self._on_landmark_selected()
        self._delete_current_landmark()

    def _delete_landmark_annotation(
        self,
        lm: str,
        *,
        confirm: bool = True,
        show_review_prompt: bool = True,
    ) -> bool:
        if getattr(self, "multi_review_mode", False):
            return self._delete_selected_multi_review_temp_annotation()
        if self.current_image_path is None or not lm:
            return False
        if not self._landmark_is_primary_for_panel(lm):
            self._set_status(self._noneditable_derived_message(lm))
            return False

        ann_key = self._current_annotation_key()
        if ann_key is None:
            return False
        current_pts = self.annotations.setdefault(ann_key, {})
        has_stored = self._landmark_has_annotation_value(lm) or bool(
            self._get_landmark_status(lm)
        )
        if not has_stored and lm not in current_pts:
            self._set_status(f"{lm} has no annotation to delete.")
            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._refresh_landmark_selectability()
            return False

        if confirm and not messagebox.askyesno(
            "Delete Landmark",
            f'Are you sure you want to delete "{lm}"?\n\n'
            "The primary annotation, stored landmark status, and dependent "
            "generated geometry for this landmark will be cleared.",
        ):
            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._refresh_landmark_selectability()
            return False

        later_to_review = self._subsequent_protocol_landmarks_with_data(lm)
        was_projected_fac = self._is_ellipse_landmark(
            lm
        ) and self._ellipse_is_projected_hemisphere(lm, current_pts.get(lm))

        current_pts.pop(lm, None)
        img_key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None
            else str(self.current_image_path)
        )
        self.landmark_meta.get(img_key, {}).pop(lm, None)
        self.hover_radii.get(img_key, {}).pop(lm, None)

        if self._is_line_landmark(lm):
            self._clear_line_preview()
        if self._is_ellipse_landmark(lm):
            self.dragging_ellipse_handle = None
            self.dragging_ellipse_whole = False
            self.dragging_ellipse_last_img_pos = None
            self._clear_auto_acetabular_points_for_ellipse(lm)
            if was_projected_fac:
                self._clear_projected_hemisphere_ac_for_ellipse(lm)
        if self._matching_ellipse_landmark_for_ac(lm) is not None:
            self._clear_projected_hemisphere_for_ac(lm)
            self._clear_auto_acetabular_points_for_ac(lm)

        if self._is_segmentation_landmark(lm):
            self.last_seed.pop(lm, None)
            mask_key = self._segmentation_mask_key()
            if mask_key is not None:
                getattr(self, "seg_masks", {}).get(mask_key, {}).pop(lm, None)
            self._remove_overlay_for(lm)
        if self._is_border_segmentation_landmark(lm):
            self._clear_it_edge_border_state(lm, clear_annotation=False)

        self._clear_femoral_axis_overlay()
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._refresh_categorical_controls()
        self._refresh_landmark_selectability()
        self._refresh_landmark_label_styles()
        self._refresh_image_listbox()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self._load_note_for_selected_landmark()
        self.dirty = True
        self._autosave_annotation_change()
        self._set_status(
            f"{lm} deleted. Annotate it or set Landmark Status before continuing."
        )
        if show_review_prompt:
            self._prompt_review_subsequent_landmarks(lm, later_to_review)
        return True

    def _on_annotated_checkbox_toggled(self, lm: str) -> None:
        pts, _quality = self._get_annotations()
        found_var = self.landmark_found.get(lm)
        if found_var is not None:
            found_var.set(self._landmark_has_annotation_value(lm))
        self._update_found_checks(pts)

    def _set_note_editor_enabled(self, enabled: bool) -> None:
        if self.note_text is None:
            return

        if enabled:
            self.note_text.configure(
                state="normal",
                bg="white",
                fg="black",
                insertbackground="black",
            )
        else:
            self.note_text.configure(
                state="disabled",
                bg="#E6E6E6",
                fg="#666666",
                insertbackground="#666666",
            )

    def _load_note_for_selected_landmark(self) -> None:
        if self.note_text is None:
            self._refresh_landmark_status_control()
            return

        lm = self.selected_landmark.get()
        can_edit = (
            bool(lm)
            and self._get_landmark_flag(lm)
            and not getattr(self, "multi_review_mode", False)
        )
        note = self._get_landmark_note(lm) if can_edit else ""

        self.note_text_internal_update = True
        self.note_text.configure(state="normal")
        self.note_text.delete("1.0", "end")
        self.note_text.insert("1.0", note)
        self.note_text.edit_modified(False)
        self.note_text_internal_update = False

        self._set_note_editor_enabled(can_edit)
        self._refresh_landmark_status_control()

    def _save_note_for_selected_landmark(self) -> None:
        if self.note_text is None or self.current_image_path is None:
            return

        lm = self.selected_landmark.get()
        if not lm:
            return

        if not self._get_landmark_flag(lm):
            self._set_landmark_note(lm, "")
            return

        note = self.note_text.get("1.0", "end-1c")
        self._set_landmark_note(lm, note)

    def _on_note_text_modified(self, _event=None) -> None:
        if self.note_text is None:
            return
        if getattr(self, "multi_review_mode", False):
            self.note_text.edit_modified(False)
            return
        if self.note_text_internal_update:
            self.note_text.edit_modified(False)
            return
        if not self.note_text.edit_modified():
            return

        self._save_note_for_selected_landmark()
        self.note_text.edit_modified(False)
        self.dirty = True
        self._maybe_autosave_current_image()

    def _refresh_landmark_status_control(self) -> None:
        combo = self.__dict__.get("landmark_status_combo")
        var = self.__dict__.get("landmark_status_var")
        if combo is None or var is None:
            return

        lm = self.selected_landmark.get().strip()
        is_annotated = bool(lm) and self._landmark_annotation_geometry_complete(lm)
        can_edit = (
            bool(lm)
            and self.current_image_path is not None
            and not getattr(self, "multi_review_mode", False)
            and self._landmark_allows_direct_status(lm)
            and not self._landmark_has_annotation_value(lm)
        )
        if is_annotated:
            status = "Annotated"
        elif can_edit:
            status = self._direct_annotation_status_display(
                self._get_landmark_status(lm)
            )
        else:
            status = ""
        var.set(status)
        try:
            combo.configure(state="readonly" if can_edit else "disabled")
        except Exception:
            pass

    def _on_landmark_status_changed(self, _event=None) -> None:
        if getattr(self, "multi_review_mode", False):
            self._refresh_landmark_status_control()
            return
        if self.current_image_path is None:
            self._refresh_landmark_status_control()
            return

        lm = self.selected_landmark.get().strip()
        if not self._landmark_allows_direct_status(lm):
            self._refresh_landmark_status_control()
            return

        status = self._normalize_direct_annotation_status(
            self.landmark_status_var.get()
        )
        if status and self._landmark_has_annotation_value(lm):
            self._set_landmark_status(lm, "")
            self._refresh_landmark_status_control()
            self._set_status(
                f"Delete {lm} before marking it missing or not applicable."
            )
            return

        self._set_landmark_status(lm, status)
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._refresh_landmark_selectability()
        self._refresh_landmark_label_styles()
        self._refresh_image_listbox()
        self.dirty = True
        self._maybe_autosave_current_image()
        self._advance_to_next_protocol_landmark_if_current_complete(lm)

    def _default_categorical_annotations(self) -> Dict[str, Any]:
        return {
            "hip_type": {"left": "", "right": ""},
            "derived_features": {},
        }

    def _parse_categorical_annotations_for_record(self, record: Dict) -> Dict[str, Any]:
        raw = record.get("categories")
        if not isinstance(raw, dict):
            raw = record.get("categorical_annotations")
        if not isinstance(raw, dict):
            return self._default_categorical_annotations()

        parsed = self._default_categorical_annotations()
        raw_hip_type = raw.get("hip_type")
        if isinstance(raw_hip_type, dict):
            for side in ("left", "right"):
                value = str(raw_hip_type.get(side, "") or "").strip().lower()
                parsed["hip_type"][side] = value if value in HIP_CATEGORY_VALUES else ""

        raw_derived = raw.get("derived_features")
        if isinstance(raw_derived, dict):
            for lm, status in raw_derived.items():
                if not isinstance(lm, str):
                    continue
                value = str(status or "").strip().lower()
                if value in DERIVED_FEATURE_STATUS_VALUES and value:
                    parsed["derived_features"][lm] = value
        return parsed

    def _current_categorical_key(self) -> str | None:
        if self.current_image_path is None:
            return None
        if self.json_path is not None:
            return self._path_key(self.current_image_path)
        return str(self.current_image_path)

    def _current_categorical_annotations(self) -> Dict[str, Any]:
        key = self._current_categorical_key()
        if key is None:
            return self._default_categorical_annotations()
        categories = self.categorical_annotations.setdefault(
            key, self._default_categorical_annotations()
        )
        categories.setdefault("hip_type", {"left": "", "right": ""})
        categories.setdefault("derived_features", {})
        return categories

    def _prepare_categorical_data(self) -> Dict[str, Any]:
        categories = self._current_categorical_annotations()
        hip_type = categories.get("hip_type", {})
        derived_features = categories.get("derived_features", {})
        clean_derived = {}
        if isinstance(derived_features, dict):
            for lm, status in derived_features.items():
                value = str(status or "").strip().lower()
                if (
                    isinstance(lm, str)
                    and lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS
                    and value in DERIVED_FEATURE_STATUS_VALUES
                    and value
                ):
                    clean_derived[lm] = value
        return {
            "hip_type": {
                "left": str(hip_type.get("left", "") or "").strip().lower()
                if isinstance(hip_type, dict)
                and str(hip_type.get("left", "") or "").strip().lower()
                in HIP_CATEGORY_VALUES
                else "",
                "right": str(hip_type.get("right", "") or "").strip().lower()
                if isinstance(hip_type, dict)
                and str(hip_type.get("right", "") or "").strip().lower()
                in HIP_CATEGORY_VALUES
                else "",
            },
            "derived_features": clean_derived,
        }

    def _allowed_derived_features_for_current_view(self) -> List[str]:
        allowed = self._get_allowed_landmarks_for_current_view()
        candidates = [
            lm
            for lm in LANDMARK_DISPLAY_ORDER
            if lm in PROTOCOL_DERIVED_FEATURE_LANDMARKS
            and (not allowed or lm in allowed)
            and self._is_protocol_derived_feature(lm)
        ]
        seen: Set[str] = set()
        out: List[str] = []
        for lm in candidates:
            if lm not in seen:
                out.append(lm)
                seen.add(lm)
        return out

    def _refresh_categorical_controls(self) -> None:
        left_hip_var = self.__dict__.get("left_hip_category_var")
        right_hip_var = self.__dict__.get("right_hip_category_var")
        derived_review_var = self.__dict__.get("derived_feature_review_var")
        derived_status_var = self.__dict__.get("derived_feature_status_var")
        if (
            left_hip_var is None
            or right_hip_var is None
            or derived_review_var is None
            or derived_status_var is None
        ):
            return

        readonly = bool(getattr(self, "multi_review_mode", False))
        has_image = self.current_image_path is not None
        categories = self._current_categorical_annotations()
        hip_type = categories.get("hip_type", {})
        if not isinstance(hip_type, dict):
            hip_type = {"left": "", "right": ""}

        left_hip_var.set(self._hip_category_display(hip_type.get("left", "")))
        right_hip_var.set(self._hip_category_display(hip_type.get("right", "")))

        selected_lm = self.selected_landmark.get().strip()
        feature_values = self._relevant_derived_features_for_landmark(selected_lm)
        derived_feature_combo = self.__dict__.get("derived_feature_combo")
        if derived_feature_combo is not None:
            derived_feature_combo["values"] = feature_values

        selected_feature = derived_review_var.get().strip()
        if selected_feature not in feature_values:
            selected_feature = feature_values[0] if feature_values else ""
        derived_review_var.set(selected_feature)

        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            derived = {}
        status = str(derived.get(selected_feature, "") or "") if selected_feature else ""
        derived_status_var.set(self._derived_feature_status_display(status))

        hip_state = "readonly" if has_image and not readonly else "disabled"
        can_classify = (
            has_image
            and not readonly
            and bool(selected_lm)
            and self._landmark_annotation_geometry_complete(selected_lm)
        )
        feature_state = "readonly" if can_classify and feature_values else "disabled"
        status_state = "readonly" if can_classify and bool(selected_feature) else "disabled"
        for widget_name, state in (
            ("left_hip_category_combo", hip_state),
            ("right_hip_category_combo", hip_state),
            ("derived_feature_combo", feature_state),
            ("derived_feature_status_combo", status_state),
        ):
            widget = self.__dict__.get(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except Exception:
                pass

        frame = self.__dict__.get("derived_feature_status_frame")
        if frame is not None:
            for child in frame.winfo_children():
                child.destroy()
            self.derived_feature_status_vars = {}
            self.derived_feature_status_widgets = {}
            for row, feature in enumerate(feature_values):
                var = tk.StringVar(
                    value=self._derived_feature_status_display(derived.get(feature, ""))
                )
                self.derived_feature_status_vars[feature] = var
                tk.Label(frame, text=feature, font=self.dialogue_font).grid(
                    row=row, column=0, sticky="w", padx=(0, 4), pady=2
                )
                combo = ttk.Combobox(
                    frame,
                    textvariable=var,
                    values=DERIVED_FEATURE_STATUS_DISPLAY_VALUES,
                    state="readonly" if can_classify else "disabled",
                    font=self.dialogue_font,
                    width=16,
                    takefocus=False,
                )
                combo.grid(row=row, column=1, sticky="ew", pady=2)
                combo.bind(
                    "<<ComboboxSelected>>",
                    lambda event, feature=feature: self._on_derived_feature_status_changed(
                        event, feature
                    ),
                )
                self.derived_feature_status_widgets[feature] = combo
            try:
                frame.grid_columnconfigure(1, weight=1)
            except Exception:
                pass
        self._refresh_landmark_label_styles()
        self._draw_categorical_shortcut_overlay()

    def _categorical_shortcut_prompt_key(self, prompt: Dict[str, Any]) -> str:
        kind = str(prompt.get("kind", ""))
        if kind == "hip":
            return f"hip:{prompt.get('side', '')}"
        if kind == "landmark_status":
            return f"status:{prompt.get('landmark', '')}"
        if kind == "derived":
            return f"derived:{prompt.get('feature', '')}"
        return kind

    def _categorical_shortcut_prompts(self) -> List[Dict[str, Any]]:
        if self.current_image_path is None or getattr(self, "multi_review_mode", False):
            return []
        if "categorical_annotations" not in self.__dict__:
            return []

        categories = self._current_categorical_annotations()
        hip_type = categories.get("hip_type", {})
        if not isinstance(hip_type, dict):
            hip_type = {}
        hip_options = [
            ("native", "Native"),
            ("prosthetic", "Prosthetic"),
            ("other", "Other"),
        ]
        prompts: List[Dict[str, Any]] = []
        for side, label in (("left", "Left Hip"), ("right", "Right Hip")):
            value = self._normalize_hip_category(hip_type.get(side, ""))
            if not value:
                prompts.append(
                    {
                        "kind": "hip",
                        "side": side,
                        "label": label,
                        "options": hip_options,
                    }
                )

        if prompts:
            return prompts

        selected_lm = self.selected_landmark.get().strip()
        if not selected_lm:
            return []

        if (
            self._landmark_allows_direct_status(selected_lm)
            and not self._landmark_has_annotation_value(selected_lm)
        ):
            status = self._normalize_direct_annotation_status(
                self._get_landmark_status(selected_lm)
            )
            if not status:
                return [
                    {
                        "kind": "landmark_status",
                        "landmark": selected_lm,
                        "label": f"{selected_lm} Landmark Status",
                        "options": [
                            ("missing", "Missing"),
                            ("not_applicable", "Not Applicable"),
                        ],
                    }
                ]

        if not self._landmark_annotation_geometry_complete(selected_lm):
            return []

        derived = categories.get("derived_features", {})
        if not isinstance(derived, dict):
            derived = {}
        status_options = [
            ("correct", "Correct"),
            ("incorrect", "Incorrect"),
            ("missing", "Missing"),
        ]
        for feature in self._relevant_derived_features_for_landmark(selected_lm):
            status = self._normalize_derived_feature_status(derived.get(feature, ""))
            if not status:
                prompts.append(
                    {
                        "kind": "derived",
                        "feature": feature,
                        "label": feature,
                        "options": status_options,
                    }
                )
        return prompts

    def _active_categorical_shortcut_prompt(self) -> Dict[str, Any] | None:
        prompts = self._categorical_shortcut_prompts()
        if not prompts:
            self._categorical_shortcut_focus_key = ""
            return None
        focus_key = self.__dict__.get("_categorical_shortcut_focus_key", "")
        for prompt in prompts:
            if self._categorical_shortcut_prompt_key(prompt) == focus_key:
                return prompt
        self._categorical_shortcut_focus_key = self._categorical_shortcut_prompt_key(
            prompts[0]
        )
        return prompts[0]

    def _draw_categorical_shortcut_overlay(self) -> None:
        canvas = self.__dict__.get("canvas")
        if canvas is None:
            return
        try:
            canvas.delete("category_shortcut")
        except Exception:
            return
        prompt = self._active_categorical_shortcut_prompt()
        if prompt is None:
            return
        options = prompt.get("options", [])
        if not isinstance(options, list):
            return
        lines = [str(prompt.get("label", ""))]
        for index, option in enumerate(options, start=1):
            if (
                isinstance(option, (list, tuple))
                and len(option) >= 2
                and str(option[1]).strip()
            ):
                lines.append(f"{index} {option[1]}")
        text = "\n".join(lines)
        if not text.strip():
            return
        try:
            width = max(int(canvas.winfo_width()), 1)
        except Exception:
            width = 1
        font = self._canvas_landmark_prompt_font()
        x = max(width - 10, 10)
        y = 10
        canvas.create_text(
            x + 1,
            y + 1,
            text=text,
            anchor="ne",
            justify="right",
            fill="white",
            font=font,
            tags="category_shortcut",
        )
        canvas.create_text(
            x,
            y,
            text=text,
            anchor="ne",
            justify="right",
            fill=self._selected_landmark_canvas_text_color(),
            font=font,
            tags="category_shortcut",
        )

    def _set_hip_category_from_shortcut(self, side: str, value: str) -> None:
        display = self._hip_category_display(value)
        if side == "left":
            self.left_hip_category_var.set(display)
        elif side == "right":
            self.right_hip_category_var.set(display)
        else:
            return
        self._on_categorical_annotation_changed()

    def _set_landmark_status_from_shortcut(self, lm: str, status: str) -> None:
        display = self._direct_annotation_status_display(status)
        var = self.__dict__.get("landmark_status_var")
        if var is not None:
            var.set(display)
            self._on_landmark_status_changed()
            return
        self._set_landmark_status(lm, status)
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._refresh_landmark_selectability()
        self._refresh_landmark_label_styles()
        self._refresh_image_listbox()
        self.dirty = True
        self._maybe_autosave_current_image()
        self._advance_to_next_protocol_landmark_if_current_complete(lm)

    def _set_derived_feature_status_from_shortcut(self, feature: str, status: str) -> None:
        display = self._derived_feature_status_display(status)
        var = self.__dict__.get("derived_feature_status_vars", {}).get(feature)
        if var is not None:
            var.set(display)
            self._on_derived_feature_status_changed(feature=feature)
            return
        categories = self._current_categorical_annotations()
        derived = categories.setdefault("derived_features", {})
        if status in DERIVED_FEATURE_STATUS_VALUES and status:
            derived[feature] = status
        else:
            derived.pop(feature, None)
        selected_lm = self.selected_landmark.get().strip()
        self._refresh_categorical_controls()
        self._refresh_landmark_selectability()
        self._refresh_landmark_label_styles()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self._refresh_image_listbox()
        self.dirty = True
        self._maybe_autosave_current_image()
        self._advance_to_next_protocol_landmark_if_current_complete(selected_lm)

    def _apply_categorical_shortcut(self, option_index: int) -> bool:
        prompt = self._active_categorical_shortcut_prompt()
        if prompt is None:
            self._set_status("No categorical annotation is ready for a number shortcut.")
            self._draw_categorical_shortcut_overlay()
            return False
        options = prompt.get("options", [])
        if not isinstance(options, list) or option_index < 1 or option_index > len(options):
            self._set_status(f"Choose 1-{len(options)} for {prompt.get('label', 'this category')}.")
            self._draw_categorical_shortcut_overlay()
            return False
        option = options[option_index - 1]
        if not isinstance(option, (list, tuple)) or len(option) < 2:
            return False
        value = str(option[0])
        label = str(option[1])
        prompt_key = self._categorical_shortcut_prompt_key(prompt)
        if prompt.get("kind") == "hip":
            self._set_hip_category_from_shortcut(str(prompt.get("side", "")), value)
        elif prompt.get("kind") == "landmark_status":
            self._set_landmark_status_from_shortcut(
                str(prompt.get("landmark", "")), value
            )
        elif prompt.get("kind") == "derived":
            self._set_derived_feature_status_from_shortcut(str(prompt.get("feature", "")), value)
        else:
            return False
        prompts = self._categorical_shortcut_prompts()
        next_prompt = None
        for candidate in prompts:
            if self._categorical_shortcut_prompt_key(candidate) != prompt_key:
                next_prompt = candidate
                break
        if next_prompt is None and prompts:
            next_prompt = prompts[0]
        self._categorical_shortcut_focus_key = (
            self._categorical_shortcut_prompt_key(next_prompt) if next_prompt else ""
        )
        self._draw_categorical_shortcut_overlay()
        self._set_status(f"{prompt.get('label', 'Category')}: {label}.")
        return True

    def _make_categorical_shortcut_handler(self, option_index: int):
        def handler(_event) -> str:
            self._apply_categorical_shortcut(option_index)
            return "break"

        return handler

    def _select_next_primary_landmark(self) -> bool:
        order = self._protocol_direct_order_for_current_view()
        if not order:
            order = [
                lm
                for lm in self._display_ordered(getattr(self, "landmarks", []))
                if self._landmark_is_primary_for_panel(lm)
                and self._landmark_is_selectable(lm)
            ]
        else:
            order = [lm for lm in order if self._landmark_is_selectable(lm)]
        if not order:
            return False
        current_primary = self._primary_group_for_landmark(self.selected_landmark.get().strip())
        try:
            idx = order.index(current_primary)
        except ValueError:
            idx = -1
        for step in range(1, len(order) + 1):
            candidate = order[(idx + step) % len(order)]
            if candidate != current_primary or len(order) == 1:
                self.selected_landmark.set(candidate)
                self._on_landmark_selected()
                return True
        return False

    def _on_tab_press(self, _event) -> str:
        if not self._select_next_primary_landmark():
            self._set_status("No primary landmark is available for Tab navigation.")
        return "break"

    def _on_categorical_annotation_changed(self, _event=None) -> None:
        if getattr(self, "multi_review_mode", False):
            self._refresh_categorical_controls()
            return
        categories = self._current_categorical_annotations()
        hip_type = categories.setdefault("hip_type", {"left": "", "right": ""})
        left = self._normalize_hip_category(self.left_hip_category_var.get())
        right = self._normalize_hip_category(self.right_hip_category_var.get())
        hip_type["left"] = left
        hip_type["right"] = right
        self.dirty = True
        self._maybe_autosave_current_image()
        self._build_landmark_panel()
        self._refresh_categorical_controls()
        self._draw_points()
        self._refresh_image_listbox()

    def _on_derived_feature_review_selected(self, _event=None) -> None:
        self._refresh_categorical_controls()

    def _on_derived_feature_status_changed(
        self, _event=None, feature: str | None = None
    ) -> None:
        if getattr(self, "multi_review_mode", False):
            self._refresh_categorical_controls()
            return
        if feature is None:
            feature = self.derived_feature_review_var.get().strip()
            status = self._normalize_derived_feature_status(
                self.derived_feature_status_var.get()
            )
        else:
            var = self.__dict__.get("derived_feature_status_vars", {}).get(feature)
            status = self._normalize_derived_feature_status(
                var.get() if var is not None else ""
            )
        if not feature:
            return
        categories = self._current_categorical_annotations()
        derived = categories.setdefault("derived_features", {})
        if status in DERIVED_FEATURE_STATUS_VALUES and status:
            derived[feature] = status
        else:
            derived.pop(feature, None)
        selected_lm = self.selected_landmark.get().strip()
        self._refresh_categorical_controls()
        self._refresh_landmark_selectability()
        self._refresh_landmark_label_styles()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self._refresh_image_listbox()
        self.dirty = True
        self._maybe_autosave_current_image()
        self._advance_to_next_protocol_landmark_if_current_complete(selected_lm)

    def _bind_shortcut(self, sequence: str, callback) -> None:
        def wrapper(event):
            if self._note_text_shortcuts_blocked():
                return "break"
            return callback(event)

        self.bind(sequence, wrapper)

    def _note_text_shortcuts_blocked(self) -> bool:
        if self.note_text is None:
            return False

        try:
            if str(self.note_text.cget("state")) != "normal":
                return False
        except Exception:
            return False

        focused = self.focus_get() is self.note_text

        try:
            px, py = self.winfo_pointerxy()
            hovered_widget = self.winfo_containing(px, py)
        except Exception:
            hovered_widget = None

        hovered = hovered_widget is self.note_text or (
            hovered_widget is not None
            and str(hovered_widget).startswith(str(self.note_text))
        )

        return focused and hovered

    def _sync_auto_tools_for_selected_landmark(self) -> None:
        lm = self.selected_landmark.get().strip()

        femoral_axis_landmarks = {"L-FA", "R-FA"}

        want_hover = lm in self.HOVER_CIRCLE_LANDMARKS
        want_femoral_axis = lm in femoral_axis_landmarks

        if want_hover:
            if not self.hover_enabled.get():
                self.hover_enabled.set(True)
            if self.femoral_axis_enabled.get():
                self.femoral_axis_enabled.set(False)
            self._toggle_hover()
            self._toggle_femoral_axis()
            return

        if want_femoral_axis:
            if not self.femoral_axis_enabled.get():
                self.femoral_axis_enabled.set(True)
            if self.hover_enabled.get():
                self.hover_enabled.set(False)
            self._toggle_femoral_axis()
            self._toggle_hover()
            return

        changed = False

        if self.hover_enabled.get():
            self.hover_enabled.set(False)
            changed = True

        if self.femoral_axis_enabled.get():
            self.femoral_axis_enabled.set(False)
            changed = True

        if changed:
            self._toggle_hover()
            self._toggle_femoral_axis()
        else:
            self._hide_hover_circle()
            self._clear_femoral_axis_overlay()

    def _change_selected_landmark(self, step: int) -> None:
        if not getattr(self, "landmarks", None):
            return
        if not self.landmarks:
            return
        current = self.selected_landmark.get()
        allowed_set = self._get_allowed_landmarks_for_current_view()
        allowed = self._display_ordered(
            [
                lm
                for lm in self.landmarks
                if lm in allowed_set and self._landmark_is_selectable(lm)
            ]
        )
        if current in allowed:
            idx = allowed.index(current)
        else:
            idx = 0
        idx = idx + step
        if idx < 0 or idx >= len(allowed):
            return
        new_lm = allowed[idx]
        if new_lm != current:
            self.selected_landmark.set(new_lm)
            self._on_landmark_selected()

    def _on_arrow_up(self, event) -> str:
        self._change_selected_landmark(-1)
        return "break"

    def _on_arrow_down(self, event) -> str:
        self._change_selected_landmark(1)
        return "break"

    # Toggles visibility for all landmarks and redraws markers.
    def _set_all_visibility(self, value: bool) -> None:
        for var in self.landmark_visibility.values():
            var.set(value)
        self._draw_points()

    def load_landmarks_from_csv(self, path: Optional[Union[Path, str]] = None) -> None:
        if path is None:
            if not self._maybe_save_before_destructive_action("load point name CSV"):
                return
            self.abs_csv_path = filedialog.askopenfilename(
                initialdir=BASE_DIR, filetypes=[("CSV File", ("*.csv"))]
            )
        else:
            self.abs_csv_path = str(path)

        self.json_path = None
        self.json_dir = None
        self.json_data = {"landmarks": [], "views": {}, "images": []}
        self.allowed_views = {}
        self.images = []
        self.image_index_map = {}
        self.current_image_index = -1
        self.saved_image_snapshots.clear()
        self.annotations.clear()
        self.categorical_annotations.clear()
        self.landmark_meta.clear()
        self.lm_settings.clear()
        self.seg_masks.clear()
        self.it_edge_border_masks.clear()
        self.last_seed.clear()
        self.pelvis_segmentation_undo_stack.clear()
        self.pelvis_segmentation_status_var.set("")
        self.pelvis_segmentation_running = False
        self._refresh_pelvis_segmentation_buttons()
        try:
            self.hover_radii.clear()
        except AttributeError:
            pass
        self.current_view_var.set("")
        if self.view_dropdown is not None:
            self.view_dropdown["values"] = ()
        self.current_image_flag = False

        isolated_data_path = Path(BASE_DIR.parent / "data")
        db_name = extract_filename(Path(self.abs_csv_path).with_suffix(".db"))

        self.db_path = Path(isolated_data_path / db_name)
        self._init_database()
        df: pd.DataFrame = pd.read_csv(self.abs_csv_path)

        self.csv_path_column = self._detect_path_column(df)

        # Removing columns that we know are not landmarks, the rest are assumed to be landmarks
        df.drop(
            columns=["image_quality", self.csv_path_column],
            inplace=True,
            errors="ignore",
        )

        self.landmarks = list(df.columns)
        if self.landmarks:
            self.selected_landmark.set(self.landmarks[0])
        self._build_landmark_panel()
        self._import_csv_to_db()

    def _init_database(self) -> None:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS annotations (
                    image_filename TEXT PRIMARY KEY,
                    image_path TEXT,
                    image_quality INTEGER DEFAULT 0,
                    data BLOB, -- JSON blob of all landmarks
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified INTEGER DEFAULT 0
                )
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS image_filename
                ON annotations(image_filename DESC)
                """)
                cols = [
                    elem[1]
                    for elem in conn.execute("""
                PRAGMA table_info(annotations)
                """)
                ]

                if "verified" not in cols:
                    conn.execute("""
                    ALTER TABLE annotations ADD COLUMN verified INTEGER DEFAULT 0
                    """)
                conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror(
                "Database Error",
                f"Failed to initialize database:\n{e}\n\nAnnotations will not be saved.",
            )
        return

    def _db_is_populated(self) -> bool:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM annotations")
                count = cursor.fetchone()[0]
            return count > 0
        except sqlite3.Error:
            return False

    def _import_csv_to_db(self) -> None:
        try:
            df = pd.read_csv(self.abs_csv_path)
        except Exception as e:
            messagebox.showerror(
                "CSV Error",
                f"Failed to read CSV file:\n{e}",
            )
            return

        col = self._detect_path_column(df)

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    path = str(row[col])
                    try:
                        quality = int(row.get("image_quality", 0))
                    except (ValueError, TypeError):
                        quality = 0
                    landmark_data = {}

                    for lm in self.landmarks:
                        val = row.get(lm, "")
                        if pd.isna(val) or not str(val).strip():
                            continue  # Skip empty values entirely

                        # Parse the string representation back to list
                        try:
                            parsed = ast.literal_eval(str(val))
                            landmark_data[lm] = (
                                parsed  # Store as actual list, not string
                            )
                        except (ValueError, SyntaxError):
                            # If parsing fails, skip this landmark
                            continue

                    cursor.execute(
                        """
                        INSERT INTO annotations (image_filename, image_path, image_quality, data)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(image_filename) DO UPDATE SET
                            image_path    = excluded.image_path,
                            image_quality = excluded.image_quality,
                            data          = excluded.data,
                            modified_at   = CURRENT_TIMESTAMP,
                            verified      = annotations.verified
                        """,
                        (
                            extract_filename(path),
                            path,
                            quality,
                            json.dumps(landmark_data),
                        ),
                    )
                conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror(
                "Database Error",
                f"Failed to import CSV data to database:\n{e}",
            )

    # Prompts to save pending changes before destructive operations.
    def _maybe_save_before_destructive_action(self, why: str = "continue") -> bool:
        if not self.current_image_path:
            return True

        if self._autosave_enabled():
            if not self._flush_pending_autosave():
                return False
            if self._current_image_has_pending_changes():
                return self._save_current_image_silently()
            return True

        if not self._current_image_has_pending_changes():
            return True

        should_save = messagebox.askyesno(
            "Unsaved annotations",
            "You have unsaved annotation changes for this image.\n"
            f"Do you want to save before you {why}?",
        )
        if should_save:
            return self.save_annotations()

        return True

    # Handles window close, offering to save unsaved annotations.
    def _on_close(self) -> None:
        # Flush any pending debounced autosave before the save prompt
        # so that autosave-on users aren't prompted about stale dirty state.
        if self._pending_hover_radius_sync_id is not None:
            self.after_cancel(self._pending_hover_radius_sync_id)
            self._pending_hover_radius_sync_id = None
            self._flush_hover_radius_dependency_refresh()
        if not self._flush_pending_autosave():
            return
        if not self._maybe_save_before_destructive_action("exit"):
            return
        self.window_close_flag = True
        if self._onedrive_backup_timer is not None:
            self.after_cancel(self._onedrive_backup_timer)
            self._onedrive_backup_timer = None
        if self.json_path is not None and self._onedrive_enabled():
            if self._is_upload_in_flight():
                # Wait for in-flight upload to complete (up to 3 seconds)
                for _ in range(30):
                    if not self._is_upload_in_flight():
                        break
                    time.sleep(0.1)
            # Always do a final upload to ensure latest annotations are backed up
            if self.onedrive_backup is not None:
                t = threading.Thread(
                    target=self.onedrive_backup.upload_backup_sync,
                    args=(self.json_path,),
                    daemon=False,
                )
                t.start()
                t.join(timeout=2.0)
        if self.db_path is not None:
            self._export_db_to_csv()
        self.destroy()

    # Opens an image, prepares canvas, and loads saved points.
    def load_image(self) -> None:
        if not self._maybe_save_before_destructive_action("load another image"):
            return
        abs_path = filedialog.askopenfilename(
            initialdir=BASE_DIR,
            filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", ".tif"))],
        )
        if not abs_path:
            return

        if getattr(self, "multi_review_mode", False):
            self._clear_multi_review_state(clear_loaded=True)
        self.absolute_current_image_path = Path(abs_path)
        self.load_image_from_path(Path(abs_path))
        if self.landmarks:
            self.selected_landmark.set(self.landmarks[0])
            self._set_active_landmark_group_from_selection()

        return

    def _get_image_index_from_directory(self) -> Tuple[int, list]:
        if self.absolute_current_image_path is None:
            return 0, []
        # Grab the directory that the current image lives in
        current_image_directory = (
            Path(self.absolute_current_image_path).resolve().parent
        )
        current_image_name = extract_filename(self.absolute_current_image_path)

        all_files = [
            file.name
            for file in current_image_directory.iterdir()
            if file.suffix.lower() in self.possible_image_suffix
        ]

        all_files.sort()

        try:
            idx = all_files.index(current_image_name)
        except ValueError:
            # If current image not found, check case-insensitive match
            current_lower = current_image_name.lower()
            for i, fname in enumerate(all_files):
                if fname.lower() == current_lower:
                    idx = i
                    break
            else:
                raise ValueError(
                    f"Current image '{current_image_name}' not found in directory. "
                    f"Available files: {all_files[:5]}..."  # Show first 5 for debugging
                )

        return idx, all_files

    def _next_image(self) -> None:
        if (
            getattr(self, "multi_review_mode", False) or self.json_path is not None
        ) and self.images:
            if self.current_image_index >= len(self.images) - 1:
                messagebox.showwarning(
                    "End of List", "You have reached the last image in the JSON."
                )
                return

            if not self._maybe_save_before_destructive_action("switch images"):
                return

            self._navigation_in_progress = True
            self._suspend_image_tree_select = True
            try:
                self.current_image_index += 1
                self.load_image_from_path(self.images[self.current_image_index])
            finally:
                self._navigation_in_progress = False
                self._suspend_image_tree_select = False
            return

        # Loads the next image in the current directory
        # self._maybe_save_before_destructive_action("load next image")
        self.save_annotations()
        if self.queue_mode:
            # Queue mode: move backward through unannotated list
            if self.queue_index >= len(self.unannotated_queue) - 1:
                messagebox.showwarning(
                    "Start of Queue",
                    "You're at the beginning of the unannotated images.",
                )
                return

            self.queue_index += 1
            prev_path = self.unannotated_queue[self.queue_index]
            self.absolute_current_image_path = prev_path
            self.load_image_from_path(prev_path)
            self._update_queue_status()

        elif self.check_csv_mode:
            if self.csv_index >= len(self.csv_path_queue) - 1:
                return
            self.csv_index += 1
            prev_path = self.csv_path_queue[self.csv_index]
            self.absolute_current_image_path = prev_path
            self.load_image_from_path(Path(prev_path))
            self._update_queue_status()

        else:
            idx, all_files = self._get_image_index_from_directory()
            current_path = self.absolute_current_image_path
            if current_path is None or not all_files:
                return
            if len(all_files) == idx + 1:
                messagebox.showwarning(
                    "End of Directory",
                    "You've reached the end of the current image directory. "
                    "Use the image list or load another JSON/image set to continue, "
                    "or use 'Prev Image' to move backward.",
                )
            else:
                self.absolute_current_image_path = Path(
                    current_path.resolve().parent / all_files[idx + 1]
                )
                self.load_image_from_path(Path(self.absolute_current_image_path))

    def _prev_image(self) -> None:
        if (
            getattr(self, "multi_review_mode", False) or self.json_path is not None
        ) and self.images:
            if self.current_image_index <= 0:
                messagebox.showwarning(
                    "Beginning of List", "You are at the first image in the JSON."
                )
                return

            if not self._maybe_save_before_destructive_action("switch images"):
                return

            self._navigation_in_progress = True
            self._suspend_image_tree_select = True
            try:
                self.current_image_index -= 1
                self.load_image_from_path(self.images[self.current_image_index])
            finally:
                self._navigation_in_progress = False
                self._suspend_image_tree_select = False
            return

        # Loads the next image in the current directory
        # self._maybe_save_before_destructive_action("load next image")
        self.save_annotations()

        if self.queue_mode:
            # Queue mode: move backward through unannotated list
            if self.queue_index <= 0:
                messagebox.showwarning(
                    "Start of Queue",
                    "You're at the beginning of the unannotated images.",
                )
                return

            self.queue_index -= 1
            prev_path = self.unannotated_queue[self.queue_index]
            self.absolute_current_image_path = prev_path
            self.load_image_from_path(prev_path)
            self._update_queue_status()

        elif self.check_csv_mode:
            if self.csv_index <= 0:
                return
            self.csv_index -= 1
            prev_path = self.csv_path_queue[self.csv_index]
            self.absolute_current_image_path = prev_path
            self.load_image_from_path(Path(prev_path))
            self._update_queue_status()
        else:
            idx, all_files = self._get_image_index_from_directory()
            current_path = self.absolute_current_image_path
            if current_path is None or not all_files:
                return
            if idx == 0:
                messagebox.showwarning(
                    "Beginning of Directory",
                    "You've reached the beginning of the current image directory. "
                    "Use the image list or load another JSON/image set to continue, "
                    "or use 'Next Image' to move forward.",
                )
            else:
                self.absolute_current_image_path = Path(
                    current_path.resolve().parent / all_files[idx - 1]
                )
                self.load_image_from_path(Path(self.absolute_current_image_path))

    def _on_pg_down(self, event) -> None:
        self._next_image()
        return

    def _on_pg_up(self, event) -> None:
        self._prev_image()
        return

    def _on_backspace(self, event) -> None:
        self._delete_current_landmark()
        return

    def _on_1_press(self, event) -> None:
        self._apply_categorical_shortcut(1)
        return

    def _on_2_press(self, event) -> None:
        self._apply_categorical_shortcut(2)
        return

    def _on_3_press(self, event) -> None:
        self._apply_categorical_shortcut(3)
        return

    def _on_4_press(self, event) -> None:
        self._apply_categorical_shortcut(4)
        return

    def change_image_quality(self, val: int) -> None:
        if getattr(self, "multi_review_mode", False):
            return
        self.current_image_quality = val
        if self.current_image_path:
            self.path_var.set(extract_filename(self.current_image_path))
            self.quality_var.set(str(self.current_image_quality))
        self.dirty = True
        self._autosave_annotation_change()
        return

    def _update_path_var(self) -> None:
        if self.current_image_path:
            self.path_var.set(extract_filename(self.current_image_path))
            self.quality_var.set(str(self.current_image_quality))
        self.dirty = True

    def _select_initial_landmark_for_current_image(self) -> None:
        selected = ""
        if getattr(self, "landmarks", None):
            candidate = self._next_incomplete_protocol_direct_landmark()
            if candidate and self._landmark_is_selectable(candidate):
                selected = candidate
            else:
                selectable = [
                    lm
                    for lm in self._protocol_direct_order_for_current_view()
                    if self._landmark_is_selectable(lm)
                ]
                if not selectable:
                    selectable = [
                        lm
                        for lm in self._display_ordered(getattr(self, "landmarks", []))
                        if self._landmark_is_primary_for_panel(lm)
                        and self._landmark_is_selectable(lm)
                    ]
                if selectable:
                    selected = selectable[0]

        self.selected_landmark.set(selected)
        self._set_active_landmark_group_from_selection()
        if selected:
            self._sync_fac_construction_mode_for_selected_landmark(selected)
            self._sync_auto_tools_for_selected_landmark()

    def load_image_from_path(self, path: Path) -> None:
        try:
            self.current_image = Image.open(path)

            if self.current_image.mode == "I;16":
                arr = np.array(self.current_image, dtype=np.uint16)

                lo = np.percentile(arr, 1)
                hi = np.percentile(arr, 99)

                arr = np.clip(arr, lo, hi)
                arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

                img = Image.fromarray(arr, mode="L").convert("RGB")
                self.current_image = img
            self.current_image = self.current_image.convert("RGB")

        except Exception as e:
            messagebox.showerror("Open Image", f"Failed to open image:\n{e}")
            return

        json_record_loaded = getattr(self, "multi_review_mode", False) or (
            self.json_path is not None and self.images
        )
        if self.view_dropdown is not None:
            self.view_dropdown["values"] = list(self.allowed_views.keys())

        if json_record_loaded:
            resolved_path = path.resolve()
            self.current_image_path = resolved_path
            self.absolute_current_image_path = resolved_path
            self.current_image_index = (
                self.images.index(resolved_path) if resolved_path in self.images else -1
            )
            key = self._path_key(resolved_path)
            record = self._get_current_image_record()
            if record is not None:
                self.current_image_flag = bool(record.get("image_flag", False))
                self.image_flag_var.set(self.current_image_flag)
                self.current_image_direction = (
                    record.get("image_direction", "AP") or "AP"
                )
                self.image_direction_var.set(self.current_image_direction)
                self.annotations[key] = self._parse_annotations_for_record(record)
                self.categorical_annotations[key] = (
                    self._parse_categorical_annotations_for_record(record)
                )
            else:
                self.current_image_flag = False
                self.image_flag_var.set(False)
                self.current_image_direction = "AP"
                self.image_direction_var.set("AP")
                self.annotations[key] = {}
                self.landmark_meta[key] = {}
                self.hover_radii[key] = {}
                self.categorical_annotations[key] = (
                    self._default_categorical_annotations()
                )
            self.current_image_quality = 0
            self.current_view_var.set(self._get_current_view())
        else:
            rel_path = PurePath(path.resolve()).relative_to(
                BASE_DIR.resolve(), walk_up=True
            )
            self.current_image_path = Path(rel_path)
            self.absolute_current_image_path = path.resolve()
            self.current_image_flag = False
            self.image_flag_var.set(False)
            self.current_image_direction = "AP"
            self.image_direction_var.set("AP")
            self.current_view_var.set("")

        self._refresh_image_flag_checkbox_style()

        self.canvas.delete("all")
        self.base_img_item = None
        self._remove_all_overlays()
        self.last_seed.clear()
        self._clear_line_preview()
        self._clear_femoral_axis_overlay()
        self.dragging_landmark = None
        self.dragging_point_index = None
        self.dragging_line_whole = False
        self.dragging_line_last_img_pos = None
        self.dragging_ellipse_handle = None
        self.dragging_ellipse_whole = False
        self.dragging_ellipse_last_img_pos = None
        self.drawing_it_edge_border = False
        self.it_edge_border_last_img_pos = None

        current_key = (
            self._path_key(self.current_image_path)
            if json_record_loaded and self.current_image_path is not None
            else str(self.current_image_path)
        )
        self.lm_settings.setdefault(current_key, {})
        self.annotations.setdefault(current_key, {})
        self.hover_radii.setdefault(current_key, {})
        self.categorical_annotations.setdefault(
            current_key, self._default_categorical_annotations()
        )

        if not json_record_loaded:
            self.load_points(show_message=False)

        self._render_base_image()
        self.mouse_crosshair_ids = []
        self.extended_crosshair_ids = []
        self.zoom_extended_crosshair_ids = []
        self._hide_hover_circle()
        self.last_mouse_canvas_pos = None
        self._update_zoom_view(None, None)
        self._hide_zoom_extended_crosshair()
        self.dirty = False
        self._update_path_var()
        self.dirty = False

        if json_record_loaded and not getattr(self, "multi_review_mode", False):
            self._prompt_for_view_if_needed()
        else:
            self._rebuild_landmark_panel_for_view()

        self._select_initial_landmark_for_current_image()
        self._refresh_selected_landmark_group_style()
        self._load_note_for_selected_landmark()
        derived_changed = False
        if not getattr(self, "multi_review_mode", False):
            derived_changed = self._update_auto_acetabular_points_for_current_image(
                snap_if_enabled=json_record_loaded
            )
            protocol_derived_changed = (
                self._update_protocol_derived_features_for_current_image(
                    mark_dirty=False
                )
            )
            if (derived_changed or protocol_derived_changed) and json_record_loaded:
                record = self._get_current_image_record()
                if record is not None:
                    record["annotations"] = self._prepare_landmark_data(for_json=True)
                    record["categories"] = self._prepare_categorical_data()

        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._restore_segmentation_state_for_current_points(pts)
        self._draw_points()
        self._refresh_multi_review_panel()
        self._refresh_categorical_controls()

        if json_record_loaded:
            self._refresh_saved_snapshot_for_current_image()

        self._refresh_image_listbox()
        self._refresh_pelvis_segmentation_buttons()

    def _restore_segmentation_state_for_current_points(
        self, pts: Dict[str, AnnotationValue]
    ) -> None:
        mask_key = self._segmentation_mask_key()
        seg_masks = getattr(self, "seg_masks", {})
        for lm in AUTO_SEGMENTATION_LANDMARKS:
            point = pts.get(lm)
            if not (
                isinstance(point, tuple)
                and len(point) == 2
                and all(isinstance(v, (int, float)) for v in point)
            ):
                self.last_seed.pop(lm, None)
                continue

            sx, sy = point
            self.last_seed[lm] = (int(sx), int(sy))
            has_saved_mask = (
                mask_key is not None and lm in seg_masks.get(mask_key, {})
            )
            if has_saved_mask:
                continue

            vis_var = self.landmark_visibility.get(lm)
            is_visible = vis_var is None or bool(vis_var.get())
            if is_visible and cv2 is not None:
                self._resegment_for(lm, apply_saved_settings=True)
            else:
                self._update_overlay_for(lm)

    def _prepare_landmark_data(self, for_json: Optional[bool] = None) -> dict:
        """Prepare landmark data for JSON or legacy database storage."""
        self._update_protocol_derived_features_for_current_image(mark_dirty=False)
        pts, _quality = self._get_annotations()
        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None and self.current_image_path is not None
            else str(self.current_image_path)
        )
        per_img_settings = self.lm_settings.get(key, {})
        per_img_meta = self.landmark_meta.get(key, {})
        per_img_masks = getattr(self, "seg_masks", {}).get(key, {})
        per_img_border_masks = getattr(self, "it_edge_border_masks", {}).get(key, {})
        use_json_format = self.json_path is not None if for_json is None else for_json

        landmark_data = {}
        all_landmarks = (
            set(pts.keys())
            | set(per_img_meta.keys())
            | set(per_img_masks.keys())
            | set(per_img_border_masks.keys())
        )

        annotation_key_source = {lm: None for lm in all_landmarks}
        for lm in self._annotation_landmark_keys(annotation_key_source):
            if use_json_format and lm not in all_landmarks:
                continue
            if not use_json_format and lm not in pts:
                continue

            meta = per_img_meta.get(lm, {})
            is_flagged = bool(meta.get("flag", False))
            note = str(meta.get("note", ""))
            status = self._normalize_direct_annotation_status(meta.get("status", ""))

            value = None
            if lm in pts:
                status = ""
                if self._is_ellipse_landmark(lm):
                    value = self._normalize_ellipse_value(pts[lm])
                    if value is None:
                        continue
                elif self._is_line_landmark(lm):
                    line_pts = self._get_line_points(lm)
                    if line_pts:
                        value = [[float(x), float(y)] for x, y in line_pts]
                else:
                    point = pts[lm]
                    if not (
                        isinstance(point, tuple)
                        and len(point) == 2
                        and all(isinstance(v, (int, float)) for v in point)
                    ):
                        continue

                    x, y = point
                    if self._is_auto_segmentation_landmark(lm):
                        st = per_img_settings.get(lm, self._current_settings_dict())
                        method_code = "FF" if st["method"] == "Flood Fill" else "ACC"
                        value = [
                            float(x),
                            float(y),
                            method_code,
                            int(st["sens"]),
                            int(st["edge_lock"]),
                            int(st["edge_width"]),
                            int(st["clahe"]),
                            int(st["grow"]),
                        ]
                    else:
                        value = [float(x), float(y)]

            if use_json_format:
                saved_mask = per_img_masks.get(lm)
                saved_border_mask = per_img_border_masks.get(lm)
                if value is None and self._is_border_segmentation_landmark(lm):
                    centroid_mask = (
                        saved_mask if saved_mask is not None else saved_border_mask
                    )
                    centroid = self._mask_centroid_in_image_space(centroid_mask)
                    if centroid is not None:
                        value = [float(centroid[0]), float(centroid[1])]

                has_saved_segmentation = (
                    self._is_segmentation_landmark(lm) and saved_mask is not None
                )
                if (
                    value is None
                    and not is_flagged
                    and not note.strip()
                    and not status
                    and not has_saved_segmentation
                ):
                    continue
                entry: Dict[str, object] = {
                    "value": value,
                    "flag": is_flagged,
                    "note": note,
                }
                if status:
                    entry["status"] = status
                saved_radius = getattr(self, "hover_radii", {}).get(key, {}).get(lm)
                if lm in self.HOVER_CIRCLE_LANDMARKS and saved_radius is not None:
                    entry["radius"] = saved_radius
                if self._is_border_segmentation_landmark(lm) and saved_mask is None:
                    saved_mask = saved_border_mask
                if (
                    self._is_segmentation_landmark(lm)
                    and saved_mask is not None
                ):
                    entry["segmentation"] = self._encode_segmentation_mask(
                        self._normalize_segmentation_mask_for_landmark(lm, saved_mask)
                    )
                if (
                    self._is_border_segmentation_landmark(lm)
                    and saved_border_mask is not None
                ):
                    entry["edge_border"] = self._encode_segmentation_mask(
                        self._normalize_segmentation_mask_for_landmark(
                            lm, saved_border_mask
                        )
                    )
                landmark_data[lm] = entry
            elif value is not None:
                landmark_data[lm] = value

        return landmark_data

    def _canonical_image_state_for_path(self, image_path: Path) -> str:
        key = self._path_key(image_path)
        idx = self.image_index_map.get(key)
        if idx is None:
            return ""

        record = self.json_data["images"][idx]
        state = {
            "image_path": record.get("image_path"),
            "image_flag": bool(record.get("image_flag", False)),
            "view": record.get("view"),
            "annotations": record.get("annotations", {}) or {},
            "categories": self._parse_categorical_annotations_for_record(record),
        }
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    def _current_image_state_string(self) -> str:
        if self.current_image_path is None:
            return ""

        record = self._get_current_image_record()
        state = {
            "image_path": record.get("image_path")
            if record
            else str(self.current_image_path),
            "image_flag": bool(self.current_image_flag),
            "view": self.current_view_var.get().strip() or None,
            "annotations": self._prepare_landmark_data(for_json=True),
            "categories": self._prepare_categorical_data(),
        }
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    def _refresh_saved_snapshot_for_current_image(self) -> None:
        if self.current_image_path is None:
            return
        self.saved_image_snapshots[self._path_key(self.current_image_path)] = (
            self._canonical_image_state_for_path(self.current_image_path)
        )

    def _current_image_has_unsaved_changes(self) -> bool:
        if getattr(self, "multi_review_mode", False):
            return False
        if self.current_image_path is None:
            return False

        key = self._path_key(self.current_image_path)
        current_state = self._current_image_state_string()
        saved_state = self.saved_image_snapshots.get(key)
        if saved_state is None:
            saved_state = self._canonical_image_state_for_path(self.current_image_path)
            self.saved_image_snapshots[key] = saved_state
        return current_state != saved_state

    def _autosave_enabled(self) -> bool:
        autosave_var = getattr(self, "autosave_var", None)
        try:
            return bool(autosave_var.get()) if autosave_var is not None else False
        except Exception:
            return False

    def _current_image_has_pending_changes(self) -> bool:
        if not self.current_image_path:
            return False
        if (
            self.json_path is not None
            and self._get_current_image_record() is not None
        ):
            return self._current_image_has_unsaved_changes()
        return bool(getattr(self, "dirty", False))

    def _save_current_image_silently(self) -> bool:
        if getattr(self, "multi_review_mode", False):
            return True
        if not self.current_image_path:
            return True
        if self.json_path is not None:
            return self._save_json_file(show_success=False)
        if getattr(self, "db_path", None) is not None:
            return self._auto_save_to_db()
        return self.save_annotations()

    def _flush_pending_autosave(self) -> bool:
        pending_id = getattr(self, "_pending_autosave_id", None)
        if pending_id is None:
            return True
        try:
            self.after_cancel(pending_id)
        except Exception:
            pass
        self._pending_autosave_id = None
        return self._execute_autosave()

    def _maybe_autosave_current_image(self) -> bool:
        if getattr(self, "multi_review_mode", False):
            return True
        self.dirty = True
        self._refresh_image_listbox()

        autosave_var = getattr(self, "autosave_var", None)
        if autosave_var is not None and bool(autosave_var.get()):
            # Cancel any previously scheduled autosave to debounce
            if self._pending_autosave_id is not None:
                self.after_cancel(self._pending_autosave_id)
            self._pending_autosave_id = self.after(100, self._execute_autosave)
            return True

        return True

    def _execute_autosave(self) -> bool:
        """Perform the actual autosave after debounce delay."""
        self._pending_autosave_id = None
        ok = self._save_current_image_silently()
        if ok:
            self.dirty = False
        return ok

    def _auto_save_to_db(self) -> bool:
        """
        Auto-save annotations to database immediately after changes.
        Returns True on success, False on failure.
        Silent operation - no user feedback unless error occurs.
        """
        if not self.current_image_path or self.db_path is None:
            return False

        try:
            pts, quality = self._get_annotations()
            landmark_data = self._prepare_landmark_data()

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO annotations (image_filename, image_path, image_quality, data, modified_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(image_filename) DO UPDATE SET
                        image_path    = excluded.image_path,
                        image_quality = excluded.image_quality,
                        data          = excluded.data,
                        modified_at   = CURRENT_TIMESTAMP,
                        verified      = annotations.verified
                    """,
                    (
                        extract_filename(self.current_image_path),
                        str(self.current_image_path),
                        quality,
                        json.dumps(landmark_data),
                    ),
                )
                conn.commit()

            self.dirty = False
            return True

        except sqlite3.Error as e:
            messagebox.showerror(
                "Database Error",
                f"Failed to auto-save annotations:\n{e}\n\nYour changes may not be saved.",
            )
            return False
        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Unexpected error while saving:\n{e}\n\nYour changes may not be saved.",
            )
            return False

    # Writes annotations and segmentation settings back to the CSV.
    def save_annotations(self) -> bool:
        """Save annotations. JSON is the primary format; SQLite/CSV kept for backup."""
        if getattr(self, "multi_review_mode", False):
            messagebox.showinfo(
                "Save",
                "Read-only data cannot be saved.",
            )
            return True
        if not self.current_image_path:
            messagebox.showwarning("Save", "No image loaded.")
            return False

        return self._save_json_file(show_success=(PLATFORM == "Windows"))

        # Get annotation data
        _pts, quality = self._get_annotations()
        landmark_data = self._prepare_landmark_data()

        # Save to database with error handling
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO annotations (image_filename, image_path, image_quality, data, modified_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(image_filename) DO UPDATE SET
                        image_path    = excluded.image_path,
                        image_quality = excluded.image_quality,
                        data          = excluded.data,
                        modified_at   = CURRENT_TIMESTAMP,
                        verified      = annotations.verified
                    """,
                    (
                        extract_filename(self.current_image_path),
                        str(self.current_image_path),
                        quality,
                        json.dumps(landmark_data),
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror(
                "Database Error",
                f"Failed to save annotations to database:\n{e}",
            )
            return False
        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Unexpected error while saving:\n{e}",
            )
            return False

        # Export to CSV (periodic, not every save)
        self._export_db_to_csv()

        self.dirty = False
        if PLATFORM == "Windows":
            messagebox.showinfo("Saved", "Annotations saved")
        return True

    def submit_for_review(self) -> None:
        return

    def _prompt_weird_landmarks(
        self, available_landmarks: List[str]
    ) -> Optional[List[str]]:
        dialog = tk.Toplevel(self)
        dialog.title("Select Landmarks to Review")
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 350) // 2
        dialog.geometry(f"400x350+{x}+{y}")

        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="Select landmarks to highlight in review:",
            font=self.heading_font,
        ).pack(anchor="w", pady=(0, 10))

        canvas_frame = tk.Frame(frame, height=200)
        canvas_frame.pack(fill="x")
        canvas_frame.pack_propagate(False)

        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        var_map: Dict[str, tk.BooleanVar] = {}
        cols = 3
        for i, lm in enumerate(available_landmarks):
            var_map[lm] = tk.BooleanVar(value=False)
            row_frame = tk.Frame(scrollable_frame)
            row_frame.grid(row=i // cols, column=i % cols, sticky="w", padx=5, pady=2)
            tk.Checkbutton(row_frame, text=lm, variable=var_map[lm]).pack(side="left")

        result: Dict[str, Any] = {"selected": None}

        def confirm():
            result["selected"] = [lm for lm, v in var_map.items() if v.get()]
            dialog.grab_release()
            dialog.destroy()

        def cancel():
            dialog.grab_release()
            dialog.destroy()

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", pady=(10, 0))
        tk.Button(
            btn_frame,
            text="Cancel",
            command=cancel,
            font=self.dialogue_font,
            width=12,
        ).pack(side="right", padx=5)
        tk.Button(
            btn_frame,
            text="OK",
            command=confirm,
            font=self.dialogue_font,
            width=12,
        ).pack(side="right")

        dialog.wait_window(dialog)

        return result["selected"]

    def _capture_zoom_views_for_landmarks(
        self, landmarks: List[str], temp_dir: Path, image_name: str
    ) -> List[Path]:
        zoom_paths = []
        pts, _quality = self._get_annotations()

        for lm in landmarks:
            if lm not in pts:
                continue
            point = pts[lm]
            if self._is_ellipse_landmark(lm):
                center = self._ellipse_center_from_value(point)
                if center is None:
                    continue
                x, y = center
                zoom_img = self._generate_zoom_at_point(x, y)
                if zoom_img:
                    zoom_path = temp_dir / f"zoom_{lm}.png"
                    zoom_img.save(zoom_path, "PNG")
                    zoom_paths.append(zoom_path)
                continue
            if self._is_line_landmark(lm):
                continue
            x, y = point[0], point[1]

            xi, yi = self._screen_to_img(float(x), float(y))
            zoom_img = self._generate_zoom_at_point(xi, yi)
            if zoom_img:
                zoom_path = temp_dir / f"zoom_{lm}.png"
                zoom_img.save(zoom_path, "PNG")
                zoom_paths.append(zoom_path)

        return zoom_paths

    def _generate_zoom_at_point(self, xi: float, yi: float) -> Optional["Image.Image"]:
        if self.current_image is None:
            return None

        iw, ih = self.current_image.size
        if not (0 <= xi < iw and 0 <= yi < ih):
            return None

        zoom_lev = max(2, min(40, float(self.zoom_percent.get())))
        half_w = max(1.0, iw / (zoom_lev * 2.0))
        half_h = max(1.0, ih / (zoom_lev * 2.0))

        src_left = float(xi) - half_w
        src_top = float(yi) - half_h
        src_right = float(xi) + half_w
        src_bottom = float(yi) + half_h

        size = 300

        try:
            out = self.current_image.transform(
                (size, size),
                Image.Transform.EXTENT,
                (src_left, src_top, src_right, src_bottom),
                resample=Image.Resampling.BICUBIC,
                fill=0,
            )
        except TypeError:
            out = self.current_image.transform(
                (size, size),
                Image.Transform.EXTENT,
                (src_left, src_top, src_right, src_bottom),
                resample=Image.Resampling.BICUBIC,
            )

        if self.enable_zoom_contrast.get():
            out_gray = out.convert("L")
            img_np = np.array(out_gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(img_np)
            out = Image.fromarray(cl)

        return out

    def _prepare_review_data(
        self, review_note: str, weird_landmarks: List[str]
    ) -> dict:
        """Prepare review submission data JSON."""
        pts, quality = self._get_annotations()
        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None and self.current_image_path is not None
            else str(self.current_image_path)
        )
        per_img_meta = self.landmark_meta.get(key, {})

        # Get all annotations with flags and notes
        annotations_data = {}
        for lm, point in pts.items():
            meta = per_img_meta.get(lm, {})
            is_flagged = bool(meta.get("flag", False))
            note = str(meta.get("note", ""))
            status = self._normalize_direct_annotation_status(meta.get("status", ""))
            if self._is_ellipse_landmark(lm):
                value = self._normalize_ellipse_value(point)
            elif self._is_line_landmark(lm):
                line_pts = self._get_line_points(lm)
                value = (
                    [[float(px), float(py)] for px, py in line_pts]
                    if line_pts
                    else None
                )
            else:
                pt = point
                x, y = pt[0], pt[1]
                value = [float(x), float(y)]
            annotations_data[lm] = {
                "value": value,
                "flag": is_flagged,
                "note": note,
            }
            if status:
                annotations_data[lm]["status"] = status

        return {
            "image_path": str(self.current_image_path),
            "image_name": PurePath(self.current_image_path).name,
            "image_flag": self.current_image_flag,
            "image_direction": self.current_image_direction,
            "image_quality": quality,
            "review_note": review_note,
            "submitted_by": getpass.getuser(),
            "submitted_at": datetime.now().isoformat(),
            "weird_landmarks": weird_landmarks,
            "annotations": annotations_data,
        }

    def _upload_for_review(
        self,
        original_img_path: Path,
        zoom_img_paths: List[Path],
        review_json_path: Path,
        image_name: str,
    ) -> bool:
        """
        Upload review files to OneDrive to_review folder.

        Structure: to_review/<image_name>/<files>
        """
        if not self._onedrive_enabled():
            logger.info("OneDrive disabled; skipping review upload")
            return False
        backup = self.onedrive_backup
        if backup is None:
            return False

        status_dialog = tk.Toplevel(self)
        status_dialog.title("Uploading")
        status_dialog.geometry("300x100")
        status_dialog.resizable(False, False)
        status_dialog.transient(self)
        status_dialog.grab_set()

        status_dialog.update_idletasks()
        x = (status_dialog.winfo_screenwidth() - 300) // 2
        y = (status_dialog.winfo_screenheight() - 100) // 2
        status_dialog.geometry(f"300x100+{x}+{y}")

        frame = tk.Frame(status_dialog, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        status_label = tk.Label(
            frame, text="Uploading to OneDrive...", font=self.dialogue_font
        )
        status_label.pack(pady=(0, 10))

        progress = tk.Canvas(
            frame,
            width=260,
            height=20,
            bg="white",
            highlightthickness=1,
            relief="sunken",
        )
        progress.pack(pady=5)

        canvas_item = progress.create_rectangle(0, 0, 0, 20, fill="#0078D4", outline="")

        def animate():
            current = progress.coords(canvas_item)[2]
            if current >= 260:
                current = 0
            progress.coords(canvas_item, 0, 0, current, 20)
            progress.after(30, animate)

        anim_id = progress.after(30, animate)

        try:
            client = backup._create_fresh_client()
            if not client:
                logger.error("No Graph client available for review upload")
                return False

            loop = asyncio.SelectorEventLoop()

            async def do_upload():
                base_folder = "pelvic-2d-points-backup/to_review"
                img_folder = f"{base_folder}/{image_name}"

                # Ensure folder exists
                await backup._ensure_folder_exists(img_folder)

                async def upload_file(
                    local_path: Optional[Path], remote_name: str
                ) -> bool:
                    if local_path is None:
                        return True
                    try:
                        with open(local_path, "rb") as f:
                            file_content = f.read()
                        drive_item_path = f"root:/{img_folder}/{remote_name}:"
                        await (
                            client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                            .items.by_drive_item_id(drive_item_path)
                            .content.put(file_content)
                        )
                        logger.info(f"Uploaded review file: {remote_name}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to upload {remote_name}: {e}")
                        return False

                # Upload original image and review JSON
                success = await upload_file(
                    original_img_path, f"{image_name}_original.tif"
                )
                success = (
                    await upload_file(review_json_path, f"{image_name}_review.json")
                    and success
                )
                for zoom_path in zoom_img_paths:
                    zoom_name = zoom_path.name
                    success = await upload_file(zoom_path, zoom_name) and success
                return success

            try:
                success = loop.run_until_complete(do_upload())
            finally:
                loop.close()
                progress.after_cancel(anim_id)
                status_dialog.grab_release()
                status_dialog.destroy()
            return success

        except Exception as e:
            logger.error(f"Review upload failed: {e}", exc_info=True)
            progress.after_cancel(anim_id)
            status_dialog.grab_release()
            status_dialog.destroy()
            return False

    def _init_onedrive_credentials(self) -> None:
        """
        Initialize OneDrive credentials at app startup.

        If credentials don't exist, this will trigger the auth dialog.
        Runs in background thread to not block GUI startup.
        """
        if not self._onedrive_enabled():
            return

        def _init():
            try:
                if self.onedrive_backup is not None:
                    self.onedrive_backup._ensure_initialized()
                logger.info("OneDrive credentials initialized")
            except Exception as e:
                logger.warning(f"OneDrive initialization failed: {e}")

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()

    def _backup_original_json_content(self, content: bytes) -> None:
        """
        Backup original JSON content to OneDrive.

        Takes pre-captured content bytes instead of reading from file,
        ensuring the pre-edit state is preserved even after atomic replace.

        Uploads to: pelvic-2d-points-backup/original_jsons/{username}/{filename}_{timestamp}.json
        """
        if self.json_path is None or not self._onedrive_enabled():
            return

        try:
            from auth import get_safe_username

            username = get_safe_username()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = self.json_path.stem
            suffix = self.json_path.suffix
            backup_name = f"{stem}_{timestamp}{suffix}"

            # Upload in background thread
            def _do_backup():
                try:
                    if self.onedrive_backup is None:
                        return
                    client = self.onedrive_backup._create_fresh_client()
                    if not client:
                        logger.warning(
                            "No Graph client available for original JSON backup"
                        )
                        return

                    import asyncio

                    loop = asyncio.SelectorEventLoop()

                    async def upload():
                        remote_folder = (
                            f"pelvic-2d-points-backup/original_jsons/{username}"
                        )
                        drive_item_path = f"root:/{remote_folder}/{backup_name}:"
                        await (
                            client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                            .items.by_drive_item_id(drive_item_path)
                            .content.put(content)
                        )
                        logger.info(f"Backed up original JSON: {backup_name}")

                    try:
                        loop.run_until_complete(upload())
                    finally:
                        loop.close()
                except Exception as e:
                    logger.warning(f"Failed to backup original JSON: {e}")

            thread = threading.Thread(target=_do_backup, daemon=True)
            thread.start()

        except Exception as e:
            logger.warning(f"Failed to initiate original JSON backup: {e}")

    def _backup_original_json(self) -> None:
        """
        Backup the original JSON to OneDrive before first modification.

        Uploads to: pelvic-2d-points-backup/original_jsons/{username}/{filename}_{timestamp}.json
        This preserves the pre-edit state for traceability.
        """
        if (
            self.json_path is None
            or not self.json_path.exists()
            or not self._onedrive_enabled()
        ):
            return

        try:
            from auth import get_safe_username

            username = get_safe_username()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = self.json_path.stem
            suffix = self.json_path.suffix
            backup_name = f"{stem}_{timestamp}{suffix}"

            # Read original content
            with self.json_path.open("rb") as f:
                file_content = f.read()

            # Upload in background thread
            def _do_backup():
                try:
                    if self.onedrive_backup is None:
                        return
                    client = self.onedrive_backup._create_fresh_client()
                    if not client:
                        logger.warning(
                            "No Graph client available for original JSON backup"
                        )
                        return

                    import asyncio

                    loop = asyncio.SelectorEventLoop()

                    async def upload():
                        remote_folder = (
                            f"pelvic-2d-points-backup/original_jsons/{username}"
                        )
                        drive_item_path = f"root:/{remote_folder}/{backup_name}:"
                        await (
                            client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                            .items.by_drive_item_id(drive_item_path)
                            .content.put(file_content)
                        )
                        logger.info(f"Backed up original JSON: {backup_name}")

                    try:
                        loop.run_until_complete(upload())
                    finally:
                        loop.close()
                except Exception as e:
                    logger.warning(f"Failed to backup original JSON: {e}")

            thread = threading.Thread(target=_do_backup, daemon=True)
            thread.start()

        except Exception as e:
            logger.warning(f"Failed to initiate original JSON backup: {e}")

    def _set_upload_in_flight(self, value: bool) -> None:
        """Thread-safe setter for the OneDrive upload in-flight flag."""
        with self._upload_flag_lock:
            self._onedrive_upload_in_flight = value

    def _is_upload_in_flight(self) -> bool:
        """Thread-safe getter for the OneDrive upload in-flight flag."""
        with self._upload_flag_lock:
            return self._onedrive_upload_in_flight

    def _onedrive_enabled(self) -> bool:
        return ONEDRIVE_ENABLED and self.onedrive_backup is not None

    def _schedule_onedrive_backup(self, delay_ms: int = 5000) -> None:
        if not self._onedrive_enabled():
            return
        if self._onedrive_backup_timer is not None:
            self.after_cancel(self._onedrive_backup_timer)
        self._onedrive_backup_timer = self.after(delay_ms, self._fire_onedrive_backup)

    def _fire_onedrive_backup(self) -> None:
        self._onedrive_backup_timer = None
        if (
            self.json_path is None
            or self._is_upload_in_flight()
            or not self._onedrive_enabled()
        ):
            return
        self._set_upload_in_flight(True)
        self._backup_to_onedrive(self.json_path)

    def _backup_to_onedrive(self, *paths: Path) -> None:
        if not self._onedrive_enabled():
            return
        files_to_backup = [p for p in paths if p is not None and p.exists()]
        if not files_to_backup:
            return

        def _on_done(success, total):
            self._set_upload_in_flight(False)
            logger.info(f"OneDrive backup: {success}/{total} files uploaded")

        if self.onedrive_backup is not None:
            self.onedrive_backup.backup_multiple(
                files_to_backup,
                callback=_on_done,
            )

    def _backup_with_progress_dialog(self, files: list) -> None:
        """
        Show a progress dialog while backing up files on close.
        Runs upload in background thread, updates GUI via polling.
        """
        if not self._onedrive_enabled():
            return
        backup = self.onedrive_backup
        if backup is None:
            return
        # Create progress dialog
        dialog = tk.Toplevel(self)
        dialog.title("Backing up to OneDrive...")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 150) // 2
        dialog.geometry(f"400x150+{x}+{y}")

        # Prevent closing via X button
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)

        status_var = tk.StringVar(value="Uploading to OneDrive...")
        status_label = ttk.Label(frame, textvariable=status_var, font=("Helvetica", 12))
        status_label.pack(pady=(0, 15))

        # Progress bar (indeterminate mode)
        progress = ttk.Progressbar(frame, mode="indeterminate", length=300)
        progress.pack(pady=10)
        progress.start(10)

        file_var = tk.StringVar(value="")
        file_label = ttk.Label(
            frame, textvariable=file_var, font=("Helvetica", 9), foreground="gray"
        )
        file_label.pack(pady=(10, 0))

        # Shared state for thread communication
        upload_state = {"done": False, "success": 0, "total": len(files), "current": ""}

        def do_upload():
            """Run uploads in background thread."""
            for fpath in files:
                if fpath.exists():
                    upload_state["current"] = fpath.name
                    try:
                        if backup.upload_backup_sync(fpath):
                            upload_state["success"] += 1
                    except Exception as e:
                        logger.warning(f"OneDrive backup failed for {fpath}: {e}")
            upload_state["done"] = True

        def check_progress():
            """Poll upload state and update GUI."""
            if upload_state["current"]:
                file_var.set(f"Uploading: {upload_state['current']}")

            if upload_state["done"]:
                progress.stop()
                dialog.destroy()
            else:
                # Check again in 100ms
                dialog.after(100, check_progress)

        # Start upload thread
        upload_thread = threading.Thread(target=do_upload, daemon=True)
        upload_thread.start()

        # Start polling
        check_progress()

        # Wait for dialog to close (blocking but GUI-responsive)
        self.wait_window(dialog)

    def _export_db_to_csv(self) -> None:
        """Export database to CSV file with atomic write (temp file + rename)."""
        current_time = datetime.now()
        # save every so often, or save when the window is being closed
        if ((current_time - self.last_update).total_seconds() > 20) or (
            self.window_close_flag
        ):
            self.last_update = datetime.now()

            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT image_path, image_quality, data FROM annotations"
                    )
                    rows = cursor.fetchall()
            except sqlite3.Error as e:
                messagebox.showerror(
                    "Database Error",
                    f"Failed to read annotations from database:\n{e}",
                )
                return

            # Build DataFrame
            records = []
            for path, quality, data_json in rows:
                if path == None:
                    continue
                if quality == None:
                    continue
                if data_json == None:
                    continue

                record = {
                    self.csv_path_column: Path(path).resolve(),
                    "image_quality": quality,
                }

                try:
                    landmark_data = json.loads(data_json) if data_json else {}
                except json.JSONDecodeError:
                    landmark_data = {}

                for lm in self.landmarks:
                    if lm in landmark_data and landmark_data[lm]:
                        record[lm] = repr(landmark_data[lm])
                    else:
                        record[lm] = ""

                records.append(record)

            df = pd.DataFrame(records)

            # Ensure column order
            cols = [self.csv_path_column, "image_quality"] + self.landmarks
            df = df[[c for c in cols if c in df.columns]]

            # Determine final CSV path
            if self.abs_csv_path is None:
                return
            csv_path = Path(self.abs_csv_path)
            if self.check_csv_mode:
                dir = csv_path.parent
                csv_name = Path(extract_filename(csv_path))
                new_name = csv_name.stem + "_CHECKED.csv"
                csv_path = Path(dir / new_name)

            # Atomic write: write to temp file, then rename
            # This prevents corruption if the app crashes mid-write
            temp_path = csv_path.with_suffix(".csv.tmp")
            try:
                df.to_csv(str(temp_path), index=False)
                # Atomic rename (on same filesystem)
                temp_path.replace(csv_path)

                # Backup to OneDrive after successful local save
                self._backup_to_onedrive(csv_path)

            except OSError as e:
                messagebox.showerror(
                    "CSV Export Error",
                    f"Failed to save CSV file:\n{e}\n\nYour data is safe in the database.",
                )
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

    def _get_annotations(self) -> Tuple[Dict[str, AnnotationValue], int]:
        """Returns (points_dict, quality) for current image."""
        if self.current_image_path is not None:
            current_filename = PurePath(self.current_image_path).name
            keys_to_try = [str(self.current_image_path)]
            if self.json_path is not None:
                keys_to_try.insert(0, self._path_key(self.current_image_path))

            for key in keys_to_try:
                if key in self.annotations:
                    quality = self.current_image_quality
                    return self.annotations[key], quality

            # Fallback: match by filename
            for key in self.annotations.keys():
                if current_filename in key:
                    quality = self.current_image_quality
                    return self.annotations[key], quality

        return {}, 0

    def load_points(self, show_message: bool = True) -> None:
        if not self.current_image_path:
            if show_message:
                messagebox.showwarning("Load Points", "No image loaded.")
            return
        if self.abs_csv_path is None:
            if show_message:
                messagebox.showerror("Load Points", "CSV path is not configured.")
            return
        if not Path(self.abs_csv_path).exists():
            if show_message:
                messagebox.showerror(
                    "Load Points", f"CSV not found: {self.abs_csv_path}"
                )
            return
        try:
            df = pd.read_csv(self.abs_csv_path)
        except Exception as e:
            if show_message:
                messagebox.showerror("Load Points", f"Failed to read CSV:\n{e}")
            return
        if df.empty:
            self.annotations[str(self.current_image_path)] = {}
            self._update_found_checks({})
            if show_message:
                messagebox.showinfo("Load Points", "No saved points for this image.")
            return
        # We know that this is image_path
        # df_img_path_col = "image_path"  # this is just grabbing image_path
        df_img_path_col = self._detect_path_column(df)

        # rowdf = df.loc[df[col0] == self.current_image_path]
        rowdf = df.loc[df[df_img_path_col] == str(self.current_image_path)]
        if rowdf.empty:
            # Check if the current image filename is in any of the rows
            # Only if that fails do we reset the points values
            current_filename = PurePath(self.current_image_path).name
            found_filename = False
            for name in list(df[df_img_path_col]):
                if current_filename in name:
                    rowdf = df.loc[df[df_img_path_col] == name]
                    found_filename = True
                    break

            if found_filename is False:
                self.annotations[str(self.current_image_path)] = {}
                self.hover_radii.pop(str(self.current_image_path), None)
                self._update_found_checks({})
                if show_message:
                    messagebox.showinfo(
                        "Load Points", "No saved points for this image."
                    )
                return
        row: pd.DataFrame = rowdf.iloc[0]
        pts = {}
        per_img_settings = self.lm_settings.setdefault(str(self.current_image_path), {})
        self.hover_radii.pop(str(self.current_image_path), None)
        for lm in self.landmarks:
            val = row.get(lm, "")
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    arr = ast.literal_eval(val)
                except Exception:
                    continue
                if self._is_ellipse_landmark(lm):
                    ellipse = self._normalize_ellipse_value(arr)
                    if ellipse is not None:
                        pts[lm] = ellipse
                    continue
                if self._is_line_landmark(lm):
                    line_pts: List[Tuple[float, float]] = []
                    if isinstance(arr, (list, tuple)):
                        for point in arr[:2]:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                try:
                                    line_pts.append((float(point[0]), float(point[1])))
                                except Exception:
                                    continue
                    if line_pts:
                        pts[lm] = line_pts
                    continue
                try:
                    x, y = float(arr[0]), float(arr[1])
                    pts[lm] = (x, y)
                except Exception:
                    continue
                if self._is_auto_segmentation_landmark(lm):
                    settings = self._settings_from_segmentation_value(arr)
                    if settings is not None:
                        per_img_settings[lm] = settings
        self.annotations[str(self.current_image_path)] = pts
        try:
            self.current_image_quality = int(row.get("image_quality", 0))
        except (ValueError, TypeError):
            self.current_image_quality = 0
        self._update_found_checks(pts)
        self._draw_points()
        for lm in AUTO_SEGMENTATION_LANDMARKS:
            if lm in pts:
                try:
                    point = pts[lm]
                    if not (
                        isinstance(point, tuple)
                        and len(point) == 2
                        and all(isinstance(v, (int, float)) for v in point)
                    ):
                        raise ValueError(f"{lm} seed must be a point landmark")
                    sx, sy = point
                    self.last_seed[lm] = (int(sx), int(sy))
                except Exception:
                    self.last_seed.pop(lm, None)
                vis_var = self.landmark_visibility.get(lm)
                if (
                    self.last_seed.get(lm) is not None
                    and vis_var is not None
                    and vis_var.get()
                    and cv2 is not None
                ):
                    self._resegment_for(lm, apply_saved_settings=True)
                else:
                    self._update_overlay_for(lm)
        if show_message:
            messagebox.showinfo("Load Points", "Points loaded from CSV.")
        self._update_path_var()

    # Updates landmark completion state from available annotation geometry.
    def _update_found_checks(self, pts_dict):
        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None and self.current_image_path is not None
            else str(self.current_image_path)
        )
        meta = self.landmark_meta.get(key, {})
        multi_image_key = None
        if (
            getattr(self, "multi_review_mode", False)
            and self.current_image_path is not None
        ):
            multi_image_key = self.multi_review_image_key_by_path.get(
                self._path_key(self.current_image_path)
            )

        for lm in self.landmarks:
            var = self.landmark_found.get(lm)
            if var is not None:
                if multi_image_key is not None:
                    is_found = any(
                        lm
                        in self.multi_review_annotations.get(multi_image_key, {}).get(
                            reviewer, {}
                        )
                        for reviewer in self.multi_review_reviewer_order
                    )
                else:
                    is_found = self._landmark_annotation_geometry_complete(lm)
                var.set(is_found)

            fvar = self.landmark_flagged.get(lm)
            fwidget = getattr(self, "landmark_flag_widgets", {}).get(lm)
            if multi_image_key is not None:
                is_flagged = any(
                    bool(
                        self.multi_review_meta.get(multi_image_key, {})
                        .get(reviewer, {})
                        .get(lm, {})
                        .get("flag", False)
                    )
                    for reviewer in self.multi_review_reviewer_order
                )
            else:
                is_flagged = bool(meta.get(lm, {}).get("flag", False))
            if fvar is not None:
                fvar.set(is_flagged)
            if fwidget is not None:
                try:
                    fwidget.configure(
                        fg="red" if is_flagged else "black",
                        activeforeground="red" if is_flagged else "black",
                        selectcolor="#FFB6B6" if is_flagged else fwidget.cget("bg"),
                    )
                except Exception:
                    pass

        self._refresh_landmark_selectability()
        self._refresh_landmark_status_control()
        self._refresh_categorical_controls()
        self._refresh_landmark_categorical_count_labels()

    def _multi_review_color(self, reviewer: str) -> str:
        try:
            idx = self.multi_review_reviewer_order.index(reviewer)
        except ValueError:
            idx = 0
        return MULTI_REVIEW_COLORS[idx % len(MULTI_REVIEW_COLORS)]

    def _draw_multi_review_points(self) -> None:
        if not self.current_image_path:
            return

        image_key = self.multi_review_image_key_by_path.get(
            self._path_key(self.current_image_path)
        )
        if not image_key:
            return

        selected_lm = self.selected_landmark.get()
        offsets = [
            (-8, -8),
            (8, -8),
            (-8, 8),
            (8, 8),
            (0, -13),
            (0, 13),
            (13, 0),
            (-13, 0),
        ]
        font = self.dialogue_font
        selected_font = self.landmark_font
        shadow_font = font.copy()
        shadow_font.configure(size=font.cget("size") + 1)

        for reviewer_idx, reviewer in enumerate(self.multi_review_reviewer_order):
            if not self._is_multi_review_reviewer_visible(reviewer):
                continue
            color = self._multi_review_color(reviewer)
            dx, dy = offsets[reviewer_idx % len(offsets)]
            reviewer_pts = self.multi_review_annotations.get(image_key, {}).get(
                reviewer, {}
            )
            reviewer_meta = self.multi_review_meta.get(image_key, {}).get(reviewer, {})

            for lm, val in reviewer_pts.items():
                vis_var = self.landmark_visibility.get(lm)
                if vis_var is not None and not vis_var.get():
                    continue

                is_selected = lm == selected_lm
                is_flagged = bool(reviewer_meta.get(lm, {}).get("flag", False))
                label_text = lm if is_selected else ""
                if is_flagged:
                    label_text = f"!{label_text}" if label_text else "!"
                label_font = selected_font if is_selected else font
                marker_width = 3 if is_selected else 2

                if self._is_ellipse_landmark(lm):
                    if self._normalize_ellipse_value(val) is None:
                        continue
                    self._draw_ellipse_annotation(
                        self.canvas,
                        val,
                        self._img_to_screen,
                        outline=color,
                        label_fill=color,
                        label=label_text,
                        font=label_font,
                        shadow_font=shadow_font,
                        width=marker_width,
                        handle_radius=5 if is_selected else 4,
                        tags="marker",
                        label_offset=(dx, dy),
                        show_handles=is_selected,
                    )
                    self._draw_projected_hemisphere_boundary(
                        self.canvas,
                        lm,
                        self._img_to_screen,
                        ellipse=val,
                        width=marker_width,
                        tags="marker",
                    )
                    continue

                if self._is_line_landmark(lm):
                    if not isinstance(val, list) or not val:
                        continue
                    screen_pts = [self._img_to_screen(x, y) for x, y in val]
                    if len(screen_pts) == 2:
                        xs1, ys1 = screen_pts[0]
                        xs2, ys2 = screen_pts[1]
                        self.canvas.create_line(
                            xs1,
                            ys1,
                            xs2,
                            ys2,
                            fill=color,
                            width=marker_width,
                            tags="marker",
                        )
                        label_x = (xs1 + xs2) / 2 + dx
                        label_y = (ys1 + ys2) / 2 + dy - 10
                    else:
                        label_x = screen_pts[0][0] + dx
                        label_y = screen_pts[0][1] + dy - 10

                    for xs, ys in screen_pts:
                        r = 5 if is_selected else 4
                        self.canvas.create_rectangle(
                            xs - r,
                            ys - r,
                            xs + r,
                            ys + r,
                            outline=color,
                            width=marker_width,
                            tags="marker",
                        )

                    if label_text:
                        self.canvas.create_text(
                            label_x - 1,
                            label_y - 1,
                            text=label_text,
                            fill="black",
                            font=shadow_font,
                            tags="marker",
                        )
                        self.canvas.create_text(
                            label_x,
                            label_y,
                            text=label_text,
                            fill=color,
                            font=label_font,
                            tags="marker",
                        )
                    continue

                if not (
                    isinstance(val, tuple)
                    and len(val) == 2
                    and all(isinstance(v, (int, float)) for v in val)
                ):
                    continue

                x, y = val
                xs, ys = self._img_to_screen(x, y)
                r = 6 if is_selected else 4
                self.canvas.create_oval(
                    xs - r,
                    ys - r,
                    xs + r,
                    ys + r,
                    outline=color,
                    width=marker_width,
                    tags="marker",
                )
                self.canvas.create_line(
                    xs - r - 2,
                    ys,
                    xs + r + 2,
                    ys,
                    fill=color,
                    width=marker_width,
                    tags="marker",
                )
                self.canvas.create_line(
                    xs,
                    ys - r - 2,
                    xs,
                    ys + r + 2,
                    fill=color,
                    width=marker_width,
                    tags="marker",
                )
                label_x = xs + dx + 7
                label_y = ys + dy - 8
                if label_text:
                    self.canvas.create_text(
                        label_x - 1,
                        label_y - 1,
                        text=label_text,
                        fill="black",
                        font=shadow_font,
                        anchor="w",
                        tags="marker",
                    )
                    self.canvas.create_text(
                        label_x,
                        label_y,
                        text=label_text,
                        fill=color,
                        font=label_font,
                        anchor="w",
                        tags="marker",
                    )

        temp_pts = self.multi_review_temp_annotations.get(image_key, {})
        for lm, val in temp_pts.items():
            vis_var = self.landmark_visibility.get(lm)
            if vis_var is not None and not vis_var.get():
                continue

            if self._is_line_landmark(lm):
                if not isinstance(val, list) or not val:
                    continue
                screen_pts = [self._img_to_screen(x, y) for x, y in val]
                if len(screen_pts) == 2:
                    self.canvas.create_line(
                        screen_pts[0][0],
                        screen_pts[0][1],
                        screen_pts[1][0],
                        screen_pts[1][1],
                        fill="white",
                        width=3,
                        dash=(5, 3),
                        tags="marker",
                    )
                for xs, ys in screen_pts:
                    r = 7
                    self.canvas.create_rectangle(
                        xs - r,
                        ys - r,
                        xs + r,
                        ys + r,
                        outline="white",
                        width=3,
                        tags="marker",
                    )
                if screen_pts:
                    self.canvas.create_text(
                        screen_pts[0][0] + 9,
                        screen_pts[0][1] - 14,
                        text=f"temp:{lm}",
                        fill="white",
                        font=selected_font,
                        anchor="w",
                        tags="marker",
                    )
                continue

            if self._is_ellipse_landmark(lm):
                if self._normalize_ellipse_value(val) is None:
                    continue
                self._draw_ellipse_annotation(
                    self.canvas,
                    val,
                    self._img_to_screen,
                    outline="white",
                    label_fill="white",
                    label=f"temp:{lm}",
                    font=selected_font,
                    width=3,
                    handle_radius=7,
                    tags="marker",
                    label_offset=(8, -8),
                    show_handles=True,
                )
                self._draw_projected_hemisphere_boundary(
                    self.canvas,
                    lm,
                    self._img_to_screen,
                    ellipse=val,
                    width=3,
                    tags="marker",
                )
                continue

            center = self._landmark_center_from_value(lm, val)
            if center is None:
                continue
            xs, ys = self._img_to_screen(center[0], center[1])
            r = 8
            self.canvas.create_oval(
                xs - r,
                ys - r,
                xs + r,
                ys + r,
                outline="white",
                width=3,
                tags="marker",
            )
            self.canvas.create_line(
                xs - r - 3,
                ys,
                xs + r + 3,
                ys,
                fill="white",
                width=3,
                tags="marker",
            )
            self.canvas.create_line(
                xs,
                ys - r - 3,
                xs,
                ys + r + 3,
                fill="white",
                width=3,
                tags="marker",
            )
            self.canvas.create_text(
                xs + 10,
                ys - 12,
                text=f"temp:{lm}",
                fill="white",
                font=selected_font,
                anchor="w",
                tags="marker",
            )

    def _draw_protocol_pod_construction_overlay(
        self,
        canvas,
        point_transform,
        *,
        tags,
        draw_pod_line: bool = False,
        width: int = 2,
    ) -> List[int]:
        geometry = self._current_protocol_pod_construction()
        if geometry is None:
            return []
        if not self._current_view_allows_landmark("POD"):
            return []
        vis_var = self.landmark_visibility.get("POD")
        if vis_var is not None and not vis_var.get():
            return []

        def point_value(name: str) -> AnnotationPoint | None:
            raw = geometry.get(name)
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None
            try:
                return float(raw[0]), float(raw[1])
            except (TypeError, ValueError):
                return None

        def point_pair(name: str) -> Tuple[AnnotationPoint, AnnotationPoint] | None:
            raw = geometry.get(name)
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None
            first_raw, second_raw = raw[0], raw[1]
            if (
                not isinstance(first_raw, (list, tuple))
                or not isinstance(second_raw, (list, tuple))
                or len(first_raw) < 2
                or len(second_raw) < 2
            ):
                return None
            try:
                return (
                    (float(first_raw[0]), float(first_raw[1])),
                    (float(second_raw[0]), float(second_raw[1])),
                )
            except (TypeError, ValueError):
                return None

        tag_values = (tags,) if isinstance(tags, str) else tuple(tags)
        tag_values = tag_values + ("pod_construction",)
        item_ids: List[int] = []
        axis = point_pair("axis")
        left_lpo = point_value("left_lpo")
        right_lpo = point_value("right_lpo")
        left_axis = point_value("left_axis_point")
        right_axis = point_value("right_axis_point")
        midpoint = point_value("midpoint")
        pod = point_pair("pod")

        if axis is not None:
            (x0, y0), (x1, y1) = axis
            sx0, sy0 = point_transform(x0, y0)
            sx1, sy1 = point_transform(x1, y1)
            item_ids.append(
                canvas.create_line(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    fill=PROTOCOL_POD_MEDIAL_AXIS_COLOR,
                    width=max(1, width),
                    dash=(6, 4),
                    tags=tag_values,
                )
            )

        if left_axis is not None and right_axis is not None:
            lx, ly = point_transform(*left_axis)
            rx, ry = point_transform(*right_axis)
            item_ids.append(
                canvas.create_line(
                    lx,
                    ly,
                    rx,
                    ry,
                    fill=PROTOCOL_POD_MEDIAL_AXIS_COLOR,
                    width=max(1, width + 1),
                    tags=tag_values,
                )
            )

        for lpo, axis_point in (
            (left_lpo, left_axis),
            (right_lpo, right_axis),
        ):
            if lpo is None or axis_point is None:
                continue
            x0, y0 = point_transform(*lpo)
            x1, y1 = point_transform(*axis_point)
            item_ids.append(
                canvas.create_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=PROTOCOL_POD_HELPER_COLOR,
                    width=max(1, width - 1),
                    dash=(3, 3),
                    tags=tag_values,
                )
            )
            r = 3 + width
            item_ids.append(
                canvas.create_oval(
                    x1 - r,
                    y1 - r,
                    x1 + r,
                    y1 + r,
                    outline=PROTOCOL_POD_HELPER_COLOR,
                    width=max(1, width - 1),
                    tags=tag_values,
                )
            )

        if midpoint is not None:
            mx, my = point_transform(*midpoint)
            r = 4 + width
            item_ids.append(
                canvas.create_oval(
                    mx - r,
                    my - r,
                    mx + r,
                    my + r,
                    fill=PROTOCOL_POD_MEDIAL_AXIS_COLOR,
                    outline="black",
                    width=1,
                    tags=tag_values,
                )
            )

        if draw_pod_line and pod is not None:
            (x0, y0), (x1, y1) = pod
            sx0, sy0 = point_transform(x0, y0)
            sx1, sy1 = point_transform(x1, y1)
            color = self._zoom_overlay_color_for_landmark("POD")
            item_ids.append(
                canvas.create_line(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    fill=color,
                    width=max(2, width + 1),
                    tags=tag_values,
                )
            )

        return item_ids

    def _selected_landmark_is_labeled(
        self, pts: Dict[str, AnnotationValue] | None = None
    ) -> bool:
        lm = self.selected_landmark.get().strip()
        if not lm:
            return False
        if self._is_ellipse_landmark(lm):
            return self._get_ellipse(lm) is not None
        if self._is_line_landmark(lm):
            return len(self._get_line_points(lm)) > 0
        if self._is_border_segmentation_landmark(lm):
            return self._landmark_annotation_geometry_complete(lm)
        if pts is None:
            pts, _quality = self._get_annotations()
        return lm in pts

    def _selected_landmark_canvas_text_color(
        self, pts: Dict[str, AnnotationValue] | None = None
    ) -> str:
        return "#FFCC66" if self._selected_landmark_is_labeled(pts) else "#FF8066"

    def _canvas_landmark_prompt_font(self):
        font = self.__dict__.get("landmark_font") or self.__dict__.get("dialogue_font")
        if font is None:
            return None
        try:
            prompt_font = font.copy()
            prompt_font.configure(size=24)
            return prompt_font
        except Exception:
            return font

    # Draws landmark markers/labels and syncs overlays and pair lines.
    def _draw_points(self, *, lightweight: bool = False) -> None:
        self.canvas.delete("marker")
        if not self.current_image_path:
            self.canvas.delete("category_shortcut")
            self._clear_line_preview()
            self._clear_femoral_axis_overlay()
            return
        # pts = self.annotations.get(self.current_image_path, {})
        if not lightweight:
            self._update_protocol_derived_features_for_current_image(mark_dirty=False)
        pts, quality = self._get_annotations()
        if not lightweight:
            self._update_found_checks(pts)
            self._refresh_landmark_selectability()
        current_image_verified = self._is_current_image_verified()
        x_curr, y_curr = self._img_to_screen(0, 0)
        label_font = self._canvas_landmark_prompt_font()

        selected_lm = self.selected_landmark.get()

        if selected_lm:
            self.canvas.create_text(
                x_curr,
                y_curr,
                text=selected_lm,
                fill=self._selected_landmark_canvas_text_color(pts),
                font=label_font,
                tags="marker",
                anchor="nw",
            )

        key = (
            self._path_key(self.current_image_path)
            if self.json_path is not None
            else str(self.current_image_path)
        )
        meta = self.landmark_meta.get(key, {})

        self._draw_protocol_pod_construction_overlay(
            self.canvas,
            self._img_to_screen,
            tags="marker",
            draw_pod_line=False,
            width=2,
        )

        for lm, val in pts.items():
            if self._is_pelvis_segmentation_landmark(lm):
                continue
            if not self._landmark_should_render_on_canvas(lm):
                continue
            vis_var = self.landmark_visibility.get(lm)
            if vis_var is not None and not vis_var.get():
                continue

            drawing_current_selected = lm == self.selected_landmark.get()
            is_flagged = bool(meta.get(lm, {}).get("flag", False))
            oval_color, text_color = self._canvas_colors_for_landmark(
                lm,
                selected=drawing_current_selected,
                flagged=is_flagged,
                image_verified=current_image_verified,
            )
            font = self._label_font_for_landmark(
                lm, selected=drawing_current_selected
            )
            shadow_font = font.copy()
            shadow_font.configure(size=font.cget("size") + 1)

            if self._is_ellipse_landmark(lm):
                ellipse = self._normalize_ellipse_value(val)
                if ellipse is None:
                    continue

                line_color = oval_color
                self._draw_ellipse_annotation(
                    self.canvas,
                    ellipse,
                    self._img_to_screen,
                    outline=line_color,
                    label_fill=text_color,
                    label=lm,
                    font=font,
                    shadow_font=shadow_font,
                    width=2,
                    handle_radius=6 if drawing_current_selected else 5,
                    tags="marker",
                    show_handles=True,
                )
                self._draw_projected_hemisphere_boundary(
                    self.canvas,
                    lm,
                    self._img_to_screen,
                    ellipse=ellipse,
                    width=3 if drawing_current_selected else 2,
                    tags="marker",
                )
                self._draw_acetabular_lower_arc_highlights(
                    self.canvas,
                    lm,
                    self._img_to_screen,
                    width=3 if drawing_current_selected else 2,
                    tags="marker",
                )
                if drawing_current_selected:
                    self._draw_fac_angle_readout(
                        self.canvas,
                        ellipse,
                        self._img_to_screen,
                        font=self.dialogue_font,
                        tags="marker",
                        fill="#FFCC66",
                    )
                continue

            if self._is_line_landmark(lm):
                line_pts = self._get_line_points(lm)
                if not line_pts:
                    continue

                screen_pts = [self._img_to_screen(x, y) for x, y in line_pts]

                line_color = oval_color
                if len(screen_pts) == 2:
                    xs1, ys1 = screen_pts[0]
                    xs2, ys2 = screen_pts[1]
                    self.canvas.create_line(
                        xs1,
                        ys1,
                        xs2,
                        ys2,
                        fill=line_color,
                        width=2,
                        tags="marker",
                    )
                    label_x = (xs1 + xs2) / 2
                    label_y = (ys1 + ys2) / 2 - 14
                else:
                    label_x, label_y = screen_pts[0][0], screen_pts[0][1] - 14

                for xs, ys in screen_pts:
                    r = 5
                    self.canvas.create_oval(
                        xs - r,
                        ys - r,
                        xs + r,
                        ys + r,
                        outline=line_color,
                        width=2,
                        tags="marker",
                    )

                self.canvas.create_text(
                    label_x - 1,
                    label_y - 1,
                    text=lm,
                    fill="black",
                    font=shadow_font,
                    tags="marker",
                )
                self.canvas.create_text(
                    label_x,
                    label_y,
                    text=lm,
                    fill=text_color,
                    font=font,
                    tags="marker",
                )
                continue

            if not (
                isinstance(val, tuple)
                and len(val) == 2
                and all(isinstance(v, (int, float)) for v in val)
            ):
                continue

            x, y = val
            y_offset_label = 16 if drawing_current_selected else 12
            xs, ys = self._img_to_screen(x, y)

            r = 5
            self.canvas.create_oval(
                xs - r,
                ys - r,
                xs + r,
                ys + r,
                outline=oval_color,
                width=2,
                tags="marker",
            )
            if lm in self.HOVER_CIRCLE_LANDMARKS:
                saved_radius = self.hover_radii.get(key, {}).get(lm)
                ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
                projected_ellipse_visible = False
                if ellipse_lm is not None and self._ellipse_is_projected_hemisphere(
                    ellipse_lm
                ):
                    fac_vis_var = self.landmark_visibility.get(ellipse_lm)
                    projected_ellipse_visible = (
                        fac_vis_var is None or bool(fac_vis_var.get())
                    )
                    if not projected_ellipse_visible:
                        self._draw_projected_hemisphere_boundary(
                            self.canvas,
                            ellipse_lm,
                            self._img_to_screen,
                            width=2,
                            tags="marker",
                        )
                elif saved_radius is not None:
                    screen_r = saved_radius * (self.disp_scale or 1.0)
                    self.canvas.create_oval(
                        xs - screen_r,
                        ys - screen_r,
                        xs + screen_r,
                        ys + screen_r,
                        outline="orange",
                        width=2,
                        tags="marker",
                    )
            if self.check_csv_mode and drawing_current_selected:
                self.canvas.create_oval(
                    xs - 10 * r,
                    ys - 10 * r,
                    xs + 10 * r,
                    ys + 10 * r,
                    outline=oval_color,
                    width=6,
                    tags="marker",
                )

            self.canvas.create_text(
                xs - 1,
                ys - y_offset_label - 1,
                text=lm,
                fill="black",
                font=shadow_font,
                tags="marker",
            )
            self.canvas.create_text(
                xs,
                ys - y_offset_label,
                text=lm,
                fill=text_color,
                font=font,
                tags="marker",
            )
        if lightweight:
            return
        for lm in SEGMENTATION_LANDMARKS:
            self._update_overlay_for(lm)
        for lm in IT_EDGE_LANDMARKS:
            self._update_it_edge_border_overlay_for(lm)
        self._update_pair_lines()
        self._update_femoral_axis_overlay()
        if getattr(self, "multi_review_mode", False):
            self._draw_multi_review_points()
        self._draw_categorical_shortcut_overlay()

    # Removes any connector lines between paired landmarks.
    def _remove_pair_lines(self):
        for key, line_id in list(self.pair_line_ids.items()):
            try:
                self.canvas.delete(line_id)
            except Exception as e:
                logger.warning(f"Failed to delete pair line {key}: {e}")
            self.pair_line_ids.pop(key, None)

    # Finds a landmark by lowercase name, returning the exact key.
    def _find_landmark_key(self, name_lower: str):
        for k in self.landmarks:
            if k.lower() == name_lower:
                return k
        return None

    # Adds or updates connector lines between DF/PF pairs if visible.
    def _update_pair_lines(self) -> None:
        if not self.current_image_path:
            self._remove_pair_lines()
            return
        # pts = self.annotations.get(self.current_image_path, {})
        pts, quality = self._get_annotations()
        pairs = [("ldf", "lpf"), ("rdf", "rpf")]
        color = "#00FFFF"
        for a, b in pairs:
            key = f"{a}_{b}"
            ka = self._find_landmark_key(a)
            kb = self._find_landmark_key(b)
            if ka is None or kb is None:
                return
            if ka in pts and kb in pts:
                va = self.landmark_visibility.get(ka)
                vb = self.landmark_visibility.get(kb)
                if (va is None or va.get()) and (vb is None or vb.get()):
                    point_a = pts[ka]
                    point_b = pts[kb]
                    if not (
                        isinstance(point_a, tuple)
                        and len(point_a) == 2
                        and all(isinstance(v, (int, float)) for v in point_a)
                        and isinstance(point_b, tuple)
                        and len(point_b) == 2
                        and all(isinstance(v, (int, float)) for v in point_b)
                    ):
                        continue
                    x1, y1 = point_a
                    x2, y2 = point_b
                    xs1, ys1 = self._img_to_screen(x1, y1)
                    xs2, ys2 = self._img_to_screen(x2, y2)
                    if key in self.pair_line_ids:
                        self.canvas.coords(self.pair_line_ids[key], xs1, ys1, xs2, ys2)
                    else:
                        self.pair_line_ids[key] = self.canvas.create_line(
                            xs1, ys1, xs2, ys2, fill=color, width=2, tags="pairline"
                        )
                    try:
                        self.canvas.tag_lower(self.pair_line_ids[key], "marker")
                    except Exception as e:
                        logger.warning(f"Failed to lower pair line {key}: {e}")
                    continue
            if key in self.pair_line_ids:
                try:
                    self.canvas.delete(self.pair_line_ids[key])
                except Exception as e:
                    logger.warning(f"Failed to delete pair line {key}: {e}")
                self.pair_line_ids.pop(key, None)

    def _toggle_extended_crosshair(self) -> None:
        enabled = self.extended_crosshair_enabled.get()
        self.crosshair_length_scale.config(state="normal" if enabled else "disabled")

        if not enabled:
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()
        else:
            if self.last_mouse_canvas_pos is not None:
                x, y = self.last_mouse_canvas_pos
                self._update_extended_crosshair(x, y)
            self._update_zoom_extended_crosshair()

    def _on_extended_crosshair_length_change(self, _value: str) -> None:
        if not self.extended_crosshair_enabled.get():
            return

        length = max(5, min(400, self.extended_crosshair_length.get()))
        self.extended_crosshair_length.set(length)

        if self.last_mouse_canvas_pos is not None:
            x, y = self.last_mouse_canvas_pos
            self._update_extended_crosshair(x, y)

        self._update_zoom_extended_crosshair()

    def _update_extended_crosshair(self, x: float, y: float) -> None:
        length = self.extended_crosshair_length.get()

        if not self.extended_crosshair_ids:
            hline_id = self.canvas.create_line(
                x - length,
                y,
                x + length,
                y,
                fill="lime",
                width=1,
                tags="extended_crosshair",
            )
            vline_id = self.canvas.create_line(
                x,
                y - length,
                x,
                y + length,
                fill="lime",
                width=1,
                tags="extended_crosshair",
            )
            self.extended_crosshair_ids = [hline_id, vline_id]
        else:
            hline_id, vline_id = self.extended_crosshair_ids
            self.canvas.coords(hline_id, x - length, y, x + length, y)
            self.canvas.coords(vline_id, x, y - length, x, y + length)

        for item_id in self.extended_crosshair_ids:
            self.canvas.tag_raise(item_id)

    def _hide_extended_crosshair(self) -> None:
        for item_id in self.extended_crosshair_ids:
            try:
                self.canvas.delete(item_id)
            except Exception as e:
                logger.warning(f"Failed to delete extended crosshair item: {e}")
        self.extended_crosshair_ids = []

    # Enables/disables the hover circle UI and hides it when disabled.
    def _toggle_hover(self) -> None:
        enabled = self.hover_enabled.get()
        radius_scale = self.__dict__.get("radius_scale")
        if radius_scale is not None:
            radius_scale.config(state="normal" if enabled else "disabled")
        if enabled and self.femoral_axis_enabled.get():
            self.femoral_axis_enabled.set(False)
            count_scale = self.__dict__.get("femoral_axis_count_scale")
            tip_scale = self.__dict__.get("femoral_axis_whisker_tip_length_scale")
            if count_scale is not None:
                count_scale.config(state="disabled")
            if tip_scale is not None:
                tip_scale.config(state="disabled")
            self._clear_femoral_axis_overlay()
        if not enabled:
            self._hide_hover_circle()

    # Keeps the hover circle radius in range and updates it live.
    def _on_radius_change(self, _value: str) -> None:
        r = max(1, min(300, self.hover_radius.get()))
        self.hover_radius.set(r)
        if self.hover_enabled.get() and self.hover_circle_id is not None:
            x0, y0, x1, y1 = self.canvas.coords(self.hover_circle_id)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self._update_hover_circle(cx, cy)
            self._update_zoom_hover_circle()

        if not self.hover_enabled.get():
            return

        lm = self.selected_landmark.get().strip()
        if lm not in self.HOVER_CIRCLE_LANDMARKS or self.current_image_path is None:
            return

        pts, _quality = self._get_annotations()
        if lm not in pts:
            return

        key = self._current_annotation_key()
        if key is None:
            return
        disp_scale = self.disp_scale or 1.0
        self.hover_radii.setdefault(key, {})[lm] = r / disp_scale
        self._schedule_hover_radius_dependency_refresh(lm, key)

    def _schedule_hover_radius_dependency_refresh(self, lm: str, key: str) -> None:
        pending_id = self.__dict__.get("_pending_hover_radius_sync_id")
        if pending_id is not None:
            try:
                self.after_cancel(pending_id)
            except Exception:
                pass
        self._pending_hover_radius_sync_landmark = lm
        self._pending_hover_radius_sync_key = key
        try:
            self._pending_hover_radius_sync_id = self.after(
                120, self._flush_hover_radius_dependency_refresh
            )
        except Exception:
            self._pending_hover_radius_sync_id = None
            self._flush_hover_radius_dependency_refresh()

    def _flush_hover_radius_dependency_refresh(self) -> None:
        lm = str(self.__dict__.get("_pending_hover_radius_sync_landmark", "") or "")
        key = str(self.__dict__.get("_pending_hover_radius_sync_key", "") or "")
        self._pending_hover_radius_sync_id = None
        self._pending_hover_radius_sync_landmark = ""
        self._pending_hover_radius_sync_key = ""
        if not lm or not key:
            return
        if key != self._current_annotation_key():
            return
        self._refresh_hover_radius_dependents(lm)

    def _refresh_hover_radius_dependents(self, lm: str) -> None:
        if self.current_image_path is None:
            return
        pts, _quality = self._get_annotations()
        if lm not in pts:
            return
        ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
        if ellipse_lm is not None and self._ellipse_is_projected_hemisphere(ellipse_lm):
            self._set_projected_hemisphere_from_ac(lm)
        elif ellipse_lm is not None and self._ellipse_snap_enabled():
            self._snap_ellipse_major_axis_to_ac_circle(ellipse_lm)
        else:
            self._update_auto_acetabular_points_for_ac(lm)
        self._update_protocol_derived_features_for_current_image(mark_dirty=True)
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._refresh_image_listbox()

    def _on_projected_hemisphere_z_change(self, _value: str) -> None:
        z_var = self.__dict__.get("projected_hemisphere_z")
        if z_var is None:
            return
        try:
            z_value = int(z_var.get())
        except (TypeError, ValueError):
            z_value = 70
        z_value = max(5, min(300, z_value))
        z_var.set(z_value)
        if self._apply_projected_hemisphere_radius_from_control():
            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True
            self._refresh_image_listbox()

    def _active_projected_hemisphere_ellipse_for_size(self) -> str | None:
        selected_var = self.__dict__.get("selected_landmark")
        lm = str(selected_var.get()).strip() if selected_var is not None else ""
        if self._is_ellipse_landmark(lm) and self._ellipse_is_projected_hemisphere(lm):
            return lm
        if self._matching_ellipse_landmark_for_ac(lm) is not None:
            ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
            if ellipse_lm is not None and self._ellipse_is_projected_hemisphere(
                ellipse_lm
            ):
                return ellipse_lm
        if self._fac_construction_mode() != FAC_CONSTRUCTION_HEMISPHERE:
            return None
        candidate = self._projected_hemisphere_placement_landmark(prefer_unset=False)
        if candidate is not None and self._ellipse_is_projected_hemisphere(candidate):
            return candidate
        return None

    def _apply_projected_hemisphere_radius_from_control(self) -> bool:
        if self.current_image_path is None:
            return False
        ellipse_lm = self._active_projected_hemisphere_ellipse_for_size()
        if ellipse_lm is None:
            return False
        return self._set_projected_hemisphere_radius_for_ellipse(
            ellipse_lm,
            self._projected_hemisphere_radius_from_control(),
        )

    def _change_projected_hemisphere_size(self, delta: int) -> bool:
        z_var = self.__dict__.get("projected_hemisphere_z")
        if z_var is None:
            return False
        try:
            old_radius = int(z_var.get())
        except (TypeError, ValueError):
            old_radius = 70
        new_radius = max(5, min(300, old_radius + delta))
        if new_radius == old_radius:
            return False
        z_var.set(new_radius)
        self._on_projected_hemisphere_z_change(str(new_radius))
        return True

    def _on_fac_tool_mousewheel(self, event):
        if self._fac_construction_mode() != FAC_CONSTRUCTION_HEMISPHERE:
            return None
        step = 2 if event.delta > 0 else -2
        self._change_projected_hemisphere_size(step)
        return "break"

    def _on_fac_tool_scroll_linux(self, direction: int):
        if self._fac_construction_mode() != FAC_CONSTRUCTION_HEMISPHERE:
            return None
        step = 2 if direction > 0 else -2
        self._change_projected_hemisphere_size(step)
        return "break"

    # Moves the hover circle with the mouse within image bounds.
    def _on_mouse_move(self, event) -> None:
        if not self.current_image:
            self.last_mouse_canvas_pos = None
            self._clear_line_preview()
            self._clear_femoral_axis_overlay()
            self._hide_mouse_crosshair()
            self._update_zoom_view(None, None)
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()
            return
        x0, y0, x1, y1 = self._display_rect()
        if x0 <= event.x < x1 and y0 <= event.y < y1:
            self.last_mouse_canvas_pos = (event.x, event.y)
            self._update_mouse_crosshair(event.x, event.y)
            self._update_zoom_view(event.x, event.y)
            self._update_line_preview(event.x, event.y)
            self._update_femoral_axis_overlay()
            if self.hover_enabled.get():
                self._update_hover_circle(
                    event.x, event.y
                )  # hover circle stays screen-space
            else:
                self._hide_hover_circle()
            if self.extended_crosshair_enabled.get():
                self._update_extended_crosshair(event.x, event.y)
            else:
                self._hide_extended_crosshair()
        else:
            self.last_mouse_canvas_pos = None
            self._clear_line_preview()
            self._clear_femoral_axis_overlay()
            self._hide_mouse_crosshair()
            self._hide_hover_circle()
            self._update_zoom_view(None, None)
            self._hide_extended_crosshair()
            self._hide_zoom_extended_crosshair()

    def _on_canvas_leave(self, _event) -> None:
        self._hide_hover_circle()
        self._clear_femoral_axis_overlay()
        self._hide_extended_crosshair()
        self._hide_mouse_crosshair()
        self._clear_line_preview()
        self.last_mouse_canvas_pos = None
        self._update_zoom_view(None, None)
        self._hide_zoom_extended_crosshair()

    # Adjusts hover radius via standard mouse wheel events.
    def _on_mousewheel(self, event) -> None:
        if self.right_mouse_held:
            step = 2 if event.delta > 0 else -2
            self._change_zoom_percent(step)
            return
        if self.femoral_axis_enabled.get() and self.selected_landmark.get() in (
            "L-FA",
            "R-FA",
        ):
            step = 2 if event.delta > 0 else -2
            self._change_femoral_axis_length(step)
            return
        if (
            self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE
            or self._active_projected_hemisphere_ellipse_for_size() is not None
        ):
            step = 2 if event.delta > 0 else -2
            self._change_projected_hemisphere_size(step)
            return
        if not self.hover_enabled.get():
            return
        step = 2 if event.delta > 0 else -2
        self._change_radius(step)

    # Adjusts hover radius for Linux button-4/5 events.
    def _on_scroll_linux(self, direction: int) -> None:
        if self.right_mouse_held:
            step = 2 if direction > 0 else -2
            self._change_zoom_percent(step)
            return
        if self.femoral_axis_enabled.get() and self.selected_landmark.get() in (
            "L-FA",
            "R-FA",
        ):
            step = 2 if direction > 0 else -2
            self._change_femoral_axis_length(step)
            return
        if (
            self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE
            or self._active_projected_hemisphere_ellipse_for_size() is not None
        ):
            step = 2 if direction > 0 else -2
            self._change_projected_hemisphere_size(step)
            return
        if not self.hover_enabled.get():
            return
        step = 2 if direction > 0 else -2
        self._change_radius(step)

    # Changes the hover radius and triggers a redraw if it changed.
    def _change_radius(self, delta: int) -> None:
        new_r = max(1, min(300, self.hover_radius.get() + delta))
        if new_r != self.hover_radius.get():
            self.hover_radius.set(new_r)
            self._on_radius_change(str(new_r))

    # Creates or updates the hover circle at the given position.
    def _update_hover_circle(self, x, y) -> None:
        r = self.hover_radius.get()
        x0, y0, x1, y1 = x - r, y - r, x + r, y + r
        if self.hover_circle_id is None:
            self.hover_circle_id = self.canvas.create_oval(
                x0, y0, x1, y1, outline="cyan", width=1, tags="hover_circle"
            )
        else:
            self.canvas.coords(self.hover_circle_id, x0, y0, x1, y1)

    # Creates or updates the hover circle in the zoom view.
    # The zoom hover circle is always centered since the zoom view
    # is centered on the mouse position. The radius scales with zoom.
    def _update_zoom_hover_circle(self) -> None:
        if self.zoom_canvas is None:
            return

        # Only show if hover tool is enabled
        if not self.hover_enabled.get():
            if self.zoom_hover_circle_id is not None:
                self.zoom_canvas.delete(self.zoom_hover_circle_id)
                self.zoom_hover_circle_id = None
            return

        # Need valid zoom source rect to compute scale
        if self.zoom_src_rect is None:
            return

        size = self._get_zoom_canvas_size()
        src_left, src_top, src_right, src_bottom = self.zoom_src_rect
        src_w = src_right - src_left
        src_h = src_bottom - src_top

        if abs(src_w) < 1e-12 or abs(src_h) < 1e-12:
            return

        # Calculate scaled radius:
        # hover_radius is in screen pixels, convert to image pixels, then to zoom canvas pixels
        # img_radius = screen_radius / disp_scale
        # zoom_radius = img_radius * (size / src_w)
        screen_radius = float(self.hover_radius.get())
        disp_scale = self.disp_scale or 1.0
        img_radius = screen_radius / disp_scale
        zoom_radius = img_radius * (size / src_w)

        # Center of zoom canvas
        cx, cy = size / 2.0, size / 2.0
        x0, y0 = cx - zoom_radius, cy - zoom_radius
        x1, y1 = cx + zoom_radius, cy + zoom_radius

        if self.zoom_hover_circle_id is None:
            self.zoom_hover_circle_id = self.zoom_canvas.create_oval(
                x0, y0, x1, y1, outline="cyan", width=1, tags="zoom_hover_circle"
            )
        else:
            self.zoom_canvas.coords(self.zoom_hover_circle_id, x0, y0, x1, y1)

    def _update_mouse_crosshair(self, x: float, y: float) -> None:
        circle_r = 8
        cross_r = 4

        if not self.mouse_crosshair_ids:
            circle_id = self.canvas.create_oval(
                x - circle_r,
                y - circle_r,
                x + circle_r,
                y + circle_r,
                outline="cyan",
                width=1,
                tags="mouse_crosshair",
            )
            hline_id = self.canvas.create_line(
                x - cross_r,
                y,
                x + cross_r,
                y,
                fill="cyan",
                width=1,
                tags="mouse_crosshair",
            )
            vline_id = self.canvas.create_line(
                x,
                y - cross_r,
                x,
                y + cross_r,
                fill="cyan",
                width=1,
                tags="mouse_crosshair",
            )
            self.mouse_crosshair_ids = [circle_id, hline_id, vline_id]
        else:
            circle_id, hline_id, vline_id = self.mouse_crosshair_ids
            self.canvas.coords(
                circle_id,
                x - circle_r,
                y - circle_r,
                x + circle_r,
                y + circle_r,
            )
            self.canvas.coords(hline_id, x - cross_r, y, x + cross_r, y)
            self.canvas.coords(vline_id, x, y - cross_r, x, y + cross_r)

        for item_id in self.mouse_crosshair_ids:
            self.canvas.tag_raise(item_id)

    def _hide_mouse_crosshair(self) -> None:
        for item_id in self.mouse_crosshair_ids:
            self.canvas.delete(item_id)
        self.mouse_crosshair_ids = []

    def _on_right_button_press(self, event) -> None:
        self.right_mouse_held = True

    def _on_right_button_release(self, event) -> None:
        self.right_mouse_held = False

    # Deletes the hover circle if present.
    def _hide_hover_circle(self) -> None:
        if self.hover_circle_id is not None:
            self.canvas.delete(self.hover_circle_id)
            self.hover_circle_id = None
        if self.zoom_hover_circle_id is not None and self.zoom_canvas is not None:
            self.zoom_canvas.delete(self.zoom_hover_circle_id)
            self.zoom_hover_circle_id = None

    # Clears all segmentation overlays and connector lines.
    def _remove_all_overlays(self):
        self._clear_it_edge_border_live_overlay()
        for lm in list(self.seg_item_ids.keys()):
            try:
                self.canvas.delete(self.seg_item_ids[lm])
            except Exception as e:
                logger.warning(f"Failed to delete overlay for {lm}: {e}")
        self.seg_item_ids.clear()
        self.seg_img_objs.clear()
        for lm in list(self.it_edge_border_item_ids.keys()):
            try:
                self.canvas.delete(self.it_edge_border_item_ids[lm])
            except Exception as e:
                logger.warning(f"Failed to delete IT edge border overlay for {lm}: {e}")
        self.it_edge_border_item_ids.clear()
        self.it_edge_border_img_objs.clear()
        self._remove_pair_lines()

    # Removes a specific landmark's overlay from the canvas.
    def _remove_overlay_for(self, lm: str) -> None:
        item_ids = self.__dict__.get("seg_item_ids", {})
        img_objs = self.__dict__.get("seg_img_objs", {})
        if lm in item_ids:
            try:
                self.canvas.delete(item_ids[lm])
            except Exception as e:
                logger.warning(f"Failed to delete overlay for {lm}: {e}")
            item_ids.pop(lm, None)
            img_objs.pop(lm, None)

    # Shows or hides a landmark overlay depending on mask and visibility.
    def _update_overlay_for(self, lm: str) -> None:
        vis = True
        vis_var = self.landmark_visibility.get(lm)
        if vis_var is not None:
            vis = bool(vis_var.get())
        has_mask = (
            self.current_image_path
            and (mask_key := self._segmentation_mask_key()) is not None
            and mask_key in self.seg_masks
            and lm in self.seg_masks[mask_key]
        )
        if not has_mask or not vis:
            self._remove_overlay_for(lm)
            return
        mask = self.seg_masks[mask_key][lm]
        fill_rgba = (
            PELVIS_SEGMENTATION_OVERLAY_RGBA
            if self._is_pelvis_segmentation_landmark(lm)
            else SEGMENTATION_OVERLAY_RGBA
        )
        self._render_overlay_for(lm, mask, fill_rgba=fill_rgba)
        self.canvas.tag_raise("marker")

    # Renders a semi-transparent RGBA overlay from a binary mask.
    def _render_overlay_for(
        self,
        lm: str,
        mask: np.ndarray,
        fill_rgba: Tuple[int, int, int, int] = SEGMENTATION_OVERLAY_RGBA,
    ) -> None:
        if mask is None or self.current_image is None:
            return

        # Ensure transform is current
        if self.disp_size == (0, 0):
            self._recompute_transform()

        disp_w, disp_h = self.disp_size
        off_x, off_y = self.disp_off

        mask_u8 = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_u8, mode="L").resize(
            (disp_w, disp_h), Image.Resampling.NEAREST
        )

        overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
        color_img = Image.new("RGBA", (disp_w, disp_h), fill_rgba)

        overlay.paste(color_img, (0, 0), mask_img)

        self.seg_img_objs[lm] = ImageTk.PhotoImage(overlay)

        if lm not in self.seg_item_ids:
            self.seg_item_ids[lm] = self.canvas.create_image(
                off_x, off_y, anchor="nw", image=self.seg_img_objs[lm], tags=f"seg_{lm}"
            )
        else:
            self.canvas.itemconfigure(
                self.seg_item_ids[lm], image=self.seg_img_objs[lm]
            )
            self.canvas.coords(self.seg_item_ids[lm], off_x, off_y)

        self.canvas.tag_lower(self.seg_item_ids[lm], "marker")
        self.canvas.tag_raise("marker")

    def _remove_it_edge_border_overlay_for(self, lm: str) -> None:
        item_ids = self.__dict__.get("it_edge_border_item_ids", {})
        img_objs = self.__dict__.get("it_edge_border_img_objs", {})
        if lm in item_ids:
            try:
                self.canvas.delete(item_ids[lm])
            except Exception as e:
                logger.warning(f"Failed to delete IT edge border overlay for {lm}: {e}")
            item_ids.pop(lm, None)
            img_objs.pop(lm, None)
        self._clear_it_edge_border_live_overlay(lm)

    def _clear_it_edge_border_live_overlay(self, lm: str | None = None) -> None:
        live_lm = self.__dict__.get("it_edge_border_live_landmark")
        if lm is not None and live_lm is not None and live_lm != lm:
            return

        canvas = self.__dict__.get("canvas")
        for attr in ("it_edge_border_live_item_id", "it_edge_border_live_dot_id"):
            item_id = self.__dict__.get(attr)
            if item_id is None or canvas is None:
                continue
            try:
                canvas.delete(item_id)
            except Exception as e:
                logger.warning(f"Failed to delete live IT edge border overlay: {e}")

        self.__dict__["it_edge_border_live_item_id"] = None
        self.__dict__["it_edge_border_live_dot_id"] = None
        self.__dict__["it_edge_border_live_landmark"] = None
        self.__dict__["it_edge_border_live_coords"] = []

    def _it_edge_border_live_stroke_width(self, lm: str, brush: int) -> float:
        mask = self._get_it_edge_border_mask(lm, create=False)
        if mask is not None and self.disp_size != (0, 0):
            disp_w, disp_h = self.disp_size
            mask_h, mask_w = mask.shape[:2]
            scale_x = disp_w / max(1, mask_w)
            scale_y = disp_h / max(1, mask_h)
            return max(1.0, float(brush) * max(scale_x, scale_y))
        return max(1.0, float(brush) * float(self.disp_scale or 1.0))

    def _it_edge_border_mode_value(self) -> str:
        mode_var = self.__dict__.get("it_edge_border_mode")
        mode = str(mode_var.get()).strip().lower() if mode_var is not None else "draw"
        return "erase" if mode == "erase" else "draw"

    def _effective_it_edge_border_brush_size(self, lm: str) -> int:
        try:
            brush = int(self.it_edge_border_brush_size.get())
        except (TypeError, ValueError):
            brush = 1
        brush = max(1, min(80, brush))
        if self._it_edge_border_mode_value() == "draw":
            brush = 1
            brush_var = self.__dict__.get("it_edge_border_brush_size")
            if brush_var is not None:
                try:
                    brush_var.set(1)
                except Exception:
                    pass
        return brush

    def _refresh_it_edge_border_brush_control(self) -> None:
        mode = self._it_edge_border_mode_value()
        if mode == "draw":
            brush_var = self.__dict__.get("it_edge_border_brush_size")
            if brush_var is not None:
                try:
                    brush_var.set(1)
                except Exception:
                    pass
        scale = self.__dict__.get("it_edge_border_brush_scale")
        if scale is None:
            return
        try:
            scale.configure(state="disabled" if mode == "draw" else "normal")
        except Exception:
            pass

    def _on_it_edge_border_mode_changed(self) -> None:
        self._refresh_it_edge_border_brush_control()

    def _draw_live_it_edge_border_segment(
        self,
        lm: str,
        start: AnnotationPoint,
        end: AnnotationPoint,
    ) -> None:
        canvas = self.__dict__.get("canvas")
        if canvas is None or self.current_image is None:
            return
        if self._it_edge_border_mode_value() == "erase":
            return

        brush = self._effective_it_edge_border_brush_size(lm)
        stroke_width = self._it_edge_border_live_stroke_width(lm, brush)

        if self.__dict__.get("it_edge_border_live_landmark") != lm:
            self._clear_it_edge_border_live_overlay()
            self.__dict__["it_edge_border_live_landmark"] = lm
            self.__dict__["it_edge_border_live_coords"] = []

        sx0, sy0 = self._img_to_screen(float(start[0]), float(start[1]))
        sx1, sy1 = self._img_to_screen(float(end[0]), float(end[1]))

        coords = self.__dict__.setdefault("it_edge_border_live_coords", [])
        if not coords:
            coords.extend([sx0, sy0])

        if abs(sx1 - coords[-2]) > 0.01 or abs(sy1 - coords[-1]) > 0.01:
            coords.extend([sx1, sy1])

        item_id = self.__dict__.get("it_edge_border_live_item_id")
        if len(coords) >= 4:
            if item_id is None:
                item_id = canvas.create_line(
                    *coords,
                    fill="#00FFFF",
                    width=stroke_width,
                    capstyle="round",
                    joinstyle="round",
                    stipple="gray50",
                    tags=("it_edge_border_live", f"it_edge_border_live_{lm}"),
                )
                self.__dict__["it_edge_border_live_item_id"] = item_id
            else:
                canvas.coords(item_id, *coords)
                canvas.itemconfigure(item_id, width=stroke_width)

        if len(coords) == 2:
            dot_id = self.__dict__.get("it_edge_border_live_dot_id")
            r = stroke_width / 2.0
            if dot_id is None:
                dot_id = canvas.create_oval(
                    sx1 - r,
                    sy1 - r,
                    sx1 + r,
                    sy1 + r,
                    fill="#00FFFF",
                    outline="",
                    stipple="gray50",
                    tags=("it_edge_border_live", f"it_edge_border_live_{lm}"),
                )
                self.__dict__["it_edge_border_live_dot_id"] = dot_id
            else:
                canvas.coords(dot_id, sx1 - r, sy1 - r, sx1 + r, sy1 + r)

        for maybe_item_id in (
            self.__dict__.get("it_edge_border_live_dot_id"),
            self.__dict__.get("it_edge_border_live_item_id"),
        ):
            if maybe_item_id is None:
                continue
            try:
                canvas.tag_lower(maybe_item_id, "marker")
            except Exception:
                pass

        for item_id in self.__dict__.get("mouse_crosshair_ids", []):
            canvas.tag_raise(item_id)
        for item_id in self.__dict__.get("extended_crosshair_ids", []):
            canvas.tag_raise(item_id)

    def _drag_update_interval_elapsed(self, attr: str, *, force: bool = False) -> bool:
        now = time.monotonic()
        if force:
            self.__dict__[attr] = now
            return True
        interval = float(
            self.__dict__.get("it_edge_border_drag_update_interval_sec", 1.0 / 30.0)
            or 0.0
        )
        last = float(self.__dict__.get(attr, 0.0) or 0.0)
        if now - last < interval:
            return False
        self.__dict__[attr] = now
        return True

    def _update_it_edge_border_overlay_for_drag(
        self, lm: str, *, force: bool = False
    ) -> None:
        if not self._drag_update_interval_elapsed(
            "it_edge_border_last_overlay_update_at",
            force=force,
        ):
            return
        self._update_it_edge_border_overlay_for(lm)

    def _update_it_edge_border_zoom_for_drag(
        self,
        canvas_x: float,
        canvas_y: float,
        *,
        force: bool = False,
    ) -> None:
        if self.__dict__.get("zoom_canvas") is None:
            return
        if not self._drag_update_interval_elapsed(
            "it_edge_border_last_zoom_update_at",
            force=force,
        ):
            return
        self._update_zoom_view(canvas_x, canvas_y)

    def _update_it_edge_border_overlay_for(self, lm: str) -> None:
        vis = True
        vis_var = self.__dict__.get("landmark_visibility", {}).get(lm)
        if vis_var is not None:
            vis = bool(vis_var.get())
        mask_key = self._segmentation_mask_key()
        border_masks = self.__dict__.get("it_edge_border_masks", {})
        filled_masks = self.__dict__.get("seg_masks", {})
        has_filled_mask = (
            self.current_image_path
            and mask_key is not None
            and mask_key in filled_masks
            and lm in filled_masks[mask_key]
        )
        if has_filled_mask:
            self._remove_it_edge_border_overlay_for(lm)
            return
        has_mask = (
            self.current_image_path
            and mask_key is not None
            and mask_key in border_masks
            and lm in border_masks[mask_key]
        )
        if not has_mask or not vis:
            self._remove_it_edge_border_overlay_for(lm)
            return
        self._render_it_edge_border_overlay_for(lm, border_masks[mask_key][lm])
        self.canvas.tag_raise("marker")

    def _render_it_edge_border_overlay_for(
        self,
        lm: str,
        mask: np.ndarray,
        fill_rgba: Tuple[int, int, int, int] = SEGMENTATION_OVERLAY_RGBA,
    ) -> None:
        if mask is None or self.current_image is None:
            return

        self._clear_it_edge_border_live_overlay(lm)

        if self.disp_size == (0, 0):
            self._recompute_transform()

        disp_w, disp_h = self.disp_size
        off_x, off_y = self.disp_off

        mask_u8 = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_u8, mode="L").resize(
            (disp_w, disp_h), Image.Resampling.NEAREST
        )

        overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
        color_img = Image.new("RGBA", (disp_w, disp_h), fill_rgba)
        overlay.paste(color_img, (0, 0), mask_img)

        img_objs = self.__dict__.setdefault("it_edge_border_img_objs", {})
        item_ids = self.__dict__.setdefault("it_edge_border_item_ids", {})
        img_objs[lm] = ImageTk.PhotoImage(overlay)

        if lm not in item_ids:
            item_ids[lm] = self.canvas.create_image(
                off_x,
                off_y,
                anchor="nw",
                image=img_objs[lm],
                tags=f"it_edge_border_{lm}",
            )
        else:
            self.canvas.itemconfigure(item_ids[lm], image=img_objs[lm])
            self.canvas.coords(item_ids[lm], off_x, off_y)

        self.canvas.tag_lower(item_ids[lm], "marker")
        self.canvas.tag_raise("marker")

    def _get_it_edge_border_mask(
        self, lm: str, create: bool = True
    ) -> np.ndarray | None:
        if not self._is_border_segmentation_landmark(lm):
            return None
        if self.current_image is None:
            return None
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return None

        expected_shape = self._mask_shape_for_landmark(lm)
        if expected_shape is None:
            return None
        border_masks = self.__dict__.setdefault("it_edge_border_masks", {})
        per_img = border_masks.setdefault(mask_key, {})
        mask = per_img.get(lm)
        if mask is not None and mask.shape == expected_shape:
            return mask
        if mask is not None:
            mask = self._resize_binary_mask(mask, expected_shape)
            per_img[lm] = mask
            return mask
        if not create:
            return None
        mask = np.zeros(expected_shape, dtype=np.uint8)
        per_img[lm] = mask
        return mask

    def _clear_it_edge_filled_mask(self, lm: str) -> None:
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return
        self.__dict__.get("seg_masks", {}).get(mask_key, {}).pop(lm, None)
        self._remove_overlay_for(lm)

    def _clear_it_edge_border_state(
        self,
        lm: str,
        *,
        clear_annotation: bool = True,
    ) -> None:
        mask_key = self._segmentation_mask_key()
        if mask_key is not None:
            self.__dict__.get("seg_masks", {}).get(mask_key, {}).pop(lm, None)
            self.__dict__.get("it_edge_border_masks", {}).get(mask_key, {}).pop(
                lm, None
            )
        self._remove_overlay_for(lm)
        self._remove_it_edge_border_overlay_for(lm)
        if clear_annotation:
            ann_key = self._current_annotation_key()
            if ann_key is not None:
                self.annotations.setdefault(ann_key, {}).pop(lm, None)
            if lm in self.landmark_found:
                self.landmark_found[lm].set(False)

    def _sync_it_edge_annotation_from_mask(
        self, lm: str, mask: np.ndarray | None
    ) -> None:
        if not self._is_border_segmentation_landmark(lm):
            return
        ann_key = self._current_annotation_key()
        if ann_key is None:
            return

        pts = self.annotations.setdefault(ann_key, {})
        centroid = self._mask_centroid_in_image_space(mask)
        if centroid is None:
            pts.pop(lm, None)
            if lm in self.landmark_found:
                self.landmark_found[lm].set(False)
            return

        pts[lm] = centroid
        if lm in self.landmark_found:
            self.landmark_found[lm].set(True)

    def _draw_it_edge_border_segment(
        self,
        lm: str,
        start: AnnotationPoint,
        end: AnnotationPoint,
        *,
        refresh_overlay: bool = True,
        sync_annotation: bool = True,
        refresh_controls: bool = True,
    ) -> None:
        mask = self._get_it_edge_border_mask(lm, create=True)
        if mask is None or cv2 is None:
            return

        height, width = mask.shape[:2]

        def clamp_pixel(point: AnnotationPoint) -> Tuple[int, int]:
            mx, my = self._image_point_to_mask_point(point, mask)
            px = int(round(mx))
            py = int(round(my))
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            return px, py

        p0 = clamp_pixel(start)
        p1 = clamp_pixel(end)
        brush = self._effective_it_edge_border_brush_size(lm)
        mode = self._it_edge_border_mode_value()
        color = 0 if mode == "erase" else 1
        cv2.line(mask, p0, p1, color=int(color), thickness=brush, lineType=cv2.LINE_8)
        self._clear_it_edge_filled_mask(lm)
        if sync_annotation:
            self._sync_it_edge_annotation_from_mask(lm, mask)
        if refresh_overlay:
            self._update_it_edge_border_overlay_for(lm)
        if refresh_controls:
            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._refresh_image_listbox()
        self.dirty = True

    def _fill_it_edge_border(self, lm: str) -> bool:
        if lm in PROTOCOL_FREEHAND_CONTOUR_LANDMARKS:
            self._set_status(
                f"{lm}: protocol contours save the traced 512-grid pixels directly."
            )
            return False
        border_mask = self._get_it_edge_border_mask(lm, create=False)
        if border_mask is None or cv2 is None:
            self._set_status(f"{lm}: draw an enclosed highlight border first.")
            return False

        border = border_mask > 0
        if not bool(np.any(border)):
            self._clear_it_edge_filled_mask(lm)
            self._set_status(f"{lm}: draw an enclosed highlight border first.")
            return False

        padded_border = np.pad(border.astype(np.uint8), 1, constant_values=0)
        free = (padded_border == 0).astype(np.uint8)
        _label_count, labels = cv2.connectedComponents(free, connectivity=4)
        outside_label = int(labels[0, 0])
        interior_padded = (free > 0) & (labels != outside_label)
        interior = interior_padded[1:-1, 1:-1]

        if int(interior.sum()) < 30:
            self._clear_it_edge_filled_mask(lm)
            self._set_status(f"{lm}: highlight border is not enclosed yet.")
            return False

        filled = (interior | border).astype(np.uint8)
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return False
        self.seg_masks.setdefault(mask_key, {})[lm] = filled

        ann_key = self._current_annotation_key()
        if ann_key is None:
            return False
        self._sync_it_edge_annotation_from_mask(lm, filled)
        pts = self.annotations.setdefault(ann_key, {})

        self._update_overlay_for(lm)
        self._update_it_edge_border_overlay_for(lm)
        self._update_found_checks(pts)
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._refresh_image_listbox()
        self._set_status(f"{lm}: filled enclosed highlight border.")
        return True

    def _fill_selected_it_edge_border(self) -> bool:
        lm = self.selected_landmark.get().strip()
        if not self._is_border_segmentation_landmark(lm):
            self._set_status("Select a manual contour mask before filling.")
            return False
        filled = self._fill_it_edge_border(lm)
        if filled:
            self._draw_points()
            self._autosave_annotation_change()
        return filled

    def _clear_selected_it_edge_border(self) -> None:
        lm = self.selected_landmark.get().strip()
        if not self._is_border_segmentation_landmark(lm):
            self._set_status("Select a manual contour mask before clearing.")
            return
        self._clear_it_edge_border_state(lm)
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._refresh_image_listbox()
        self._autosave_annotation_change()
        self._set_status(f"{lm}: highlight and segmentation cleared.")

    def _set_status(self, message: str) -> None:
        try:
            self.queue_status_var.set(message)
        except Exception:
            logger.info(message)

    def _set_pelvis_segmentation_status(self, message: str) -> None:
        try:
            self.pelvis_segmentation_status_var.set(message)
        except Exception:
            logger.info(message)
        self._set_status(message)

    def _refresh_pelvis_segmentation_buttons(self) -> None:
        readonly = bool(getattr(self, "multi_review_mode", False))
        has_image = self.current_image is not None and self.current_image_path is not None
        running = bool(getattr(self, "pelvis_segmentation_running", False))
        can_segment = has_image and not readonly and not running

        if self.segment_pelvis_btn is not None:
            self.segment_pelvis_btn.configure(
                state="normal" if can_segment else "disabled"
            )

        mask_key = self._segmentation_mask_key()
        has_undo = bool(
            mask_key is not None
            and self.pelvis_segmentation_undo_stack.get(mask_key)
        )
        if self.undo_pelvis_btn is not None:
            self.undo_pelvis_btn.configure(
                state="normal" if can_segment and has_undo else "disabled"
            )

    def _selected_pelvis_segmentation_backend(self) -> str:
        valid = {mode for mode, _label in PELVIS_SEGMENTATION_BACKEND_OPTIONS}
        try:
            mode = self.pelvis_segmentation_backend_var.get().strip().lower()
        except Exception:
            mode = "auto"
        return mode if mode in valid else "auto"

    def _on_pelvis_segmentation_backend_changed(self) -> None:
        mode = self._selected_pelvis_segmentation_backend()
        label = self._pelvis_segmentation_backend_label(mode)
        self._set_pelvis_segmentation_status(f"Pelvis backend: {label}.")
        self._refresh_pelvis_segmentation_buttons()

    def _pelvis_segmentation_backend_label(self, mode: str) -> str:
        for candidate, label in PELVIS_SEGMENTATION_BACKEND_OPTIONS:
            if candidate == mode:
                return label
        return "Auto"

    def _get_pelvis_segmenter(self, mode: str) -> PelvisSegmentationBackend:
        if mode not in self.pelvis_segmenters:
            self.pelvis_segmenters[mode] = PelvisSegmentationBackend(mode=mode)
        return self.pelvis_segmenters[mode]

    def _current_pelvis_mask(self) -> np.ndarray | None:
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return None
        return self.seg_masks.get(mask_key, {}).get(PELVIS_SEGMENTATION_LANDMARK)

    def _push_pelvis_undo_state(self) -> None:
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return
        current = self._current_pelvis_mask()
        undo_state = None if current is None else current.copy()
        stack = self.pelvis_segmentation_undo_stack.setdefault(mask_key, [])
        stack.append(undo_state)
        if len(stack) > 20:
            del stack[0]
        self._refresh_pelvis_segmentation_buttons()

    def _segment_pelvis_current_image(self) -> None:
        if getattr(self, "multi_review_mode", False):
            messagebox.showinfo(
                "Pelvis Segmentation",
                "Multi-reviewer mode is read-only; no segmentation was changed.",
            )
            return

        if self.current_image is None or self.current_image_path is None:
            messagebox.showwarning("Pelvis Segmentation", "No image loaded.")
            return

        if self.pelvis_segmentation_running:
            return

        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return

        image = self.current_image.copy()
        mode = self._selected_pelvis_segmentation_backend()
        label = self._pelvis_segmentation_backend_label(mode)
        self.pelvis_segmentation_running = True
        self._set_pelvis_segmentation_status(f"Segmenting pelvis with {label}...")
        self._refresh_pelvis_segmentation_buttons()

        def worker() -> None:
            try:
                result = self._get_pelvis_segmenter(mode).segment(image)
                mask = result.mask.copy()
                backend = result.backend
                detail = result.detail
                self.after(
                    0,
                    lambda: self._finish_pelvis_segmentation(
                        mask_key,
                        mask,
                        backend,
                        detail,
                        None,
                    ),
                )
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                self.after(
                    0,
                    lambda: self._finish_pelvis_segmentation(
                        mask_key,
                        None,
                        "",
                        "",
                        message,
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_pelvis_segmentation(
        self,
        expected_mask_key: str,
        mask: np.ndarray | None,
        backend: str,
        detail: str,
        error: str | None,
    ) -> None:
        self.pelvis_segmentation_running = False

        if self._segmentation_mask_key() != expected_mask_key:
            self._set_pelvis_segmentation_status(
                "Pelvis segmentation discarded; image changed."
            )
            self._refresh_pelvis_segmentation_buttons()
            return

        if error is not None:
            self._set_pelvis_segmentation_status(f"Pelvis segmentation failed: {error}")
            messagebox.showerror("Pelvis Segmentation", error)
            self._refresh_pelvis_segmentation_buttons()
            return

        if mask is None:
            self._set_pelvis_segmentation_status("Pelvis segmentation returned no mask.")
            self._refresh_pelvis_segmentation_buttons()
            return

        source = backend if not detail else f"{backend} ({detail})"
        self._apply_pelvis_segmentation_mask(mask, source=source)

    def _apply_pelvis_segmentation_mask(self, mask: np.ndarray, source: str) -> bool:
        if self.current_image is None or self.current_image_path is None:
            return False

        width, height = self.current_image.size
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_u8.shape != (height, width):
            try:
                mask_u8 = cv2.resize(
                    mask_u8,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
            except Exception:
                self._set_pelvis_segmentation_status(
                    "Pelvis segmentation failed: mask size did not match the image."
                )
                return False

        if int(mask_u8.sum()) < 30:
            self._set_pelvis_segmentation_status(
                "Pelvis segmentation failed: mask was empty."
            )
            return False

        mask_key = self._segmentation_mask_key()
        ann_key = self._current_annotation_key()
        if mask_key is None or ann_key is None:
            return False

        self._push_pelvis_undo_state()
        self.seg_masks.setdefault(mask_key, {})[PELVIS_SEGMENTATION_LANDMARK] = mask_u8
        self.annotations.setdefault(ann_key, {}).pop(
            PELVIS_SEGMENTATION_LANDMARK, None
        )
        self.landmark_meta.setdefault(ann_key, {}).pop(
            PELVIS_SEGMENTATION_LANDMARK, None
        )
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._refresh_image_listbox()
        self._autosave_annotation_change()
        self._set_pelvis_segmentation_status(f"Pelvis segmented with {source}.")
        self._refresh_pelvis_segmentation_buttons()
        return True

    def _undo_pelvis_segmentation(self) -> None:
        if getattr(self, "multi_review_mode", False):
            return

        mask_key = self._segmentation_mask_key()
        ann_key = self._current_annotation_key()
        if mask_key is None or ann_key is None:
            return

        stack = self.pelvis_segmentation_undo_stack.get(mask_key, [])
        if not stack:
            self._set_pelvis_segmentation_status("No pelvis segmentation to undo.")
            self._refresh_pelvis_segmentation_buttons()
            return

        previous = stack.pop()
        if previous is None:
            self.seg_masks.get(mask_key, {}).pop(PELVIS_SEGMENTATION_LANDMARK, None)
            self._remove_overlay_for(PELVIS_SEGMENTATION_LANDMARK)
        else:
            self.seg_masks.setdefault(mask_key, {})[PELVIS_SEGMENTATION_LANDMARK] = (
                previous.copy()
            )

        self.annotations.setdefault(ann_key, {}).pop(
            PELVIS_SEGMENTATION_LANDMARK, None
        )
        self.landmark_meta.setdefault(ann_key, {}).pop(
            PELVIS_SEGMENTATION_LANDMARK, None
        )
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        self._refresh_image_listbox()
        self._autosave_annotation_change()
        self._set_pelvis_segmentation_status("Pelvis segmentation undone.")
        self._refresh_pelvis_segmentation_buttons()

    # Handles click to place a landmark and optionally run segmentation.
    def _place_multi_review_temp_annotation(self, x: float, y: float) -> None:
        lm = self.selected_landmark.get()
        if not lm:
            if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE:
                lm = self._select_projected_hemisphere_placement_landmark() or ""
            if lm:
                self._set_status(f"{lm}: projected hemisphere selected.")
            else:
                messagebox.showwarning(
                    "No Landmark", "Please select a landmark in the list."
                )
                return

        temp_pts = self._current_multi_review_temp_annotations()
        if self._is_ellipse_landmark(lm):
            construction = (
                FAC_CONSTRUCTION_HEMISPHERE
                if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE
                else None
            )
            temp_pts[lm] = self._default_ellipse_for_center((x, y), construction)
        elif self._is_line_landmark(lm):
            existing = temp_pts.get(lm)
            line_pts: List[AnnotationPoint] = []
            if isinstance(existing, list):
                for point in existing[:2]:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            line_pts.append((float(point[0]), float(point[1])))
                        except (TypeError, ValueError):
                            pass

            if len(line_pts) >= 2:
                line_pts = [(x, y)]
            else:
                line_pts.append((x, y))
            temp_pts[lm] = line_pts
        else:
            temp_pts[lm] = (x, y)

        self._refresh_multi_review_panel()
        self._draw_points()
        self._refresh_zoom_landmark_overlay()

    def _on_left_press(self, event) -> None:
        x0, y0, x1, y1 = self._display_rect()
        if not self.current_image:
            return
        if self.current_image_path is None:
            return
        if not (x0 <= event.x < x1 and y0 <= event.y < y1):
            return

        self.last_mouse_canvas_pos = (event.x, event.y)
        self._update_mouse_crosshair(event.x, event.y)
        self._update_zoom_view(event.x, event.y)

        xi, yi = self._screen_to_img(event.x, event.y)
        x, y = self._clamp_img_point(xi, yi)

        if getattr(self, "multi_review_mode", False):
            self._place_multi_review_temp_annotation(x, y)
            return

        if not self._image_categories_complete():
            self._set_status("Select Left Hip and Right Hip image categories before landmarking.")
            return

        lm = self.selected_landmark.get()
        if not lm:
            if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE:
                lm = self._select_projected_hemisphere_placement_landmark() or ""
            if lm:
                self._set_status(f"{lm}: projected hemisphere selected.")
            else:
                messagebox.showwarning(
                    "No Landmark", "Please select a landmark in the list."
                )
                return

        if self._is_protocol_derived_feature(lm) or self._is_derived_acetabular_landmark(lm):
            self._set_status(self._noneditable_derived_message(lm))
            return
        if not self._landmark_is_selectable(lm):
            self._set_status(self._locked_by_protocol_order_message(lm))
            return

        if lm not in {"L-FA", "R-FA"} and self.femoral_axis_enabled.get():
            self.femoral_axis_enabled.set(False)
            self._toggle_femoral_axis()

        if self._is_intermediate_acetabular_landmark(lm):
            if not self._set_intermediate_acetabular_point(lm, (x, y)):
                return
            self.dragging_landmark = lm
            self.dragging_point_index = None
            self.dragging_line_whole = False
            self.dragging_line_last_img_pos = None
            self.dragging_ellipse_handle = None
            self.dragging_ellipse_whole = False
            self.dragging_ellipse_last_img_pos = None
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True
            return

        if self._is_border_segmentation_landmark(lm):
            if cv2 is None:
                messagebox.showerror(
                    "OpenCV missing",
                    'cv2 is not available. Install with "pip install opencv-python".',
                )
                return
            self.drawing_it_edge_border = True
            self.it_edge_border_last_img_pos = (x, y)
            self.dragging_landmark = None
            self.dragging_point_index = None
            self.dragging_line_whole = False
            self.dragging_line_last_img_pos = None
            self.dragging_ellipse_handle = None
            self.dragging_ellipse_whole = False
            self.dragging_ellipse_last_img_pos = None
            self.it_edge_border_last_overlay_update_at = 0.0
            self.it_edge_border_last_zoom_update_at = 0.0
            self._clear_it_edge_border_live_overlay(lm)
            self._draw_it_edge_border_segment(
                lm,
                (x, y),
                (x, y),
                refresh_overlay=False,
                sync_annotation=False,
                refresh_controls=False,
            )
            if self._it_edge_border_mode_value() == "erase":
                self._update_it_edge_border_overlay_for_drag(lm, force=True)
            else:
                self._draw_live_it_edge_border_segment(lm, (x, y), (x, y))
            self._update_it_edge_border_zoom_for_drag(event.x, event.y, force=True)
            return

        if self._is_ellipse_landmark(lm):
            hit_handle = self._find_ellipse_handle_hit(lm, event.x, event.y)
            if hit_handle is not None:
                self.dragging_landmark = lm
                self.dragging_ellipse_handle = hit_handle
                self.dragging_ellipse_whole = False
                self.dragging_ellipse_last_img_pos = None
                self.dragging_point_index = None
                self.dragging_line_whole = False
                return

            ellipse = self._get_ellipse(lm)
            if ellipse is not None and self._is_ellipse_hit(lm, event.x, event.y):
                self.dragging_landmark = lm
                self.dragging_ellipse_handle = None
                self.dragging_ellipse_whole = True
                self.dragging_ellipse_last_img_pos = (x, y)
                self.dragging_point_index = None
                self.dragging_line_whole = False
                return

            if ellipse is not None:
                self._set_status(
                    f"{lm}: drag ellipse endpoints to adjust, or uncheck to clear."
                )
                return

            if not self._check_left_right_order_for_landmark(lm, x, y):
                return

            if self._fac_construction_mode() == FAC_CONSTRUCTION_HEMISPHERE:
                if not self._set_projected_hemisphere_from_center(lm, (x, y)):
                    self._set_status(f"{lm}: projected hemisphere could not be placed.")
                    return
                self.dragging_landmark = lm
                self.dragging_ellipse_handle = None
                self.dragging_ellipse_whole = True
                self.dragging_ellipse_last_img_pos = (x, y)
                self.dragging_point_index = None
                self.dragging_line_whole = False
                self.dragging_line_last_img_pos = None
            else:
                self._set_ellipse(
                    lm,
                    self._default_ellipse_for_center(
                        (x, y), FAC_CONSTRUCTION_SNAPPED
                    ),
                    FAC_CONSTRUCTION_SNAPPED,
                )
            if (
                self._fac_construction_mode() == FAC_CONSTRUCTION_SNAPPED
                and self._ellipse_snap_enabled()
            ):
                self._snap_ellipse_major_axis_to_ac_circle(lm)
            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True
            self._autosave_annotation_change()
            if self._fac_construction_mode() != FAC_CONSTRUCTION_HEMISPHERE:
                self._advance_to_next_protocol_landmark_if_current_complete(lm)
            return

        if self._is_line_landmark(lm):
            hit_idx = self._find_line_point_hit(lm, event.x, event.y)
            if hit_idx is not None:
                self.dragging_landmark = lm
                self.dragging_point_index = hit_idx
                self.dragging_line_whole = False
                self.dragging_line_last_img_pos = None
                return

            pts = self._get_line_points(lm)
            if len(pts) == 2 and self._is_line_hit(lm, event.x, event.y):
                self.dragging_landmark = lm
                self.dragging_point_index = None
                self.dragging_line_whole = True
                self.dragging_line_last_img_pos = (x, y)
                return

            if len(pts) == 0:
                if not self._check_left_right_order_for_landmark(lm, x, y):
                    return
                self._set_line_points(lm, [(x, y)])
            elif len(pts) == 1:
                mean_x = (pts[0][0] + x) / 2.0
                if not self._check_left_right_order_for_landmark(lm, mean_x, y):
                    return
                self._set_line_points(lm, [pts[0], (x, y)])
                self._clear_line_preview()
            elif len(pts) == 2:
                # Two points already placed — require explicit deletion to re-place
                self._set_status(f"{lm}: already has 2 endpoints. Uncheck to clear.")
                return
            else:
                if not self._check_left_right_order_for_landmark(lm, x, y):
                    return
                self._set_line_points(lm, [(x, y)])
                self._clear_line_preview()

            pts, _quality = self._get_annotations()
            self._update_found_checks(pts)
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True
            self._autosave_annotation_change()
            self._advance_to_next_protocol_landmark_if_current_complete(lm)
            return

        if not self._check_left_right_order_for_landmark(lm, x, y):
            return

        ann_key = self._current_annotation_key()
        if ann_key is None:
            return
        self.annotations.setdefault(ann_key, {})[lm] = (x, y)
        self._set_landmark_status(lm, "")
        if lm in self.HOVER_CIRCLE_LANDMARKS:
            img_key = (
                self._path_key(self.current_image_path)
                if self.json_path is not None
                else str(self.current_image_path)
            )
            disp_scale = self.disp_scale or 1.0
            self.hover_radii.setdefault(img_key, {})[lm] = (
                self.hover_radius.get() / disp_scale
            )
            ellipse_lm = self._matching_ellipse_landmark_for_ac(lm)
            if ellipse_lm is not None:
                if (
                    self._ellipse_is_projected_hemisphere(ellipse_lm)
                ):
                    self._set_projected_hemisphere_from_ac(lm)
                    self.dragging_landmark = ellipse_lm
                    self.dragging_ellipse_handle = None
                    self.dragging_ellipse_whole = True
                    self.dragging_ellipse_last_img_pos = (x, y)
                    self.dragging_point_index = None
                    self.dragging_line_whole = False
                    self.dragging_line_last_img_pos = None
                elif self._ellipse_snap_enabled():
                    self._snap_ellipse_major_axis_to_ac_circle(ellipse_lm)
                else:
                    self._update_auto_acetabular_points_for_ac(lm)
        pts, _quality = self._get_annotations()
        self._update_found_checks(pts)
        self._draw_points()
        self._refresh_zoom_landmark_overlay()
        self.dirty = True
        if self._is_auto_segmentation_landmark(lm):
            if cv2 is None:
                messagebox.showerror(
                    "OpenCV missing",
                    'cv2 is not available. Install with "pip install opencv-python".',
                )
                return
            self.last_seed[lm] = (int(x), int(y))
            self._store_current_settings_for(lm)
            self._resegment_for(lm)
        self._autosave_annotation_change()
        self._advance_to_next_protocol_landmark_if_current_complete(lm)
        return

    def _check_left_right_order_for_landmark(
        self,
        lm: str,
        new_x: float,
        new_y: float,
    ) -> bool:
        if lm.startswith("L-"):
            other_lm = "R-" + lm[2:]
            is_left_landmark = True
        elif lm.startswith("R-"):
            other_lm = "L-" + lm[2:]
            is_left_landmark = False
        else:
            return True

        allowed = self._get_allowed_landmarks_for_current_view()
        if other_lm not in allowed:
            return True

        pts, _quality = self._get_annotations()
        if other_lm not in pts:
            return True

        # Use x-position of the already-placed corresponding landmark.
        # For line landmarks, use the mean x of the existing line points.
        if self._is_ellipse_landmark(other_lm):
            other_center = self._ellipse_center_from_value(pts[other_lm])
            if other_center is None:
                return True
            other_x = other_center[0]
        elif self._is_line_landmark(other_lm):
            other_pts = self._get_line_points(other_lm)
            if not other_pts:
                return True
            other_x = sum(px for px, _py in other_pts) / len(other_pts)
        else:
            other_pt = pts[other_lm]
            if isinstance(other_pt, tuple):
                other_x = float(other_pt[0])
            else:
                return True

        is_pa = self.current_image_direction == "PA"

        if is_left_landmark:
            if is_pa:
                is_valid = new_x < other_x
                side_word = "LEFT"
            else:
                is_valid = new_x > other_x
                side_word = "RIGHT"
            bad_msg = (
                f'"{lm}" must be to the {side_word} of "{other_lm}" '
                f"({self.current_image_direction} image).\n\n"
                f"Current click x = {new_x:.1f}\n"
                f"{other_lm} x = {other_x:.1f}"
            )
        else:
            if is_pa:
                is_valid = new_x > other_x
                side_word = "RIGHT"
            else:
                is_valid = new_x < other_x
                side_word = "LEFT"
            bad_msg = (
                f'"{lm}" must be to the {side_word} of "{other_lm}" '
                f"({self.current_image_direction} image).\n\n"
                f"Current click x = {new_x:.1f}\n"
                f"{other_lm} x = {other_x:.1f}"
            )

        if not is_valid:
            messagebox.showwarning("Left/Right Landmark Order", bad_msg)
            return False

        return True

    def _on_left_drag(self, event) -> None:
        if not self.current_image:
            return

        drawing_border = bool(self.drawing_it_edge_border)
        self.last_mouse_canvas_pos = (event.x, event.y)
        self._update_mouse_crosshair(event.x, event.y)

        if self.extended_crosshair_enabled.get():
            self._update_extended_crosshair(event.x, event.y)
        else:
            self._hide_extended_crosshair()

        if not drawing_border:
            self._update_zoom_view(event.x, event.y)

        if self.hover_enabled.get():
            self._update_hover_circle(event.x, event.y)
        else:
            self._hide_hover_circle()

        if getattr(self, "multi_review_mode", False):
            return

        x0, y0, x1, y1 = self._display_rect()
        if not (x0 <= event.x < x1 and y0 <= event.y < y1):
            return

        xi, yi = self._screen_to_img(event.x, event.y)
        x, y = self._clamp_img_point(xi, yi)

        if drawing_border:
            lm = self.selected_landmark.get().strip()
            if self._is_border_segmentation_landmark(lm):
                last = self.it_edge_border_last_img_pos or (x, y)
                self._draw_it_edge_border_segment(
                    lm,
                    last,
                    (x, y),
                    refresh_overlay=False,
                    sync_annotation=False,
                    refresh_controls=False,
                )
                if self._it_edge_border_mode_value() == "erase":
                    self._update_it_edge_border_overlay_for_drag(lm)
                else:
                    self._draw_live_it_edge_border_segment(lm, last, (x, y))
                self._update_it_edge_border_zoom_for_drag(event.x, event.y)
                self.it_edge_border_last_img_pos = (x, y)
            return

        if self.dragging_landmark is None:
            return

        if self._is_intermediate_acetabular_landmark(self.dragging_landmark):
            if self._set_intermediate_acetabular_point(self.dragging_landmark, (x, y)):
                self._draw_points()
                self._refresh_zoom_landmark_overlay()
                self.dirty = True
            return

        if self._is_ellipse_landmark(self.dragging_landmark):
            if self.dragging_ellipse_handle is not None:
                self._move_ellipse_handle(
                    self.dragging_landmark,
                    self.dragging_ellipse_handle,
                    (x, y),
                    sync_dependents=False,
                )
                self._draw_points(lightweight=True)
                self.dirty = True
                return

            if self.dragging_ellipse_whole:
                if self.dragging_ellipse_last_img_pos is None:
                    self.dragging_ellipse_last_img_pos = (x, y)
                    return

                last_x, last_y = self.dragging_ellipse_last_img_pos
                dx = x - last_x
                dy = y - last_y
                if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                    return

                self._shift_ellipse(
                    self.dragging_landmark, (dx, dy), sync_dependents=False
                )
                self.dragging_ellipse_last_img_pos = (x, y)
                self._draw_points(lightweight=True)
                self.dirty = True
                return

            return

        pts = self._get_line_points(self.dragging_landmark)
        if not pts:
            return

        if self.dragging_point_index is not None:
            if self.dragging_point_index >= len(pts):
                return

            pts[self.dragging_point_index] = (x, y)
            self._set_line_points(self.dragging_landmark, pts)
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True
            return

        if self.dragging_line_whole:
            if len(pts) != 2:
                return

            if self.dragging_line_last_img_pos is None:
                self.dragging_line_last_img_pos = (x, y)
                return

            last_x, last_y = self.dragging_line_last_img_pos
            dx = x - last_x
            dy = y - last_y

            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                return

            new_pts = []
            for px, py in pts:
                nx, ny = self._clamp_img_point(px + dx, py + dy)
                new_pts.append((nx, ny))

            actual_dx = new_pts[0][0] - pts[0][0]
            actual_dy = new_pts[0][1] - pts[0][1]
            new_pts = [
                self._clamp_img_point(px + actual_dx, py + actual_dy) for px, py in pts
            ]

            self._set_line_points(self.dragging_landmark, new_pts)
            self.dragging_line_last_img_pos = (x, y)
            self._draw_points()
            self._refresh_zoom_landmark_overlay()
            self.dirty = True

    def _on_left_release(self, event) -> None:
        x0, y0, x1, y1 = self._display_rect()
        finishing_border_stroke = bool(self.drawing_it_edge_border)
        if self.current_image and x0 <= event.x < x1 and y0 <= event.y < y1:
            self.last_mouse_canvas_pos = (event.x, event.y)
            self._update_mouse_crosshair(event.x, event.y)
            if not finishing_border_stroke:
                self._update_zoom_view(event.x, event.y)
            self._update_line_preview(event.x, event.y)

        if getattr(self, "multi_review_mode", False):
            return

        if self.drawing_it_edge_border:
            lm = self.selected_landmark.get().strip()
            if (
                self.current_image
                and x0 <= event.x < x1
                and y0 <= event.y < y1
                and self._is_border_segmentation_landmark(lm)
            ):
                xi, yi = self._screen_to_img(event.x, event.y)
                x, y = self._clamp_img_point(xi, yi)
                last = self.it_edge_border_last_img_pos or (x, y)
                self._draw_it_edge_border_segment(
                    lm,
                    last,
                    (x, y),
                    refresh_overlay=False,
                    sync_annotation=False,
                    refresh_controls=False,
                )
            self.drawing_it_edge_border = False
            self.it_edge_border_last_img_pos = None
            if self._is_border_segmentation_landmark(lm):
                if self.dirty:
                    mask = self._get_it_edge_border_mask(lm, create=False)
                    self._sync_it_edge_annotation_from_mask(lm, mask)
                    pts, _quality = self._get_annotations()
                    self._update_found_checks(pts)
                    self._refresh_image_listbox()
                if self.dirty:
                    self._clear_it_edge_border_live_overlay(lm)
                    self._draw_points()
                    if self.current_image and x0 <= event.x < x1 and y0 <= event.y < y1:
                        self._update_it_edge_border_zoom_for_drag(
                            event.x,
                            event.y,
                            force=True,
                        )
                    self._autosave_annotation_change()
                    self._advance_to_next_protocol_landmark_if_current_complete(lm)
            return

        if self.dragging_landmark is not None:
            released_lm = self.dragging_landmark
            was_ellipse_drag = self._is_ellipse_landmark(released_lm) and (
                self.dragging_ellipse_handle is not None or self.dragging_ellipse_whole
            )
            self.dragging_landmark = None
            self.dragging_point_index = None
            self.dragging_line_whole = False
            self.dragging_line_last_img_pos = None
            self.dragging_ellipse_handle = None
            self.dragging_ellipse_whole = False
            self.dragging_ellipse_last_img_pos = None
            if was_ellipse_drag:
                self._update_auto_acetabular_points_for_ellipse(released_lm)
                pts, _quality = self._get_annotations()
                self._update_found_checks(pts)
                self._draw_points()
                self._refresh_zoom_landmark_overlay()
                self._advance_to_next_protocol_landmark_if_current_complete(released_lm)
            self._autosave_annotation_change()

    def _delete_current_landmark(self) -> None:
        lm = self.selected_landmark.get().strip()
        if not lm:
            messagebox.showwarning(
                "No Landmark", "Please select a primary landmark in the list."
            )
            return
        self._delete_landmark_annotation(lm, confirm=True, show_review_prompt=True)
        return

    # Triggers re-segmentation if the selected landmark supports segmentation.
    def _resegment_selected_if_needed(self) -> None:
        lm = self.selected_landmark.get()
        if self._is_auto_segmentation_landmark(lm):
            self._store_current_settings_for(lm)
            self._resegment_for(lm)

    # Runs the chosen segmentation method for a landmark and updates overlay.
    def _resegment_for(self, lm: str, apply_saved_settings: bool = False) -> None:
        if not self._is_auto_segmentation_landmark(lm):
            return
        if self.current_image_path is None:
            return
        if apply_saved_settings:
            self._apply_settings_to_ui_for(lm)
        else:
            self._store_current_settings_for(lm)
        seed = self.last_seed.get(lm)
        if seed is None or cv2 is None or self.current_image is None:
            return
        x, y = seed
        mask = self._segment_with_fallback(x, y, lm)

        if mask is None:
            mask_key = self._segmentation_mask_key()
            if mask_key is not None:
                self.seg_masks.get(mask_key, {}).pop(lm, None)
            self._remove_overlay_for(lm)
            self.dirty = True
            self._refresh_image_listbox()
            return
        mask = self._grow_shrink(mask, self.grow_shrink.get())
        mask_key = self._segmentation_mask_key()
        if mask_key is None:
            return
        self.seg_masks.setdefault(mask_key, {})[lm] = mask
        self._update_overlay_for(lm)
        self.dirty = True
        self._refresh_image_listbox()

    def _segment_with_fallback(self, x: int, y: int, lm: str) -> np.ndarray | None:
        _ = lm
        methods = (
            (self._segment_adaptive_cc, self._segment_ff)
            if self.method.get() == "Adaptive CC"
            else (self._segment_ff, self._segment_adaptive_cc)
        )
        for segment in methods:
            mask = segment(x, y)
            if mask is not None:
                return mask
        return None

    def _percentile_contrast_stretch(
        self, img: np.ndarray, low_percent: int, high_percent: int
    ) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        p_low, p_high = np.percentile(gray, (low_percent, high_percent))

        if p_high <= p_low:
            return img

        return np.clip((img - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)

    # Converts image to preprocessed grayscale (CLAHE + blur).
    def _preprocess_gray(self):
        img_rgb = np.array(self.current_image)
        if img_rgb.shape[-1] == 3:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb

        if self.use_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    # Segments a region via flood fill with optional edge-lock barrier.
    def _segment_ff(self, x: int, y: int) -> np.ndarray | None:
        gray = self._preprocess_gray()
        h, w = gray.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        barrier = np.zeros((h, w), np.uint8)
        if self.edge_lock.get():
            edges = cv2.Canny(gray, 40, 120)
            k = max(1, min(5, int(self.edge_lock_width.get())))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
            )
            barrier = cv2.dilate(edges, kernel, iterations=1)
            barrier = (barrier > 0).astype(np.uint8)
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        ff_mask[0, :], ff_mask[-1, :], ff_mask[:, 0], ff_mask[:, -1] = 1, 1, 1, 1
        if self.edge_lock.get():
            ff_mask[1:-1, 1:-1][barrier == 1] = 1
        sens = int(self.fill_sensitivity.get())
        tol = max(1, min(80, 2 * sens + 2))
        img_ff = gray.copy()
        flags = cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8)
        try:
            _area, _, _, _ = cv2.floodFill(
                img_ff,
                ff_mask,
                seedPoint=(int(x), int(y)),
                newVal=0,
                loDiff=tol,
                upDiff=tol,
                flags=flags,
            )
        except cv2.error:
            return None
        region = (ff_mask[1:-1, 1:-1] == 255).astype(np.uint8)
        region = self._sanity_and_clean(region)
        return region

    # Segments a region via adaptive thresholding and connected components.
    def _segment_adaptive_cc(self, x: int, y: int) -> np.ndarray | None:
        gray = self._preprocess_gray()
        h, w = gray.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        sens = int(self.fill_sensitivity.get())
        block = max(11, 2 * (5 + sens // 2) + 1)
        C = max(2, min(15, 12 - sens // 5))
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, C
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        labels = cv2.connectedComponentsWithStats(
            (thr > 0).astype(np.uint8), connectivity=8
        )[1]
        lbl = labels[int(y), int(x)]
        if lbl == 0:
            r = 3
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            patch = labels[y0:y1, x0:x1]
            u = np.unique(patch)
            u = u[u != 0]
            if u.size == 0:
                return None
            lbl = int(u[0])
        region = (labels == lbl).astype(np.uint8)
        region = self._sanity_and_clean(region)
        return region

    # Rejects implausible masks and performs a closing for cleanup.
    def _sanity_and_clean(self, mask: np.ndarray) -> np.ndarray | None:
        h, w = mask.shape[:2]
        area = int(mask.sum())
        if area < 30:
            return None
        if area > 0.7 * w * h:
            return None
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
        return (mask > 0).astype(np.uint8)

    # Dilates (positive) or erodes (negative) a mask by the requested steps.
    def _grow_shrink(self, mask: np.ndarray, steps: int) -> np.ndarray:
        if steps == 0:
            return (mask > 0).astype(np.uint8)
        k = min(25, max(1, abs(steps)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        if steps > 0:
            out = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        else:
            out = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return (out > 0).astype(np.uint8)

    # Returns a dict of the current UI segmentation settings.
    def _current_settings_dict(self) -> Dict:
        return {
            "method": self.method.get(),
            "sens": int(self.fill_sensitivity.get()),
            "edge_lock": 1 if self.edge_lock.get() else 0,
            "edge_width": int(self.edge_lock_width.get()),
            "clahe": 1 if self.use_clahe.get() else 0,
            "grow": int(self.grow_shrink.get()),
        }

    # Stores current UI settings for a specific landmark on this image.
    def _store_current_settings_for(self, lm: str) -> None:
        key = self._segmentation_mask_key()
        if key is None:
            return
        per_img = self.lm_settings.setdefault(key, {})
        per_img[lm] = self._current_settings_dict()

    # Applies saved settings for a landmark back into the UI controls.
    def _apply_settings_to_ui_for(self, lm: str) -> None:
        key = self._segmentation_mask_key()
        st = self.lm_settings.get(key, {}).get(lm) if key is not None else None
        if not st:
            return
        self.method.set(st["method"])
        is_ff = st["method"] == "Flood Fill"
        self.use_ff.set(is_ff)
        self.use_adap_cc.set(not is_ff)
        try:
            self.fill_sensitivity.set(int(st["sens"]))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to apply sensitivity setting for {lm}: {e}")
        try:
            self.edge_lock.set(bool(int(st["edge_lock"])))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to apply edge_lock setting for {lm}: {e}")
        try:
            self.edge_lock_width.set(int(st["edge_width"]))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to apply edge_width setting for {lm}: {e}")
        try:
            self.use_clahe.set(bool(int(st["clahe"])))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to apply clahe setting for {lm}: {e}")
        try:
            self.grow_shrink.set(int(st["grow"]))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to apply grow setting for {lm}: {e}")

    # Toggle visibility.
    def _set_selected_visibility(self, visible: bool) -> None:
        lm = self.selected_landmark.get()
        if not lm:
            return
        var = self.landmark_visibility.get(lm)
        if var is None:
            return
        var.set(visible)
        self._draw_points()

    def _on_arrow_left(self, event) -> None:
        self._set_selected_visibility(False)

    def _on_arrow_right(self, event) -> None:
        self._set_selected_visibility(True)

    def _format_shortcuts(
        self,
        rows,
        width: int = 60,
        gap_min: int = 2,
        leader: str = ".",
    ):
        """
        2-column text block with an 'hfill' made of leader characters (e.g. dots).
        Looks good even with proportional fonts.
        """

        key_w = max(len(k) for k, _ in rows)
        lines = []

        for keys, action in rows:
            left = f"{keys:<{key_w}}"
            # Fill the middle with dots/spaces so the action ends at width-ish
            fill_len = max(gap_min, width - len(left) - len(action))
            fill = (leader * fill_len) if leader else (" " * fill_len)
            lines.append(f"{left} {fill} {action}")

        return "\n".join(lines)

    def _on_h_press(self, event=None) -> None:
        shortcuts = [
            ("<up> / ↑ / d", "Previous landmark"),
            ("<down> / ↓ / f", "Next landmark"),
            ("<left> / <--", "Hide selected landmark"),
            ("<right> / -->", "Show selected landmark"),
            ("b / Ctrl+b", "Previous image"),
            ("n / Ctrl+n", "Next image"),
            ("1-9", "Choose displayed option"),
            ("Tab", "Next primary landmark"),
            ("Backspace / Delete", "Delete selected landmark"),
            ("h", "Show this help"),
            ("?", "Show landmark reference"),
            ("Mouse click", "Place landmark"),
            ("Mouse wheel", "Adjust hover radius"),
        ]
        help_text = self._format_shortcuts(
            shortcuts,
            width=60,
            leader=".",
        )

        win = tk.Toplevel(self)
        win.title("Help & Keyboard Shortcuts")
        win.transient(self)
        win.resizable(False, False)

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(
            frm,
            text="Keyboard Shortcuts",
            font=self.heading_font,
        ).pack(anchor="w", pady=(0, 8))

        text = tk.Text(
            frm,
            wrap="none",
            height=len(shortcuts) + 1,
            borderwidth=0,
            highlightthickness=0,
        )
        text.pack(fill="both", expand=True)

        text.configure(font=self.dialogue_font)

        text.insert("1.0", help_text)
        text.configure(state="disabled")

        ttk.Button(
            frm,
            text="Close",
            command=win.destroy,
        ).pack(anchor="e", pady=(10, 0))

    def _open_landmark_reference(self, event=None) -> None:
        """Open or focus the landmark reference popup."""
        if self._landmark_ref is None:
            return

        if self._landmark_ref_dialog is not None:
            try:
                self._landmark_ref_dialog.lift()
                self._landmark_ref_dialog.focus_force()
                return
            except tk.TclError:
                # Window was destroyed outside our tracking
                self._landmark_ref_dialog = None

        self._landmark_ref_dialog = LandmarkReferenceDialog(
            parent=self,
            reference=self._landmark_ref,
            current_landmark=self.selected_landmark.get() or None,
            on_close=self._on_landmark_ref_dialog_closed,
        )

    def _on_landmark_ref_dialog_closed(self) -> None:
        """Clear the dialog reference when the popup is closed."""
        self._landmark_ref_dialog = None

    def _configure_linux_fonts(self) -> None:
        # (optional) scaling tweak, only on Linux
        self.tk.call("tk", "scaling", 1.25)

        # Pick whatever you decided works well in your env
        self.heading_font = tkfont.Font(
            family="Liberation Sans", size=15, weight="bold"
        )
        self.dialogue_font = tkfont.Font(
            family="Liberation Sans", size=12, weight="bold"
        )

        # Tk named fonts (classic tk widgets)
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkFixedFont",
            "TkMenuFont",
            "TkHeadingFont",
        ):
            f = tkfont.nametofont(name)
            f.configure(family="Liberation Sans", size=12)

        # ttk styles
        style = ttk.Style(self)
        style.theme_use(style.theme_use())  # force theme init / refresh

        style.configure(".", font=self.dialogue_font)
        for s in (
            "TLabel",
            "TButton",
            "TCheckbutton",
            "TRadiobutton",
            "TEntry",
            "TCombobox",
            "TMenubutton",
            "TNotebook",
            "TNotebook.Tab",
            "Treeview",
            "Treeview.Heading",
            "TLabelframe.Label",
        ):
            style.configure(s, font=self.dialogue_font)
        return

    def _configure_annotation_label_fonts(self) -> None:
        base_font = self.__dict__.get("dialogue_font")
        if base_font is None:
            return
        self.primary_label_font = base_font.copy()
        self.primary_label_font.configure(weight="bold")
        self.contour_label_font = base_font.copy()
        self.contour_label_font.configure(slant="roman", weight="normal")
        self.derived_label_font = base_font.copy()
        self.derived_label_font.configure(weight="normal", underline=False)
        self.categorical_shortcut_font = base_font.copy()
        try:
            size = int(self.categorical_shortcut_font.cget("size"))
        except Exception:
            size = 10
        if size >= 0:
            self.categorical_shortcut_font.configure(size=max(8, size - 2), weight="normal")
        else:
            self.categorical_shortcut_font.configure(size=min(-8, size + 2), weight="normal")

    def _configure_landmark_table_header_font(self) -> None:
        base_font = self.__dict__.get("dialogue_font") or self.__dict__.get(
            "heading_font"
        )
        if base_font is None:
            self.landmark_table_header_font = tkfont.nametofont("TkHeadingFont").copy()
            return
        self.landmark_table_header_font = base_font.copy()
        try:
            size = int(self.landmark_table_header_font.cget("size"))
        except Exception:
            size = 12
        step = 1 if size >= 0 else -1
        self.landmark_table_header_font.configure(size=size + step, weight="bold")

    def _refresh_image_listbox(self) -> None:
        if self.image_tree is None:
            return

        self._suspend_image_tree_select = True
        try:
            for item in self.image_tree.get_children():
                self.image_tree.delete(item)

            for idx, path in enumerate(self.images):
                progress = self._image_progress_text(path)
                self.image_tree.insert(
                    "",
                    "end",
                    iid=str(idx),
                    values=(extract_filename(path), progress),
                    tags=("done",) if self._image_progress_done(path) else (),
                )

            self.image_tree.tag_configure("done", foreground="green")

            if 0 <= self.current_image_index < len(self.images):
                iid = str(self.current_image_index)
                if self.image_tree.exists(iid):
                    self.image_tree.selection_set(iid)
                    self.image_tree.focus(iid)
                    self.image_tree.see(iid)
            else:
                self.image_tree.selection_remove(self.image_tree.selection())
        finally:
            if not getattr(self, "_navigation_in_progress", False):
                self._suspend_image_tree_select = False

    def _on_image_list_select(self, _event=None) -> None:
        if self.image_tree is None or getattr(
            self, "_suspend_image_tree_select", False
        ):
            return

        selection = self.image_tree.selection()
        if not selection:
            return

        idx = int(selection[0])
        if idx == self.current_image_index or not (0 <= idx < len(self.images)):
            return

        self._navigation_in_progress = True
        self._suspend_image_tree_select = True
        try:
            if not self._maybe_save_before_destructive_action("switch images"):
                self._refresh_image_listbox()
                return

            self.current_image_index = idx
            self.load_image_from_path(self.images[idx])
        finally:
            self._navigation_in_progress = False
            self._suspend_image_tree_select = False

    def _landmark_counts_for_progress(self, lm: str) -> bool:
        return not (
            self._is_protocol_derived_feature(lm)
            or self._is_derived_acetabular_landmark(lm)
            or self._is_intermediate_acetabular_landmark(lm)
        )

    def _count_allowed_landmarks_for_current_image(self, image_path: Path) -> int:
        key = self._path_key(image_path)
        idx = self.image_index_map.get(key)
        if idx is None:
            return 0

        if (
            self.current_image_path is not None
            and self._path_key(self.current_image_path) == key
        ):
            view = self.current_view_var.get().strip()
        else:
            record = self.json_data["images"][idx]
            view = record.get("view")

        if not view or view not in self.allowed_views:
            return 0

        return sum(
            1
            for lm in self.allowed_views.get(view, [])
            if self._landmark_counts_for_progress(lm)
        )

    def _count_completed_landmarks_for_current_image(self, image_path: Path) -> int:
        key = self._path_key(image_path)
        idx = self.image_index_map.get(key)
        if idx is None:
            return 0

        is_current_image = (
            self.current_image_path is not None
            and self._path_key(self.current_image_path) == key
        )
        if is_current_image:
            view = self.current_view_var.get().strip()
            annotations = self.annotations.get(key, {}) or {}
            meta = self.landmark_meta.get(key, {})
            allowed = (
                set(self.allowed_views.get(view, []))
                if view and view in self.allowed_views
                else set(annotations.keys())
            )
            return sum(
                1
                for lm in allowed
                if self._landmark_counts_for_progress(lm)
                and (
                    self._direct_landmark_complete(lm)
                    if lm in PROTOCOL_DIRECT_LANDMARKS
                    else annotations.get(lm) is not None
                    or bool(
                        self._normalize_direct_annotation_status(
                            meta.get(lm, {}).get("status", "")
                        )
                    )
                )
            )

        record = self.json_data["images"][idx]
        view = record.get("view")
        annotations = record.get("annotations", {}) or {}

        if view and view in self.allowed_views:
            allowed = set(self.allowed_views.get(view, []))
        else:
            allowed = set(annotations.keys())

        done = 0
        for lm in allowed:
            if not self._landmark_counts_for_progress(lm):
                continue
            raw = annotations.get(lm)
            if raw is None:
                continue

            if isinstance(raw, dict):
                value = raw.get("value")
                status = self._normalize_direct_annotation_status(
                    raw.get("status", "")
                )
            else:
                value = raw
                status = ""

            if value is not None or status:
                done += 1

        return done

    def _image_progress_text(self, image_path: Path) -> str:
        if getattr(self, "multi_review_mode", False):
            done, total = self._multi_review_counts_for_image_path(image_path)
            return f"{done}/{total}"

        key = self._path_key(image_path)
        idx = self.image_index_map.get(key)
        if idx is None:
            return "0/?"

        if (
            self.current_image_path is not None
            and self._path_key(self.current_image_path) == key
        ):
            view = self.current_view_var.get().strip()
        else:
            record = self.json_data["images"][idx]
            view = record.get("view")

        done = self._count_completed_landmarks_for_current_image(image_path)

        if not view or view not in self.allowed_views:
            return f"{done}/?"

        total = self._count_allowed_landmarks_for_current_image(image_path)
        return f"{done}/{total}"

    def _image_progress_done(self, image_path: Path) -> bool:
        if getattr(self, "multi_review_mode", False):
            done, total = self._multi_review_counts_for_image_path(image_path)
            return total > 0 and done >= total

        key = self._path_key(image_path)
        idx = self.image_index_map.get(key)
        if idx is None:
            return False

        if (
            self.current_image_path is not None
            and self._path_key(self.current_image_path) == key
        ):
            view = self.current_view_var.get().strip()
        else:
            record = self.json_data["images"][idx]
            view = record.get("view")

        if not view or view not in self.allowed_views:
            return False

        done = self._count_completed_landmarks_for_current_image(image_path)
        total = self._count_allowed_landmarks_for_current_image(image_path)
        return total > 0 and done >= total

    def _widget_y_in_inner(self, widget) -> int:
        y = 0
        w = widget
        while w is not None and w is not self.lp_inner:
            y += w.winfo_y()
            parent_name = w.winfo_parent()
            if not parent_name:
                break
            w = w.nametowidget(parent_name)
        return y

    def _get_csv_images_from_directory(self, dir: Path) -> list[Path]:
        try:
            df = pd.read_csv(self.abs_csv_path)
        except Exception:
            return []

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT image_filename FROM annotations WHERE verified = 0"
                )
                rows = cursor.fetchall()
        except sqlite3.Error:
            return []

        csv_files = [elem[0] for elem in rows if elem is not None]

        local_dir_csv_files = []
        for root, dirs, files in dir.walk():
            for file in files:
                if file in csv_files:
                    local_dir_csv_files.append(Path(root / file))

        return local_dir_csv_files

    def _find_unannotated_images(self) -> None:
        """Scan directory for images not in CSV, enter queue mode."""
        if not hasattr(self, "abs_csv_path") or not self.abs_csv_path:
            messagebox.showwarning(
                "No CSV", "Please load a CSV file first (Load CSV button)"
            )
            return

        # Select directory to scan
        scan_dir = filedialog.askdirectory(
            initialdir=BASE_DIR, title="Select folder to scan for unannotated images"
        )

        if not scan_dir:
            return

        scan_path = Path(scan_dir)

        # Load existing annotations from CSV
        try:
            df = pd.read_csv(self.abs_csv_path)
            col = self._detect_path_column(df)

            # Build a set of files that are "truly annotated"
            # (either have landmarks OR have been quality-rated as bad)
            annotated_filenames: Set = set()

            for _, row in df.iterrows():
                # filename = Path(str(row[col])).name.lower()
                filename = extract_filename(row[col])

                # Check if any landmark columns have values
                has_landmarks = any(
                    pd.notna(row.get(lm)) and str(row.get(lm, "")).strip()
                    for lm in self.landmarks
                )

                # Check image quality
                quality = row.get("image_quality", 0)
                try:
                    quality = int(quality)
                except (ValueError, TypeError):
                    quality = 0

                # Consider "annotated" if:
                # 1. Has landmarks, OR
                # 2. Has no landmarks but quality != 0 (marked as bad image)
                if has_landmarks or quality != 0:
                    annotated_filenames.add(filename)

        except Exception as e:
            messagebox.showerror("CSV Error", f"Failed to read CSV:\n{e}")
            return

        # Recursively find all image files
        all_images = []
        for dirpath, dirnames, filenames in scan_path.walk():
            if "duplicates" in dirnames:
                dirnames.remove("duplicates")
            for file in filenames:
                if Path(file).suffix.lower() in self.possible_image_suffix:
                    all_images.append(dirpath / file)
                    pass
                pass
            pass

        # Filter to unannotated only
        unannotated = [
            img
            for img in all_images
            if (img.name.lower() not in annotated_filenames)
            and (img.parent.name != "duplicates")
        ]

        if not unannotated:
            messagebox.showinfo(
                "All Done!", f"All images in {scan_path.name} are already annotated!"
            )
            return

        # Sort for consistent ordering
        unannotated.sort()

        # Enter queue mode
        self.unannotated_queue = unannotated
        self.queue_index = 0
        self.queue_mode = True

        # Load first unannotated image
        self.absolute_current_image_path = unannotated[0]
        self.load_image_from_path(unannotated[0])

        self._update_queue_status()

        messagebox.showinfo(
            "Queue Mode",
            f"Found {len(unannotated)} unannotated images.\n\n"
            f"Use Next/Prev (or N/B keys) to cycle through them.\n"
            f"Click 'Exit Queue Mode' to return to normal browsing.",
        )
        if self.exit_queue_btn is not None:
            self.exit_queue_btn.config(state="normal")
        return

    def _update_queue_status(self) -> None:
        """Update the queue status label."""
        if self.queue_mode and self.unannotated_queue:
            self.queue_status_var.set(
                f"Queue: {self.queue_index + 1} / {len(self.unannotated_queue)} unannotated"
            )
        elif self.check_csv_mode and self.csv_path_queue:
            self.queue_status_var.set(
                f"Queue: {self.csv_index + 1} / {len(self.csv_path_queue)} remaining"
            )
        else:
            self.queue_status_var.set("")

    def _exit_queue_mode(self) -> None:
        """Return to normal directory browsing."""
        self.queue_mode = False
        self.unannotated_queue.clear()
        self.queue_index = 0
        self._update_queue_status()
        if self.exit_queue_btn is not None:
            self.exit_queue_btn.config(state="disabled")
        messagebox.showinfo("Queue Mode", "Returned to normal directory browsing.")

    def _change_method_to_ff(self) -> None:
        self.use_adap_cc.set(False)
        self.use_ff.set(True)
        self.method.set("Flood Fill")
        self._resegment_selected_if_needed()
        return

    def _change_method_to_acc(self) -> None:
        self.use_adap_cc.set(True)
        self.use_ff.set(False)
        self.method.set("Adaptive CC")
        self._resegment_selected_if_needed()
        return

    def _on_segmentation_method_changed(self) -> None:
        is_ff = self.method.get() == "Flood Fill"
        self.use_ff.set(is_ff)
        self.use_adap_cc.set(not is_ff)
        self._resegment_selected_if_needed()

    def _check_csv_images(self):
        self.abs_csv_path = filedialog.askopenfilename(
            initialdir=BASE_DIR / "data/csv", filetypes=[("CSV File", ("*.csv"))]
        )
        if self.abs_csv_path:
            df: pd.DataFrame = pd.read_csv(self.abs_csv_path)
            csv_path_column = self._detect_path_column(df)
            self.load_landmarks_from_csv(self.abs_csv_path)
            self.check_csv_mode = True
            if self.csv_local_image_directory_path == None:
                self.csv_local_image_directory_path = filedialog.askdirectory()
            self.csv_path_queue = self._get_csv_images_from_directory(
                Path(self.csv_local_image_directory_path)
            )
            if len(self.csv_path_queue) == 0:
                self.csv_local_image_directory_path = filedialog.askdirectory()

            self.csv_path_queue = self._get_csv_images_from_directory(
                Path(self.csv_local_image_directory_path)
            )

            self.absolute_current_image_path = Path(self.csv_path_queue[0])
            self.load_image_from_path(self.absolute_current_image_path)
            self._update_queue_status()

        return

    def _exit_csv_check_mode(self):
        self.check_csv_mode = False
        self.csv_path_queue.clear()
        self.csv_index = 0
        if self.exit_csv_check_btn is not None:
            self.exit_csv_check_btn.config(state="disabled")
        self._update_queue_status()

        return


if __name__ == "__main__":
    # Feature 1 change
    # Feature 2 change
    # Testing Tags
    app = AnnotationGUI()
    app.option_add("*Label.font", "helvetica 20 bold")
    app.mainloop()
