#!/usr/bin/env python3

# pyright: reportMissingImports=false

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import main as annotator_main
from main import (
    AnnotationGUI,
    MANUAL_SEGMENTATION_MASK_SHAPE,
    MANUAL_SEGMENTATION_MASK_SIZE,
    PELVIS_SEGMENTATION_BACKEND_OPTIONS,
    PELVIS_SEGMENTATION_LANDMARK,
    PROTOCOL_DIRECT_LANDMARKS,
    PROTOCOL_DERIVED_CLASSIFIED_COLOR,
    PROTOCOL_DERIVED_UNCLASSIFIED_COLOR,
    PROTOCOL_LEFT_DIRECT_LANDMARKS,
    PROTOCOL_LEFT_DISPLAY_ORDER,
    PROTOCOL_RIGHT_DIRECT_LANDMARKS,
    PROTOCOL_STATUS_MISSING_COLOR,
    PROTOCOL_STATUS_NOT_APPLICABLE_COLOR,
)


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_gui_stub(tmp_path: Path):
    gui = AnnotationGUI.__new__(AnnotationGUI)
    gui.json_path = tmp_path / "annotations.json"
    gui.json_dir = tmp_path
    gui.json_data = {"landmarks": [], "views": {}, "images": []}
    gui.image_index_map = {}
    gui.images = []
    gui.landmarks = ["L-FA", "L-FAC", "LOB", "RKNEE"]
    gui.allowed_views = {"ap": list(gui.landmarks)}
    gui.line_landmarks = {"L-FA", "R-FA", "L-D-FA", "R-D-FA", "POD"}
    gui.ellipse_landmarks = {"L-FAC", "R-FAC", "L-C-FAC", "R-C-FAC"}
    gui.annotations = {}
    gui.categorical_annotations = {}
    gui.seg_masks = {}
    gui.it_edge_border_masks = {}
    gui.protocol_pod_constructions = {}
    gui.seg_img_objs = {}
    gui.seg_item_ids = {}
    gui.it_edge_border_img_objs = {}
    gui.it_edge_border_item_ids = {}
    gui.pelvis_segmentation_undo_stack = {}
    gui.pelvis_segmentation_running = False
    gui.pelvis_segmentation_backend_var = FakeVar("auto")
    gui.pelvis_segmenters = {}
    gui.segment_pelvis_btn = None
    gui.undo_pelvis_btn = None
    gui.pelvis_segmentation_status_label = None
    gui.pelvis_segmentation_status_var = FakeVar("")
    gui.last_seed = {}
    gui.lm_settings = {}
    gui.landmark_meta = {}
    gui.landmark_visibility = {}
    gui.landmark_found = {}
    gui.landmark_found_widgets = {}
    gui.landmark_flagged = {}
    gui.landmark_flag_widgets = {}
    gui.multi_review_mode = False
    gui.multi_review_reviewer_order = []
    gui.multi_review_annotations = {}
    gui.multi_review_meta = {}
    gui.multi_review_image_key_by_path = {}
    gui.multi_review_temp_annotations = {}
    gui.multi_review_visibility = {}
    gui.current_image_path = tmp_path / "image.png"
    gui.current_image_quality = 0
    gui.image_tree = None
    gui._categorical_shortcut_focus_key = ""
    gui.current_image_flag = True
    gui.current_view_var = FakeVar("ap")
    gui.left_hip_category_var = FakeVar("")
    gui.right_hip_category_var = FakeVar("")
    gui.derived_feature_review_var = FakeVar("")
    gui.derived_feature_status_var = FakeVar("")
    gui.landmark_status_var = FakeVar("")
    gui.landmark_status_combo = None
    gui.derived_feature_combo = None
    gui.derived_feature_status_combo = None
    gui.derived_feature_status_vars = {}
    gui.derived_feature_status_widgets = {}
    gui.derived_feature_status_frame = None
    gui.landmark_correct_labels = {}
    gui.landmark_delete_widgets = {}
    gui.saved_image_snapshots = {}
    gui.autosave_var = FakeVar(True)
    gui._pending_autosave_id = None
    gui._active_landmark_group = ""
    gui.db_path = None
    gui.hover_radii = {}
    gui.method = FakeVar("Flood Fill")
    gui.fill_sensitivity = FakeVar(18)
    gui.edge_lock = FakeVar(True)
    gui.edge_lock_width = FakeVar(2)
    gui.use_clahe = FakeVar(True)
    gui.grow_shrink = FakeVar(3)
    gui.it_edge_border_mode = FakeVar("draw")
    gui.it_edge_border_brush_size = FakeVar(1)
    gui.it_edge_border_brush_scale = None
    gui.selected_landmark = FakeVar("")
    gui.fac_construction_mode = FakeVar("snapped_ac_circle")
    gui.projected_hemisphere_z = FakeVar(70)
    gui.ellipse_snap_to_ac_enabled = FakeVar(True)
    gui.ellipse_lock_major_when_snapped_enabled = FakeVar(True)
    gui.hover_enabled = FakeVar(False)
    gui.femoral_axis_enabled = FakeVar(False)
    gui.right_mouse_held = False
    gui.current_image_direction = "AP"
    gui.zoom_seg_img_obj = None
    gui.disp_scale = 1.0
    return gui


def test_inject_landmarks_constrains_views_to_pdf_protocol(tmp_path):
    gui = make_gui_stub(tmp_path)
    landmarks = ["L-IT", "R-IT", "L-FAC", "R-FAC", "L-E-IT"]
    views = {
        "AP Bilateral": ["L-IT", "R-IT", "L-E-IT", "L-FAC"],
        "AP Unilateral (Left)": ["L-IT", "L-E-IT"],
        "AP Unilateral (Right)": ["R-IT", "R-E-IT"],
    }

    updated = gui._inject_ellipse_landmarks(landmarks, views)

    assert updated[:15] == [
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
    ]
    assert "L-E-IT" not in updated
    assert "L-FAC" not in updated
    assert "L-A-SAB" not in updated
    assert views["AP Bilateral"] == updated
    assert views["AP Unilateral (Left)"] == [
        lm for lm in updated if not lm.startswith("R-") and lm != "POD"
    ]
    assert views["AP Unilateral (Right)"] == [
        lm for lm in updated if not lm.startswith("L-") and lm != "POD"
    ]


def test_protocol_direct_order_matches_updated_pdf_section_4(tmp_path):
    assert PROTOCOL_LEFT_DIRECT_LANDMARKS == [
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
    assert PROTOCOL_RIGHT_DIRECT_LANDMARKS == [
        lm.replace("L-", "R-", 1) for lm in PROTOCOL_LEFT_DIRECT_LANDMARKS
    ]
    assert "L-FA" not in PROTOCOL_DIRECT_LANDMARKS
    assert "R-FA" not in PROTOCOL_DIRECT_LANDMARKS
    assert PROTOCOL_LEFT_DISPLAY_ORDER.index("L-C-LT") < (
        PROTOCOL_LEFT_DISPLAY_ORDER.index("L-C-PMF")
    )
    assert PROTOCOL_LEFT_DISPLAY_ORDER.index("L-C-PMF") < (
        PROTOCOL_LEFT_DISPLAY_ORDER.index("L-C-DMF")
    )
    assert PROTOCOL_LEFT_DISPLAY_ORDER.index("L-D-FA") > (
        PROTOCOL_LEFT_DISPLAY_ORDER.index("L-C-DMF")
    )

    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-PMF", "L-C-LT", "L-C-DMF", "L-D-FA", "L-FA"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")

    assert gui._protocol_direct_order_for_current_view() == [
        "L-C-LT",
        "L-C-PMF",
        "L-C-DMF",
    ]


def test_segmentation_mask_round_trips_for_e_it_landmark(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    key = gui._path_key(gui.current_image_path)
    mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    mask[200:260, 180:240] = 1

    record = {
        "annotations": {
            "L-E-IT": {
                "value": [10, 20, "ACC", 7, 1, 5, 0, 2],
                "segmentation": gui._encode_segmentation_mask(mask),
                "flag": True,
                "note": "edge",
            }
        }
    }

    parsed = gui._parse_annotations_for_record(record)

    assert parsed == {"L-E-IT": (10.0, 20.0)}
    assert "L-E-IT" not in gui.lm_settings.get(key, {})
    assert np.array_equal(gui.seg_masks[key]["L-E-IT"], mask)
    assert gui.landmark_meta[key]["L-E-IT"] == {"flag": True, "note": "edge"}

    gui.annotations[key] = parsed
    data = gui._prepare_landmark_data()

    assert data["L-E-IT"]["value"] == [10.0, 20.0]
    decoded = gui._decode_segmentation_mask(data["L-E-IT"]["segmentation"])
    assert np.array_equal(decoded, mask)


def test_old_it_edge_masks_migrate_to_fixed_512_grid_on_parse(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    key = gui._path_key(gui.current_image_path)
    old_mask = np.zeros((20, 30), dtype=np.uint8)
    old_mask[4:12, 8:20] = 1

    record = {
        "annotations": {
            "L-E-IT": {
                "value": None,
                "segmentation": gui._encode_segmentation_mask(old_mask),
                "edge_border": gui._encode_segmentation_mask(old_mask),
                "flag": False,
                "note": "",
            }
        }
    }

    parsed = gui._parse_annotations_for_record(record)

    assert parsed["L-E-IT"][0] > 0
    assert parsed["L-E-IT"][1] > 0
    assert gui.seg_masks[key]["L-E-IT"].shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )
    assert gui.it_edge_border_masks[key]["L-E-IT"].shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )


def test_pelvis_mask_round_trips_as_mask_only_annotation(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-FA", "LOB"]
    key = gui._path_key(gui.current_image_path)
    mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    gui.seg_masks[key] = {PELVIS_SEGMENTATION_LANDMARK: mask}

    data = gui._prepare_landmark_data(for_json=True)

    assert data[PELVIS_SEGMENTATION_LANDMARK]["value"] is None
    decoded = gui._decode_segmentation_mask(
        data[PELVIS_SEGMENTATION_LANDMARK]["segmentation"]
    )
    assert np.array_equal(decoded, mask)

    gui.seg_masks = {}
    parsed = gui._parse_annotations_for_record({"annotations": data})

    assert PELVIS_SEGMENTATION_LANDMARK not in parsed
    assert np.array_equal(gui.seg_masks[key][PELVIS_SEGMENTATION_LANDMARK], mask)


def test_pelvis_segmentation_undo_restores_previous_mask(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(8, 8))
    key = gui._segmentation_mask_key()
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1, 1] = 1
    second = np.ones((8, 8), dtype=np.uint8)
    gui.seg_masks[key] = {PELVIS_SEGMENTATION_LANDMARK: first}
    gui._draw_points = lambda: None
    gui._refresh_zoom_landmark_overlay = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._autosave_annotation_change = lambda: True
    gui._set_status = lambda message: setattr(gui, "status", message)
    gui._remove_overlay_for = lambda _lm: None

    assert gui._apply_pelvis_segmentation_mask(second, source="test") is True
    assert np.array_equal(gui.seg_masks[key][PELVIS_SEGMENTATION_LANDMARK], second)

    gui._undo_pelvis_segmentation()

    assert np.array_equal(gui.seg_masks[key][PELVIS_SEGMENTATION_LANDMARK], first)
    assert gui.pelvis_segmentation_status_var.get() == "Pelvis segmentation undone."


def test_visible_pelvis_backend_options_are_pelvis_specific():
    modes = [mode for mode, _label in PELVIS_SEGMENTATION_BACKEND_OPTIONS]

    assert "hf_yolo11s_pelvis_xray" in modes
    assert "replicate_unetpp_pelvis" in modes
    assert "rf_ericyum_pelvis_ap" in modes
    assert "rf_sooyeon_pelvis_ap" in modes
    assert "rf_yolov8_pelvis_xray" in modes
    assert "pelvis_onnx" not in modes
    assert "pelvis_torchscript" not in modes
    assert "hip_pelvis_annotator" not in modes
    assert "sam" not in modes
    assert "sam2" not in modes
    assert "medsam" not in modes


def test_it_edge_border_fill_creates_saved_segmentation_mask(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    gui.current_image = SimpleNamespace(size=(20, 20))
    gui.landmark_visibility = {"L-E-IT": FakeVar(True)}
    gui.landmark_found = {"L-E-IT": FakeVar(False)}
    gui._update_overlay_for = lambda _lm: None
    gui._update_it_edge_border_overlay_for = lambda _lm: None
    gui._update_found_checks = lambda _pts: None
    gui._refresh_zoom_landmark_overlay = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._set_status = lambda message: setattr(gui, "status", message)

    key = gui._segmentation_mask_key()
    border = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    border[128, 128:384] = 1
    border[383, 128:384] = 1
    border[128:384, 128] = 1
    border[128:384, 383] = 1
    gui.it_edge_border_masks[key] = {"L-E-IT": border}

    assert gui._fill_it_edge_border("L-E-IT") is True

    filled = gui.seg_masks[key]["L-E-IT"]
    assert filled.shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )
    assert filled[256, 256] == 1
    assert filled[128, 256] == 1
    assert filled[0, 0] == 0
    assert gui.annotations[key]["L-E-IT"] == (9.5, 9.5)
    assert gui.landmark_found["L-E-IT"].get() is True


def test_it_edge_highlighter_uses_fixed_512_grid_for_any_image_size(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    gui.current_image = SimpleNamespace(size=(1200, 300))
    gui.landmark_found = {"L-E-IT": FakeVar(False)}
    gui.landmark_visibility = {"L-E-IT": FakeVar(True)}
    gui.it_edge_border_brush_size.set(1)
    gui._update_it_edge_border_overlay_for = lambda _lm: None
    gui._update_found_checks = lambda _pts: None
    gui._refresh_image_listbox = lambda: None

    gui._draw_it_edge_border_segment("L-E-IT", (0.0, 0.0), (1199.0, 299.0))

    key = gui._segmentation_mask_key()
    mask = gui.it_edge_border_masks[key]["L-E-IT"]
    assert mask.shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )
    assert mask[0, 0] == 1
    assert mask[-1, -1] == 1
    assert "L-E-IT" not in gui.seg_masks.get(key, {})
    assert abs(gui.annotations[key]["L-E-IT"][0] - 599.5) < 1.0
    assert abs(gui.annotations[key]["L-E-IT"][1] - 149.5) < 1.0


def test_protocol_freehand_contour_uses_fixed_512_grid_and_saves_mask(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.current_image = SimpleNamespace(size=(1200, 300))
    gui.landmark_found = {"L-C-ILI": FakeVar(False)}
    gui.landmark_visibility = {"L-C-ILI": FakeVar(True)}
    gui.it_edge_border_brush_size.set(1)
    gui._update_it_edge_border_overlay_for = lambda _lm: None
    gui._update_found_checks = lambda _pts: None
    gui._refresh_image_listbox = lambda: None
    gui._set_status = lambda message: setattr(gui, "status", message)

    gui._draw_it_edge_border_segment("L-C-ILI", (0.0, 0.0), (1199.0, 299.0))

    key = gui._segmentation_mask_key()
    mask = gui.it_edge_border_masks[key]["L-C-ILI"]
    assert mask.shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )
    assert "L-C-ILI" not in gui.seg_masks.get(key, {})

    data = gui._prepare_landmark_data(for_json=True)
    decoded = gui._decode_segmentation_mask(data["L-C-ILI"]["segmentation"])
    assert np.array_equal(decoded, mask)
    assert np.array_equal(
        gui._decode_segmentation_mask(data["L-C-ILI"]["edge_border"]),
        mask,
    )
    assert gui._fill_it_edge_border("L-C-ILI") is False


def test_protocol_ilium_contour_generates_sip_and_lip_from_mask(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.current_image = SimpleNamespace(size=(512, 512))
    gui.landmark_found = {"L-SIP": FakeVar(False), "L-LIP": FakeVar(False)}
    key = gui._segmentation_mask_key()
    mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    mask[20, 100] = 1
    mask[20, 300] = 1
    mask[60, 400] = 1
    gui.it_edge_border_masks[key] = {"L-C-ILI": mask}

    assert gui._update_protocol_derived_features_for_current_image() is True

    assert gui.annotations[key]["L-SIP"] == (300.0, 20.0)
    assert gui.annotations[key]["L-LIP"] == (400.0, 60.0)
    assert gui.landmark_found["L-SIP"].get() is True
    assert gui.landmark_found["L-LIP"].get() is True


def test_protocol_right_ilium_lateral_is_image_left(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["R-C-ILI", "R-SIP", "R-LIP"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.current_image = SimpleNamespace(size=(512, 512))
    gui.landmark_found = {"R-SIP": FakeVar(False), "R-LIP": FakeVar(False)}
    key = gui._segmentation_mask_key()
    mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    mask[20, 100] = 1
    mask[20, 300] = 1
    mask[60, 50] = 1
    gui.it_edge_border_masks[key] = {"R-C-ILI": mask}

    assert gui._update_protocol_derived_features_for_current_image() is True

    assert gui.annotations[key]["R-SIP"] == (100.0, 20.0)
    assert gui.annotations[key]["R-LIP"] == (50.0, 60.0)
    assert gui.landmark_found["R-SIP"].get() is True
    assert gui.landmark_found["R-LIP"].get() is True


def test_protocol_pod_uses_medial_axis_perpendicular_between_lpo_projections(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-PO", "R-C-PO", "L-LPO", "R-LPO", "POD"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.current_image = SimpleNamespace(size=(512, 512))
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    key = gui._segmentation_mask_key()

    left_mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    right_mask = np.zeros_like(left_mask)
    for y in range(100, 301):
        left_mask[y, 310] = 1
        left_mask[y, 356] = 1
        right_mask[y, 202] = 1
        right_mask[y, 156] = 1
    left_mask[140, 376] = 1
    right_mask[260, 136] = 1
    gui.it_edge_border_masks[key] = {
        "L-C-PO": left_mask,
        "R-C-PO": right_mask,
    }

    assert gui._update_protocol_derived_features_for_current_image() is True

    assert gui.annotations[key]["L-LPO"] == (376.0, 140.0)
    assert gui.annotations[key]["R-LPO"] == (136.0, 260.0)
    assert gui.annotations[key]["POD"] == [(356.0, 200.0), (156.0, 200.0)]

    geometry = gui.protocol_pod_constructions[key]
    assert geometry["midpoint"] == (256.0, 200.0)
    assert geometry["left_axis_point"] == (256.0, 140.0)
    assert geometry["right_axis_point"] == (256.0, 260.0)

    class FakeCanvas:
        def __init__(self):
            self.created = []

        def create_line(self, *coords, **kwargs):
            self.created.append(("line", coords, kwargs))
            return len(self.created)

        def create_oval(self, *coords, **kwargs):
            self.created.append(("oval", coords, kwargs))
            return len(self.created)

    canvas = FakeCanvas()
    item_ids = gui._draw_protocol_pod_construction_overlay(
        canvas,
        lambda x, y: (x, y),
        tags="marker",
        draw_pod_line=True,
    )

    assert len(item_ids) >= 8
    assert any(
        kind == "line" and kwargs.get("fill") == "#B875FF"
        for kind, _coords, kwargs in canvas.created
    )
    assert any(
        kind == "line" and kwargs.get("fill") == "black"
        for kind, _coords, kwargs in canvas.created
    )


def test_protocol_pod_medial_axis_includes_pubic_symphysis_contours(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = [
        "L-C-PO",
        "R-C-PO",
        "L-C-PS",
        "R-C-PS",
        "L-LPO",
        "R-LPO",
        "POD",
    ]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.current_image = SimpleNamespace(size=(512, 512))
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    key = gui._segmentation_mask_key()

    left_mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    right_mask = np.zeros_like(left_mask)
    for y in range(100, 301):
        left_mask[y, 310] = 1
        left_mask[y, 356] = 1
        right_mask[y, 202] = 1
        right_mask[y, 156] = 1
    left_mask[140, 376] = 1
    right_mask[260, 136] = 1

    left_ps_mask = np.zeros_like(left_mask)
    right_ps_mask = np.zeros_like(left_mask)
    for y in range(320, 441):
        left_ps_mask[y, 340] = 1
        left_ps_mask[y, 388] = 1
        right_ps_mask[y, 252] = 1
        right_ps_mask[y, 204] = 1

    gui.it_edge_border_masks[key] = {
        "L-C-PO": left_mask,
        "R-C-PO": right_mask,
        "L-C-PS": left_ps_mask,
        "R-C-PS": right_ps_mask,
    }

    assert gui._update_protocol_derived_features_for_current_image() is True

    geometry = gui.protocol_pod_constructions[key]
    assert geometry["axis_sample_sources"] == ["PO", "PS"]
    axis_start, axis_end = geometry["axis"]
    assert axis_end[0] > axis_start[0] + 20.0
    assert geometry["midpoint"][0] > 256.0


def test_protocol_ac_circle_and_fac_ellipse_generate_protocol_derived_points(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-C-AC", "L-C-FAC", "L-AC", "L-DAB", "L-SAB"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.landmark_found = {
        "L-AC": FakeVar(False),
        "L-DAB": FakeVar(False),
        "L-SAB": FakeVar(False),
    }
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"L-C-AC": (50.0, 50.0)}
    gui.hover_radii[key] = {"L-C-AC": 25.0}

    gui._set_ellipse(
        "L-C-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[25, 50], [75, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )
    gui._update_protocol_derived_features_for_current_image()

    assert gui.annotations[key]["L-AC"] == (50.0, 50.0)
    assert gui.annotations[key]["L-DAB"] == (25.0, 50.0)
    assert gui.annotations[key]["L-SAB"] == (75.0, 50.0)
    assert gui.landmark_found["L-AC"].get() is True
    assert gui.landmark_found["L-DAB"].get() is True
    assert gui.landmark_found["L-SAB"].get() is True


def test_protocol_right_acetabulum_uses_image_left_as_lateral(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["R-C-AC", "R-C-FAC", "R-AC", "R-DAB", "R-SAB"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.landmark_found = {
        "R-AC": FakeVar(False),
        "R-DAB": FakeVar(False),
        "R-SAB": FakeVar(False),
    }
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"R-C-AC": (150.0, 50.0)}
    gui.hover_radii[key] = {"R-C-AC": 25.0}

    gui._set_ellipse(
        "R-C-FAC",
        {
            "type": "ellipse",
            "center": [150, 50],
            "major": [[125, 50], [175, 50]],
            "minor": [[150, 40], [150, 60]],
        },
    )
    gui._update_protocol_derived_features_for_current_image()

    assert gui.annotations[key]["R-AC"] == (150.0, 50.0)
    assert gui.annotations[key]["R-DAB"] == (175.0, 50.0)
    assert gui.annotations[key]["R-SAB"] == (125.0, 50.0)
    assert gui.landmark_found["R-AC"].get() is True
    assert gui.landmark_found["R-DAB"].get() is True
    assert gui.landmark_found["R-SAB"].get() is True


def test_protocol_derived_canvas_colors_follow_category_status(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP"]
    key = gui._current_categorical_key()

    assert gui._canvas_colors_for_landmark(
        "L-SIP",
        selected=False,
        flagged=False,
        image_verified=True,
    ) == ("black", "black")
    assert gui._panel_label_color_for_landmark("L-C-ILI") == "#007C89"

    gui.categorical_annotations[key] = {
        "hip_type": {"left": "", "right": ""},
        "derived_features": {"L-SIP": "correct"},
    }
    assert gui._canvas_colors_for_landmark(
        "L-SIP",
        selected=False,
        flagged=False,
        image_verified=False,
    ) == (PROTOCOL_DERIVED_CLASSIFIED_COLOR, PROTOCOL_DERIVED_CLASSIFIED_COLOR)
    assert gui._landmark_should_render_on_canvas("L-SIP") is True

    gui.categorical_annotations[key]["derived_features"]["L-SIP"] = "incorrect"
    assert gui._canvas_colors_for_landmark(
        "L-SIP",
        selected=False,
        flagged=False,
        image_verified=False,
    ) == (PROTOCOL_DERIVED_UNCLASSIFIED_COLOR, PROTOCOL_DERIVED_UNCLASSIFIED_COLOR)
    assert gui._landmark_should_render_on_canvas("L-SIP") is True

    gui.categorical_annotations[key]["derived_features"]["L-SIP"] = "missing"
    assert gui._landmark_should_render_on_canvas("L-SIP") is False


def test_categorical_annotations_round_trip(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._current_categorical_key()
    parsed = gui._parse_categorical_annotations_for_record(
        {
            "categories": {
                "hip_type": {"left": "native", "right": "prosthetic"},
                "derived_features": {
                    "L-SIP": "correct",
                    "L-LIP": "missing",
                    "bad": "correct",
                },
            }
        }
    )
    gui.categorical_annotations[key] = parsed

    assert parsed["hip_type"] == {"left": "native", "right": "prosthetic"}
    assert parsed["derived_features"]["L-SIP"] == "correct"

    data = gui._prepare_categorical_data()

    assert data == {
        "hip_type": {"left": "native", "right": "prosthetic"},
        "derived_features": {"L-SIP": "correct", "L-LIP": "missing"},
    }


def test_direct_annotation_status_round_trip_for_missing_landmark(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    record = {
        "annotations": {
            "RKNEE": {
                "value": None,
                "status": "not applicable",
                "flag": False,
                "note": "not present",
            },
            "L-FA": {
                "value": [[1.0, 2.0], [3.0, 4.0]],
                "status": "missing",
                "flag": False,
                "note": "",
            },
        }
    }

    pts = gui._parse_annotations_for_record(record)
    gui.annotations[key] = pts

    assert "RKNEE" not in pts
    assert gui.landmark_meta[key]["RKNEE"]["status"] == "not_applicable"
    assert gui._get_landmark_status("RKNEE") == "not_applicable"
    assert gui._get_landmark_status("L-FA") == ""

    data = gui._prepare_landmark_data(for_json=True)

    assert data["RKNEE"] == {
        "value": None,
        "flag": False,
        "note": "not present",
        "status": "not_applicable",
    }
    assert "status" not in data["L-FA"]


def test_primary_landmarking_requires_left_and_right_hip_categories(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": ""},
        "derived_features": {},
    }

    assert gui._landmark_is_selectable("L-C-ILI") is False

    gui.categorical_annotations[key]["hip_type"]["right"] = "native"

    assert gui._landmark_is_selectable("L-C-ILI") is True


def test_section_4_order_locks_future_direct_annotations_until_current_is_done(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI", "L-C-PS", "L-DSI"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.annotations[key] = {}
    gui.landmark_meta[key] = {}
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {},
    }

    assert gui._landmark_is_selectable("L-C-ILI") is True
    assert gui._landmark_is_selectable("L-C-PS") is False
    assert gui._landmark_is_selectable("L-DSI") is False

    gui._set_landmark_status("L-C-ILI", "not applicable")

    assert gui._landmark_is_selectable("L-C-ILI") is True
    assert gui._landmark_is_selectable("L-C-PS") is True
    assert gui._landmark_is_selectable("L-DSI") is False

    gui.annotations[key]["L-C-PS"] = (1.0, 2.0)
    gui.it_edge_border_masks[key] = {
        "L-C-PS": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8)
    }
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {"L-SPS": "correct", "L-IPS": "missing"},
    }

    assert gui._landmark_is_selectable("L-DSI") is True


def test_section_4_order_requires_categorical_statuses_for_drawn_contour(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP", "L-C-PS"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.annotations[key] = {"L-C-ILI": (1.0, 2.0)}
    gui.landmark_meta[key] = {}
    gui.it_edge_border_masks[key] = {
        "L-C-ILI": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8)
    }
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {},
    }

    assert gui._landmark_is_selectable("L-C-PS") is False

    gui.categorical_annotations[key]["derived_features"]["L-SIP"] = "correct"
    assert gui._landmark_is_selectable("L-C-PS") is False

    gui.categorical_annotations[key]["derived_features"]["L-LIP"] = "missing"
    assert gui._landmark_is_selectable("L-C-PS") is True


def test_categorical_correct_count_summarizes_primary_and_derived_rows(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "", "right": ""},
        "derived_features": {"L-SIP": "correct", "L-LIP": "missing"},
    }

    assert gui._categorical_correct_count_for_landmark("L-C-ILI") == (1, 2)
    assert gui._categorical_correct_text_for_landmark("L-C-ILI") == "1/2"
    assert gui._categorical_correct_count_for_landmark("L-SIP") == (1, 1)
    assert gui._categorical_correct_count_for_landmark("L-LIP") == (0, 1)


def test_landmark_panel_groups_derived_rows_under_primary_sources(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP", "L-C-PS", "L-SPS", "L-IPS"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")

    groups, orphans = gui._landmark_panel_groups(gui.landmarks)

    assert groups == [
        ("L-C-ILI", ["L-SIP", "L-LIP"]),
        ("L-C-PS", ["L-SPS", "L-IPS"]),
    ]
    assert orphans == []


def test_multi_source_derived_feature_belongs_to_last_source_in_order(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = [
        "L-C-GT",
        "L-SGT",
        "L-LGT",
        "L-C-DMF",
        "L-D-FA",
    ]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.it_edge_border_masks[key] = {
        "L-C-GT": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8),
        "L-C-DMF": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8),
    }
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {"L-SGT": "correct", "L-LGT": "correct"},
    }

    assert gui._relevant_derived_features_for_landmark("L-C-GT") == [
        "L-SGT",
        "L-LGT",
    ]
    assert gui._relevant_derived_features_for_landmark("L-C-DMF") == ["L-D-FA"]

    groups, orphans = gui._landmark_panel_groups(gui.landmarks)

    assert groups == [
        ("L-C-GT", ["L-SGT", "L-LGT"]),
        ("L-C-DMF", ["L-D-FA"]),
    ]
    assert orphans == []
    assert gui._direct_landmark_complete("L-C-GT") is True
    assert gui._direct_landmark_complete("L-C-DMF") is False

    gui.categorical_annotations[key]["derived_features"]["L-D-FA"] = "correct"

    assert gui._direct_landmark_complete("L-C-DMF") is True


def test_chained_multi_source_derived_feature_belongs_to_last_source_in_order(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-PO", "L-LPO", "R-C-PO", "R-LPO", "POD"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")

    assert gui._direct_sources_for_derived_feature("POD") == {
        "L-C-PO",
        "R-C-PO",
    }
    assert gui._derived_feature_owner_landmark("POD") == "R-C-PO"
    assert gui._relevant_derived_features_for_landmark("L-C-PO") == ["L-LPO"]
    assert gui._relevant_derived_features_for_landmark("R-C-PO") == [
        "R-LPO",
        "POD",
    ]

    groups, orphans = gui._landmark_panel_groups(gui.landmarks)

    assert groups == [
        ("L-C-PO", ["L-LPO"]),
        ("R-C-PO", ["R-LPO", "POD"]),
    ]
    assert orphans == []


def test_annotated_state_ignores_landmark_status_without_drawn_pixels(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.annotations[key] = {}
    gui.landmark_meta[key] = {"L-C-ILI": {"status": "missing"}}
    gui.landmark_found = {"L-C-ILI": FakeVar(False)}

    gui._update_found_checks({})

    assert gui.landmark_found["L-C-ILI"].get() is False

    gui.it_edge_border_masks[key] = {
        "L-C-ILI": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8)
    }
    gui._update_found_checks({})

    assert gui.landmark_found["L-C-ILI"].get() is True


def test_update_found_checks_refreshes_status_and_categorical_controls(tmp_path):
    gui = make_gui_stub(tmp_path)
    calls = []
    gui.landmarks = ["L-C-ILI"]
    gui.landmark_found = {"L-C-ILI": FakeVar(False)}
    gui._refresh_landmark_selectability = lambda: calls.append("selectability")
    gui._refresh_landmark_status_control = lambda: calls.append("status")
    gui._refresh_categorical_controls = lambda: calls.append("categorical")
    gui._refresh_landmark_categorical_count_labels = lambda: calls.append("labels")

    gui._update_found_checks({})

    assert "status" in calls
    assert "categorical" in calls


def test_landmark_group_green_status_requires_primary_and_all_derived(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI", "L-SIP", "L-LIP"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_group_members = {"L-C-ILI": ["L-C-ILI", "L-SIP", "L-LIP"]}
    gui.annotations[key] = {}
    gui.landmark_meta[key] = {}
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {"L-SIP": "correct", "L-LIP": "missing"},
    }

    assert gui._landmark_group_has_status("L-C-ILI") is False

    gui.landmark_meta[key] = {"L-C-ILI": {"status": "missing"}}
    assert gui._landmark_group_has_status("L-C-ILI") is True

    gui.categorical_annotations[key]["derived_features"].pop("L-LIP")
    assert gui._landmark_group_has_status("L-C-ILI") is False

    gui.categorical_annotations[key]["derived_features"]["L-LIP"] = "incorrect"
    assert gui._landmark_group_has_status("L-C-ILI") is True


def test_landmark_group_green_status_allows_primary_without_derived(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-DSI"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_group_members = {"L-DSI": ["L-DSI"]}
    gui.landmark_meta[key] = {"L-DSI": {"status": "not_applicable"}}

    assert gui._landmark_group_has_status("L-DSI") is True


def test_primary_landmark_status_text_and_color_follow_annotation_or_status(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-C-ILI"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.annotations[key] = {}
    gui.landmark_meta[key] = {}

    assert gui._primary_landmark_status_text("L-C-ILI") == "-"

    gui.landmark_meta[key] = {"L-C-ILI": {"status": "missing"}}
    assert gui._primary_landmark_status_text("L-C-ILI") == "M"
    assert gui._primary_landmark_status_color("L-C-ILI") == PROTOCOL_STATUS_MISSING_COLOR

    gui.landmark_meta[key] = {"L-C-ILI": {"status": "not_applicable"}}
    assert gui._primary_landmark_status_text("L-C-ILI") == "N/A"
    assert (
        gui._primary_landmark_status_color("L-C-ILI")
        == PROTOCOL_STATUS_NOT_APPLICABLE_COLOR
    )

    gui.it_edge_border_masks[key] = {
        "L-C-ILI": np.ones(MANUAL_SEGMENTATION_MASK_SHAPE, dtype=np.uint8)
    }
    assert gui._primary_landmark_status_text("L-C-ILI") == "A"
    assert gui._primary_landmark_status_color("L-C-ILI") == PROTOCOL_DERIVED_CLASSIFIED_COLOR


def test_direct_annotation_status_counts_as_image_progress(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.images = [gui.current_image_path]
    gui.image_index_map = {key: 0}
    gui.allowed_views = {"ap": ["RKNEE", "L-FA"]}
    gui.json_data = {
        "landmarks": gui.landmarks,
        "views": gui.allowed_views,
        "images": [{"image_path": "image.png", "view": "ap", "annotations": {}}],
    }
    gui.annotations[key] = {"L-FA": [(0.0, 0.0), (1.0, 1.0)]}
    gui.landmark_meta[key] = {
        "RKNEE": {"flag": False, "note": "", "status": "missing"}
    }

    assert gui._count_completed_landmarks_for_current_image(gui.current_image_path) == 2
    assert gui._image_progress_text(gui.current_image_path) == "2/2"
    assert gui._image_progress_done(gui.current_image_path) is True


def test_zoom_mask_rect_maps_image_space_to_512_mask_space(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(1000, 500))
    mask = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )

    rect = gui._image_rect_to_mask_rect((250.0, 125.0, 750.0, 375.0), mask)

    assert rect is not None
    assert np.allclose(
        rect,
        (
            250.0 * 511.0 / 999.0,
            125.0 * 511.0 / 499.0,
            750.0 * 511.0 / 999.0,
            375.0 * 511.0 / 499.0,
        ),
    )


def test_it_edge_border_line_round_trips_as_binary_mask_annotation(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    gui.current_image = SimpleNamespace(size=(20, 20))
    key = gui._segmentation_mask_key()
    border = np.zeros(
        (MANUAL_SEGMENTATION_MASK_SIZE, MANUAL_SEGMENTATION_MASK_SIZE),
        dtype=np.uint8,
    )
    border[256, 128:384] = 1
    gui.it_edge_border_masks[key] = {"L-E-IT": border}

    data = gui._prepare_landmark_data()

    assert data["L-E-IT"]["value"] == [9.5, 9.5]
    assert np.array_equal(
        gui._decode_segmentation_mask(data["L-E-IT"]["segmentation"]),
        border,
    )
    assert np.array_equal(
        gui._decode_segmentation_mask(data["L-E-IT"]["edge_border"]),
        border,
    )

    gui.seg_masks = {}
    gui.it_edge_border_masks = {}
    parsed = gui._parse_annotations_for_record({"annotations": data})

    assert parsed == {"L-E-IT": (9.5, 9.5)}
    assert np.array_equal(gui.seg_masks[key]["L-E-IT"], border)
    assert np.array_equal(gui.it_edge_border_masks[key]["L-E-IT"], border)


def test_it_edge_trace_drag_can_defer_expensive_refreshes(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    gui.current_image = SimpleNamespace(size=(20, 20))
    gui.landmark_found = {"L-E-IT": FakeVar(False)}
    gui.it_edge_border_brush_size.set(1)
    gui.dirty = False
    calls = []
    gui._update_it_edge_border_overlay_for = lambda _lm: calls.append("overlay")
    gui._update_found_checks = lambda _pts: calls.append("checks")
    gui._refresh_image_listbox = lambda: calls.append("list")

    gui._draw_it_edge_border_segment(
        "L-E-IT",
        (2.0, 2.0),
        (10.0, 2.0),
        refresh_overlay=False,
        sync_annotation=False,
        refresh_controls=False,
    )

    key = gui._segmentation_mask_key()
    assert calls == []
    assert gui.dirty is True
    assert "L-E-IT" not in gui.annotations.get(key, {})
    assert int(gui.it_edge_border_masks[key]["L-E-IT"].sum()) > 0


def test_contour_highlighter_draw_mode_forces_single_grid_pixel_brush(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI"]
    gui.current_image = SimpleNamespace(size=(512, 512))
    gui.it_edge_border_mode = FakeVar("draw")
    gui.it_edge_border_brush_size = FakeVar(21)

    gui._draw_it_edge_border_segment(
        "L-C-ILI",
        (10.0, 10.0),
        (20.0, 10.0),
        refresh_overlay=False,
        sync_annotation=False,
        refresh_controls=False,
    )

    key = gui._segmentation_mask_key()
    ys, _xs = np.nonzero(gui.it_edge_border_masks[key]["L-C-ILI"] > 0)
    assert set(ys.tolist()) == {10}
    assert gui.it_edge_border_brush_size.get() == 1


def test_open_it_edge_trace_autosaves_on_mouse_release(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-E-IT"]
    gui.current_image = SimpleNamespace(size=(20, 20))
    gui.landmark_visibility = {"L-E-IT": FakeVar(True)}
    gui.landmark_found = {"L-E-IT": FakeVar(False)}
    gui.selected_landmark.set("L-E-IT")
    gui.it_edge_border_brush_size.set(1)
    gui.drawing_it_edge_border = True
    gui.it_edge_border_last_img_pos = (2.0, 2.0)
    gui.dirty = False
    calls = []
    gui._display_rect = lambda: (0, 0, 20, 20)
    gui._screen_to_img = lambda x, y: (x, y)
    gui._update_mouse_crosshair = lambda *_args: None
    gui._update_zoom_view = lambda *_args: None
    gui._update_line_preview = lambda *_args: None
    gui._update_overlay_for = lambda _lm: None
    gui._update_it_edge_border_overlay_for = lambda _lm: None
    gui._update_found_checks = lambda _pts: None
    gui._refresh_image_listbox = lambda: None
    gui._draw_points = lambda: calls.append("draw")
    gui._set_status = lambda message: setattr(gui, "status", message)
    gui._autosave_annotation_change = lambda: calls.append("save") or True

    gui._on_left_release(SimpleNamespace(x=10, y=2))

    key = gui._segmentation_mask_key()
    assert calls == ["draw", "save"]
    assert gui.annotations[key]["L-E-IT"] == (6.0, 2.0)
    assert "L-E-IT" not in gui.seg_masks.get(key, {})
    assert gui.it_edge_border_masks[key]["L-E-IT"].shape == (
        MANUAL_SEGMENTATION_MASK_SIZE,
        MANUAL_SEGMENTATION_MASK_SIZE,
    )
    assert int(gui.it_edge_border_masks[key]["L-E-IT"].sum()) > 0


def test_segmentation_display_prefers_combined_filled_mask_for_e_it(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._segmentation_mask_key()
    border = np.zeros((4, 4), dtype=np.uint8)
    border[0, :] = 1
    filled = np.ones((4, 4), dtype=np.uint8)
    gui.it_edge_border_masks[key] = {"L-E-IT": border}
    gui.seg_masks[key] = {"L-E-IT": filled}

    assert np.array_equal(gui._segmentation_mask_for_display("L-E-IT"), filled)


def test_snapped_fac_ellipse_defines_left_and_right_acetabular_border_points(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = [
        "L-AC",
        "R-AC",
        "L-FAC",
        "R-FAC",
        "L-A-DAB",
        "L-A-SAB",
        "R-A-DAB",
        "R-A-SAB",
    ]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.landmark_found = {
        "L-A-DAB": FakeVar(False),
        "L-A-SAB": FakeVar(False),
        "R-A-DAB": FakeVar(False),
        "R-A-SAB": FakeVar(False),
    }
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"L-AC": (50.0, 50.0), "R-AC": (150.0, 50.0)}
    gui.hover_radii[key] = {"L-AC": 25.0, "R-AC": 25.0}

    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[25, 50], [75, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )
    gui._set_ellipse(
        "R-FAC",
        {
            "type": "ellipse",
            "center": [150, 50],
            "major": [[125, 50], [175, 50]],
            "minor": [[150, 40], [150, 60]],
        },
    )

    assert gui.annotations[key]["L-A-DAB"] == (25.0, 50.0)
    assert gui.annotations[key]["L-A-SAB"] == (75.0, 50.0)
    assert gui.annotations[key]["R-A-DAB"] == (175.0, 50.0)
    assert gui.annotations[key]["R-A-SAB"] == (125.0, 50.0)
    assert gui.landmark_found["L-A-DAB"].get() is True
    assert gui.landmark_found["R-A-SAB"].get() is True

    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[20, 50], [90, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )

    assert "L-A-DAB" not in gui.annotations[key]
    assert "L-A-SAB" not in gui.annotations[key]


def test_loaded_snapped_fac_geometry_recomputes_derived_acetabular_points(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {
        "L-A-DAB": FakeVar(False),
        "L-A-SAB": FakeVar(False),
    }
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-AC": (50.0, 50.0),
        "L-FAC": {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[25.0, 50.0], [75.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        },
    }
    gui.hover_radii[key] = {"L-AC": 25.0}

    assert (
        gui._update_auto_acetabular_points_for_current_image(snap_if_enabled=False)
        is True
    )

    assert gui.annotations[key]["L-A-DAB"] == (25.0, 50.0)
    assert gui.annotations[key]["L-A-SAB"] == (75.0, 50.0)
    assert gui.landmark_found["L-A-DAB"].get() is True
    assert gui.landmark_found["L-A-SAB"].get() is True


def test_intermediate_acetabular_points_require_snapped_fac_geometry(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB", "L-A-IDAB", "L-A-ISAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {}
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-AC": (50.0, 50.0),
        "L-FAC": {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[20.0, 50.0], [90.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        },
    }
    gui.hover_radii[key] = {"L-AC": 25.0}

    assert gui._intermediate_acetabular_landmark_ready("L-A-ISAB") is False
    assert gui._landmark_is_selectable("L-A-ISAB") is False

    assert (
        gui._update_auto_acetabular_points_for_current_image(snap_if_enabled=True)
        is True
    )

    assert gui._intermediate_acetabular_landmark_ready("L-A-ISAB") is True
    assert gui._landmark_is_selectable("L-A-ISAB") is True


def test_intermediate_acetabular_points_project_to_lower_arc_and_keep_order(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB", "L-A-IDAB", "L-A-ISAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {
        "L-A-IDAB": FakeVar(False),
        "L-A-ISAB": FakeVar(False),
    }
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-AC": (50.0, 50.0),
        "L-FAC": {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[25.0, 50.0], [75.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        },
    }
    gui.hover_radii[key] = {"L-AC": 25.0}
    gui._update_auto_acetabular_points_for_current_image()

    assert gui._set_intermediate_acetabular_point("L-A-ISAB", (35.0, 90.0)) is True
    assert gui._set_intermediate_acetabular_point("L-A-IDAB", (30.0, 90.0)) is True

    isab = gui.annotations[key]["L-A-ISAB"]
    idab = gui.annotations[key]["L-A-IDAB"]
    assert isab[1] > 50.0
    assert idab[1] > 50.0
    isab_ratio = gui._lower_arc_ratio_for_point("L-FAC", isab)
    idab_ratio = gui._lower_arc_ratio_for_point("L-FAC", idab)
    assert isab_ratio is not None
    assert idab_ratio is not None
    assert idab_ratio > isab_ratio
    assert gui.landmark_found["L-A-IDAB"].get() is True


def test_acetabular_lower_arc_highlight_draws_sab_and_dab_segments(tmp_path):
    class FakeCanvas:
        def __init__(self):
            self.lines = []
            self.raised = []

        def create_line(self, *coords, **kwargs):
            self.lines.append((coords, kwargs))
            return len(self.lines)

        def tag_raise(self, item_id):
            self.raised.append(item_id)

    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB", "L-A-IDAB", "L-A-ISAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-AC": (50.0, 50.0),
        "L-FAC": {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[25.0, 50.0], [75.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        },
    }
    gui.hover_radii[key] = {"L-AC": 25.0}
    gui._update_auto_acetabular_points_for_current_image()
    gui._set_intermediate_acetabular_point("L-A-ISAB", (35.0, 90.0))
    gui._set_intermediate_acetabular_point("L-A-IDAB", (65.0, 90.0))
    canvas = FakeCanvas()

    item_ids = gui._draw_acetabular_lower_arc_highlights(
        canvas,
        "L-FAC",
        lambda x, y: (x, y),
        width=2,
    )

    assert item_ids == [1, 2]
    assert len(canvas.lines) == 2
    assert all(
        kwargs["fill"] == "#FF4DA6" and kwargs["width"] == 4
        for _coords, kwargs in canvas.lines
    )


def test_projected_hemisphere_creates_ac_and_acetabular_points(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB", "L-A-IDAB", "L-A-ISAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    key = gui._path_key(gui.current_image_path)

    assert gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0)) is True

    pts = gui.annotations[key]
    assert pts["L-FAC"]["construction"] == "projected_hemisphere"
    assert pts["L-AC"] == (100.0, 100.0)
    assert gui.hover_radii[key]["L-AC"] == 70.0
    assert pts["L-A-DAB"] == (30.0, 100.0)
    assert pts["L-A-SAB"] == (170.0, 100.0)
    assert gui._intermediate_acetabular_landmark_ready("L-A-ISAB") is True

    assert gui._set_intermediate_acetabular_point("L-A-ISAB", (60.0, 140.0)) is True
    assert gui._set_intermediate_acetabular_point("L-A-IDAB", (65.0, 140.0)) is True
    isab_ratio = gui._lower_arc_ratio_for_point("L-FAC", pts["L-A-ISAB"])
    idab_ratio = gui._lower_arc_ratio_for_point("L-FAC", pts["L-A-IDAB"])
    assert isab_ratio is not None
    assert idab_ratio is not None
    assert idab_ratio > isab_ratio
    assert gui.landmark_found["L-A-IDAB"].get() is True


def test_projected_hemisphere_uses_z_control_for_new_size(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui.projected_hemisphere_z = FakeVar(90)
    key = gui._path_key(gui.current_image_path)

    assert gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0)) is True

    assert gui.hover_radii[key]["L-AC"] == 90.0
    assert gui.annotations[key]["L-A-SAB"] == (190.0, 100.0)
    assert gui.annotations[key]["L-A-DAB"] == (10.0, 100.0)


def test_projected_hemisphere_can_start_from_blank_with_no_selected_landmark(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    key = gui._path_key(gui.current_image_path)
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {},
    }
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui.fac_construction_mode = FakeVar("projected_hemisphere")
    gui.selected_landmark = FakeVar("")
    gui._display_rect = lambda: (0, 0, 240, 240)
    gui._screen_to_img = lambda x, y: (x, y)
    gui._update_mouse_crosshair = lambda x, y: None
    gui._update_zoom_view = lambda x, y: None
    gui._draw_points = lambda: None
    gui._refresh_zoom_landmark_overlay = lambda: None
    gui._auto_save_to_db = lambda: None
    gui._set_status = lambda message: None

    gui._on_left_press(SimpleNamespace(x=110, y=115))

    key = gui._path_key(gui.current_image_path)
    assert gui.selected_landmark.get() == "L-FAC"
    assert gui.annotations[key]["L-FAC"]["construction"] == "projected_hemisphere"
    assert gui.annotations[key]["L-AC"] == (110.0, 115.0)
    assert gui.dragging_landmark == "L-FAC"
    assert gui.dragging_ellipse_whole is True


def test_projected_hemisphere_z_wheel_resizes_existing_projection(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui.fac_construction_mode = FakeVar("projected_hemisphere")
    gui.selected_landmark = FakeVar("L-FAC")
    gui._draw_points = lambda: None
    gui._refresh_zoom_landmark_overlay = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0))

    assert gui._change_projected_hemisphere_size(10) is True

    key = gui._path_key(gui.current_image_path)
    assert gui.projected_hemisphere_z.get() == 80
    assert gui.hover_radii[key]["L-AC"] == 80.0
    assert gui.annotations[key]["L-A-SAB"] == (180.0, 100.0)
    assert gui.annotations[key]["L-A-DAB"] == (20.0, 100.0)


def test_projected_hemisphere_major_drag_changes_inclination_not_z(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0))

    gui._move_ellipse_handle("L-FAC", "major_end", (100.0, 170.0))

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    center, major0, major1, _minor0, _minor1 = axes
    radius = ((major1[0] - major0[0]) ** 2 + (major1[1] - major0[1]) ** 2) ** 0.5 / 2
    assert center == (100.0, 100.0)
    assert abs(radius - 70.0) < 0.1
    assert major0 == (100.0, 30.0)
    assert major1 == (100.0, 170.0)


def test_projected_hemisphere_minor_drag_changes_anteversion_not_inclination(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0))

    gui._move_ellipse_handle("L-FAC", "minor_end", (100.0, 160.0))

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    _center, major0, major1, minor0, minor1 = axes
    assert major0 == (30.0, 100.0)
    assert major1 == (170.0, 100.0)
    assert minor0 == (100.0, 40.0)
    assert minor1 == (100.0, 160.0)


def test_number_shortcuts_choose_categories_not_projection_or_quality(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(240, 240))
    gui.landmarks = ["L-AC", "L-FAC", "L-A-DAB", "L-A-SAB"]
    gui.allowed_views = {"AP Unilateral (Left)": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Unilateral (Left)")
    gui.landmark_found = {lm: FakeVar(False) for lm in gui.landmarks}
    gui.fac_construction_mode = FakeVar("projected_hemisphere")
    gui.selected_landmark = FakeVar("L-FAC")
    gui.quality_var = FakeVar("0")
    gui.current_image_quality = 0
    gui._maybe_autosave_current_image = lambda: None
    gui._build_landmark_panel = lambda: None
    gui._refresh_categorical_controls = lambda: None
    gui._draw_points = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._set_status = lambda message: setattr(gui, "last_status", message)
    gui._set_projected_hemisphere_from_center("L-FAC", (100.0, 100.0))

    before = gui._fac_angle_measurements(gui._get_ellipse("L-FAC"))
    gui._on_1_press(None)
    after_left_category = gui._fac_angle_measurements(gui._get_ellipse("L-FAC"))
    gui._on_2_press(None)
    after_right_category = gui._fac_angle_measurements(gui._get_ellipse("L-FAC"))
    categories = gui._current_categorical_annotations()

    assert before is not None
    assert after_left_category == before
    assert after_right_category == before
    assert categories["hip_type"] == {"left": "native", "right": "prosthetic"}
    assert gui.current_image_quality == 0


def test_number_shortcuts_choose_primary_landmark_status(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.selected_landmark = FakeVar("L-C-ILI")
    key = gui._current_categorical_key()
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {},
    }
    gui._refresh_categorical_controls = lambda: None
    gui._refresh_landmark_selectability = lambda: None
    gui._refresh_landmark_label_styles = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._maybe_autosave_current_image = lambda: None
    gui._advance_to_next_protocol_landmark_if_current_complete = lambda lm: None
    gui._set_status = lambda message: setattr(gui, "last_status", message)

    prompt = gui._active_categorical_shortcut_prompt()
    assert prompt is not None
    assert prompt["kind"] == "landmark_status"
    assert prompt["label"] == "L-C-ILI Landmark Status"
    assert prompt["options"] == [
        ("missing", "Missing"),
        ("not_applicable", "Not Applicable"),
    ]

    assert gui._apply_categorical_shortcut(2) is True
    assert gui._get_landmark_status("L-C-ILI") == "not_applicable"


def test_number_shortcuts_choose_derived_feature_status(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-C-ILI", "L-SIP"]
    gui.allowed_views = {"AP Bilateral": list(gui.landmarks)}
    gui.current_view_var = FakeVar("AP Bilateral")
    gui.selected_landmark = FakeVar("L-C-ILI")
    key = gui._current_categorical_key()
    gui.categorical_annotations[key] = {
        "hip_type": {"left": "native", "right": "native"},
        "derived_features": {},
    }
    gui._landmark_annotation_geometry_complete = lambda lm: lm == "L-C-ILI"
    gui._landmark_has_annotation_value = lambda lm: lm == "L-C-ILI"
    gui._refresh_categorical_controls = lambda: None
    gui._refresh_landmark_selectability = lambda: None
    gui._refresh_landmark_label_styles = lambda: None
    gui._draw_points = lambda: None
    gui._refresh_zoom_landmark_overlay = lambda: None
    gui._refresh_image_listbox = lambda: None
    gui._maybe_autosave_current_image = lambda: None
    gui._advance_to_next_protocol_landmark_if_current_complete = lambda lm: None
    gui._set_status = lambda message: setattr(gui, "last_status", message)

    prompt = gui._active_categorical_shortcut_prompt()
    assert prompt is not None
    assert prompt["label"] == "L-SIP"

    assert gui._apply_categorical_shortcut(2) is True
    assert gui.categorical_annotations[key]["derived_features"] == {
        "L-SIP": "incorrect"
    }


def test_projected_hemisphere_serializes_construction_method(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.landmarks = ["L-FAC"]
    gui.annotations[key] = {
        "L-FAC": {
            "type": "ellipse",
            "construction": "projected_hemisphere",
            "center": [50.0, 50.0],
            "major": [[25.0, 50.0], [75.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        }
    }

    data = gui._prepare_landmark_data()

    assert data["L-FAC"]["value"]["construction"] == "projected_hemisphere"


def test_projected_hemisphere_boundary_draws_visible_and_missing_arcs(tmp_path):
    class FakeCanvas:
        def __init__(self):
            self.lines = []
            self.raised = []

        def create_line(self, *coords, **kwargs):
            self.lines.append((coords, kwargs))
            return len(self.lines)

        def tag_raise(self, item_id):
            self.raised.append(item_id)

    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.landmarks = ["L-FAC"]
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-FAC": {
            "type": "ellipse",
            "construction": "projected_hemisphere",
            "center": [50.0, 50.0],
            "major": [[25.0, 50.0], [75.0, 50.0]],
            "minor": [[50.0, 40.0], [50.0, 60.0]],
        }
    }
    canvas = FakeCanvas()

    item_ids = gui._draw_projected_hemisphere_boundary(
        canvas,
        "L-FAC",
        lambda x, y: (x, y),
        width=2,
    )

    assert item_ids == [1, 2]
    assert len(canvas.lines) == 2
    assert "dash" not in canvas.lines[0][1]
    assert canvas.lines[1][1]["dash"] == (5, 4)
    assert all(kwargs["fill"] == "orange" for _coords, kwargs in canvas.lines)
    solid_y_values = canvas.lines[0][0][1::2]
    dashed_y_values = canvas.lines[1][0][1::2]
    assert sum(solid_y_values) / len(solid_y_values) < 50.0
    assert sum(dashed_y_values) / len(dashed_y_values) > 50.0


def test_parse_annotations_for_record_extracts_meta_and_segmentation_settings(tmp_path):
    gui = make_gui_stub(tmp_path)

    record = {
        "annotations": {
            "L-FA": {"value": [[1, 2], [3, 4]], "flag": True, "note": "shaft"},
            "LOB": {"value": [10, 20, "ACC", 7, 1, 5, 0, 2], "flag": False, "note": ""},
            "RKNEE": [30, 40],
        }
    }

    parsed = gui._parse_annotations_for_record(record)

    assert parsed == {
        "L-FA": [(1.0, 2.0), (3.0, 4.0)],
        "LOB": (10.0, 20.0),
        "RKNEE": (30.0, 40.0),
    }
    key = gui._path_key(gui.current_image_path)
    assert gui.landmark_meta[key]["L-FA"] == {"flag": True, "note": "shaft"}
    assert gui.landmark_meta[key]["RKNEE"] == {"flag": False, "note": ""}
    assert gui.lm_settings[key]["LOB"] == {
        "method": "Adaptive CC",
        "sens": 7,
        "edge_lock": 1,
        "edge_width": 5,
        "clahe": 0,
        "grow": 2,
    }


def test_prepare_landmark_data_includes_meta_and_line_landmarks(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-FA": [(1.0, 2.0), (3.0, 4.0)],
        "LOB": (10.0, 20.0),
    }
    gui.lm_settings[key] = {
        "LOB": {
            "method": "Flood Fill",
            "sens": 9,
            "edge_lock": 1,
            "edge_width": 3,
            "clahe": 1,
            "grow": 4,
        }
    }
    gui.landmark_meta[key] = {
        "L-FA": {"flag": True, "note": "line note"},
        "RKNEE": {"flag": False, "note": "missing point note"},
    }

    data = gui._prepare_landmark_data()

    assert data == {
        "L-FA": {
            "value": [[1.0, 2.0], [3.0, 4.0]],
            "flag": True,
            "note": "line note",
        },
        "LOB": {
            "value": [10.0, 20.0, "FF", 9, 1, 3, 1, 4],
            "flag": False,
            "note": "",
        },
        "RKNEE": {
            "value": None,
            "flag": False,
            "note": "missing point note",
        },
    }


def test_parse_annotations_for_record_extracts_ellipse_landmark(tmp_path):
    gui = make_gui_stub(tmp_path)

    record = {
        "annotations": {
            "L-FAC": {
                "value": {
                    "type": "ellipse",
                    "center": [10, 20],
                    "major": [[0, 20], [20, 20]],
                    "minor": [[10, 15], [10, 25]],
                },
                "flag": True,
                "note": "cup",
            }
        }
    }

    parsed = gui._parse_annotations_for_record(record)

    assert parsed["L-FAC"] == {
        "type": "ellipse",
        "center": [10.0, 20.0],
        "major": [[0.0, 20.0], [20.0, 20.0]],
        "minor": [[10.0, 15.0], [10.0, 25.0]],
    }
    key = gui._path_key(gui.current_image_path)
    assert gui.landmark_meta[key]["L-FAC"] == {"flag": True, "note": "cup"}


def test_prepare_landmark_data_serializes_ellipse_landmark(tmp_path):
    gui = make_gui_stub(tmp_path)
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {
        "L-FAC": {
            "type": "ellipse",
            "center": [10.0, 20.0],
            "major": [[0.0, 20.0], [20.0, 20.0]],
            "minor": [[10.0, 15.0], [10.0, 25.0]],
        }
    }
    gui.landmark_meta[key] = {
        "L-FAC": {"flag": False, "note": "ellipse note"},
    }

    data = gui._prepare_landmark_data()

    assert data["L-FAC"] == {
        "value": {
            "type": "ellipse",
            "center": [10.0, 20.0],
            "major": [[0.0, 20.0], [20.0, 20.0]],
            "minor": [[10.0, 15.0], [10.0, 25.0]],
        },
        "flag": False,
        "note": "ellipse note",
    }


def test_fac_angle_measurements_use_major_axis_and_axis_ratio(tmp_path):
    gui = make_gui_stub(tmp_path)

    measurements = gui._fac_angle_measurements(
        {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[20.0, 50.0], [80.0, 50.0]],
            "minor": [[50.0, 35.0], [50.0, 65.0]],
        }
    )

    assert measurements is not None
    inclination, anteversion = measurements
    assert abs(inclination - 0.0) < 0.01
    assert abs(anteversion - 30.0) < 0.01


def test_fac_angle_measurements_report_acute_inclination_to_horizontal(tmp_path):
    gui = make_gui_stub(tmp_path)

    measurements = gui._fac_angle_measurements(
        {
            "type": "ellipse",
            "center": [50.0, 50.0],
            "major": [[20.0, 80.0], [80.0, 20.0]],
            "minor": [[35.0, 35.0], [65.0, 65.0]],
        }
    )

    assert measurements is not None
    inclination, _anteversion = measurements
    assert abs(inclination - 45.0) < 0.01


def test_move_ellipse_major_endpoint_keeps_opposite_endpoint_fixed(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.disp_scale = 1.0
    gui.landmark_found = {}
    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[30, 50], [70, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )

    gui._move_ellipse_handle("L-FAC", "major_end", (90, 70))

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    center, major0, major1, minor0, minor1 = axes
    assert center == (60.0, 60.0)
    assert major0 == (30.0, 50.0)
    assert major1 == (90.0, 70.0)

    major_vec = ((major1[0] - major0[0]) / 2, (major1[1] - major0[1]) / 2)
    minor_vec = ((minor1[0] - minor0[0]) / 2, (minor1[1] - minor0[1]) / 2)
    dot = major_vec[0] * minor_vec[0] + major_vec[1] * minor_vec[1]
    assert abs(dot) < 5.0
    assert abs((minor_vec[0] ** 2 + minor_vec[1] ** 2) ** 0.5 - 10.0) < 0.2


def test_snap_ellipse_major_axis_projects_endpoints_to_ac_hover_circle(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.disp_scale = 1.0
    gui.landmark_found = {}
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"L-AC": (50.0, 50.0)}
    gui.hover_radii[key] = {"L-AC": 25.0}
    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [55, 50],
            "major": [[20, 50], [90, 50]],
            "minor": [[55, 40], [55, 60]],
        },
    )

    assert gui._snap_ellipse_major_axis_to_ac_circle("L-FAC") is True

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    _center, major0, major1, _minor0, _minor1 = axes
    assert major0 == (25.0, 50.0)
    assert major1 == (75.0, 50.0)


def test_snap_enabled_constrains_dragged_major_endpoint_to_ac_circle(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.disp_scale = 1.0
    gui.landmark_found = {}
    gui.ellipse_snap_to_ac_enabled = FakeVar(True)
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"L-AC": (50.0, 50.0)}
    gui.hover_radii[key] = {"L-AC": 25.0}
    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[25, 50], [75, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )

    gui._move_ellipse_handle("L-FAC", "major_end", (50, 90))

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    _center, major0, major1, _minor0, _minor1 = axes
    assert major0 == (25.0, 50.0)
    assert major1 == (50.0, 75.0)
    assert abs(((major1[0] - 50.0) ** 2 + (major1[1] - 50.0) ** 2) ** 0.5 - 25.0) < 0.1


def test_lock_major_option_keeps_snapped_major_axis_fixed_when_dragging_minor(
    tmp_path,
):
    gui = make_gui_stub(tmp_path)
    gui.current_image = SimpleNamespace(size=(200, 200))
    gui.disp_scale = 1.0
    gui.landmark_found = {}
    gui.ellipse_lock_major_when_snapped_enabled = FakeVar(True)
    key = gui._path_key(gui.current_image_path)
    gui.annotations[key] = {"L-AC": (50.0, 50.0)}
    gui.hover_radii[key] = {"L-AC": 25.0}
    gui._set_ellipse(
        "L-FAC",
        {
            "type": "ellipse",
            "center": [50, 50],
            "major": [[25, 50], [75, 50]],
            "minor": [[50, 40], [50, 60]],
        },
    )

    gui._move_ellipse_handle("L-FAC", "minor_end", (50, 80))

    ellipse = gui._get_ellipse("L-FAC")
    axes = gui._ellipse_axis_points(ellipse)
    assert axes is not None
    center, major0, major1, minor0, minor1 = axes
    assert center == (50.0, 50.0)
    assert major0 == (25.0, 50.0)
    assert major1 == (75.0, 50.0)
    assert minor0 == (50.0, 20.0)
    assert minor1 == (50.0, 80.0)


def test_current_image_has_unsaved_changes_uses_json_snapshot(tmp_path):
    gui = make_gui_stub(tmp_path)
    image_path = gui.current_image_path
    key = gui._path_key(image_path)
    gui.json_data["images"] = [
        {
            "image_path": "image.png",
            "image_flag": False,
            "view": "ap",
            "annotations": {},
            "resolved_image_path": key,
        }
    ]
    gui.image_index_map[key] = 0

    assert gui._current_image_has_unsaved_changes() is True

    gui.current_image_flag = False
    assert gui._current_image_has_unsaved_changes() is False

    gui.saved_image_snapshots[key] = json.dumps(
        {
            "image_path": "image.png",
            "image_flag": False,
            "view": "ap",
            "annotations": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    gui.current_view_var.set("lat")
    assert gui._current_image_has_unsaved_changes() is True


def test_autosave_mode_saves_transitions_without_prompt(tmp_path, monkeypatch):
    gui = make_gui_stub(tmp_path)
    gui.autosave_var.set(True)
    gui.dirty = True
    saves = []

    def unexpected_prompt(*_args, **_kwargs):
        raise AssertionError("autosave-on transitions should not prompt")

    monkeypatch.setattr(annotator_main.messagebox, "askyesno", unexpected_prompt)
    gui._current_image_has_pending_changes = lambda: True
    gui._save_current_image_silently = lambda: saves.append("save") or True

    assert gui._maybe_save_before_destructive_action("switch images") is True
    assert saves == ["save"]


def test_manual_save_mode_prompts_before_transition(tmp_path, monkeypatch):
    gui = make_gui_stub(tmp_path)
    gui.autosave_var.set(False)
    saves = []

    monkeypatch.setattr(
        annotator_main.messagebox,
        "askyesno",
        lambda _title, _message: True,
    )
    gui._current_image_has_pending_changes = lambda: True
    gui.save_annotations = lambda: saves.append("save") or True

    assert gui._maybe_save_before_destructive_action("switch landmarks") is True
    assert saves == ["save"]


def test_landmark_group_transition_prompts_only_between_primary_groups(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmark_group_frames = {"L-C-DMF": object(), "R-C-DMF": object()}
    gui.landmark_group_members = {
        "L-C-DMF": ["L-C-DMF", "L-D-FA"],
        "R-C-DMF": ["R-C-DMF", "R-D-FA"],
    }
    gui._active_landmark_group = "L-C-DMF"
    calls = []
    gui._maybe_save_before_destructive_action = lambda why: calls.append(why) or True

    assert gui._handle_landmark_group_transition("L-D-FA") is True
    assert gui._active_landmark_group == "L-C-DMF"
    assert calls == []

    assert gui._handle_landmark_group_transition("R-C-DMF") is True
    assert gui._active_landmark_group == "R-C-DMF"
    assert calls == ["switch landmarks"]


def test_multi_review_parses_annotations_and_counts_review_content(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.landmarks = ["L-FA", "RKNEE"]
    gui.multi_review_reviewer_order = ["andrew", "jenna"]

    record = {
        "annotations": {
            "L-FA": {"value": [[1, 2], [3, 4]], "flag": True, "note": "check"},
            "RKNEE": {"value": [10, 20], "flag": False, "note": ""},
        }
    }

    pts, meta = gui._parse_multi_review_record_annotations(record)

    assert pts == {"L-FA": [(1.0, 2.0), (3.0, 4.0)], "RKNEE": (10.0, 20.0)}
    assert meta["L-FA"] == {"flag": True, "note": "check"}

    image_path = tmp_path / "image.png"
    image_key = "fluoro_images_round_1/image.png"
    gui.multi_review_image_key_by_path = {gui._path_key(image_path): image_key}
    gui.multi_review_annotations = {
        image_key: {
            "andrew": pts,
            "jenna": {},
        }
    }
    gui.multi_review_meta = {
        image_key: {
            "andrew": meta,
            "jenna": {"RKNEE": {"flag": True, "note": ""}},
        }
    }

    assert gui._multi_review_counts_for_image_path(image_path) == (2, 2)


def test_multi_review_delete_selected_temp_point(tmp_path):
    gui = make_gui_stub(tmp_path)
    gui.multi_review_mode = True
    gui.selected_landmark = FakeVar("RKNEE")
    image_key = "fluoro_images_round_1/image.png"
    gui.multi_review_image_key_by_path = {
        gui._path_key(gui.current_image_path): image_key
    }
    gui.multi_review_temp_annotations = {
        image_key: {"RKNEE": (2.0, 3.0), "L-FA": [(1.0, 1.0)]}
    }

    calls = []
    gui._clear_line_preview = lambda: calls.append("clear")
    gui._refresh_multi_review_panel = lambda: calls.append("panel")
    gui._draw_points = lambda: calls.append("draw")
    gui._refresh_zoom_landmark_overlay = lambda: calls.append("zoom")

    deleted = gui._delete_selected_multi_review_temp_annotation()

    assert deleted is True
    assert gui.multi_review_temp_annotations[image_key] == {"L-FA": [(1.0, 1.0)]}
    assert calls == ["clear", "panel", "draw", "zoom"]
