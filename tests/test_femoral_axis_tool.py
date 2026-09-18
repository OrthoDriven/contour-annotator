#!/usr/bin/env python3

# pyright: reportMissingImports=false

from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import AnnotationGUI


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeScale:
    def __init__(self):
        self.state = None

    def config(self, *, state):
        self.state = state


class FakeFont:
    def __init__(self, size=12):
        self.size = size

    def copy(self):
        return FakeFont(self.size)

    def configure(self, **kwargs):
        if "size" in kwargs:
            self.size = kwargs["size"]

    def cget(self, key):
        if key == "size":
            return self.size
        return None


class FakeCanvas:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.raised = []
        self.next_id = 1

    def _create(self, *coords, **kwargs):
        item_id = self.next_id
        self.next_id += 1
        self.created.append((item_id, coords, kwargs))
        return item_id

    def create_line(self, *coords, **kwargs):
        return self._create(*coords, **kwargs)

    def create_oval(self, *coords, **kwargs):
        return self._create(*coords, **kwargs)

    def create_rectangle(self, *coords, **kwargs):
        return self._create(*coords, **kwargs)

    def create_text(self, *coords, **kwargs):
        return self._create(*coords, **kwargs)

    def delete(self, item_id):
        self.deleted.append(item_id)

    def tag_raise(self, item):
        self.raised.append(item)

    def winfo_width(self):
        return 800


def make_gui_stub():
    gui = AnnotationGUI.__new__(AnnotationGUI)
    gui.hover_enabled = FakeVar(False)
    gui.femoral_axis_enabled = FakeVar(False)
    gui.selected_landmark = FakeVar("")
    gui.femoral_axis_count = FakeVar(5)
    gui.femoral_axis_proj_length = FakeVar(60)
    gui.femoral_axis_whisker_tip_length = FakeVar(10)
    gui.femoral_axis_item_ids = []
    gui.radius_scale = FakeScale()
    gui.femoral_axis_count_scale = FakeScale()
    gui.femoral_axis_whisker_tip_length_scale = FakeScale()
    gui.canvas = FakeCanvas()
    gui.current_image = object()
    gui.multi_review_mode = False
    return gui


def test_toggle_hover_disables_femoral_axis_tool():
    gui = make_gui_stub()
    gui.hover_enabled.set(True)
    gui.femoral_axis_enabled.set(True)
    calls = []
    gui._hide_hover_circle = lambda: calls.append("hide_hover")
    gui._clear_femoral_axis_overlay = lambda: calls.append("clear_femoral_axis")

    gui._toggle_hover()

    assert gui.radius_scale.state == "normal"
    assert gui.femoral_axis_enabled.get() is False
    assert gui.femoral_axis_count_scale.state == "disabled"
    assert gui.femoral_axis_whisker_tip_length_scale.state == "disabled"
    assert calls == ["clear_femoral_axis"]


def test_mousewheel_adjusts_femoral_axis_length_for_selected_fa_landmark():
    gui = make_gui_stub()
    gui.right_mouse_held = False
    gui.femoral_axis_enabled.set(True)
    gui.selected_landmark.set("L-FA")
    gui.hover_enabled.set(True)
    deltas = []
    gui._change_femoral_axis_length = lambda delta: deltas.append(delta)

    gui._on_mousewheel(SimpleNamespace(delta=120))

    assert deltas == [2]


def test_hover_radius_change_debounces_dependent_refresh():
    gui = make_gui_stub()
    key = "image.png"
    gui.current_image_path = Path("/tmp/image.png")
    gui.hover_enabled.set(True)
    gui.hover_radius = FakeVar(25)
    gui.hover_circle_id = None
    gui.selected_landmark.set("L-C-AC")
    gui.disp_scale = 1.0
    gui.hover_radii = {}
    gui._current_annotation_key = lambda: key
    gui._get_annotations = lambda: ({"L-C-AC": (10.0, 20.0)}, 0)

    scheduled = {}
    cancelled = []
    next_id = [0]

    def fake_after(delay, callback):
        next_id[0] += 1
        token = f"after-{next_id[0]}"
        scheduled[token] = (delay, callback)
        return token

    gui.after = fake_after
    gui.after_cancel = lambda token: cancelled.append(token)
    refreshes = []
    gui._refresh_hover_radius_dependents = lambda lm: refreshes.append(lm)

    gui._on_radius_change("25")
    first_token = gui._pending_hover_radius_sync_id
    gui.hover_radius.set(35)
    gui._on_radius_change("35")
    second_token = gui._pending_hover_radius_sync_id

    assert gui.hover_radii[key]["L-C-AC"] == 35.0
    assert first_token != second_token
    assert cancelled == [first_token]
    assert refreshes == []

    scheduled[second_token][1]()

    assert refreshes == ["L-C-AC"]
    assert gui._pending_hover_radius_sync_id is None


def test_update_femoral_axis_overlay_draws_whiskers_and_tip_caps():
    gui = make_gui_stub()
    gui.femoral_axis_enabled.set(True)
    gui.selected_landmark.set("L-FA")
    gui.femoral_axis_count.set(2)
    gui.femoral_axis_proj_length.set(12)
    gui.femoral_axis_whisker_tip_length.set(4)
    gui._get_active_femoral_axis_line_screen = lambda: (0.0, 0.0, 100.0, 0.0)

    gui._update_femoral_axis_overlay()

    assert len(gui.canvas.created) == 6
    assert len(gui.femoral_axis_item_ids) == 6
    assert gui.canvas.raised[-1] == "marker"


def test_draw_points_renders_ellipse_on_main_canvas(tmp_path):
    gui = make_gui_stub()
    gui.current_image_path = tmp_path / "image.tif"
    gui.json_path = tmp_path / "annotations.json"
    gui.db_path = None
    gui.current_image_quality = 0
    gui.landmarks = ["L-FAC"]
    gui.ellipse_landmarks = {"L-FAC", "R-FAC"}
    gui.line_landmarks = {"L-FA", "R-FA"}
    gui.landmark_visibility = {"L-FAC": FakeVar(True)}
    gui.landmark_found = {"L-FAC": FakeVar(False)}
    gui.landmark_flagged = {}
    gui.landmark_flag_widgets = {}
    gui.landmark_found_widgets = {}
    gui.landmark_meta = {gui._path_key(gui.current_image_path): {}}
    gui.hover_radii = {}
    gui.pair_line_ids = {}
    gui.current_image = object()
    gui.selected_landmark.set("L-FAC")
    gui.dialogue_font = FakeFont(12)
    gui.landmark_font = FakeFont(18)
    gui.disp_off = (0, 0)
    gui.disp_scale = 1.0
    gui.annotations = {
        gui._path_key(gui.current_image_path): {
            "L-FAC": {
                "type": "ellipse",
                "center": [50.0, 50.0],
                "major": [[20.0, 50.0], [80.0, 50.0]],
                "minor": [[50.0, 35.0], [50.0, 65.0]],
            }
        }
    }
    gui._img_to_screen = lambda x, y: (x, y)
    gui._update_overlay_for = lambda _lm: None
    gui._update_pair_lines = lambda: None
    gui._update_femoral_axis_overlay = lambda: None

    gui._draw_points()

    created_lines = [
        (coords, kwargs)
        for _item_id, coords, kwargs in gui.canvas.created
        if kwargs.get("tags") == "marker" and "fill" in kwargs
    ]
    canvas_texts = [
        kwargs["text"]
        for _item_id, _coords, kwargs in gui.canvas.created
        if "text" in kwargs
    ]
    assert len(created_lines) >= 3
    assert any(kwargs.get("dash") == (2, 4) for _coords, kwargs in created_lines)
    assert "I 0.0 deg\nA 30.0 deg" in canvas_texts
    assert len(gui.canvas.raised) >= 3


def test_multi_review_points_do_not_draw_reviewer_names_on_canvas():
    gui = make_gui_stub()
    gui.current_image_path = Path("/tmp/image.tif")
    gui.multi_review_reviewer_order = ["andrew", "paris"]
    gui.multi_review_visibility = {
        "andrew": FakeVar(True),
        "paris": FakeVar(True),
    }
    gui.multi_review_image_key_by_path = {"/tmp/image.tif": "image.tif"}
    gui.multi_review_annotations = {
        "image.tif": {
            "andrew": {"L-FHC": (10, 20), "L-FA": [(30, 40), (50, 60)]},
            "paris": {"L-FHC": (12, 22)},
        }
    }
    gui.multi_review_meta = {
        "image.tif": {
            "andrew": {"L-FHC": {"flag": True}},
            "paris": {},
        }
    }
    gui.multi_review_temp_annotations = {"image.tif": {}}
    gui.landmark_visibility = {}
    gui.selected_landmark.set("L-FHC")
    gui.dialogue_font = FakeFont(12)
    gui.landmark_font = FakeFont(18)
    gui._path_key = lambda path: str(path)
    gui._img_to_screen = lambda x, y: (x, y)
    gui._is_line_landmark = lambda lm: lm == "L-FA"

    gui._draw_multi_review_points()

    canvas_texts = [
        kwargs["text"] for _, _, kwargs in gui.canvas.created if "text" in kwargs
    ]
    assert canvas_texts
    assert all("andrew" not in text for text in canvas_texts)
    assert all("paris" not in text for text in canvas_texts)
    assert "!L-FHC" in canvas_texts
    assert "L-FHC" in canvas_texts
