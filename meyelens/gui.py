import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import toml
from PyQt6 import QtCore, QtGui, QtWidgets

from .meye import Meye


VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
MODEL_FILTER = "PyTorch models (*.pt *.pth *.ckpt);;All files (*)"


def prepare_frame(frame_bgr, settings):
    frame = frame_bgr

    if settings["flip_vertical"]:
        frame = cv2.flip(frame, 0)

    if settings["crop_enabled"]:
        height, width = frame.shape[:2]
        x0 = max(0, min(int(settings["crop_x"]), width - 1))
        y0 = max(0, min(int(settings["crop_y"]), height - 1))
        size = max(1, int(settings["crop_size"]))
        x1 = min(x0 + size, width)
        y1 = min(y0 + size, height)
        frame = frame[y0:y1, x0:x1]

    if settings["invert"]:
        frame = cv2.bitwise_not(frame)

    return frame


def feature_value(features, name, key, default=math.nan):
    return features.get(name, {}).get(key, default)


def tuple_value(value, index, default=math.nan):
    try:
        return value[index]
    except (IndexError, TypeError):
        return default


def feature_to_row(prefix, feature):
    centroid = feature.get("centroid", (math.nan, math.nan))
    ellipse = feature.get("ellipse", {})
    major_axis = ellipse.get("major_axis", {})
    minor_axis = ellipse.get("minor_axis", {})
    major_p1 = major_axis.get("p1", (math.nan, math.nan))
    major_p2 = major_axis.get("p2", (math.nan, math.nan))
    minor_p1 = minor_axis.get("p1", (math.nan, math.nan))
    minor_p2 = minor_axis.get("p2", (math.nan, math.nan))

    return {
        f"{prefix}_valid": bool(feature.get("valid", False)),
        f"{prefix}_x": tuple_value(centroid, 1),
        f"{prefix}_y": tuple_value(centroid, 0),
        f"{prefix}_area": feature.get("area", math.nan),
        f"{prefix}_ellipse_valid": bool(ellipse.get("valid", False)),
        f"{prefix}_major_diameter": ellipse.get("major_diameter", math.nan),
        f"{prefix}_minor_diameter": ellipse.get("minor_diameter", math.nan),
        f"{prefix}_orientation_deg": ellipse.get("orientation_deg", math.nan),
        f"{prefix}_ovality": ellipse.get("ovality", math.nan),
        f"{prefix}_eccentricity": ellipse.get("eccentricity", math.nan),
        f"{prefix}_major_p1_x": tuple_value(major_p1, 1),
        f"{prefix}_major_p1_y": tuple_value(major_p1, 0),
        f"{prefix}_major_p2_x": tuple_value(major_p2, 1),
        f"{prefix}_major_p2_y": tuple_value(major_p2, 0),
        f"{prefix}_minor_p1_x": tuple_value(minor_p1, 1),
        f"{prefix}_minor_p1_y": tuple_value(minor_p1, 0),
        f"{prefix}_minor_p2_x": tuple_value(minor_p2, 1),
        f"{prefix}_minor_p2_y": tuple_value(minor_p2, 0),
    }


def result_to_row(result, frame_index, source_time_ms=math.nan):
    features = result.features or {}
    row = {
        "frame_index": int(frame_index),
        "source_time_ms": source_time_ms,
        "inference_ms": result.inference_time_ms,
        "inference_fps": result.inference_fps,
    }
    row.update(feature_to_row("pupil", features.get("pupil", {})))
    row.update(feature_to_row("eye", features.get("eye", {})))
    return row


def image_to_pixmap(image):
    if image.ndim == 2:
        height, width = image.shape
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        )
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        qimage = QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )

    return QtGui.QPixmap.fromImage(qimage.copy())


class AspectPixmapLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = None

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = pixmap
        self._rescale()

    def _rescale(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        super().setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class ROIRectItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, size, view):
        super().__init__(0, 0, size, size)
        self.view = view
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True,
        )
        self.setPen(QtGui.QPen(QtGui.QColor("red"), 2))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            x = value.x()
            y = value.y()
            max_x = max(0, self.view.image_width - self.rect().width())
            max_y = max(0, self.view.image_height - self.rect().height())
            x = min(max(x, 0), max_x)
            y = min(max(y, 0), max_y)
            self.view.roi_changed.emit(int(x), int(y), int(self.rect().width()))
            return QtCore.QPointF(x, y)

        return super().itemChange(change, value)


class ROIView(QtWidgets.QGraphicsView):
    roi_changed = QtCore.pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.image_width = 0
        self.image_height = 0
        self.roi_item = None
        self.setMinimumSize(420, 300)
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def has_image(self):
        return self.image_width > 0 and self.image_height > 0

    def set_image(self, image):
        self.scene().clear()
        self.roi_item = None
        self.image_height, self.image_width = image.shape[:2]
        self.scene().addPixmap(image_to_pixmap(image))
        self.setSceneRect(0, 0, self.image_width, self.image_height)
        self.fitInView(
            self.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )

    def set_roi(self, x, y, size):
        if not self.has_image():
            return

        size = max(1, min(int(size), self.image_width, self.image_height))
        x = max(0, min(int(x), self.image_width - size))
        y = max(0, min(int(y), self.image_height - size))

        if self.roi_item is None:
            self.roi_item = ROIRectItem(size, self)
            self.scene().addItem(self.roi_item)
        else:
            self.roi_item.setRect(0, 0, size, size)

        self.roi_item.setPos(x, y)
        self.roi_changed.emit(x, y, size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.sceneRect().isNull():
            self.fitInView(
                self.sceneRect(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            )


class BatchVideoWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, float, float, str, int, int)
    finished = QtCore.pyqtSignal(object, int, bool)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, meye, video_paths, settings):
        super().__init__()
        self.meye = meye
        self.video_paths = [Path(path) for path in video_paths]
        self.settings = dict(settings)
        self._cancel_requested = False

    @QtCore.pyqtSlot()
    def cancel(self):
        self._cancel_requested = True

    def _video_plan(self, path):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        start = min(max(int(self.settings["start_frame"]), 1), max(frame_count, 1))
        requested_end = int(self.settings["end_frame"])
        end = frame_count if requested_end <= 0 else min(requested_end, frame_count)
        return {
            "path": path,
            "frame_count": frame_count,
            "fps": fps,
            "start": start,
            "end": max(start - 1, end),
            "total": max(0, end - start + 1),
        }

    def _output_paths(self, source_path, file_count):
        output_dir_text = str(self.settings.get("output_dir", "")).strip()
        output_dir = Path(output_dir_text) if output_dir_text else source_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        requested_name = str(self.settings.get("output_name", "")).strip()
        requested_stem = Path(requested_name).stem if requested_name else ""
        if requested_stem and file_count == 1:
            stem = requested_stem
        elif requested_stem:
            stem = f"{requested_stem}_{source_path.stem}"
        else:
            stem = f"{source_path.stem}_meye"

        csv_path = output_dir / f"{stem}.csv"
        overlay_path = (
            output_dir / f"{stem}_overlay.mp4"
            if self.settings["save_video"]
            else None
        )
        return csv_path, overlay_path

    @QtCore.pyqtSlot()
    def run(self):
        outputs = []
        total_processed = 0
        started_at = time.perf_counter()

        try:
            plans = [self._video_plan(path) for path in self.video_paths]
            total_frames = sum(plan["total"] for plan in plans)

            for file_index, plan in enumerate(plans, start=1):
                if self._cancel_requested:
                    break

                source_path = plan["path"]
                capture = cv2.VideoCapture(str(source_path))
                capture.set(cv2.CAP_PROP_POS_FRAMES, plan["start"] - 1)
                csv_path, overlay_path = self._output_paths(
                    source_path, len(plans)
                )
                csv_handle = None
                csv_writer = None
                video_writer = None
                file_processed = 0

                try:
                    while (
                        not self._cancel_requested
                        and file_processed < plan["total"]
                    ):
                        ok, frame = capture.read()
                        if not ok:
                            break

                        source_frame_index = plan["start"] - 1 + file_processed
                        prepared = prepare_frame(frame, self.settings)
                        if prepared.size == 0:
                            raise RuntimeError(
                                "The selected crop produced an empty frame."
                            )

                        source_time_ms = float(
                            capture.get(cv2.CAP_PROP_POS_MSEC)
                        )
                        result = self.meye.predict(prepared)
                        row = result_to_row(
                            result,
                            source_frame_index,
                            source_time_ms=source_time_ms,
                        )

                        if csv_writer is None:
                            csv_handle = csv_path.open(
                                "w", newline="", encoding="utf-8"
                            )
                            csv_writer = csv.DictWriter(
                                csv_handle, fieldnames=list(row)
                            )
                            csv_writer.writeheader()
                        csv_writer.writerow(row)

                        if overlay_path is not None:
                            overlay = self.meye.overlay(
                                prepared,
                                result,
                                alpha=self.settings["mask_opacity"],
                                draw_text=False,
                            )
                            if video_writer is None:
                                height, width = overlay.shape[:2]
                                video_writer = cv2.VideoWriter(
                                    str(overlay_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    plan["fps"],
                                    (width, height),
                                    True,
                                )
                                if not video_writer.isOpened():
                                    raise RuntimeError(
                                        "Could not create overlay video: "
                                        f"{overlay_path}"
                                    )
                            video_writer.write(overlay)

                        file_processed += 1
                        total_processed += 1
                        elapsed = max(time.perf_counter() - started_at, 1e-9)
                        processing_fps = total_processed / elapsed
                        eta = (
                            (total_frames - total_processed) / processing_fps
                            if processing_fps > 0
                            else math.inf
                        )
                        self.progress.emit(
                            total_processed,
                            total_frames,
                            processing_fps,
                            eta,
                            source_path.name,
                            file_index,
                            len(plans),
                        )
                finally:
                    capture.release()
                    if csv_handle is not None:
                        csv_handle.close()
                    if video_writer is not None:
                        video_writer.release()

                if file_processed:
                    outputs.append(
                        {
                            "source": str(source_path),
                            "csv": str(csv_path),
                            "overlay": (
                                str(overlay_path)
                                if overlay_path is not None
                                else ""
                            ),
                            "frames": file_processed,
                        }
                    )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(outputs, total_processed, self._cancel_requested)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.meye = None
        self.loaded_model_key = None
        self.preview_frame = None
        self.preview_frame_count = 0
        self.preview_frame_index = 0
        self.preview_video_fps = 0.0
        self._updating_roi = False
        self._updating_timeline = False
        self.worker_thread = None
        self.worker = None

        self.setWindowTitle("MEYElens Offline Analysis")
        self._build_ui()

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(150)
        self.preview_timer.timeout.connect(self.preview_selected_frame)

        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self.advance_playback)

        default_model = Path(__file__).resolve().parent / "models" / "meye_default.pt"
        if default_model.exists():
            self.model_path_edit.setText(str(default_model))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_browse_button = QtWidgets.QPushButton("Browse")
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cpu", "mps", "cuda"])

        self.video_list = QtWidgets.QListWidget()
        self.video_list.setMinimumHeight(100)
        self.video_add_button = QtWidgets.QPushButton("Add videos")
        self.video_remove_button = QtWidgets.QPushButton("Remove")
        self.video_clear_button = QtWidgets.QPushButton("Clear")

        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_button = QtWidgets.QPushButton("Browse")
        self.output_name_edit = QtWidgets.QLineEdit()
        self.output_name_edit.setPlaceholderText(
            "Blank uses <video>_meye; batch adds each source name"
        )

        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(1, 99_999_999)
        self.previous_frame_button = QtWidgets.QPushButton("<")
        self.previous_frame_button.setFixedWidth(34)
        self.play_button = QtWidgets.QPushButton("Play")
        self.next_frame_button = QtWidgets.QPushButton(">")
        self.next_frame_button.setFixedWidth(34)

        self.start_frame_spin = QtWidgets.QSpinBox()
        self.start_frame_spin.setRange(1, 99_999_999)
        self.end_frame_spin = QtWidgets.QSpinBox()
        self.end_frame_spin.setRange(1, 99_999_999)

        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setTracking(True)
        self.timeline_label = QtWidgets.QLabel("No video loaded")

        self.pupil_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.pupil_threshold_spin.setRange(0.01, 0.99)
        self.pupil_threshold_spin.setSingleStep(0.01)
        self.pupil_threshold_spin.setValue(0.50)
        self.eye_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.eye_threshold_spin.setRange(0.01, 0.99)
        self.eye_threshold_spin.setSingleStep(0.01)
        self.eye_threshold_spin.setValue(0.50)

        self.show_pupil_mask_check = QtWidgets.QCheckBox("Pupil coloring")
        self.show_pupil_mask_check.setChecked(True)
        self.show_eye_mask_check = QtWidgets.QCheckBox("Eye coloring")
        self.show_eye_mask_check.setChecked(True)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_value_label = QtWidgets.QLabel("45%")

        self.morphology_check = QtWidgets.QCheckBox("Morphology")
        self.morphology_check.setChecked(True)
        self.keep_biggest_check = QtWidgets.QCheckBox("Keep biggest component")
        self.keep_biggest_check.setChecked(True)
        self.fill_holes_check = QtWidgets.QCheckBox("Fill holes")
        self.fill_holes_check.setChecked(True)
        self.feature_labels_check = QtWidgets.QCheckBox("Feature labels")

        self.kernel_spin = QtWidgets.QSpinBox()
        self.kernel_spin.setRange(1, 99)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(5)

        self.invert_check = QtWidgets.QCheckBox("Invert image")
        self.flip_check = QtWidgets.QCheckBox("Flip vertically")
        self.crop_check = QtWidgets.QCheckBox("Enable crop")
        self.crop_check.setChecked(True)
        self.save_video_check = QtWidgets.QCheckBox("Save overlay video")

        self.crop_x_spin = QtWidgets.QSpinBox()
        self.crop_x_spin.setRange(0, 20_000)
        self.crop_y_spin = QtWidgets.QSpinBox()
        self.crop_y_spin.setRange(0, 20_000)
        self.crop_size_spin = QtWidgets.QSpinBox()
        self.crop_size_spin.setRange(1, 20_000)
        self.crop_size_spin.setValue(256)

        self.settings_save_button = QtWidgets.QPushButton("Save settings")
        self.settings_load_button = QtWidgets.QPushButton("Load settings")
        self.preview_button = QtWidgets.QPushButton("Preview frame")
        self.run_button = QtWidgets.QPushButton("Run analysis")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setWordWrap(True)

        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)
        form.addRow("Model:", self._path_row(self.model_path_edit, self.model_browse_button))
        form.addRow("Device:", self.device_combo)
        form.addRow("Videos:", self.video_list)

        video_buttons = QtWidgets.QHBoxLayout()
        video_buttons.addWidget(self.video_add_button)
        video_buttons.addWidget(self.video_remove_button)
        video_buttons.addWidget(self.video_clear_button)
        form.addRow(video_buttons)
        form.addRow("Output directory:", self._path_row(self.output_dir_edit, self.output_dir_button))
        form.addRow("Output filename:", self.output_name_edit)

        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Start"))
        range_row.addWidget(self.start_frame_spin)
        range_row.addWidget(QtWidgets.QLabel("End"))
        range_row.addWidget(self.end_frame_spin)
        form.addRow("Analysis range:", range_row)

        frame_row = QtWidgets.QHBoxLayout()
        frame_row.addWidget(self.previous_frame_button)
        frame_row.addWidget(self.frame_spin)
        frame_row.addWidget(self.play_button)
        frame_row.addWidget(self.next_frame_button)
        form.addRow("Preview frame:", frame_row)

        form.addRow("Pupil threshold:", self.pupil_threshold_spin)
        form.addRow("Eye threshold:", self.eye_threshold_spin)
        form.addRow("Morphology kernel:", self.kernel_spin)

        processing_row = QtWidgets.QHBoxLayout()
        processing_row.addWidget(self.morphology_check)
        processing_row.addWidget(self.keep_biggest_check)
        processing_row.addWidget(self.fill_holes_check)
        form.addRow(processing_row)
        form.addRow(self.feature_labels_check)

        mask_row = QtWidgets.QHBoxLayout()
        mask_row.addWidget(self.show_pupil_mask_check)
        mask_row.addWidget(self.show_eye_mask_check)
        form.addRow("Masks:", mask_row)

        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_value_label)
        form.addRow("Mask opacity:", opacity_row)

        transform_row = QtWidgets.QHBoxLayout()
        transform_row.addWidget(self.invert_check)
        transform_row.addWidget(self.flip_check)
        form.addRow(transform_row)
        form.addRow(self.crop_check)

        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addWidget(QtWidgets.QLabel("X"))
        crop_row.addWidget(self.crop_x_spin)
        crop_row.addWidget(QtWidgets.QLabel("Y"))
        crop_row.addWidget(self.crop_y_spin)
        crop_row.addWidget(QtWidgets.QLabel("Size"))
        crop_row.addWidget(self.crop_size_spin)
        form.addRow("Crop:", crop_row)
        form.addRow(self.save_video_check)

        settings_row = QtWidgets.QHBoxLayout()
        settings_row.addWidget(self.settings_save_button)
        settings_row.addWidget(self.settings_load_button)
        form.addRow(settings_row)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        form.addRow(button_row)
        form.addRow("Progress:", self.progress_bar)
        form.addRow(self.progress_label)

        form_scroll = QtWidgets.QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setWidget(form_widget)
        form_scroll.setMinimumWidth(440)

        self.roi_view = ROIView()
        self.processed_label = AspectPixmapLabel("Processed preview")
        self.processed_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(420, 300)
        self.processed_label.setFrameStyle(
            QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Sunken
        )
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        preview_widget = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_widget)
        preview_layout.addWidget(QtWidgets.QLabel("Input frame (drag the crop):"))
        preview_layout.addWidget(self.roi_view, stretch=1)
        preview_layout.addWidget(self.timeline_slider)
        preview_layout.addWidget(self.timeline_label)
        preview_layout.addWidget(QtWidgets.QLabel("Processed overlay:"))
        preview_layout.addWidget(self.processed_label, stretch=1)
        preview_layout.addWidget(self.info_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(form_scroll)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(splitter)

        self.model_browse_button.clicked.connect(self.browse_model)
        self.video_add_button.clicked.connect(self.add_videos)
        self.video_remove_button.clicked.connect(self.remove_selected_videos)
        self.video_clear_button.clicked.connect(self.clear_videos)
        self.video_list.currentRowChanged.connect(self.selected_video_changed)
        self.output_dir_button.clicked.connect(self.browse_output_directory)
        self.preview_button.clicked.connect(self.preview_selected_frame)
        self.play_button.clicked.connect(self.toggle_playback)
        self.previous_frame_button.clicked.connect(lambda: self.step_preview_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self.step_preview_frame(1))
        self.frame_spin.valueChanged.connect(self.frame_spin_changed)
        self.timeline_slider.valueChanged.connect(self.timeline_changed)
        self.timeline_slider.sliderReleased.connect(self.preview_selected_frame)
        self.run_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.settings_save_button.clicked.connect(self.save_gui_settings)
        self.settings_load_button.clicked.connect(self.load_gui_settings)
        self.roi_view.roi_changed.connect(self.roi_changed)
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_value_label.setText(f"{value}%")
        )

        for widget in (self.crop_x_spin, self.crop_y_spin, self.crop_size_spin):
            widget.valueChanged.connect(self.crop_spin_changed)

        for widget in (
            self.pupil_threshold_spin,
            self.eye_threshold_spin,
            self.kernel_spin,
            self.opacity_slider,
            self.morphology_check,
            self.keep_biggest_check,
            self.fill_holes_check,
            self.feature_labels_check,
            self.show_pupil_mask_check,
            self.show_eye_mask_check,
            self.invert_check,
            self.crop_check,
        ):
            if isinstance(widget, QtWidgets.QAbstractButton):
                widget.toggled.connect(self.refresh_preview)
            else:
                widget.valueChanged.connect(self.refresh_preview)

        self.flip_check.toggled.connect(self.flip_changed)

    @staticmethod
    def _path_row(line_edit, button):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    def video_paths(self):
        return [
            self.video_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self.video_list.count())
        ]

    def current_video_path(self):
        item = self.video_list.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else ""

    def settings(self):
        kernel = int(self.kernel_spin.value())
        if kernel % 2 == 0:
            kernel += 1
        return {
            "flip_vertical": self.flip_check.isChecked(),
            "invert": self.invert_check.isChecked(),
            "crop_enabled": self.crop_check.isChecked(),
            "crop_x": self.crop_x_spin.value(),
            "crop_y": self.crop_y_spin.value(),
            "crop_size": self.crop_size_spin.value(),
            "pupil_threshold": self.pupil_threshold_spin.value(),
            "eye_threshold": self.eye_threshold_spin.value(),
            "morphology": self.morphology_check.isChecked(),
            "keep_biggest": self.keep_biggest_check.isChecked(),
            "fill_holes": self.fill_holes_check.isChecked(),
            "feature_labels": self.feature_labels_check.isChecked(),
            "show_pupil_mask": self.show_pupil_mask_check.isChecked(),
            "show_eye_mask": self.show_eye_mask_check.isChecked(),
            "mask_opacity": self.opacity_slider.value() / 100.0,
            "kernel": kernel,
            "save_video": self.save_video_check.isChecked(),
            "start_frame": self.start_frame_spin.value(),
            "end_frame": self.end_frame_spin.value(),
            "output_dir": self.output_dir_edit.text().strip(),
            "output_name": self.output_name_edit.text().strip(),
        }

    def apply_model_settings(self):
        settings = self.settings()
        self.meye.set_threshold("pupil", settings["pupil_threshold"])
        self.meye.set_threshold("eye", settings["eye_threshold"])
        self.meye.use_morphology = settings["morphology"]
        self.meye.keep_biggest = settings["keep_biggest"]
        self.meye.fill_holes = settings["fill_holes"]
        self.meye.morphology_kernel_size = settings["kernel"]
        self.meye.compute_features = True
        self.meye.draw_area = settings["feature_labels"]
        self.meye.draw_text = False
        self.meye.show_pupil_mask = settings["show_pupil_mask"]
        self.meye.show_eye_mask = settings["show_eye_mask"]

    def ensure_model_loaded(self):
        model_path = self.model_path_edit.text().strip()
        device = self.device_combo.currentText()
        model_key = (model_path, device)
        if not model_path or not Path(model_path).is_file():
            QtWidgets.QMessageBox.critical(
                self, "Model error", "Select a valid PyTorch model file."
            )
            return False
        if self.meye is None or model_key != self.loaded_model_key:
            try:
                self.meye = Meye(
                    model_path=model_path,
                    gpu_device=device,
                    verbose=False,
                )
                self.loaded_model_key = model_key
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Model error", str(exc))
                return False
        self.apply_model_settings()
        return True

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select model",
            self.model_path_edit.text(),
            MODEL_FILTER,
        )
        if path:
            self.model_path_edit.setText(path)
            self.loaded_model_key = None

    def add_videos(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add videos", "", VIDEO_FILTER
        )
        existing = set(self.video_paths())
        for path in paths:
            if path in existing:
                continue
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.video_list.addItem(item)
            existing.add(path)
        if self.video_list.count() and self.video_list.currentRow() < 0:
            self.video_list.setCurrentRow(0)

    def remove_selected_videos(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        if self.video_list.count() and self.video_list.currentRow() < 0:
            self.video_list.setCurrentRow(0)
        elif not self.video_list.count():
            self.clear_preview_metadata()

    def clear_videos(self):
        self.stop_playback()
        self.video_list.clear()
        self.clear_preview_metadata()

    def selected_video_changed(self, _row):
        self.stop_playback()
        self.preview_frame = None
        self.load_video_metadata()

    def browse_output_directory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_edit.text() or str(Path.home()),
        )
        if path:
            self.output_dir_edit.setText(path)

    def clear_preview_metadata(self):
        self.preview_frame_count = 0
        self.preview_video_fps = 0.0
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_label.setText("No video loaded")

    def load_video_metadata(self):
        video_path = self.current_video_path()
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            self.clear_preview_metadata()
            return False
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 0.0

        self.preview_frame_count = frame_count
        self.preview_video_fps = fps
        maximum = max(frame_count - 1, 0)
        self._updating_timeline = True
        self.frame_spin.setMaximum(max(frame_count, 1))
        self.start_frame_spin.setMaximum(max(frame_count, 1))
        self.end_frame_spin.setMaximum(max(frame_count, 1))
        self.start_frame_spin.setValue(1)
        self.end_frame_spin.setValue(max(frame_count, 1))
        self.timeline_slider.setRange(0, maximum)
        self.timeline_slider.setEnabled(frame_count > 0)
        frame_index = min(self.frame_spin.value() - 1, maximum)
        self.frame_spin.setValue(frame_index + 1)
        self.timeline_slider.setValue(frame_index)
        self._updating_timeline = False
        self.update_timeline_label(frame_index)
        return frame_count > 0

    def update_timeline_label(self, frame_index):
        if self.preview_frame_count <= 0:
            self.timeline_label.setText("No video loaded")
            return
        if self.preview_video_fps > 0:
            current_seconds = frame_index / self.preview_video_fps
            duration_seconds = self.preview_frame_count / self.preview_video_fps
            time_text = f" | {current_seconds:.2f} / {duration_seconds:.2f} s"
        else:
            time_text = ""
        self.timeline_label.setText(
            f"Frame {frame_index + 1} / {self.preview_frame_count}{time_text}"
        )

    def frame_spin_changed(self, value):
        if self._updating_timeline:
            return
        frame_index = max(0, value - 1)
        self._updating_timeline = True
        self.timeline_slider.setValue(frame_index)
        self._updating_timeline = False
        self.update_timeline_label(frame_index)
        self.schedule_preview()

    def timeline_changed(self, frame_index):
        if self._updating_timeline:
            return
        self._updating_timeline = True
        self.frame_spin.setValue(frame_index + 1)
        self._updating_timeline = False
        self.update_timeline_label(frame_index)
        self.schedule_preview()

    def step_preview_frame(self, offset):
        if self.preview_frame_count <= 0 and not self.load_video_metadata():
            return
        value = min(
            max(self.frame_spin.value() + int(offset), 1),
            max(self.preview_frame_count, 1),
        )
        self.frame_spin.setValue(value)
        self.preview_timer.stop()
        self.preview_selected_frame()

    def schedule_preview(self):
        if self.preview_frame is not None and self.meye is not None:
            self.preview_timer.start()

    def toggle_playback(self):
        if self.playback_timer.isActive():
            self.stop_playback()
            return
        if self.preview_frame_count <= 0 and not self.load_video_metadata():
            return
        interval = (
            max(1, round(1000 / self.preview_video_fps))
            if self.preview_video_fps > 0
            else 33
        )
        self.playback_timer.start(interval)
        self.play_button.setText("Pause")
        self.advance_playback()

    def stop_playback(self):
        self.playback_timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")

    def advance_playback(self):
        if self.frame_spin.value() >= self.preview_frame_count:
            self.stop_playback()
            return
        self._updating_timeline = True
        next_frame = self.frame_spin.value() + 1
        self.frame_spin.setValue(next_frame)
        self.timeline_slider.setValue(next_frame - 1)
        self._updating_timeline = False
        self.update_timeline_label(next_frame - 1)
        self.preview_timer.stop()
        self.preview_selected_frame()

    def preview_selected_frame(self):
        video_path = self.current_video_path()
        if not video_path:
            QtWidgets.QMessageBox.critical(
                self, "Video error", "Add and select a video file."
            )
            return
        if not self.ensure_model_loaded():
            self.stop_playback()
            return
        if self.preview_frame_count <= 0:
            self.load_video_metadata()
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            QtWidgets.QMessageBox.critical(
                self, "Video error", "Could not open the selected video."
            )
            self.stop_playback()
            return
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = min(
            max(self.frame_spin.value() - 1, 0), max(frame_count - 1, 0)
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            QtWidgets.QMessageBox.critical(
                self, "Video error", "Could not read the selected frame."
            )
            self.stop_playback()
            return

        self.preview_frame = frame
        self.preview_frame_count = frame_count
        self.preview_frame_index = frame_index
        self._updating_timeline = True
        self.frame_spin.setValue(frame_index + 1)
        self.timeline_slider.setRange(0, max(frame_count - 1, 0))
        self.timeline_slider.setValue(frame_index)
        self._updating_timeline = False
        self.update_timeline_label(frame_index)
        self.update_input_preview()
        self.refresh_preview()

    def update_input_preview(self):
        if self.preview_frame is None:
            return
        display = self.preview_frame
        if self.flip_check.isChecked():
            display = cv2.flip(display, 0)
        self.roi_view.set_image(display)
        self._updating_roi = True
        self.roi_view.set_roi(
            self.crop_x_spin.value(),
            self.crop_y_spin.value(),
            self.crop_size_spin.value(),
        )
        self._updating_roi = False

    @QtCore.pyqtSlot()
    def refresh_preview(self, _value=None):
        if self.preview_frame is None or self.meye is None:
            return
        try:
            settings = self.settings()
            self.apply_model_settings()
            prepared = prepare_frame(self.preview_frame, settings)
            if prepared.size == 0:
                raise RuntimeError("The selected crop produced an empty frame.")
            result = self.meye.predict(prepared)
            overlay = self.meye.overlay(
                prepared,
                result,
                alpha=settings["mask_opacity"],
                draw_text=False,
            )
        except Exception as exc:
            self.info_label.setText(str(exc))
            return

        self.processed_label.set_source_pixmap(image_to_pixmap(overlay))
        pupil = result.features.get("pupil", {})
        eye = result.features.get("eye", {})
        self.info_label.setText(
            f"Frame {self.preview_frame_index + 1}/{self.preview_frame_count} | "
            f"Device: {self.meye.device} | "
            f"Inference: {result.inference_time_ms:.1f} ms\n"
            f"Pupil area: {pupil.get('area', 0)} | "
            f"Eye area: {eye.get('area', 0)}"
        )

    def flip_changed(self, _checked):
        self.update_input_preview()
        self.refresh_preview()

    def roi_changed(self, x, y, size):
        for widget, value in (
            (self.crop_x_spin, x),
            (self.crop_y_spin, y),
            (self.crop_size_spin, size),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        if not self._updating_roi and self.crop_check.isChecked():
            self.refresh_preview()

    def crop_spin_changed(self, _value):
        if self.roi_view.has_image():
            self.roi_view.set_roi(
                self.crop_x_spin.value(),
                self.crop_y_spin.value(),
                self.crop_size_spin.value(),
            )

    def save_gui_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GUI settings", "meyelens_gui.toml", "TOML files (*.toml)"
        )
        if not path:
            return
        data = {
            "gui": {
                **self.settings(),
                "model_path": self.model_path_edit.text().strip(),
                "device": self.device_combo.currentText(),
                "videos": self.video_paths(),
                "preview_frame": self.frame_spin.value(),
            }
        }
        try:
            with Path(path).open("w", encoding="utf-8") as handle:
                toml.dump(data, handle)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Settings error", str(exc))

    def load_gui_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load GUI settings", "", "TOML files (*.toml)"
        )
        if not path:
            return
        try:
            values = toml.load(path).get("gui", {})
            self._apply_loaded_settings(values)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Settings error", str(exc))

    def _apply_loaded_settings(self, values):
        saved_start = values.get("start_frame")
        saved_end = values.get("end_frame")
        saved_preview = int(values.get("preview_frame", 1))
        spin_widgets = {
            "crop_x": self.crop_x_spin,
            "crop_y": self.crop_y_spin,
            "crop_size": self.crop_size_spin,
            "pupil_threshold": self.pupil_threshold_spin,
            "eye_threshold": self.eye_threshold_spin,
            "kernel": self.kernel_spin,
        }
        check_widgets = {
            "flip_vertical": self.flip_check,
            "invert": self.invert_check,
            "crop_enabled": self.crop_check,
            "morphology": self.morphology_check,
            "keep_biggest": self.keep_biggest_check,
            "fill_holes": self.fill_holes_check,
            "feature_labels": self.feature_labels_check,
            "show_pupil_mask": self.show_pupil_mask_check,
            "show_eye_mask": self.show_eye_mask_check,
            "save_video": self.save_video_check,
        }
        for key, widget in spin_widgets.items():
            if key in values:
                widget.setValue(values[key])
        for key, widget in check_widgets.items():
            if key in values:
                widget.setChecked(bool(values[key]))
        if "mask_opacity" in values:
            self.opacity_slider.setValue(round(float(values["mask_opacity"]) * 100))
        if "model_path" in values:
            self.model_path_edit.setText(str(values["model_path"]))
            self.loaded_model_key = None
        if "device" in values:
            index = self.device_combo.findText(str(values["device"]))
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        self.output_dir_edit.setText(str(values.get("output_dir", "")))
        self.output_name_edit.setText(str(values.get("output_name", "")))

        self.video_list.clear()
        for path in values.get("videos", []):
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setToolTip(str(path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            self.video_list.addItem(item)
        if self.video_list.count():
            self.video_list.setCurrentRow(0)
            if saved_start is not None:
                self.start_frame_spin.setValue(int(saved_start))
            if saved_end is not None:
                self.end_frame_spin.setValue(int(saved_end))
            self.frame_spin.setValue(saved_preview)

    def start_analysis(self):
        paths = self.video_paths()
        if not paths:
            QtWidgets.QMessageBox.critical(
                self, "Video error", "Add at least one video file."
            )
            return
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            QtWidgets.QMessageBox.critical(
                self, "Video error", f"Video does not exist: {missing[0]}"
            )
            return
        if self.start_frame_spin.value() > self.end_frame_spin.value():
            QtWidgets.QMessageBox.critical(
                self,
                "Frame range error",
                "The start frame must not be after the end frame.",
            )
            return
        if not self.ensure_model_loaded():
            return

        self.stop_playback()
        self.set_processing_state(True)
        self.worker_thread = QtCore.QThread(self)
        self.worker = BatchVideoWorker(self.meye, paths, self.settings())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def set_processing_state(self, active):
        for widget in (
            self.preview_button,
            self.play_button,
            self.run_button,
            self.video_add_button,
            self.video_remove_button,
            self.video_clear_button,
        ):
            widget.setEnabled(not active)
        self.cancel_button.setEnabled(active)
        self.progress_bar.setVisible(active)
        if active:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Preparing analysis...")

    def update_progress(
        self,
        current,
        total,
        processing_fps,
        eta_seconds,
        filename,
        file_index,
        file_count,
    ):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        eta_text = (
            f"{eta_seconds:.0f} s"
            if np.isfinite(eta_seconds)
            else "calculating"
        )
        self.progress_label.setText(
            f"Video {file_index}/{file_count}: {filename} | "
            f"{processing_fps:.2f} frames/s | ETA {eta_text}"
        )

    def cancel_analysis(self):
        if self.worker is not None:
            self.worker.cancel()
            self.progress_label.setText("Cancelling after the current frame...")

    def analysis_finished(self, outputs, frame_count, cancelled):
        self.set_processing_state(False)
        self.worker = None
        self.worker_thread = None
        title = "Analysis cancelled" if cancelled else "Analysis complete"
        message = f"Processed {frame_count} frames across {len(outputs)} video(s)."
        if outputs:
            message += "\n\n" + "\n".join(item["csv"] for item in outputs)
        self.progress_label.setText(message)
        QtWidgets.QMessageBox.information(self, title, message)

    def analysis_failed(self, message):
        self.set_processing_state(False)
        self.worker = None
        self.worker_thread = None
        self.progress_label.setText(message)
        QtWidgets.QMessageBox.critical(self, "Analysis failed", message)

    def closeEvent(self, event):
        self.stop_playback()
        if self.worker is not None:
            self.worker.cancel()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1350, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
