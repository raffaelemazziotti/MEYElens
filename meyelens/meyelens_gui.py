#!/usr/bin/env python3
"""
Pupil analysis GUI (PyQt6) with interactive ROI

- Loads a CNN model that outputs (mask, info)
- Lets you choose a video and processing options in a GUI
- Shows preview inside the GUI:
    * full (flipped) frame with draggable square ROI
    * processed overlay preview (mask + centroid), based on cropped+resized input
- Run full analysis:
    * CSV with pupil size, centroid, eye/blink prob
    * Optional overlay video with prediction mask + centroid

Assumptions:
    model(input) -> (mask, info)
    mask: (1, H, W, 1)  probability map
    info: (1, 2)        [eyeProbability, blinkProbability]
"""

import os
import sys
from importlib.resources import files

import cv2 as cv
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from skimage.measure import label, regionprops

from meyelens.meye import Meye


def morphProcessing(sourceImg: np.ndarray,
                    threshold: float,
                    imclosing: int,
                    meye_model: Meye | None):
    """
    Binarize the prediction and keep the largest component using the core MEYE
    implementation when possible. For non-default closing sizes we fall back to
    the custom kernel logic so the UI control still works.
    """
    if meye_model is not None and imclosing == 13:
        return meye_model.morphProcessing(sourceImg, thr=threshold)

    binarized = sourceImg > threshold
    label_img = label(binarized)
    regions = regionprops(label_img)
    if len(regions) == 0:
        return np.zeros(sourceImg.shape, dtype="uint8"), (np.nan, np.nan)

    regions.sort(key=lambda x: x.area, reverse=True)
    centroid = regions[0].centroid
    if len(regions) > 1:
        for rg in regions[1:]:
            label_img[rg.coords[:, 0], rg.coords[:, 1]] = 0
    label_img[label_img != 0] = 1
    biggestRegion = (label_img * 255).astype(np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (imclosing, imclosing))
    morph = cv.morphologyEx(biggestRegion, cv.MORPH_CLOSE, kernel)
    return morph, centroid


def preprocess_frame_for_model(frame_bgr: np.ndarray,
                               settings: dict,
                               requiredFrameSize: tuple[int, int]) -> np.ndarray:
    """
    Apply processing in this order:
        BGR -> gray -> flip (optional) -> crop (optional) -> resize -> invert (optional)

    Returns gray uint8 frame of size requiredFrameSize.
    """
    frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    # 1) Flip (new reference system)
    if settings["FLIP_UD"]:
        frame = cv.flip(frame, 0)

    # 2) Optional square crop
    if settings["CROP_ENABLED"]:
        h, w = frame.shape[:2]
        x0 = int(settings["CROP_X_TOP"])
        y0 = int(settings["CROP_Y_TOP"])
        size = int(settings["CROP_SIZE"])

        if x0 < w and y0 < h:
            x1 = min(x0 + size, w)
            y1 = min(y0 + size, h)
            if x1 > x0 and y1 > y0:
                frame = frame[y0:y1, x0:x1]
            else:
                print("WARNING: invalid crop size, using full frame for this sample.")
        else:
            print("WARNING: crop origin outside image, using full frame for this sample.")

    # 3) Resize
    frame = cv.resize(frame, tuple(requiredFrameSize))

    # 4) Optional inversion
    if settings["INVERTIMAGE"]:
        frame = cv.bitwise_not(frame)

    return frame


# ---------------- ROI VIEW (INTERACTIVE) ---------------- #

class ROIRectItem(QtWidgets.QGraphicsRectItem):
    """Movable square ROI constrained inside the image."""

    def __init__(self, x, y, w, h, view: "ROIView"):
        super().__init__(x, y, w, h)
        self.view = view
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        # Needed so itemChange is called on move
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        pen = QtGui.QPen(QtGui.QColor("red"), 2)
        self.setPen(pen)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            newPos = value
            x = newPos.x()
            y = newPos.y()
            if self.view.image_width > 0 and self.view.image_height > 0:
                max_x = max(0, self.view.image_width - self.rect().width())
                max_y = max(0, self.view.image_height - self.rect().height())
                x = min(max(x, 0), max_x)
                y = min(max(y, 0), max_y)
            self.view.roiChanged.emit(int(x), int(y), int(self.rect().width()))
            return QtCore.QPointF(x, y)
        return super().itemChange(change, value)


class ROIView(QtWidgets.QGraphicsView):
    """
    QGraphicsView that shows the input frame and a draggable square ROI.
    Scene coordinates == image pixel coordinates.
    """

    roiChanged = QtCore.pyqtSignal(int, int, int)  # x, y, size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.pixmap_item = None
        self.roi_item: ROIRectItem | None = None
        self.image_width = 0
        self.image_height = 0
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def hasImage(self) -> bool:
        return self.image_width > 0 and self.image_height > 0

    def setImage(self, img_gray: np.ndarray):
        """Set the displayed image (numpy uint8, HxW)."""
        self.scene().clear()
        self.pixmap_item = None
        self.roi_item = None

        h, w = img_gray.shape
        self.image_width = w
        self.image_height = h

        qimg = QtGui.QImage(
            img_gray.data,
            w,
            h,
            img_gray.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene().addPixmap(pix)
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def setROI(self, x: int, y: int, size: int):
        """Create or move the ROI rectangle."""
        if not self.hasImage():
            return

        size = max(1, min(size, self.image_width, self.image_height))
        x = max(0, min(x, self.image_width - size))
        y = max(0, min(y, self.image_height - size))

        if self.roi_item is None:
            self.roi_item = ROIRectItem(0, 0, size, size, self)
            self.roi_item.setPos(x, y)
            self.scene().addItem(self.roi_item)
        else:
            self.roi_item.setRect(0, 0, size, size)
            self.roi_item.setPos(x, y)

        self.roiChanged.emit(int(x), int(y), int(size))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None and not self.scene().sceneRect().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# ---------------- MAIN WINDOW ---------------- #

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.default_model_path = self._resolve_packaged_model()
        self.meye_model: Meye | None = None
        self.model_path_loaded = None
        # True when we move ROI programmatically (Preview, Flip, etc.)
        self._updating_roi_from_code = False

        # Cached preview data (so we can reprocess on-the-fly)
        self.preview_frame_bgr = None
        self.preview_numFrames = 0
        self.preview_frame_index = 0

        self.setWindowTitle("Pupil analysis GUI (PyQt6, interactive ROI)")
        self._build_ui()

        if self.default_model_path:
            self.modelPathEdit.setText(self.default_model_path)

    # ---------- UI building ---------- #

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Left controls (form)
        self.modelPathEdit = QtWidgets.QLineEdit()
        self.modelBrowseBtn = QtWidgets.QPushButton("Browse…")

        self.videoPathEdit = QtWidgets.QLineEdit()
        self.videoBrowseBtn = QtWidgets.QPushButton("Browse…")

        self.frameSpin = QtWidgets.QSpinBox()
        self.frameSpin.setRange(1, 9999999)
        self.frameSpin.setValue(10)

        self.thresholdSpin = QtWidgets.QDoubleSpinBox()
        self.thresholdSpin.setRange(0.0, 1.0)
        self.thresholdSpin.setSingleStep(0.01)
        self.thresholdSpin.setDecimals(3)
        self.thresholdSpin.setValue(0.1)

        self.imclosingSpin = QtWidgets.QSpinBox()
        self.imclosingSpin.setRange(1, 200)
        self.imclosingSpin.setValue(13)

        self.invertCheck = QtWidgets.QCheckBox("Invert image")
        self.flipCheck = QtWidgets.QCheckBox("Flip upside-down")
        self.flipCheck.setChecked(False)

        self.cropEnableCheck = QtWidgets.QCheckBox("Enable crop")
        self.cropEnableCheck.setChecked(True)

        self.cropXSpin = QtWidgets.QSpinBox()
        self.cropXSpin.setRange(0, 5000)
        self.cropXSpin.setValue(0)

        self.cropYSpin = QtWidgets.QSpinBox()
        self.cropYSpin.setRange(0, 5000)
        self.cropYSpin.setValue(0)

        self.cropSizeSpin = QtWidgets.QSpinBox()
        self.cropSizeSpin.setRange(1, 5000)
        self.cropSizeSpin.setValue(256)

        self.saveVideoCheck = QtWidgets.QCheckBox("Save overlay video")
        self.saveVideoCheck.setChecked(False)

        self.previewBtn = QtWidgets.QPushButton("Preview frame")
        self.runBtn = QtWidgets.QPushButton("Run full analysis")

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)

        leftForm = QtWidgets.QFormLayout()

        # Model row
        modelRow = QtWidgets.QHBoxLayout()
        modelRow.addWidget(self.modelPathEdit)
        modelRow.addWidget(self.modelBrowseBtn)
        leftForm.addRow("Model file:", modelRow)

        # Video row
        videoRow = QtWidgets.QHBoxLayout()
        videoRow.addWidget(self.videoPathEdit)
        videoRow.addWidget(self.videoBrowseBtn)
        leftForm.addRow("Video file:", videoRow)

        leftForm.addRow("Frame to preview:", self.frameSpin)
        leftForm.addRow("Threshold:", self.thresholdSpin)
        leftForm.addRow("IMCLOSING (kernel radius):", self.imclosingSpin)

        binRow = QtWidgets.QHBoxLayout()
        binRow.addWidget(self.invertCheck)
        binRow.addWidget(self.flipCheck)
        leftForm.addRow(binRow)

        leftForm.addRow(self.cropEnableCheck)

        cropRow = QtWidgets.QHBoxLayout()
        cropRow.addWidget(QtWidgets.QLabel("Crop X top:"))
        cropRow.addWidget(self.cropXSpin)
        cropRow.addWidget(QtWidgets.QLabel("Crop Y top:"))
        cropRow.addWidget(self.cropYSpin)
        cropRow.addWidget(QtWidgets.QLabel("Crop size:"))
        cropRow.addWidget(self.cropSizeSpin)
        leftForm.addRow(cropRow)

        leftForm.addRow(self.saveVideoCheck)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.previewBtn)
        btnRow.addWidget(self.runBtn)
        leftForm.addRow(btnRow)

        leftForm.addRow("Progress:", self.progressBar)

        # Right side: preview
        self.roiView = ROIView()
        self.roiView.setMinimumSize(320, 240)

        self.processedLabel = QtWidgets.QLabel("Processed preview will appear here.")
        self.processedLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # Make overlay preview square so it doesn't stretch horizontally
        self.processedLabel.setFixedSize(320, 320)
        self.processedLabel.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Sunken)
        self.processedLabel.setScaledContents(True)

        self.infoLabel = QtWidgets.QLabel("")
        self.infoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        rightLayout = QtWidgets.QVBoxLayout()
        rightLayout.addWidget(QtWidgets.QLabel("Input frame (drag ROI):"))
        rightLayout.addWidget(self.roiView)
        rightLayout.addSpacing(8)
        rightLayout.addWidget(QtWidgets.QLabel("Processed overlay preview:"))
        rightLayout.addWidget(self.processedLabel)
        rightLayout.addWidget(self.infoLabel)

        # Combine left + right
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(leftForm, stretch=0)
        mainLayout.addLayout(rightLayout, stretch=1)

        central.setLayout(mainLayout)

        # Connections
        self.modelBrowseBtn.clicked.connect(self.on_browse_model)
        self.videoBrowseBtn.clicked.connect(self.on_browse_video)
        self.previewBtn.clicked.connect(self.on_preview_clicked)
        self.runBtn.clicked.connect(self.on_run_clicked)

        self.roiView.roiChanged.connect(self.on_roi_changed)
        self.cropXSpin.valueChanged.connect(self.on_crop_spin_changed)
        self.cropYSpin.valueChanged.connect(self.on_crop_spin_changed)
        self.cropSizeSpin.valueChanged.connect(self.on_crop_spin_changed)

        # Recompute preview when processing params change
        self.thresholdSpin.valueChanged.connect(self.on_processing_param_changed)
        self.imclosingSpin.valueChanged.connect(self.on_processing_param_changed)
        self.invertCheck.toggled.connect(self.on_processing_param_changed)
        self.cropEnableCheck.toggled.connect(self.on_processing_param_changed)
        self.flipCheck.toggled.connect(self.on_flip_changed)

    # ---------- Helpers ---------- #

    def get_settings(self) -> dict:
        return {
            "VIDEOPATH": self.videoPathEdit.text().strip(),
            "FRAMENUMBER": int(self.frameSpin.value()),
            "THRESHOLD": float(self.thresholdSpin.value()),
            "IMCLOSING": int(self.imclosingSpin.value()),
            "INVERTIMAGE": bool(self.invertCheck.isChecked()),
            "FLIP_UD": bool(self.flipCheck.isChecked()),
            "CROP_ENABLED": bool(self.cropEnableCheck.isChecked()),
            "CROP_X_TOP": int(self.cropXSpin.value()),
            "CROP_Y_TOP": int(self.cropYSpin.value()),
            "CROP_SIZE": int(self.cropSizeSpin.value()),
            "SAVE_VIDEO": bool(self.saveVideoCheck.isChecked()),
        }

    def ensure_model_loaded(self) -> bool:
        requested_model_path = self.modelPathEdit.text().strip()
        effective_model_path = requested_model_path or self.default_model_path or None

        if effective_model_path is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Please select a model file first. No packaged model was found.",
            )
            return False

        model_key = effective_model_path or "<package default>"
        if (self.meye_model is None) or (model_key != self.model_path_loaded):
            try:
                self.meye_model = Meye(model=effective_model_path)
                self.model_path_loaded = model_key
                print(f"Model loaded from {model_key}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error loading model", str(e))
                return False

        return True

    def _resolve_packaged_model(self) -> str:
        """Return the bundled model path (inside the package) if it exists."""
        try:
            candidate = files("meyelens.models").joinpath("meye-2022-01-24.h5")
            if candidate.is_file():
                return str(candidate)
        except Exception as exc:
            print(f"Could not resolve packaged model path: {exc}")
        return ""

    def _get_required_frame_size(self):
        if self.meye_model is None:
            raise RuntimeError("Model not loaded")
        return self.meye_model.requiredFrameSize

    def _predict_frame(self, frame_bgr: np.ndarray, settings: dict):
        """Preprocess a frame and run the MEYE model, returning mask, centroid and probabilities."""
        requiredFrameSize = self._get_required_frame_size()
        proc_frame = preprocess_frame_for_model(frame_bgr, settings, requiredFrameSize)

        networkInput = proc_frame.astype(np.float32) / 255.0
        networkInput = networkInput[None, :, :, None]
        mask, info = self.meye_model.model(networkInput, training=False)

        mask_arr = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
        info_arr = info.numpy() if hasattr(info, "numpy") else np.array(info)

        prediction = mask_arr[0, :, :, 0]

        morphedMask, centroid = morphProcessing(
            prediction,
            threshold=settings["THRESHOLD"],
            imclosing=settings["IMCLOSING"],
            meye_model=self.meye_model,
        )

        eye_prob = float(info_arr[0, 0])
        blink_prob = float(info_arr[0, 1])

        if morphedMask.dtype != np.uint8:
            morphedMask = morphedMask.astype(np.uint8)

        return proc_frame, morphedMask, centroid, eye_prob, blink_prob

    # ---------- Slots (UI callbacks) ---------- #

    def on_browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select model file",
            "",
            "Keras models (*.h5 *.keras);;All files (*)",
        )
        if path:
            self.modelPathEdit.setText(path)

    def on_browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.videoPathEdit.setText(path)

    def on_preview_clicked(self):
        if not self.ensure_model_loaded():
            return
        settings = self.get_settings()
        self.preview_one_frame(settings)

    def on_run_clicked(self):
        if not self.ensure_model_loaded():
            return
        settings = self.get_settings()
        self.process_full_video(settings)

    def on_roi_changed(self, x: int, y: int, size: int):
        """Called whenever ROI moves (by mouse or by code)."""
        # Always sync spinboxes to ROI, but don't emit their signals while doing it
        self.cropXSpin.blockSignals(True)
        self.cropYSpin.blockSignals(True)
        self.cropSizeSpin.blockSignals(True)
        self.cropXSpin.setValue(x)
        self.cropYSpin.setValue(y)
        self.cropSizeSpin.setValue(size)
        self.cropXSpin.blockSignals(False)
        self.cropYSpin.blockSignals(False)
        self.cropSizeSpin.blockSignals(False)

        # If ROI was moved by code (Preview/Flip), do not trigger recompute
        if self._updating_roi_from_code:
            return

        # User drag: recompute preview only if crop is enabled (checkbox)
        if self.preview_frame_bgr is not None and self.cropEnableCheck.isChecked():
            settings = self.get_settings()
            self._update_processed_preview(settings)

    def on_crop_spin_changed(self, _value):
        """Spinboxes changed -> update ROI rectangle (and thus crop if enabled)."""
        if not self.roiView.hasImage():
            return
        x = self.cropXSpin.value()
        y = self.cropYSpin.value()
        size = self.cropSizeSpin.value()
        # This will emit roiChanged; _updating_roi_from_code is False here,
        # so if crop is enabled, preview will be recomputed.
        self.roiView.setROI(x, y, size)

    def on_processing_param_changed(self, _value=None):
        """Threshold / IMCLOSING / invert / cropEnabled changed."""
        if self.preview_frame_bgr is None:
            return
        settings = self.get_settings()
        self._update_processed_preview(settings)

    def on_flip_changed(self, _value=None):
        """Flip checkbox changed: need to update input frame and preview."""
        if self.preview_frame_bgr is None:
            return
        settings = self.get_settings()

        # Update full frame display
        frame_gray = cv.cvtColor(self.preview_frame_bgr, cv.COLOR_BGR2GRAY)
        if settings["FLIP_UD"]:
            frame_gray = cv.flip(frame_gray, 0)

        self.roiView.setImage(frame_gray)

        # Silent ROI placement (do not treat as user drag)
        self._updating_roi_from_code = True
        self.roiView.setROI(settings["CROP_X_TOP"], settings["CROP_Y_TOP"], settings["CROP_SIZE"])
        self._updating_roi_from_code = False

        # Update processed preview
        self._update_processed_preview(settings)

    # ---------- Core actions (preview + full video) ---------- #

    def preview_one_frame(self, settings: dict):
        videopath = settings["VIDEOPATH"]
        if not os.path.isfile(videopath):
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid video path.")
            return

        cap = cv.VideoCapture(videopath)
        numFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        frameN = settings["FRAMENUMBER"]
        if frameN <= 0 or frameN > numFrames:
            frameN = max(1, numFrames // 2)  # middle frame fallback

        cap.set(cv.CAP_PROP_POS_FRAMES, frameN - 1)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", "Could not read frame from video.")
            return

        # Cache preview frame info
        self.preview_frame_bgr = frame_bgr
        self.preview_numFrames = numFrames
        self.preview_frame_index = frameN

        # full frame (gray, flipped if needed, BEFORE crop/resize)
        frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if settings["FLIP_UD"]:
            frame_gray = cv.flip(frame_gray, 0)

        # show full frame + ROI in GUI
        self.roiView.setImage(frame_gray)
        # Silent ROI placement (no recompute, no auto-enable crop)
        self._updating_roi_from_code = True
        self.roiView.setROI(settings["CROP_X_TOP"], settings["CROP_Y_TOP"], settings["CROP_SIZE"])
        self._updating_roi_from_code = False

        # compute processed preview (cropped+resized, depending on checkbox)
        self._update_processed_preview(settings)

    def _update_processed_preview(self, settings: dict):
        """Recompute processed overlay preview for the currently cached preview frame."""
        if self.preview_frame_bgr is None or self.meye_model is None:
            return

        proc_frame, morphedMask, centroid, eyeProbability, blinkProbability = self._predict_frame(
            self.preview_frame_bgr, settings
        )

        # Build overlay for processed preview (cropped+resized)
        overlay = Meye.overlay_roi(morphedMask, cv.cvtColor(proc_frame, cv.COLOR_GRAY2BGR))
        rgb = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        self.processedLabel.setPixmap(pix)  # label is fixed square, pixmap scaled inside

        if self.preview_numFrames > 0:
            self.infoLabel.setText(
                f"Frame {self.preview_frame_index}/{self.preview_numFrames}\n"
                f"Eye Probability:  {eyeProbability:6.2%}\n"
                f"Blink Probability:{blinkProbability:6.2%}"
            )
        else:
            self.infoLabel.setText(
                f"Eye Probability:  {eyeProbability:6.2%}\n"
                f"Blink Probability:{blinkProbability:6.2%}"
            )

    def process_full_video(self, settings: dict):
        videopath = settings["VIDEOPATH"]
        if not os.path.isfile(videopath):
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid video path.")
            return

        cap = cv.VideoCapture(videopath)
        numFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        requiredFrameSize = self._get_required_frame_size()

        rows = []
        video_writer = None
        video_out_path = None

        if settings["SAVE_VIDEO"]:
            fps = cap.get(cv.CAP_PROP_FPS)
            fps_value = fps if isinstance(fps, (int, float, np.floating)) else np.nan
            if fps_value <= 0 or np.isnan(fps_value):
                fps_value = 30.0
            h_out, w_out = requiredFrameSize
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            base_name = os.path.basename(videopath).rsplit(".", 1)[0]
            video_out_path = os.path.join(
                os.path.dirname(videopath),
                f"{base_name}_pupil_overlay.mp4",
            )
            video_writer = cv.VideoWriter(
                video_out_path,
                fourcc,
                fps_value,
                (w_out, h_out),
                isColor=True,
            )

        # Progress bar
        self.progressBar.setVisible(True)
        self.progressBar.setMaximum(numFrames)
        self.progressBar.setValue(0)

        try:
            for i in range(numFrames):
                ok, frame_bgr = cap.read()
                if not ok:
                    print(f"Could not read frame {i+1}, stopping.")
                    break

                frame, morphedMask, centroid, eyeProbability, blinkProbability = self._predict_frame(
                    frame_bgr, settings
                )

                rows.append(
                    {
                        "frameN": int(i + 1),
                        "pupilSize": float(np.sum(morphedMask) / 255.0),
                        "pupCntr_x": float(centroid[1]),
                        "pupCntr_y": float(centroid[0]),
                        "eyeProb": eyeProbability,
                        "blinkProb": blinkProbability,
                    }
                )

                if settings["SAVE_VIDEO"] and video_writer is not None:
                    overlay_base = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                    overlay = Meye.overlay_roi(morphedMask, overlay_base)
                    if not np.isnan(centroid[0]):
                        cv.drawMarker(
                            overlay,
                            (int(centroid[1]), int(centroid[0])),
                            color=(0, 255, 0),
                            markerType=cv.MARKER_CROSS,
                            markerSize=12,
                            thickness=2,
                        )
                    video_writer.write(overlay)

                if (i != 0) and (i % 400 == 0):
                    print(f"Processing frames... ({i}/{numFrames})")

                # update progress
                self.progressBar.setValue(i + 1)
                QtWidgets.QApplication.processEvents()

        finally:
            cap.release()
            if settings["SAVE_VIDEO"] and video_writer is not None:
                video_writer.release()
                print(f"Overlay video saved to {video_out_path}")

            self.progressBar.setVisible(False)

            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "frameN",
                        "pupilSize",
                        "pupCntr_x",
                        "pupCntr_y",
                        "eyeProb",
                        "blinkProb",
                    ],
                )
                base_name = os.path.basename(videopath).rsplit(".", 1)[0]
                output_csv_path = os.path.join(
                    os.path.dirname(videopath),
                    f"{base_name}_pupil.csv",
                )
                df.to_csv(output_csv_path, index=False)
                print(f"Data saved to {output_csv_path}")

                msg = f"CSV saved:\n{output_csv_path}"
                if video_out_path is not None:
                    msg += f"\nVideo saved:\n{video_out_path}"
                QtWidgets.QMessageBox.information(self, "Done", msg)


# ---------------- MAIN ---------------- #

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
