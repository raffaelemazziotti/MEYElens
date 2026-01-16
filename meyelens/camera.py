"""
camera.py
=========

A small OpenCV-based camera wrapper with:

- Optional calibration loading from a TOML file.
- Optional frame undistortion (requires calibration).
- Fixed resolution / framerate configuration.
- Auto-exposure or manual exposure control.
- Interactive ROI selection and optional cropping.
- A simple preview window for quick diagnostics.

This module is written to be friendly to Sphinx autodoc. If you use NumPy-style
docstrings, enable ``sphinx.ext.napoleon`` in your ``conf.py``.

Notes
-----
OpenCV camera properties (auto-exposure, exposure units, gain, FPS) can behave
differently across operating systems, backends (DirectShow/MSMF/V4L2), and
camera drivers. The setters in this class *attempt* to apply requested values,
but your hardware may ignore some properties.

Dependencies
------------
- opencv-python
- numpy
- toml

Example
-------
>>> from camera import Camera
>>> with Camera(camera_index=0, resolution=(640, 480), auto_exposure=True) as cam:
...     frame = cam.get_frame()
...     if frame is not None:
...         cam.show(frame, "One frame")
...         cv2.waitKey(0)
...     cv2.destroyAllWindows()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import toml

CropTuple = Tuple[int, int, int, int]  # (top, left, height, width)


class Camera:
    """
    High-level wrapper around :class:`cv2.VideoCapture`.

    Parameters
    ----------
    camera_index
        Index passed to :class:`cv2.VideoCapture`. Common values are ``0`` or ``1``.
        Some systems support special values (e.g. ``-1``) but this is backend-dependent.
    calibration_file
        Path to a TOML file containing camera calibration parameters.
        Expected keys:
        - ``camera_matrix``: 3x3 array-like
        - ``distortion_coefficients``: array-like (e.g. 1x5, 1x8)
    undistort
        If ``True`` and calibration parameters are available, frames are undistorted
        on read.
    exposure
        Manual exposure value passed to OpenCV when auto-exposure is disabled.
        The numeric meaning is backend/driver-dependent.
    framerate
        Requested camera FPS. Note: many cameras ignore this, or it depends on
        resolution/exposure.
    resolution
        Requested camera resolution as ``(width, height)``.
    auto_exposure
        If ``True`` attempt to enable auto-exposure; if ``False`` attempt to disable
        auto-exposure and apply manual exposure/gain settings.
    crop
        Optional crop rectangle stored as ``(top, left, height, width)``.
        If provided, frames returned by :meth:`get_frame` are cropped accordingly.

    Attributes
    ----------
    cap
        The underlying :class:`cv2.VideoCapture`.
    camera_matrix
        Loaded camera intrinsic matrix (or ``None`` if not available).
    dist_coeffs
        Loaded distortion coefficients (or ``None`` if not available).
    crop
        Crop rectangle in ``(top, left, height, width)`` format, or ``None``.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    """

    def __init__(
        self,
        camera_index: int = 0,
        calibration_file: Union[str, Path] = "camera_calibration.toml",
        undistort: bool = False,
        exposure: float = 0,
        framerate: float = 30,
        resolution: Tuple[int, int] = (640, 480),
        auto_exposure: bool = True,
        crop: Optional[CropTuple] = None,
    ) -> None:
        # Open camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        # Calibration / undistortion state
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self._new_camera_matrix: Optional[np.ndarray] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h for OpenCV ROI

        # Configuration
        self.undistort = bool(undistort)
        self.exposure = exposure
        self.framerate = framerate
        self.resolution = resolution
        self.auto_exposure = bool(auto_exposure)

        # Optional crop (top, left, height, width)
        self.crop: Optional[CropTuple] = self._validate_crop(crop)

        # Apply settings
        self.load_calibration(calibration_file)
        self.set_resolution(self.resolution)
        self.set_auto_exposure(self.auto_exposure)

        # Many backends tie achievable FPS to exposure; keep original behavior:
        # only set framerate when auto-exposure is disabled.
        if not self.auto_exposure:
            self.set_framerate(self.framerate)

    def __enter__(self) -> "Camera":
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release camera resources on context exit."""
        self.close()

    @staticmethod
    def _validate_crop(crop: Optional[CropTuple]) -> Optional[CropTuple]:
        """
        Validate and normalize a crop tuple.

        Parameters
        ----------
        crop
            ``(top, left, height, width)`` or ``None``.

        Returns
        -------
        Optional[CropTuple]
            The validated crop or ``None``.

        Raises
        ------
        ValueError
            If the crop tuple is not a 4-tuple or contains invalid values.
        """
        if crop is None:
            return None
        if len(crop) != 4:
            raise ValueError("crop must be a 4-tuple: (top, left, height, width).")
        top, left, height, width = crop
        if height <= 0 or width <= 0:
            raise ValueError("crop height and width must be positive.")
        if top < 0 or left < 0:
            raise ValueError("crop top and left must be non-negative.")
        return int(top), int(left), int(height), int(width)

    def load_calibration(self, calibration_file: Union[str, Path]) -> None:
        """
        Load camera calibration from a TOML file.

        Parameters
        ----------
        calibration_file
            TOML file path. Must contain at least ``camera_matrix`` and
            ``distortion_coefficients`` keys.

        Notes
        -----
        If loading fails for any reason, the camera continues operating without
        calibration.
        """
        calibration_path = Path(calibration_file)
        if not calibration_path.exists():
            print(
                f"### CAMERA ### Calibration file '{calibration_path}' not found. "
                "Proceeding without calibration."
            )
            return

        try:
            with calibration_path.open("r", encoding="utf-8") as f:
                calibration_data = toml.load(f)

            self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(calibration_data["distortion_coefficients"], dtype=np.float64)

            if self.camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix must be 3x3, got {self.camera_matrix.shape}.")

            print(f"### CAMERA ### Camera calibration parameters loaded from '{calibration_path}'.")

            # Precompute matrices for undistortion if we already know the resolution.
            self._prepare_undistort_matrices()

        except KeyError as e:
            print(f"### CAMERA ### Missing key in calibration file: {e}. Proceeding without calibration.")
            self.camera_matrix = None
            self.dist_coeffs = None
        except Exception as e:
            print(f"### CAMERA ### Error loading calibration file ({e}). Proceeding without calibration.")
            self.camera_matrix = None
            self.dist_coeffs = None

    def _prepare_undistort_matrices(self) -> None:
        """
        Precompute undistortion matrices if calibration is available.

        This is called after calibration load and after resolution changes.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._new_camera_matrix = None
            self._roi = None
            return

        width, height = self.resolution
        if width <= 0 or height <= 0:
            self._new_camera_matrix = None
            self._roi = None
            return

        self._new_camera_matrix, self._roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (width, height),
            alpha=1.0,
            newImgSize=(width, height),
        )

    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """
        Attempt to set the capture resolution.

        Parameters
        ----------
        resolution
            ``(width, height)``.
        """
        self.resolution = resolution
        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot set resolution.")
            return

        width, height = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        print(f"### CAMERA ### Resolution set (requested) to {width}x{height}.")

        # Update undistortion matrices in case resolution changed.
        self._prepare_undistort_matrices()

    def set_framerate(self, framerate: float) -> None:
        """
        Attempt to set the capture framerate.

        Parameters
        ----------
        framerate
            Requested FPS.
        """
        self.framerate = framerate
        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot set framerate.")
            return

        self.cap.set(cv2.CAP_PROP_FPS, float(framerate))
        print(f"### CAMERA ### Framerate set (requested) to {framerate:.2f} FPS.")

    def set_auto_exposure(self, enabled: bool) -> None:
        """
        Attempt to enable/disable auto-exposure.

        Parameters
        ----------
        enabled
            If ``True``, attempt to enable auto-exposure. If ``False``, attempt to
            disable auto-exposure and apply manual exposure settings.

        Notes
        -----
        OpenCV uses different conventions depending on backend:

        - Some backends expect ``CAP_PROP_AUTO_EXPOSURE`` to be 0.25 for manual and
          0.75 for auto (common on V4L2).
        - Others accept 0/1.

        Here we try a reasonable approach while keeping original intent.
        """
        self.auto_exposure = bool(enabled)

        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change auto-exposure setting.")
            return

        if enabled:
            # Try both conventions.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print("### CAMERA ### Auto-exposure enabled (requested).")
        else:
            # Try both conventions.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

            # Apply manual exposure values as best-effort.
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
            print(f"### CAMERA ### Auto-exposure disabled (requested). Manual exposure set to {self.exposure}.")

    def set_exposure(self, exposure: float) -> bool:
        """
        Attempt to set manual exposure.

        Parameters
        ----------
        exposure
            Manual exposure value passed to OpenCV.

        Returns
        -------
        bool
            ``True`` if the camera is open and we attempted to set the property,
            ``False`` otherwise.

        Notes
        -----
        Many drivers require auto-exposure to be disabled for this to take effect.
        """
        self.exposure = exposure
        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change exposure.")
            return False

        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        print(f"### CAMERA ### Exposure set (requested) to {exposure}.")
        return True

    def get_frame(self, flip_vertical: bool = True, apply_crop: bool = True) -> Optional[np.ndarray]:
        """
        Capture a frame.

        Parameters
        ----------
        flip_vertical
            If ``True``, flip the frame vertically (OpenCV flipCode=0), matching
            the original behavior.
        apply_crop
            If ``True`` and :attr:`crop` is set, crop the returned frame.

        Returns
        -------
        Optional[numpy.ndarray]
            BGR image (H x W x 3) if successful, otherwise ``None``.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        if flip_vertical:
            frame = cv2.flip(frame, 0)

        # Apply undistortion only if enabled and calibration is available.
        if self.undistort and (self.camera_matrix is not None) and (self.dist_coeffs is not None):
            frame = self._undistort_frame(frame)

        # Apply crop if configured.
        if apply_crop and self.crop is not None:
            top, left, height, width = self.crop

            # Guard against out-of-bounds in case resolution changed.
            h, w = frame.shape[:2]
            top2 = max(0, min(top, h))
            left2 = max(0, min(left, w))
            bottom2 = max(0, min(top2 + height, h))
            right2 = max(0, min(left2 + width, w))

            frame = frame[top2:bottom2, left2:right2]

        return frame

    def _undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a frame using loaded calibration.

        Parameters
        ----------
        frame
            Input BGR image.

        Returns
        -------
        numpy.ndarray
            Undistorted image. If precomputed matrices are unavailable, returns
            the original frame.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame

        if self._new_camera_matrix is None:
            # Fallback: compute with current frame shape.
            h, w = frame.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            undist = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_mtx)
            x, y, rw, rh = roi
            return undist[y:y + rh, x:x + rw] if rw > 0 and rh > 0 else undist

        undist = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self._new_camera_matrix)
        if self._roi is None:
            return undist

        x, y, w, h = self._roi
        return undist[y:y + h, x:x + w] if w > 0 and h > 0 else undist

    @staticmethod
    def show(frame: np.ndarray, name: str = "Frame") -> None:
        """
        Display a frame in an OpenCV window.

        Parameters
        ----------
        frame
            Frame to display.
        name
            Window name.
        """
        cv2.imshow(name, frame)

    def wait_key(self, key: str = "q", delay_ms: int = 1) -> bool:
        """
        Check whether a given key was pressed in the last OpenCV event loop iteration.

        Parameters
        ----------
        key
            Single character key to detect (e.g. ``"q"``).
        delay_ms
            Delay in milliseconds passed to :func:`cv2.waitKey`.

        Returns
        -------
        bool
            ``True`` if the key was pressed, else ``False``.
        """
        if len(key) != 1:
            raise ValueError("key must be a single character.")
        return (cv2.waitKey(int(delay_ms)) & 0xFF) == ord(key)

    def preview(self, window_name: str = "Camera Preview") -> None:
        """
        Open a live preview window with a framerate readout.

        Controls
        --------
        - ESC: exit preview
        - 'o': increase exposure by +1 (only meaningful if manual exposure is supported)
        - 'p': decrease exposure by -1

        Parameters
        ----------
        window_name
            Name of the OpenCV window.
        """
        print("### CAMERA ### Starting preview mode. Press ESC to exit. 'o'/'p' adjust exposure.")

        frame_count = 0
        start_time = time.time()
        real_fps = 0.0

        while True:
            frame = self.get_frame()
            if frame is None:
                print("### CAMERA ### Failed to capture frame.")
                continue

            # Update measured FPS every ~2 seconds.
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                real_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Read back what OpenCV thinks the resolution is (may differ from requested).
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Overlay FPS + resolution.
            cv2.putText(frame, f"FPS: {real_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Resolution: {width}x{height}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            self.show(frame, window_name)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("o"):
                self.set_exposure(self.exposure + 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")
            elif key == ord("p"):
                self.set_exposure(self.exposure - 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")

        cv2.destroyWindow(window_name)

    def close(self) -> None:
        """
        Release camera resources and close OpenCV windows.

        Notes
        -----
        Calling :func:`cv2.destroyAllWindows` is global (it closes all OpenCV windows),
        so if you manage multiple windows externally you may prefer to destroy specific
        windows yourself.
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def select_roi(self, window_name: str = "Select ROI") -> None:
        """
        Interactively select a rectangular ROI and store it in :attr:`crop`.

        Workflow
        --------
        - Drag with left mouse button to draw a rectangle.
        - Press 's' to save the selection.
        - Press 'r' to reset the selection.
        - Press ESC to exit without changing :attr:`crop`.

        The selection is stored as ``(top, left, height, width)`` and will be applied
        by :meth:`get_frame` when ``apply_crop=True``.

        Parameters
        ----------
        window_name
            OpenCV window name used for ROI selection.
        """
        frame = self.get_frame(apply_crop=False)
        if frame is None:
            print("### CAMERA ### Cannot select ROI: failed to capture a frame.")
            return

        # We keep ROI in OpenCV-ish format during drawing: (left, top, width, height).
        drawing = False
        roi = [0, 0, 0, 0]  # left, top, width, height

        def on_mouse(event, x, y, flags, param) -> None:
            nonlocal drawing, roi
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                roi[0] = x  # left
                roi[1] = y  # top
                roi[2] = 0
                roi[3] = 0
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                roi[2] = x - roi[0]
                roi[3] = y - roi[1]
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                roi[2] = x - roi[0]
                roi[3] = y - roi[1]

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frame = self.get_frame(apply_crop=False)
            if frame is None:
                print("### CAMERA ### Failed to capture frame during ROI selection.")
                continue

            display = frame.copy()

            left, top, w, h = roi
            if drawing or (w != 0 and h != 0):
                cv2.rectangle(display, (left, top), (left + w, top + h), (0, 255, 0), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("### CAMERA ### ROI selection cancelled.")
                break

            if key == ord("r"):
                roi = [0, 0, 0, 0]
                print("### CAMERA ### ROI reset.")
                continue

            if key == ord("s"):
                left, top, w, h = roi

                # Normalize negative width/height (dragging up/left).
                if w < 0:
                    left += w
                    w = abs(w)
                if h < 0:
                    top += h
                    h = abs(h)

                # Clamp to frame bounds.
                H, W = frame.shape[:2]
                x1 = max(0, min(left, W))
                y1 = max(0, min(top, H))
                x2 = max(0, min(left + w, W))
                y2 = max(0, min(top + h, H))

                if x2 <= x1 or y2 <= y1:
                    print("### CAMERA ### Invalid ROI (empty). Please select again.")
                    continue

                # Store as (top, left, height, width) for slicing: frame[top:top+h, left:left+w]
                self.crop = (int(y1), int(x1), int(y2 - y1), int(x2 - x1))
                print(f"### CAMERA ### Selected ROI stored as crop={self.crop}")
                break

        cv2.destroyWindow(window_name)


# If you want a runnable quick-test, uncomment the block below.
# (For Sphinx, leaving it commented keeps autodoc output cleaner.)
#
# if __name__ == "__main__":
#     with Camera(camera_index=0, crop=(90, 210, 260, 280)) as cam:
#         cam.preview()
