import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MEYElens requires PyTorch 2.2 or newer. Install the CPU, CUDA, or "
        "Apple MPS build for your platform from "
        "https://pytorch.org/get-started/locally/ before importing meyelens."
    ) from exc
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import toml
import copy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .fileio import BufferedFileWriter


class Camera:
    """
        High-level camera interface based on OpenCV ``cv2.VideoCapture``.

        The ``Camera`` class provides a small convenience wrapper around OpenCV
        camera acquisition. It is designed for MEYELens workflows, where a camera
        is used to acquire eye images for pupil/eye segmentation, preview,
        recording, or calibration.

        The class supports:

        - opening a camera by index;
        - setting requested resolution and frame rate;
        - enabling or disabling auto-exposure;
        - setting manual exposure values (not all OS support manual exposure);
        - optional camera undistortion from a TOML calibration file;
        - optional rectangular cropping of the eye region;
        - live camera preview;
        - interactive ROI selection.

        Parameters
        ----------
        camera_index : int, default=0
            Camera index passed to ``cv2.VideoCapture``.
            The default value ``0`` usually corresponds to the first available
            camera. Use ``1``, ``2``, etc. when multiple cameras are connected.

        calibration_file : str or pathlib.Path, default="camera_calibration.toml"
            Path to a TOML file containing camera calibration parameters.
            The calibration file must be a TOML file with the following top-level keys:

            camera_matrix : list of list of float
                A 3 x 3 OpenCV camera intrinsic matrix.

            distortion_coefficients : list of float
                OpenCV distortion coefficients. Usually five values are used:
                [k1, k2, p1, p2, k3].

            Example
            -------
            camera_matrix = [
                [615.23, 0.0, 319.50],
                [0.0, 614.87, 239.75],
                [0.0, 0.0, 1.0],
            ]

            distortion_coefficients = [-0.128, 0.042, 0.0008, -0.0004, -0.006]

            The file is expected to contain at least:

            ``camera_matrix``
                A 3 x 3 camera matrix.

            ``distortion_coefficients``
                Distortion coefficients compatible with OpenCV undistortion.

            If the file is missing, incomplete, or invalid, the camera still opens
            and acquisition continues without calibration.

        undistort : bool, default=False
            If ``True`` and valid calibration parameters are available, each frame
            returned by :meth:`get_frame` is undistorted using OpenCV.

            If ``False``, frames are returned without geometric correction.

        exposure : int or float, default=0
            Manual exposure value requested from the camera when
            ``auto_exposure=False``.

            The interpretation of exposure values depends on the camera driver and
            operating system. Some cameras expect negative values, others expect
            positive integer-like values.

        framerate : int or float, default=30
            Requested camera frame rate in frames per second.

            This value is passed to ``cv2.CAP_PROP_FPS``. The actual achieved frame
            rate may differ depending on the camera, driver, backend, lighting
            conditions, exposure settings, and USB bandwidth. It works aparently only on linux.

        resolution : tuple of int, default=(640, 480)
            Requested camera resolution as ``(width, height)``.

            This value is passed to ``cv2.CAP_PROP_FRAME_WIDTH`` and
            ``cv2.CAP_PROP_FRAME_HEIGHT``. The actual resolution may differ if the
            camera does not support the requested size.

        auto_exposure : bool, default=True
            If ``True``, the class attempts to enable camera auto-exposure.

            If ``False``, the class attempts to disable auto-exposure and apply the
            value provided in ``exposure``.

            Camera exposure control is backend-dependent. Some cameras or operating
            systems may ignore these requests.

        crop : tuple of int or None, default=None
            Optional rectangular crop applied to frames returned by
            :meth:`get_frame`.

            The crop must be provided as:

            ``(top, left, height, width)``

            where ``top`` and ``left`` are the coordinates of the upper-left corner,
            and ``height`` and ``width`` define the crop size in pixels.

            If ``None``, no crop is applied.

        Attributes
        ----------
        cap : cv2.VideoCapture
            OpenCV video capture object.

        camera_matrix : numpy.ndarray or None
            Camera intrinsic matrix loaded from the calibration file.
            ``None`` if no valid calibration was loaded.

        dist_coeffs : numpy.ndarray or None
            Camera distortion coefficients loaded from the calibration file.
            ``None`` if no valid calibration was loaded.

        undistort : bool
            Whether frames should be undistorted when calibration is available.

        exposure : int or float
            Current requested manual exposure value.

        framerate : int or float
            Current requested frame rate.

        resolution : tuple
            Current requested resolution as ``(width, height)``.

        auto_exposure : bool
            Whether auto-exposure is currently requested.

        crop : tuple of int or None
            Current crop rectangle as ``(top, left, height, width)``.
            This can be set manually or by calling :meth:`select_roi`.

        Raises
        ------
        RuntimeError
            If the camera cannot be opened with the requested ``camera_index``.

        ValueError
            If ``crop`` is provided but is not a valid
            ``(top, left, height, width)`` tuple.

        Methods
        -------
        load_calibration(calibration_file)
            Load camera calibration parameters from a TOML file.

        set_resolution(resolution)
            Request a new camera resolution as ``(width, height)``.

        set_framerate(framerate)
            Request a new camera frame rate in frames per second.

        set_auto_exposure(enabled)
            Enable or disable camera auto-exposure.

        set_exposure(exposure)
            Request a manual exposure value.

        get_frame(flip_vertical=True, apply_crop=True)
            Acquire one frame from the camera, optionally flipping, undistorting,
            and cropping it.

        show(frame, name="Frame")
            Display a frame in an OpenCV window.

        wait_key(key="q", delay_ms=1)
            Wait for a key press and return ``True`` if the requested key was
            pressed.

        refresh(delay_ms=1)
            Refresh OpenCV windows by calling ``cv2.waitKey``.

        preview(window_name="Camera Preview")
            Show a live camera preview with FPS and resolution information.

        select_roi(window_name="Select ROI")
            Interactively select a rectangular region of interest and store it in
            ``self.crop``.

        close()
            Release the camera and close OpenCV windows.

        Notes
        -----
        OpenCV camera settings are requests, not guarantees. Depending on the
        hardware and backend, the camera may not exactly apply the requested
        resolution, frame rate, exposure, or auto-exposure mode. Use
        :meth:`preview` to inspect the live image and estimated acquisition rate.

        By default, :meth:`get_frame` flips frames vertically
        with ``flip_vertical=True``. This is useful for some MEYELens camera
        setups, but users can disable it by calling:

        ``frame = cam.get_frame(flip_vertical=False)``

        The class can be used as a context manager. This is the recommended usage,
        because the camera is released automatically when the ``with`` block exits.

    """

    def __init__(
        self,
        camera_index=0,
        calibration_file="camera_calibration.toml",
        undistort=False,
        exposure=0,
        framerate=30,
        resolution=(640, 480),
        auto_exposure=True,
        crop=None,
    ):
        """
            Open a camera and configure acquisition settings.

            Parameters
            ----------
            camera_index : int, default=0
                Camera index passed to ``cv2.VideoCapture``. The default value ``0``
                usually opens the first available camera.

            calibration_file : str or pathlib.Path, default="camera_calibration.toml"
                Path to a TOML camera-calibration file.

                The file should contain the following top-level keys:

                ``camera_matrix``
                    A 3 x 3 OpenCV camera intrinsic matrix.

                ``distortion_coefficients``
                    OpenCV distortion coefficients, usually
                    ``[k1, k2, p1, p2, k3]``.

                If the file is missing or invalid, the camera still opens and the class
                proceeds without calibration.

            undistort : bool, default=False
                If ``True``, apply OpenCV undistortion to frames returned by
                :meth:`get_frame`, provided that valid calibration parameters were
                loaded from ``calibration_file``.

            exposure : int or float, default=0
                Manual exposure value requested from the camera when
                ``auto_exposure=False``.

                The meaning and valid range of this value depend on the camera,
                operating system, and OpenCV backend.

            framerate : int or float, default=30
                Requested camera frame rate in frames per second.

                This value is passed to ``cv2.CAP_PROP_FPS``. The actual frame rate may
                differ from the requested value.

            resolution : tuple of int, default=(640, 480)
                Requested camera resolution as ``(width, height)``.

                This value is passed to ``cv2.CAP_PROP_FRAME_WIDTH`` and
                ``cv2.CAP_PROP_FRAME_HEIGHT``. The actual resolution may differ if the
                camera does not support the requested size.

            auto_exposure : bool, default=True
                If ``True``, attempt to enable camera auto-exposure.

                If ``False``, attempt to disable auto-exposure and apply the manual
                value provided in ``exposure``.

                Exposure control is hardware- and backend-dependent. Some cameras may
                ignore these requests.

            crop : tuple of int or None, default=None
                Optional rectangular crop applied to frames returned by
                :meth:`get_frame`.

                The crop must be provided as ``(top, left, height, width)`` in pixels.

                ``top`` and ``left`` define the upper-left corner of the crop.
                ``height`` and ``width`` define the crop size.

                If ``None``, no crop is applied.

            Raises
            ------
            RuntimeError
                If the camera cannot be opened with the requested ``camera_index``.

            ValueError
                If ``crop`` is not ``None`` and is not a valid
                ``(top, left, height, width)`` tuple.

            Notes
            -----
            OpenCV camera properties are requests rather than guarantees. The actual
            resolution, frame rate, exposure mode, or exposure value may differ from the
            requested settings depending on the camera and driver.

            The constructor calls, in order:

            1. ``cv2.VideoCapture(camera_index)``
            2. :meth:`load_calibration`
            3. :meth:`set_resolution`
            4. :meth:`set_auto_exposure`
            5. :meth:`set_framerate`, only when ``auto_exposure=False``
        """

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        self.camera_matrix = None
        self.dist_coeffs = None
        self._new_camera_matrix = None
        self._roi = None

        self.undistort = bool(undistort)
        self.exposure = exposure
        self.framerate = framerate
        self.resolution = resolution
        self.auto_exposure = bool(auto_exposure)

        self.crop = self._validate_crop(crop)

        self.load_calibration(calibration_file)
        self.set_resolution(self.resolution)
        self.set_auto_exposure(self.auto_exposure)

        if not self.auto_exposure:
            self.set_framerate(self.framerate)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _validate_crop(crop):
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

    def load_calibration(self, calibration_file):
        """
            Load camera calibration parameters from a TOML file.

            The calibration file is used only when ``undistort=True``. If valid
            calibration parameters are loaded, frames returned by :meth:`get_frame`
            can be undistorted using OpenCV.

            The file must contain the following top-level TOML keys:

            ``camera_matrix``
                A 3 x 3 OpenCV intrinsic camera matrix.

            ``distortion_coefficients``
                OpenCV distortion coefficients. The most common format is
                ``[k1, k2, p1, p2, k3]``.

            Parameters
            ----------
            calibration_file : str or pathlib.Path
                Path to the TOML calibration file.

            Attributes Modified
            -------------------
            camera_matrix : numpy.ndarray or None
                Set to the loaded 3 x 3 camera matrix if loading succeeds.
                Set to ``None`` if loading fails.

            dist_coeffs : numpy.ndarray or None
                Set to the loaded distortion coefficients if loading succeeds.
                Set to ``None`` if loading fails.

            _new_camera_matrix : numpy.ndarray or None
                Updated by the internal undistortion setup when calibration succeeds.

            _roi : tuple or None
                Updated by the internal undistortion setup when calibration succeeds.

            Returns
            -------
            None
                This method modifies the camera object in place.

            Notes
            -----
            Camera calibration is optional but useful when geometrically accurate images are
            needed. It corrects lens distortion, which can otherwise affect pupil position,
            ellipse shape, area, and gaze-related measurements. For simple pupil-area
            recordings with a centered eye ROI, calibration is usually not required.

            If the file does not exist, the method prints a warning and continues
            without calibration.

            If the file exists but does not contain the required keys, or if the
            calibration parameters are invalid, the method prints a warning and
            continues without calibration.

            Camera calibration follows the standard OpenCV chessboard calibration
            procedure. See the official OpenCV tutorial:

            https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

            The resulting `camera_matrix` and `distortion_coefficients` should be saved
            in a TOML file.

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

            self.camera_matrix = np.array(
                calibration_data["camera_matrix"],
                dtype=np.float64,
            )

            self.dist_coeffs = np.array(
                calibration_data["distortion_coefficients"],
                dtype=np.float64,
            )

            if self.camera_matrix.shape != (3, 3):
                raise ValueError(
                    f"camera_matrix must be 3x3, got {self.camera_matrix.shape}."
                )

            print(
                f"### CAMERA ### Camera calibration parameters loaded from "
                f"'{calibration_path}'."
            )

            self._prepare_undistort_matrices()

        except KeyError as e:
            print(
                f"### CAMERA ### Missing key in calibration file: {e}. "
                "Proceeding without calibration."
            )
            self.camera_matrix = None
            self.dist_coeffs = None

        except Exception as e:
            print(
                f"### CAMERA ### Error loading calibration file ({e}). "
                "Proceeding without calibration."
            )
            self.camera_matrix = None
            self.dist_coeffs = None

    def _prepare_undistort_matrices(self):
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

    def set_resolution(self, resolution):
        self.resolution = resolution

        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot set resolution.")
            return

        width, height = resolution

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        print(f"### CAMERA ### Resolution set (requested) to {width}x{height}.")

        self._prepare_undistort_matrices()

    def set_framerate(self, framerate):
        self.framerate = framerate

        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot set framerate.")
            return

        self.cap.set(cv2.CAP_PROP_FPS, float(framerate))

        print(f"### CAMERA ### Framerate set (requested) to {framerate:.2f} FPS.")

    def set_auto_exposure(self, enabled):
        self.auto_exposure = bool(enabled)

        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change auto-exposure setting.")
            return

        if enabled:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print("### CAMERA ### Auto-exposure enabled (requested).")

        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

            print(
                f"### CAMERA ### Auto-exposure disabled (requested). "
                f"Manual exposure set to {self.exposure}."
            )

    def set_exposure(self, exposure):
        self.exposure = exposure

        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change exposure.")
            return False

        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        print(f"### CAMERA ### Exposure set (requested) to {exposure}.")

        return True

    def get_frame(self, flip_vertical=True, apply_crop=True):

        """
            Acquire one frame from the camera.

            The returned frame can optionally be flipped vertically, undistorted using
            the loaded camera calibration, and cropped to the selected region of
            interest.

            Parameters
            ----------
            flip_vertical : bool, default=True
                If ``True``, flip the frame vertically using ``cv2.flip(frame, 0)``.

            apply_crop : bool, default=True
                If ``True`` and ``self.crop`` is not ``None``, return only the selected
                crop region.

            Returns
            -------
            numpy.ndarray or None
                The acquired frame as an OpenCV image array, or ``None`` if frame
                acquisition failed.

            Notes
            -----
            If ``self.undistort=True`` and valid calibration parameters are available,
            the frame is undistorted before cropping.
        """

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        if flip_vertical:
            frame = cv2.flip(frame, 0)

        if self.undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
            frame = self._undistort_frame(frame)

        if apply_crop and self.crop is not None:
            top, left, height, width = self.crop

            h, w = frame.shape[:2]

            top2 = max(0, min(top, h))
            left2 = max(0, min(left, w))
            bottom2 = max(0, min(top2 + height, h))
            right2 = max(0, min(left2 + width, w))

            frame = frame[top2:bottom2, left2:right2]

        return frame

    def _undistort_frame(self, frame):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame

        if self._new_camera_matrix is None:
            h, w = frame.shape[:2]

            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (w, h),
                1,
                (w, h),
            )

            undist = cv2.undistort(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_mtx,
            )

            x, y, rw, rh = roi

            if rw > 0 and rh > 0:
                return undist[y:y + rh, x:x + rw]

            return undist

        undist = cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self._new_camera_matrix,
        )

        if self._roi is None:
            return undist

        x, y, w, h = self._roi

        if w > 0 and h > 0:
            return undist[y:y + h, x:x + w]

        return undist

    @staticmethod
    def show(frame, name="Frame"):
        cv2.imshow(name, frame)

    def wait_key(self, key="q", delay_ms=1):
        if len(key) != 1:
            raise ValueError("key must be a single character.")

        return (cv2.waitKey(int(delay_ms)) & 0xFF) == ord(key)

    def refresh(self, delay_ms=1):
        cv2.waitKey(int(delay_ms))

    def preview(self, window_name="Camera Preview"):

        """
            Show a live preview of the camera image.

            The preview window displays the current camera frame together with the
            estimated acquisition frame rate and the current camera resolution.

            During preview, the user can adjust manual exposure with the keyboard (if supported by OS and drivers):

            - ``o`` increases the exposure value by 1;
            - ``p`` decreases the exposure value by 1;
            - ``ESC`` closes the preview window.

            Parameters
            ----------
            window_name : str, default="Camera Preview"
                Name of the OpenCV window used to display the live camera preview.

            Returns
            -------
            None
                This method runs an interactive preview loop until the user presses
                ``ESC``.

            Notes
            -----
            Frames are acquired using :meth:`get_frame`, so the preview uses the current
            camera settings, including vertical flipping, undistortion, and cropping.

            Exposure changes are applied by calling :meth:`set_exposure`. Whether the
            camera responds to exposure changes depends on the camera hardware, driver,
            operating system, and OpenCV backend.

            The displayed FPS is estimated from the number of frames acquired over
            approximately two-second intervals.
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

            frame_count += 1
            elapsed = time.time() - start_time

            if elapsed >= 2.0:
                real_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cv2.putText(
                frame,
                f"FPS: {real_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

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

            if key == 27:
                break

            if key == ord("o"):
                self.set_exposure(self.exposure + 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")

            elif key == ord("p"):
                self.set_exposure(self.exposure - 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")

        cv2.destroyWindow(window_name)

    def close(self):
        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()

    def select_roi(self, window_name="Select ROI"):
        """
            Interactively select a rectangular region of interest.

            This method opens a live camera window and lets the user draw a rectangular
            ROI with the mouse. When the ROI is saved, it is stored in ``self.crop`` as:

            ``(top, left, height, width)``

            The selected crop is then automatically applied by :meth:`get_frame` when
            ``apply_crop=True``.

            Parameters
            ----------
            window_name : str, default="Select ROI"
                Name of the OpenCV window used for interactive ROI selection.

            Returns
            -------
            None
                This method modifies ``self.crop`` in place.

            Keyboard Controls
            -----------------
            s
                Save the currently selected ROI.

            r
                Reset the current selection and draw a new ROI.

            ESC
                Cancel ROI selection and keep the previous value of ``self.crop``.

            Notes
            -----
            The ROI is selected on uncropped frames by calling
            ``get_frame(apply_crop=False)``. This allows the user to redefine the ROI
            even if a previous crop is already active.

            The selected rectangle is clipped to the image boundaries before it is
            stored. If the user drags from bottom-right to top-left, negative width or
            height values are automatically converted to a valid rectangle.

            If no valid frame can be acquired, the method prints a warning and returns
            without modifying ``self.crop``.

        """

        frame = self.get_frame(apply_crop=False)

        if frame is None:
            print("### CAMERA ### Cannot select ROI: failed to capture a frame.")
            return

        print("")
        print("### CAMERA ROI SELECTION ###")
        print("  1. Drag with LEFT mouse button to draw the ROI rectangle.")
        print("  2. Press 's' to save the selected ROI.")
        print("  3. Press 'r' to reset the selection and draw again.")
        print("  4. Press ESC to cancel without changing the current ROI.")
        print("")
        print("### CAMERA ### ROI format after saving:")
        print("  crop = (top, left, height, width)")
        print("")

        drawing = False
        roi = [0, 0, 0, 0]  # left, top, width, height

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, roi

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                roi[0] = x
                roi[1] = y
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
                cv2.rectangle(
                    display,
                    (left, top),
                    (left + w, top + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    "s: save | r: reset | ESC: cancel",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            else:
                cv2.putText(
                    display,
                    "Drag with LEFT mouse button to select ROI",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display,
                    "s: save | r: reset | ESC: cancel",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("### CAMERA ### ROI selection cancelled. Current crop unchanged.")
                break

            if key == ord("r"):
                roi = [0, 0, 0, 0]
                print("### CAMERA ### ROI reset. Draw a new rectangle.")
                continue

            if key == ord("s"):
                left, top, w, h = roi

                if w == 0 or h == 0:
                    print("### CAMERA ### No ROI selected yet. Drag a rectangle first.")
                    continue

                if w < 0:
                    left += w
                    w = abs(w)

                if h < 0:
                    top += h
                    h = abs(h)

                H, W = frame.shape[:2]

                x1 = max(0, min(left, W))
                y1 = max(0, min(top, H))
                x2 = max(0, min(left + w, W))
                y2 = max(0, min(top + h, H))

                if x2 <= x1 or y2 <= y1:
                    print("### CAMERA ### Invalid ROI. Please select again.")
                    continue

                self.crop = (
                    int(y1),
                    int(x1),
                    int(y2 - y1),
                    int(x2 - x1),
                )

                print("")
                print("### CAMERA ### ROI selected.")
                print(f"### CAMERA ### crop = {self.crop}  # (top, left, height, width)")
                print(
                    f"### CAMERA ### corners = "
                    f"top_left=({int(x1)}, {int(y1)}), "
                    f"bottom_right=({int(x2)}, {int(y2)})  # (x, y)"
                )
                print("")

                break

        cv2.destroyWindow(window_name)


@dataclass
class MeyeResult:
    """
       Container for the output of a MEYELens prediction.

       A ``MeyeResult`` object is returned by :meth:`Meye.predict`. It stores the
       binary segmentation masks, optional probability maps, optional extracted
       mask features, and inference timing information for one input frame.

       Attributes
       ----------
       probabilities : dict
           Dictionary containing probability maps for each predicted class.

           Keys are class names, usually:

           - ``"pupil"``
           - ``"eye"``

           Values are 2D NumPy arrays with the same height and width as the input
           frame. Probability maps are included only when
           ``Meye.return_probabilities=True``. Otherwise, this dictionary is empty.

       masks : dict
           Dictionary containing binary segmentation masks for each predicted
           class.

           Keys are class names, usually:

           - ``"pupil"``
           - ``"eye"``

           Values are 2D ``uint8`` NumPy arrays with the same height and width as
           the input frame. Mask pixels are encoded as:

           - ``0`` for background;
           - ``255`` for the segmented object.

       features : dict
           Dictionary containing geometric features extracted from the binary
           masks.

           Keys are class names, usually ``"pupil"`` and ``"eye"``. Values are
           feature dictionaries returned by ``MeyeMaskFeatures.compute_mask``.

           Each feature dictionary may include:

           - ``valid``
           - ``area``
           - ``centroid``
           - ``ellipse``

           If feature computation is disabled with ``Meye.compute_features=False``,
           this dictionary is empty.

       inference_time_ms : float
           Time required to run neural-network inference for the frame, in
           milliseconds.

           This timing measures the model prediction step. It does not include
           camera acquisition, drawing, file writing, or downstream analysis.

       inference_fps : float
           Inference speed expressed as frames per second, computed from
           ``inference_time_ms``.

    """
    probabilities: dict
    masks: dict
    features: dict
    inference_time_ms: float
    inference_fps: float


class MeyeMaskFeatures:

    """
        Extract geometric features from MEYELens binary masks.

        ``MeyeMaskFeatures`` is a stateless helper class used to compute simple
        geometric measurements from segmentation masks, usually the ``"pupil"`` and
        ``"eye"`` masks returned by :meth:`Meye.predict`.

        The class is intentionally separate from :class:`Meye`: the ``Meye`` class
        handles neural-network inference, thresholding, morphology, and preview
        logic, while ``MeyeMaskFeatures`` handles measurement of the resulting
        binary masks.

        All methods are static. The class does not need to be instantiated.

        Computed Features
        -----------------
        For each input mask, the output is a dictionary with the following fields:

        valid : bool
            ``True`` if the mask contains at least one positive pixel.
            ``False`` if the mask is empty.

        area : int
            Number of positive pixels in the mask.

        centroid : tuple of float
            Mean position of all positive pixels, returned as ``(row, col)``.

            This follows NumPy image coordinates, where ``row`` corresponds to the
            vertical image coordinate and ``col`` corresponds to the horizontal image
            coordinate.

        ellipse : dict
            Dictionary containing the ellipse fitted to the largest external
            contour in the mask.

            If no valid ellipse can be fitted, the ellipse dictionary has
            ``valid=False`` and all numeric values are ``NaN``.

        Ellipse Features
        ----------------
        The ``ellipse`` dictionary contains:

        valid : bool
            ``True`` if an ellipse was successfully fitted.
            ``False`` otherwise.

        center : tuple of float
            Ellipse center as ``(row, col)``.

        major_diameter : float
            Diameter of the major axis of the fitted ellipse, in pixels.

        minor_diameter : float
            Diameter of the minor axis of the fitted ellipse, in pixels.

        orientation_deg : float
            Orientation of the major axis in degrees.

        ovality : float
            Shape elongation index computed as:

            ``1 - minor_diameter / major_diameter``

            Values close to ``0`` indicate a nearly circular shape. Larger values
            indicate a more elongated shape.

        eccentricity : float
            Ellipse eccentricity computed as:

            ``sqrt(1 - (minor_diameter / major_diameter) ** 2)``

            Values close to ``0`` indicate a nearly circular ellipse. Values closer
            to ``1`` indicate a more elongated ellipse.

        major_axis : dict
            Endpoints of the major axis. Contains:

            ``p1``
                First endpoint as ``(row, col)``.

            ``p2``
                Second endpoint as ``(row, col)``.

        minor_axis : dict
            Endpoints of the minor axis. Contains:

            ``p1``
                First endpoint as ``(row, col)``.

            ``p2``
                Second endpoint as ``(row, col)``.

        Notes
        -----
        Input masks are interpreted as binary masks. Any pixel value greater than
        zero is treated as part of the object.

        Ellipse fitting is performed on the largest external contour. This is useful
        when small isolated components are present, because the largest component is
        usually the pupil or eye region after morphology.

        A valid mask does not always produce a valid ellipse. OpenCV requires at
        least five contour points to fit an ellipse. Therefore, a mask can have
        ``valid=True`` while ``ellipse["valid"]`` is ``False``.

    """

    @staticmethod
    def compute_all(masks):
        @staticmethod
        def compute_all(masks):
            """
                Compute features for all masks in a dictionary.

                Parameters
                ----------
                masks : dict
                    Dictionary mapping mask names to binary masks. Keys are usually
                    ``"pupil"`` and ``"eye"``. Values are 2D NumPy arrays.

                Returns
                -------
                dict
                    Dictionary mapping each mask name to the output of
                    :meth:`compute_mask`.

            """
        features = {}

        for name, mask in masks.items():
            features[name] = MeyeMaskFeatures.compute_mask(mask)

        return features

    @staticmethod
    def compute_mask(mask):
        """
           Compute geometric features for one binary mask.

           Parameters
           ----------
           mask : numpy.ndarray
               Two-dimensional binary mask. Pixels greater than zero are treated as
               object pixels.

           Returns
           -------
           dict
               Feature dictionary with ``valid``, ``area``, ``centroid``, and
               ``ellipse`` fields.

           Notes
           -----
           If the mask is empty, ``valid`` is ``False``, ``area`` is ``0``,
           ``centroid`` is ``(np.nan, np.nan)``, and ``ellipse`` is an invalid ellipse
           dictionary.
        """
        binary = mask > 0
        ys, xs = np.where(binary)

        output = {
            "valid": False,
            "area": 0,
            "centroid": (np.nan, np.nan),
            "ellipse": MeyeMaskFeatures.empty_ellipse(),
        }

        if len(xs) == 0:
            return output

        output["valid"] = True
        output["area"] = int(len(xs))
        output["centroid"] = (float(np.mean(ys)), float(np.mean(xs)))
        output["ellipse"] = MeyeMaskFeatures.fit_ellipse(mask)

        return output

    @staticmethod
    def empty_ellipse():
        return {
            "valid": False,
            "center": (np.nan, np.nan),
            "major_diameter": np.nan,
            "minor_diameter": np.nan,
            "orientation_deg": np.nan,
            "ovality": np.nan,
            "eccentricity": np.nan,
            "major_axis": {
                "p1": (np.nan, np.nan),
                "p2": (np.nan, np.nan),
            },
            "minor_axis": {
                "p1": (np.nan, np.nan),
                "p2": (np.nan, np.nan),
            },
        }

    @staticmethod
    def fit_ellipse(mask):
        """
            Fit an ellipse to the largest external contour of a binary mask.

            Parameters
            ----------
            mask : numpy.ndarray
                Two-dimensional binary mask. Pixels greater than zero are treated as
                object pixels.

            Returns
            -------
            dict
                Ellipse feature dictionary containing center, major/minor diameters,
                orientation, ovality, eccentricity, and axis endpoints.

                If no ellipse can be fitted, returns :meth:`empty_ellipse`.

            Notes
            -----
            OpenCV requires at least five contour points to fit an ellipse. If the
            largest contour has fewer than five points, the returned ellipse is marked
            as invalid.

            Coordinates are returned as ``(row, col)`` for consistency with NumPy image
            indexing.
        """
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return MeyeMaskFeatures.empty_ellipse()

        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 5:
            return MeyeMaskFeatures.empty_ellipse()

        ellipse_raw = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), angle_deg = ellipse_raw

        axis_a = float(axis_a)
        axis_b = float(axis_b)
        angle_deg = float(angle_deg)

        if axis_a >= axis_b:
            major_diameter = axis_a
            minor_diameter = axis_b
            major_angle_deg = angle_deg
        else:
            major_diameter = axis_b
            minor_diameter = axis_a
            major_angle_deg = angle_deg + 90.0

        major_angle_deg = major_angle_deg % 180.0

        if major_diameter <= 0:
            ovality = np.nan
            eccentricity = np.nan
        else:
            ratio = minor_diameter / major_diameter
            ovality = 1.0 - ratio
            eccentricity = math.sqrt(max(0.0, 1.0 - ratio * ratio))

        theta = math.radians(major_angle_deg)

        major_dx = math.cos(theta) * major_diameter / 2.0
        major_dy = math.sin(theta) * major_diameter / 2.0

        minor_theta = theta + math.pi / 2.0
        minor_dx = math.cos(minor_theta) * minor_diameter / 2.0
        minor_dy = math.sin(minor_theta) * minor_diameter / 2.0

        major_p1 = (float(cy - major_dy), float(cx - major_dx))
        major_p2 = (float(cy + major_dy), float(cx + major_dx))

        minor_p1 = (float(cy - minor_dy), float(cx - minor_dx))
        minor_p2 = (float(cy + minor_dy), float(cx + minor_dx))

        return {
            "valid": True,
            "center": (float(cy), float(cx)),
            "major_diameter": float(major_diameter),
            "minor_diameter": float(minor_diameter),
            "orientation_deg": float(major_angle_deg),
            "ovality": float(ovality),
            "eccentricity": float(eccentricity),
            "major_axis": {
                "p1": major_p1,
                "p2": major_p2,
            },
            "minor_axis": {
                "p1": minor_p1,
                "p2": minor_p2,
            },
        }

    @staticmethod
    def draw(
        image,
        features,
        draw_area=True,
        draw_centroid=True,
        draw_ellipse=True,
        draw_axis_points=True,
    ):
        """
            Draw mask features on an image.

            Parameters
            ----------
            image : numpy.ndarray
                Image on which features are drawn. The image is modified in place and
                also returned.

            features : dict
                Dictionary of feature dictionaries, usually produced by
                :meth:`compute_all`.

            draw_area : bool, default=True
                If ``True``, draw text showing mask name, area, ovality, and ellipse
                orientation when available.

            draw_centroid : bool, default=True
                If ``True``, draw the mask centroid.

            draw_ellipse : bool, default=True
                If ``True``, draw the fitted ellipse when available.

            draw_axis_points : bool, default=True
                If ``True``, draw the major and minor ellipse axes and their endpoints.

            Returns
            -------
            numpy.ndarray
                The input image with the requested feature overlays.

            Notes
            -----
            The input image is modified in place.

            Feature coordinates are stored as ``(row, col)``, but OpenCV drawing uses
            ``(x, y)``. This method performs the conversion internally.
        """

        for idx, (name, feat) in enumerate(features.items()):
            if not feat.get("valid", False):
                continue

            color = MeyeMaskFeatures.feature_color(name, idx)

            centroid_row, centroid_col = feat["centroid"]
            ellipse = feat["ellipse"]

            if draw_centroid:
                cv2.circle(
                    image,
                    (int(round(centroid_col)), int(round(centroid_row))),
                    4,
                    color,
                    -1,
                )

            if draw_area:
                text = f"{name}: area={feat['area']}"

                if ellipse.get("valid", False):
                    text += (
                        f" ov={ellipse['ovality']:.2f}"
                        f" ang={ellipse['orientation_deg']:.1f}"
                    )

                cv2.putText(
                    image,
                    text,
                    (int(round(centroid_col)) + 6, int(round(centroid_row)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if ellipse.get("valid", False):
                center_row, center_col = ellipse["center"]
                major = ellipse["major_diameter"]
                minor = ellipse["minor_diameter"]
                angle = ellipse["orientation_deg"]

                if draw_ellipse:
                    cv2.ellipse(
                        image,
                        (
                            (float(center_col), float(center_row)),
                            (float(major), float(minor)),
                            float(angle),
                        ),
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                if draw_axis_points:
                    major_p1 = ellipse["major_axis"]["p1"]
                    major_p2 = ellipse["major_axis"]["p2"]
                    minor_p1 = ellipse["minor_axis"]["p1"]
                    minor_p2 = ellipse["minor_axis"]["p2"]

                    major_p1_xy = (int(round(major_p1[1])), int(round(major_p1[0])))
                    major_p2_xy = (int(round(major_p2[1])), int(round(major_p2[0])))
                    minor_p1_xy = (int(round(minor_p1[1])), int(round(minor_p1[0])))
                    minor_p2_xy = (int(round(minor_p2[1])), int(round(minor_p2[0])))

                    cv2.line(image, major_p1_xy, major_p2_xy, color, 1, cv2.LINE_AA)
                    cv2.line(image, minor_p1_xy, minor_p2_xy, color, 1, cv2.LINE_AA)

                    cv2.circle(image, major_p1_xy, 4, color, -1)
                    cv2.circle(image, major_p2_xy, 4, color, -1)
                    cv2.circle(image, minor_p1_xy, 3, color, -1)
                    cv2.circle(image, minor_p2_xy, 3, color, -1)

        return image

    @staticmethod
    def feature_color(name, idx):
        """
            Return the drawing color for a feature name.

            Parameters
            ----------
            name : str
                Feature or mask name. Special colors are used for ``"pupil"`` and
                ``"eye"``.

            idx : int
                Index of the feature in the drawing order. Used to select a fallback
                color for unknown names.

            Returns
            -------
            tuple of int
                OpenCV BGR color tuple.

            Notes
            -----
            Returned colors are in BGR order, not RGB order, because they are used with
            OpenCV drawing functions.
        """
        if name == "pupil":
            return (0, 0, 255)

        if name == "eye":
            return (0, 255, 0)

        palette = [
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        return palette[idx % len(palette)]


class Meye:
    """
        MEYELens neural-network interface for pupil and eye segmentation.

        ``Meye`` loads a trained PyTorch segmentation model and applies it to eye
        images. It predicts binary masks for the pupil and eye, optionally computes
        geometric mask features, applies post-processing morphology, and provides
        interactive preview tools for tuning segmentation thresholds.

        The class is designed as the main inference interface of MEYELens. A typical
        workflow is:

        1. Open a camera with :class:`Camera`.
        2. Create a ``Meye`` object.
        3. Acquire a frame.
        4. Run :meth:`predict`.
        5. Use the returned :class:`MeyeResult` for recording, analysis, preview, or
           gaze calibration.

        ``Meye`` uses a persistent TOML configuration file. The first time the class
        is used, a default configuration file is created automatically. When
        thresholds or preview options are changed during interactive preview, the
        configuration file is updated so that the next session starts with the last
        saved settings.

        Parameters
        ----------
        model_path : str or pathlib.Path or None, default=None
            Path to a trained MEYELens model checkpoint.

            If provided, this path overrides the model path stored in the TOML
            configuration file and is saved to that configuration.

            If ``None``, the model path is read from the configuration file. By
            default, this points to:

            ``models/meye_default.pt``

            relative to the source file.

        config_path : str or pathlib.Path or None, default=None
            Path to the TOML configuration file.

            If ``None``, the default path is used:

            ``~/.meyelens/meye_config.toml``

            If the file does not exist, it is created automatically using
            :attr:`DEFAULT_CONFIG`.

        gpu_device : {"auto", "cpu", int, str}, default="auto"
            Device used for inference.

            Supported values are:

            ``"auto"``
                Use CUDA if available, otherwise Apple MPS if available,
                otherwise CPU.

            ``"cpu"``
                Force CPU inference.

            int
                Use a specific CUDA device index, for example ``0`` for
                ``"cuda:0"``.

            str
                Any valid PyTorch device string, such as ``"cuda:0"``.

            If CUDA or MPS is requested but unavailable, ``Meye`` prints a message
            and falls back to CPU inference.

        use_half_precision : bool or None, default=None
            Whether to use half-precision inference.

            If ``None``, the value is read from the configuration file.
            If ``True``, half precision is used only when the selected device is a
            CUDA device. CPU and MPS inference always use full precision.

        reset_config : bool, default=False
            If ``True``, overwrite the configuration file with default settings
            before loading the model.

        verbose : bool, default=True
            If ``True``, print configuration, device, model-loading information, and
            current settings.

        Attributes
        ----------
        model : torch.nn.Module
            Loaded PyTorch segmentation model in evaluation mode.

        model_path : str
            Path to the currently loaded model checkpoint.

        config_path : pathlib.Path
            Path to the TOML configuration file.

        config : dict
            Runtime configuration dictionary loaded from the TOML file.

        device : torch.device
            PyTorch device used for inference.

        class_names : list of str
            Names of output classes. For the standard MEYELens model these are:

            - ``"pupil"``
            - ``"eye"``

        thresholds : dict
            Segmentation thresholds for each class.

            Standard keys are:

            - ``"pupil"``
            - ``"eye"``

        threshold_step : float
            Amount by which thresholds are changed during interactive preview.

        use_morphology : bool
            Whether to apply OpenCV morphological opening and closing to predicted
            masks.

        keep_biggest : bool
            Whether to keep only the largest connected component in each predicted
            mask.

        fill_holes : bool
            Whether to fill internal holes in each predicted mask.

        morphology_kernel_size : int
            Kernel size used for morphological operations.

        morphology_open_iterations : int
            Number of morphological opening iterations.

        morphology_close_iterations : int
            Number of morphological closing iterations.

        return_probabilities : bool
            If ``True``, :meth:`predict` returns probability maps in addition to
            binary masks.

        compute_features : bool
            If ``True``, :meth:`predict` computes geometric mask features using
            :class:`MeyeMaskFeatures`.

        use_half_precision : bool
            Whether half-precision inference is actually active.

        input_size : int
            Input image size expected by the model. This is inferred from the model
            checkpoint.

        required_frame_size : tuple of int
            Model input size as ``(input_size, input_size)``.

        normalization : str
            Image normalization strategy used internally.

            Current values are:

            ``"imagenet"``
                ImageNet normalization, used for SegFormer-style models.

            ``"none"``
                No ImageNet normalization, used for EfficientFormerV2-style models.

        resize_to_original : bool
            Always ``True``. Output masks and probability maps are resized back to
            the original input-frame size.

        last_result : MeyeResult or None
            Last prediction result returned by :meth:`predict`.

        Methods
        -------
        default_config_path()
            Return the default TOML configuration path.

        default_model_path()
            Return the default model checkpoint path.

        make_default_config()
            Return a fresh copy of the default configuration dictionary.

        reset_config(keep_model=True)
            Reset the TOML configuration to default values.

        get_threshold(class_name)
            Return the current segmentation threshold for one class.

        set_threshold(class_name, value)
            Set the segmentation threshold for one class.

        print_settings()
            Print current thresholds, morphology settings, feature settings, and
            preview settings.

        predict(frame)
            Predict pupil and eye masks from one frame and return a
            :class:`MeyeResult`.

        morphology(mask)
            Apply configured mask-cleaning operations to one binary mask.

        keep_biggest_component(mask)
            Keep only the largest connected component in a binary mask.

        fill_holes_in_mask(mask)
            Fill internal holes in a binary mask.

        overlay(frame, result, alpha=0.45, draw_text=None)
            Draw masks, features, and optional status text on a frame.

        print_preview_commands()
            Print keyboard commands used by the interactive preview.

        handle_preview_key(key)
            Handle one keyboard input during preview.

        preview(cam, window_name="Meye Preview")
            Run interactive MEYELens preview using an external camera-like object.

        preview_camera(camera_index=0, flip_180=False, camera_width=0,
                       camera_height=0, window_name="Meye Preview")
            Run interactive MEYELens preview directly from an OpenCV camera index.

        Prediction Output
        -----------------
        :meth:`predict` returns a :class:`MeyeResult` object with:

        masks : dict
            Binary masks for each predicted class.

            Standard keys are:

            - ``result.masks["pupil"]``
            - ``result.masks["eye"]``

            Each mask is a 2D ``uint8`` array with values ``0`` for background and
            ``255`` for the segmented region.

        probabilities : dict
            Probability maps for each class. This dictionary is empty unless
            ``return_probabilities=True``.

        features : dict
            Geometric features computed from the masks. This dictionary is empty
            unless ``compute_features=True``.

        inference_time_ms : float
            Neural-network inference time in milliseconds.

        inference_fps : float
            Inference speed expressed as frames per second.

        Configuration File
        ------------------
        The default configuration file is:

        ``~/.meyelens/meye_config.toml``

        It contains three main sections.

        ``[model]``

        ``path``
            Path to the model checkpoint.

        ``[inference]``

        ``pupil_threshold``
            Threshold used to binarize the pupil probability map.

        ``eye_threshold``
            Threshold used to binarize the eye probability map.

        ``threshold_step``
            Step used when changing thresholds interactively.

        ``use_morphology``
            Whether to apply morphological opening and closing.

        ``keep_biggest``
            Whether to keep only the largest connected component.

        ``fill_holes``
            Whether to fill internal holes in masks.

        ``morphology_kernel_size``
            Kernel size for morphology.

        ``morphology_open_iterations``
            Number of opening iterations.

        ``morphology_close_iterations``
            Number of closing iterations.

        ``return_probabilities``
            Whether to return probability maps.

        ``compute_features``
            Whether to compute mask features.

        ``use_half_precision``
            Whether to request half-precision inference.

        ``[preview]``

        ``show_pupil_mask``
            Whether to draw the pupil mask in preview overlays.

        ``show_eye_mask``
            Whether to draw the eye mask in preview overlays.

        ``draw_area``
            Whether to draw area text.

        ``draw_centroid``
            Whether to draw centroids.

        ``draw_ellipse``
            Whether to draw fitted ellipses.

        ``draw_axis_points``
            Whether to draw major/minor ellipse axes.

        ``draw_text``
            Whether to draw status text.

        Preview Keyboard Controls
        -------------------------
        q or ESC
            Quit preview.

        + or =
            Increase pupil threshold.

        - or _
            Decrease pupil threshold.

        ]
            Increase eye threshold.

        [
            Decrease eye threshold.

        m
            Toggle morphological opening/closing.

        b
            Toggle keeping only the largest connected component.

        u
            Toggle hole filling.

        p
            Toggle pupil mask overlay.

        o
            Toggle eye mask overlay.

        a
            Toggle area text.

        c
            Toggle centroid drawing.

        e
            Toggle ellipse drawing.

        x
            Toggle ellipse axis drawing.

        f
            Toggle feature computation.

        t
            Toggle status text overlay.

        s
            Print current settings.

        r
            Reset saved TOML configuration while keeping the current model path.

        h
            Print preview help.

        Supported Checkpoint Types
        --------------------------
        ``Meye`` can load several checkpoint styles:

        - full PyTorch modules;
        - state dictionaries with common keys such as ``state_dict``,
          ``model_state_dict``, ``model_state``, ``net``, ``network``, or
          ``weights``;
        - SegFormer-style checkpoints;
        - EfficientFormerV2 two-head FPN checkpoints.

        The architecture is inferred from the checkpoint contents and metadata.

        Notes
        -----
        Input frames can be grayscale or BGR OpenCV images.

        Grayscale frames are internally converted to RGB. BGR frames are converted
        to RGB before preprocessing.

        Output masks and probability maps are resized back to the original frame
        size, so the user usually does not need to handle model input resizing
        manually.

        Thresholds are clipped to the range ``0.01`` to ``0.99`` by
        :meth:`set_threshold`.

    """

    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "meye_default.pt"
    DEFAULT_CONFIG_PATH = Path.home() / ".meyelens" / "meye_config.toml"

    EFFICIENTFORMER_FPN_MODE = "topdown_all_smooth"

    DEFAULT_CONFIG = {
        "model": {
            "path": str(DEFAULT_MODEL_PATH),
        },
        "inference": {
            "pupil_threshold": 0.50,
            "eye_threshold": 0.50,
            "threshold_step": 0.05,
            "use_morphology": True,
            "keep_biggest": True,
            "fill_holes": True,

            "morphology_kernel_size": 5,
            "morphology_open_iterations": 1,
            "morphology_close_iterations": 1,

            "return_probabilities": False,
            "compute_features": True,
            "use_half_precision": False,
        },
        "preview": {
            "show_pupil_mask": True,
            "show_eye_mask": True,
            "draw_area": True,
            "draw_centroid": True,
            "draw_ellipse": True,
            "draw_axis_points": True,
            "draw_text": True,
        },
    }

    class _SegFormerWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hf_model = SegformerForSemanticSegmentation(config)

        def forward(self, x):
            return self.hf_model(pixel_values=x).logits

    class _ModuleOutputWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return Meye._extract_logits(out)

    class _ConvBNAct(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _EfficientFormerV2TwoHeadFPNWrapper(nn.Module):
        def __init__(self, backbone_name, decoder_channels=128):
            super().__init__()

            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "EfficientFormerV2 checkpoints require timm. "
                    "Install it with: pip install timm"
                ) from exc

            self.encoder = timm.create_model(
                backbone_name,
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

            encoder_channels = self.encoder.feature_info.channels()

            self.lateral = nn.ModuleList(
                [
                    nn.Conv2d(int(ch), int(decoder_channels), kernel_size=1)
                    for ch in encoder_channels
                ]
            )

            self.smooth = nn.ModuleList(
                [
                    Meye._ConvBNAct(
                        int(decoder_channels),
                        int(decoder_channels),
                        kernel_size=3,
                        padding=1,
                    )
                    for _ in encoder_channels
                ]
            )

            self.pupil_head = nn.Conv2d(int(decoder_channels), 1, kernel_size=1)
            self.eye_head = nn.Conv2d(int(decoder_channels), 1, kernel_size=1)

        def forward(self, x):
            feats = self.encoder(x)

            laterals = [
                lateral(feat)
                for lateral, feat in zip(self.lateral, feats)
            ]

            p3 = self.smooth[3](laterals[3])

            p2 = laterals[2] + F.interpolate(
                p3,
                size=laterals[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            p2 = self.smooth[2](p2)

            p1 = laterals[1] + F.interpolate(
                p2,
                size=laterals[1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            p1 = self.smooth[1](p1)

            p0 = laterals[0] + F.interpolate(
                p1,
                size=laterals[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            p0 = self.smooth[0](p0)

            pupil_logits = self.pupil_head(p0)
            eye_logits = self.eye_head(p0)

            return torch.cat([pupil_logits, eye_logits], dim=1)

    def __init__(
        self,
        model_path=None,
        config_path=None,
        gpu_device="auto",
        use_half_precision=None,
        reset_config=False,
        verbose=True,
    ):
        """
            Initialize the MEYELens segmentation model.

            The constructor loads the TOML configuration file, resolves the model path,
            selects the inference device, loads the trained PyTorch checkpoint, and
            prepares the runtime settings used by :meth:`predict`, :meth:`overlay`, and
            the interactive preview methods.

            If no configuration file exists, a default one is created automatically.
            The default configuration file is located at:

            ``~/.meyelens/meye_config.toml``

            Parameters
            ----------
            model_path : str or pathlib.Path or None, default=None
                Path to a trained MEYELens model checkpoint.

                If provided, this path overrides the model path stored in the TOML
                configuration file. The new path is also saved to the configuration
                file so that future ``Meye`` instances use the same checkpoint.

                If ``None``, the model path is read from the configuration file.

            config_path : str or pathlib.Path or None, default=None
                Path to the TOML configuration file.

                If ``None``, the default path is used:

                ``~/.meyelens/meye_config.toml``

                If the file does not exist, it is created automatically using
                :attr:`DEFAULT_CONFIG`.

            gpu_device : {"auto", "cpu", int, str}, default="auto"
                Device used for model inference.

                Supported values are:

                ``"auto"``
                    Use CUDA if available, otherwise Apple MPS if available,
                    otherwise CPU.

                ``"cpu"``
                    Force CPU inference.

                int
                    Use a specific CUDA device index, for example ``0`` for
                    ``"cuda:0"``.

                str
                    Any valid PyTorch device string, for example ``"cuda:0"``.

                If CUDA or MPS is requested but unavailable, ``Meye`` prints a
                message and falls back to CPU inference.

            use_half_precision : bool or None, default=None
                Whether to request half-precision inference.

                If ``None``, the value is read from the configuration file.

                If ``True``, half precision is used only when inference runs on CUDA.
                On CPU, inference remains in full precision even if half precision is
                requested.

                If a value is provided, it overrides the value stored in the TOML
                configuration file and is saved to that file.

            reset_config : bool, default=False
                If ``True``, overwrite the TOML configuration file with default settings
                before loading the model.

                This is useful when the saved configuration should be reset to a clean
                state.

            verbose : bool, default=True
                If ``True``, print information about the configuration file, selected
                device, loaded model architecture, input size, output channels, and
                current segmentation settings.

            Attributes Initialized
            ----------------------
            config_path : pathlib.Path
                Path to the active TOML configuration file.

            config : dict
                Configuration dictionary loaded from the TOML file and merged with the
                default configuration.

            model_path : str
                Path to the model checkpoint that will be loaded.

            class_names : list of str
                Output class names. For the standard MEYELens model these are
                ``["pupil", "eye"]``.

            thresholds : dict
                Segmentation thresholds for each class.

            device : torch.device
                Device used for inference.

            model : torch.nn.Module
                Loaded PyTorch model in evaluation mode.

            input_size : int
                Input image size inferred from the checkpoint.

            required_frame_size : tuple of int
                Required model input size as ``(input_size, input_size)``.

            normalization : str
                Image normalization strategy selected from the detected model
                architecture.

            last_result : MeyeResult or None
                Last result returned by :meth:`predict`. Initialized as ``None``.

            Raises
            ------
            FileNotFoundError
                If the model checkpoint file does not exist.

            RuntimeError
                If the checkpoint architecture cannot be detected or if the model state
                cannot be loaded.

            ImportError
                If the checkpoint requires an optional dependency that is not installed,
                for example ``timm`` for EfficientFormerV2 checkpoints.

            Notes
            -----
            The constructor may create or update the TOML configuration file.

            The model architecture is inferred automatically from the checkpoint. The
            current implementation supports full PyTorch modules, SegFormer-style
            checkpoints, and EfficientFormerV2 two-head FPN checkpoints.

            Half precision is applied only after the model is loaded and only when the
            selected device is CUDA.
        """

        self.verbose = bool(verbose)

        self.config_path = Path(config_path) if config_path is not None else self.DEFAULT_CONFIG_PATH
        self.config = self._load_or_create_config(reset=reset_config)

        if model_path is not None:
            self.config["model"]["path"] = str(model_path)
            self._write_config_file(self.config)

        self.model_path = str(self.config["model"]["path"])

        configured_model = Path(self.model_path)
        if (
            not configured_model.exists()
            and configured_model.name == self.DEFAULT_MODEL_PATH.name
            and self.DEFAULT_MODEL_PATH.exists()
        ):
            self.model_path = str(self.DEFAULT_MODEL_PATH)
            self.config["model"]["path"] = self.model_path
            self._write_config_file(self.config)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                "Meye model file was not found.\n"
                f"Current model path: {self.model_path}\n\n"
                "Pass a model path explicitly:\n"
                "    meye = Meye(model_path='models/your_model.pt')\n\n"
                "or edit the TOML configuration:\n"
                f"    {self.config_path}"
            )

        self.class_names = ["pupil", "eye"]

        self.thresholds = {
            "pupil": float(self.config["inference"]["pupil_threshold"]),
            "eye": float(self.config["inference"]["eye_threshold"]),
        }

        self.threshold_step = float(self.config["inference"]["threshold_step"])

        self.use_morphology = bool(self.config["inference"]["use_morphology"])
        self.keep_biggest = bool(self.config["inference"]["keep_biggest"])
        self.fill_holes = bool(self.config["inference"]["fill_holes"])

        self.morphology_kernel_size = int(self.config["inference"]["morphology_kernel_size"])
        self.morphology_open_iterations = int(self.config["inference"]["morphology_open_iterations"])
        self.morphology_close_iterations = int(self.config["inference"]["morphology_close_iterations"])

        self.return_probabilities = bool(self.config["inference"]["return_probabilities"])
        self.compute_features = bool(self.config["inference"]["compute_features"])

        self.show_pupil_mask = bool(self.config["preview"]["show_pupil_mask"])
        self.show_eye_mask = bool(self.config["preview"]["show_eye_mask"])

        self.draw_area = bool(self.config["preview"]["draw_area"])
        self.draw_centroid = bool(self.config["preview"]["draw_centroid"])
        self.draw_ellipse = bool(self.config["preview"]["draw_ellipse"])
        self.draw_axis_points = bool(self.config["preview"]["draw_axis_points"])
        self.draw_text = bool(self.config["preview"]["draw_text"])

        if use_half_precision is None:
            self.use_half_precision_requested = bool(
                self.config["inference"]["use_half_precision"]
            )
        else:
            self.use_half_precision_requested = bool(use_half_precision)
            self.config["inference"]["use_half_precision"] = self.use_half_precision_requested
            self._write_config_file(self.config)

        self.device = self._resolve_device(gpu_device)

        self.use_half_precision = bool(
            self.use_half_precision_requested and self.device.type == "cuda"
        )

        self.normalization = "imagenet"
        self.resize_to_original = True

        if self.verbose:
            print(f"### MEYE ### Config: {self.config_path}")
            print(f"### MEYE ### Using device: {self.device}")

            if self.device.type == "cuda":
                print(f"### MEYE ### GPU: {torch.cuda.get_device_name(self.device)}")

        self.model, inferred_input_size, num_labels = self._load_model(self.model_path)

        self.input_size = int(inferred_input_size)
        self.required_frame_size = (self.input_size, self.input_size)

        self._fix_class_names(num_labels)
        self._ensure_thresholds_for_classes()

        self.last_result = None

        if self.verbose:
            self.print_settings()

    @classmethod
    def default_config_path(cls):
        return cls.DEFAULT_CONFIG_PATH

    @classmethod
    def default_model_path(cls):
        return cls.DEFAULT_MODEL_PATH

    @classmethod
    def make_default_config(cls):
        return copy.deepcopy(cls.DEFAULT_CONFIG)

    def _load_or_create_config(self, reset=False):
        if reset:
            config = self.make_default_config()
            self._write_config_file(config)
            return config

        if not self.config_path.exists():
            config = self.make_default_config()
            self._write_config_file(config)
            return config

        try:
            loaded = toml.load(self.config_path)
        except Exception as exc:
            print(f"### MEYE ### Could not read config file: {exc}")
            print("### MEYE ### Recreating default config.")
            config = self.make_default_config()
            self._write_config_file(config)
            return config

        config = self._merge_config(self.make_default_config(), loaded)
        self._write_config_file(config)

        return config

    def _write_config_file(self, config):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with self.config_path.open("w", encoding="utf-8") as f:
            toml.dump(config, f)

    def _save_config(self):
        self._sync_config_from_runtime()
        self._write_config_file(self.config)

    @staticmethod
    def _merge_config(defaults, loaded):
        merged = copy.deepcopy(defaults)

        for section, values in loaded.items():
            if section not in merged:
                merged[section] = values
                continue

            if isinstance(values, dict) and isinstance(merged[section], dict):
                for key, value in values.items():
                    merged[section][key] = value
            else:
                merged[section] = values

        return merged

    def reset_config(self, keep_model=True):
        current_model_path = self.model_path

        self.config = self.make_default_config()

        if keep_model:
            self.config["model"]["path"] = current_model_path

        self._sync_runtime_from_config()
        self._write_config_file(self.config)

        print(f"### MEYE ### Config reset: {self.config_path}")

    def _sync_runtime_from_config(self):
        self.model_path = str(self.config["model"]["path"])

        self.thresholds["pupil"] = float(self.config["inference"]["pupil_threshold"])
        self.thresholds["eye"] = float(self.config["inference"]["eye_threshold"])

        self.threshold_step = float(self.config["inference"]["threshold_step"])

        self.use_morphology = bool(self.config["inference"]["use_morphology"])
        self.keep_biggest = bool(self.config["inference"]["keep_biggest"])
        self.fill_holes = bool(self.config["inference"]["fill_holes"])

        self.morphology_kernel_size = int(self.config["inference"]["morphology_kernel_size"])
        self.morphology_open_iterations = int(self.config["inference"]["morphology_open_iterations"])
        self.morphology_close_iterations = int(self.config["inference"]["morphology_close_iterations"])

        self.return_probabilities = bool(self.config["inference"]["return_probabilities"])
        self.compute_features = bool(self.config["inference"]["compute_features"])
        self.use_half_precision_requested = bool(self.config["inference"]["use_half_precision"])

        self.show_pupil_mask = bool(self.config["preview"]["show_pupil_mask"])
        self.show_eye_mask = bool(self.config["preview"]["show_eye_mask"])

        self.draw_area = bool(self.config["preview"]["draw_area"])
        self.draw_centroid = bool(self.config["preview"]["draw_centroid"])
        self.draw_ellipse = bool(self.config["preview"]["draw_ellipse"])
        self.draw_axis_points = bool(self.config["preview"]["draw_axis_points"])
        self.draw_text = bool(self.config["preview"]["draw_text"])

    def _sync_config_from_runtime(self):
        self.config["model"]["path"] = str(self.model_path)

        self.config["inference"]["pupil_threshold"] = float(self.thresholds["pupil"])
        self.config["inference"]["eye_threshold"] = float(self.thresholds["eye"])
        self.config["inference"]["threshold_step"] = float(self.threshold_step)

        self.config["inference"]["use_morphology"] = bool(self.use_morphology)
        self.config["inference"]["keep_biggest"] = bool(self.keep_biggest)
        self.config["inference"]["fill_holes"] = bool(self.fill_holes)

        self.config["inference"]["morphology_kernel_size"] = int(self.morphology_kernel_size)
        self.config["inference"]["morphology_open_iterations"] = int(self.morphology_open_iterations)
        self.config["inference"]["morphology_close_iterations"] = int(self.morphology_close_iterations)

        self.config["inference"]["return_probabilities"] = bool(self.return_probabilities)
        self.config["inference"]["compute_features"] = bool(self.compute_features)
        self.config["inference"]["use_half_precision"] = bool(self.use_half_precision_requested)

        self.config["preview"]["show_pupil_mask"] = bool(self.show_pupil_mask)
        self.config["preview"]["show_eye_mask"] = bool(self.show_eye_mask)

        self.config["preview"]["draw_area"] = bool(self.draw_area)
        self.config["preview"]["draw_centroid"] = bool(self.draw_centroid)
        self.config["preview"]["draw_ellipse"] = bool(self.draw_ellipse)
        self.config["preview"]["draw_axis_points"] = bool(self.draw_axis_points)
        self.config["preview"]["draw_text"] = bool(self.draw_text)

    @staticmethod
    def _resolve_device(gpu_device="auto"):
        if gpu_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            if torch.backends.mps.is_available():
                return torch.device("mps")

            return torch.device("cpu")

        if gpu_device == "cpu":
            return torch.device("cpu")

        if isinstance(gpu_device, int):
            if not torch.cuda.is_available():
                print("### MEYE ### CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")

            return torch.device(f"cuda:{gpu_device}")

        gpu_device = str(gpu_device)

        if gpu_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print("### MEYE ### CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")

        if gpu_device == "mps" and not torch.backends.mps.is_available():
            print("### MEYE ### MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        return torch.device(gpu_device)

    def _fix_class_names(self, num_labels):
        num_labels = int(num_labels)

        if len(self.class_names) < num_labels:
            for i in range(len(self.class_names), num_labels):
                self.class_names.append(f"class_{i}")

        if len(self.class_names) > num_labels:
            self.class_names = self.class_names[:num_labels]

    def _ensure_thresholds_for_classes(self):
        for class_name in self.class_names:
            if class_name not in self.thresholds:
                self.thresholds[class_name] = 0.5

    @staticmethod
    def _extract_logits(output):
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("logits", "out", "masks", "pred", "prediction"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        if hasattr(output, "logits") and torch.is_tensor(output.logits):
            return output.logits

        raise RuntimeError("Could not extract logits from model output.")

    @staticmethod
    def _get_state_dict_from_checkpoint(checkpoint):
        if isinstance(checkpoint, dict):
            for key in (
                "model_state",
                "model_state_dict",
                "state_dict",
                "net",
                "network",
                "weights",
            ):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]

            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]

        return checkpoint

    @staticmethod
    def _strip_prefix_if_needed(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict

        cleaned = state_dict
        prefixes = ("module.", "model.", "net.", "network.")

        changed = True

        while changed:
            changed = False
            keys = list(cleaned.keys())

            for prefix in prefixes:
                if keys and all(k.startswith(prefix) for k in keys):
                    cleaned = {
                        k[len(prefix):]: v
                        for k, v in cleaned.items()
                    }
                    changed = True
                    break

        return cleaned

    @staticmethod
    def _load_model_state(model, state_dict, strict=True):
        attempts = [
            state_dict,
            {
                f"hf_model.{k}": v
                for k, v in state_dict.items()
            },
            {
                k[len("hf_model."):] if k.startswith("hf_model.") else k: v
                for k, v in state_dict.items()
            },
        ]

        last_error = None

        for candidate in attempts:
            try:
                model.load_state_dict(candidate, strict=strict)
                return
            except RuntimeError as exc:
                last_error = exc

        raise last_error

    @staticmethod
    def _infer_input_size_from_checkpoint(checkpoint, fallback=224):
        if isinstance(checkpoint, dict):
            containers = [
                checkpoint,
                checkpoint.get("args", {}),
                checkpoint.get("config", {}),
            ]

            for container in containers:
                if not isinstance(container, dict):
                    continue

                for key in ("resolution", "input_size", "img_size", "image_size", "crop_size"):
                    if key in container:
                        value = container[key]

                        if isinstance(value, (tuple, list)):
                            return int(value[0])

                        return int(value)

        return int(fallback)

    @staticmethod
    def _looks_like_segformer(state_dict):
        if not isinstance(state_dict, dict):
            return False

        keys = list(state_dict.keys())

        return any("segformer.encoder.patch_embeddings" in k for k in keys) or any(
            "hf_model.segformer" in k for k in keys
        )

    @staticmethod
    def _infer_segformer_config_from_state_dict(state_dict):
        hidden_sizes = []

        for stage_idx in range(4):
            candidates = [
                f"hf_model.segformer.encoder.patch_embeddings.{stage_idx}.proj.weight",
                f"segformer.encoder.patch_embeddings.{stage_idx}.proj.weight",
            ]

            key = next((k for k in candidates if k in state_dict), None)

            if key is None:
                raise RuntimeError(f"Could not infer SegFormer hidden size for stage {stage_idx}")

            hidden_sizes.append(state_dict[key].shape[0])

        depths = []

        for stage_idx in range(4):
            block_indices = []

            prefixes = [
                f"hf_model.segformer.encoder.block.{stage_idx}.",
                f"segformer.encoder.block.{stage_idx}.",
            ]

            for key in state_dict.keys():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        rest = key[len(prefix):]
                        block_id = rest.split(".")[0]

                        if block_id.isdigit():
                            block_indices.append(int(block_id))

            if not block_indices:
                raise RuntimeError(f"Could not infer SegFormer depth for stage {stage_idx}")

            depths.append(max(block_indices) + 1)

        decoder_key = next(
            (
                k for k in (
                    "hf_model.decode_head.linear_c.0.proj.weight",
                    "decode_head.linear_c.0.proj.weight",
                )
                if k in state_dict
            ),
            None,
        )

        classifier_key = next(
            (
                k for k in (
                    "hf_model.decode_head.classifier.weight",
                    "decode_head.classifier.weight",
                )
                if k in state_dict
            ),
            None,
        )

        if decoder_key is None or classifier_key is None:
            raise RuntimeError("Could not infer SegFormer decode head.")

        decoder_hidden_size = state_dict[decoder_key].shape[0]
        num_labels = state_dict[classifier_key].shape[0]

        config = SegformerConfig(
            num_channels=3,
            num_labels=int(num_labels),
            hidden_sizes=hidden_sizes,
            depths=depths,
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            decoder_hidden_size=int(decoder_hidden_size),
            classifier_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        if num_labels != 2:
            print(f"### MEYE ### Warning: expected 2 output channels, found {num_labels}.")

        return config, int(num_labels)

    @staticmethod
    def _normalize_name(value):
        if value is None:
            return ""

        return str(value).lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _metadata_value(cls, checkpoint, names):
        if not isinstance(checkpoint, dict):
            return None

        containers = [
            checkpoint,
            checkpoint.get("args", {}),
            checkpoint.get("config", {}),
        ]

        for container in containers:
            if not isinstance(container, dict):
                continue

            for name in names:
                if name in container:
                    return container[name]

        return None

    @classmethod
    def _looks_like_efficientformerv2(cls, state_dict, model_path=None, checkpoint=None):
        arch = cls._normalize_name(
            cls._metadata_value(
                checkpoint,
                ("architecture", "arch", "model_type", "model_name", "backbone", "variant"),
            )
        )

        path = cls._normalize_name(model_path)

        if "efficientformerv2" in arch or "efficientformer_v2" in arch:
            return True

        if "effformerv2" in path or "efficientformerv2" in path:
            return True

        if not isinstance(state_dict, dict):
            return False

        keys = [k.lower() for k in state_dict.keys()]

        has_encoder = any(k.startswith("encoder.stem_conv") for k in keys) or any(
            k.startswith("encoder.stages_") for k in keys
        )

        has_two_heads = (
            any(k.startswith("lateral.") for k in keys)
            and any(k.startswith("smooth.") for k in keys)
            and any(k.startswith("pupil_head.") for k in keys)
            and any(k.startswith("eye_head.") for k in keys)
        )

        return has_encoder and has_two_heads

    @classmethod
    def _infer_efficientformerv2_variant(cls, checkpoint=None, model_path=None):
        value = cls._metadata_value(
            checkpoint,
            ("variant", "backbone", "model_name", "architecture", "arch", "model_type"),
        )

        combined = f"{cls._normalize_name(value)} {cls._normalize_name(model_path)}"

        if "s0" in combined:
            return "s0"

        if "s1" in combined:
            return "s1"

        if "s2" in combined:
            return "s2"

        if "efficientformerv2_l" in combined or "efficientformer_v2_l" in combined:
            return "l"

        if "_l_" in combined or combined.endswith("_l") or "_l." in combined:
            return "l"

        raise RuntimeError("Could not infer EfficientFormerV2 variant.")

    @staticmethod
    def _efficientformerv2_timm_name(variant):
        mapping = {
            "s0": "efficientformerv2_s0",
            "s1": "efficientformerv2_s1",
            "s2": "efficientformerv2_s2",
            "l": "efficientformerv2_l",
        }

        variant = str(variant).lower()

        if variant not in mapping:
            raise RuntimeError(f"Unsupported EfficientFormerV2 variant: {variant}")

        return mapping[variant]

    @staticmethod
    def _infer_decoder_channels_from_state_dict(state_dict, fallback=128):
        if not isinstance(state_dict, dict):
            return int(fallback)

        for key, value in state_dict.items():
            if key.endswith("lateral.0.weight") and hasattr(value, "shape"):
                return int(value.shape[0])

        return int(fallback)

    def _build_efficientformerv2_model(self, checkpoint, state_dict, model_path):
        variant = self._infer_efficientformerv2_variant(
            checkpoint=checkpoint,
            model_path=model_path,
        )

        backbone_name = self._efficientformerv2_timm_name(variant)
        decoder_channels = self._infer_decoder_channels_from_state_dict(state_dict)

        if self.verbose:
            print("### MEYE ### Inferred EfficientFormerV2 two-head FPN config:")
            print(f"  variant: {variant}")
            print(f"  backbone: {backbone_name}")
            print(f"  decoder_channels: {decoder_channels}")
            print("  num_labels: 2")
            print("  heads: pupil_head + eye_head")

        model = self._EfficientFormerV2TwoHeadFPNWrapper(
            backbone_name=backbone_name,
            decoder_channels=decoder_channels,
        )

        return model, 2

    def _load_full_torch_module(self, checkpoint, input_size):
        model = None

        if isinstance(checkpoint, nn.Module):
            model = checkpoint

        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model", None), nn.Module):
            model = checkpoint["model"]

        if model is None:
            return None

        model = self._ModuleOutputWrapper(model)
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            x = torch.zeros(1, 3, int(input_size), int(input_size), device=self.device)
            logits = self._extract_logits(model(x))

        if logits.ndim != 4:
            raise RuntimeError(f"Expected model output [B, C, H, W], got {tuple(logits.shape)}")

        return model, int(logits.shape[1])

    def _load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        inferred_input_size = self._infer_input_size_from_checkpoint(checkpoint)

        full_module = self._load_full_torch_module(
            checkpoint,
            input_size=inferred_input_size,
        )

        if full_module is not None:
            model, num_labels = full_module
            architecture = "full_torch_module"
            self.normalization = "imagenet"

        else:
            state_dict = self._get_state_dict_from_checkpoint(checkpoint)
            state_dict = self._strip_prefix_if_needed(state_dict)

            if self._looks_like_segformer(state_dict):
                architecture = "segformer"
                self.normalization = "imagenet"

                config, num_labels = self._infer_segformer_config_from_state_dict(state_dict)
                model = self._SegFormerWrapper(config)
                self._load_model_state(model, state_dict, strict=True)

            elif self._looks_like_efficientformerv2(
                state_dict,
                model_path=model_path,
                checkpoint=checkpoint,
            ):
                architecture = "efficientformerv2_two_head_fpn"
                self.normalization = "none"

                model, num_labels = self._build_efficientformerv2_model(
                    checkpoint,
                    state_dict,
                    model_path,
                )
                self._load_model_state(model, state_dict, strict=True)

            else:
                raise RuntimeError(f"Could not detect checkpoint architecture: {model_path}")

            model.to(self.device)
            model.eval()

        if self.use_half_precision:
            model.half()
            if self.verbose:
                print("### MEYE ### Using half precision inference.")

        if self.verbose:
            print(f"### MEYE ### Loaded architecture: {architecture}")
            print("### MEYE ### Model loaded correctly.")
            print(f"### MEYE ### Using input size: {inferred_input_size}")
            print(f"### MEYE ### Output channels: {num_labels}")
            print(f"### MEYE ### Normalization: {self.normalization}")

        return model, int(inferred_input_size), int(num_labels)

    def get_threshold(self, class_name):
        return float(self.thresholds.get(class_name, 0.5))

    def set_threshold(self, class_name, value):
        self.thresholds[class_name] = float(np.clip(value, 0.01, 0.99))

    def print_settings(self):
        threshold_text = " | ".join(
            [
                f"{name}_thr={self.get_threshold(name):.2f}"
                for name in self.class_names
            ]
        )

        print(
            "### MEYE settings ### "
            f"{threshold_text} | "
            f"normalization={self.normalization} | "
            f"morphology={self.use_morphology} | "
            f"keep_biggest={self.keep_biggest} | "
            f"fill_holes={self.fill_holes} | "
            f"kernel={self.morphology_kernel_size} | "
            f"open_iter={self.morphology_open_iterations} | "
            f"close_iter={self.morphology_close_iterations} | "
            f"features={self.compute_features} | "
            f"pupil_mask={self.show_pupil_mask} | "
            f"eye_mask={self.show_eye_mask} | "
            f"area={self.draw_area} | "
            f"centroid={self.draw_centroid} | "
            f"ellipse={self.draw_ellipse} | "
            f"axis_points={self.draw_axis_points} | "
            f"text={self.draw_text}"
        )

    def _preprocess(self, frame_bgr_or_gray):
        if len(frame_bgr_or_gray.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            frame_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        x = resized.astype(np.float32) / 255.0

        if self.normalization in ("imagenet", "torch", "timm"):
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        elif self.normalization in ("none", "raw", "zero_one", "0_1"):
            pass

        else:
            raise ValueError(
                f"Unknown internal normalization: {self.normalization}. "
                "Expected 'imagenet' or 'none'."
            )

        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1).unsqueeze(0)

        if self.use_half_precision:
            x = x.half()

        return x.to(self.device, non_blocking=True)

    def predict(self, frame):
        """
            Predict pupil and eye masks from one image frame.

            This method runs the loaded MEYELens neural network on a single input frame,
            converts the model output logits to probability maps, thresholds the
            probabilities into binary masks, optionally applies mask post-processing,
            optionally extracts geometric features, and returns all results in a
            :class:`MeyeResult` object.

            Parameters
            ----------
            frame : numpy.ndarray
                Input image.

                The frame can be either:

                - a grayscale image with shape ``(height, width)``;
                - a BGR OpenCV image with shape ``(height, width, 3)``.

                BGR images are internally converted to RGB before model inference.
                Grayscale images are internally converted to RGB.

            Returns
            -------
            MeyeResult
                Prediction result for the input frame.

            Attributes Modified
            -------------------
            last_result : MeyeResult
                Updated to the result returned by this method.

            Raises
            ------
            RuntimeError
                If the model output cannot be interpreted as a tensor of logits with
                shape ``(batch, channels, height, width)``.

            ValueError
                If the input frame has an unsupported shape or if internal
                preprocessing encounters an unsupported normalization mode.
        """

        target_h, target_w = frame.shape[:2]

        x = self._preprocess(frame)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            logits = self.model(x)
            logits = self._extract_logits(logits)

            if logits.ndim != 4:
                raise RuntimeError(f"Expected logits [B, C, H, W], got {tuple(logits.shape)}")

            logits = F.interpolate(
                logits,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

            probs_tensor = torch.sigmoid(logits)[0]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        t1 = time.perf_counter()

        inference_time_ms = (t1 - t0) * 1000.0
        inference_fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else np.inf

        probs_np = probs_tensor.detach().float().cpu().numpy()

        probabilities = {}
        masks = {}

        for idx, class_name in enumerate(self.class_names):
            if idx >= probs_np.shape[0]:
                break

            prob = probs_np[idx].astype(np.float32)
            mask = (prob > self.get_threshold(class_name)).astype(np.uint8) * 255

            if self.use_morphology or self.keep_biggest:
                mask = self.morphology(mask)

            mask_out = cv2.resize(
                mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

            masks[class_name] = mask_out

            if self.return_probabilities:
                probabilities[class_name] = cv2.resize(
                    prob,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )

        features = {}

        if self.compute_features:
            features = MeyeMaskFeatures.compute_all(masks)

        result = MeyeResult(
            probabilities=probabilities,
            masks=masks,
            features=features,
            inference_time_ms=inference_time_ms,
            inference_fps=inference_fps,
        )

        self.last_result = result

        return result

    def morphology(self, mask):
        clean = (mask > 0).astype(np.uint8) * 255

        # 1. open/close morphology.
        if self.use_morphology and self.morphology_kernel_size > 1:
            kernel_size = int(self.morphology_kernel_size)

            if kernel_size % 2 == 0:
                kernel_size += 1

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )

            if self.morphology_open_iterations > 0:
                clean = cv2.morphologyEx(
                    clean,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=self.morphology_open_iterations,
                )

            if self.morphology_close_iterations > 0:
                clean = cv2.morphologyEx(
                    clean,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=self.morphology_close_iterations,
                )

        # 2. keep largest connected component.
        if self.keep_biggest:
            clean = self.keep_biggest_component(clean)

        # 3. fill internal holes.
        if self.fill_holes:
            clean = self.fill_holes_in_mask(clean)

        return clean

    @staticmethod
    def keep_biggest_component(mask):
        binary = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        if num_labels <= 1:
            return np.zeros_like(mask, dtype=np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest_label = 1 + int(np.argmax(areas))

        clean = np.zeros_like(mask, dtype=np.uint8)
        clean[labels == biggest_label] = 255

        return clean

    @staticmethod
    def fill_holes_in_mask(mask):
        """
        Fill internal holes in a binary mask.

        Input and output are uint8 masks:
            0   = background
            255 = object
        """
        binary = (mask > 0).astype(np.uint8)

        h, w = binary.shape[:2]

        # Flood-fill the background from the image border.
        flood = binary.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        cv2.floodFill(
            flood,
            flood_mask,
            seedPoint=(0, 0),
            newVal=1,
        )

        # Pixels not reached by the flood-fill and not already object
        # are internal holes.
        holes = (flood == 0)

        filled = binary.copy()
        filled[holes] = 1

        return (filled * 255).astype(np.uint8)

    def overlay(self, frame, result, alpha=0.45, draw_text=None):
        if draw_text is None:
            draw_text = self.draw_text

        if len(frame.shape) == 2:
            base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            base = frame.copy()

        out = base.copy()

        for idx, class_name in enumerate(self.class_names):
            if class_name not in result.masks:
                continue

            if class_name == "pupil" and not self.show_pupil_mask:
                continue

            if class_name == "eye" and not self.show_eye_mask:
                continue

            mask = result.masks[class_name] > 0

            if mask.shape[:2] != out.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (out.shape[1], out.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

            color = self._class_color(class_name, idx)

            out[mask] = (
                (1.0 - alpha) * out[mask] + alpha * np.array(color)
            ).astype(np.uint8)

        if self.compute_features and result.features:
            MeyeMaskFeatures.draw(
                out,
                result.features,
                draw_area=self.draw_area,
                draw_centroid=self.draw_centroid,
                draw_ellipse=self.draw_ellipse,
                draw_axis_points=self.draw_axis_points,
            )

        if draw_text:
            self._draw_text(out, result)

        return out

    @staticmethod
    def _class_color(class_name, idx):
        if class_name == "pupil":
            return (0, 0, 255)

        if class_name == "eye":
            return (0, 255, 0)

        palette = [
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        return palette[idx % len(palette)]

    def _draw_text(self, image, result):
        lines = [
            f"Inference: {result.inference_fps:.1f} FPS ({result.inference_time_ms:.1f} ms)",
            (
                f"pupil_thr={self.get_threshold('pupil'):.2f} | "
                f"eye_thr={self.get_threshold('eye'):.2f} | "
                f"morph={int(self.use_morphology)} | "
                f"big={int(self.keep_biggest)} | "
                f"holes={int(self.fill_holes)}"
            ),
            (
                f"pupil={int(self.show_pupil_mask)} | "
                f"eye={int(self.show_eye_mask)} | "
                f"area={int(self.draw_area)} | "
                f"cent={int(self.draw_centroid)} | "
                f"ell={int(self.draw_ellipse)} | "
                f"axes={int(self.draw_axis_points)}"
            ),
        ]

        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (20, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def print_preview_commands(self):
        print("")
        print("### MEYE preview commands ###")
        print("  q / ESC : quit preview")
        print("  + / =   : increase pupil threshold")
        print("  - / _   : decrease pupil threshold")
        print("  ]       : increase eye threshold")
        print("  [       : decrease eye threshold")
        print("  m       : toggle morphology")
        print("  b       : toggle keep biggest object")
        print("  u       : toggle fill holes")
        print("  p       : toggle pupil mask overlay")
        print("  o       : toggle eye mask overlay")
        print("  a       : toggle area text")
        print("  c       : toggle centroid")
        print("  e       : toggle ellipse")
        print("  x       : toggle ellipse axis points")
        print("  f       : toggle feature computation")
        print("  t       : toggle text overlay")
        print("  s       : print current settings")
        print("  r       : reset saved TOML configuration")
        print("  h       : print this help")
        print("")

    def handle_preview_key(self, key):
        changed = False

        if key in (ord("q"), 27):
            return False

        if key in (ord("+"), ord("=")):
            self.set_threshold("pupil", self.get_threshold("pupil") + self.threshold_step)
            print(f"### MEYE ### pupil_threshold = {self.get_threshold('pupil'):.2f}")
            changed = True

        elif key in (ord("-"), ord("_")):
            self.set_threshold("pupil", self.get_threshold("pupil") - self.threshold_step)
            print(f"### MEYE ### pupil_threshold = {self.get_threshold('pupil'):.2f}")
            changed = True

        elif key == ord("]"):
            self.set_threshold("eye", self.get_threshold("eye") + self.threshold_step)
            print(f"### MEYE ### eye_threshold = {self.get_threshold('eye'):.2f}")
            changed = True

        elif key == ord("["):
            self.set_threshold("eye", self.get_threshold("eye") - self.threshold_step)
            print(f"### MEYE ### eye_threshold = {self.get_threshold('eye'):.2f}")
            changed = True

        elif key == ord("m"):
            self.use_morphology = not self.use_morphology
            print(f"### MEYE ### use_morphology = {self.use_morphology}")
            changed = True

        elif key == ord("b"):
            self.keep_biggest = not self.keep_biggest
            print(f"### MEYE ### keep_biggest = {self.keep_biggest}")
            changed = True

        elif key == ord("u"):
            self.fill_holes = not self.fill_holes
            print(f"### MEYE ### fill_holes = {self.fill_holes}")
            changed = True

        elif key == ord("p"):
            self.show_pupil_mask = not self.show_pupil_mask
            print(f"### MEYE ### show_pupil_mask = {self.show_pupil_mask}")
            changed = True

        elif key == ord("o"):
            self.show_eye_mask = not self.show_eye_mask
            print(f"### MEYE ### show_eye_mask = {self.show_eye_mask}")
            changed = True

        elif key == ord("a"):
            self.draw_area = not self.draw_area
            print(f"### MEYE ### draw_area = {self.draw_area}")
            changed = True

        elif key == ord("c"):
            self.draw_centroid = not self.draw_centroid
            print(f"### MEYE ### draw_centroid = {self.draw_centroid}")
            changed = True

        elif key == ord("e"):
            self.draw_ellipse = not self.draw_ellipse
            print(f"### MEYE ### draw_ellipse = {self.draw_ellipse}")
            changed = True

        elif key == ord("x"):
            self.draw_axis_points = not self.draw_axis_points
            print(f"### MEYE ### draw_axis_points = {self.draw_axis_points}")
            changed = True

        elif key == ord("f"):
            self.compute_features = not self.compute_features
            print(f"### MEYE ### compute_features = {self.compute_features}")
            changed = True

        elif key == ord("t"):
            self.draw_text = not self.draw_text
            print(f"### MEYE ### draw_text = {self.draw_text}")
            changed = True

        elif key == ord("r"):
            self.reset_config(keep_model=True)
            changed = False

        elif key == ord("s"):
            self.print_settings()

        elif key == ord("h"):
            self.print_preview_commands()

        if changed:
            self._save_config()

        return True

    def preview(self, cam, window_name="Meye Preview"):
        """
            Run an interactive MEYELens segmentation preview.

            This method acquires frames from the camera object, runs
            :meth:`predict` on each frame, draws the prediction overlay with
            :meth:`overlay`, and displays the result in an OpenCV window.

            The preview is useful for checking segmentation quality before recording,
            tuning pupil and eye thresholds, enabling or disabling morphology, and
            selecting which features should be drawn on the overlay.

            Parameters
            ----------
            cam : object
                Camera-like object used for frame acquisition and display.
                A :class:`Camera` instance satisfies these requirements.

            window_name : str, default="Meye Preview"
                Name of the OpenCV window used to display the live preview.

            Returns
            -------
            None
                This method runs an interactive loop until the user presses ``q`` or
                ``ESC``.

            Keyboard Controls
            -----------------
            q or ESC
                Quit preview.

            + or =
                Increase pupil threshold.

            - or _
                Decrease pupil threshold.

            ]
                Increase eye threshold.

            [
                Decrease eye threshold.

            m
                Toggle morphological opening/closing.

            b
                Toggle keeping only the largest connected component.

            u
                Toggle hole filling.

            p
                Toggle pupil mask overlay.

            o
                Toggle eye mask overlay.

            a
                Toggle area text.

            c
                Toggle centroid drawing.

            e
                Toggle ellipse drawing.

            x
                Toggle major/minor ellipse axis drawing.

            f
                Toggle feature computation.

            t
                Toggle status text overlay.

            s
                Print current settings.

            r
                Reset the saved TOML configuration while keeping the current model path.

            h
                Print preview help.

            Notes
            -----
            When a setting is changed interactively, the updated value is saved to the
            active TOML configuration file. This means that threshold and preview
            settings persist across sessions.

            Frames are processed using the current ``Meye`` settings, including
            thresholds, morphology, connected-component filtering, hole filling, feature
            computation, and overlay options.

            If ``cam.get_frame()`` returns ``None``, the method prints a warning and
            continues the preview loop.

        """

        print("### MEYE ### Starting preview.")
        self.print_preview_commands()
        self.print_settings()

        while True:
            frame = cam.get_frame()

            if frame is None:
                print("### MEYE ### Could not read frame.")
                continue

            result = self.predict(frame)
            overlay = self.overlay(frame, result)

            cam.show(overlay, name=window_name)

            key = cv2.waitKey(1) & 0xFF

            if not self.handle_preview_key(key):
                break

        cv2.destroyWindow(window_name)

    def preview_camera(
        self,
        camera_index=0,
        flip_180=False,
        camera_width=0,
        camera_height=0,
        window_name="Meye Preview",
    ):
        """
        Run live preview directly from an OpenCV camera index.

        Example
        -------
            meye = Meye()
            meye.preview_camera(camera_index=0)
        """

        cap = cv2.VideoCapture(camera_index)

        if camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)

        if camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")

        print("### MEYE ### Starting OpenCV preview.")
        self.print_preview_commands()
        self.print_settings()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("### MEYE ### Could not read frame.")
                continue

            if flip_180:
                frame = cv2.flip(frame, -1)

            result = self.predict(frame)
            overlay = self.overlay(frame, result)

            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1) & 0xFF

            if not self.handle_preview_key(key):
                break

        cap.release()
        cv2.destroyWindow(window_name)


class MeyeRecorder:
    """
        Synchronous MEYELens recorder with buffered file writing.

        ``MeyeRecorder`` records pupil and eye measurements from a camera stream
        using a :class:`Meye` segmentation model. For each saved sample, it captures
        one frame, runs MEYELens prediction on that exact frame, extracts geometric
        features, attaches trigger values, and queues one complete row for writing
        to disk.

        The acquisition pipeline is synchronous:

        1. acquire frame from the camera;
        2. run :meth:`Meye.predict`;
        3. extract pupil and eye features;
        4. attach trigger values;
        5. queue the completed row for file writing.

        Only the final disk-writing step is asynchronous, through
        :class:`BufferedFileWriter`. This design keeps camera acquisition,
        prediction, feature extraction, and trigger assignment aligned to the same
        frame, while avoiding direct disk I/O inside the time-sensitive acquisition
        loop.

        captures one camera frame, predicts pupil and eye masks for that same frame,
        extracts features from that result, attaches ``trg1=1`` to that row, and
        queues the row for writing.

        Parameters
        ----------
        cam : object
            Camera-like object used to acquire frames.
            A :class:`Camera` instance satisfies these requirements.

        meye : Meye
            MEYE segmentation object.

        filename : str, default="meye"
            Base filename used for the output recording file.

            A timestamp is prepended and the file extension is added by
            :class:`BufferedFileWriter`.

        path_to_file : str or pathlib.Path or None, default=None
            Directory where the recording file is saved.

            If ``None``, files are saved in:

            ``~/Documents/meyeDATA``

        buffer_size : int, default=1000
            Maximum number of completed data rows allowed in the writer queue.

            If the acquisition loop produces rows faster than the background writer
            can save them, the queue grows until it reaches this limit. The behavior
            after that depends on ``overflow_policy``.

        sep : str, default=";"
            Field separator used in the output text file.

        show_preview : bool, default=False
            If ``True``, show a live MEYELens overlay while frames are recorded.

            Keeping this ``False`` gives better acquisition performance.

        preview_window_name : str, default="Meye Recording"
            Name of the OpenCV window used when ``show_preview=True``.

        save_inference_timing : bool, default=True
            If ``True``, include ``inference_ms`` and ``inference_fps`` columns in
            the output file.

        flush_every : int, default=30
            Flush the file buffer after this many written rows.

            This parameter is passed to :class:`BufferedFileWriter`.

        flush_interval : float, default=0.0
            Flush the file buffer at least every this many seconds.

            If ``0.0``, time-based flushing is disabled and flushing is controlled
            by ``flush_every`` and by final close.

        overflow_policy : {"raise", "block", "drop"}, default="raise"
            Behavior when the writer queue is full.

            ``"raise"``
                Raise an error. This is recommended for experiments because it makes
                missed writes explicit.

            ``"block"``
                Wait until queue space is available. This prevents data loss but can
                pause acquisition.

            ``"drop"``
                Drop the new row and print a warning. This avoids blocking but loses
                data.

        Attributes
        ----------
        cam : object
            Camera-like object used by the recorder.

        meye : Meye
            MEYELens inference object used by the recorder.

        filename : str
            Base output filename.

        path_to_file : pathlib.Path
            Output directory.

        buffer_size : int
            Maximum number of queued rows for the background writer.

        sep : str
            Field separator used in the output file.

        show_preview : bool
            Whether to show the live prediction overlay during recording.

        preview_window_name : str
            Preview window name used when ``show_preview=True``.

        save_inference_timing : bool
            Whether inference timing columns are included in the output file.

        flush_every : int
            Row-based file flush interval.

        flush_interval : float
            Time-based file flush interval.

        overflow_policy : str
            Queue overflow behavior used by the file writer.

        writer : BufferedFileWriter or None
            Active buffered file writer. Set by :meth:`start` and cleared by
            :meth:`stop`.

        time_start : float or None
            Reference time from ``time.perf_counter()`` used to compute relative
            timestamps.

        running : bool
            Whether recording is currently active.

        frame : numpy.ndarray or None
            Last acquired camera frame.

        result : MeyeResult or None
            Last MEYELens prediction result.

        frame_index : int
            Index of the next frame to be recorded.

        trg1, trg2, ..., trg9 : int or float
            Last trigger values passed to :meth:`save_frame`.

        headers : list of str
            Column names written to the recording file.

        Methods
        -------
        start(metadata=None)
            Open the output file, write metadata and headers, and start recording.
            ``metadata`` is an optional dictionary of user-defined information about
            the recording session, such as subject ID, session name, task condition,
            stimulus parameters, experiment name, or notes. These values are written at
            the top of the output file as comment lines before the data table.

        stop()
            Stop recording, flush queued rows, and close the file writer.

        close()
            Release camera resources.

        close_all()
            Stop recording and release all resources.

        save_frame(trg1=0, ..., trg9=0)
            Capture one frame, run prediction, attach triggers, write one data row,
            and return the row data as a dictionary.

        get_data()
            Capture one frame and return extracted data without writing to disk.

        get_last_data()
            Return the extracted data from the last acquired frame.

        preview()
            Run the interactive MEYELens preview using the recorder camera.

        Output File
        -----------
        The output file is a separator-separated text file created by
        :class:`BufferedFileWriter`. It contains:

        - metadata lines at the top, prefixed with ``#``;
        - one header row;
        - one data row per successful call to :meth:`save_frame`.

        The default separator is ``";"``.

        The output columns include:

        ``frame_index``
            Sequential frame index.

        ``t_call``
            Time, in seconds, when :meth:`save_frame` was called, relative to
            :meth:`start`.

        ``t_frame``
            Time, in seconds, immediately after frame acquisition.

        ``t_pred``
            Time, in seconds, immediately after MEYELens prediction.

        ``pupil_x``, ``pupil_y``, ``pupil_area``
            Pupil centroid and area.

        ``pupil_major_diameter``, ``pupil_minor_diameter``,
        ``pupil_orientation_deg``, ``pupil_ovality``, ``pupil_eccentricity``
            Pupil ellipse features.

        ``pupil_major_p1_x``, ``pupil_major_p1_y``,
        ``pupil_major_p2_x``, ``pupil_major_p2_y``
            Endpoints of the pupil major axis.

        ``pupil_minor_p1_x``, ``pupil_minor_p1_y``,
        ``pupil_minor_p2_x``, ``pupil_minor_p2_y``
            Endpoints of the pupil minor axis.

        ``eye_x``, ``eye_y``, ``eye_area``
            Eye centroid and area.

        ``eye_major_diameter``, ``eye_minor_diameter``,
        ``eye_orientation_deg``, ``eye_ovality``, ``eye_eccentricity``
            Eye ellipse features.

        ``eye_major_p1_x``, ``eye_major_p1_y``,
        ``eye_major_p2_x``, ``eye_major_p2_y``
            Endpoints of the eye major axis.

        ``eye_minor_p1_x``, ``eye_minor_p1_y``,
        ``eye_minor_p2_x``, ``eye_minor_p2_y``
            Endpoints of the eye minor axis.

        ``inference_ms``, ``inference_fps``
            Inference timing columns. Present only when
            ``save_inference_timing=True``.

        ``trg1`` ... ``trg9``
            User-defined trigger channels.

        Metadata
        --------
        :meth:`start` writes user metadata together with automatically generated
        MEYELens metadata.

        User metadata can include experiment information such as subject, session,
        condition, task name, or stimulus parameters.

        Automatic metadata may include:

        - model path;
        - MEYELens configuration path;
        - class names;
        - pupil and eye thresholds;
        - normalization mode;
        - morphology settings;
        - input size;
        - probability/feature settings;
        - half-precision setting;
        - camera crop;
        - recorder mode;
        - writer buffer settings.

        Notes
        -----
        ``MeyeRecorder`` requires feature computation. If ``meye.compute_features``
        is ``False`` when :meth:`start` is called, the recorder enables it
        automatically because the output columns depend on mask features.

        Trigger values belong to the exact frame acquired during the corresponding
        :meth:`save_frame` call. This makes trigger assignment explicit and
        frame-locked.

        If frame acquisition fails, :meth:`save_frame` prints a warning and returns
        ``None``. No row is written for that call.

        The output uses image coordinates where ``x`` is the column coordinate and
        ``y`` is the row coordinate. Internally, :class:`MeyeMaskFeatures` stores
        centroids and ellipse points as ``(row, col)``, and the recorder converts
        them to ``x``/``y`` columns.
    """

    def __init__(
        self,
        cam,
        meye,
        filename="meye",
        path_to_file=None,
        buffer_size=1000,
        sep=";",
        show_preview=False,
        preview_window_name="Meye Recording",
        save_inference_timing=True,
        flush_every=30,
        flush_interval=0.0,
        overflow_policy="raise",
    ):
        """
            Initialize a MEYELens recorder.

            The constructor stores the camera and MEYELens objects, prepares output-file
            settings, initializes trigger values, and creates the list of output column
            names. It does not start recording and does not open the output file.
            Recording begins only after calling :meth:`start`.

            Parameters
            ----------
            cam : object
                Camera-like object used to acquire frames.

                The object must provide:

                ``cam.get_frame()``
                    Acquire and return one image frame as a NumPy array, or return
                    ``None`` if acquisition fails.

                ``cam.close()``
                    Release camera resources.

                If ``show_preview=True``, the object must also provide:

                ``cam.show(image, name=...)``
                    Display an image in an OpenCV window.

                ``cam.refresh(delay_ms=...)``
                    Refresh OpenCV windows.

                A :class:`Camera` instance satisfies these requirements.

            meye : Meye
                MEYELens inference object used to predict pupil and eye masks.

                The object must provide:

                ``meye.predict(frame)``
                    Run segmentation on one frame and return a :class:`MeyeResult`.

                ``meye.overlay(frame, result, draw_text=True)``
                    Create a preview image showing masks, features, and optional status
                    text.

            filename : str, default="meye"
                Base name of the output recording file.

                A timestamp is prepended and the file extension is added by
                :class:`BufferedFileWriter`. For example, ``filename="subject01"`` may
                produce a file such as ``20260603_121530-subject01.txt``.

            path_to_file : str or pathlib.Path or None, default=None
                Directory where the recording file will be saved.

                If ``None``, the default output directory is:

                ``~/Documents/meyeDATA``

                The directory is created automatically when recording starts.

            buffer_size : int, default=1000
                Maximum number of completed data rows that can wait in the asynchronous
                writer queue.

                If the acquisition loop produces rows faster than the background writer
                can save them, the queue grows up to this limit. The behavior when the
                queue is full is controlled by ``overflow_policy``.

            sep : str, default=";"
                Separator used between columns in the output text file.

            show_preview : bool, default=False
                If ``True``, display a live MEYELens overlay during recording.

                For best recording performance, especially when high frame rates are
                needed, keep this set to ``False``.

            preview_window_name : str, default="Meye Recording"
                Name of the OpenCV preview window used when ``show_preview=True``.

            save_inference_timing : bool, default=True
                If ``True``, include ``inference_ms`` and ``inference_fps`` columns in
                the output file.

            flush_every : int, default=30
                Flush the file buffer after this many written rows.

                This controls how often already written rows are pushed from Python's
                file buffer to the operating system. It does not control how often rows
                are removed from the writer queue.

            flush_interval : float, default=0.0
                Flush the file buffer at least every this many seconds.

                If ``0.0``, time-based flushing is disabled. The file is always flushed
                when :meth:`stop` or :meth:`close_all` closes the writer.

            overflow_policy : {"raise", "block", "drop"}, default="raise"
                Behavior when the asynchronous writer queue is full.

                ``"raise"``
                    Raise a ``RuntimeError``. Recommended for experiments, because it
                    makes missed writes explicit.

                ``"block"``
                    Wait until queue space is available. This prevents data loss but can
                    pause the acquisition loop.

                ``"drop"``
                    Drop the new row and print a warning. This avoids blocking but can
                    lose data.

            Attributes Initialized
            ----------------------
            cam : object
                Camera-like object passed to the constructor.

            meye : Meye
                MEYELens inference object passed to the constructor.

            path_to_file : pathlib.Path
                Output directory.

            writer : BufferedFileWriter or None
                Buffered writer used during recording. Initialized as ``None`` and
                created by :meth:`start`.

            time_start : float or None
                Reference timestamp for the recording clock. Initialized as ``None`` and
                set by :meth:`start`.

            running : bool
                Whether recording is active. Initialized as ``False``.

            frame : numpy.ndarray or None
                Last acquired frame. Initialized as ``None``.

            result : MeyeResult or None
                Last MEYELens prediction result. Initialized as ``None``.

            frame_index : int
                Index of the next frame to be recorded. Initialized as ``0``.

            trg1, trg2, ..., trg9 : int
                Trigger channels. All initialized to ``0``.

            headers : list of str
                Output column names generated by the internal header builder.

            Notes
            -----
            The constructor does not create the output file. The file is created when
            :meth:`start` is called.

            Trigger values are provided later to :meth:`save_frame`. They are stored in
            the output row corresponding to the exact frame acquired during that call.

            """
        self.cam = cam
        self.meye = meye

        self.filename = filename

        if path_to_file is None:
            self.path_to_file = Path.home() / "Documents" / "meyeDATA"
        else:
            self.path_to_file = Path(path_to_file).expanduser()

        self.buffer_size = int(buffer_size)
        self.sep = sep

        self.show_preview = bool(show_preview)
        self.preview_window_name = preview_window_name
        self.save_inference_timing = bool(save_inference_timing)

        self.flush_every = int(flush_every)
        self.flush_interval = float(flush_interval)
        self.overflow_policy = overflow_policy

        self.writer = None
        self.time_start = None
        self.running = False

        self.frame = None
        self.result = None
        self.frame_index = 0

        self.trg1 = 0
        self.trg2 = 0
        self.trg3 = 0
        self.trg4 = 0
        self.trg5 = 0
        self.trg6 = 0
        self.trg7 = 0
        self.trg8 = 0
        self.trg9 = 0

        self.headers = self._make_headers()

    # ============================================================
    # Headers and metadata
    # ============================================================

    def _make_headers(self):
        headers = [
            "frame_index",

            # Timing
            "t_call",
            "t_frame",
            "t_pred",

            # Pupil centroid and area
            "pupil_x",
            "pupil_y",
            "pupil_area",

            # Pupil ellipse
            "pupil_major_diameter",
            "pupil_minor_diameter",
            "pupil_orientation_deg",
            "pupil_ovality",
            "pupil_eccentricity",

            # Pupil ellipse major axis points
            "pupil_major_p1_x",
            "pupil_major_p1_y",
            "pupil_major_p2_x",
            "pupil_major_p2_y",

            # Pupil ellipse minor axis points
            "pupil_minor_p1_x",
            "pupil_minor_p1_y",
            "pupil_minor_p2_x",
            "pupil_minor_p2_y",

            # Eye centroid and area
            "eye_x",
            "eye_y",
            "eye_area",

            # Eye ellipse
            "eye_major_diameter",
            "eye_minor_diameter",
            "eye_orientation_deg",
            "eye_ovality",
            "eye_eccentricity",

            # Eye ellipse major axis points
            "eye_major_p1_x",
            "eye_major_p1_y",
            "eye_major_p2_x",
            "eye_major_p2_y",

            # Eye ellipse minor axis points
            "eye_minor_p1_x",
            "eye_minor_p1_y",
            "eye_minor_p2_x",
            "eye_minor_p2_y",
        ]

        if self.save_inference_timing:
            headers += [
                "inference_ms",
                "inference_fps",
            ]

        headers += [
            "trg1",
            "trg2",
            "trg3",
            "trg4",
            "trg5",
            "trg6",
            "trg7",
            "trg8",
            "trg9",
        ]

        return headers

    def _make_meye_metadata(self):
        metadata = {}

        if hasattr(self.meye, "model_path"):
            metadata["model_path"] = str(self.meye.model_path)

        if hasattr(self.meye, "config_path"):
            metadata["meye_config_path"] = str(self.meye.config_path)

        if hasattr(self.meye, "class_names"):
            metadata["class_names"] = ",".join(map(str, self.meye.class_names))

        if hasattr(self.meye, "thresholds"):
            for name, value in self.meye.thresholds.items():
                metadata[f"{name}_threshold"] = value

        for attr in [
            "normalization",
            "use_morphology",
            "keep_biggest",
            "compute_features",
            "morphology_kernel_size",
            "morphology_open_iterations",
            "morphology_close_iterations",
            "input_size",
            "return_probabilities",
            "use_half_precision",
        ]:
            if hasattr(self.meye, attr):
                metadata[attr] = getattr(self.meye, attr)

        if hasattr(self.cam, "crop"):
            metadata["camera_crop"] = self.cam.crop

        metadata["recorder"] = "MeyeRecorder"
        metadata["recorder_mode"] = (
            "synchronous_frame_prediction_trigger_assignment__buffered_file_writing"
        )
        metadata["buffer_size"] = self.buffer_size
        metadata["flush_every"] = self.flush_every
        metadata["flush_interval"] = self.flush_interval
        metadata["overflow_policy"] = self.overflow_policy

        return metadata

    # ============================================================
    # Start / stop
    # ============================================================

    def start(self, metadata=None):
        """
            Start recording and open the output file.

            This method creates a :class:`BufferedFileWriter`, writes metadata and the
            column header to the output file, resets the recording clock and frame
            index, and sets ``self.running=True``.

            Parameters
            ----------
            metadata : dict or None, default=None
                Optional user-defined metadata describing the recording session.

                Metadata are written at the top of the output file as comment lines
                before the data table. They are useful for storing information needed to
                interpret the recording later, such as:

                - subject or participant ID;
                - session name or number;
                - experimental condition;
                - task name;
                - stimulus parameters;
                - acquisition notes.

                Example:

                ``{"subject": "S01", "session": "pre", "condition": "baseline"}``

                If ``None``, only automatically generated MEYELens and recorder
                metadata are written.

            Returns
            -------
            None

            Notes
            -----
            If recording is already active, this method prints a message and returns
            without creating a new file.

            User metadata are combined with automatically generated metadata describing
            the MEYELens model, configuration file, class names, thresholds,
            morphology settings, camera crop, and writer settings.

            If ``self.meye.compute_features`` is ``False``, this method enables it
            automatically because the recorder output columns require mask features.

            The recording clock starts when this method is called. The timestamps saved
            by :meth:`save_frame` are relative to this start time.
        """

        if self.running:
            print("### MEYE RECORDER ### Already running.")
            return

        if metadata is None:
            metadata = {}

        if hasattr(self.meye, "compute_features") and not self.meye.compute_features:
            print(
                "### MEYE RECORDER ### Enabling meye.compute_features=True "
                "because recording requires feature columns."
            )
            self.meye.compute_features = True

        metadata = dict(metadata)
        metadata.update(self._make_meye_metadata())

        self.writer = BufferedFileWriter(
            self.path_to_file,
            filename=self.filename,
            buffer_size=self.buffer_size,
            headers=self.headers,
            metadata=metadata,
            sep=self.sep,
            flush_every=self.flush_every,
            flush_interval=self.flush_interval,
            overflow_policy=self.overflow_policy,
        )

        self.time_start = time.perf_counter()
        self.frame_index = 0
        self.running = True

        print("### MEYE RECORDER ### Start.")
        print(
            "### MEYE RECORDER ### Frame acquisition, prediction, and trigger "
            "assignment are synchronous."
        )
        print("### MEYE RECORDER ### File writing is buffered.")

    def stop(self):
        """
            Stop recording and close the output file.

            This method stops the active recording, drains the writer queue, flushes the
            file buffer, closes the output file, and sets ``self.running=False``.

            Returns
            -------
            None

            Notes
            -----
            This method does not release the camera. To release camera resources, call
            :meth:`close`. To both stop recording and release the camera, call
            :meth:`close_all`.

            It is safe to call this method even when no writer is active. In that case,
            the method simply sets ``self.running=False``.
        """

        if self.writer is not None:
            print("### MEYE RECORDER ### Closing file writer.")
            self.writer.close()
            self.writer = None

        self.running = False

    def close(self):
        """
            Release camera resources.

            This method calls ``self.cam.close()`` when a camera object is available.

            Returns
            -------
            None

            Notes
            -----
            This method does not stop the file writer by itself. If recording is active,
            call :meth:`stop` before releasing the camera, or use :meth:`close_all`.
        """

        if self.cam is not None:
            print("### MEYE RECORDER ### Closing camera.")
            self.cam.close()

    def close_all(self):
        """
            Stop recording and release all recorder resources.

            This convenience method calls :meth:`stop` and then :meth:`close`. It is the
            recommended cleanup method at the end of an experiment because it ensures
            that queued rows are written to disk and that camera resources are released.

            Returns
            -------
            None

            Notes
            -----
            This method flushes and closes the writer through :meth:`stop`, then closes
            the camera through :meth:`close`.
        """

        self.stop()
        self.close()
        print("### MEYE RECORDER ### Closed.")

    # ============================================================
    # Recording
    # ============================================================

    def save_frame(
        self,
        trg1=0,
        trg2=0,
        trg3=0,
        trg4=0,
        trg5=0,
        trg6=0,
        trg7=0,
        trg8=0,
        trg9=0,
    ):
        """
            Capture one frame, run MEYELens prediction, and save one data row.

            This method is the main recording operation of :class:`MeyeRecorder`.
            It acquires one camera frame, runs :meth:`Meye.predict` on that exact frame,
            extracts pupil and eye features, attaches the provided trigger values, and
            queues one complete row for writing to the output file.

            Frame acquisition, prediction, feature extraction, and trigger assignment
            are synchronous. Only the final file-writing step is buffered through
            :class:`BufferedFileWriter`.

            Parameters
            ----------
            trg1, trg2, trg3, trg4, trg5, trg6, trg7, trg8, trg9 : int or float, default=0
                User-defined trigger values saved in the output row.

                Trigger values are arbitrary numeric codes defined by the experiment.
                They can be used to mark stimulus onset, trial number, condition,
                response events, task phase, or other experimental variables.

                The trigger values belong to the frame acquired during this exact
                :meth:`save_frame` call.

            Returns
            -------
            dict or None
                Dictionary containing the extracted data for the saved frame, or
                ``None`` if recording has not started or if frame acquisition fails.

                The returned dictionary includes timing values, pupil features, eye
                features, and inference timing. Trigger values are written to the file
                row but are not included in the returned dictionary.

            Notes
            -----
            Recording must be started with :meth:`start` before calling this method. If
            the recorder is not running, the method prints a warning and returns
            ``None``.

            The method records three timestamps, all relative to the time when
            :meth:`start` was called:

            ``t_call``
                Time when :meth:`save_frame` was called.

            ``t_frame``
                Time immediately after camera frame acquisition.

            ``t_pred``
                Time immediately after MEYELens prediction.

            If frame acquisition fails and ``cam.get_frame()`` returns ``None``, no row
            is written and the method returns ``None``.

            If ``show_preview=True``, the method also draws and displays a MEYELens
            overlay after writing the row. Live preview can reduce acquisition
            performance.

            The output file row is queued for asynchronous writing. Calling this method
            does not necessarily mean that the row has already been physically written
            to disk, but :meth:`stop` and :meth:`close_all` flush and close the writer.

        """

        if not self.running or self.time_start is None or self.writer is None:
            print("### MEYE RECORDER ### Recording not started.")
            return None

        current_frame_index = self.frame_index

        self.trg1 = trg1
        self.trg2 = trg2
        self.trg3 = trg3
        self.trg4 = trg4
        self.trg5 = trg5
        self.trg6 = trg6
        self.trg7 = trg7
        self.trg8 = trg8
        self.trg9 = trg9

        triggers = [
            trg1,
            trg2,
            trg3,
            trg4,
            trg5,
            trg6,
            trg7,
            trg8,
            trg9,
        ]

        t_call = time.perf_counter() - self.time_start

        self.frame = self.cam.get_frame()
        t_frame = time.perf_counter() - self.time_start

        if self.frame is None:
            print("### MEYE RECORDER ### Could not acquire frame.")
            return None

        self.result = self.meye.predict(self.frame)
        t_pred = time.perf_counter() - self.time_start

        data = self._result_to_dict(
            result=self.result,
            frame_index=current_frame_index,
            t_call=t_call,
            t_frame=t_frame,
            t_pred=t_pred,
        )

        row = self._dict_to_row(data, triggers)
        self.writer.write_sv(row)

        if self.show_preview:
            overlay = self.meye.overlay(
                self.frame,
                self.result,
                draw_text=True,
            )
            self.cam.show(overlay, name=self.preview_window_name)
            self.cam.refresh(delay_ms=1)

        self.frame_index += 1

        return data

    def get_data(self):
        """
            Acquire one new frame and return extracted data without writing to disk.

            This method captures a new frame from the camera, runs :meth:`Meye.predict`,
            extracts pupil and eye features, and returns the result as a dictionary.
            Unlike :meth:`save_frame`, it does not queue a row for file writing and does
            not save anything to the output file.

            Returns
            -------
            dict or None
                Dictionary containing timing values, pupil features, eye features, and
                inference timing for the newly acquired frame.

                Returns ``None`` if frame acquisition fails.

            Notes
            -----
            This method can be used before or during recording.

            If recording has already been started with :meth:`start`, timestamps are
            relative to the recording start time.

            If recording has not been started, a temporary local reference time is used
            for the returned timestamps.

            This method updates ``self.frame`` and ``self.result`` with the newly
            acquired frame and prediction result.

            No trigger values are attached and no data are written to disk.
        """

        if self.time_start is None:
            local_start = time.perf_counter()
        else:
            local_start = self.time_start

        t_call = time.perf_counter() - local_start

        self.frame = self.cam.get_frame()
        t_frame = time.perf_counter() - local_start

        if self.frame is None:
            return None

        self.result = self.meye.predict(self.frame)
        t_pred = time.perf_counter() - local_start

        return self._result_to_dict(
            result=self.result,
            frame_index=self.frame_index,
            t_call=t_call,
            t_frame=t_frame,
            t_pred=t_pred,
        )

    def get_last_data(self):
        """
            Return extracted data from the most recent prediction.

            This method does not acquire a new frame and does not run a new MEYELens
            prediction. It reuses ``self.result``, which is the most recent result
            produced by :meth:`save_frame` or :meth:`get_data`, and converts it to the
            same dictionary format used for recorder output.

            Returns
            -------
            dict or None
                Dictionary containing pupil features, eye features, and inference timing
                from the most recent prediction.

                Returns ``None`` if no prediction result is available yet.

            Notes
            -----
            Because no new frame is acquired, this method is useful when the latest
            result should be inspected or reused without changing the camera/model
            state.

            The returned timing fields ``t_call``, ``t_frame``, and ``t_pred`` are set
            to ``math.nan`` because no new acquisition or prediction is performed.

            The returned ``frame_index`` corresponds to the last saved frame when used
            after :meth:`save_frame`. If no frame has been saved yet, it returns ``0``.
        """

        if self.result is None:
            return None

        last_index = max(0, self.frame_index - 1)

        return self._result_to_dict(
            result=self.result,
            frame_index=last_index,
            t_call=math.nan,
            t_frame=math.nan,
            t_pred=math.nan,
        )

    # ============================================================
    # Data extraction
    # ============================================================

    def _dict_to_row(self, data, triggers):
        row = [
            data["frame_index"],

            data["t_call"],
            data["t_frame"],
            data["t_pred"],

            data["pupil_x"],
            data["pupil_y"],
            data["pupil_area"],

            data["pupil_major_diameter"],
            data["pupil_minor_diameter"],
            data["pupil_orientation_deg"],
            data["pupil_ovality"],
            data["pupil_eccentricity"],

            data["pupil_major_p1_x"],
            data["pupil_major_p1_y"],
            data["pupil_major_p2_x"],
            data["pupil_major_p2_y"],

            data["pupil_minor_p1_x"],
            data["pupil_minor_p1_y"],
            data["pupil_minor_p2_x"],
            data["pupil_minor_p2_y"],

            data["eye_x"],
            data["eye_y"],
            data["eye_area"],

            data["eye_major_diameter"],
            data["eye_minor_diameter"],
            data["eye_orientation_deg"],
            data["eye_ovality"],
            data["eye_eccentricity"],

            data["eye_major_p1_x"],
            data["eye_major_p1_y"],
            data["eye_major_p2_x"],
            data["eye_major_p2_y"],

            data["eye_minor_p1_x"],
            data["eye_minor_p1_y"],
            data["eye_minor_p2_x"],
            data["eye_minor_p2_y"],
        ]

        if self.save_inference_timing:
            row += [
                data["inference_ms"],
                data["inference_fps"],
            ]

        row += list(triggers)

        return row

    def _result_to_dict(
        self,
        result,
        frame_index,
        t_call=math.nan,
        t_frame=math.nan,
        t_pred=math.nan,
    ):
        features = getattr(result, "features", {}) or {}

        pupil = features.get("pupil", {})
        eye = features.get("eye", {})

        pupil_centroid = pupil.get("centroid", (math.nan, math.nan))
        eye_centroid = eye.get("centroid", (math.nan, math.nan))

        # MeyeMaskFeatures uses (row, col). Save as x=col, y=row.
        pupil_y = self._safe_get_tuple(pupil_centroid, 0)
        pupil_x = self._safe_get_tuple(pupil_centroid, 1)

        eye_y = self._safe_get_tuple(eye_centroid, 0)
        eye_x = self._safe_get_tuple(eye_centroid, 1)

        pupil_ellipse = pupil.get("ellipse", {})
        eye_ellipse = eye.get("ellipse", {})

        data = {
            "frame_index": frame_index,

            "t_call": t_call,
            "t_frame": t_frame,
            "t_pred": t_pred,

            "pupil_x": pupil_x,
            "pupil_y": pupil_y,
            "pupil_area": pupil.get("area", math.nan),

            "eye_x": eye_x,
            "eye_y": eye_y,
            "eye_area": eye.get("area", math.nan),

            "inference_ms": getattr(result, "inference_time_ms", math.nan),
            "inference_fps": getattr(result, "inference_fps", math.nan),
        }

        data.update(self._ellipse_to_dict("pupil", pupil_ellipse))
        data.update(self._ellipse_to_dict("eye", eye_ellipse))

        return data

    def _ellipse_to_dict(self, prefix, ellipse):
        major_axis = ellipse.get("major_axis", {})
        minor_axis = ellipse.get("minor_axis", {})

        major_p1 = major_axis.get("p1", (math.nan, math.nan))
        major_p2 = major_axis.get("p2", (math.nan, math.nan))

        minor_p1 = minor_axis.get("p1", (math.nan, math.nan))
        minor_p2 = minor_axis.get("p2", (math.nan, math.nan))

        return {
            f"{prefix}_major_diameter": ellipse.get("major_diameter", math.nan),
            f"{prefix}_minor_diameter": ellipse.get("minor_diameter", math.nan),
            f"{prefix}_orientation_deg": ellipse.get("orientation_deg", math.nan),
            f"{prefix}_ovality": ellipse.get("ovality", math.nan),
            f"{prefix}_eccentricity": ellipse.get("eccentricity", math.nan),

            f"{prefix}_major_p1_x": self._safe_get_tuple(major_p1, 1),
            f"{prefix}_major_p1_y": self._safe_get_tuple(major_p1, 0),
            f"{prefix}_major_p2_x": self._safe_get_tuple(major_p2, 1),
            f"{prefix}_major_p2_y": self._safe_get_tuple(major_p2, 0),

            f"{prefix}_minor_p1_x": self._safe_get_tuple(minor_p1, 1),
            f"{prefix}_minor_p1_y": self._safe_get_tuple(minor_p1, 0),
            f"{prefix}_minor_p2_x": self._safe_get_tuple(minor_p2, 1),
            f"{prefix}_minor_p2_y": self._safe_get_tuple(minor_p2, 0),
        }

    @staticmethod
    def _safe_get_tuple(value, index, default=math.nan):
        try:
            return value[index]
        except Exception:
            return default

    # ============================================================
    # Preview
    # ============================================================

    def preview(self):
        """
            Run the interactive MEYELens preview using the recorder camera.

            This is a convenience method that calls :meth:`Meye.preview` with the
            camera object stored in the recorder. It can be used to check the camera
            image, inspect segmentation quality, tune pupil and eye thresholds, and
            adjust preview options before starting an acquisition.

            Returns
            -------
            None
                This method runs an interactive preview loop until the user quits the
                preview with ``q`` or ``ESC``.

            Notes
            -----
            This method does not start recording and does not write data to disk.

            It uses the same camera object stored in ``self.cam`` and the same
            MEYELens model stored in ``self.meye``.

            Any threshold or preview setting changed during the interactive preview is
            handled by :meth:`Meye.preview`. In the standard :class:`Meye`
            implementation, these changes are saved to the active TOML configuration
            file.
        """
        self.meye.preview(self.cam)


class EyeVideoRecorder:
    """
        Raw eye-video recorder with synchronized frame triggers.

        ``EyeVideoRecorder`` records camera frames to a video file and writes a
        synchronized trigger table with one row per saved frame. It is intended for
        experiments where the raw eye movie should be stored without running
        MEYELens segmentation online.

        The class uses the same :class:`Camera` object used by the rest of the
        MEYELens acquisition workflow, but it does not require a :class:`Meye`
        instance and it never calls :meth:`Meye.predict`. This makes it useful when:

        - the goal is to archive raw eye videos for later offline analysis;
        - online segmentation would be too slow or unnecessary;
        - stimulus or behavioural triggers must be synchronized to each video frame;
        - raw image acquisition should be kept separate from pupil-feature extraction.

        Output Structure
        ----------------
        By default, recordings are saved in the standard MEYELens data folder:

        ``~/Documents/meyeDATA``

        Each call to :meth:`start` creates one timestamped session folder:

        ``YYYYMMDD_HHMMSS-eye_video``

        Inside the session folder, two files are created with predictable names:

        ``video.mp4``
            Raw eye video. The frames are the images returned by
            :meth:`Camera.get_frame`, therefore any active camera crop, vertical flip,
            or undistortion setting is already applied.

        ``triggers.csv``
            Comma-separated frame table. The file contains optional metadata lines
            prefixed by ``#``, followed by one row per written video frame.

        Trigger Table
        -------------
        The trigger file contains the following columns:

        ``frame_index``
            Zero-based index of the written video frame.

        ``t_call``
            Time, in seconds, at which :meth:`save_frame` was called, relative to
            recording start.

        ``t_frame``
            Time, in seconds, immediately before the frame is written to disk,
            relative to recording start.

        ``video_time_sec``
            Nominal video timestamp computed as ``frame_index / fps``.

        ``trg1`` ... ``trg9``
            User-defined trigger channels. These can be used to store stimulus
            onsets, trial numbers, response codes, condition labels, or any other
            frame-wise experimental variable.

        One trigger row is written only when a frame is successfully acquired and
        written to the video file. Failed camera acquisitions therefore do not create
        trigger rows.

        Parameters
        ----------
        cam : Camera-like object
            Camera object used for frame acquisition. A :class:`Camera` instance is
            the intended input. The object must provide a :meth:`get_frame` method.

        filename : str, default="eye_video"
            Session name appended to the timestamped output folder. For example,
            ``filename="subject01"`` creates a folder similar to:

            ``20260616_153000-subject01``

        path_to_file : str or pathlib.Path or None, default=None
            Root directory in which the timestamped session folder is created.

            If ``None``, the default root is:

            ``~/Documents/meyeDATA``

        fps : int or float, default=30
            Frame rate written in the video file header. This should match the
            intended acquisition frame rate. The actual acquisition timing is stored
            separately in ``t_call`` and ``t_frame``.

        codec : str, default="mp4v"
            Four-character OpenCV video codec passed to
            ``cv2.VideoWriter_fourcc``. The default creates MP4-compatible output
            on most systems.

        trigger_filename : str, default="triggers.csv"
            Name of the trigger table inside the session folder.

        video_filename : str, default="video.mp4"
            Name of the video file inside the session folder.

        show_preview : bool, default=False
            If ``True``, show a live OpenCV preview of the raw frame during
            recording. The preview is for monitoring only and does not alter the
            saved video.

        preview_window_name : str, default="Eye video recorder"
            Name of the OpenCV preview window.

        draw_preview_text : bool, default=True
            If ``True`` and ``show_preview=True``, draw the current frame index and
            a compact trigger summary on the preview window. The text is not written
            to the saved video.

        Attributes
        ----------
        session_dir : pathlib.Path or None
            Timestamped output folder created by :meth:`start`.

        video_path : pathlib.Path or None
            Full path to the saved video file.

        trigger_path : pathlib.Path or None
            Full path to the saved trigger CSV file.

        frame_index : int
            Number of frames written so far. Also used as the index of the next
            frame to be saved.

        started : bool
            Whether a recording is currently active.

        closed : bool
            Whether the current recording resources have been closed.

        Methods
        -------
        start(metadata=None)
            Create the session folder, open the video and trigger files, write
            metadata, and start recording.

        save_frame(trg1=0, ..., trg9=0)
            Acquire one frame from the camera, append it to the video, and write
            the corresponding trigger row.

        stop()
            Flush and close the trigger file, release the video writer, and close
            the preview window if needed.

        close_all()
            Alias-like convenience method for :meth:`stop`, matching the style of
            :class:`MeyeRecorder`.

        Notes
        -----
        ``EyeVideoRecorder`` deliberately stores raw frames only. It does not compute
        pupil masks, eye masks, pupil area, centroids, ellipses, or inference timing.
        To extract pupil features later, the saved video can be processed offline
        with :class:`Meye` or with the MEYELens offline analysis GUI.

        The video timestamp stored in ``video_time_sec`` is nominal and depends on
        the requested ``fps``. For experimental timing, prefer the measured
        ``t_call`` and ``t_frame`` columns in ``triggers.csv``.

        The class can be used as a context manager, but the camera is not released
        by :meth:`stop` or :meth:`close_all`. Camera ownership remains with the code
        that created the :class:`Camera` object, typically a ``with Camera(...)`` block.

        Example
        -------
        .. code-block:: python

            from meyelens import Camera, EyeVideoRecorder

            with Camera(camera_index=0, framerate=30) as cam:
                cam.select_roi()

                recorder = EyeVideoRecorder(
                    cam=cam,
                    filename="subject01_baseline",
                    fps=30,
                    show_preview=True,
                )

                try:
                    recorder.start(
                        metadata={
                            "subject": "S01",
                            "condition": "baseline",
                        }
                    )

                    for frame in range(300):
                        recorder.save_frame(
                            trg1=1 if frame == 100 else 0,
                            trg2=frame,
                        )

                finally:
                    recorder.close_all()
    """

    def __init__(
        self,
        cam,
        filename="eye_video",
        path_to_file=None,
        fps=30,
        codec="mp4v",
        trigger_filename="triggers.csv",
        video_filename="video.mp4",
        show_preview=False,
        preview_window_name="Eye video recorder",
        draw_preview_text=True,
    ):
        self.cam = cam
        self.filename = str(filename)
        self.fps = float(fps)
        self.codec = str(codec)
        self.trigger_filename = str(trigger_filename)
        self.video_filename = str(video_filename)

        self.show_preview = bool(show_preview)
        self.preview_window_name = str(preview_window_name)
        self.draw_preview_text = bool(draw_preview_text)

        if path_to_file is None:
            self.root = Path.home() / "Documents" / "meyeDATA"
        else:
            self.root = Path(path_to_file).expanduser()

        self.session_dir = None
        self.video_path = None
        self.trigger_path = None

        self.video_writer = None
        self.trigger_file_handle = None
        self.trigger_headers = None

        self.frame_index = 0
        self.started = False
        self.closed = True

        self.t0 = None

    def start(self, metadata=None):
        """
        Start the raw video recording.

        Parameters
        ----------
        metadata : dict or None
            Optional metadata written at the top of ``triggers.csv`` as comment
            lines.
        """
        if self.started:
            raise RuntimeError("EyeVideoRecorder.start() called more than once.")

        metadata = dict(metadata or {})

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = self.filename if self.filename else "eye_video"

        self.session_dir = self.root / f"{timestamp}-{safe_name}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.session_dir / self.video_filename
        self.trigger_path = self.session_dir / self.trigger_filename

        frame = self.cam.get_frame()

        if frame is None:
            raise RuntimeError("Could not acquire first frame. Recording not started.")

        height, width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            (width, height),
            True,
        )

        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            raise RuntimeError(f"Could not create video file: {self.video_path}")

        metadata.update(
            {
                "recorder": "EyeVideoRecorder",
                "video_file": self.video_filename,
                "trigger_file": self.trigger_filename,
                "fps": self.fps,
                "width": width,
                "height": height,
                "camera_crop": getattr(self.cam, "crop", None),
                "camera_index": getattr(self.cam, "camera_index", ""),
            }
        )

        headers = [
            "frame_index",
            "t_call",
            "t_frame",
            "video_time_sec",
            "trg1",
            "trg2",
            "trg3",
            "trg4",
            "trg5",
            "trg6",
            "trg7",
            "trg8",
            "trg9",
        ]

        self._open_trigger_file(metadata=metadata, headers=headers)

        self.frame_index = 0
        self.t0 = time.perf_counter()
        self.started = True
        self.closed = False

        print(f"## EyeVideoRecorder ## Session folder: {self.session_dir}")
        print(f"## EyeVideoRecorder ## Video: {self.video_path}")
        print(f"## EyeVideoRecorder ## Triggers: {self.trigger_path}")

    def _open_trigger_file(self, metadata, headers):
        """
        Open the fixed-name trigger CSV file and write metadata/header rows.
        """
        self.trigger_file_handle = self.trigger_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )

        for key, value in metadata.items():
            self.trigger_file_handle.write(f"# {key}: {value}\n")

        self.trigger_file_handle.write(",".join(headers) + "\n")
        self.trigger_file_handle.flush()

        self.trigger_headers = list(headers)

    def save_frame(
        self,
        trg1=0,
        trg2=0,
        trg3=0,
        trg4=0,
        trg5=0,
        trg6=0,
        trg7=0,
        trg8=0,
        trg9=0,
    ):
        """
        Acquire one camera frame, write it to video, and append one trigger row.

        Returns
        -------
        bool
            ``True`` if a frame was acquired and written.
            ``False`` if the camera returned ``None``.
        """
        if not self.started:
            raise RuntimeError("Call EyeVideoRecorder.start() before save_frame().")

        if self.closed:
            raise RuntimeError("EyeVideoRecorder.save_frame() called after close().")

        t_call = time.perf_counter() - self.t0

        frame = self.cam.get_frame()

        if frame is None:
            return False

        self._write_frame_and_triggers(
            frame=frame,
            t_call=t_call,
            trg1=trg1,
            trg2=trg2,
            trg3=trg3,
            trg4=trg4,
            trg5=trg5,
            trg6=trg6,
            trg7=trg7,
            trg8=trg8,
            trg9=trg9,
        )

        return True

    def _write_frame_and_triggers(
        self,
        frame,
        t_call,
        trg1=0,
        trg2=0,
        trg3=0,
        trg4=0,
        trg5=0,
        trg6=0,
        trg7=0,
        trg8=0,
        trg9=0,
    ):
        """
        Write an already-acquired frame and append its trigger row.
        """
        if frame.ndim == 2:
            frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_to_write = frame

        t_frame = time.perf_counter() - self.t0
        video_time_sec = self.frame_index / self.fps if self.fps > 0 else np.nan

        self.video_writer.write(frame_to_write)

        row = [
            self.frame_index,
            f"{float(t_call):.9f}",
            f"{float(t_frame):.9f}",
            f"{float(video_time_sec):.9f}",
            trg1,
            trg2,
            trg3,
            trg4,
            trg5,
            trg6,
            trg7,
            trg8,
            trg9,
        ]

        self.trigger_file_handle.write(",".join(map(str, row)) + "\n")

        if self.frame_index % 50 == 0:
            self.trigger_file_handle.flush()

        if self.show_preview:
            self._show_preview(
                frame=frame_to_write,
                trg1=trg1,
                trg2=trg2,
                trg3=trg3,
            )

        self.frame_index += 1

    def _show_preview(self, frame, trg1=0, trg2=0, trg3=0):
        """
        Show a raw-frame preview during recording.
        """
        preview = frame.copy()

        if self.draw_preview_text:
            cv2.putText(
                preview,
                f"frame={self.frame_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"trg1={trg1} trg2={trg2} trg3={trg3}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.preview_window_name, preview)
        cv2.waitKey(1)

    def stop(self):
        """
        Stop recording and close video/trigger files.
        """
        if self.closed:
            return

        if self.trigger_file_handle is not None:
            self.trigger_file_handle.flush()
            self.trigger_file_handle.close()
            self.trigger_file_handle = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.show_preview:
            try:
                cv2.destroyWindow(self.preview_window_name)
            except cv2.error:
                pass

        self.closed = True
        self.started = False

    def close_all(self):
        """
        Stop recording and release recorder-owned resources.

        The camera itself is not closed here if it is managed by a ``with Camera(...)``
        block.
        """
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_all()
        return False


class MeyeGazeCalibrator:
    """
        In-memory gaze calibration model for MEYELens.

        ``MeyeGazeCalibrator`` learns a mapping from MEYELens pupil/eye features to
        two-dimensional screen or stimulus coordinates. It is designed for simple
        one-camera gaze estimation workflows where the user presents calibration
        targets, collects MEYELens features while the participant fixates each
        target, fits a regression model, and then predicts gaze position from new
        MEYELens results.

        The calibrator works entirely in memory. It does not save calibration
        samples, fitted models, or predictions to disk.

        The typical workflow is:

        1. create a calibrator with the size of the coordinate space;
        2. call :meth:`next_target` to obtain the next calibration target;
        3. show that target on screen;
        4. collect samples with :meth:`add_sample`;
        5. repeat for all targets;
        6. fit the calibration model with :meth:`fit`;
        7. predict gaze from new MEYELens results with :meth:`predict_xy`.

        Parameters
        ----------
        screen_width : float
            Width of the calibration coordinate space.

            Target coordinates are centered around zero. For example, if
            ``screen_width=1.8``, generated x coordinates span approximately
            ``-0.9`` to ``+0.9``.

        screen_height : float
            Height of the calibration coordinate space.

            Target coordinates are centered around zero. For example, if
            ``screen_height=1.8``, generated y coordinates span approximately
            ``-0.9`` to ``+0.9``.

        num_points : int, default=9
            Number of calibration target positions to generate.

            If ``num_points <= 5``, targets are selected from the center and four
            corners.

            If ``num_points > 5`` and ``random_points=False``, targets are sampled
            from a regular grid.

            If ``random_points=True``, the target set contains the center, four
            corners, and additional random positions.

        random_points : bool, default=False
            If ``True``, generate additional random target positions after the
            center and corner positions.

            If ``False``, generate target positions from a regular grid.

        random_seed : int or None, default=None
            Seed used by NumPy's random number generator.

            Set this value to make the target order and random positions
            reproducible.

        feature_columns : list of str or None, default=None
            Feature names used as predictors for the calibration model.

            If ``None``, :attr:`DEFAULT_FEATURE_COLUMNS` is used.

            The default features include normalized pupil displacement relative to
            the eye ellipse, normalized pupil and eye positions, normalized pupil
            size, pupil and eye ovality, and eye-orientation sine/cosine features.

        model_type : {"ridge_poly", "ridge", "mlp"}, default="ridge_poly"
            Default regression model used by :meth:`fit`.

            ``"ridge_poly"``
                Ridge regression on polynomial features. This is the recommended
                default for small calibration datasets.

            ``"ridge"``
                Linear ridge regression.

            ``"mlp"``
                Multi-layer perceptron regression. This can model nonlinear
                relationships but usually requires more calibration data.

        Attributes
        ----------
        screen_width : float
            Width of the calibration coordinate space.

        screen_height : float
            Height of the calibration coordinate space.

        num_points : int
            Number of generated calibration targets.

        random_points : bool
            Whether random calibration positions are used.

        random_seed : int or None
            Random seed used for target generation and MLP initialization.

        rng : numpy.random.Generator
            Random number generator used to create and shuffle calibration targets.

        feature_columns : list of str
            Names of features used as model predictors.

        target_columns : list of str
            Names of target coordinate columns. These are ``["target_x",
            "target_y"]``.

        model_type : str
            Current/default model type used for fitting.

        positions : list of tuple
            Generated calibration target positions as ``(x, y)`` tuples.

        current_index : int
            Index of the current target in ``positions``.

        current_target : dict or None
            Current target dictionary, or ``None`` if no target is active.

        current_target_id : int or None
            ID of the current target.

        current_target_sample_index : int
            Sample counter within the current target. This is useful for discarding
            the first samples after a target appears.

        samples : list of dict
            Calibration samples collected in memory.

            Each sample contains target coordinates, timing information, extracted
            features, and any optional extra fields.

        model : sklearn estimator or None
            Fitted calibration model. Set by :meth:`fit`.

        target_scaler : sklearn.preprocessing.StandardScaler
            Scaler used internally to standardize target coordinates during model
            fitting.

        is_fitted : bool
            Whether the calibration model has been fitted.

        fit_info : dict
            Summary information from the most recent fit, including model type,
            number of samples, selected features, MSE, and R2.

        Methods
        -------
        reset_targets(reshuffle=True)
            Reset the target sequence and optionally reshuffle/regenerate target
            positions.

        next_target()
            Advance to the next calibration target and return its target
            dictionary.

        get_current_target()
            Return the current target dictionary.

        extract_features_from_result(result)
            Extract gaze-calibration features from a :class:`MeyeResult`.

        add_sample(result, timestamp=None, extra=None)
            Extract features from a :class:`MeyeResult` and add one calibration
            sample for the current target.

        add_features(feature_dict, target=None, timestamp=None, extra=None)
            Add one already-extracted feature dictionary to the calibration sample
            list.

        clear_samples()
            Remove all collected calibration samples.

        get_samples()
            Return a copy of the collected sample list.

        prepare_training_data(skip_samples=5, aggregate="median",
                              min_pupil_area=None, min_eye_area=None)
            Convert collected samples into predictor and target arrays for model
            fitting.

        fit(model_type=None, degree=2, alpha=1.0, skip_samples=5,
            aggregate="median", min_pupil_area=None, min_eye_area=None,
            mlp_hidden_layer_sizes=(32, 16), mlp_max_iter=2000)
            Fit the gaze calibration model.

        predict_features(features)
            Predict gaze coordinates from a feature dictionary or feature array.

        predict_result(result)
            Predict gaze coordinates directly from a :class:`MeyeResult`.

        predict_xy(result)
            Predict gaze coordinates directly from a :class:`MeyeResult` and return
            scalar ``x`` and ``y`` values.

        Coordinate System
        -----------------
        Calibration targets are generated in a coordinate space centered around
        zero.

        For example, in PsychoPy ``units="norm"``, a common setup is:

        >>> screen_width = 1.8
        >>> screen_height = 1.8

        This produces target coordinates approximately within:

        ``x = -0.9 ... +0.9``

        ``y = -0.9 ... +0.9``

        The predicted gaze coordinates are returned in the same coordinate system
        as the calibration targets.

        Feature Strategy
        ----------------
        The default feature set is based on the relative geometry of the pupil and
        eye masks.

        A key idea is to compute the pupil position relative to the eye ellipse and
        rotate this displacement into the eye ellipse coordinate system:

        ``pupil_dx_eye_axis_norm``

        ``pupil_dy_eye_axis_norm``

        These features are normalized by the eye ellipse size and can remain useful
        when the eye is tilted, rotated, or vertically oriented.

        The default feature set also includes normalized pupil/eye position,
        normalized pupil size, pupil and eye ovality, and sine/cosine encoding of
        eye orientation.

        Notes
        -----
        This class is intended for lightweight gaze calibration, not for storing a
        permanent gaze model. If calibration data or models need to be reused across
        sessions, they must be saved externally by the user.

        The default model, ``"ridge_poly"``, is usually more stable than an MLP when
        only a small number of calibration targets is available.

        Samples with missing or invalid features are removed during
        :meth:`prepare_training_data`.

        By default, :meth:`prepare_training_data` skips the first few samples from
        each target using ``skip_samples=5``. This is useful because the eye may
        still be moving immediately after a new calibration target appears.

    """

    DEFAULT_FEATURE_COLUMNS = [
        "pupil_dx_eye_axis_norm",
        "pupil_dy_eye_axis_norm",
        "pupil_x_norm_frame",
        "pupil_y_norm_frame",
        "eye_x_norm_frame",
        "eye_y_norm_frame",
        "pupil_area_norm",
        "pupil_major_diameter_norm",
        "pupil_minor_diameter_norm",
        "pupil_ovality",
        "eye_ovality",
        "eye_orientation_sin",
        "eye_orientation_cos",
    ]

    def __init__(
        self,
        screen_width,
        screen_height,
        num_points=9,
        random_points=False,
        random_seed=None,
        feature_columns=None,
        model_type="ridge_poly",
    ):
        """
            Initialize an in-memory MEYELens gaze calibrator.

            The constructor defines the calibration coordinate space, generates the
            calibration target positions, selects the feature columns used for model
            fitting, and initializes the internal sample list and regression model
            state.

            Parameters
            ----------
            screen_width : float
                Width of the calibration coordinate space.

                Target coordinates are centered around zero. For example, if
                ``screen_width=1.8``, generated x coordinates span approximately
                ``-0.9`` to ``+0.9``.

            screen_height : float
                Height of the calibration coordinate space.

                Target coordinates are centered around zero. For example, if
                ``screen_height=1.8``, generated y coordinates span approximately
                ``-0.9`` to ``+0.9``.

            num_points : int, default=9
                Number of calibration target positions to generate.

                If ``num_points <= 5``, the target list is built from the center and
                four corners.

                If ``num_points > 5`` and ``random_points=False``, target positions are
                selected from a regular grid.

                If ``random_points=True``, the target list contains the center, four
                corners, and additional random positions.

            random_points : bool, default=False
                If ``True``, generate additional random calibration target positions.

                If ``False``, generate calibration targets from a regular grid when
                ``num_points > 5``.

            random_seed : int or None, default=None
                Seed for NumPy's random number generator.

                Use a fixed value to make target generation and target order
                reproducible.

            feature_columns : list of str or None, default=None
                Names of the feature columns used as predictors during calibration
                model fitting.

                If ``None``, :attr:`DEFAULT_FEATURE_COLUMNS` is used.

            model_type : {"ridge_poly", "ridge", "mlp"}, default="ridge_poly"
                Default model type used by :meth:`fit`.

                ``"ridge_poly"``
                    Ridge regression on polynomial features. Recommended when the
                    number of calibration points is limited.

                ``"ridge"``
                    Linear ridge regression.

                ``"mlp"``
                    Multi-layer perceptron regression. More flexible, but usually
                    requires more calibration samples.

            Attributes Initialized
            ----------------------
            screen_width : float
                Width of the calibration coordinate space.

            screen_height : float
                Height of the calibration coordinate space.

            num_points : int
                Number of generated calibration targets.

            random_points : bool
                Whether random target positions are used.

            random_seed : int or None
                Random seed used for target generation.

            rng : numpy.random.Generator
                Random generator used to generate and shuffle target positions.

            feature_columns : list of str
                Feature names used as model predictors.

            target_columns : list of str
                Target coordinate names, initialized as ``["target_x", "target_y"]``.

            model_type : str
                Default regression model type.

            positions : list of tuple
                Generated calibration target positions as ``(x, y)`` tuples.

            current_index : int
                Index of the current calibration target. Initialized to ``-1`` because
                no target is active yet.

            current_target : dict or None
                Current target dictionary. Initialized as ``None``.

            current_target_id : int or None
                ID of the current target. Initialized as ``None``.

            current_target_sample_index : int
                Sample counter within the current target. Initialized as ``0``.

            samples : list
                In-memory list of collected calibration samples. Initialized as an empty
                list.

            model : sklearn estimator or None
                Fitted calibration model. Initialized as ``None``.

            target_scaler : sklearn.preprocessing.StandardScaler
                Scaler used internally to standardize target coordinates before model
                fitting.

            is_fitted : bool
                Whether the calibration model has been fitted. Initialized as
                ``False``.

            fit_info : dict
                Summary dictionary from the most recent model fit. Initialized as an
                empty dictionary.

            Notes
            -----
            The generated target coordinates use the same coordinate system that should
            later be used for gaze predictions. For example, if calibration targets are
            shown in PsychoPy ``units="norm"``, predicted gaze coordinates will also be
            in that same coordinate system.

            The constructor only prepares the calibrator. Use :meth:`next_target` to
            iterate through calibration positions, :meth:`add_sample` or
            :meth:`add_features` to collect data, and :meth:`fit` to train the gaze
            model.
        """

        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)

        self.num_points = int(num_points)
        self.random_points = bool(random_points)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        if feature_columns is None:
            feature_columns = self.DEFAULT_FEATURE_COLUMNS

        self.feature_columns = list(feature_columns)
        self.target_columns = ["target_x", "target_y"]

        self.model_type = model_type

        self.positions = self._generate_positions()

        self.current_index = -1
        self.current_target = None
        self.current_target_id = None
        self.current_target_sample_index = 0

        self.samples = []

        self.model = None
        self.target_scaler = StandardScaler()

        self.is_fitted = False
        self.fit_info = {}

    # ============================================================
    # Target generation
    # ============================================================

    def _generate_positions(self):
        """
        Generate calibration positions.

        If num_points <= 5:
            use center + four corners.

        If num_points > 5 and random_points=False:
            use a regular grid.

        If random_points=True:
            use center + corners + random points.
        """

        half_w = self.screen_width / 2.0
        half_h = self.screen_height / 2.0

        fixed_points = [
            (0.0, 0.0),
            (-half_w, half_h),
            (half_w, half_h),
            (-half_w, -half_h),
            (half_w, -half_h),
        ]

        if self.num_points <= 5:
            positions = fixed_points[:self.num_points]
            self.rng.shuffle(positions)
            return positions

        if self.random_points:
            n_random = self.num_points - len(fixed_points)
            random_positions = []

            for _ in range(n_random):
                x = self.rng.uniform(-half_w, half_w)
                y = self.rng.uniform(-half_h, half_h)
                random_positions.append((float(x), float(y)))

            positions = fixed_points + random_positions
            self.rng.shuffle(positions)
            return positions

        grid_size = int(math.ceil(math.sqrt(self.num_points)))

        xs = np.linspace(-half_w, half_w, grid_size)
        ys = np.linspace(-half_h, half_h, grid_size)

        positions = []

        for y in ys:
            for x in xs:
                positions.append((float(x), float(y)))

        positions = positions[:self.num_points]
        self.rng.shuffle(positions)

        return positions

    def reset_targets(self, reshuffle=True):
        """
            Reset the calibration target sequence.

            This method resets the current target state so that calibration can start
            again from the first target. Optionally, it regenerates and reshuffles the
            target positions.

            Parameters
            ----------
            reshuffle : bool, default=True
                If ``True``, regenerate the calibration target positions by calling the
                internal target-generation method.

                If ``False``, keep the existing target positions and only reset the
                current target index.

            Returns
            -------
            None

            Attributes Modified
            -------------------
            positions : list of tuple
                Regenerated if ``reshuffle=True``.

            current_index : int
                Reset to ``-1``.

            current_target : dict or None
                Reset to ``None``.

            current_target_id : int or None
                Reset to ``None``.

            current_target_sample_index : int
                Reset to ``0``.

            Notes
            -----
            This method does not delete collected calibration samples. To remove
            existing samples, call :meth:`clear_samples`.
        """
        if reshuffle:
            self.positions = self._generate_positions()

        self.current_index = -1
        self.current_target = None
        self.current_target_id = None
        self.current_target_sample_index = 0

    def next_target(self):
        """
            Advance to the next calibration target.

            This method moves the calibrator to the next target position in the current
            target sequence and stores it as ``self.current_target``. The returned
            target dictionary can be used to draw the calibration stimulus on screen.

            Returns
            -------
            dict or None
                Dictionary describing the next calibration target, or ``None`` if all
                targets have already been completed.

                The dictionary contains:

                ``target_id`` : int
                    Integer identifier of the target within the current sequence.

                ``target_x`` : float
                    Horizontal target coordinate.

                ``target_y`` : float
                    Vertical target coordinate.

            Attributes Modified
            -------------------
            current_index : int
                Incremented by one.

            current_target : dict or None
                Set to the new target dictionary, or ``None`` when the sequence is
                finished.

            current_target_id : int or None
                Set to the current target index, or ``None`` when the sequence is
                finished.

            current_target_sample_index : int
                Reset to ``0`` every time a new target is selected.

            Notes
            -----
            Target coordinates are expressed in the coordinate system defined by
            ``screen_width`` and ``screen_height`` when the calibrator was initialized.

            This method does not draw anything on screen. The user is responsible for
            presenting the returned target position in the experiment interface.

        """

        self.current_index += 1
        self.current_target_sample_index = 0

        if self.current_index >= len(self.positions):
            self.current_target = None
            self.current_target_id = None
            return None

        x, y = self.positions[self.current_index]

        self.current_target_id = self.current_index

        self.current_target = {
            "target_id": self.current_target_id,
            "target_x": float(x),
            "target_y": float(y),
        }

        return self.current_target

    def get_current_target(self):
        """
            Return the current calibration target.

            Returns
            -------
            dict or None
                Current target dictionary, or ``None`` if no target is currently active.

                When a target is active, the dictionary contains:

                ``target_id`` : int
                    Integer identifier of the current target.

                ``target_x`` : float
                    Horizontal target coordinate.

                ``target_y`` : float
                    Vertical target coordinate.

            Notes
            -----
            This method does not advance the calibration sequence. Use
            :meth:`next_target` to move to the next target.
        """
        return self.current_target

    # ============================================================
    # Feature extraction
    # ============================================================

    @staticmethod
    def extract_features_from_result(result):
        """
        Extract gaze features from a Meye result.

        Expected Meye features:
            result.features["pupil"]["centroid"] = (row, col)
            result.features["eye"]["centroid"] = (row, col)
            result.features["pupil"]["ellipse"]
            result.features["eye"]["ellipse"]
        """

        features = {}

        result_features = getattr(result, "features", {}) or {}

        pupil = result_features.get("pupil", {})
        eye = result_features.get("eye", {})

        pupil_centroid = pupil.get("centroid", (np.nan, np.nan))
        eye_centroid = eye.get("centroid", (np.nan, np.nan))

        pupil_y = MeyeGazeCalibrator._safe_tuple_get(pupil_centroid, 0)
        pupil_x = MeyeGazeCalibrator._safe_tuple_get(pupil_centroid, 1)

        eye_y = MeyeGazeCalibrator._safe_tuple_get(eye_centroid, 0)
        eye_x = MeyeGazeCalibrator._safe_tuple_get(eye_centroid, 1)

        pupil_area = pupil.get("area", np.nan)
        eye_area = eye.get("area", np.nan)

        pupil_ellipse = pupil.get("ellipse", {})
        eye_ellipse = eye.get("ellipse", {})

        pupil_major = pupil_ellipse.get("major_diameter", np.nan)
        pupil_minor = pupil_ellipse.get("minor_diameter", np.nan)
        pupil_orientation = pupil_ellipse.get("orientation_deg", np.nan)
        pupil_ovality = pupil_ellipse.get("ovality", np.nan)
        pupil_eccentricity = pupil_ellipse.get("eccentricity", np.nan)

        eye_major = eye_ellipse.get("major_diameter", np.nan)
        eye_minor = eye_ellipse.get("minor_diameter", np.nan)
        eye_orientation = eye_ellipse.get("orientation_deg", np.nan)
        eye_ovality = eye_ellipse.get("ovality", np.nan)
        eye_eccentricity = eye_ellipse.get("eccentricity", np.nan)

        features["pupil_x"] = pupil_x
        features["pupil_y"] = pupil_y
        features["eye_x"] = eye_x
        features["eye_y"] = eye_y

        features["pupil_area"] = pupil_area
        features["eye_area"] = eye_area

        features["pupil_major_diameter"] = pupil_major
        features["pupil_minor_diameter"] = pupil_minor
        features["pupil_orientation_deg"] = pupil_orientation
        features["pupil_ovality"] = pupil_ovality
        features["pupil_eccentricity"] = pupil_eccentricity

        features["eye_major_diameter"] = eye_major
        features["eye_minor_diameter"] = eye_minor
        features["eye_orientation_deg"] = eye_orientation
        features["eye_ovality"] = eye_ovality
        features["eye_eccentricity"] = eye_eccentricity

        # Raw pupil displacement from eye center.
        dx = pupil_x - eye_x
        dy = pupil_y - eye_y

        features["pupil_dx_px"] = dx
        features["pupil_dy_px"] = dy

        # Rotate displacement into the eye ellipse coordinate system.
        # eye_orientation_deg is the major-axis orientation.
        theta = np.deg2rad(eye_orientation)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Projection onto major and minor axes.
        dx_eye_axis = dx * cos_t + dy * sin_t
        dy_eye_axis = -dx * sin_t + dy * cos_t

        features["pupil_dx_eye_axis_px"] = dx_eye_axis
        features["pupil_dy_eye_axis_px"] = dy_eye_axis

        # Normalize by eye ellipse size.
        features["pupil_dx_eye_axis_norm"] = MeyeGazeCalibrator._safe_divide(
            dx_eye_axis,
            eye_major,
        )
        features["pupil_dy_eye_axis_norm"] = MeyeGazeCalibrator._safe_divide(
            dy_eye_axis,
            eye_minor,
        )

        # Normalize positions by eye size.
        features["pupil_dx_eye_major_norm"] = MeyeGazeCalibrator._safe_divide(
            dx,
            eye_major,
        )
        features["pupil_dy_eye_minor_norm"] = MeyeGazeCalibrator._safe_divide(
            dy,
            eye_minor,
        )

        # Approximate frame normalization using eye ellipse dimensions.
        # Useful when no full frame size is stored in result.
        features["pupil_x_norm_frame"] = MeyeGazeCalibrator._safe_divide(
            pupil_x,
            max(eye_major, eye_minor) if not np.isnan(eye_major) and not np.isnan(eye_minor) else np.nan,
        )
        features["pupil_y_norm_frame"] = MeyeGazeCalibrator._safe_divide(
            pupil_y,
            max(eye_major, eye_minor) if not np.isnan(eye_major) and not np.isnan(eye_minor) else np.nan,
        )
        features["eye_x_norm_frame"] = MeyeGazeCalibrator._safe_divide(
            eye_x,
            max(eye_major, eye_minor) if not np.isnan(eye_major) and not np.isnan(eye_minor) else np.nan,
        )
        features["eye_y_norm_frame"] = MeyeGazeCalibrator._safe_divide(
            eye_y,
            max(eye_major, eye_minor) if not np.isnan(eye_major) and not np.isnan(eye_minor) else np.nan,
        )

        features["pupil_area_norm"] = MeyeGazeCalibrator._safe_divide(
            pupil_area,
            eye_area,
        )
        features["pupil_major_diameter_norm"] = MeyeGazeCalibrator._safe_divide(
            pupil_major,
            eye_major,
        )
        features["pupil_minor_diameter_norm"] = MeyeGazeCalibrator._safe_divide(
            pupil_minor,
            eye_minor,
        )

        if np.isnan(eye_orientation):
            features["eye_orientation_sin"] = np.nan
            features["eye_orientation_cos"] = np.nan
        else:
            features["eye_orientation_sin"] = float(np.sin(theta))
            features["eye_orientation_cos"] = float(np.cos(theta))

        features["inference_ms"] = getattr(result, "inference_time_ms", np.nan)
        features["inference_fps"] = getattr(result, "inference_fps", np.nan)

        return features

    @staticmethod
    def _safe_tuple_get(value, index, default=np.nan):
        try:
            return value[index]
        except Exception:
            return default

    @staticmethod
    def _safe_divide(a, b, default=np.nan):
        try:
            if b is None or np.isnan(b) or b == 0:
                return default
            return float(a / b)
        except Exception:
            return default

    # ============================================================
    # Sample collection
    # ============================================================

    def add_sample(self, result, timestamp=None, extra=None):
        """
            Add one calibration sample from a MEYELens result.

            This method extracts gaze-calibration features from a :class:`MeyeResult`
            and associates them with the current calibration target. The resulting
            sample is stored in memory in ``self.samples``.

            Parameters
            ----------
            result : MeyeResult
                MEYELens prediction result from which gaze features are extracted.

                The result is expected to contain pupil and eye features in
                ``result.features``.

            timestamp : float or None, default=None
                Optional timestamp assigned to the sample.

                If ``None``, the current value of ``time.perf_counter()`` is used.

            extra : dict or None, default=None
                Optional additional fields to add to the sample dictionary.

                This can be used to store extra information such as trial number,
                condition, fixation duration, frame index, or quality flags.

            Returns
            -------
            dict
                Sample dictionary added to ``self.samples``.

                The dictionary contains target information, timestamp, extracted gaze
                features, and any additional fields provided through ``extra``.

            Raises
            ------
            RuntimeError
                If no calibration target is currently active.

            Notes
            -----
            Before calling this method, call :meth:`next_target` to activate a
            calibration target.

            This method is a convenience wrapper around :meth:`add_features`. It first
            calls :meth:`extract_features_from_result` and then stores the resulting
            feature dictionary with the current target.
        """

        if self.current_target is None:
            raise RuntimeError("No current target. Call next_target() first.")

        features = self.extract_features_from_result(result)

        return self.add_features(
            feature_dict=features,
            target=self.current_target,
            timestamp=timestamp,
            extra=extra,
        )

    def add_features(self, feature_dict, target=None, timestamp=None, extra=None):
        """
            Add one calibration sample from an already-extracted feature dictionary.

            This method stores one feature dictionary together with a target position.
            It is useful when MEYELens features have already been extracted, or when
            calibration samples are being created from previously stored data.

            Parameters
            ----------
            feature_dict : dict
                Dictionary of gaze features.

                The keys should include the feature names used by ``self.feature_columns``
                if the sample will later be used for model fitting.

            target : dict or tuple or list or None, default=None
                Target position associated with this sample.

                Supported formats are:

                ``None``
                    Use the current target stored in ``self.current_target``.

                ``(target_x, target_y)``
                    Manual target position. In this case, ``target_id`` and
                    ``target_sample_index`` are stored as ``np.nan``.

                dict
                    Target dictionary. It must contain ``"target_x"`` and
                    ``"target_y"``. It may also contain ``"target_id"``.

                If the provided target is exactly ``self.current_target``, the internal
                ``current_target_sample_index`` counter is incremented after adding the
                sample.

            timestamp : float or None, default=None
                Optional timestamp assigned to the sample.

                If ``None``, the current value of ``time.perf_counter()`` is used.

            extra : dict or None, default=None
                Optional additional fields to merge into the sample dictionary.

                Values in ``extra`` are added after the feature dictionary, so they can
                be used to attach trial information, frame indices, quality flags, or
                other experiment-specific variables.

            Returns
            -------
            dict
                Sample dictionary added to ``self.samples``.

                The dictionary contains:

                ``time``
                    Sample timestamp.

                ``target_id``
                    Target identifier, if available.

                ``target_sample_index``
                    Index of the sample within the current target, if available.

                ``target_x``
                    Horizontal target coordinate.

                ``target_y``
                    Vertical target coordinate.

                plus all fields from ``feature_dict`` and ``extra``.

            Raises
            ------
            RuntimeError
                If ``target=None`` and no current target is active.

            ValueError
                If ``target`` is not ``None``, a tuple/list, or a dictionary.

            Notes
            -----
            This method does not check whether all features required by
            ``self.feature_columns`` are present. Missing or invalid features are
            filtered later by :meth:`prepare_training_data`.

            Target coordinates should be expressed in the same coordinate system used
            when the calibrator was initialized.
        """

        if timestamp is None:
            timestamp = time.perf_counter()

        if target is None:
            if self.current_target is None:
                raise RuntimeError("No current target. Call next_target() or pass target.")

            target = self.current_target

        if isinstance(target, (tuple, list)):
            target_dict = {
                "target_id": np.nan,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
            }
            target_sample_index = np.nan

        elif isinstance(target, dict):
            target_dict = target
            target_sample_index = self.current_target_sample_index

            if target is self.current_target:
                self.current_target_sample_index += 1

        else:
            raise ValueError("target must be None, tuple/list, or dict.")

        sample = {
            "time": float(timestamp),
            "target_id": target_dict.get("target_id", np.nan),
            "target_sample_index": target_sample_index,
            "target_x": float(target_dict["target_x"]),
            "target_y": float(target_dict["target_y"]),
        }

        sample.update(dict(feature_dict))

        if extra is not None:
            sample.update(dict(extra))

        self.samples.append(sample)

        return sample

    def clear_samples(self):
        """
            Remove all collected calibration samples.

            This method clears the in-memory sample list used for model fitting.

            Returns
            -------
            None

            Notes
            -----
            This method does not reset the calibration target sequence and does not
            modify the fitted model. To restart the target sequence, use
            :meth:`reset_targets`.
        """
        self.samples = []

    def get_samples(self):
        """
            Return a copy of the collected calibration samples.

            Returns
            -------
            list of dict
                List containing the currently stored calibration samples.

                A new list object is returned, but the sample dictionaries inside the
                list are not deep-copied.

            Notes
            -----
            The returned list can be inspected or saved externally by the user.
        """
        return list(self.samples)

    # ============================================================
    # Training data preparation
    # ============================================================

    def prepare_training_data(
        self,
        skip_samples=5,
        aggregate="median",
        min_pupil_area=None,
        min_eye_area=None,
    ):
        """
            Prepare calibration samples for model fitting.

            This method converts the in-memory calibration samples stored in
            ``self.samples`` into predictor and target arrays suitable for fitting a
            gaze calibration model.

            It filters out samples that are likely to be unreliable and optionally
            aggregates valid samples within each calibration target.

            Parameters
            ----------
            skip_samples : int or None, default=5
                Number of initial samples to discard for each calibration target.

                This is useful because the participant's eye may still be moving
                immediately after a new target appears. For example, with
                ``skip_samples=5``, the first five samples collected after each target
                onset are excluded.

                If ``None``, no samples are skipped based on
                ``target_sample_index``.

            aggregate : {"median", "mean", None}, default="median"
                How to combine valid samples within each calibration target.

                ``"median"``
                    Return one median feature vector and one median target coordinate
                    per target. This is the recommended default because it is robust to
                    occasional noisy samples.

                ``"mean"``
                    Return one mean feature vector and one mean target coordinate per
                    target.

                ``None``
                    Return all valid samples without target-wise aggregation.

            min_pupil_area : int or float or None, default=None
                Optional minimum pupil area required for a sample to be included.

                Samples with ``sample["pupil_area"] < min_pupil_area`` are discarded.
                If ``None``, no pupil-area threshold is applied.

            min_eye_area : int or float or None, default=None
                Optional minimum eye area required for a sample to be included.

                Samples with ``sample["eye_area"] < min_eye_area`` are discarded.
                If ``None``, no eye-area threshold is applied.

            Returns
            -------
            X : numpy.ndarray
                Predictor matrix with shape ``(n_samples, n_features)``.

                Columns correspond to ``self.feature_columns``.

            y : numpy.ndarray
                Target coordinate matrix with shape ``(n_samples, 2)``.

                The two columns are ``target_x`` and ``target_y``.

            valid : list of dict
                List of valid sample entries used to build ``X`` and ``y``.

                Each entry contains:

                ``target_id``
                    Target identifier.

                ``X``
                    Feature vector for one valid sample.

                ``y``
                    Target coordinate vector for one valid sample.

                If ``aggregate`` is ``"median"`` or ``"mean"``, this list still
                contains the valid pre-aggregation samples.

            Raises
            ------
            RuntimeError
                If no calibration samples are available.

            RuntimeError
                If no valid samples remain after filtering.

            ValueError
                If ``aggregate`` is not ``"median"``, ``"mean"``, or ``None``.

            Notes
            -----
            A sample is excluded if any feature listed in ``self.feature_columns`` is
            missing, ``None``, or ``NaN``.

            A sample is also excluded if ``target_x`` or ``target_y`` is missing or
            ``NaN``.

            This method does not modify ``self.samples``. Filtering and aggregation are
            applied only to the returned training arrays.

            This method does not perform blink interpolation or signal cleaning. For
            calibration, unreliable samples should generally be rejected before fitting,
            for example by using ``skip_samples``, ``min_pupil_area``, and
            ``min_eye_area``.

        """

        if not self.samples:
            raise RuntimeError("No calibration samples available.")

        valid = []

        for sample in self.samples:
            if skip_samples is not None:
                if sample.get("target_sample_index", 0) < skip_samples:
                    continue

            if min_pupil_area is not None:
                if sample.get("pupil_area", np.nan) < min_pupil_area:
                    continue

            if min_eye_area is not None:
                if sample.get("eye_area", np.nan) < min_eye_area:
                    continue

            row = []
            has_nan = False

            for col in self.feature_columns:
                value = sample.get(col, np.nan)

                if value is None or np.isnan(value):
                    has_nan = True
                    break

                row.append(float(value))

            if has_nan:
                continue

            target_x = sample.get("target_x", np.nan)
            target_y = sample.get("target_y", np.nan)

            if np.isnan(target_x) or np.isnan(target_y):
                continue

            valid.append(
                {
                    "target_id": sample.get("target_id", np.nan),
                    "X": np.array(row, dtype=float),
                    "y": np.array([target_x, target_y], dtype=float),
                }
            )

        if not valid:
            raise RuntimeError("No valid calibration samples after filtering.")

        if aggregate is None:
            X = np.vstack([v["X"] for v in valid])
            y = np.vstack([v["y"] for v in valid])
            return X, y, valid

        if aggregate not in ("median", "mean"):
            raise ValueError("aggregate must be 'median', 'mean', or None.")

        grouped = {}

        for item in valid:
            target_id = item["target_id"]
            grouped.setdefault(target_id, {"X": [], "y": []})
            grouped[target_id]["X"].append(item["X"])
            grouped[target_id]["y"].append(item["y"])

        X_list = []
        y_list = []

        for target_id, group in grouped.items():
            X_group = np.vstack(group["X"])
            y_group = np.vstack(group["y"])

            if aggregate == "median":
                X_list.append(np.median(X_group, axis=0))
                y_list.append(np.median(y_group, axis=0))
            else:
                X_list.append(np.mean(X_group, axis=0))
                y_list.append(np.mean(y_group, axis=0))

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y, valid

    # ============================================================
    # Model fitting
    # ============================================================

    def fit(
        self,
        model_type=None,
        degree=2,
        alpha=1.0,
        skip_samples=5,
        aggregate="median",
        min_pupil_area=None,
        min_eye_area=None,
        mlp_hidden_layer_sizes=(32, 16),
        mlp_max_iter=2000,
    ):
        """
            Fit the gaze calibration model.

            This method prepares the collected calibration samples, fits a regression
            model that maps MEYE pupil/eye features to target coordinates, stores
            the fitted model in ``self.model``, and returns a summary of the fitting
            procedure.

            Parameters
            ----------
            model_type : {"ridge_poly", "ridge", "mlp"} or None, default=None
                Type of regression model to fit.

                If ``None``, the current value of ``self.model_type`` is used.

                ``"ridge_poly"``
                    Polynomial feature expansion followed by ridge regression. This is
                    the recommended default when the number of calibration targets is
                    limited.

                ``"ridge"``
                    Linear ridge regression.

                ``"mlp"``
                    Multi-layer perceptron regression. This can model nonlinear
                    mappings, but usually requires more calibration samples than ridge
                    models.

            degree : int, default=2
                Polynomial degree used when ``model_type="ridge_poly"``.

                Ignored when ``model_type`` is ``"ridge"`` or ``"mlp"``.

            alpha : float, default=1.0
                Regularization strength.

                For ridge models, this is the ridge penalty. For the MLP model, this is
                the L2 penalty passed to ``MLPRegressor``.

            skip_samples : int or None, default=5
                Number of initial samples to discard for each calibration target before
                fitting.

                Passed to :meth:`prepare_training_data`.

            aggregate : {"median", "mean", None}, default="median"
                How to combine valid samples within each calibration target before
                fitting.

                Passed to :meth:`prepare_training_data`.

            min_pupil_area : int or float or None, default=None
                Optional minimum pupil area required for a sample to be used for
                fitting.

                Passed to :meth:`prepare_training_data`.

            min_eye_area : int or float or None, default=None
                Optional minimum eye area required for a sample to be used for fitting.

                Passed to :meth:`prepare_training_data`.

            mlp_hidden_layer_sizes : tuple of int, default=(32, 16)
                Hidden-layer sizes used when ``model_type="mlp"``.

                Ignored for ridge models.

            mlp_max_iter : int, default=2000
                Maximum number of optimization iterations used when
                ``model_type="mlp"``.

                Ignored for ridge models.

            Returns
            -------
            dict
                Fit summary dictionary stored in ``self.fit_info``.

                The dictionary contains:

                ``model_type``
                    Fitted model type.

                ``degree``
                    Polynomial degree for ``"ridge_poly"``, otherwise ``None``.

                ``alpha``
                    Regularization strength.

                ``skip_samples``
                    Number of initial samples skipped per target.

                ``aggregate``
                    Aggregation method used before fitting.

                ``n_samples``
                    Number of samples used for fitting after filtering and aggregation.

                ``n_raw_samples``
                    Number of raw samples stored in ``self.samples``.

                ``feature_columns``
                    Feature columns used as predictors.

                ``target_columns``
                    Target coordinate columns.

                ``mse``
                    Mean squared error computed on the training data.

                ``r2``
                    Coefficient of determination computed on the training data.

            Attributes Modified
            -------------------
            model : sklearn estimator
                Set to the fitted regression pipeline.

            model_type : str
                Updated to the fitted model type.

            is_fitted : bool
                Set to ``True`` after fitting succeeds.

            fit_info : dict
                Updated with the returned fit summary.

            Raises
            ------
            RuntimeError
                If there are no calibration samples.

            RuntimeError
                If no valid samples remain after filtering.

            RuntimeError
                If fewer than three calibration samples are available after preparation.

            ValueError
                If ``model_type`` is not ``"ridge_poly"``, ``"ridge"``, or ``"mlp"``.

            Notes
            -----
            Target coordinates are standardized internally before model fitting using
            ``self.target_scaler``. Predictions are transformed back to the original
            target coordinate system.

            The reported ``mse`` and ``r2`` are computed on the same data used for
            fitting. They are useful as quick diagnostics, but they should not be
            interpreted as independent validation performance.
        """

        if model_type is None:
            model_type = self.model_type

        X, y, used = self.prepare_training_data(
            skip_samples=skip_samples,
            aggregate=aggregate,
            min_pupil_area=min_pupil_area,
            min_eye_area=min_eye_area,
        )

        if len(X) < 3:
            raise RuntimeError(f"Not enough calibration points: {len(X)}")

        y_scaled = self.target_scaler.fit_transform(y)

        if model_type == "ridge_poly":
            model = Pipeline(
                steps=[
                    ("feature_scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                    ("regression", Ridge(alpha=alpha)),
                ]
            )

        elif model_type == "ridge":
            model = Pipeline(
                steps=[
                    ("feature_scaler", StandardScaler()),
                    ("regression", Ridge(alpha=alpha)),
                ]
            )

        elif model_type == "mlp":
            model = Pipeline(
                steps=[
                    ("feature_scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=mlp_hidden_layer_sizes,
                            activation="relu",
                            solver="adam",
                            alpha=alpha,
                            max_iter=mlp_max_iter,
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            )

        else:
            raise ValueError("model_type must be 'ridge_poly', 'ridge', or 'mlp'.")

        model.fit(X, y_scaled)

        pred_scaled = model.predict(X)
        pred = self.target_scaler.inverse_transform(pred_scaled)

        mse = mean_squared_error(y, pred)
        r2 = r2_score(y, pred)

        self.model = model
        self.model_type = model_type
        self.is_fitted = True

        self.fit_info = {
            "model_type": model_type,
            "degree": int(degree) if model_type == "ridge_poly" else None,
            "alpha": float(alpha),
            "skip_samples": int(skip_samples),
            "aggregate": aggregate,
            "n_samples": int(len(X)),
            "n_raw_samples": int(len(self.samples)),
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "mse": float(mse),
            "r2": float(r2),
        }

        print(
            "### MeyeGazeCalibrator ### Fit complete. "
            f"model={model_type}, n={len(X)}, MSE={mse:.4f}, R2={r2:.4f}"
        )

        return self.fit_info

    # ============================================================
    # Prediction
    # ============================================================

    def predict_features(self, features):
        """
            Predict gaze coordinates from feature values.

            This method applies the fitted calibration model to one or more feature
            vectors and returns predicted gaze coordinates in the calibration coordinate
            system.

            Parameters
            ----------
            features : dict or array-like
                Feature values used for prediction.

                If a dictionary is provided, values are extracted in the order defined
                by ``self.feature_columns``.

                If an array is provided, it must contain feature values in the same
                order as ``self.feature_columns``. A one-dimensional array is
                interpreted as one sample. A two-dimensional array is interpreted as
                multiple samples.

            Returns
            -------
            numpy.ndarray
                Predicted gaze coordinates with shape ``(n_samples, 2)``.

                The two columns correspond to predicted ``x`` and ``y`` coordinates in
                the same coordinate system used during calibration.

            Raises
            ------
            RuntimeError
                If the calibration model has not been fitted yet.

            RuntimeError
                If the input feature array contains ``NaN`` values.

            Notes
            -----
            The calibration model must be fitted with :meth:`fit` before calling this
            method.
        """

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Calibration model is not fitted yet.")

        if isinstance(features, dict):
            X = np.array(
                [[features.get(col, np.nan) for col in self.feature_columns]],
                dtype=float,
            )
        else:
            X = np.asarray(features, dtype=float)

            if X.ndim == 1:
                X = X.reshape(1, -1)

        if np.any(np.isnan(X)):
            raise RuntimeError("Input gaze features contain NaN.")

        pred_scaled = self.model.predict(X)
        pred = self.target_scaler.inverse_transform(pred_scaled)

        return pred

    def predict_result(self, result):
        """
            Predict gaze coordinates from a MEYE result.

            This method extracts gaze-calibration features from a :class:`MeyeResult`
            and applies the fitted calibration model.

            Parameters
            ----------
            result : MeyeResult
                MEYE prediction result containing pupil and eye features.

            Returns
            -------
            numpy.ndarray
                Predicted gaze coordinates with shape ``(1, 2)``.

                The two columns correspond to predicted ``x`` and ``y`` coordinates in
                the calibration coordinate system.

            Raises
            ------
            RuntimeError
                If the calibration model has not been fitted yet.

            RuntimeError
                If the extracted features contain ``NaN`` values.

            Notes
            -----
            This is a convenience wrapper around
            :meth:`extract_features_from_result` and :meth:`predict_features`.
        """

        features = self.extract_features_from_result(result)
        return self.predict_features(features)

    def predict_xy(self, result):
        """
            Predict gaze coordinates as scalar x and y values.

            This method predicts gaze position directly from a :class:`MeyeResult` and
            returns the first predicted sample as two Python floats.

            Parameters
            ----------
            result : MeyeResult
                MEYELens prediction result containing pupil and eye features.

            Returns
            -------
            x : float
                Predicted horizontal gaze coordinate.

            y : float
                Predicted vertical gaze coordinate.

            Raises
            ------
            RuntimeError
                If the calibration model has not been fitted yet.

            RuntimeError
                If the extracted features contain ``NaN`` values.

            Notes
            -----
            This is the most convenient method for live gaze display, because it returns
            scalar coordinates that can be assigned directly to a visual stimulus
            position.

        """

        pred = self.predict_result(result)
        return float(pred[0, 0]), float(pred[0, 1])



if __name__ == "__main__":
    meye = Meye()
    with Camera(camera_index=1) as cam:
        cam.select_roi()
        meye.preview(cam)
