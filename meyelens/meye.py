import time

import cv2
import numpy as np
import tensorflow as tf
from skimage.measure import label, regionprops
from tensorflow.keras.models import load_model

from .camera import Camera
from .fileio import BufferedFileWriter, FileWriter

try:  # Py ≥ 3.9
    from importlib.resources import files
except ImportError:  # Py ≤ 3.8
    from importlib_resources import files


class Meye:
    """
    Pupil segmentation and basic shape extraction using a pre-trained neural network.

    This class loads a Keras/TensorFlow segmentation model and exposes a :meth:`predict`
    method that:

    1. converts the input frame to grayscale (if needed)
    2. resizes it to the model input size (hardcoded to 128x128 in this implementation)
    3. runs inference to obtain a pupil mask
    4. optionally performs morphological post-processing to isolate the pupil region
    5. optionally fits an ellipse to estimate major/minor diameters and orientation

    Attributes
    ----------
    model_path : str or pathlib.Path
        Path to the Keras model file used for inference.
    model : tensorflow.keras.Model
        Loaded Keras model.
    requiredFrameSize : tuple[int, int]
        Expected model input frame size (height, width) derived from the model input.
        Note: this implementation still resizes to 128x128 in :meth:`predict`.
    centroid : tuple[float, float] or float
        Centroid (row, col) of the largest detected pupil region after post-processing.
        Set to ``(np.nan, np.nan)`` when no pupil is found.
    pupil_size : float
        Number of non-zero pixels in the resized pupil mask (in the original image size).
    major_diameter : float
        Major axis length from an ellipse fit (in pixels), if available.
    minor_diameter : float
        Minor axis length from an ellipse fit (in pixels), if available.
    orientation : float
        Ellipse orientation angle (degrees), if available.

    Notes
    -----
    - This class prints GPU availability on initialization (no logging).
    - The model is assumed to return two outputs (``mask, info``). Only ``mask`` is used.
    - Coordinate conventions:
        - centroid from :func:`skimage.measure.regionprops` is (row, col).
        - the recorders write centroid as (x=col, y=row) by swapping indices.
    """

    def __init__(self, model=None):
        """
        Initialize the detector and load the segmentation model.

        Parameters
        ----------
        model : str or pathlib.Path or None, optional
            Path to a ``.h5`` Keras model. If ``None``, loads the bundled
            ``meye-2022-01-24.h5`` model from the ``meyelens.models`` package.
        """
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

        if model is None:
            self.model_path = files("meyelens.models").joinpath("meye-2022-01-24.h5")
        else:
            self.model_path = model

        self.model = load_model(self.model_path)

        # Derive required input size from the model's first input tensor.
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.requiredFrameSize = tuple(shape[1:3])

        # Public outputs (updated on every predict call).
        self.centroid = np.nan
        self.pupil_size = np.nan
        self.major_diameter = np.nan
        self.minor_diameter = np.nan
        self.orientation = np.nan

    def predict(self, img, post_proc: bool = True, morph: bool = True, fill_ellipse: bool = False):
        """
        Predict a pupil mask and centroid from an input image.

        Parameters
        ----------
        img : numpy.ndarray
            Input frame, grayscale (H, W) or BGR (H, W, 3).
        post_proc : bool, optional
            If ``True``, apply :meth:`morphProcessing` to binarize, keep the
            largest connected component, and perform morphological closing.
            If ``False``, the raw network output is used and centroid is set to (0, 0).
        morph : bool, optional
            If ``True``, attempt ellipse fitting on the post-processed mask to estimate
            major/minor diameters and orientation.
        fill_ellipse : bool, optional
            If ``True``, replace the mask with a filled ellipse fitted to the contour
            (useful for smoothing irregular segmentations).

        Returns
        -------
        pupil_resized : numpy.ndarray
            Processed pupil mask resized back to the original image size.
            Pixel values are 0/255 when post-processing is enabled.
        centroid : tuple[float, float]
            Centroid of the detected pupil region in (row, col) format.

        Notes
        -----
        - The model input is normalized to [0, 1] and shaped as (1, H, W, 1).
        - This implementation resizes inputs to 128x128 regardless of
          :attr:`requiredFrameSize`.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to model input size used by this implementation.
        crop = cv2.resize(img, [128, 128])

        networkInput = crop.astype(np.float32) / 255.0
        networkInput = networkInput[None, :, :, None]

        mask, info = self.model(networkInput)  # 'info' is unused but kept for model compatibility
        pupil = mask[0, :, :, 0]

        if post_proc:
            pupil, centroid = self.morphProcessing(pupil)
        else:
            centroid = (0, 0)

        # Optional ellipse fitting to estimate geometric features.
        if morph:
            contours, _ = cv2.findContours(pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                try:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    self.major_diameter = max(axes)
                    self.minor_diameter = min(axes)
                    self.orientation = angle
                except Exception as e:
                    self.major_diameter = np.nan
                    self.minor_diameter = np.nan
                    self.orientation = np.nan
                    print(e)

        if fill_ellipse:
            pupil = self.fit_ellipse_and_fill(pupil)

        # Resize mask back to the original frame size.
        pupil_resized = cv2.resize(pupil, (img.shape[1], img.shape[0]))

        # Update public state.
        self.pupil_size = np.sum(pupil_resized > 0)
        self.centroid = centroid

        return pupil_resized, centroid

    def morphProcessing(self, sourceImg, thr: float = 0.8):
        """
        Post-process the raw model output to isolate the pupil region.

        Steps
        -----
        1. Threshold the model output at ``thr``
        2. Label connected components
        3. Keep only the largest component
        4. Apply morphological closing with an elliptical kernel

        Parameters
        ----------
        sourceImg : numpy.ndarray
            Raw model output mask (float array in [0, 1]).
        thr : float, optional
            Threshold used to binarize the mask.

        Returns
        -------
        morph : numpy.ndarray
            Post-processed binary mask as uint8 with values 0 or 255.
        centroid : tuple[float, float]
            Centroid of the largest component in (row, col) format.
            Returns ``(np.nan, np.nan)`` if no component is found.
        """
        binarized = sourceImg > thr
        label_img = label(binarized)
        regions = regionprops(label_img)

        if len(regions) == 0:
            morph = np.zeros(sourceImg.shape, dtype="uint8")
            centroid = (np.nan, np.nan)
            return morph, centroid

        # Keep only the largest connected component.
        regions.sort(key=lambda x: x.area, reverse=True)
        centroid = regions[0].centroid

        for rg in regions[1:]:
            label_img[rg.coords[:, 0], rg.coords[:, 1]] = 0

        label_img[label_img != 0] = 1
        biggest_region = (label_img * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        morph = cv2.morphologyEx(biggest_region, cv2.MORPH_CLOSE, kernel)

        return morph, centroid

    @staticmethod
    def fit_ellipse_and_fill(mask):
        """
        Fit an ellipse to the pupil mask and return a filled ellipse mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask (uint8) typically with values 0/255.

        Returns
        -------
        numpy.ndarray
            New mask where the pupil is represented by a filled ellipse
            (0/255). If ellipse fitting is not possible, returns the input mask.

        Notes
        -----
        - Uses the convex hull of the largest contour to stabilize ellipse fitting.
        - OpenCV requires at least 5 points to fit an ellipse.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)

        if len(hull) >= 5:
            ellipse = cv2.fitEllipse(hull)
            (x, y), (major_axis, minor_axis), angle = ellipse

            filled_image = np.zeros_like(mask, dtype=np.uint8)
            cv2.ellipse(
                filled_image,
                (int(x), int(y)),
                (int(major_axis / 2), int(minor_axis / 2)),
                angle,
                0,
                360,
                255,
                thickness=-1,
            )
            return filled_image

        return mask

    @staticmethod
    def overlay_roi(mask, roi, ratios=(0.7, 0.3)):
        """
        Overlay a pupil mask over a region of interest (ROI) image.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask to overlay (expected 0/255).
        roi : numpy.ndarray
            Base image (typically BGR) to overlay on.
        ratios : tuple[float, float], optional
            Blending ratios for (roi, mask_color) passed to
            :func:`cv2.addWeighted`.

        Returns
        -------
        numpy.ndarray
            Blended image for visualization.
        """
        overlay = roi.copy()
        mask_colored = Meye.mask2color(mask, 1)
        overlay = cv2.addWeighted(overlay, ratios[0], mask_colored, ratios[1], 0)
        return overlay

    @staticmethod
    def mask2color(mask, channel: int = 1):
        """
        Convert a single-channel mask into a 3-channel color image.

        Parameters
        ----------
        mask : numpy.ndarray
            2D mask.
        channel : int, optional
            Channel to place the mask in:
            - 0: red
            - 1: green
            - 2: blue

        Returns
        -------
        numpy.ndarray
            3-channel image with the mask in the selected channel.
        """
        if channel == 0:
            out = cv2.merge([mask, np.zeros_like(mask), np.zeros_like(mask)])
        elif channel == 1:
            out = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])
        else:
            out = cv2.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
        return out

    def preview(self, cam):
        """
        Run a real-time preview loop showing pupil segmentation overlay.

        Parameters
        ----------
        cam : Camera
            Camera instance providing frames via :meth:`Camera.get_frame`.

        Returns
        -------
        None

        Notes
        -----
        Press "q" in the preview window to exit.
        """
        print('### MEYE ### Press "q" to exit preview.')
        while True:
            frame = cam.get_frame()
            predicted = self.predict(frame)[0]
            overlay = self.overlay_roi(predicted, frame)
            cam.show(overlay, name="Preview")
            if cam.wait_key("q"):
                break
        cv2.destroyWindow("Preview")


class MeyeRecorder:
    """
    Synchronous frame-by-frame recorder using :class:`Meye` and :class:`FileWriter`.

    This recorder captures frames from a :class:`~.camera.Camera`, runs pupil detection,
    and writes one line per frame to a semicolon-separated text file.

    The output columns are:

    - time (seconds since start)
    - x, y (centroid coordinates; written as col, row)
    - pupil (mask area in pixels)
    - major_diameter, minor_diameter, orientation (ellipse fit; may be NaN)
    - trg1 ... trg9 (user-defined trigger values)

    Parameters
    ----------
    cam_ind : int, optional
        Camera index passed to :class:`~.camera.Camera`.
    model : str or pathlib.Path or None, optional
        Path to the Keras model file. If ``None``, the packaged model is used.
    show_preview : bool, optional
        If ``True``, display an overlay window during recording.
    filename : str, optional
        Base filename used by :class:`FileWriter` (timestamp is added automatically).
    folder_path : str, optional
        Output folder where the text file is created.
    sep : str, optional
        Column separator used in the output file.

    Notes
    -----
    This recorder writes synchronously to disk at each :meth:`save_frame` call.
    """

    def __init__(
        self,
        cam_ind=0,
        model=None,
        show_preview=False,
        filename="meye",
        folder_path="Data",
        sep=";",
    ):
        self.cam = Camera(cam_ind)
        self.filename = filename
        self.show_preview = show_preview
        self.meye = Meye(model=model)

        self.folder_path = folder_path
        self.sep = sep

        self.writer = None
        self.frame = None
        self.predicted = None

        # Trigger placeholders (kept for external code that may read these attributes).
        self.trg1 = 0
        self.trg2 = 0
        self.trg3 = 0
        self.trg4 = 0
        self.trg5 = 0
        self.trg6 = 0
        self.trg7 = 0
        self.trg8 = 0
        self.trg9 = 0

    def start(self) -> None:
        """
        Start recording and initialize the output file.

        Returns
        -------
        None
        """
        print("### MEYE RECORDER ### Start.")
        self.running = True

        self.writer = FileWriter(path_to_file=self.folder_path, filename=self.filename, sep=self.sep)
        self.writer.write_sv(
            [
                "time",
                "x",
                "y",
                "pupil",
                "major_diameter",
                "minor_diameter",
                "orientation",
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
        )
        self.time_start = time.time()

    def preview(self) -> None:
        """
        Open a live preview window (segmentation overlay).

        Returns
        -------
        None
        """
        self.meye.preview(self.cam)

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
    ) -> None:
        """
        Capture one frame, run pupil detection, and append a row to the output file.

        Parameters
        ----------
        trg1, trg2, trg3, trg4, trg5, trg6, trg7, trg8, trg9 : int, optional
            Trigger values to save alongside the measurements.

        Returns
        -------
        None
        """
        if self.time_start is None or self.writer is None:
            print("### MEYE ASYNC RECORDER ### Recording not started!")
            return

        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]

        timestamp = time.time() - self.time_start

        # Note: centroid is (row, col); here we store x=col, y=row.
        self.writer.write_sv(
            [
                timestamp,
                self.meye.centroid[1],
                self.meye.centroid[0],
                self.meye.pupil_size,
                self.meye.major_diameter,
                self.meye.minor_diameter,
                self.meye.orientation,
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
        )

        if self.show_preview:
            overlay = self.meye.overlay_roi(self.predicted, self.frame)
            self.cam.show(overlay, name="Recording")

    def get_data(self):
        """
        Capture one frame and return instantaneous pupil metrics.

        Returns
        -------
        dict
            Dictionary containing:

            - ``centroid``: (x, y) as (col, row)
            - ``size``: pupil mask area in pixels
            - ``major_diameter``: ellipse major axis in pixels (or NaN)
            - ``minor_diameter``: ellipse minor axis in pixels (or NaN)
            - ``orientation``: ellipse angle (degrees) (or NaN)
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]

        data = {
            "centroid": (self.meye.centroid[1], self.meye.centroid[0]),
            "size": self.meye.pupil_size,
            "major_diameter": self.meye.major_diameter,
            "minor_diameter": self.meye.minor_diameter,
            "orientation": self.meye.orientation,
        }
        return data

    def stop(self) -> None:
        """
        Stop recording and close the output file.

        Returns
        -------
        None
        """
        print("### MEYE RECORDER ### File Closed.")
        self.writer.close()

    def close(self) -> None:
        """
        Release the camera resource.

        Returns
        -------
        None
        """
        print("### MEYE RECORDER ### Camera Closed.")
        self.cam.close()

    def close_all(self) -> None:
        """
        Stop recording and release all resources (file + camera).

        Returns
        -------
        None
        """
        print("### MEYE RECORDER ### Closed.")
        self.stop()
        self.close()


class MeyeAsyncRecorder:
    """
    Asynchronous frame-by-frame recorder using :class:`Meye` and :class:`BufferedFileWriter`.

    This recorder is similar to :class:`MeyeRecorder`, but data rows are queued in memory and
    written to disk by a background thread, reducing I/O latency inside tight loops.

    Parameters
    ----------
    cam_ind : int, optional
        Camera index passed to :class:`~.camera.Camera`.
    model : str or pathlib.Path or None, optional
        Path to the Keras model file. If ``None``, the packaged model is used.
    show_preview : bool, optional
        If ``True``, display an overlay window during recording.
    path_to_file : str, optional
        Output folder where the text file is created.
    filename : str, optional
        Base filename used by :class:`BufferedFileWriter` (timestamp is added automatically).
    buffer_size : int, optional
        Queue size for :class:`BufferedFileWriter`. When full, new rows are discarded and
        a warning is printed by the writer (no logging).
    sep : str, optional
        Column separator used in the output file.
    cam_crop : list[int] or tuple[int, int, int, int] or None, optional
        Crop passed to :class:`~.camera.Camera` (implementation-dependent).

    Notes
    -----
    You must call :meth:`stop` (or :meth:`close_all`) to flush remaining queued data.
    """

    def __init__(
        self,
        cam_ind=0,
        model=None,
        show_preview=False,
        path_to_file="Data",
        filename="meye",
        buffer_size=100,
        sep=";",
        cam_crop=None,
    ):
        self.cam = Camera(cam_ind, crop=cam_crop)
        self.show_preview = show_preview
        self.meye = Meye(model=model)

        self.frame = None
        self.predicted = None

        self.filename = filename
        self.path_to_file = path_to_file
        self.buffer_size = buffer_size
        self.sep = sep

        self.writer = None
        self.time_start = None

    def start(self, metadata=None) -> None:
        """
        Start recording and initialize the asynchronous output writer.

        Parameters
        ----------
        metadata : dict or None, optional
            Metadata passed to :class:`BufferedFileWriter` and written as comment
            lines at the top of the file.

        Returns
        -------
        None
        """
        self.writer = BufferedFileWriter(
            self.path_to_file,
            filename=self.filename,
            buffer_size=self.buffer_size,
            headers=[
                "time",
                "x",
                "y",
                "pupil",
                "major_diameter",
                "minor_diameter",
                "orientation",
                "trg1",
                "trg2",
                "trg3",
                "trg4",
                "trg5",
                "trg6",
                "trg7",
                "trg8",
                "trg9",
            ],
            metadata=metadata,
            sep=self.sep,
        )
        self.time_start = time.time()
        print("### MEYE ASYNC RECORDER ### Start.")

    def preview(self) -> None:
        """
        Open a live preview window (segmentation overlay).

        Returns
        -------
        None
        """
        self.meye.preview(self.cam)

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
    ) -> None:
        """
        Capture one frame, run pupil detection, and queue a row for disk writing.

        Parameters
        ----------
        trg1, trg2, trg3, trg4, trg5, trg6, trg7, trg8, trg9 : int, optional
            Trigger values to save alongside the measurements.

        Returns
        -------
        None
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]

        timestamp = time.time() - self.time_start

        # Note: centroid is (row, col); here we store x=col, y=row.
        self.writer.write_sv(
            [
                timestamp,
                self.meye.centroid[1],
                self.meye.centroid[0],
                self.meye.pupil_size,
                self.meye.major_diameter,
                self.meye.minor_diameter,
                self.meye.orientation,
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
        )

        if self.show_preview:
            overlay = self.meye.overlay_roi(self.predicted, self.frame)
            self.cam.show(overlay, name="Recording")

    def get_data(self):
        """
        Capture one frame and return instantaneous pupil metrics.

        Returns
        -------
        dict
            Dictionary containing:

            - ``centroid``: (x, y) as (col, row)
            - ``size``: pupil mask area in pixels
            - ``major_diameter``: ellipse major axis in pixels (or NaN)
            - ``minor_diameter``: ellipse minor axis in pixels (or NaN)
            - ``orientation``: ellipse angle (degrees) (or NaN)
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]

        data = {
            "centroid": (self.meye.centroid[1], self.meye.centroid[0]),
            "size": self.meye.pupil_size,
            "major_diameter": self.meye.major_diameter,
            "minor_diameter": self.meye.minor_diameter,
            "orientation": self.meye.orientation,
        }
        return data

    def stop(self) -> None:
        """
        Stop recording and close the output file (flushes queued data).

        Returns
        -------
        None
        """
        print("### MEYE ASYNC RECORDER ### File Closed.")
        self.writer.close()

    def close(self) -> None:
        """
        Release the camera resource.

        Returns
        -------
        None
        """
        print("### MEYE ASYNC RECORDER ### Camera Closed.")
        self.cam.close()

    def close_all(self) -> None:
        """
        Stop recording and release all resources (writer + camera).

        Returns
        -------
        None
        """
        print("### MEYE ASYNC RECORDER ### Closed.")
        self.stop()
        self.close()


# if __name__ == '__main__':
#     from camera import Camera
#     cam = Camera(1, crop=[100, 220, 250, 270])
#     meye = Meye()
#     meye.preview(cam)
