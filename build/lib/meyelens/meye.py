import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from skimage.measure import label, regionprops
import time
from .camera import Camera
from .fileio import FileWriter, BufferedFileWriter
try:                                 # Py ≥ 3.9
    from importlib.resources import files
except ImportError:                  # Py ≤ 3.8
    from importlib_resources import files

class Meye:
    """
    Meye Class
    ==========
    This class is designed for pupil detection and analysis from images using a pre-trained deep learning model.
    """

    def __init__(self, model=None): #r'models/meye-2022-01-24.h5'):
        """
        Initializes the Meye class by loading the specified model and checking for GPU availability.
        """
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if model==None:
            self.model_path = files("meyelens.models").joinpath("meye-2022-01-24.h5")
        else:
            self.model_path = model

        self.model = load_model(self.model_path)
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.requiredFrameSize = tuple(shape[1:3])
        self.centroid = np.nan
        self.pupil_size = np.nan
        self.major_diameter = np.nan
        self.minor_diameter = np.nan
        self.orientation = np.nan

    def predict(self, img, post_proc=True, morph=True, fill_ellipse=False):
        """
        Processes the input image to predict the pupil mask and centroid.

        Parameters:
        -----------
        img (np.ndarray): Input image (grayscale or RGB).
        post_proc (bool): Whether to apply post-processing to the prediction.
        morph (bool): Whether to apply morphological processing.
        fill_ellipse (bool): Whether to fill the pupil mask with a fitted ellipse.

        Returns:
        --------
        pupil_resized (np.ndarray): Processed pupil mask.
        centroid (tuple): Centroid coordinates of the detected pupil.
        """
        if len(img.shape) == 3:  # Convert to grayscale if RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        crop = cv2.resize(img, [128, 128])  # Resize to required input size
        networkInput = crop.astype(np.float32) / 255.0  # Normalize image
        networkInput = networkInput[None, :, :, None]  # Add batch and channel dimensions
        mask, info = self.model(networkInput)
        pupil = mask[0, :, :, 0]

        if post_proc:
            pupil, centroid = self.morphProcessing(pupil)
        else:
            centroid = (0, 0)

        if morph:
            contours, _ = cv2.findContours(pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)  # Select the largest contour
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

        pupil_resized = cv2.resize(pupil, (img.shape[1], img.shape[0]))
        self.pupil_size = np.sum(pupil_resized > 0)
        self.centroid = centroid

        return pupil_resized, centroid

    def morphProcessing(self, sourceImg, thr=0.8):
        """
        Applies morphological processing to refine the detected pupil region.
        """
        binarized = sourceImg > thr  # Binarize image
        label_img = label(binarized)
        regions = regionprops(label_img)
        if len(regions) == 0:
            morph = np.zeros(sourceImg.shape, dtype='uint8')
            centroid = (np.nan, np.nan)
            return morph, centroid

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
        Fits an ellipse to the segmented pupil mask and fills the image with the fitted ellipse.
        Uses contour refinement and convex hull to improve shape accuracy.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask  # Return original if no contours are found

        largest_contour = max(contours, key=cv2.contourArea)

        # Refining contour using convex hull
        hull = cv2.convexHull(largest_contour)

        if len(hull) >= 5:  # OpenCV requires at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(hull)
            (x, y), (major_axis, minor_axis), angle = ellipse

            # Create a blank image
            filled_image = np.zeros_like(mask, dtype=np.uint8)

            # Draw the best-fitting ellipse, preserving its natural shape
            cv2.ellipse(filled_image, (int(x), int(y)),
                        (int(major_axis / 2), int(minor_axis / 2)),
                        angle, 0, 360, 255, thickness=-1)

            return filled_image
        else:
            return mask

    @staticmethod
    def overlay_roi(mask, roi, ratios=(0.7, 0.3)):
        """
        Overlays the mask onto the region of interest (ROI) with specified blending ratios.
        """
        overlay = roi.copy()
        mask = Meye.mask2color(mask, 1)
        overlay = cv2.addWeighted(overlay, ratios[0], mask, ratios[1], 0)
        return overlay

    @staticmethod
    def mask2color(mask, channel=1):
        """
        Converts a binary mask into a color image with the specified channel.
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
        Launches a real-time preview of the pupil detection process.
        """
        print('### MEYE ### Press "q" to exit preview.')
        while True:
            frame = cam.get_frame()
            predicted = self.predict(frame)[0]
            overlay = self.overlay_roi(predicted, frame)
            cam.show(overlay, name='Preview')
            if cam.wait_key('q'):
                break
        cv2.destroyWindow("Preview")

class MeyeRecorder:
    """
    MeyeRecorder Class
    ==================
    This class provides functionality to record and analyze video frames using the Meye class for pupil detection.
    It also allows for real-time preview and data logging.

    Attributes:
    -----------
    - cam (Camera): Camera object for capturing video frames.
    - show_preview (bool): Whether to display a live preview of the recording.
    - meye (Meye): Instance of the Meye class for pupil detection.
    - writer (FileWriter): File writer for saving recorded data.
    - frame (np.ndarray): Current frame from the camera.
    - predicted (np.ndarray): Pupil mask predicted for the current frame.
    - trgX (int): Trigger values (trg1 to trg9) for logging events.

    Methods:
    --------
    - __init__(cam_ind: int = -1, model: str = r'models/meye-2022-01-24.h5', show_preview: bool = False):
        Initializes the recorder with the specified camera and model.

    - start():
        Starts the recording process and initializes data logging.

    - preview():
        Displays a real-time preview of pupil detection.

    - save_frame(trg1=0, trg2=0, trg3=0, trg4=0, trg5=0, trg6=0, trg7=0, trg8=0, trg9=0):
        Captures and processes the current frame, logs the data, and optionally displays a preview.

    - stop():
        Stops the recording process and closes all resources.

    - get_data():
        Returns instantaneous data on pupil detection.
    """

    def __init__(self, cam_ind=0, model=r'models/meye-2022-01-24.h5', show_preview=False, filename='meye', folder_path='Data', sep=';'):
        """
        Initializes the MeyeRecorder with a camera and model.

        Parameters:
        -----------
        cam_ind (int): Camera index for video capture (-1 for default camera).
        model (str): Path to the pre-trained model file.
        show_preview (bool): Whether to display a live preview.
        filename (str): Name of the file for saving recorded data.
        """
        self.cam = Camera(cam_ind)
        self.filename = filename
        self.show_preview = show_preview
        self.meye = Meye(model=model)
        self.folder_path = folder_path
        self.sep = sep
        self.writer = None
        self.frame = None
        self.predicted = None
        self.trg1 = 0
        self.trg2 = 0
        self.trg3 = 0
        self.trg4 = 0
        self.trg5 = 0
        self.trg6 = 0
        self.trg7 = 0
        self.trg8 = 0
        self.trg9 = 0

    def start(self):
        """
        Starts the recording process and initializes the file writer for data logging.
        """
        print(f"### MEYE RECORDER ### Start.")
        self.running = True
        self.writer = FileWriter(path_to_file=self.folder_path, filename=self.filename,sep=self.sep)
        self.writer.write_sv(['time', 'x', 'y', 'pupil', 'major_diameter', 'minor_diameter', 'orientation',
                              'trg1', 'trg2', 'trg3', 'trg4', 'trg5', 'trg6', 'trg7', 'trg8', 'trg9'])
        self.time_start = time.time()

    def preview(self):
        """
        Displays a real-time preview of pupil detection using the connected camera.
        """
        self.meye.preview(self.cam)

    def save_frame(self, trg1=0, trg2=0, trg3=0, trg4=0, trg5=0, trg6=0, trg7=0, trg8=0, trg9=0):
        """
        Captures and processes the current frame, logs the data, and optionally displays a preview.

        Parameters:
        -----------
        trg1 to trg9 (int): Trigger values for logging events (default is 0 for all).
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]
        timestamp = time.time() - self.time_start
        self.writer.write_sv([timestamp, self.meye.centroid[1], self.meye.centroid[0], self.meye.pupil_size,
                              self.meye.major_diameter, self.meye.minor_diameter, self.meye.orientation,
                              trg1, trg2, trg3, trg4, trg5, trg6, trg7, trg8, trg9])
        if self.show_preview:
            overlay = self.meye.overlay_roi(self.predicted, self.frame)
            self.cam.show(overlay, name='Recording')

    def get_data(self):
        """
        Retrieves instantaneous data on pupil detection.

        Returns:
        --------
        dict: Dictionary containing pupil detection metrics including centroid, size, major diameter,
              minor diameter, and orientation.
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]
        data = {'centroid': (self.meye.centroid[1], self.meye.centroid[0]),
                'size': self.meye.pupil_size,
                'major_diameter': self.meye.major_diameter,
                'minor_diameter': self.meye.minor_diameter,
                'orientation': self.meye.orientation}
        return data

    def stop(self):
        """
        Stops the recording process, closes the file writer, and releases the camera.
        """
        print(f"### MEYE RECORDER ### File Closed.")
        self.writer.close()

    def close(self):
        """
        Releases the camera resource.
        """
        print(f"### MEYE RECORDER ### Camera Closed.")
        self.cam.close()

    def close_all(self):
        """
        Stops recording and releases all resources, including the camera and file writer.
        """

        print(f"### MEYE RECORDER ### Closed.")
        self.stop()
        self.close()

# TODO test this class and compare with MeyeRecorder
class MeyeAsyncRecorder:
    """
    MeyeAsyncRecorder Class
    =======================
    This class provides functionality to record and analyze video frames using the Meye class for pupil detection,
    while asynchronously saving data to a file using BufferedFileWriter.

    Attributes:
    -----------
    - cam (Camera): Camera object for capturing video frames.
    - show_preview (bool): Whether to display a live preview of the recording.
    - meye (Meye): Instance of the Meye class for pupil detection.
    - writer (BufferedFileWriter): Asynchronous file writer for saving recorded data.
    - frame (np.ndarray): Current frame from the camera.
    - predicted (np.ndarray): Pupil mask predicted for the current frame.

    Methods:
    --------
    - __init__(cam_ind=0, model='', show_preview=False, path_to_file='.', filename='meye', buffer_size=100):
        Initializes the recorder with the specified camera, model, and file writer.

    - start(metadata=None):
        Starts the recording process and initializes data logging.

    - preview():
        Displays a real-time preview of pupil detection.

    - save_frame(trg1=0, trg2=0, trg3=0, trg4=0, trg5=0, trg6=0, trg7=0, trg8=0, trg9=0):
        Captures and processes the current frame, logs the data, and optionally displays a preview.

    - get_data():
        Returns instantaneous data on pupil detection.

    - stop():
        Stops the recording process and closes all resources.
    """

    def __init__(self, cam_ind=0, model=r'models/meye-2022-01-24.h5', show_preview=False,
                 path_to_file='Data', filename='meye', buffer_size=100,sep=';', cam_crop=None):
        """
        Initializes the MeyeAsyncRecorder with a camera, Meye model, and asynchronous file writer.

        Parameters:
        -----------
        cam_ind (int): Camera index for video capture (-1 for default camera).
        model (str): Path to the pre-trained model file.
        show_preview (bool): Whether to display a live preview.
        path_to_file (str): Directory to save the recorded data.
        filename (str): Base filename for the saved file.
        buffer_size (int): Buffer size for the asynchronous file writer.
        """
        self.cam = Camera(cam_ind,crop=cam_crop)
        self.show_preview = show_preview
        self.meye = Meye(model=model)
        self.frame = None
        self.predicted = None
        self.filename = filename
        self.path_to_file = path_to_file
        self.buffer_size = buffer_size
        self.sep = sep


    def start(self, metadata=None):
        """
        Starts the recording process and initializes the file writer with optional metadata.

        Parameters:
        -----------
        metadata (dict): Metadata to include in the file (default is None).
        """
        #if metadata:
        #    self.writer.metadata.update(metadata)
        #self.writer._write_metadata()  # Write metadata and headers to the file
        self.writer = BufferedFileWriter(self.path_to_file, filename=self.filename, buffer_size=self.buffer_size,
                                         headers=['time', 'x', 'y', 'pupil', 'major_diameter',
                                                  'minor_diameter', 'orientation', 'trg1', 'trg2',
                                                  'trg3', 'trg4', 'trg5', 'trg6', 'trg7', 'trg8', 'trg9'],metadata=metadata,sep=self.sep)
        self.time_start = time.time()
        print("### MEYE ASYNC RECORDER ### Start.")

    def preview(self):
        """
        Displays a real-time preview of pupil detection using the connected camera.
        """
        self.meye.preview(self.cam)

    def save_frame(self, trg1=0, trg2=0, trg3=0, trg4=0, trg5=0, trg6=0, trg7=0, trg8=0, trg9=0):
        """
        Captures and processes the current frame, logs the data asynchronously, and optionally displays a preview.

        Parameters:
        -----------
        trg1 to trg9 (int): Trigger values for logging events (default is 0 for all).
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]
        timestamp = time.time() - self.time_start
        self.writer.write_sv([timestamp, self.meye.centroid[1], self.meye.centroid[0], self.meye.pupil_size,
                              self.meye.major_diameter, self.meye.minor_diameter, self.meye.orientation,
                              trg1, trg2, trg3, trg4, trg5, trg6, trg7, trg8, trg9])
        if self.show_preview:
            overlay = self.meye.overlay_roi(self.predicted, self.frame)
            self.cam.show(overlay, name='Recording')

    def get_data(self):
        """
        Retrieves instantaneous data on pupil detection.

        Returns:
        --------
        dict: Dictionary containing pupil detection metrics including centroid, size, major diameter,
              minor diameter, and orientation.
        """
        self.frame = self.cam.get_frame()
        self.predicted = self.meye.predict(self.frame)[0]
        data = {'centroid': (self.meye.centroid[1], self.meye.centroid[0]),
                'size': self.meye.pupil_size,
                'major_diameter': self.meye.major_diameter,
                'minor_diameter': self.meye.minor_diameter,
                'orientation': self.meye.orientation}
        return data

    def stop(self):
        """
        Stops the recording process and closes the file writer and camera resources.
        """
        #print("### MEYE ASYNC RECORDER ### Stop.")
        #self.writer.close()
        #self.cam.close()
        print(f"### MEYE ASYNC RECORDER ### File Closed.")
        self.writer.close()

    def close(self):
        """
        Releases the camera resource.
        """
        print(f"### MEYE ASYNC RECORDER ### Camera Closed.")
        self.cam.close()

    def close_all(self):
        """
        Stops recording and releases all resources, including the camera and file writer.
        """

        print(f"### MEYE ASYNC RECORDER ### Closed.")
        self.stop()
        self.close()


if  __name__ == '__main__':
    from camera import Camera
    cam = Camera(0)
    meye = Meye()
    meye.preview(cam)