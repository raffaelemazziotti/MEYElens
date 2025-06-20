from camera import Camera
from meye import Meye
from fileio import FileWriter,BufferedFileWriter
import time

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