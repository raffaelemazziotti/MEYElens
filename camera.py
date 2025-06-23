import cv2
import numpy as np
import time
import toml


class Camera:
    def __init__(self, camera_ind=-1, calibration_file="camera_calibration.toml", undistort=False, exposure=0,
                 framerate=30, resolution=(640, 480), auto_exposure=True, crop=None):
        self.cap = cv2.VideoCapture(camera_ind)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.roi = None
        self.undistort = undistort
        self.exposure = exposure
        self.framerate = framerate  # Fixed framerate set at initialization
        self.resolution = resolution  # Fixed resolution set at initialization
        self.auto_exposure = auto_exposure  # Enable or disable auto exposure

        self.load_calibration(calibration_file)
        self.set_resolution()
        self.set_auto_exposure()
        if not self.auto_exposure:
            self.set_framerate()  # Framerate is set only if auto-exposure is disabled
        if crop:
            self.crop = crop
        else:
            self.crop = None

    def load_calibration(self, calibration_file):
        try:
            with open(calibration_file, "r") as f:
                calibration_data = toml.load(f)
                self.camera_matrix = np.array(calibration_data["camera_matrix"])
                self.dist_coeffs = np.array(calibration_data["distortion_coefficients"])
                print("### CAMERA ### Camera calibration parameters loaded successfully.")
        except FileNotFoundError:
            print(f"### CAMERA ### Calibration file '{calibration_file}' not found. Proceeding without calibration.")
        except KeyError as e:
            print(f"### CAMERA ### Missing key in calibration file: {e}. Proceeding without calibration.")
        except Exception as e:
            print(f"### CAMERA ### Error loading calibration file: {e}. Proceeding without calibration.")

    def set_resolution(self):
        if self.cap.isOpened():
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"### CAMERA ### Resolution set to {width}x{height}.")

    def set_framerate(self):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            print(f"### CAMERA ### Framerate set to {self.framerate} FPS.")

    def set_auto_exposure(self):
        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change auto-exposure setting.")
            return

        if self.auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Fully enable auto exposure
            print("### CAMERA ### Auto-exposure enabled.")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure
            self.cap.set(cv2.CAP_PROP_GAIN, 50)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            print(f"### CAMERA ### Auto-exposure disabled. Manual exposure set to {self.exposure}.")

    def set_exposure(self, exposure):
        self.exposure = exposure
        if not self.cap.isOpened():
            print("### CAMERA ### Camera is not opened. Cannot change exposure.")
            return False

        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"### CAMERA ### Exposure successfully set to {exposure}.")
        return True

    def preview(self):
        print("### CAMERA ### Starting preview mode. Press 'q' to exit.")
        print("### CAMERA ### Press 'o' to increase exposure, 'p' to decrease exposure.")

        frame_count = 0
        start_time = time.time()
        real_fps = 0

        while True:
            frame = self.get_frame()
            if frame is None:
                print("### CAMERA ### Failed to capture frame.")
                continue

            # Calculate real framerate every 2 seconds
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 2:
                real_fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Display real framerate and resolution on frame
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv2.putText(frame, f"FPS: {real_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.show(frame, "Camera Preview")

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('o'):
                self.set_exposure(self.exposure + 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")
            elif key == ord('p'):
                self.set_exposure(self.exposure - 1)
                print(f"### CAMERA ### Current Exposure: {self.exposure}")

        cv2.destroyAllWindows()

    def get_frame(self, flip=True, if_crop=True):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if flip:
            frame = cv2.flip(frame, 0)
        if if_crop and self.crop:
                frame = frame[self.crop[0]:self.crop[0] + self.crop[2], self.crop[1]:self.crop[1]+self.crop[3]]
        return frame

    @staticmethod
    def show(frame, name='Frame'):
        cv2.imshow(name, frame)

    def wait_key(self, key='q'):
        """
        Waits for a specified key press.

        Parameters:
        -----------
        key (str): Key to detect.

        Returns:
        --------
        bool: True if the specified key was pressed, False otherwise.
        """
        return cv2.waitKey(1) & 0xFF == ord(key)

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def select_roi(self):
        """
        Interactively select an arbitrary rectangular ROI from a single captured frame.
        Press 's' to confirm, 'r' to reset, or ESC to exit without selection.
        The selected ROI is stored in self.crop as (top, left, height, width).
        """
        frame = self.get_frame(if_crop=False)
        temp_frame = frame.copy()
        drawing = False
        if self.crop:
            roi = list(self.crop)
        else:
            roi = [0, 0, 0, 0]

        def draw_rectangle(event, x, y, flags, param):
            nonlocal drawing, roi, frame
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

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", draw_rectangle)

        while True:
            frame = self.get_frame(if_crop=False)
            temp_frame = frame.copy()
            clone = frame.copy()

            temp_frame = frame.copy()
            if drawing or roi[2] != 0 or roi[3] != 0:
                cv2.rectangle(
                    temp_frame,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Select ROI", temp_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('s'):
                x, y, w, h = roi
                if w < 0:
                    x += w
                    w = abs(w)
                if h < 0:
                    y += h
                    h = abs(h)

                x1 = max(0, min(x, frame.shape[1]))
                y1 = max(0, min(y, frame.shape[0]))
                x2 = max(0, min(x + w, frame.shape[1]))
                y2 = max(0, min(y + h, frame.shape[0]))

                cropped_roi = clone[y1:y2, x1:x2]
                if cropped_roi.size == 0:
                    print("### CAMERA ### Invalid ROI. Please select again.")
                    continue

                self.crop = (y1, x1, y2 - y1, x2 - x1)
                print(f'### CAMERA ### Selected ROI: {self.crop}')
                cv2.destroyWindow("Select ROI")
                return
            elif key == ord('r'):
                frame = clone.copy()
                roi = [0, 0, 0, 0]

        cv2.destroyWindow("Select ROI")


if __name__ == "__main__":

    cam = Camera(0,crop=[162, 249, 229, 246])
    cam.select_roi()
    cam.preview()
    cam.close()