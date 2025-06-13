import cv2
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class ExperimentReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.video_path = os.path.join(folder_path, "pupillometry.avi")
        self.csv_path = os.path.join(folder_path, "expinfo.csv")
        self.metadata = {}
        self.frame_info = []  # List of dicts with keys: frame_index, timestamp, signal
        self.frame_info = pd.read_csv(self.csv_path, comment='#')
        self.fps = np.mean(1/self.frame_info['timestamp'].diff().loc[1:].values)

        with open(self.csv_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    parts = line[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        self.metadata[key] = value
                else:
                   break
        self.cap = cv2.VideoCapture(self.video_path)

    def __getitem__(self, index):
        return (self.frame_info.loc[index],self.get_frame(index))

    def __len__(self):
        return self.get_frame_count()

    def get_metadata(self):
        return self.metadata

    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def __iter__(self):
        for index in range(0,self.get_frame_count()):
            yield (self.frame_info.loc[index],self.get_frame(index))

    def play_video(self, delay=30, repeat=True):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            signal = None
            if frame_idx < len(self.frame_info):
                signal = self.frame_info['signal'].loc[frame_idx]
            if signal is not None:
                cv2.putText(frame, f"Signal: {signal}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
            cv2.imshow("Playback", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            if frame_idx<self.get_frame_count():
                frame_idx += 1
            elif repeat:
                frame_idx = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                break

        cv2.destroyAllWindows()

    def visualize_fps_stability(self):
        if len(self.frame_info) < 2:
            print("Not enough frame info to calculate FPS.")
            return

        timestamps = np.array([entry["timestamp"] for entry in self.frame_info])
        dt = np.diff(timestamps)
        instantaneous_fps = 1.0 / dt

        mid_times = (timestamps[:-1] + timestamps[1:]) / 2
        mean_fps = np.mean(instantaneous_fps)
        stability_index = 1.0 - (np.std(instantaneous_fps) / mean_fps)
        stable_indices = np.where(np.abs(instantaneous_fps - mean_fps) < 0.05 * mean_fps)[0]

        plt.figure()
        plt.plot(mid_times, instantaneous_fps, label="Instantaneous FPS")

        plt.xlabel("Time (s)")
        plt.ylabel("FPS")
        plt.show()

    def visualize_triggers(self):
        if not self.frame_info:
            print("No frame info available to visualize triggers.")
            return

        frame_indices = np.array([entry["frame_index"] for entry in self.frame_info])
        triggers = np.array([entry["signal"] for entry in self.frame_info])

        plt.figure()
        plt.plot(frame_indices, triggers, marker='o', linestyle='-', label="Trigger values")
        plt.xlabel("Frame Index")
        plt.ylabel("Trigger Value")
        plt.title("Trigger Values Over Frames")
        plt.legend()
        plt.show()

    def close(self):
        self.cap.release()

class FastVideoRecorder:
    def __init__(self, fname="experiment", dest_folder=".", fps=30.0, frame_size=(640, 480), metadata=None):
        # Create the destination folder if it does not exist.
        os.makedirs(dest_folder, exist_ok=True)

        # Create a new folder with a timestamp and the user-defined fname.
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(dest_folder, f"{timestamp_str}-{fname}")
        os.makedirs(self.output_folder, exist_ok=True)

        # Hardcoded file names: pupillometri.avi for video and expinfo.csv for timestamps and signals.
        self.video_path = os.path.join(self.output_folder, "pupillometry.avi")
        self.timestamp_path = os.path.join(self.output_folder, "expinfo.csv")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, frame_size, isColor=False)

        # Open CSV file for writing metadata and frame info.
        self.timestamp_fh = open(self.timestamp_path, "w")
        if metadata is not None:
            for key, value in metadata.items():
                self.timestamp_fh.write(f"# {key}: {value}\n")
        self.timestamp_fh.write("frame_index,timestamp,signal\n")
        self.frame_index = 0

    def record_frame(self, frame, signal):
        current_time = time.time()
        self.timestamp_fh.write(f"{self.frame_index},{current_time},{signal}\n")
        self.frame_index += 1

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        self.timestamp_fh.close()

class FrameRateManager:

    def __init__(self,fps, duration=10):
        self.fps = fps
        self.interframe = 1/fps
        self.time_grab = 0
        self.framecount=0
        self.duration = duration

    def start(self):
        self.nextframetime = time.time()
        self.start_time = time.time()
        self.loop_duration = time.time() + self.duration

    def is_ready(self):
        isready = time.time() >= self.nextframetime
        if isready:
            self.time_grab = time.time()
            self.framecount += 1
        return isready

    def set_frame_time(self,overhead=0.0005):
        self.grab_dur = time.time() - self.time_grab
        self.nextframetime = time.time() + self.interframe - (self.grab_dur) - overhead

    def is_finished(self):
        finished = time.time()>=self.loop_duration
        if finished:
            self.end_time = time.time() - self.start_time
            print('Duration:', self.end_time, 'Number of Frames:', self.framecount)
        return finished