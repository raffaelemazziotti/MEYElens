import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentReader:
    """
    Read back a recorded experiment folder (video + per-frame CSV + metadata).

    This class is designed to mirror the output structure produced by
    :class:`FastVideoRecorder`. It expects a folder containing:

    - a video file (default name used here: ``pupillometry.avi``)
    - a CSV file named ``expinfo.csv`` with:
        - optional metadata lines starting with ``#`` in the form ``# key: value``
        - a header row
        - per-frame rows containing at least a ``timestamp`` column

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Folder containing the recorded video and ``expinfo.csv``.

    Attributes
    ----------
    folder_path : str
        Base folder of the recording.
    video_path : str
        Path to the video file.
    csv_path : str
        Path to the CSV file with timestamps/signals.
    metadata : dict
        Metadata parsed from comment lines in the CSV.
    frame_info : pandas.DataFrame
        Frame-by-frame table loaded from the CSV (comment lines ignored).
    fps : float
        Estimated FPS computed from timestamp differences.
    cap : cv2.VideoCapture
        OpenCV video capture handle.

    Notes
    -----
    - ``fps`` is estimated as the mean of ``1 / diff(timestamp)``; this assumes
      timestamps are in seconds and monotonic.
    - This class does not automatically close the capture: call :meth:`close`.
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path

        # Expected file structure.
        self.video_path = os.path.join(folder_path, "pupillometry.avi")
        self.csv_path = os.path.join(folder_path, "expinfo.csv")

        self.metadata = {}

        # Load per-frame info, ignoring comment lines.
        self.frame_info = pd.read_csv(self.csv_path, comment="#")

        # Estimate FPS from timestamp differences.
        self.fps = np.mean(1 / self.frame_info["timestamp"].diff().loc[1:].values)

        # Parse metadata from comment lines at the top of the CSV.
        with open(self.csv_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    parts = line[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        self.metadata[key] = value
                else:
                    # Stop at first non-comment line (header row).
                    break

        self.cap = cv2.VideoCapture(self.video_path)

    def __getitem__(self, index):
        """
        Random-access a frame and its corresponding CSV row.

        Parameters
        ----------
        index : int
            Frame index.

        Returns
        -------
        tuple[pandas.Series, numpy.ndarray or None]
            ``(frame_info_row, frame)``.
        """
        return (self.frame_info.loc[index], self.get_frame(index))

    def __len__(self):
        """
        Return the number of frames in the video.

        Returns
        -------
        int
            Frame count from OpenCV metadata.
        """
        return self.get_frame_count()

    def get_metadata(self):
        """
        Return parsed metadata.

        Returns
        -------
        dict
            Metadata dictionary from the CSV comment lines.
        """
        return self.metadata

    def get_frame_count(self):
        """
        Get total number of frames in the video.

        Returns
        -------
        int
            Number of frames according to OpenCV.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, index):
        """
        Retrieve a frame by index.

        Parameters
        ----------
        index : int
            Frame index (0-based).

        Returns
        -------
        numpy.ndarray or None
            The frame (as returned by OpenCV), or ``None`` if reading fails.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def __iter__(self):
        """
        Iterate over all frames, yielding CSV row + frame.

        Yields
        ------
        tuple[pandas.Series, numpy.ndarray or None]
            ``(frame_info_row, frame)`` for each frame index.
        """
        for index in range(0, self.get_frame_count()):
            yield (self.frame_info.loc[index], self.get_frame(index))

    def play_video(self, delay: int = 30, repeat: bool = True):
        """
        Play the recorded video with an overlay of the per-frame ``signal`` value.

        Parameters
        ----------
        delay : int, optional
            Delay passed to :func:`cv2.waitKey` in milliseconds. Smaller values play faster.
        repeat : bool, optional
            If ``True``, loop the video when it ends.

        Notes
        -----
        Press ``q`` to quit playback.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            signal = None
            if frame_idx < len(self.frame_info):
                # "signal" is assumed to exist as a column in expinfo.csv.
                signal = self.frame_info["signal"].loc[frame_idx]

            if signal is not None:
                cv2.putText(
                    frame,
                    f"Signal: {signal}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255),
                    2,
                )

            cv2.imshow("Playback", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

            # Advance frame index and handle repeat logic.
            if frame_idx < self.get_frame_count():
                frame_idx += 1
            elif repeat:
                frame_idx = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                break

        cv2.destroyAllWindows()

    def visualize_fps_stability(self):
        """
        Plot instantaneous FPS over time derived from recorded timestamps.

        This function computes:

        - ``dt = diff(timestamps)``
        - ``instantaneous_fps = 1 / dt``

        Then plots instantaneous FPS against the mid-time between consecutive frames.

        Notes
        -----
        This method expects that ``timestamp`` is a numeric column in seconds.

        Potential issue in original code
        --------------------------------
        The previous implementation attempted to iterate over ``self.frame_info`` as if it
        were a list of dicts (``entry["timestamp"]``). Here we use DataFrame columns directly.
        """
        if len(self.frame_info) < 2:
            print("Not enough frame info to calculate FPS.")
            return

        timestamps = self.frame_info["timestamp"].to_numpy(dtype=float)
        dt = np.diff(timestamps)

        # Guard against zero or negative dt (could happen if timestamps are malformed).
        valid = dt > 0
        if not np.all(valid):
            dt = dt[valid]
            mid_times = ((timestamps[:-1] + timestamps[1:]) / 2)[valid]
        else:
            mid_times = (timestamps[:-1] + timestamps[1:]) / 2

        instantaneous_fps = 1.0 / dt

        mean_fps = np.mean(instantaneous_fps)
        stability_index = 1.0 - (np.std(instantaneous_fps) / mean_fps) if mean_fps > 0 else np.nan

        # "stable_indices" kept for potential downstream usage/debugging.
        stable_indices = np.where(np.abs(instantaneous_fps - mean_fps) < 0.05 * mean_fps)[0]

        plt.figure()
        plt.plot(mid_times, instantaneous_fps, label="Instantaneous FPS")
        plt.xlabel("Time (s)")
        plt.ylabel("FPS")
        plt.show()

    def visualize_triggers(self):
        """
        Plot trigger/signal values across frame indices.

        Notes
        -----
        Potential issue in original code:
        ``if not self.frame_info:`` is ambiguous for a DataFrame. Here we use ``empty``.
        Also, the original code treated ``frame_info`` like a list of dicts; this version
        uses DataFrame columns.
        """
        if self.frame_info is None or self.frame_info.empty:
            print("No frame info available to visualize triggers.")
            return

        # These columns are expected to be present in expinfo.csv.
        frame_indices = self.frame_info["frame_index"].to_numpy()
        triggers = self.frame_info["signal"].to_numpy()

        plt.figure()
        plt.plot(frame_indices, triggers, marker="o", linestyle="-", label="Trigger values")
        plt.xlabel("Frame Index")
        plt.ylabel("Trigger Value")
        plt.title("Trigger Values Over Frames")
        plt.legend()
        plt.show()

    def close(self):
        """
        Release the OpenCV video capture handle.

        Returns
        -------
        None
        """
        self.cap.release()


class FastVideoRecorder:
    """
    Simple video + CSV recorder for experiments.

    This class writes:

    - a grayscale video file (MJPG codec)
    - a CSV file named ``expinfo.csv`` with optional metadata comment lines and
      one row per recorded frame.

    The output folder is created as::

        <dest_folder>/<timestamp>-<name>/

    Parameters
    ----------
    name : str, optional
        Name appended to the output folder.
    dest_folder : str, optional
        Base destination directory.
    fps : float, optional
        Target frames per second passed to OpenCV VideoWriter.
    frame_size : tuple[int, int], optional
        Frame size (width, height) expected by OpenCV VideoWriter.
    metadata : dict or None, optional
        Optional metadata written to the top of ``expinfo.csv`` as ``# key: value``.
    filename : str, optional
        Video filename inside the output folder.

    Attributes
    ----------
    output_folder : str
        Created output folder path.
    video_path : str
        Full path to the recorded video file.
    timestamp_path : str
        Full path to the per-frame CSV file (``expinfo.csv``).
    frame_index : int
        Incremented each time :meth:`record_frame` is called.
    """

    def __init__(
        self,
        name="experiment",
        dest_folder=".",
        fps=20.0,
        frame_size=(640, 480),
        metadata=None,
        filename="eye.avi",
    ):
        # Ensure base destination exists.
        os.makedirs(dest_folder, exist_ok=True)

        # Create per-recording folder.
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(dest_folder, f"{timestamp_str}-{name}")
        os.makedirs(self.output_folder, exist_ok=True)

        self.video_path = os.path.join(self.output_folder, filename)
        self.timestamp_path = os.path.join(self.output_folder, "expinfo.csv")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # We write grayscale frames (isColor=False). OpenCV expects frame_size as (width, height).
        self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, frame_size, isColor=False)

        # Open CSV file and write metadata (comment lines) + header.
        self.timestamp_fh = open(self.timestamp_path, "w")
        if metadata is not None:
            for key, value in metadata.items():
                self.timestamp_fh.write(f"# {key}: {value}\n")

        self.timestamp_fh.write("frame_index,timestamp,signal,trial\n")
        self.frame_index = 0

    def record_frame(self, frame, signal="", trial_n=""):
        """
        Record a frame to video and append a row to ``expinfo.csv``.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame to write. If BGR, it is converted to grayscale before writing.
        signal : str or int, optional
            Signal/trigger value to store for this frame.
        trial_n : str or int, optional
            Trial identifier to store for this frame.

        Notes
        -----
        - Timestamp is recorded using :func:`time.time` (seconds since epoch).
        - ``frame_index`` starts at 0 and increments per call.
        """
        current_time = time.time()
        self.timestamp_fh.write(f"{self.frame_index},{current_time},{signal},{trial_n}\n")
        self.frame_index += 1

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.writer.write(frame)

    def release(self):
        """
        Release the video writer and close the CSV file.

        Returns
        -------
        None
        """
        self.writer.release()
        self.timestamp_fh.close()


class FrameRateManager:
    """
    Utility class to help maintain a target frame rate in a polling loop.

    Typical usage
    -------------
    >>> frm = FrameRateManager(fps=30, duration=5)
    >>> frm.start()
    >>> while not frm.is_finished():
    ...     if frm.is_ready():
    ...         # acquire frame / do work here
    ...         frm.set_frame_time()

    Parameters
    ----------
    fps : float
        Target frames per second.
    duration : float, optional
        Maximum loop duration in seconds for :meth:`is_finished`.

    Attributes
    ----------
    fps : float
        Target FPS.
    interframe : float
        Target inter-frame interval (seconds).
    time_grab : float
        Timestamp saved when a new frame cycle begins (set in :meth:`is_ready`).
    duration : float
        Duration used to determine loop end.
    framecount : int
        Counts how many times the loop was "ready" (i.e., frames acquired).
    """

    def __init__(self, fps, duration: float = 10):
        self.fps = fps
        self.interframe = 1 / fps
        self.time_grab = 0
        self.duration = duration

    def start(self):
        """
        Initialize timing variables and reset counters for a new run.

        Returns
        -------
        None
        """
        self.nextframetime = time.time()
        self.start_time = self.nextframetime
        self.loop_duration = self.nextframetime + self.duration
        self.framecount = 0

    def is_ready(self) -> bool:
        """
        Check if it is time to process/acquire the next frame.

        Returns
        -------
        bool
            ``True`` if current time has reached the next scheduled frame time.
            When ``True``, also updates internal counters.
        """
        isready = time.time() >= self.nextframetime
        if isready:
            self.time_grab = time.time()
            self.framecount += 1
        return isready

    def set_frame_time(self, overhead: float = 0.0005):
        """
        Schedule the next frame time based on processing overhead.

        This method measures time elapsed since :meth:`is_ready` last set
        :attr:`time_grab` and subtracts it from the nominal inter-frame interval.

        Parameters
        ----------
        overhead : float, optional
            Small constant to compensate for additional overhead (seconds).

        Returns
        -------
        None
        """
        self.grab_dur = time.time() - self.time_grab
        self.nextframetime = time.time() + self.interframe - (self.grab_dur) - overhead

    def is_finished(self) -> bool:
        """
        Check whether the run duration has elapsed.

        Returns
        -------
        bool
            ``True`` if the current time is beyond the configured duration.

        Notes
        -----
        When finished, prints the actual duration and number of frames processed.
        """
        finished = time.time() >= self.loop_duration
        if finished:
            self.end_time = time.time() - self.start_time
            print("Duration:", self.end_time, "Number of Frames:", self.framecount)
        return finished
