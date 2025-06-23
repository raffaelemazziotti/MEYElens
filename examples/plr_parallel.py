import numpy as np
import time
import threading
from queue import Queue, Empty
from psychopy import visual
from camera import Camera
from offline import FrameRateManager, FastVideoRecorder

# === Parameters ===
fps = 20
n_trials = 10
dest_folder = "data"
gamma = 2.177573479554467

# Durations in seconds
pretrial_duration = 5
flash_duration = 0.5
posttrial_duration = 5

# Square parameters
square_size = 800




# === Camera capturing and recording threads ===
def camera_capture(cam, frame_queue, stop_event):
    while not stop_event.is_set():
        frame = cam.get_frame()
        if not frame_queue.full():
            frame_queue.put(frame)

def camera_record(recorder, frame_queue, timestamps, trial_phase_queue, stop_event):
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            trial, phase = trial_phase_queue.get_nowait()
            recorder.record_frame(frame, trial, phase)
            timestamps.append(time.time())
        except Empty:
            continue

# === Setup ===
cam1 = Camera(0)
recorder_cam1 = FastVideoRecorder(name="plr_parallel_cam1", fps=fps, dest_folder=dest_folder)
frame_queue_cam1 = Queue(maxsize=10)
trial_phase_queue_cam1 = Queue()
timestamps_cam1 = []
stop_event_cam1 = threading.Event()

cam2 = Camera(1)
recorder_cam2 = FastVideoRecorder(name="plr_parallel_cam2", fps=fps, dest_folder=dest_folder)
frame_queue_cam2 = Queue(maxsize=10)
trial_phase_queue_cam2 = Queue()
timestamps_cam2 = []
stop_event_cam2 = threading.Event()

thread_capture_cam1 = threading.Thread(target=camera_capture, args=(cam1, frame_queue_cam1, stop_event_cam1))
thread_record_cam1 = threading.Thread(target=camera_record, args=(recorder_cam1, frame_queue_cam1, timestamps_cam1, trial_phase_queue_cam1, stop_event_cam1))

thread_capture_cam2 = threading.Thread(target=camera_capture, args=(cam2, frame_queue_cam2, stop_event_cam2))
thread_record_cam2 = threading.Thread(target=camera_record, args=(recorder_cam2, frame_queue_cam2, timestamps_cam2, trial_phase_queue_cam2, stop_event_cam2))

# Preview cameras first
cam1.preview()
cam2.preview()
print("Camera previews closed. Starting PLR paradigm...")

# Start capturing threads after previews
thread_capture_cam1.start()
thread_record_cam1.start()
thread_capture_cam2.start()
thread_record_cam2.start()

# === PsychoPy Window ===
black = [-1, -1, -1]
white = [1, 1, 1]

win = visual.Window(fullscr=True, color=black, units='pix', gamma=gamma)
fixation = visual.Circle(win, radius=5, fillColor='red', lineColor='red', pos=(0, 0))
flash_square = visual.Rect(win, width=square_size, height=square_size, fillColor=white, lineColor=white, pos=(0, 0))

frm = FrameRateManager(fps)

# Helper to enqueue trial-phase info
def enqueue_trial_phase(trial, phase):
    trial_phase_queue_cam1.put((trial, phase))
    trial_phase_queue_cam2.put((trial, phase))

# === Pretrial ===
frm.duration = pretrial_duration
frm.start()
while not frm.is_finished():
    if frm.is_ready():
        fixation.draw()
        win.flip()
        enqueue_trial_phase(-1, "pretrial")
        frm.set_frame_time(overhead=0)

# === Trials ===
for trial in range(n_trials):
    print(f"Starting Trial {trial+1}/{n_trials}")

    # Flash
    frm.duration = flash_duration
    frm.start()
    while not frm.is_finished():
        if frm.is_ready():
            flash_square.draw()
            fixation.draw()
            win.flip()
            enqueue_trial_phase(trial, "flash")
            frm.set_frame_time(overhead=0)

    # Posttrial
    frm.duration = posttrial_duration
    frm.start()
    while not frm.is_finished():
        if frm.is_ready():
            fixation.draw()
            win.flip()
            enqueue_trial_phase(trial, "posttrial")
            frm.set_frame_time(overhead=0)

# Cleanup
stop_event_cam1.set()
stop_event_cam2.set()

thread_capture_cam1.join()
thread_record_cam1.join()
thread_capture_cam2.join()
thread_record_cam2.join()

print("Experiment completed.")