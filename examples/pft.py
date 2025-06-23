import numpy as np
import random
import time
from psychopy import visual
from camera import Camera
from offline import FrameRateManager, FastVideoRecorder
# === Parameters ===

trial_duration = 20
int_duration = 5
fps = 20
gamma = 2.177573479554467
timestamps = []
frequencies = [1, 1.1111, 1.25, 1.4286, 1.6667, 2, 2.5, 3.3333, 5]
randomize_frequencies = True




cam = Camera(0)
black = [-1, -1, -1]
white = [1, 1, 1]
gray = [0, 0, 0]

frm = FrameRateManager(fps, trial_duration)
frm_int = FrameRateManager(fps, int_duration)
recorder = FastVideoRecorder(name="frequency_sweep", fps=fps, dest_folder="data")

square_size = 800  # Square size in pixels

cam.preview()
print("Camera preview closed. Starting paradigm...")

# === PsychoPy Window Setup ===
win = visual.Window(fullscr=True, color=gray, units='pix', gamma=gamma)

fixation = visual.Circle(win, radius=5, fillColor='red', lineColor='red', pos=(0, 0))
square = visual.Rect(win, width=square_size, height=square_size, fillColor=black, lineColor=black, pos=(0, 0))

if randomize_frequencies:
    random.shuffle(frequencies)

for freq in frequencies:
    # === Intertrial period ===
    frm.duration = int_duration
    frm.start()
    print(f"Start intertrial before stimulation at {freq}Hz")
    print(freq)
    while not frm.is_finished():
        if frm.is_ready():
            fixation.draw()
            win.flip()

            frame = cam.get_frame()
            timestamps.append(time.time())
            recorder.record_frame(frame, np.NaN, np.NaN)

            frm.set_frame_time(overhead=0)

    # === Stimulation period ===
    frm.duration = trial_duration
    frm.start()
    square_color = white
    frame_counter = 0
    frames_per_half_cycle = round((fps / freq) / 2)
    print(frames_per_half_cycle)

    print(f"Start stimulation at {freq}Hz")

    while not frm.is_finished():
        if frm.is_ready():
            square.fillColor = square_color
            square.lineColor = square_color

            square.draw()
            fixation.draw()
            win.flip()

            frame = cam.get_frame()
            timestamps.append(time.time())
            recorder.record_frame(frame, freq, square_color == white)

            frame_counter += 1
            if frame_counter >= frames_per_half_cycle:
                square_color = black if square_color == white else white
                frame_counter = 0

            frm.set_frame_time(overhead=0)

print("Experiment completed.")
