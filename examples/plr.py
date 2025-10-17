import time
from psychopy import visual
from meyelens.camera import Camera
from meyelens.offline import FastVideoRecorder, FrameRateManager

# === Parameters ===
fps = 20
n_trials = 10
intensity = 0 #stimulation intensity, 1 is white, -1 is black
dest_folder = "data"
gamma = 2.177573479554467

# Define durations in seconds
pretrial_duration = 5      # 5 seconds
flash_duration = 0.5       # 0.5 seconds
posttrial_duration = 5     # 5 seconds

# Square parameters
square_size = 800

# === Setup ===
cam = Camera(0)
black = [-1, -1, -1]
white = [intensity, intensity, intensity]
timestamps = []

recorder = FastVideoRecorder(name=f"plr_{intensity}", fps=fps, dest_folder=dest_folder)
cam.preview()
print("Camera preview closed. Starting PLR paradigm...")

# === PsychoPy Window ===
win = visual.Window(fullscr=True, color=black, units='pix', gamma=gamma)
fixation = visual.Circle(win, radius=5, fillColor='red', lineColor='red', pos=(0, 0))
flash_square = visual.Rect(win, width=square_size, height=square_size, fillColor=white, lineColor=white, pos=(0, 0))


frm = FrameRateManager(fps)

# === Pretrial ===
frm.duration = pretrial_duration
frm.start()
while not frm.is_finished():
    if frm.is_ready():
        fixation.draw()
        win.flip()

        frame = cam.get_frame()
        timestamps.append(time.time())
        recorder.record_frame(frame, signal = "pretrial", trial_n=-1)
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

            frame = cam.get_frame()
            timestamps.append(time.time())
            recorder.record_frame(frame, trial_n=trial, signal="flash")
            frm.set_frame_time(overhead=0)

    # Posttrial
    frm.duration = posttrial_duration
    frm.start()
    while not frm.is_finished():
        if frm.is_ready():
            fixation.draw()
            win.flip()

            frame = cam.get_frame()
            timestamps.append(time.time())
            recorder.record_frame(frame, trial_n= trial, signal ="posttrial")
            frm.set_frame_time(overhead=0)

print("Experiment completed.")
