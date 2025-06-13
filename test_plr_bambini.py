import cv2
from psychopy import visual, core, event
import numpy as np
from ffpyplayer.player import MediaPlayer

# Load video and audio
video_path = r"C:\Users\pupil\Downloads\ciuffi_puppa.mp4"
cap = cv2.VideoCapture(video_path)
player = MediaPlayer(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1.0 / fps

# Get original frame size and aspect ratio
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
aspect_ratio = frame_width / frame_height

# Create PsychoPy window
win = visual.Window(fullscr=True, color=[-1,-1,-1], units="pix")
screen_width, screen_height = win.size

# Resize keeping aspect ratio
scale = 0.2
target_height = screen_height * scale
target_width = target_height * aspect_ratio
video_size = (target_width, target_height)

# Flash settings
flash_every_n_frames = 120    # every 120 frames (≈2 seconds at 60Hz)
flash_duration = 10            # show white for 3 frames
frame_counter = 0

# Playback loop
while cap.isOpened():
    audio_frame, val = player.get_frame()
    if val == 'eof':
        break

    audio_pts = player.get_pts()
    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_time = current_frame_idx * frame_duration

    if video_time > audio_pts:
        core.wait(0.001)
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # Background switching logic
    if (frame_counter % flash_every_n_frames) < flash_duration:
        win.color = [1, 1, 1]  # white
    else:
        win.color = [-1,-1,-1]  # gray

    #win.flip(clearBuffer=True)

    # Convert to grayscale and normalize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = (gray.astype(float) - 128) / 128
    gray_rgb = cv2.flip(gray, 0)

    stim = visual.ImageStim(win, image=gray_rgb, size=video_size, pos=(0, 0), colorSpace='rgb')
    stim.draw()
    win.flip()

    frame_counter += 1

    if "escape" in event.getKeys():
        break

cap.release()
win.close()
core.quit()
