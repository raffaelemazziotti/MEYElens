"""
Raw eye-video recording with EyeVideoRecorder.

This example records the eye camera stream without running MEYELens pupil/eye
segmentation. It saves:

    1. a raw eye video;
    2. a CSV file with one trigger/timing row per saved video frame.

Default output structure
------------------------

The recorder creates one timestamped session folder in the standard MEYELens
data directory:

    ~/Documents/meyeDATA/YYYYMMDD_HHMMSS-eye_video/

Inside the session folder:

    video.mp4
        Raw eye movie.

    triggers.csv
        Frame-by-frame trigger and timing table.

Why use this recorder?
----------------------

Use EyeVideoRecorder when you want to:

    - save raw eye videos for later offline analysis;
    - avoid online pupil segmentation during the experiment;
    - keep acquisition as fast and lightweight as possible;
    - synchronize frame-wise triggers with the saved video.

The recorder does not need a Meye object and does not call Meye.predict().
"""

from meyelens import Camera, EyeVideoRecorder


# ============================================================
# Recording settings
# ============================================================

CAMERA_INDEX = 0

TARGET_FPS = 30
RESOLUTION = (640, 480)

N_FRAMES = 300

OUTPUT_FILENAME = "eye_video"


# ============================================================
# Create camera and recorder
# ============================================================

with Camera(
    camera_index=CAMERA_INDEX,
    framerate=TARGET_FPS,
    resolution=RESOLUTION,
) as cam:

    # Select the eye region.
    #
    # Controls:
    #   drag with left mouse button : draw ROI
    #   s                          : save ROI
    #   r                          : reset ROI
    #   ESC                        : cancel ROI selection
    #
    # The selected crop is applied automatically to every frame returned
    # by cam.get_frame(), and therefore also to the saved video.
    cam.select_roi(window_name="Select eye ROI")

    # Create the raw eye-video recorder.
    #
    # This recorder only saves camera frames and trigger values.
    # It does not create a Meye object and does not run segmentation.
    recorder = EyeVideoRecorder(
        cam=cam,
        filename=OUTPUT_FILENAME,
        fps=TARGET_FPS,
        show_preview=True,
    )

    try:
        # Start the recording session.
        #
        # This creates:
        #
        #   ~/Documents/meyeDATA/YYYYMMDD_HHMMSS-eye_video/video.mp4
        #   ~/Documents/meyeDATA/YYYYMMDD_HHMMSS-eye_video/triggers.csv
        #
        # Metadata are written at the top of triggers.csv as comment lines.
        recorder.start(
            metadata={
                "subject": "S01",
                "condition": "raw_eye_video_test",
                "camera_index": CAMERA_INDEX,
                "target_fps": TARGET_FPS,
                "resolution": RESOLUTION,
                "n_requested_frames": N_FRAMES,
                "trg1": "example event pulse",
                "trg2": "frame counter",
                "trg3": "example block code",
            }
        )

        # ------------------------------------------------------------
        # Recording loop
        # ------------------------------------------------------------
        #
        # Each call to save_frame():
        #
        #   1. acquires one frame from the camera;
        #   2. writes that frame to video.mp4;
        #   3. appends one row to triggers.csv.
        #
        # A trigger row is written only if the frame was successfully
        # acquired and written to the video.
        #
        # Here we create a simple example:
        #
        #   trg1 = 1 only at frame 100
        #   trg2 = current frame number
        #   trg3 = block/condition code
        #
        for frame_number in range(N_FRAMES):

            event_pulse = 1 if frame_number == 100 else 0
            block_code = 1

            ok = recorder.save_frame(
                trg1=event_pulse,
                trg2=frame_number,
                trg3=block_code,
            )

            if not ok:
                print(f"Frame {frame_number}: camera frame was not acquired.")

    finally:
        # Always close the recorder safely.
        #
        # This flushes and closes triggers.csv and releases the video writer.
        # The Camera itself is released automatically by the with Camera(...) block.
        recorder.close_all()


# ============================================================
# Print output paths
# ============================================================

print("")
print("Raw eye-video recording completed.")
print(f"Session folder: {recorder.session_dir}")
print(f"Video file:     {recorder.video_path}")
print(f"Trigger file:   {recorder.trigger_path}")