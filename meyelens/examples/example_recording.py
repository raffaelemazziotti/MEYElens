from meyelens import Camera, Meye, MeyeRecorder


# Create the pupil/eye segmentation model.
# The model path and tuning parameters are loaded from the Meye TOML config.
meye = Meye()

# Open the camera.
# The camera is automatically released when the with-block ends.
with Camera(camera_index=0) as cam:

    # Select the region of interest.
    # Drag with the mouse, press "s" to save, "r" to reset, or ESC to cancel.
    cam.select_roi()

    # Optional preview before recording.
    # Use this to tune thresholds and visualization settings.
    # Exit the preview with "q" or ESC.
    meye.preview(cam)

    # Create the recorder.
    recorder = MeyeRecorder(
        cam=cam, # camera instance
        meye=meye, # meye instance
        filename="meye_recording", # file name that is appended at the end of the filex
        show_preview=False, # show_preview=True displays the Meye overlay while frames are saved. Set to False is faster.
    )

    # Start the recording file and save metadata.
    recorder.start(metadata={"subject": "S01", "condition": "test"})

    # Record 300 frames.
    # trg1 is set to 1 only at frame 100; otherwise it is 0.
    for i in range(300):
        recorder.save_frame(trg1=1 if i == 100 else 0)

    # Stop recording, flush the file writer, and close the camera.
    recorder.close_all()
