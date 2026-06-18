from meyelens import Camera, Meye


# Use a custom MEYELens model.
# The model path is saved in the MEYELens config file, so future sessions
# will use this model unless you change it again.
meye = Meye(model_path="path/to/your_custom_model.pt")


# Open the camera.
with Camera(camera_index=0) as cam:

    # Select the eye region.
    cam.select_roi()

    # Preview segmentation with the custom model.
    # Use this to tune thresholds and check the prediction quality.
    # Exit the preview with "q" or ESC.
    meye.preview(cam)

    # Acquire one frame.
    frame = cam.get_frame()

    if frame is not None:
        # Run prediction with the custom model.
        result = meye.predict(frame)

        print("Inference FPS:", result.inference_fps)
        print("Features:")
        print(result.features)