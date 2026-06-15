from meyelens import Camera, Meye

# Create the segmentation model.
meye = Meye()

# Open the camera.
with Camera(camera_index=0) as cam:

    # select the roi
    cam.select_roi()

    # Run the normal interactive Meye preview first.
    # Here you can tune thresholds and toggle mask/features.
    # Exit the preview with "q" or ESC.
    meye.preview(cam)

    # After closing the preview, acquire 30 feature samples.
    # Each sample contains the current pupil/eye area, centroid,
    # ellipse parameters, and ellipse-axis points.
    points = []

    for _ in range(30):
        frame = cam.get_frame()

        if frame is None:
            continue

        result = meye.predict(frame)
        points.append(result.features)

    # Print the 30 collected samples and exit.
    print("\n### 30 FEATURE SAMPLES ###")

    for i, sample in enumerate(points, 1):
        print(f"\nSample {i}")
        print(sample)
