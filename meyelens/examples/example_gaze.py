from meyelens import Camera, Meye, MeyeGazeCalibrator
from psychopy import visual, event, core
from meyelens import AdaptiveGazeFilter
import time, math


# ----------------------------
# 1 CAM gaze estimation - Settings
# ----------------------------

MODEL_TYPE = "mlp"   # "ridge_poly", "ridge", or "mlp"

NUM_POINTS = 20
FIXATION_DELAY_SEC = 0.45   # wait after target appears before collecting
SAMPLES_PER_TARGET = 60     # number of valid Meye samples per target

SCREEN_WIDTH = 1.8          # PsychoPy norm coordinates: -0.9 to +0.9
SCREEN_HEIGHT = 1.8

# move elsewhere


# ----------------------------
# Objects
# ----------------------------

meye = Meye()
gaze_filter = AdaptiveGazeFilter(
    min_cutoff=0.8,
    beta=0.08,
    d_cutoff=1.0,
    jump_reset=0.6,
)


calibrator = MeyeGazeCalibrator(
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT,
    num_points=NUM_POINTS,
    model_type=MODEL_TYPE,
    random_points=False,
)

win = visual.Window(
    size=(1000, 1000),
    units="norm",
    color="black",
    fullscr=False,
)

target_dot = visual.Circle(
    win,
    radius=0.035,
    fillColor="red",
    lineColor="red",
)

gaze_dot = visual.Circle(
    win,
    radius=0.025,
    fillColor="green",
    lineColor="green",
)


# ----------------------------
# Calibration
# ----------------------------

with Camera(camera_index=0) as cam:
    # Select eye ROI first.
    cam.select_roi()

    # Optional: tune thresholds before calibration.
    meye.preview(cam)

    # Go through each calibration target.
    while True:
        target = calibrator.next_target()

        if target is None:
            break

        target_x = target["target_x"]
        target_y = target["target_y"]

        target_dot.pos = (target_x, target_y)

        # 1. Show target in red and wait before collecting.
        target_dot.fillColor = "red"
        target_dot.lineColor = "red"

        timer = core.Clock()
        while timer.getTime() < FIXATION_DELAY_SEC:
            target_dot.draw()
            win.flip()

            if "escape" in event.getKeys():
                win.close()
                core.quit()

        # 2. Change target to green when sample collection starts.
        target_dot.fillColor = "green"
        target_dot.lineColor = "green"

        n_collected = 0

        while n_collected < SAMPLES_PER_TARGET:
            target_dot.draw()
            win.flip()

            frame = cam.get_frame()

            if frame is None:
                continue

            result = meye.predict(frame)

            try:
                calibrator.add_sample(result)
                n_collected += 1
            except Exception:
                # Skip frames where pupil/eye features are invalid.
                continue

            if "escape" in event.getKeys():
                win.close()
                core.quit()

    # Fit using the model selected above.
    fit_info = calibrator.fit(
        model_type=MODEL_TYPE,
        degree=2,
        skip_samples=0,
        aggregate="median",
    )

    print("Fit info:", fit_info)

    # ----------------------------
    # Live prediction test
    # ----------------------------

    last_gaze = (0.0, 0.0)

    while True:
        frame = cam.get_frame()

        if frame is None:
            continue

        result = meye.predict(frame)

        try:
            gaze_x, gaze_y = calibrator.predict_xy(result)
            last_gaze = (gaze_x, gaze_y)
        except RuntimeError:
            # Current frame has NaN features.
            # Keep showing the last valid prediction.
            gaze_x, gaze_y = last_gaze

        gaze_dot.pos = (gaze_x, gaze_y)
        gaze_dot.pos = gaze_filter.update(gaze_x, gaze_y)

        gaze_dot.draw()
        win.flip()

        if "escape" in event.getKeys():
            break

win.close()
