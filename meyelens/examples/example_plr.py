from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from psychopy import visual, core, event

from meyelens import (
    Camera,
    DeBlink,
    Filters,
    Meye,
    MeyeReader,
    MeyeRecorder,
    TrialEpochs,
)


# ============================================================
# Pupillary light reflex recording with Meye
# ============================================================

CAMERA_INDEX = 0

N_TRIALS = 5
BASELINE_SEC = 5.0
FLASH_SEC = 0.5
RECOVERY_SEC = 4.0

OUTPUT_FILENAME = "plr_meyept_test" # the default path is Documents/meyeDATA/



def analyze_plr(path):
    """
    Load the saved recording, clean the pupil trace, extract epochs,
    and plot the pupillary light reflex.
    """

    # Read the file created by MeyeRecorder.
    reader = MeyeReader(path)
    df = reader.data

    # Extract time and raw pupil-area trace.
    time = df["t_frame"].to_numpy()
    pupil_raw = df["pupil_area"].to_numpy()

    # Remove blinks and missing-pupil samples.
    # remove_zeros=True is useful because pupil_area=0 usually means
    # that the pupil was not detected.
    deblink = DeBlink(
        threshold=0.25,
        flankers=5,
        remove_zeros=True,
    )
    pupil_clean = deblink.clean(pupil_raw)

    # Estimate sampling frequency from frame timestamps.
    fs = round(1.0 / df["t_frame"].diff().median())

    # Bandpass filter the cleaned pupil trace.
    # Low cutoff removes very slow drift.
    # High cutoff removes fast noise.
    filt = Filters(fs)
    pupil_filtered = filt.bandpass(
        pupil_clean,
        lowcut=0.01,
        highcut=4.0,
    )

    # Extract epochs around flash onset.
    # trg1 marks the first frame of the flash.
    trialer = TrialEpochs(
        signal=pupil_filtered,
        time=time,
        triggers={"light": df["trg1"].to_numpy()},
    )

    epochs = trialer.extract(
        tmin=-5,
        tmax=5,
        baseline=(-5, 0),
        transform="zscore",
        n_points=400,
    )

    # ------------------------------------------------------------
    # Plot the complete recording
    # ------------------------------------------------------------

    plt.figure(figsize=(10, 4))
    plt.plot(time, pupil_raw, alpha=0.3, label="raw")
    plt.plot(time, pupil_filtered, label="clean + filtered")

    # Add one vertical line for each flash onset.
    for flash_time in epochs["event_times"]:
        plt.axvline(flash_time, color="black", linestyle="--", alpha=0.4)

    plt.xlabel("Time (s)")
    plt.ylabel("Pupil area")
    plt.title("Complete pupil trace")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot individual trials and mean response
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Time zero is flash onset.
    plt.axvline(0, color="black", linestyle="--")

    # Horizontal zero is the baseline after z-score transform.
    plt.axhline(0, color="black", linestyle="--")

    # Mark flash duration.
    plt.axvspan(0, FLASH_SEC, alpha=0.15, label="flash")

    # Individual trial traces.
    plt.plot(
        epochs["time"],
        epochs["epochs"].T,
        alpha=0.35,
    )

    # Average response.
    plt.plot(
        epochs["time"],
        epochs["mean"],
        color="black",
        linewidth=2,
        label="mean",
    )

    plt.xlabel("Time from flash onset (s)")
    plt.ylabel("Pupil response, z-score")
    plt.title("Pupillary light reflex")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return epochs


# ============================================================
# Create model and camera
# ============================================================

meye = Meye()

with Camera(camera_index=CAMERA_INDEX) as cam:
    # Select the eye region. Select the ROI with the mouse the confirm with 's'
    cam.select_roi(window_name="Select eye ROI")

    # Preview segmentation before recording.
    # Use this to tune pupil/eye thresholds.
    # Press q or ESC to exit preview.
    meye.preview(cam)

    # ------------------------------------------------------------
    # Create PsychoPy stimuli
    # ------------------------------------------------------------

    win = visual.Window(
        fullscr=True,
        units="height",
        color="black",
    )

    # Small red fixation dot shown throughout the experiment.
    fixation = visual.Circle(
        win,
        radius=0.012,
        fillColor="red",
        lineColor="red",
        pos=(0, 0),
    )

    # White flash stimulus.
    flash_square = visual.Rect(
        win,
        width=0.25,
        height=0.25,
        fillColor="white",
        lineColor="white",
        pos=(0, 0),
    )

    # ------------------------------------------------------------
    # Create recorder
    # ------------------------------------------------------------

    recorder = MeyeRecorder(
        cam=cam,
        meye=meye,
        filename=OUTPUT_FILENAME,
        path_to_file=None,      # default output folder
        show_preview=False,     # set to False for performances
    )

    saved_path = None

    try:
        # Start writing data to disk.
        recorder.start(
            metadata={
                "experiment": "pupillary_light_reflex",
                "n_trials": N_TRIALS,
                "baseline_sec": BASELINE_SEC,
                "flash_sec": FLASH_SEC,
                "recovery_sec": RECOVERY_SEC,
            }
        )

        saved_path = Path(recorder.writer.path)

        # Each trial has three phases:
        # baseline -> flash -> recovery
        trial_duration = BASELINE_SEC + FLASH_SEC + RECOVERY_SEC

        for trial in range(1, N_TRIALS + 1):
            print(f"Trial {trial}/{N_TRIALS}")

            # Trial clock starts from zero for each trial.
            clock = core.Clock()

            # Used to send trg1=1 only once, at flash onset.
            flash_started = False

            while clock.getTime() < trial_duration:
                # Allow the user to stop the experiment.
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

                t = clock.getTime()

                # ------------------------------------------------
                # Decide what phase we are in
                # ------------------------------------------------

                if t < BASELINE_SEC:
                    # Baseline: black screen.
                    flash_on = False
                    trg1 = 0          # no flash onset
                    trg3 = 0          # stimulus state: no flash

                elif t < BASELINE_SEC + FLASH_SEC:
                    # Flash period: white square is visible.
                    flash_on = True
                    trg3 = 1          # stimulus state: flash on

                    # Send a trigger pulse only on the first flash frame.
                    if not flash_started:
                        trg1 = 1      # flash onset trigger
                        flash_started = True
                    else:
                        trg1 = 0

                else:
                    # Recovery: black screen again.
                    flash_on = False
                    trg1 = 0
                    trg3 = 0

                # ------------------------------------------------
                # Draw current frame
                # ------------------------------------------------

                win.color = "black"

                if flash_on:
                    flash_square.draw()

                fixation.draw()

                # Flip shows the selected stimulus state on screen.
                win.flip()

                # ------------------------------------------------
                # Record one synchronized frame/result
                # ------------------------------------------------

                # save_frame captures the camera frame, runs Meye,
                # extracts pupil features, attaches triggers, and writes a row.
                recorder.save_frame(
                    trg1=trg1,        # flash onset pulse
                    trg2=trial,       # trial number
                    trg3=trg3,        # stimulus state
                )

    except KeyboardInterrupt:
        print("Experiment interrupted.")

    finally:
        # Always stop recording and close the PsychoPy window safely.
        recorder.stop()
        win.close()

    # Analyze the recording after the experiment.
    if saved_path is not None and saved_path.exists():
        print(f"Data saved to: {saved_path}")
        epochs = analyze_plr(saved_path)

    # Close camera and writer resources.
    recorder.close_all()
