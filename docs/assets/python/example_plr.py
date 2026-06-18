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
OUTPUT_FILENAME = "plr_meyept_test"  # default path: Documents/meyeDATA/


# ============================================================
# Experiment timing parameters
# ============================================================

N_TRIALS = 5

# Initial recording before the trial sequence starts.
# This is useful to stabilize the pupil trace before the first event.
BASELINE_SEC = 5.0

# Per-trial timing.
# Each trial follows:
# prestim -> stim -> poststim
PRESTIM_SEC = 2.0
STIM_SEC = 0.5
POSTSTIM_SEC = 3.0


# ============================================================
# Processing parameters
# ============================================================

DEBLINK_THRESHOLD = 0.25
DEBLINK_FLANKERS = 5
DEBLINK_REMOVE_ZEROS = True

FILTER_LOWCUT_HZ = 0.01
FILTER_HIGHCUT_HZ = 4.0


# ============================================================
# Epoch extraction and display parameters
# ============================================================

# Epochs are extracted around stimulus onset.
# Because each trial includes PRESTIM_SEC before the stimulus and POSTSTIM_SEC
# after the stimulus, these values should not exceed those limits.
EPOCH_TMIN = -PRESTIM_SEC
EPOCH_TMAX = POSTSTIM_SEC

EPOCH_BASELINE = (-PRESTIM_SEC, 0.0)
EPOCH_TRANSFORM = "zscore"
EPOCH_N_POINTS = 400

PLOT_COMPLETE_TRACE = True
PLOT_EPOCHS = True


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
        threshold=DEBLINK_THRESHOLD,
        flankers=DEBLINK_FLANKERS,
        remove_zeros=DEBLINK_REMOVE_ZEROS,
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
        lowcut=FILTER_LOWCUT_HZ,
        highcut=FILTER_HIGHCUT_HZ,
    )

    # Extract epochs around stimulus onset.
    # trg1 marks the first frame of the stimulus.
    trialer = TrialEpochs(
        signal=pupil_filtered,
        time=time,
        triggers={"light": df["trg1"].to_numpy()},
    )

    epochs = trialer.extract(
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=EPOCH_BASELINE,
        transform=EPOCH_TRANSFORM,
        n_points=EPOCH_N_POINTS,
    )

    # ------------------------------------------------------------
    # Plot the complete recording
    # ------------------------------------------------------------

    if PLOT_COMPLETE_TRACE:
        plt.figure(figsize=(10, 4))
        plt.plot(time, pupil_raw, alpha=0.3, label="raw")
        plt.plot(time, pupil_filtered, label="clean + filtered")

        # Add one vertical line for each stimulus onset.
        for stim_time in epochs["event_times"]:
            plt.axvline(stim_time, color="black", linestyle="--", alpha=0.4)

        plt.xlabel("Time (s)")
        plt.ylabel("Pupil area")
        plt.title("Complete pupil trace")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Plot individual trials and mean response
    # ------------------------------------------------------------

    if PLOT_EPOCHS:
        plt.figure(figsize=(8, 5))

        # Time zero is stimulus onset.
        plt.axvline(0, color="black", linestyle="--")

        # Horizontal zero is the baseline after z-score transform.
        plt.axhline(0, color="black", linestyle="--")

        # Mark stimulus duration.
        plt.axvspan(0, STIM_SEC, alpha=0.15, label="stimulus")

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

        plt.xlabel("Time from stimulus onset (s)")
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
    # Select the eye region. Select the ROI with the mouse then confirm with 's'.
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

    # White stimulus.
    stim_square = visual.Rect(
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
        show_preview=False,     # set to False for performance
    )

    saved_path = None

    try:
        # Start writing data to disk.
        recorder.start(
            metadata={
                "experiment": "pupillary_light_reflex",
                "n_trials": N_TRIALS,
                "baseline_sec": BASELINE_SEC,
                "prestim_sec": PRESTIM_SEC,
                "stim_sec": STIM_SEC,
                "poststim_sec": POSTSTIM_SEC,
                "deblink_threshold": DEBLINK_THRESHOLD,
                "deblink_flankers": DEBLINK_FLANKERS,
                "deblink_remove_zeros": DEBLINK_REMOVE_ZEROS,
                "filter_lowcut_hz": FILTER_LOWCUT_HZ,
                "filter_highcut_hz": FILTER_HIGHCUT_HZ,
                "epoch_tmin": EPOCH_TMIN,
                "epoch_tmax": EPOCH_TMAX,
                "epoch_baseline": EPOCH_BASELINE,
                "epoch_transform": EPOCH_TRANSFORM,
                "epoch_n_points": EPOCH_N_POINTS,
                "trg1": "stimulus onset pulse",
                "trg2": "trial number",
                "trg3": "stimulus state",
            }
        )

        saved_path = Path(recorder.writer.path)

        # --------------------------------------------------------
        # Initial baseline recording
        # --------------------------------------------------------

        print(f"Baseline recording: {BASELINE_SEC} s")

        baseline_clock = core.Clock()

        while baseline_clock.getTime() < BASELINE_SEC:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            win.color = "black"
            fixation.draw()
            win.flip()

            recorder.save_frame(
                trg1=0,      # no stimulus onset
                trg2=0,      # no trial
                trg3=0,      # stimulus off
            )

        # --------------------------------------------------------
        # Trial sequence
        # --------------------------------------------------------

        trial_duration = PRESTIM_SEC + STIM_SEC + POSTSTIM_SEC

        for trial in range(1, N_TRIALS + 1):
            print(f"Trial {trial}/{N_TRIALS}")

            clock = core.Clock()

            # Used to send trg1=1 only once, at stimulus onset.
            stim_started = False

            while clock.getTime() < trial_duration:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

                t = clock.getTime()

                # ------------------------------------------------
                # Decide what phase we are in
                # ------------------------------------------------

                if t < PRESTIM_SEC:
                    # Pre-stimulus period: black screen with fixation.
                    stim_on = False
                    trg1 = 0
                    trg3 = 0

                elif t < PRESTIM_SEC + STIM_SEC:
                    # Stimulus period: white square is visible.
                    stim_on = True
                    trg3 = 1

                    # Send a trigger pulse only on the first stimulus frame.
                    if not stim_started:
                        trg1 = 1
                        stim_started = True
                    else:
                        trg1 = 0

                else:
                    # Post-stimulus period: black screen with fixation.
                    stim_on = False
                    trg1 = 0
                    trg3 = 0

                # ------------------------------------------------
                # Draw current frame
                # ------------------------------------------------

                win.color = "black"

                if stim_on:
                    stim_square.draw()

                fixation.draw()
                win.flip()

                # ------------------------------------------------
                # Record one synchronized frame/result
                # ------------------------------------------------

                recorder.save_frame(
                    trg1=trg1,        # stimulus onset pulse
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