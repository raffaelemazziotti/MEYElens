"""
Visual oddball task with Meye recording.

Instructions
------------
Keep your gaze on the red fixation dot.

You will see brief gratings:

    0 degrees      = standard stimulus
    +45 degrees    = target / oddball
    -45 degrees    = distractor

Press SPACE only for the +45 degree grating.
Do not press for the 0 degree or -45 degree gratings.

The task starts with fixation only, then a few standard stimuli, then the
random oddball sequence. At the end, fixation remains on screen for a few
seconds before the test closes.

Press ESC at any time to stop the experiment.
"""

from pathlib import Path
import random

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
# Settings
# ============================================================

CAMERA_INDEX = 0

INITIAL_FIXATION_SEC = 4.0
FINAL_FIXATION_SEC = 5.0

N_INITIAL_STANDARDS = 4
N_RANDOM_TRIALS = 80

ODDBALL_PROB = 0.15
DISTRACTOR_PROB = 0.15
MIN_STANDARDS_BETWEEN_RARE = 3

STIM_INTERVAL_SEC = 2.0
STIM_DURATION_SEC = 0.25
RESPONSE_WINDOW_SEC = 1.5

OUTPUT_FILENAME = "visual_oddball_meyept_test"

TRG_STANDARD = 1
TRG_ODDBALL = 2
TRG_DISTRACTOR = 3


# ============================================================
# Sequence generation
# ============================================================

def make_random_sequence(
    n_trials,
    oddball_prob,
    distractor_prob,
    min_standards_between_rare,
    seed=None,
):
    """
    Create the random part of the sequence.

    Rare events are oddballs and distractors.
    After each rare event, at least min_standards_between_rare standards
    are forced before another rare event can appear.
    """
    rng = random.Random(seed)

    sequence = []
    standards_since_rare = min_standards_between_rare

    for _ in range(n_trials):
        rare_allowed = standards_since_rare >= min_standards_between_rare

        if not rare_allowed:
            sequence.append("standard")
            standards_since_rare += 1
            continue

        r = rng.random()

        if r < oddball_prob:
            sequence.append("oddball")
            standards_since_rare = 0

        elif r < oddball_prob + distractor_prob:
            sequence.append("distractor")
            standards_since_rare = 0

        else:
            sequence.append("standard")
            standards_since_rare += 1

    return sequence


def make_full_sequence():
    """
    Full stimulation sequence.

    The experiment has:
        1. fixation-only adaptation, not included here;
        2. a few standard-only stimulation trials;
        3. the random oddball/distractor sequence;
        4. final fixation-only period, not included here.
    """
    initial_standards = ["standard"] * N_INITIAL_STANDARDS

    random_sequence = make_random_sequence(
        n_trials=N_RANDOM_TRIALS,
        oddball_prob=ODDBALL_PROB,
        distractor_prob=DISTRACTOR_PROB,
        min_standards_between_rare=MIN_STANDARDS_BETWEEN_RARE,
        seed=42,
    )

    return initial_standards + random_sequence


# ============================================================
# Analysis
# ============================================================

def estimate_sampling_rate(df, time_column="t_frame"):
    dt = df[time_column].diff().median()

    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Could not estimate sampling rate.")

    return float(round(1.0 / dt))


def analyze_oddball(path):
    reader = MeyeReader(path)
    df = reader.data

    time = df["t_frame"].to_numpy()
    pupil_raw = df["pupil_area"].to_numpy()

    # Clean blinks and missing-pupil samples.
    deblink = DeBlink(
        threshold=0.25,
        flankers=5,
        remove_zeros=True,
    )
    pupil_clean = deblink.clean(pupil_raw)

    # Filter the pupil trace.
    fs = estimate_sampling_rate(df)
    print(f"Estimated sampling rate: {fs:.1f} Hz")

    filt = Filters(fs)
    pupil_filtered = filt.bandpass(
        pupil_clean,
        lowcut=0.01,
        highcut=min(4.0, fs * 0.45),
    )

    # Build one binary trigger trace per condition.
    standard_trigger = (df["trg1"].to_numpy() == TRG_STANDARD).astype(int)
    oddball_trigger = (df["trg1"].to_numpy() == TRG_ODDBALL).astype(int)
    distractor_trigger = (df["trg1"].to_numpy() == TRG_DISTRACTOR).astype(int)

    print("Detected events")
    print("  standard:", int(standard_trigger.sum()))
    print("  oddball:", int(oddball_trigger.sum()))
    print("  distractor:", int(distractor_trigger.sum()))

    # Extract epochs.
    # The initial and final fixation periods make the epoch boundaries safe.
    trialer = TrialEpochs(
        signal=pupil_filtered,
        time=time,
        triggers={
            "standard": standard_trigger,
            "oddball": oddball_trigger,
            "distractor": distractor_trigger,
        },
    )

    epochs = trialer.extract(
        tmin=-1.0,
        tmax=4.0,
        baseline=(-1.0, 0.0),
        transform="zscore",
        n_points=300,
    )

    # ------------------------------------------------------------
    # Plot complete trace
    # ------------------------------------------------------------

    plt.figure(figsize=(11, 4))
    plt.plot(time, pupil_raw, alpha=0.25, label="raw")
    plt.plot(time, pupil_filtered, linewidth=2, label="clean + filtered")

    for condition in epochs["conditions"].values():
        for event_time in condition["event_times"]:
            plt.axvline(event_time, color="black", linestyle="--", alpha=0.08)

    plt.xlabel("Time (s)")
    plt.ylabel("Pupil area")
    plt.title("Complete pupil trace")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot evoked response
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.axvline(0, color="black", linestyle="--")
    plt.axhline(0, color="black", linestyle="--")
    plt.axvspan(0, STIM_DURATION_SEC, alpha=0.15, label="stimulus")

    for name, condition in epochs["conditions"].items():
        plt.plot(
            epochs["time"],
            condition["mean"],
            linewidth=2,
            label=f"{name} n={condition['n_trials']}",
        )

        plt.fill_between(
            epochs["time"],
            condition["mean"] - condition["sem"],
            condition["mean"] + condition["sem"],
            alpha=0.2,
        )

    plt.xlabel("Time from stimulus onset (s)")
    plt.ylabel("Pupil response, z-score")
    plt.title("Visual oddball pupil response")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return epochs


# ============================================================
# Main experiment
# ============================================================

sequence = make_full_sequence()

print("Sequence summary")
print("  initial fixation only:", INITIAL_FIXATION_SEC, "s")
print("  initial standards:", N_INITIAL_STANDARDS)
print("  random trials:", N_RANDOM_TRIALS)
print("  final fixation only:", FINAL_FIXATION_SEC, "s")
print("  total stimulation trials:", len(sequence))
print("  standard:", sequence.count("standard"))
print("  oddball:", sequence.count("oddball"))
print("  distractor:", sequence.count("distractor"))


meye = Meye()

with Camera(camera_index=CAMERA_INDEX) as cam:
    cam.select_roi(window_name="Select eye ROI")
    meye.preview(cam)

    win = visual.Window(
        fullscr=True,
        units="height",
        color=0.0,
    )

    fixation = visual.Circle(
        win,
        radius=0.01,
        fillColor="red",
        lineColor="red",
        pos=(0, 0),
    )

    grating = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        size=0.35,
        sf=8,
        contrast=0.8,
        opacity=1.0,
        pos=(0, 0),
    )

    # ------------------------------------------------------------
    # Instruction screen with example stimuli
    # ------------------------------------------------------------

    instruction_title = visual.TextStim(
        win,
        text="Visual oddball task",
        pos=(0, 0.36),
        color="white",
        height=0.045,
    )

    instruction_text = visual.TextStim(
        win,
        text=(
            "Keep your gaze on the red fixation dot.\n\n"
            "Press SPACE only when you see the +45 degree grating.\n"
            "Do not press for the 0 degree standard.\n"
            "Do not press for the -45 degree distractor.\n\n"
            "The task starts with fixation only, then a few standards.\n"
            "At the end, keep fixating until the screen closes.\n\n"
            "Press SPACE to start."
        ),
        pos=(0, -0.35),
        color="white",
        height=0.027,
        wrapWidth=1.3,
    )

    standard_demo = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        size=0.22,
        sf=8,
        contrast=0.8,
        ori=0,
        pos=(-0.45, 0.08),
    )

    oddball_demo = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        size=0.22,
        sf=8,
        contrast=0.8,
        ori=45,
        pos=(0.0, 0.08),
    )

    distractor_demo = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        size=0.22,
        sf=8,
        contrast=0.8,
        ori=-45,
        pos=(0.45, 0.08),
    )

    standard_label = visual.TextStim(
        win,
        text="0°\nStandard\nNo response",
        pos=(-0.45, -0.12),
        color="white",
        height=0.025,
    )

    oddball_label = visual.TextStim(
        win,
        text="+45°\nTarget\nPress SPACE",
        pos=(0.0, -0.12),
        color="lime",
        height=0.025,
    )

    distractor_label = visual.TextStim(
        win,
        text="-45°\nDistractor\nNo response",
        pos=(0.45, -0.12),
        color="white",
        height=0.025,
    )

    instruction_title.draw()
    standard_demo.draw()
    oddball_demo.draw()
    distractor_demo.draw()
    standard_label.draw()
    oddball_label.draw()
    distractor_label.draw()
    instruction_text.draw()
    win.flip()

    event.waitKeys(keyList=["space"])
    event.clearEvents()

    recorder = MeyeRecorder(
        cam=cam,
        meye=meye,
        filename=OUTPUT_FILENAME,
        path_to_file=None,
        show_preview=False,
    )

    saved_path = None
    behavior = []

    try:
        recorder.start(
            metadata={
                "experiment": "visual_oddball",
                "initial_fixation_sec": INITIAL_FIXATION_SEC,
                "final_fixation_sec": FINAL_FIXATION_SEC,
                "n_initial_standards": N_INITIAL_STANDARDS,
                "n_random_trials": N_RANDOM_TRIALS,
                "n_total_stimulation_trials": len(sequence),
                "stim_interval_sec": STIM_INTERVAL_SEC,
                "stim_duration_sec": STIM_DURATION_SEC,
                "response_window_sec": RESPONSE_WINDOW_SEC,
                "standard_orientation": 0,
                "oddball_orientation": 45,
                "distractor_orientation": -45,
                "trg1": "1=standard, 2=oddball, 3=distractor",
                "trg2": "trial number",
                "trg3": "stimulus visible",
                "trg4": "spacebar response",
            }
        )

        saved_path = Path(recorder.writer.path)

        # --------------------------------------------------------
        # Initial fixation-only adaptation
        # --------------------------------------------------------

        print("Initial fixation period")

        fixation_clock = core.Clock()

        while fixation_clock.getTime() < INITIAL_FIXATION_SEC:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            fixation.draw()
            win.flip()

            recorder.save_frame(
                trg1=0,
                trg2=0,
                trg3=0,
                trg4=0,
            )

        event.clearEvents()

        # --------------------------------------------------------
        # Stimulation sequence
        # --------------------------------------------------------

        for trial, trial_type in enumerate(sequence, start=1):
            print(f"Trial {trial}/{len(sequence)}: {trial_type}")

            if trial_type == "standard":
                orientation = 0
                event_code = TRG_STANDARD

            elif trial_type == "oddball":
                orientation = 45
                event_code = TRG_ODDBALL

            elif trial_type == "distractor":
                orientation = -45
                event_code = TRG_DISTRACTOR

            else:
                raise RuntimeError(f"Unknown trial type: {trial_type}")

            trial_clock = core.Clock()
            stimulus_started = False
            responded = False
            response_time = np.nan

            while trial_clock.getTime() < STIM_INTERVAL_SEC:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

                t = trial_clock.getTime()

                stim_on = t < STIM_DURATION_SEC
                trg3 = 1 if stim_on else 0

                if stim_on:
                    grating.ori = orientation
                    grating.draw()

                fixation.draw()
                win.flip()

                keys = event.getKeys(
                    keyList=["space"],
                    timeStamped=trial_clock,
                )

                if keys and not responded and t <= RESPONSE_WINDOW_SEC:
                    responded = True
                    response_time = keys[0][1]

                if stim_on and not stimulus_started:
                    trg1 = event_code
                    stimulus_started = True
                else:
                    trg1 = 0

                trg4 = 1 if keys else 0

                recorder.save_frame(
                    trg1=trg1,
                    trg2=trial,
                    trg3=trg3,
                    trg4=trg4,
                )

            correct = (
                (trial_type == "oddball" and responded)
                or (trial_type != "oddball" and not responded)
            )

            behavior.append(
                {
                    "trial": trial,
                    "trial_type": trial_type,
                    "responded": int(responded),
                    "response_time": response_time,
                    "correct": int(correct),
                }
            )

            event.clearEvents()

        # --------------------------------------------------------
        # Final fixation-only period
        # --------------------------------------------------------

        print("Final fixation period")

        fixation_clock = core.Clock()

        while fixation_clock.getTime() < FINAL_FIXATION_SEC:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            fixation.draw()
            win.flip()

            recorder.save_frame(
                trg1=0,
                trg2=0,
                trg3=0,
                trg4=0,
            )

    except KeyboardInterrupt:
        print("Experiment interrupted.")

    finally:
        recorder.stop()
        win.close()

    # ------------------------------------------------------------
    # Behavior summary
    # ------------------------------------------------------------

    if len(behavior) > 0:
        print("")
        print("Behavior summary")

        for trial_type in ["standard", "oddball", "distractor"]:
            rows = [b for b in behavior if b["trial_type"] == trial_type]

            if len(rows) == 0:
                continue

            response_rate = np.mean([b["responded"] for b in rows]) * 100
            accuracy = np.mean([b["correct"] for b in rows]) * 100

            print(
                f"  {trial_type}: "
                f"n={len(rows)}, "
                f"response={response_rate:.1f}%, "
                f"accuracy={accuracy:.1f}%"
            )

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------

    if saved_path is not None and saved_path.exists():
        print(f"Data saved to: {saved_path}")
        epochs = analyze_oddball(saved_path)

    recorder.close_all()
