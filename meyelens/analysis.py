import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt
import numpy as np


class AdaptiveGazeFilter:
    """
    Adaptive gaze filter for live gaze pointer smoothing.

    It uses a One Euro filter:
    - strong smoothing when gaze is stable
    - weaker smoothing when gaze moves quickly

    Parameters
    ----------
    min_cutoff:
        Base smoothing. Lower = smoother during fixation, but more lag.
        Try 0.6 to 1.2.

    beta:
        Motion adaptation. Higher = faster response during eye movements.
        Try 0.02 to 0.15.

    d_cutoff:
        Smoothing applied to velocity estimate.
        Usually 1.0 is fine.

    jump_reset:
        If prediction jumps more than this distance, optionally reset the filter.
        Useful for very large gaze shifts or recovery from bad samples.
        In PsychoPy norm units, try 0.5 or None.
    """

    def __init__(
        self,
        min_cutoff=0.8,
        beta=0.08,
        d_cutoff=1.0,
        jump_reset=None,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.jump_reset = jump_reset

        self.last_time = None

        self.x = None
        self.y = None

        self.dx = 0.0
        self.dy = 0.0

    def reset(self):
        self.last_time = None
        self.x = None
        self.y = None
        self.dx = 0.0
        self.dy = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _lowpass(new_value, previous_value, alpha):
        return alpha * new_value + (1.0 - alpha) * previous_value

    def update(self, x, y, timestamp=None):
        x = float(x)
        y = float(y)

        if timestamp is None:
            timestamp = time.perf_counter()

        if self.x is None or self.y is None or self.last_time is None:
            self.x = x
            self.y = y
            self.last_time = timestamp
            return self.x, self.y

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 0:
            return self.x, self.y

        # Optional hard reset for very large jumps.
        if self.jump_reset is not None:
            jump = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

            if jump > self.jump_reset:
                self.x = x
                self.y = y
                self.dx = 0.0
                self.dy = 0.0
                return self.x, self.y

        # Raw velocity.
        raw_dx = (x - self.x) / dt
        raw_dy = (y - self.y) / dt

        # Smooth velocity estimate.
        alpha_d = self._alpha(self.d_cutoff, dt)

        self.dx = self._lowpass(raw_dx, self.dx, alpha_d)
        self.dy = self._lowpass(raw_dy, self.dy, alpha_d)

        speed = math.sqrt(self.dx ** 2 + self.dy ** 2)

        # Increase cutoff when speed increases.
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)

        self.x = self._lowpass(x, self.x, alpha)
        self.y = self._lowpass(y, self.y, alpha)

        return self.x, self.y


class DeBlink:
    """
    Blink cleaner for pupil traces.

    Main logic
    ----------
    A sample is marked as blink/artifact if:

        abs(derivative) / max(abs(derivative)) > threshold

    or, if remove_zeros=True:

        signal == zero_value

    Then flankers are added around the detected samples. The detected samples are
    replaced with NaN and interpolated.

    Main use
    --------
    deblink = DeBlink(threshold=0.25, flankers=5)

    pupil_clean = deblink.clean(pupil_area)
    blink_mask = deblink.get_blink_mask(pupil_area)
    diagnostics = deblink.get_diagnostics(pupil_area)
    """

    def __init__(
        self,
        threshold=0.25,
        flankers=3,
        interpolation="linear",
        interpolation_order=2,
        fill_edges=True,
        remove_zeros=True,
        zero_value=0.0,
    ):
        self.threshold = float(threshold)
        self.flankers = int(flankers)
        self.interpolation = str(interpolation)
        self.interpolation_order = int(interpolation_order)
        self.fill_edges = bool(fill_edges)
        self.remove_zeros = bool(remove_zeros)
        self.zero_value = float(zero_value)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        if self.flankers < 0:
            raise ValueError("flankers must be >= 0.")

    # ============================================================
    # Main API
    # ============================================================

    def clean(self, signal):
        """
        Return only the cleaned pupil signal.
        """
        signal = self._as_1d_float(signal)
        blink_mask = self.get_blink_mask(signal)

        clean = signal.copy()
        clean[blink_mask] = np.nan

        return self.interpolate(clean)

    def get_blink_mask(self, signal):
        """
        Return the final blink/artifact mask.

        True means that the sample is marked as blink/artifact and will be
        interpolated.
        """
        signal = self._as_1d_float(signal)

        derivative_mask = self.get_derivative_mask(signal)
        zero_mask = self.get_zero_mask(signal)

        blink_mask = derivative_mask | zero_mask
        blink_mask = self.add_flankers(blink_mask, self.flankers)

        return blink_mask

    def get_derivative_mask(self, signal):
        """
        Return the blink/artifact mask based only on the derivative.
        """
        signal = self._as_1d_float(signal)

        derivative = self.compute_derivative(signal)
        abs_derivative = np.abs(derivative)
        norm_abs_derivative = self.normalize_by_max(abs_derivative)

        return norm_abs_derivative > self.threshold

    def get_zero_mask(self, signal):
        """
        Return the mask for zero-valued samples.

        If remove_zeros=False, this returns an all-False mask.
        """
        signal = self._as_1d_float(signal)

        if not self.remove_zeros:
            return np.zeros_like(signal, dtype=bool)

        return np.isfinite(signal) & (signal == self.zero_value)

    def get_diagnostics(self, signal):
        """
        Return intermediate arrays useful for plotting/debugging.
        """
        signal = self._as_1d_float(signal)

        derivative = self.compute_derivative(signal)
        abs_derivative = np.abs(derivative)
        norm_abs_derivative = self.normalize_by_max(abs_derivative)

        derivative_mask = norm_abs_derivative > self.threshold
        zero_mask = self.get_zero_mask(signal)

        blink_mask_raw = derivative_mask | zero_mask
        blink_mask = self.add_flankers(blink_mask_raw, self.flankers)

        clean = signal.copy()
        clean[blink_mask] = np.nan
        clean = self.interpolate(clean)

        return {
            "clean": clean,
            "blink_mask": blink_mask,
            "blink_mask_raw": blink_mask_raw,
            "derivative_mask": derivative_mask,
            "zero_mask": zero_mask,
            "derivative": derivative,
            "abs_derivative": abs_derivative,
            "norm_abs_derivative": norm_abs_derivative,
        }

    # ============================================================
    # DataFrame helper
    # ============================================================

    def clean_dataframe(
        self,
        df,
        column="pupil_area",
        output_column=None,
        blink_column=None,
        zero_column=None,
        derivative_column=None,
    ):
        """
        Return a copy of df with a cleaned pupil column.

        Parameters
        ----------
        df
            Input pandas DataFrame.

        column
            Input signal column.

        output_column
            Output cleaned signal column. If None, uses f"{column}_clean".

        blink_column
            Optional output column for the final blink mask.

        zero_column
            Optional output column for zero-valued samples.

        derivative_column
            Optional output column for abs(derivative) / max(abs(derivative)).
        """
        if output_column is None:
            output_column = f"{column}_clean"

        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")

        diagnostics = self.get_diagnostics(df[column].to_numpy())

        out = df.copy()
        out[output_column] = diagnostics["clean"]

        if blink_column is not None:
            out[blink_column] = diagnostics["blink_mask"].astype(int)

        if zero_column is not None:
            out[zero_column] = diagnostics["zero_mask"].astype(int)

        if derivative_column is not None:
            out[derivative_column] = diagnostics["norm_abs_derivative"]

        return out

    # ============================================================
    # Core operations
    # ============================================================

    @staticmethod
    def compute_derivative(signal):
        """
        Compute derivative while preserving signal length.

        np.gradient is used instead of np.diff so the derivative and mask have
        the same length as the original signal.
        """
        signal = DeBlink._as_1d_float(signal)

        if len(signal) < 2:
            return np.zeros_like(signal, dtype=float)

        return np.gradient(signal)

    @staticmethod
    def normalize_by_max(x):
        """
        Normalize a non-negative signal by its maximum.

        Intended for abs(derivative):

            0 = no change
            1 = largest detected change
        """
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)

        valid = np.isfinite(x)

        if not np.any(valid):
            return out

        xmax = np.nanmax(x[valid])

        if xmax <= 0:
            return out

        out[valid] = x[valid] / xmax
        out[~valid] = 0.0

        return out

    @staticmethod
    def add_flankers(mask, flankers):
        """
        Expand a boolean mask by N samples on each side.
        """
        mask = np.asarray(mask, dtype=bool)

        if flankers <= 0:
            return mask.copy()

        expanded = mask.copy()
        blink_indices = np.where(mask)[0]
        n = len(mask)

        for idx in blink_indices:
            start = max(0, idx - flankers)
            stop = min(n, idx + flankers + 1)
            expanded[start:stop] = True

        return expanded

    def interpolate(self, signal_with_nan):
        """
        Interpolate NaN samples.
        """
        signal_with_nan = self._as_1d_float(signal_with_nan)
        s = pd.Series(signal_with_nan, dtype=float)

        kwargs = {
            "method": self.interpolation,
            "limit_direction": "both" if self.fill_edges else "forward",
        }

        if self.interpolation in {"spline", "polynomial"}:
            kwargs["order"] = self.interpolation_order

        try:
            clean = s.interpolate(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Interpolation failed with method={self.interpolation!r}. "
                "Try interpolation='linear'."
            ) from exc

        if self.fill_edges:
            clean = clean.bfill().ffill()

        return clean.to_numpy(dtype=float)

    # ============================================================
    # Plotting
    # ============================================================

    def plot_diagnostics(self, signal, time=None, show_clean=True):
        """
        Plot the pupil trace, cleaned trace, derivative, threshold, and mask.

        Parameters
        ----------
        signal
            One-dimensional pupil trace.

        time
            Optional time vector. If None, sample index is used.

        show_clean
            If True, plot the cleaned interpolated trace.
        """
        signal = self._as_1d_float(signal)

        if time is None:
            time = np.arange(len(signal))
            xlabel = "Sample"
        else:
            time = np.asarray(time, dtype=float)
            if len(time) != len(signal):
                raise ValueError("time must have the same length as signal.")
            xlabel = "Time"

        diagnostics = self.get_diagnostics(signal)

        clean = diagnostics["clean"]
        norm_abs_derivative = diagnostics["norm_abs_derivative"]
        blink_mask = diagnostics["blink_mask"]
        derivative_mask = diagnostics["derivative_mask"]
        zero_mask = diagnostics["zero_mask"]

        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 6),
        )

        axes[0].plot(time, signal, label="raw")

        if show_clean:
            axes[0].plot(time, clean, label="clean")

        if np.any(blink_mask):
            axes[0].scatter(
                time[blink_mask],
                signal[blink_mask],
                s=18,
                label="blink/artifact",
            )

        if np.any(zero_mask):
            axes[0].scatter(
                time[zero_mask],
                signal[zero_mask],
                s=35,
                marker="x",
                label="zero samples",
            )

        axes[0].set_ylabel("Pupil signal")
        axes[0].set_title("DeBlink pupil trace")
        axes[0].legend(loc="best")

        axes[1].plot(
            time,
            norm_abs_derivative,
            label="abs(derivative) / max",
        )

        axes[1].axhline(
            self.threshold,
            linestyle="--",
            label=f"threshold = {self.threshold:.2f}",
        )

        if np.any(derivative_mask):
            axes[1].scatter(
                time[derivative_mask],
                norm_abs_derivative[derivative_mask],
                s=18,
                label="derivative detections",
            )

        if np.any(blink_mask):
            axes[1].fill_between(
                time,
                0,
                1,
                where=blink_mask,
                alpha=0.2,
                label="final mask + flankers",
            )

        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("Normalized abs derivative")
        axes[1].set_title("Derivative and zero-value detection")
        axes[1].legend(loc="best")

        fig.tight_layout()
        return fig

    # ============================================================
    # Setters
    # ============================================================

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

    def set_flankers(self, flankers):
        self.flankers = int(flankers)

        if self.flankers < 0:
            raise ValueError("flankers must be >= 0.")

    def set_interpolation(self, interpolation, order=None):
        self.interpolation = str(interpolation)

        if order is not None:
            self.interpolation_order = int(order)

    def set_remove_zeros(self, remove_zeros=True, zero_value=0.0):
        self.remove_zeros = bool(remove_zeros)
        self.zero_value = float(zero_value)

    # ============================================================
    # Validation
    # ============================================================

    @staticmethod
    def _as_1d_float(signal):
        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional.")

        return signal


class Filters:
    """
    Frequency filter for pupil traces.

    Supports:
        - lowpass
        - highpass
        - bandpass

    Main use
    --------
    filt = PupilFilter(fs=30)

    pupil_low = filt.lowpass(pupil, cutoff=4.0)
    pupil_high = filt.highpass(pupil, cutoff=0.05)
    pupil_band = filt.bandpass(pupil, lowcut=0.05, highcut=4.0)

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    order : int
        Butterworth filter order.

    method : str
        "sos" or "ba".

        "sos" is recommended because it is more numerically stable.
    """

    def __init__(
        self,
        fs,
        order=4,
        method="sos",
    ):
        self.fs = float(fs)
        self.order = int(order)
        self.method = str(method)

        if self.fs <= 0:
            raise ValueError("fs must be > 0.")

        if self.order <= 0:
            raise ValueError("order must be > 0.")

        if self.method not in {"sos", "ba"}:
            raise ValueError("method must be 'sos' or 'ba'.")

    # ============================================================
    # Public filters
    # ============================================================

    def lowpass(self, signal, cutoff):
        """
        Lowpass filter.

        Keeps frequencies below cutoff.
        """
        signal = self._as_1d_float(signal)
        cutoff = float(cutoff)

        self._check_cutoff(cutoff)

        return self._filter(
            signal=signal,
            cutoff=cutoff,
            btype="lowpass",
        )

    def highpass(self, signal, cutoff):
        """
        Highpass filter.

        Keeps frequencies above cutoff.
        """
        signal = self._as_1d_float(signal)
        cutoff = float(cutoff)

        self._check_cutoff(cutoff)

        return self._filter(
            signal=signal,
            cutoff=cutoff,
            btype="highpass",
        )

    def bandpass(self, signal, lowcut, highcut):
        """
        Bandpass filter.

        Keeps frequencies between lowcut and highcut.
        """
        signal = self._as_1d_float(signal)
        lowcut = float(lowcut)
        highcut = float(highcut)

        self._check_band(lowcut, highcut)

        return self._filter(
            signal=signal,
            cutoff=[lowcut, highcut],
            btype="bandpass",
        )

    # ============================================================
    # DataFrame helpers
    # ============================================================

    def lowpass_dataframe(
        self,
        df,
        column="pupil_area",
        output_column=None,
        cutoff=4.0,
    ):
        if output_column is None:
            output_column = f"{column}_lowpass"

        out = df.copy()
        out[output_column] = self.lowpass(out[column].to_numpy(), cutoff=cutoff)

        return out

    def highpass_dataframe(
        self,
        df,
        column="pupil_area",
        output_column=None,
        cutoff=0.05,
    ):
        if output_column is None:
            output_column = f"{column}_highpass"

        out = df.copy()
        out[output_column] = self.highpass(out[column].to_numpy(), cutoff=cutoff)

        return out

    def bandpass_dataframe(
        self,
        df,
        column="pupil_area",
        output_column=None,
        lowcut=0.05,
        highcut=4.0,
    ):
        if output_column is None:
            output_column = f"{column}_bandpass"

        out = df.copy()
        out[output_column] = self.bandpass(
            out[column].to_numpy(),
            lowcut=lowcut,
            highcut=highcut,
        )

        return out

    # ============================================================
    # Internal filtering
    # ============================================================

    def _filter(self, signal, cutoff, btype):
        """
        Apply zero-phase Butterworth filtering.

        Uses filtfilt/sosfiltfilt, so there is no phase shift.
        """

        signal = self._fill_nan(signal)

        nyquist = self.fs / 2.0

        if self.method == "sos":
            sos = butter(
                N=self.order,
                Wn=np.asarray(cutoff, dtype=float) / nyquist,
                btype=btype,
                output="sos",
            )
            return sosfiltfilt(sos, signal)

        b, a = butter(
            N=self.order,
            Wn=np.asarray(cutoff, dtype=float) / nyquist,
            btype=btype,
            output="ba",
        )
        return filtfilt(b, a, signal)

    def _fill_nan(self, signal):
        """
        Fill NaNs before filtering.

        Frequency filters cannot handle NaNs.
        This uses linear interpolation and edge filling.
        """
        signal = signal.astype(float).copy()

        if not np.any(~np.isfinite(signal)):
            return signal

        x = np.arange(len(signal))
        valid = np.isfinite(signal)

        if valid.sum() == 0:
            return np.zeros_like(signal, dtype=float)

        if valid.sum() == 1:
            return np.full_like(signal, signal[valid][0], dtype=float)

        signal[~valid] = np.interp(
            x[~valid],
            x[valid],
            signal[valid],
        )

        return signal

    # ============================================================
    # Validation
    # ============================================================

    def _check_cutoff(self, cutoff):
        nyquist = self.fs / 2.0

        if cutoff <= 0:
            raise ValueError("cutoff must be > 0.")

        if cutoff >= nyquist:
            raise ValueError(
                f"cutoff must be lower than Nyquist frequency ({nyquist:.3f} Hz)."
            )

    def _check_band(self, lowcut, highcut):
        nyquist = self.fs / 2.0

        if lowcut <= 0:
            raise ValueError("lowcut must be > 0.")

        if highcut <= 0:
            raise ValueError("highcut must be > 0.")

        if lowcut >= highcut:
            raise ValueError("lowcut must be lower than highcut.")

        if highcut >= nyquist:
            raise ValueError(
                f"highcut must be lower than Nyquist frequency ({nyquist:.3f} Hz)."
            )

    @staticmethod
    def _as_1d_float(signal):
        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional.")

        return signal


class MeyeReader:
    """
    Minimal reader for files created with MeyeRecorder.

    The path must be provided explicitly.

    Expected file format
    --------------------
    # metadata_key: metadata_value
    # metadata_key: metadata_value
    frame_index;t_call;t_frame;...;trg1;trg2;...
    0;0.001;0.010;...

    Basic usage
    -----------
    reader = MeyeReader("path/to/recording.txt")

    df = reader.data
    metadata = reader.metadata

    reader.print_summary()
    """

    def __init__(
        self,
        path,
        sep=";",
        comment="#",
    ):
        self.path = Path(path).expanduser()
        self.sep = sep
        self.comment = comment

        if not self.path.exists():
            raise FileNotFoundError(f"Recording file not found: {self.path}")

        self.metadata = self._read_metadata()
        self.data = self._read_data()

    # ============================================================
    # Loading
    # ============================================================

    def _read_metadata(self):
        metadata = {}

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                if not line.startswith(self.comment):
                    break

                line = line[len(self.comment):].strip()

                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                metadata[key.strip()] = self._parse_metadata_value(value.strip())

        return metadata

    def _read_data(self):
        return pd.read_csv(
            self.path,
            sep=self.sep,
            comment=self.comment,
        )

    @staticmethod
    def _parse_metadata_value(value):
        if value in ("True", "true"):
            return True

        if value in ("False", "false"):
            return False

        if value in ("None", "none", "null"):
            return None

        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    # ============================================================
    # Basic access
    # ============================================================

    def copy(self):
        """
        Return a copy of the recording table.
        """
        return self.data.copy()

    def columns(self):
        """
        Return the column names.
        """
        return list(self.data.columns)

    def get_metadata(self):
        """
        Return a copy of the metadata dictionary.
        """
        return dict(self.metadata)

    def print_summary(self):
        """
        Print file, metadata, and column information.
        """
        print("")
        print("### MEYE READER SUMMARY ###")
        print(f"File: {self.path}")
        print(f"Rows: {len(self.data)}")
        print(f"Columns: {len(self.data.columns)}")

        print("")
        print("Metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        print("")
        print("Columns:")
        for col in self.data.columns:
            print(f"  {col}")

        print("")


class TrialEpochs:
    """
    Extract stimulus-locked epochs from one signal and one or more trigger traces.

    This class does not perform trial rejection or signal cleaning.
    Clean/filter the signal before passing it here.

    Example
    -------
    .. code-block:: python

       trialer = TrialEpochs(
           signal=pupil_signal,
           time=time,
           triggers={
               "standard": trg_standard,
               "oddball": trg_oddball,
               "distractor": trg_distractor,
           },
       )

       epochs = trialer.extract(
           tmin=-1.0,
           tmax=4.0,
           baseline=(-1.0, 0.0),
           transform="zscore",
       )

    Transform options
    -----------------
    - ``"none"``: raw epoch.
    - ``"delta"``: epoch minus baseline mean.
    - ``"percent"``: percent change from baseline mean.
    - ``"zscore"``: baseline-normalized z-score.
    """

    def __init__(
        self,
        signal,
        time,
        triggers,
    ):
        self.signal = self._as_1d_float(signal, name="signal")
        self.time = self._as_1d_float(time, name="time")

        if len(self.signal) != len(self.time):
            raise ValueError("signal and time must have the same length.")

        if len(self.signal) < 2:
            raise ValueError("signal and time must contain at least two samples.")

        if not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly increasing.")

        self.triggers = self._prepare_triggers(triggers)

        for name, trg in self.triggers.items():
            if len(trg) != len(self.signal):
                raise ValueError(
                    f"Trigger {name!r} has length {len(trg)}, "
                    f"but signal has length {len(self.signal)}."
                )

    # ============================================================
    # Public API
    # ============================================================

    def extract(
        self,
        tmin=-1.0,
        tmax=4.0,
        baseline=None,
        transform="none",
        trigger_value=1,
        edge="rising",
        n_points=300,
        exclude_first=0,
        exclude_last=0,
    ):
        """
        Extract epochs for all trigger traces.

        Parameters
        ----------
        tmin, tmax
            Epoch window in seconds relative to trigger onset.

        baseline
            None or tuple, for example (-1.0, 0.0).
            Baseline window is relative to trigger onset.

        transform
            "none", "delta", "percent", or "zscore".

        trigger_value
            Trigger value marking stimulus onset.

        edge
            ``"rising"`` detects transitions into the trigger value.
            ``"level"`` detects every sample equal to the trigger value.

        n_points
            Number of points in the interpolated epoch time base.

        exclude_first
            Number of first detected events to exclude for each condition.

        exclude_last
            Number of last detected events to exclude for each condition.

        Returns
        -------
        dict
            Contains the common epoch time base and a ``conditions`` mapping.
            Each condition stores epochs, event indices, event times, mean, SEM,
            and trial count. With one trigger, those fields are also exposed at
            the top level.
        """

        tmin = float(tmin)
        tmax = float(tmax)
        n_points = int(n_points)
        exclude_first = int(exclude_first)
        exclude_last = int(exclude_last)

        if tmax <= tmin:
            raise ValueError("tmax must be greater than tmin.")

        if n_points < 2:
            raise ValueError("n_points must be >= 2.")

        if exclude_first < 0:
            raise ValueError("exclude_first must be >= 0.")

        if exclude_last < 0:
            raise ValueError("exclude_last must be >= 0.")

        self._check_baseline(tmin, tmax, baseline, transform)

        epoch_time = np.linspace(tmin, tmax, n_points)

        output = {
            "time": epoch_time,
            "conditions": {},
            "settings": {
                "tmin": tmin,
                "tmax": tmax,
                "baseline": baseline,
                "transform": transform,
                "trigger_value": trigger_value,
                "edge": edge,
                "n_points": n_points,
                "exclude_first": exclude_first,
                "exclude_last": exclude_last,
            },
        }

        for condition_name, trigger in self.triggers.items():
            event_indices = self.find_events(
                trigger=trigger,
                value=trigger_value,
                edge=edge,
            )

            event_indices = self._exclude_events(
                event_indices,
                exclude_first=exclude_first,
                exclude_last=exclude_last,
                condition_name=condition_name,
            )

            event_times = self.time[event_indices]

            condition = self._extract_condition(
                condition_name=condition_name,
                event_indices=event_indices,
                event_times=event_times,
                epoch_time=epoch_time,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                transform=transform,
            )

            output["conditions"][condition_name] = condition

        if len(self.triggers) == 1:
            only_name = list(self.triggers.keys())[0]
            only = output["conditions"][only_name]

            output["condition_name"] = only_name
            output["epochs"] = only["epochs"]
            output["event_indices"] = only["event_indices"]
            output["event_times"] = only["event_times"]
            output["mean"] = only["mean"]
            output["sem"] = only["sem"]
            output["n_trials"] = only["n_trials"]

        return output

    def get_event_times(
        self,
        trigger_name=None,
        trigger_value=1,
        edge="rising",
    ):
        """
        Return event times for one trigger.
        """
        trigger_name = self._resolve_trigger_name(trigger_name)
        trigger = self.triggers[trigger_name]

        indices = self.find_events(
            trigger=trigger,
            value=trigger_value,
            edge=edge,
        )

        return self.time[indices]

    def get_event_indices(
        self,
        trigger_name=None,
        trigger_value=1,
        edge="rising",
    ):
        """
        Return event indices for one trigger.
        """
        trigger_name = self._resolve_trigger_name(trigger_name)
        trigger = self.triggers[trigger_name]

        return self.find_events(
            trigger=trigger,
            value=trigger_value,
            edge=edge,
        )

    # ============================================================
    # Core extraction
    # ============================================================

    def _extract_condition(
        self,
        condition_name,
        event_indices,
        event_times,
        epoch_time,
        tmin,
        tmax,
        baseline,
        transform,
    ):
        if len(event_indices) == 0:
            raise RuntimeError(f"No events found for condition {condition_name!r}.")

        epochs = []

        for event_index, event_time in zip(event_indices, event_times):
            start_time = event_time + tmin
            stop_time = event_time + tmax

            if start_time < self.time[0]:
                raise RuntimeError(
                    f"Epoch for condition {condition_name!r} at event index "
                    f"{event_index} starts before recording begins. "
                    f"event_time={event_time:.6f}, start_time={start_time:.6f}, "
                    f"recording_start={self.time[0]:.6f}. "
                    "Use a shorter pre-stimulus window or exclude_first."
                )

            if stop_time > self.time[-1]:
                raise RuntimeError(
                    f"Epoch for condition {condition_name!r} at event index "
                    f"{event_index} ends after recording ends. "
                    f"event_time={event_time:.6f}, stop_time={stop_time:.6f}, "
                    f"recording_end={self.time[-1]:.6f}. "
                    "Use a shorter post-stimulus window or exclude_last."
                )

            rel_time = self.time - event_time

            window_mask = (
                (rel_time >= tmin)
                & (rel_time <= tmax)
            )

            if window_mask.sum() < 2:
                raise RuntimeError(
                    f"Not enough samples inside epoch window for condition "
                    f"{condition_name!r} at event index {event_index}."
                )

            rel = rel_time[window_mask]
            sig = self.signal[window_mask]

            if np.any(~np.isfinite(rel)):
                raise RuntimeError(
                    f"Non-finite time values inside epoch for condition "
                    f"{condition_name!r} at event index {event_index}."
                )

            if np.any(~np.isfinite(sig)):
                raise RuntimeError(
                    f"Non-finite signal values inside epoch for condition "
                    f"{condition_name!r} at event index {event_index}. "
                    "Clean or interpolate the signal before epoch extraction."
                )

            epoch = np.interp(
                epoch_time,
                rel,
                sig,
            )

            epoch = self.apply_baseline_transform(
                epoch=epoch,
                epoch_time=epoch_time,
                baseline=baseline,
                transform=transform,
            )

            epochs.append(epoch)

        epochs = np.vstack(epochs)
        mean = np.mean(epochs, axis=0)
        sem = np.std(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0]) if epochs.shape[0] > 1 else np.zeros(epochs.shape[1])

        return {
            "epochs": epochs,
            "event_indices": np.asarray(event_indices, dtype=int),
            "event_times": np.asarray(event_times, dtype=float),
            "mean": mean,
            "sem": sem,
            "n_trials": int(epochs.shape[0]),
        }

    @staticmethod
    def apply_baseline_transform(
        epoch,
        epoch_time,
        baseline=None,
        transform="none",
    ):
        """
        Apply baseline correction or transformation to one epoch.
        """
        epoch = np.asarray(epoch, dtype=float).copy()
        epoch_time = np.asarray(epoch_time, dtype=float)

        if transform is None:
            transform = "none"

        transform = str(transform).lower()

        if transform == "none":
            return epoch

        if baseline is None:
            raise ValueError("baseline must be provided when transform is not 'none'.")

        b0, b1 = baseline
        baseline_mask = (epoch_time >= b0) & (epoch_time <= b1)

        if not np.any(baseline_mask):
            raise ValueError("baseline window does not overlap epoch time.")

        baseline_values = epoch[baseline_mask]

        if np.any(~np.isfinite(baseline_values)):
            raise RuntimeError("Baseline contains non-finite values.")

        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values, ddof=1)

        if transform == "delta":
            return epoch - baseline_mean

        if transform == "percent":
            if baseline_mean == 0:
                raise RuntimeError("Cannot compute percent change because baseline mean is zero.")
            return ((epoch - baseline_mean) / baseline_mean) * 100.0

        if transform == "zscore":
            if baseline_std == 0:
                raise RuntimeError("Cannot compute z-score because baseline std is zero.")
            return (epoch - baseline_mean) / baseline_std

        raise ValueError("transform must be 'none', 'delta', 'percent', or 'zscore'.")

    # ============================================================
    # Event detection
    # ============================================================

    @staticmethod
    def find_events(
        trigger,
        value=1,
        edge="rising",
    ):
        """
        Find trigger events.

        edge="rising":
            detect transition from not-value to value.

        edge="level":
            detect every sample equal to value.
        """
        trigger = np.asarray(trigger)
        edge = str(edge).lower()

        active = trigger == value

        if edge == "level":
            return np.where(active)[0]

        if edge == "rising":
            active_int = active.astype(int)
            diff = np.diff(active_int, prepend=0)
            return np.where(diff == 1)[0]

        raise ValueError("edge must be 'rising' or 'level'.")

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _exclude_events(
        event_indices,
        exclude_first=0,
        exclude_last=0,
        condition_name="condition",
    ):
        event_indices = np.asarray(event_indices, dtype=int)

        if exclude_first > 0:
            event_indices = event_indices[exclude_first:]

        if exclude_last > 0:
            event_indices = event_indices[:-exclude_last]

        if len(event_indices) == 0:
            raise RuntimeError(
                f"No events left for condition {condition_name!r} "
                "after exclude_first/exclude_last."
            )

        return event_indices

    @staticmethod
    def _check_baseline(tmin, tmax, baseline, transform):
        if transform is None:
            transform = "none"

        transform = str(transform).lower()

        if transform == "none":
            return

        if baseline is None:
            raise ValueError("baseline must be provided when transform is not 'none'.")

        b0, b1 = baseline

        if b1 <= b0:
            raise ValueError("baseline end must be greater than baseline start.")

        if b0 < tmin or b1 > tmax:
            raise ValueError(
                "baseline must be inside the epoch window. "
                f"Got baseline={baseline}, epoch=({tmin}, {tmax})."
            )

    def _resolve_trigger_name(self, trigger_name):
        if trigger_name is not None:
            if trigger_name not in self.triggers:
                raise ValueError(f"Unknown trigger name: {trigger_name}")
            return trigger_name

        if len(self.triggers) != 1:
            raise ValueError(
                "trigger_name must be provided when multiple triggers exist."
            )

        return list(self.triggers.keys())[0]

    @staticmethod
    def _prepare_triggers(triggers):
        """
        Accept:
            array-like               -> {"trigger": array}
            dict of array-like        -> user-defined conditions
            list/tuple of array-like  -> {"trg1": ..., "trg2": ...}
        """
        if isinstance(triggers, dict):
            return {
                str(name): np.asarray(values)
                for name, values in triggers.items()
            }

        if isinstance(triggers, (list, tuple)):
            return {
                f"trg{i + 1}": np.asarray(values)
                for i, values in enumerate(triggers)
            }

        return {
            "trigger": np.asarray(triggers)
        }

    @staticmethod
    def _as_1d_float(x, name="array"):
        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

        return x
