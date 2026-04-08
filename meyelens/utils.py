import time


class _PerfCounterClock:
    """Minimal clock wrapper based on ``time.perf_counter``."""

    def __init__(self):
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def getTime(self) -> float:
        return time.perf_counter() - self._start


class CountdownTimer:
    """
    Countdown timer based on a high-precision local clock.

    This utility wraps a ``perf_counter``-based clock to provide a simple
    countdown interface for time-limited tasks (e.g., stimulus display windows,
    response deadlines, trial timeouts).

    Parameters
    ----------
    duration : float
        Total countdown duration in seconds.

    Attributes
    ----------
    duration : float
        Total countdown duration in seconds.
    clock : _PerfCounterClock
        Perf-counter-based clock used for time tracking.
    is_running : bool
        ``True`` when the countdown is active.

    Notes
    -----
    - If the timer is not running, :meth:`get_time_left` returns 0 (matching your
      current behavior and avoiding exceptions).
    - This class is intentionally minimal and does not use logging.
    """

    def __init__(self, duration: float):
        """
        Initialize the countdown timer.

        Parameters
        ----------
        duration : float
            Countdown duration in seconds.
        """
        self.duration = duration
        self.clock = _PerfCounterClock()
        self.is_running = False

    def start(self) -> None:
        """
        Start (or restart) the countdown.

        Resets the internal clock and marks the timer as running.

        Returns
        -------
        None
        """
        self.clock.reset()
        self.is_running = True

    def get_time_left(self) -> float:
        """
        Get the remaining time in the countdown.

        Returns
        -------
        float
            Remaining time in seconds. If the timer is not running or the
            countdown has completed, returns 0.
        """
        if not self.is_running:
            # Kept intentionally non-raising for drop-in simplicity.
            return 0.0

        elapsed = self.clock.getTime()
        time_left = max(0.0, self.duration - elapsed)
        return time_left

    def is_finished(self) -> bool:
        """
        Check whether the countdown has completed.

        Returns
        -------
        bool
            ``True`` if remaining time is 0, otherwise ``False``.
        """
        return self.get_time_left() <= 0.0

    def stop(self) -> None:
        """
        Stop (pause) the countdown.

        This does not reset elapsed time; it simply marks the timer as inactive.
        With the current design, :meth:`get_time_left` will return 0 while stopped.

        Returns
        -------
        None
        """
        self.is_running = False
