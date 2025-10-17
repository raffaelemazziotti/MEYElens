from psychopy import core


class CountdownTimer:
    """
    A countdown timer for managing time-limited tasks with precise control.

    Attributes:
        duration (float): Total countdown duration in seconds.
        clock (core.Clock): A PsychoPy clock instance for precise timing.
        is_running (bool): Indicates if the countdown is currently active.
    """
    def __init__(self, duration):
        """
        Initializes the CountdownTimer class.

        Args:
            duration (float): The countdown duration in seconds.
        """
        self.duration = duration
        self.clock = core.Clock()  # Core PsychoPy clock for precise timing
        self.is_running = False

    def start(self):
        """
        Starts the countdown timer.

        Resets the internal clock and marks the timer as active.
        """
        self.clock.reset()
        self.is_running = True

    def get_time_left(self):
        """
        Calculates the remaining time in the countdown.

        Returns:
            float: The remaining time in seconds. If the countdown has ended, returns 0.

        """
        if not self.is_running:
            #raise RuntimeError("CountdownTimer has not been started.")
            return 0

        elapsed = self.clock.getTime()
        time_left = max(0, self.duration - elapsed)
        return time_left

    def is_finished(self):
        """
        Checks whether the countdown has finished.

        Returns:
            bool: True if the countdown has ended, otherwise False.
        """
        return self.get_time_left() <= 0

    def stop(self):
        """
        Stops the countdown timer.

        Marks the timer as inactive, effectively pausing the countdown.
        """
        self.is_running = False