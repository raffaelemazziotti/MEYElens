import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue


class BufferedFileWriter:
    """
    Buffered asynchronous text-file writer.

    This class is designed for time-sensitive acquisition loops.

    The acquisition loop calls:

        writer.write_sv(row)

    and the row is placed into an in-memory queue. A background thread removes
    rows from the queue and writes them to disk.

    This avoids blocking the acquisition loop on disk I/O.

    Important distinction
    ---------------------
    There are two different buffers:

    - Queue buffer: rows waiting in memory before the background thread writes
      them.
    - File buffer: rows already written by Python to the file object, but not
      necessarily flushed to the operating system yet.

    The parameters ``flush_every`` and ``flush_interval`` refer to the
    file buffer, not to the queue.

    Normal close behavior
    ---------------------
    Calling :meth:`close` guarantees that:

    - no new rows are accepted;
    - the queue is drained;
    - all queued rows are written;
    - the file buffer is flushed;
    - the file is closed.

    Therefore, under normal termination, queued data are not lost.

    Overflow behavior
    -----------------
    If the queue becomes full, the behavior is controlled by
    ``overflow_policy``:

    - ``"raise"``:
        Raise an error. Recommended for experiments, because a full queue means
        the writer cannot keep up.

    - ``"block"``:
        Wait until there is space in the queue. This prevents data loss but can
        block the acquisition loop.

    - ``"drop"``:
        Discard the new row and print a warning. This avoids blocking, but data
        are lost.

    File format
    -----------
    The output file contains:

    - optional metadata lines, prefixed by ``#``;
    - one header row;
    - one data row per call to :meth:`write` or :meth:`write_sv`.

    Example
    -------
    .. code-block:: python

       writer = BufferedFileWriter(
           path_to_file="Data",
           filename="subject01",
           headers=["time", "x", "y", "trg1"],
           metadata={"subject": "subject01", "condition": "baseline"},
           overflow_policy="raise",
       )

       writer.write_sv([0.001, 123.4, 88.2, 1])
       writer.close()
    """

    VALID_OVERFLOW_POLICIES = {"raise", "block", "drop"}

    def __init__(
        self,
        path_to_file,
        filename: str = "data",
        buffer_size: int = 1000,
        metadata=None,
        headers=None,
        sep: str = ";",
        extension: str = ".txt",
        flush_every: int = 50,
        flush_interval: float = 1.0,
        overflow_policy: str = "raise",
        encoding: str = "utf-8",
    ):
        """
        Initialize the writer.

        Parameters
        ----------
        path_to_file : str or pathlib.Path
            Directory where the output file will be created.

        filename : str, optional
            Base filename. A timestamp is prepended and ``extension`` is appended.

            Example:
                filename="subject01"

            produces something like:
                20260519_154201-subject01.txt

        buffer_size : int, optional
            Maximum number of rows allowed in the queue.

            If the acquisition loop produces rows faster than the background
            thread can write them, the queue grows. Once it reaches
            ``buffer_size``, the behavior depends on ``overflow_policy``.

        metadata : dict or None, optional
            Metadata written at the top of the file as comment lines:

                # subject: S01
                # condition: baseline

            Metadata are written once at file creation.

        headers : list[str] or None, optional
            Column names. Written as the first non-comment row.

            If ``None``, defaults to:

                ["timestamp", "value"]

        sep : str, optional
            Separator used for headers and data rows. Default is ``";"``.

        extension : str, optional
            File extension. Default is ``".txt"``.

        flush_every : int, optional
            Flush the Python file buffer after this many written rows.

            This does not drain the queue. It only forces rows that have already
            been written to the file object to be pushed from Python's buffer to
            the operating system.

            Smaller value:
                safer in case of crash, but slower.

            Larger value:
                faster, but more recently written rows may remain unflushed if
                the process crashes.

        flush_interval : float, optional
            Flush the Python file buffer at least every this many seconds.

            This is useful when the acquisition rate is low. For example, if
            ``flush_every=50`` but only 5 rows are written per second, then
            without ``flush_interval`` the file might not flush for 10 seconds.

            The file is always flushed on :meth:`close`.

        overflow_policy : {"raise", "block", "drop"}, optional
            Behavior when the queue is full.

            "raise":
                Raise a RuntimeError immediately. Recommended for experiments.

            "block":
                Block until queue space is available. No data loss, but the
                acquisition loop may pause.

            "drop":
                Discard the new row and print a warning. Fast, but data loss is
                possible.

        encoding : str, optional
            File encoding. Default is ``"utf-8"``.
        """

        if overflow_policy not in self.VALID_OVERFLOW_POLICIES:
            raise ValueError(
                f"overflow_policy must be one of {self.VALID_OVERFLOW_POLICIES}, "
                f"got {overflow_policy!r}."
            )

        self.path_to_file = Path(path_to_file).expanduser()
        self.path_to_file.mkdir(parents=True, exist_ok=True)

        if not extension.startswith("."):
            extension = "." + extension

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_filename = filename if filename else "data"

        self.filename = f"{timestamp}-{safe_filename}{extension}"
        self.path = self.path_to_file / self.filename

        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")

        self.metadata = metadata or {}
        self.headers = headers or ["timestamp", "value"]
        self.sep = sep

        self.flush_every = max(1, int(flush_every))
        self.flush_interval = max(0.0, float(flush_interval))

        self.overflow_policy = overflow_policy
        self.encoding = encoding

        self.buffer = Queue(maxsize=self.buffer_size)
        self.stop_event = threading.Event()

        self.closed = False

        self.rows_queued = 0
        self.rows_written = 0
        self.rows_dropped = 0

        self.file = open(
            self.path,
            "w",
            encoding=self.encoding,
            newline="\n",
        )

        self._write_metadata_and_header()

        self.thread = threading.Thread(
            target=self._write_loop,
            name="BufferedFileWriterThread",
            daemon=True,
        )
        self.thread.start()

        print(f"## BufferedFileWriter ## Writing to: {self.path}")

    def __enter__(self):
        """
        Enable use as a context manager.

        Example
        -------
        with BufferedFileWriter("Data", filename="test") as writer:
            writer.write_sv([1, 2, 3])
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _write_metadata_and_header(self) -> None:
        """
        Write metadata and header row.

        Metadata lines start with ``#`` so they can be ignored easily when
        reading the file as a table.
        """

        for key, value in self.metadata.items():
            self.file.write(f"# {key}: {value}\n")

        self.file.write(self.sep.join(map(str, self.headers)) + "\n")
        self.file.flush()

    def _write_loop(self) -> None:
        """
        Background writer thread.

        The loop continues until:

        - ``stop_event`` is set, and
        - the queue is empty.

        This allows :meth:`close` to flush all queued rows before closing.
        """

        rows_since_flush = 0
        last_flush_time = time.perf_counter()

        while not self.stop_event.is_set() or not self.buffer.empty():
            try:
                line = self.buffer.get(timeout=0.05)
            except Empty:
                rows_since_flush, last_flush_time = self._flush_if_needed(
                    rows_since_flush,
                    last_flush_time,
                    force_by_interval=True,
                )
                continue

            try:
                self.file.write(line + "\n")
                self.rows_written += 1
                rows_since_flush += 1

                rows_since_flush, last_flush_time = self._flush_if_needed(
                    rows_since_flush,
                    last_flush_time,
                    force_by_interval=True,
                )

            finally:
                self.buffer.task_done()

        self.file.flush()

    def _flush_if_needed(
        self,
        rows_since_flush: int,
        last_flush_time: float,
        force_by_interval: bool = True,
    ):
        """
        Flush file buffer if row-count or time-based threshold is reached.

        Returns
        -------
        tuple[int, float]
            Updated ``rows_since_flush`` and ``last_flush_time``.
        """

        should_flush = False

        if rows_since_flush >= self.flush_every:
            should_flush = True

        if force_by_interval and self.flush_interval > 0:
            elapsed = time.perf_counter() - last_flush_time
            if rows_since_flush > 0 and elapsed >= self.flush_interval:
                should_flush = True

        if should_flush:
            self.file.flush()
            rows_since_flush = 0
            last_flush_time = time.perf_counter()

        return rows_since_flush, last_flush_time

    def write(self, string: str) -> bool:
        """
        Queue a pre-formatted row.

        Parameters
        ----------
        string : str
            Row to write, without trailing newline.

        Returns
        -------
        bool
            ``True`` if the row was queued.
            ``False`` only if ``overflow_policy="drop"`` and the queue is full.

        Raises
        ------
        RuntimeError
            If called after :meth:`close`.

            If ``overflow_policy="raise"`` and the queue is full.
        """

        if self.closed:
            raise RuntimeError("BufferedFileWriter.write() called after close().")

        line = str(string)

        if self.overflow_policy == "block":
            self.buffer.put(line)
            self.rows_queued += 1
            return True

        if self.overflow_policy == "drop":
            try:
                self.buffer.put_nowait(line)
                self.rows_queued += 1
                return True
            except Full:
                self.rows_dropped += 1
                print(
                    "## BufferedFileWriter ## WARNING: queue is full. "
                    f"Dropping row. Dropped rows: {self.rows_dropped}"
                )
                return False

        if self.overflow_policy == "raise":
            try:
                self.buffer.put_nowait(line)
                self.rows_queued += 1
                return True
            except Full:
                raise RuntimeError(
                    "BufferedFileWriter queue is full. "
                    "The writer cannot keep up with acquisition. "
                    "Increase buffer_size, reduce write rate, or use "
                    "overflow_policy='block' or 'drop'."
                )

        raise RuntimeError(f"Unknown overflow_policy: {self.overflow_policy!r}")

    def write_sv(self, values) -> bool:
        """
        Queue a separator-separated row.

        Parameters
        ----------
        values : iterable
            Values to serialize. Each element is converted to text.

        Returns
        -------
        bool
            Same return value as :meth:`write`.
        """

        line = self.sep.join(self._format_value(value) for value in values)
        return self.write(line)

    @staticmethod
    def _format_value(value) -> str:
        """
        Convert a Python value to a compact string representation.

        Notes
        -----
        - ``None`` is written as an empty field.
        - floats are written with compact precision.
        """

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.10g}"

        return str(value)

    def qsize(self) -> int:
        """
        Return the approximate number of rows currently waiting in the queue.
        """

        return self.buffer.qsize()

    def stats(self) -> dict:
        """
        Return writer statistics.

        Returns
        -------
        dict
            Dictionary with queued, written, dropped, and current queue size.
        """

        return {
            "rows_queued": self.rows_queued,
            "rows_written": self.rows_written,
            "rows_dropped": self.rows_dropped,
            "queue_size": self.qsize(),
            "closed": self.closed,
            "path": str(self.path),
        }

    def close(self) -> None:
        """
        Stop the writer, drain the queue, flush the file buffer, and close.

        This method is safe to call more than once.
        """

        if self.closed:
            return

        # Prevent new writes from this point.
        self.closed = True

        # Tell the background thread to stop after the queue is empty.
        self.stop_event.set()

        # Wait until every queued item has called task_done().
        self.buffer.join()

        # Wait for the writer thread to exit.
        self.thread.join(timeout=5.0)

        if self.thread.is_alive():
            raise RuntimeError(
                "BufferedFileWriter thread did not stop within timeout. "
                "File may not have been fully flushed."
            )

        self.file.flush()
        self.file.close()

        print(
            "## BufferedFileWriter ## Closed. "
            f"Queued: {self.rows_queued}, "
            f"Written: {self.rows_written}, "
            f"Dropped: {self.rows_dropped}"
        )
