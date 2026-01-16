import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue


class FileWriter:
    """
    Simple synchronous text file writer.

    This class creates a timestamped ``.txt`` file and exposes convenience
    methods to write either a single string (one line) or a list of values
    separated by a custom delimiter.

    Notes
    -----
    - This writer is **synchronous**: each call writes directly to disk.
    - The filename is always timestamped to reduce accidental overwrites.
    - The file is opened immediately on initialization and must be closed
      by calling :meth:`close`.
    """

    def __init__(self, path_to_file, filename: str = "", append: bool = False, sep: str = ";"):
        """
        Initialize the writer and open the output file.

        Parameters
        ----------
        path_to_file : str or pathlib.Path
            Directory where the file will be created.
        filename : str, optional
            Base filename (without extension). A timestamp is prepended and the
            extension ``.txt`` is appended.
        append : bool, optional
            If ``True``, open the file in append mode (``'a'``). If ``False``,
            overwrite/create the file (``'w'``).
        sep : str, optional
            Separator used by :meth:`write_sv` to join list values.
        """
        self.path_str = path_to_file
        self.filename_str = time.strftime("%Y%m%d_%H%M%S-") + filename + ".txt"
        self.path = Path(self.path_str).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path / self.filename_str

        # Choose file open mode.
        mode = "a" if append else "w"
        self.file = open(self.path, mode)
        self.sep = sep

    def write(self, stringa: str) -> None:
        """
        Write a single line to the file.

        Parameters
        ----------
        stringa : str
            Line content. A newline character is automatically appended.
        """
        self.file.write(stringa + "\n")

    def write_sv(self, lista) -> None:
        """
        Write a list of values as a separator-joined line.

        Parameters
        ----------
        lista : iterable
            Values to serialize. Each element is converted to ``str``.
            The resulting line is written with a trailing newline.
        """
        line = self.sep.join(str(elem) for elem in lista)
        self.write(line)

    def is_open(self) -> bool:
        """
        Check whether the underlying file handle is still open.

        Returns
        -------
        bool
            ``True`` if the file is open, otherwise ``False``.
        """
        return not self.file.closed

    def close(self) -> None:
        """
        Close the underlying file handle.

        Notes
        -----
        After closing, further write attempts will raise an exception.
        """
        self.file.close()


class BufferedFileWriter:
    """
    Buffered asynchronous text file writer.

    This writer uses an in-memory :class:`queue.Queue` as a buffer and a
    background thread to flush lines to disk. This is useful for time-critical
    data acquisition loops where direct disk writes would introduce latency.

    The file format is:

    - optional metadata lines (prefixed with ``#``)
    - a header row (separator-joined)
    - data rows (one per buffered entry)

    Notes
    -----
    - The background thread is started automatically at initialization.
    - Call :meth:`close` to stop the thread and ensure all queued data is written.
    - If the buffer is full, new entries are discarded and a ``print`` warning
      is emitted (per your request, no logging is used).
    """

    def __init__(
        self,
        path_to_file,
        filename: str = "",
        buffer_size: int = 100,
        metadata=None,
        headers=None,
        sep: str = ";",
    ):
        """
        Initialize the BufferedFileWriter.

        Parameters
        ----------
        path_to_file : str or pathlib.Path
            Directory where the file will be created.
        filename : str, optional
            Base filename (without extension). A timestamp is prepended and the
            extension ``.txt`` is appended.
        buffer_size : int, optional
            Maximum number of queued lines allowed before new values are dropped.
        metadata : dict, optional
            Metadata written at the top of the file as comment lines in the form
            ``# key: value``.
        headers : list of str, optional
            Column names written as the first non-metadata row.
        sep : str, optional
            Separator used for header and list serialization (default ``';'``).
        """
        self.path_str = path_to_file
        self.filename_str = time.strftime("%Y%m%d_%H%M%S-") + filename + ".txt"
        self.path = Path(self.path_str).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path / self.filename_str

        self.buffer_size = buffer_size
        self.metadata = metadata or {}
        self.headers = headers or ["timestamp", "value"]
        self.sep = sep

        # Open the file for writing and write metadata + header immediately.
        self.file = open(self.path, "w")
        self._write_metadata()

        # Queue used as a bounded buffer (drops values on overflow in write()).
        self.buffer = Queue(maxsize=buffer_size)

        # Event signaling the thread to stop; thread continues until queue drained.
        self.stop_event = threading.Event()

        # Background daemon thread that writes queued lines to disk.
        self.thread = threading.Thread(target=self._write_to_file, daemon=True)
        self.thread.start()

    def _write_metadata(self) -> None:
        """
        Write metadata and header row at the beginning of the file.

        Notes
        -----
        Metadata entries are written as comment lines (prefixed with ``#``),
        followed by a header row joined with :attr:`sep`.
        """
        for key, value in self.metadata.items():
            self.file.write(f"# {key}: {value}\n")

        # Header row
        self.file.write(self.sep.join(self.headers) + "\n")
        self.file.flush()

    def _write_to_file(self) -> None:
        """
        Background thread loop that drains the queue and writes lines to disk.

        The loop exits only when:
        - :attr:`stop_event` is set, AND
        - the buffer queue is empty

        This ensures :meth:`close` can flush remaining queued data reliably.
        """
        while not self.stop_event.is_set() or not self.buffer.empty():
            try:
                # Retrieve a queued line; timeout allows periodic stop checks.
                data = self.buffer.get(timeout=0.1)

                self.file.write(data + "\n")
                self.file.flush()

                self.buffer.task_done()
            except Empty:
                # Nothing available right now; try again.
                continue

            # Small sleep to reduce tight-loop CPU usage.
            time.sleep(0.001)

    def write(self, string: str) -> None:
        """
        Queue a pre-formatted line for writing.

        Parameters
        ----------
        string : str
            The line to write (without a trailing newline). A newline will be
            appended by the writer thread.

        Notes
        -----
        If the buffer is full, the value is discarded and a warning is printed.
        """
        try:
            self.buffer.put_nowait(string)
        except Full:
            print(
                "## BufferedFileWriter ## WARNING: Buffer is full. Discarding value. "
                "Increase buffer size or reduce data to write."
            )

    def write_sv(self, lista) -> None:
        """
        Queue a list of values as a separator-joined line.

        Parameters
        ----------
        lista : iterable
            Values to serialize. Each element is converted to ``str`` and joined
            with :attr:`sep`.
        """
        line = self.sep.join(map(str, lista))
        self.write(line)

    def close(self) -> None:
        """
        Stop the background thread and close the file.

        This method:
        1. Signals the thread to stop
        2. Waits for the thread to finish flushing queued data
        3. Closes the underlying file handle

        Notes
        -----
        Always call this method to avoid losing buffered data.
        """
        self.stop_event.set()
        self.thread.join()
        self.file.close()
