import time
from pathlib import Path
import threading
from queue import Queue, Empty, Full


class FileWriter:

    def __init__(self, path_to_file, filename='', append=False, sep=';'):
        self.path_str = path_to_file
        self.filename_str = time.strftime("%Y%m%d_%H%M%S-") + filename + '.txt'
        self.path = Path(self.path_str) / self.filename_str
        if append:
            self.file = open(self.path, 'a')
        else:
            self.file = open(self.path, 'w')
        self.sep = sep

    def write(self, stringa):
        self.file.write(stringa + "\n")

    def write_sv(self, lista):
        lista = self.sep.join(str(elem) for elem in lista)
        self.write(lista)

    def is_open(self):
        return not self.file.closed

    def close(self):
        self.file.close()

class BufferedFileWriter:
    """
    A class that writes data to a TXT file using a memory buffer and a background thread
    for asynchronous disk writing, improving performance for time-critical tasks.
    """

    def __init__(self, path_to_file, filename='', buffer_size=100, metadata=None, headers=None,sep=';'):

        """
        Initialize the BufferedFileWriter.

        Args:
            path_to_file (str): Directory where the file will be saved.
            filename (str): Base filename. A timestamp is prepended.
            buffer_size (int): Maximum number of entries to hold in the buffer before writing to disk.
            metadata (dict, optional): Metadata to include at the beginning of the file.
            headers (list, optional): List of column names for the TXT file.
            sep: (string, optional): string separator (default ';')
        """
        self.path_str = path_to_file
        self.filename_str = time.strftime("%Y%m%d_%H%M%S-") + filename + '.txt'
        self.path = Path(self.path_str) / self.filename_str
        self.buffer_size = buffer_size
        self.metadata = metadata or {}
        self.headers = headers or ["timestamp", "value"]
        self.sep = sep

        # Open the file for writing and write metadata
        self.file = open(self.path, 'w')
        self._write_metadata()

        # Internal buffer and threading setup
        self.buffer = Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._write_to_file, daemon=True)

        # Start the background thread
        self.thread.start()

    def _write_metadata(self,sep=';'):
        """
        Write metadata at the beginning of the file as comments.
        """
        for key, value in self.metadata.items():
            self.file.write(f"# {key}: {value}\n")
        self.file.write(self.sep.join(self.headers) + "\n")  # Write the header row
        self.file.flush()

    def _write_to_file(self):
        """
        Background thread function to write buffered data to the file.
        """
        while not self.stop_event.is_set() or not self.buffer.empty():
            try:
                # Retrieve data from the buffer
                data = self.buffer.get(timeout=0.1)
                self.file.write(data + "\n")
                self.file.flush()  # Ensure data is written to disk
                self.buffer.task_done()
            except Empty:
                continue
            time.sleep(0.001)

    def write(self, string):
        """
        Add a string to the buffer to be written to the file.

        Args:
            string (str): The string to write.
        """
        try:
            self.buffer.put_nowait(string)
        except Full:
            print("## BufferedFileWriter ## WARNING: Buffer is full. Discarding value. Increase buffer size or reduce data to write.")

    def write_sv(self, lista):
        """
        Add a list of values to the buffer, joined by a separator, to be written to the file.

        Args:
            lista (list): List of values to write.
            sep (str): Separator to use for joining the list.
        """
        string = self.sep.join(map(str, lista))
        self.write(string)

    def close(self):
        """
        Close the file, ensuring all buffered data is written first.
        """
        self.stop_event.set()
        self.thread.join()  # Wait for the background thread to finish
        self.file.close()