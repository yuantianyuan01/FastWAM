import hashlib
import io
import logging
import os
import socket
import sys
import threading
from contextlib import closing, contextmanager
from typing import cast

from torch import multiprocessing as mp

from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR

HUMAN_LOG_LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error", "none")
"""
Available log levels: "debug", "info", "warning", "error", "none"
"""

_LOGGER: logging.Logger | None = None
_LOG_FILE: str | None = None

# Thread-local storage for worker loggers
_worker_logger_storage = threading.local()


class ColoredFormatter(logging.Formatter):
    """Format a log string with colors.

    This implementation taken (with modifications) from
    https://stackoverflow.com/a/384125.
    """

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "ERROR": RED,
        "CRITICAL": MAGENTA,
    }

    def __init__(self, fmt: str, datefmt: str | None = None, use_color=True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            levelname_with_color = (
                self.COLOR_SEQ % (30 + self.COLORS[levelname]) + levelname + self.RESET_SEQ
            )
            record.levelname = levelname_with_color
            formated_record = logging.Formatter.format(self, record)
            record.levelname = levelname  # Resetting levelname as `record` might be used elsewhere
            return formated_record
        else:
            return logging.Formatter.format(self, record)


def get_logger() -> logging.Logger:
    """Get a `logging.Logger` to stderr. It can be called whenever we wish to
    log some message. Messages can get mixed-up
    (https://docs.python.org/3.6/library/multiprocessing.html#logging), but it
    works well in most cases.

    # Returns

    logger: the `logging.Logger` object
    """
    if _new_logger():
        if mp.current_process().name == "MainProcess":
            _new_logger(logging.DEBUG)
        _set_log_formatter()
    return _LOGGER


def _human_log_level_to_int(human_log_level):
    human_log_level = human_log_level.lower().strip()
    assert human_log_level in HUMAN_LOG_LEVELS, f"unknown human_log_level {human_log_level}"

    if human_log_level == "debug":
        log_level = logging.DEBUG
    elif human_log_level == "info":
        log_level = logging.INFO
    elif human_log_level == "warning":
        log_level = logging.WARNING
    elif human_log_level == "error":
        log_level = logging.ERROR
    elif human_log_level == "none":
        log_level = logging.CRITICAL + 1
    else:
        raise NotImplementedError(f"Unknown log level {human_log_level}.")
    return log_level


def init_logging(human_log_level: str = "info", log_file: str | None = None) -> None:
    """Init the `logging.Logger`.

    It should be called only once in the app (e.g. in `main`). It sets
    the log_level to one of `HUMAN_LOG_LEVELS`. And sets up handlers
    for stderr and optionally a log file. The logging level is propagated to all subprocesses.

    Args:
        human_log_level: Log level as a human-readable string. One of "debug", "info", "warning", "error", "none".
        log_file: Optional path to a log file. If provided, logs will also be written to this file.
                  All worker loggers will also write to the same file with worker ID prefixes.
    """
    global _LOG_FILE
    _LOG_FILE = log_file
    _new_logger(_human_log_level_to_int(human_log_level))
    _set_log_formatter()


def update_log_level(logger, human_log_level: str) -> None:
    logger.setLevel(_human_log_level_to_int(human_log_level))


def find_free_port(address: str = "127.0.0.1") -> int:
    """Finds a free port for distributed training.

    # Returns

    port: port number that can be used to listen
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return port


def _new_logger(log_level: int | None = None) -> bool:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = mp.get_logger()
        if log_level is not None:
            get_logger().setLevel(log_level)
        return True
    if log_level is not None:
        get_logger().setLevel(log_level)
    return False


def _set_log_formatter():
    assert _LOGGER is not None

    if _LOGGER.getEffectiveLevel() <= logging.CRITICAL:
        add_style_to_logs = True  # In case someone wants to turn this off manually.

        if add_style_to_logs:
            default_format = (
                "[$BOLD%(asctime)s$RESET %(levelname)s %(filename)s:%(lineno)d] %(message)s"
            )
            default_format = default_format.replace("$BOLD", ColoredFormatter.BOLD_SEQ).replace(
                "$RESET", ColoredFormatter.RESET_SEQ
            )
        else:
            default_format = "%(asctime)s %(levelname)s: %(message)s\t[%(filename)s:%(lineno)d]"

        short_date_format = "%m/%d %H:%M:%S"
        log_format = "default"
        if log_format == "default":
            fmt = default_format
            datefmt = short_date_format
        elif log_format == "defaultMilliseconds":
            fmt = default_format
            datefmt = None
        else:
            fmt = log_format
            datefmt = short_date_format

        if add_style_to_logs:
            formatter = ColoredFormatter(
                fmt=fmt,
                datefmt=datefmt,
            )
        else:
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # Create console handler (stderr)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.addFilter(cast(logging.Filter, _MolmoSpacesMessageFilter(os.getcwd())))
        _LOGGER.addHandler(ch)

        # Create file handler if log_file is specified
        if _LOG_FILE:
            # Ensure the directory exists
            log_dir = os.path.dirname(_LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create file handler with plain formatter (no colors for file)
            file_formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]",
                datefmt=short_date_format,
            )
            fh = logging.FileHandler(_LOG_FILE)
            fh.setFormatter(file_formatter)
            fh.addFilter(cast(logging.Filter, _MolmoSpacesMessageFilter(os.getcwd())))
            _LOGGER.addHandler(fh)

        sys.excepthook = _excepthook
        # sys.stdout = cast(io.TextIOWrapper, _StreamToLogger())

    return _LOGGER


class _StreamToLogger:
    def __init__(self) -> None:
        self.linebuf = ""

    def write(self, buf) -> None:
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line[-1] == "\n":
                cast(logging.Logger, _LOGGER).info(line.rstrip())
            else:
                self.linebuf += line

    def flush(self) -> None:
        if self.linebuf != "":
            cast(logging.Logger, _LOGGER).info(self.linebuf.rstrip())
        self.linebuf = ""


class _WorkerStreamToLogger:
    """Stream logger that redirects output to the current worker's logger"""

    def __init__(self, worker_logger: logging.Logger, worker_id: int) -> None:
        self.worker_logger = worker_logger
        self.worker_id = worker_id
        self.linebuf = ""
        self._thread_id = threading.get_ident()

    def write(self, buf) -> None:
        # Only process output from the thread that created this stream
        if threading.get_ident() != self._thread_id:
            return

        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line[-1] == "\n":
                self.worker_logger.info(line.rstrip())
            else:
                self.linebuf += line

    def flush(self) -> None:
        # Only process output from the thread that created this stream
        if threading.get_ident() != self._thread_id:
            return

        if self.linebuf != "":
            self.worker_logger.info(self.linebuf.rstrip())
        self.linebuf = ""


def _excepthook(*args) -> None:
    # noinspection PyTypeChecker
    get_logger().error(msg="Uncaught exception:", exc_info=args)


class _MolmoSpacesMessageFilter:
    def __init__(self, working_directory: str) -> None:
        self.working_directory = working_directory

    # noinspection PyMethodMayBeStatic
    def filter(self, record):
        # TODO: Does this work when pip-installing MolmoSpaces?
        return int(
            self.working_directory in record.pathname
            or str(ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR) in record.pathname
            or "main" in record.pathname
        )


class ImportChecker:
    def __init__(self, msg=None) -> None:
        self.msg = msg

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, value, traceback) -> bool:
        if exc_type is ModuleNotFoundError and self.msg is not None:
            value.msg += self.msg
        return exc_type is None


def get_worker_logger(worker_id: int) -> logging.Logger:
    """Create a logger specific to a worker that includes the worker ID in all messages"""
    # Create a new logger for this worker
    worker_logger = logging.getLogger(f"worker_{worker_id}")

    # Only add handler if it doesn't already exist
    if not worker_logger.handlers:
        # Assign a consistent color to this worker based on worker_id
        # Use a hash of worker_id to get a deterministic color
        color_hash = hashlib.md5(str(worker_id).encode()).hexdigest()
        color_index = int(color_hash[:2], 16) % 8  # Use first 2 hex chars to get 0-7

        # Create a custom formatter that includes worker ID with color
        default_format = "[$BOLD%(asctime)s $WORKER_COLORWorker %(worker_id)s$RESET$RESET %(levelname)s %(filename)s:%(lineno)d] %(message)s"
        default_format = default_format.replace("$BOLD", ColoredFormatter.BOLD_SEQ).replace(
            "$RESET", ColoredFormatter.RESET_SEQ
        )

        # Create a formatter that captures the worker_id and applies color
        class WorkerFormatter(ColoredFormatter):
            def __init__(self, fmt, datefmt, worker_id, color_index) -> None:
                super().__init__(fmt, datefmt)
                self.worker_id = worker_id
                self.color_index = color_index

            def format(self, record):
                record.worker_id = self.worker_id
                # Apply the worker color
                worker_color_seq = ColoredFormatter.COLOR_SEQ % (30 + self.color_index)
                formatted = super().format(record)
                formatted = formatted.replace("$WORKER_COLOR", worker_color_seq)
                return formatted

        worker_formatter = WorkerFormatter(
            fmt=default_format,
            datefmt="%m/%d %H:%M:%S",
            worker_id=worker_id,
            color_index=color_index,
        )

        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(worker_formatter)
        worker_logger.addHandler(ch)

        # Add file handler if main logger has file logging enabled
        if _LOG_FILE:
            # Use the same log file as the main logger
            # Create file handler with plain formatter (no colors for file)
            class WorkerFileFormatter(logging.Formatter):
                def __init__(self, fmt, datefmt, worker_id) -> None:
                    super().__init__(fmt, datefmt)
                    self.worker_id = worker_id

                def format(self, record):
                    record.worker_id = self.worker_id
                    return super().format(record)

            worker_file_formatter = WorkerFileFormatter(
                fmt="%(asctime)s %(levelname)s: [Worker %(worker_id)s] %(message)s\t[%(filename)s: %(lineno)d]",
                datefmt="%m/%d %H:%M:%S",
                worker_id=worker_id,
            )

            fh = logging.FileHandler(_LOG_FILE)
            fh.setFormatter(worker_file_formatter)
            worker_logger.addHandler(fh)

        # Set the same level as the main logger
        main_logger = get_logger()
        worker_logger.setLevel(main_logger.getEffectiveLevel())

        # Prevent propagation to avoid duplicate logs
        worker_logger.propagate = False

        # Configure the root molmo_spaces logger to use worker logger's handlers
        # This ensures module-level loggers (like log = logging.getLogger(__name__))
        # in molmo_spaces modules work properly in worker processes
        molmo_spaces_logger = logging.getLogger("molmo_spaces")
        molmo_spaces_logger.handlers = worker_logger.handlers.copy()
        molmo_spaces_logger.setLevel(worker_logger.level)
        molmo_spaces_logger.propagate = False  # Don't propagate to root to avoid duplication

    return worker_logger


def setup_worker_stdout(worker_logger: logging.Logger, worker_id: int = None) -> None:
    """Set up stdout redirection for a worker thread to use the worker's logger"""
    # Store the current stdout for this thread (before worker redirection)
    if not hasattr(_worker_logger_storage, "previous_stdout"):
        _worker_logger_storage.previous_stdout = sys.stdout

    # Extract worker_id from logger name if not provided
    if worker_id is None:
        logger_name = worker_logger.name
        if logger_name.startswith("worker_"):
            try:
                worker_id = int(logger_name.split("_")[1])
            except (IndexError, ValueError):
                worker_id = 0

    # Create worker-specific stream logger and redirect stdout
    worker_stream = _WorkerStreamToLogger(worker_logger, worker_id)
    _worker_logger_storage.worker_stream = worker_stream
    sys.stdout = cast(io.TextIOWrapper, worker_stream)


def restore_worker_stdout() -> None:
    """Restore the previous stdout for the current thread"""
    if hasattr(_worker_logger_storage, "previous_stdout"):
        sys.stdout = _worker_logger_storage.previous_stdout
        delattr(_worker_logger_storage, "previous_stdout")
    if hasattr(_worker_logger_storage, "worker_stream"):
        delattr(_worker_logger_storage, "worker_stream")


@contextmanager
def worker_stdout_context(worker_logger: logging.Logger, worker_id: int = None):
    """Context manager for worker-specific stdout redirection"""
    yield
    # setup_worker_stdout(worker_logger, worker_id)
    # try:
    #     yield
    # finally:
    #     restore_worker_stdout()
