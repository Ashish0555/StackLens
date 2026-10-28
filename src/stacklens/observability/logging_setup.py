"""Logging setup for console and file output.

Most of this file is about other people's libraries. The guardrail stack pulls
in transformers, presidio and structlog, and each of them logs at INFO or
WARNING about things that are normal here: a recogniser skipped for the wrong
language, a model being loaded, a progress bar for weights read from disk.
None of it is a problem, all of it reaches the terminal, and a screen of
warnings during a question teaches the reader to ignore warnings.

They are quietened for the console and left intact in the file, so the noise
is still there when someone actually needs it.
"""

from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from stacklens.config import get_settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_FILE = "stacklens.log"

# Five files of two megabytes: enough history to explain yesterday, and a
# ceiling the disk can be sized against.
_LOG_MAX_BYTES = 2 * 1024 * 1024
_LOG_BACKUPS = 5

# Libraries that report normal conditions at WARNING. presidio names every
# recogniser it skips for the wrong language, once per question; transformers
# announces every model it loads; huggingface_hub asks for a token on every
# anonymous request.
_NOISY = (
    "presidio-analyzer",
    "presidio_analyzer",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
    "llm_guard",
    "torch",
    "urllib3",
    "httpx",
)


def setup_logging(*, verbose: bool = False, to_file: bool = True) -> None:
    """Configure the root logger once, for the whole process.

    Console output follows the verbose flag; the file always keeps INFO so a
    problem reported after the fact is still traceable.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    if to_file:
        root.addHandler(_build_file_handler(get_settings().log_dir))

    quieten_dependencies()


def quieten_dependencies() -> None:
    """Stop third party libraries from narrating routine work.

    **`--verbose` does not lift this.** Verbose means "show me what this
    application is doing", not "show me presidio's opinion about Italian
    recognisers". Those two were the same switch once, and the result was an
    API that logged eleven warnings per question about languages it never
    asked for.

    What is genuinely wrong still surfaces: these are set to ERROR, not
    silenced.
    """
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.ERROR)

    # transformers keeps its own verbosity, separate from the logging module,
    # and draws a progress bar for weights it reads from local disk.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _quieten_transformers()
    _quieten_structlog()

    # torch deprecates its own APIs to a user who never called them.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")


def _quieten_transformers() -> None:
    """Turn off the library's own verbosity, which ignores the logging module."""
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def _quieten_structlog() -> None:
    """llm_guard logs through structlog, which bypasses handler levels."""
    try:
        import structlog
    except ImportError:
        return

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
        cache_logger_on_first_use=True,
    )


def _build_file_handler(log_dir: Path) -> logging.Handler:
    """A rotating file, not a growing one.

    The log had reached six megabytes in a few days of use, and a file that
    only grows is a file nobody deletes and eventually a disk nobody expected
    to fill.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / _LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    return handler
