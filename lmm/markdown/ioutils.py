"""
Utilities to read/write markdown files to/from disc and handle errors
consistently for the lmm package.

This module provides robust I/O operations for markdown files with comprehensive
error handling, file size validation, encoding detection, and integration with
the lmm logging system.

Key Features:
- Automatic encoding detection with fallback strategies
- Configurable file size limits with warnings and hard limits
- Comprehensive error handling through LoggerBase abstraction
- Integration with markdown parsing and block structures
- Support for both file paths and direct string content

Logger Usage Patterns:
    The module supports different logger implementations for various use cases:

    1. ConsoleLogger - For interactive development and debugging:
        >>> from lmm.utils.logging import ConsoleLogger
        >>> logger = ConsoleLogger(__name__)
        >>> content = load_markdown("file.md", logger=logger)
        # Errors and warnings printed to console

    2. FileLogger - For production logging to files:
        >>> from lmm.utils.logging import FileLogger
        >>> from pathlib import Path
        >>> logger = FileLogger(__name__, Path("app.log"))
        >>> content = load_markdown("file.md", logger=logger)
        # Errors and warnings written to app.log

    3. ExceptionConsoleLogger - For strict error handling:
        >>> from lmm.utils.logging import ExceptionConsoleLogger
        >>> logger = ExceptionConsoleLogger(__name__)
        >>> content = load_markdown("file.md", logger=logger)
        # Raises RuntimeError on any error condition

    4. LoglistLogger - For testing and programmatic access:
        >>> from lmm.utils.logging import LoglistLogger
        >>> logger = LoglistLogger()
        >>> content = load_markdown("file.md", logger=logger)
        >>> errors = logger.get_logs(level=3)  # Get error-level logs only

Module Relationships:
    This module serves as the I/O layer between file system operations and the
    markdown parsing system:

    File System ←→ ioutils.py ←→ parse_markdown.py ←→ Application

    - Depends on lmm.utils.ioutils for basic file validation
    - Depends on lmm.utils.logging for error reporting abstraction
    - Integrates with lmm.markdown.parse_markdown for block structures
    - Used by higher-level modules for markdown file processing

Performance Characteristics:
    - File size checking: O(1) - single stat() call
    - Encoding detection: O(n) where n is detection sample size (1-10KB)
    - UTF-8 detection: Fast path with 1KB sample
    - Chardet detection: Slower but more accurate with 10KB sample
    - Memory usage: Proportional to file size (entire file loaded into memory)
    - Recommended limits: 50MB max, 10MB warning (configurable)

    For large files, consider:
    - Increasing max_size_mb parameter if needed
    - Using streaming approaches for files > 100MB
    - Monitoring memory usage in production environments
"""

from pathlib import Path
from lmm.utils.ioutils import validate_file
from lmm.utils.ioutils import string_to_path_or_string
from .parse_markdown import Block, ErrorBlock
from .parse_markdown import serialize_blocks, blocklist_errors

# Set up default logger
from lmm.utils.logging import get_logger, LoggerBase  # fmt: skip
logger: LoggerBase = get_logger(__name__)


def _check_file_size(
    file_path: Path,
    max_size_mb: float,
    warn_size_mb: float,
    logger: LoggerBase,
) -> bool:
    """
    Check file size against limits and log warnings/errors as appropriate.

    Args:
        file_path: Path to the file to check
        max_size_mb: Maximum allowed file size in MB
        warn_size_mb: File size in MB that triggers a warning
        logger: Logger for reporting issues

    Returns:
        True if file size is acceptable, False if it exceeds max_size_mb
    """
    try:
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Skip checks if limits are negative (effectively disables the check)
        if max_size_mb > 0 and file_size_mb > max_size_mb:
            logger.error(
                f"File {file_path} is too large ({file_size_mb:.1f}MB). "
                f"Maximum allowed size is {max_size_mb}MB."
            )
            return False
        elif warn_size_mb > 0 and file_size_mb > warn_size_mb:
            logger.warning(
                f"File {file_path} is large ({file_size_mb:.1f}MB). "
                f"Consider using smaller files for better performance."
            )

        return True
    except OSError as e:
        logger.error(
            f"Could not check file size for {file_path}: {e}"
        )
        return False


# Import chardet at module level for easier mocking
try:
    import chardet
except ImportError:
    chardet = None


def _detect_encoding(file_path: Path, logger: LoggerBase) -> str:
    """
    Detect file encoding using multiple strategies.

    Args:
        file_path: Path to the file
        logger: Logger for reporting detection results

    Returns:
        Detected encoding string (defaults to 'utf-8' if detection fails)
    """
    # Try UTF-8 first (most common for markdown files)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Read first 1KB to test
        logger.info(f"File {file_path} detected as UTF-8 encoding")
        return 'utf-8'
    except UnicodeDecodeError:
        pass

    # Try to use chardet if available
    if chardet is not None:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(
                    10240
                )  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            if result['encoding'] and result['confidence'] > 0.7:
                detected_encoding = result['encoding']
                logger.info(
                    f"File {file_path} detected as {detected_encoding} encoding "
                    f"(confidence: {result['confidence']:.2f})"
                )
                return detected_encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
    else:
        logger.info(
            "chardet library not available, using fallback encoding detection"
        )

    # Fallback to common encodings
    fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in fallback_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Test read
            logger.info(
                f"File {file_path} using fallback encoding: {encoding}"
            )
            return encoding
        except UnicodeDecodeError:
            continue

    # Last resort - use utf-8 with error handling
    logger.warning(
        f"Could not detect encoding for {file_path}, using UTF-8 with error replacement"
    )
    return 'utf-8'


# Load markdown
def load_markdown(
    source: str | Path,
    logger: LoggerBase = logger,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    encoding: str | None = None,
    auto_detect_encoding: bool = True,
) -> str:
    """
    Loads a text file (intended for markdown files).
    The purpose of this function is to catch errors through
    a LoggerBase object, instead of raising errors in the I/O.

    Args:
        source (str | Path): the source file. If the source is a
            multiline string, or if its not a file, returns the
            string itself.
        logger (LoggerBase): a logger object (defaults to console).
        max_size_mb (float): maximum file size in MB (default: 50.0).
        warn_size_mb (float): file size in MB to trigger warning (default: 10.0).
        encoding (str | None): specific encoding to use. If None and
            auto_detect_encoding is True, encoding will be detected automatically.
        auto_detect_encoding (bool): whether to automatically detect file encoding
            (default: True).

    Note: I/O errors will be conveyed to the logger object. Use an
        ExceptionConsoleLogger object to raise errors.
    """

    # Make the source a Path object if it points to file
    if isinstance(source, str):
        source = string_to_path_or_string(source)

    # Load if Path object, or return
    if isinstance(source, Path):
        if validate_file(source, logger) is None:
            return ""

        # Check file size limits
        if not _check_file_size(
            source, max_size_mb, warn_size_mb, logger
        ):
            return ""

        # Determine encoding to use
        file_encoding = encoding
        if file_encoding is None and auto_detect_encoding:
            file_encoding = _detect_encoding(source, logger)
        elif file_encoding is None:
            file_encoding = 'utf-8'  # Default fallback

        try:
            # Handle potential encoding errors gracefully
            if file_encoding == 'utf-8':
                content = source.read_text(
                    encoding=file_encoding, errors='replace'
                )
            else:
                content = source.read_text(encoding=file_encoding)
        except (IOError, OSError) as e:
            logger.error(f"I/O error reading file {source}: {e}")
            return ""
        except UnicodeDecodeError as e:
            logger.error(
                f"Encoding error reading file {source} with {file_encoding}: {e}"
            )
            # Try UTF-8 with error replacement as last resort
            try:
                content = source.read_text(
                    encoding='utf-8', errors='replace'
                )
                logger.warning(
                    f"Fallback to UTF-8 with error replacement for {source}"
                )
            except Exception as fallback_e:
                logger.error(
                    f"Final fallback failed for {source}: {fallback_e}"
                )
                return ""
        except Exception as e:
            logger.error(
                f"Unexpected error reading file {source}: {e}"
            )
            return ""
    else:
        content = source

    return content


def save_markdown(
    dest: str | Path,
    content: list[Block] | str,
    logger: LoggerBase = logger,
) -> bool:
    """
    Save markdown blocks to a file.

    Args:
        dest (str | Path): the file to save the markdown to.
        logger (LoggerBase): a logger object, defaulting to
            a console logger.

    Returns:
        a boolean indicating success or failure.

    Note: I/O errors are conveyed through the logger object. Use an
        ExceptionConsoleLogger object to raise errors.
    """
    if not content:
        return False

    try:
        # Check save path
        save_path = Path(dest)
        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize content if it is a list of blocks
        match content:
            case str():
                pass
            case list():
                content = serialize_blocks(content)
            case _:
                logger.critical('Invalid object given to serialize')
                return False

        if not content:
            logger.warning("Empty markdown")
            return False  # no file created

        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(content)

    except (IOError, OSError) as e:
        logger.error(f"I/O error saving markdown to {dest}: {str(e)}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error saving markdown to {dest}: {str(e)}"
        )
        return False
        # Note: Don't fail here as we've already processed the file
        # Just couldn't save it

    return True


def report_error_blocks(
    blocks: list[Block], logger: LoggerBase = logger
) -> list[Block]:
    """
    Checks the existence of error blocks. If there are any, they are
    reported to the logger object.

    Args:
        blocks: the block list to check for error blocks
        logger (LoggerBase): a logger object, defaulting to
            a console logger, which reports the errors.

    Returns:
        a list without error blocks.

    Note: I/O errors are conveyed through the logger object. Use an
        ExceptionConsoleLogger object to raise errors.
        Use blocklist_errors to filter the block list for error
        blocks.
    """
    if not blocks:
        return []

    errblocks: list[ErrorBlock] = blocklist_errors(blocks)
    if not errblocks:  # all ok
        return blocks

    # Handle single error block (usually file loading failure)
    if len(blocks) == 1 and len(errblocks) == 1:
        _report_single_error_block(errblocks[0], logger)
        return []

    # Handle multiple error blocks
    _report_multiple_error_blocks(errblocks, logger)
    return [b for b in blocks if not isinstance(b, ErrorBlock)]


def _report_single_error_block(
    error_block: ErrorBlock, logger: LoggerBase
) -> None:
    """Report a single error block (typically from file loading failure)."""
    error_parts = [
        "Error loading markdown file:",
        error_block.content,
    ]
    if error_block.errormsg:
        error_parts.append(error_block.errormsg)
    logger.error("\n".join(error_parts))


def _report_multiple_error_blocks(
    error_blocks: list[ErrorBlock], logger: LoggerBase
) -> None:
    """Report multiple error blocks from parsing failures."""
    for block in error_blocks:
        error_parts = ["Errors when parsing markdown file"]
        if block.errormsg:
            error_parts.append(block.errormsg)
        if str(block.origin).strip():
            error_parts.extend(
                [
                    "",
                    " Offending content:",
                    "------------",
                    str(block.origin),
                    "------------",
                ]
            )
        logger.warning("\n".join(error_parts))
