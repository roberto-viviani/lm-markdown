"""
Utilities to read/write markdown files to/from disc and handle errors
consistently for the lmm package.
"""

from pathlib import Path
from lmm.utils.ioutils import validate_file
from lmm.utils.ioutils import string_to_path_or_string
from .parse_markdown import Block, ErrorBlock
from .parse_markdown import serialize_blocks, blocklist_errors

# Set up default logger
from lmm.utils.logging import get_logger, LoggerBase  # fmt: skip
logger: LoggerBase = get_logger(__name__)


# Load markdown
def load_markdown(
    source: str | Path, logger: LoggerBase = logger
) -> str:
    """
    Loads a text file (intended for markdown files).
    The purpose of this function is to catch errors through
    a LoggerBase object, instead of raising errors in the I/O.

    Args:
        source (str | Path): the source file. If the source is a
            multiline string, or if its not a file, returns the
            string itself.
        logger (LoggerBase): a looger object (defaults to console).

    Note: I/O errors will be conveyed to the logger object. Use an
        ExceptionConsoleLogger object to raise errors.
    """

    # Make the source a Path object if it points to file
    if isinstance(source, str):
        source = string_to_path_or_string(source)

    # Load if Path object, or return
    if isinstance(source, Path):
        if not validate_file(source):
            return ""
        try:
            content = source.read_text(encoding='utf-8')
        except IOError as e:
            logger.error(f"I/O error reading file {source}: {e}")
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

    except Exception as e:
        logger.error(f"Error saving markdown to {dest}: {str(e)}")
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
    if 0 == len(errblocks):  # all ok
        return blocks

    if len(blocks) == 1:  # only one error block
        # case for a single error block coding for failure to
        # load or parse the markdown file
        errmsg = (
            "Error loading markdown file:\n"
            + f"{errblocks[0].content}"
        )
        if errblocks[0].errormsg:
            errmsg += "\n" + errblocks[0].errormsg
        logger.error(errmsg)
        return []
    else:
        # in all other cases, we enumerate the error messages
        for b in errblocks:
            errinfo = "Errors when parsing markdown file"
            if b.errormsg:
                errinfo += ":\n" + b.errormsg
            if str(b.origin).strip():
                errinfo += (
                    "\n\n Offending content:\n------------\n"
                    + str(b.origin)
                    + "\n------------"
                )
            logger.warning(errinfo)
        return [b for b in blocks if not isinstance(b, ErrorBlock)]
