"""
Utilities to read/write to/from disc and print errors to console.
Errors are not propagated, but functions return null value.
"""

from pathlib import Path
from .parse_markdown import Block, ErrorBlock
from .parse_markdown import serialize_blocks, blocklist_errors
from lmm.utils.ioutils import validate_file

# Set up default logger
from lmm.utils.logging import get_logger, ILogger  # fmt: skip
logger: ILogger = get_logger(__name__)


# Load markdown
def load_markdown(
    source: str | Path, logger: ILogger = logger
) -> str:
    # loads the markdown file. No error thrown, instead
    # message printed to console and empty string returned.

    # Check if the input is a Path object
    if isinstance(source, Path):
        if not validate_file(source):
            return ""
        try:
            content = source.read_text(encoding='utf-8')
        except IOError as e:
            logger.error(f"I/O error reading file {source}: {e}")
            return ""
    else:
        # Check if the string is a single line
        if '\n' not in source:
            # Treat it as a file path
            path = validate_file(source)
            if not path:
                return ""
            if path.is_file():  # type
                try:
                    content = path.read_text(encoding='utf-8')
                except IOError as e:
                    logger.error(
                        f"I/O error reading file {source}: " + f"{e}"
                    )
                    return ""
            else:
                logger.error(f"File not found: {source}")
                return ""
        else:
            # Treat it as raw content
            content = source

    return content


# Save markdown
def save_markdown(
    dest: str | Path,
    content: list[Block] | str,
    logger: ILogger = logger,
) -> bool:
    """Save markdown blocks to a file. Returns success or failure."""
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


# Print error blocks
def report_error_blocks(
    blocks: list[Block], logger: ILogger = logger
) -> list[Block]:
    """Checks the existence of error blocks. If there are any,
    they are printed to the console.
    Returns: A list without error blocks."""

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
                errinfo += "\n\n Offending content:\n------------\n" \
                    + str(b.origin) + "\n------------"
            logger.warning(errinfo)
        return [b for b in blocks if not isinstance(b, ErrorBlock)]
