"""
Operations on markdown files to support LM markdown use. Here,
scan checks that that markdown is well-formed, adds a header if missing,
and returns a list of blocks with a header block first, or a list of
blocks with error blocks for problems.

Main functions:
    scan: general checks on blocklist, mainly header
    markdown_scan: checks on markdown file (load)
    save_scan: saves markdown with timestamp verification

Behaviour:
    Functions in this module use the custom `LoggerBase` class from 
    the `lmm.utils.logging` package for error handling. The logger is 
    passed as the last argument to functions that require it. Errors 
    are logged rather than raised, except for validation errors in 
    `markdown_scan` and `save_scan`.
    
    File size limits: `markdown_scan` accepts `max_size_mb` (default 50.0) 
    and `warn_size_mb` (default 10.0) parameters. Files exceeding `warn_size_mb` 
    trigger a warning, while files exceeding `max_size_mb` will not be loaded 
    and an error is logged.
"""

# protected members accessed
# pyright: reportPrivateUsage=false

from pathlib import Path
from datetime import datetime
from pydantic import validate_call

import lmm.utils.ioutils as iou
from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
)
import lmm.markdown.parse_markdown as mkd
from lmm.markdown.parse_yaml import MetadataDict
from lmm.markdown.ioutils import save_markdown
from .scan_keys import LAST_MODIFIED_KEY

from lmm.utils.logging import LoggerBase, get_logger

logger: LoggerBase = get_logger(__name__)


def blocklist_scan(blocks: list[Block], default_title: str = "Title") -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first.

    Args:
        blocks: the list of blocks to process.
        default_title: the default title to use when no title is found
            or when the title is "Title".

    Returns:
        the processed list of blocks. If the input contains only
        ErrorBlocks, returns the list as-is without adding a header,
        as ErrorBlocks signal that the block list is not valid.

    Examples:
        ```
        >>> from lmm.markdown.parse_markdown import parse_markdown_text
        >>> blocks = parse_markdown_text("# My Document\\n\\nSome text")
        >>> result = blocklist_scan(blocks)
        >>> isinstance(result[0], HeaderBlock)
        True
        >>> result[0].content['title']
        'My Document'
        ```
    """

    if not blocks:  # Empty list
        return [HeaderBlock.from_default()]

    # Validate first block and ensure first block is header,
    # creating one if necessary
    match blocks[0]:
        case HeaderBlock() | MetadataBlock() as bl:
            if (
                'title' not in bl.content
                or bl.content['title'] == "Title"
            ):
                bl.content['title'] = default_title
                if not bl.comment:
                    bl.comment = "**Default title added**"
            # replace first with header
            blocks[0] = HeaderBlock._from_metadata_block(bl)
        case HeadingBlock() as bl:
            metadata: MetadataDict = {'title': bl.content}
            blocks.insert(
                0,
                HeaderBlock(
                    content=metadata,
                    comment="**Default header added**",
                ),
            )
        case TextBlock():
            metadata: MetadataDict = {'title': default_title}
            blocks.insert(
                0,
                HeaderBlock(
                    content=metadata,
                    comment="**Default header added**",
                ),
            )
        case ErrorBlock():
            pass

    return blocks


@validate_call(config={'arbitrary_types_allowed': True})
def markdown_scan(
    sourcefile: str | Path,
    save: bool | str | Path = True,
    *,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first. When
    a title is missing, uses the filename stem as the default title.

    Args:
        sourcefile: the file to load the markdown from
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.
        max_size_mb: the max size, in MB, of the file to load
        warn_size_mb: the size of the input file that results in
            a warning
        logger: a logger object (defaults to console logging)

    Returns:
        the processed list of blocks.

    Note:
        if an error occurs and the blocklist becomes empty,
        it does not alter the source file.

    Examples:
        ```python
        # Scan a markdown file and save changes. Timestamp added
        blocks = markdown_scan("document.md", save=True)
        
        # Scan without saving
        blocks = markdown_scan("document.md", save=False)
        
        # Scan and save to different file, timestamp added
        blocks = markdown_scan("source.md", save="output.md")
        ```
    """

    # Source validation
    source: Path | None = iou.validate_file(sourcefile, logger)
    if not source:
        return []
    # For type-checking
    source = Path(source)

    # load_blocks is guaranteed to return an empty list or a list
    # of blocks.
    blocks: list[Block] = mkd.load_blocks(
        source,
        max_size_mb=max_size_mb,
        warn_size_mb=warn_size_mb,
        logger=logger,
    )
    if not blocks:  # Empty list check
        logger.warning(f"No blocks found in file: {source}")
        return []
    if mkd.blocklist_haserrors(blocks):
        logger.warning(f"Errors found while scanning {source}")

    # Use blocklist_scan with filename stem as default title
    # This ensures missing titles are replaced with the filename
    blocks = blocklist_scan(blocks, default_title=source.stem)
    if not blocks:
        return []

    # Save and return
    match save:
        case False:
            pass
        case True:
            save_scan(source, blocks, logger=logger)
        case str() | Path():
            save_markdown(save, blocks, logger=logger)
        case _:  # ignore
            pass

    return blocks


@validate_call(config={'arbitrary_types_allowed': True})
def save_scan(
    destfile: str | Path,
    blocks: list[Block],
    *,
    verify_unchanged: bool = True,
    logger: LoggerBase = logger,
) -> bool:
    """
    Save blocks to markdown file with optional timestamp verification.
    
    This function provides a safe save mechanism that can verify the file
    hasn't been modified since it was loaded, preventing accidental overwrites
    of concurrent changes. A timestamp is stored in the header metadata block
    using the key '~last_modified'.
    
    Args:
        destfile: Destination file path (string or Path object)
        blocks: List of Block objects to save (must have HeaderBlock first)
        verify_unchanged: If True, check timestamp to verify file hasn't 
            changed since load. Defaults to True for safety.
        logger: Logger object for error reporting
    
    Returns:
        True if saved successfully, False otherwise
    
    Examples:
        ```
        >>> # Basic save to new file
        >>> from lmm.scan.scan import markdown_scan, save_scan
        >>> blocks = markdown_scan("test.md", save=False)
        >>> save_scan("output.md", blocks)
        True
        
        >>> # Load, modify, and save with verification
        >>> blocks = markdown_scan("test.md", save=False)
        >>> blocks[0].content['author'] = 'New Author'
        >>> save_scan("test.md", blocks, verify_unchanged=True)
        True
        
        >>> # Force save without verification
        >>> save_scan("test.md", blocks, verify_unchanged=False)
        True
        ```
    
    Note:
        - The timestamp is stored in blocks[0].content['~last_modified']
        - If verify_unchanged=True and timestamps don't match, returns False
        - Missing timestamps are handled gracefully (first save or legacy file)
        - Errors are logged through the logger object
    """
    # Validate inputs
    if not blocks:
        logger.error("Cannot save empty block list")
        return False
    
    if not isinstance(blocks[0], HeaderBlock):
        logger.error("First block must be a HeaderBlock")
        return False
    
    # Convert to Path
    dest_path = Path(destfile)
    
    # Verify timestamp if requested and file exists
    if verify_unchanged and dest_path.exists():
        try:
            # Load existing file to check timestamp
            existing_blocks = markdown_scan(dest_path, save=False, logger=logger)
            
            if not existing_blocks:
                logger.warning(
                    f"Could not load existing file {dest_path} for "
                    "timestamp verification, proceeding anyway"
                )
            else:
                # Get timestamps
                existing_timestamp: str = existing_blocks[0].content.get(LAST_MODIFIED_KEY) # type: ignore
                current_timestamp: str = blocks[0].content.get(LAST_MODIFIED_KEY) # type: ignore
                
                # Compare timestamps
                if existing_timestamp and current_timestamp:
                    if existing_timestamp != current_timestamp:
                        logger.warning(
                            f"File {dest_path} has been modified since load. "
                            f"Expected timestamp: {current_timestamp}, "
                            f"found: {existing_timestamp}. Save aborted."
                        )
                        return False
                elif existing_timestamp and not current_timestamp:
                    logger.info(
                        f"Blocks to save have no timestamp, but file {dest_path} "
                        "does. This may indicate the blocks were not loaded via "
                        "save_scan. Proceeding with save."
                    )
                # If neither has timestamp or only current has one, proceed
                
        except Exception as e:
            logger.error(f"Error during timestamp verification: {e}")
            return False
    
    # Update timestamp with current time
    blocks[0].content[LAST_MODIFIED_KEY] = datetime.now().isoformat()
    
    # Save using save_markdown
    success = save_markdown(dest_path, blocks, logger)
    
    if success:
        logger.info(f"Successfully saved {dest_path}")
    
    return success


def scan(
    sourcefile: str | Path,
    save: bool | str | Path = True,
    *,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> None:
    """General check that the markdown is suitable for work.
    This is a wrapper around markdown_scan that catches exceptions and 
    logs them, suitable for command-line interface use.

    Args:
        sourcefile: the file to load the markdown from
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.
        max_size_mb: the max size, in MB, of the file to load
        warn_size_mb: the size of the input file that results in
            a warning
        logger: a logger object (defaults to console logging)

    Returns:
        None. Errors are logged instead of raised.
    """

    try:
        markdown_scan(
            sourcefile,
            save,
            max_size_mb=max_size_mb,
            warn_size_mb=warn_size_mb,
            logger=logger,
        )
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Error scanning {sourcefile}: {e}")


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface
    from lmm.markdown.parse_markdown import blocklist_haserrors

    def call_markdown_scan(filename: str, target: str) -> list[Block]:
        blocks: list[Block] = markdown_scan(filename, target)
        if not blocklist_haserrors(blocks):
            print("No errors in markdown.")
        return blocks

    create_interface(call_markdown_scan, sys.argv)
