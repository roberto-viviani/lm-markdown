"""
Operations on markdown files to support LM markdown use. Here,
scan checks that that markdown is well-formed, adds a header if missing,
and returns a list of blocks with a header block first, or a list of
blocks with error blocks for problems.

Main functions:
    scan: general checks on blocklist, mainly header
    markdown_scan: checks on markdown file
"""

# protected members accessed
# pyright: reportPrivateUsage=false

from pathlib import Path
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

from lmm.utils.logging import LoggerBase, get_logger

logger: LoggerBase = get_logger(__name__)


def scan(blocks: list[Block]) -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first.

    Args:
        blocks: the list of blocks to process.

    Returns:
        the processed list of blocks.
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
                bl.content['title'] = "Title"
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
            metadata: MetadataDict = {'title': "Title"}
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
    logger: LoggerBase = logger,
) -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first.

    Args:
        sourcefile: the file to load the markdown from
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.
        logger: a logger object (defaults to console logging)

    Returns:
        the processed list of blocks.

    Note:
        if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    # Source validation
    source = iou.validate_file(sourcefile, logger)
    if not source:
        return []
    # For type-checking
    source = Path(source)

    # load_blocks is guaranteed to return an empty list or a list
    # of blocks.
    blocks = mkd.load_blocks(source, logger)
    if not blocks:  # Empty list check
        logger.warning(f"No blocks found in file: {source}")
        return []
    if mkd.blocklist_haserrors(blocks):
        logger.warning(f"Errors found while scanning {source}")

    # Validate first block and ensure first block is header,
    # creating one if necessary
    match blocks[0]:
        case HeaderBlock() | MetadataBlock() as bl:
            if (
                'title' not in bl.content
                or bl.content['title'] == "Title"
            ):
                bl.content['title'] = source.stem
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
            metadata: MetadataDict = {
                'title': source.stem,
                'comment': "",
            }
            blocks.insert(
                0,
                HeaderBlock(
                    content=metadata,
                    comment="**Default header added**",
                ),
            )
        case _:
            pass

    # call naked version
    blocks = scan(blocks)
    if not blocks:
        return []

    # Save and return
    match save:
        case False:
            pass
        case True:
            save_markdown(source, blocks, logger)
        case str() | Path():
            save_markdown(save, blocks, logger)
        case _:  # ignore
            pass

    return blocks


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface
    from lmm.markdown.parse_markdown import blocklist_haserrors

    def call_markdown_scan(filename: str, target: str) -> list[Block]:
        blocks = markdown_scan(filename, target)
        if not blocklist_haserrors(blocks):
            print("No errors in markdown.")
        return blocks

    create_interface(call_markdown_scan, sys.argv)
