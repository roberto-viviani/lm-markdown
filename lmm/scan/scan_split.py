"""Splits a blocklist using a splitter.

This implementation uses langchain to split the text of text
blocks. Metadata is inherited from the original block, except for
textid and UUID fields.

Note: metadata are not inherited from previous blocks or headings.
    In general, text blocks will have been populated with
    metadata prior to calling this function.

Main classes:
    NullTextSplitter: a splitter that does not split (scan_split
        becomes a no-opt)

Main functions:
    scan_split  take a blocklist and split the text blocks
"""

from pathlib import Path
from pydantic import validate_call

from .scan_keys import TEXTID_KEY, UUID_KEY

# lm markdown
from lmm.markdown.parse_markdown import (
    Block,
    TextBlock,
    MetadataBlock,
)
from lmm.markdown.parse_markdown import (
    load_blocks,
    save_blocks,
    blocklist_haserrors,
)


# langchain
from langchain_text_splitters import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from lmm.utils.logging import LoggerBase, get_logger

# Set up logger
logger = get_logger(__name__)


class NullTextSplitter(TextSplitter):
    """A langchain text splitter that does not split"""

    def split_text(self, text: str) -> list[str]:
        return [text]


defaultSplitter: TextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=False
)


def blocks_to_splitted_blocks(
    blocks: list[Block], text_splitter: TextSplitter
) -> list[Block]:
    """Transform a blocklist by applying text splitting to the text
    block prior to ingestion. Metadata are inherited from the original
    block.

    Args:
        blocks: a list of markdown blocks
        text_splitter: a langchain text splitter

    Returns:
        a list of markdown blocks
    """

    if isinstance(text_splitter, NullTextSplitter):
        return blocks

    def _split_text_block(bl: TextBlock) -> list[TextBlock]:
        doc: Document = Document(
            page_content=bl.get_content(), metadata={}
        )
        docs = text_splitter.split_documents([doc])
        return [TextBlock(content=d.page_content) for d in docs]

    # split
    newblocks: list[Block] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            splits = _split_text_block(b)
            if newblocks and isinstance(newblocks[-1], MetadataBlock):
                curmeta: MetadataBlock = newblocks[-1].deep_copy()
                newblocks.append(splits[0])
                # do not inherit textid's and UUID's
                curmeta.content.pop(TEXTID_KEY, "")
                curmeta.content.pop(UUID_KEY, "")
                for s in splits[1:]:
                    if curmeta.content:
                        newblocks.append(curmeta.deep_copy())
                    newblocks.append(s)
            else:
                newblocks.extend(splits)
        else:
            newblocks.append(b)
    return newblocks


def scan_split(
    blocks: list[Block], text_splitter: TextSplitter = defaultSplitter
) -> list[Block]:
    """Scan syntax for splitter

    Args:
        blocks: a list of markdown blocks
        text_splitter (opt): a langchain text splitter
            (defaults to a character text splitter, chunk size
            1000, overlap 200). To switch off splitting, use
            NullTextSplitter

    Returns:
        a list of markdown blocks
    """
    return blocks_to_splitted_blocks(blocks, text_splitter)


@validate_call(config={'arbitrary_types_allowed': True})
def markdown_split(
    sourcefile: str | Path,
    save: bool | str | Path = False,
    text_splitter: TextSplitter = defaultSplitter,
    logger: LoggerBase = logger,
) -> list[Block]:
    """Interface to apply split to documents (interactive use)

    Args:
        sourcefile: the file containing the markdown document to split
        save: a boolean value indicating whether the split document
            should be saved to disk
        text_splitter (opt): a langchain text splitter
            (defaults to a character text splitter, chunk size
            1000, overlap 200). To switch off splitting, use
            NullTextSplitter

    Note:
        if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    blocks = load_blocks(sourcefile, logger=logger)
    if not blocks:
        return []
    if blocklist_haserrors(blocks):
        save_blocks(sourcefile, blocks, logger)
        logger.warning("Problems in markdown, fix before continuing")
        return []

    blocks = scan_split(blocks, text_splitter)
    if not blocks:
        return []

    match save:
        case False:
            pass
        case True:
            save_blocks(sourcefile, blocks, logger)
        case str() | Path():
            save_blocks(save, blocks, logger)
        case _:  # ignore
            pass

    return blocks


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface

    def main_f(x, y):  # type: ignore[no-untyped-def]
        markdown_split(
            x, y, defaultSplitter  # type: ignore[no-untyped-def]
        )

    create_interface(main_f, sys.argv)  # type: ignore[no-untyped-def]
