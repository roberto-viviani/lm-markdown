"""
Operations on markdown files to support LM markdown use

Main functions:
    scan: general checks on blocklist, mainly header
    markdown_scan: checks on markdown file
"""

# protected members accessed
# pyright: reportPrivateUsage=false

from pathlib import Path
from typing import Callable
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
from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    TextNode,
    post_order_traversal,
)
from lmm.markdown.ioutils import save_markdown
from .scan_keys import TXTHASH_KEY

from lmm.utils.logging import ILogger, get_logger

logger: ILogger = get_logger(__name__)


def scan(blocks: list[Block]) -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first.

    Args:
        blocks: the list of blocks to process.

    Returns: the processed list of blocks.
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
    save: bool | str | Path = False,
    logger: ILogger = logger,
) -> list[Block]:
    """General check that the markdown is suitable for work,
    returning a list of blocks with a header block first.

    Args: source: the file to load the markdown from
          save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Returns: the processed list of blocks.

    Note: if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    # Source validation
    source = iou.validate_file(sourcefile)
    if not source:
        return []
    # For type-checking
    source = Path(source)

    # load_blocks is guaranteed to return an empty list or a list
    # of blocks.
    blocks = mkd.load_blocks(source)
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
            save_markdown(source, blocks)
        case str() | Path():
            save_markdown(save, blocks)
        case _:  # ignore
            pass

    return blocks


def post_order_aggregation(
    root_node: MarkdownNode,
    aggregate_func: Callable[[str], str],
    output_key: str,
    hashed: bool = True,
    hash_key: str = TXTHASH_KEY,
) -> None:
    """
    Executes a post-order traversal on the markdown tree, with bottom-
    -up aggregation of the synthetised attributes in the parent nodes
    from the content data member of children text nodes. The
    synthetised attribute is computed by aggregate_func and recursi-
    vely stored in the output_key field of the metadata member of the
    parent node.

    Note: this function differs from tree.extract_content in that a
    hash is computed to verify that the content was changed before
    calling the aggregate function.

    Args:
        root_node: The root node of the markdown tree
        aggregate_func: Function to process the collected content
            before storing. The collected content is provided as a
            string list. The function may return an empty string if
            there is no/not enough material to sythetise, leaving
            it for synthesis at the next level.
        output_key: the key in the metadata where the sythetised
            attributes should be stored
        hashed: if true, stores a hash of the content used for
            aggregation, and if  the content changes recomputes the
            aggregation. If false, the aggregation is computed only
            if the output key is missing from the metadata or its
            value is empty.
    """
    from lmm.utils.hash import base_hash

    delimiter: str = "\n\n"

    def _process_node(node: MarkdownNode) -> None:
        # Skip leaf nodes (they don't have children to synthetise)
        if isinstance(node, TextNode):
            return

        # do not repeat aggregation is the node is a parent of just
        # one parent node, as the content will be the same
        if isinstance(node, HeadingNode):
            if node.count_children() == 0:
                return
            if node.count_children() == 1 and isinstance(
                node.children[0], HeadingNode
            ):
                return

        # For parent nodes, collect content from children
        collected_content: list[str] = []

        def _collect_text(node: MarkdownNode) -> None:
            for child in node.children:
                if isinstance(child, TextNode):
                    # Collect content from direct TextBlock children
                    collected_content.append(child.get_content())
                else:
                    # Collect synth outputs from parent children that
                    # have them, and if not look in their children
                    text = (
                        child.metadata[output_key]
                        if output_key in child.metadata
                        else ""
                    )
                    if text:  # text is not empty
                        collected_content.append(
                            str(
                                child.get_metadata_for_key(
                                    output_key, ""
                                )
                            )
                        )
                    else:  # recursion to children down the tree
                        _collect_text(child)

        # start the recursion
        _collect_text(node)

        # If we collected any content, process it and store it in
        # metadata
        if collected_content:
            joined_content = delimiter.join(collected_content)

            # If there is the output, check that the joined content
            # corresponds to the hash
            if hashed:
                new_hash = base_hash(joined_content)
                if (
                    node.metadata
                    and output_key in node.metadata
                    and node.metadata[output_key]
                    and hash_key in node.metadata
                ):
                    if node.metadata[hash_key] == new_hash:
                        return
            # If there is the output, check it is not empty
            if (
                node.metadata
                and output_key in node.metadata
                and node.metadata[output_key]
            ):
                return

            # we need to recompute
            synth_content = aggregate_func(joined_content)
            if not synth_content:
                return

            # Initialize metadata dictionary if it doesn't exist
            if not node.metadata:
                node.metadata = {}

            # Store the synthesized property in metadata
            node.metadata[output_key] = synth_content
            if hashed:
                node.metadata[hash_key] = new_hash  # type: ignore
                # (bound if hashed)

    post_order_traversal(root_node, _process_node)
    # TODO: verify what happens to header
    # TODO: verify what happends to document without text


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
