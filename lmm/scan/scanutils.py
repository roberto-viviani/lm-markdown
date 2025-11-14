"""
Utilities for scan modules.

Important functions:
    preproc_for_markdown
    post_order_hashed_aggregation
"""

from typing import TypeGuard
from collections.abc import Callable
import re

from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    TextNode,
    post_order_traversal,
)
from lmm.utils.hash import base_hash
from lmm.utils.logging import LoggerBase, ConsoleLogger
from .scan_keys import TXTHASH_KEY, FREEZE_KEY, TITLES_TEMP_KEY


def preproc_for_markdown(response: str) -> str:
    # replace square brackets containing the character '\' to one
    # that is enclosed between '$$' for rendering in markdown
    response = re.sub(r"\\\[|\\\]", "$$", response)
    response = re.sub(r"\\\(|\\\)", "$", response)
    return response


def post_order_hashed_aggregation(
    root_node: MarkdownNode,
    aggregate_func: Callable[[str], str],
    output_key: str,
    hashed: bool = True,
    hash_key: str = TXTHASH_KEY,
    *,
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
    logger: LoggerBase = ConsoleLogger(),
) -> None:
    """
    Executes a post-order traversal on the markdown tree, with bottom-
    -up aggregation of the synthetic attributes in the parent nodes
    from the content data member of children text nodes. The synthetic
    attribute is computed by aggregate_func and recursively stored in
    the output_key field of the metadata member of the parent node.

    Note:
        this function differs from tree.extract_content in that a
        hash is computed to verify that the content was changed before
        calling the aggregate function.

    Note:
        aggregate_func is only called if there is content to
            aggregate.
        This avoids calls to llm's without content. aggregate_func
            itself may return empty for insufficient content.

    Args:
        root_node: The root node of the markdown tree
        aggregate_func: Function to process the collected content
            before storing. The collected content is provided as a
            string list. The function may return an empty string if
            there is no/not enough material to synthetise, leaving
            it for synthesis at the next level.
        output_key: the key in the metadata where the sythetised
            attributes should be stored
        hashed: if true, stores a hash of the content used for
            aggregation, and if  the content changes recomputes the
            aggregation. If false, the aggregation is computed only
            if the output key is missing from the metadata or its
            value is empty.
        hash_key: the key in the metadata where the hash ist read
            and stored.
        filter_func: a predicate function on the nodes to be
            aggregated. Only nodes where filter_func(node) is True
            will be aggregated.
        logger: a logger object.

    Note:
        If hashed is false, no re-computing of the output value takes
        place if there is already any. To recompute, use
        extract_content.

    Raises:
        ValueError: If validation fails for any of the following:
            - hashed is True and output_key equals hash_key
            - output_key is None or empty string
    """

    # this to inform type checker about assumption on node type
    def _is_heading_node(
        node: MarkdownNode,
    ) -> TypeGuard[HeadingNode]:
        return isinstance(node, HeadingNode)

    # this again for type checker, setting None to ""
    def _node_property(
        node: MarkdownNode, key: str, append: str = ""
    ) -> str:
        prpty: str | None = node.get_metadata_string_for_key(key, "")
        return (prpty + append) if prpty else ""

    # Validate output_key (treated as coding error)
    if not output_key or not output_key.strip():
        raise ValueError(
            "output_key must be a non-empty string. "
            f"Received: {repr(output_key)}"
        )

    # Validate that output_key and hash_key are different when
    # hashing is enabled (treated as coding error)
    if hashed and output_key == hash_key:
        raise ValueError(
            "output_key and hash_key cannot be the same when "
            f"hashed=True. Both are set to '{output_key}'. This "
            "would cause the hash value to overwrite the aggregated "
            "output."
        )

    if root_node.is_header_node() and not filter_func(root_node):
        logger.warning("Aggregation skipped for document")
        return

    delimiter: str = "\n\n"
    any_content_processed = False

    def _process_node(node: MarkdownNode) -> None:
        nonlocal any_content_processed
        # Skip leaf nodes (they don't have children to synthetise)
        if isinstance(node, TextNode):
            return

        if not _is_heading_node(node):
            # coding error, new node type
            raise ValueError(
                "Unreachable code reached: unexpected node type"
            )

        # do not compute aggregation if there is a parent node
        # with a "frozen" property to prevent updates
        if node.fetch_metadata_for_key(FREEZE_KEY, True, False):
            return

        # do not repeat aggregation if the node is a parent of just
        # one heading node, as the content will be the same
        if node.count_children() == 0:
            return
        if node.count_children() == 1 and isinstance(
            node.children[0], HeadingNode
        ):
            return

        # collect content from children (it is a heading node)
        collected_content: list[str] = []

        def _collect_text(node: MarkdownNode) -> None:
            # Recursively collects text from a node

            if not filter_func(node):
                return

            for child in node.children:

                if not filter_func(child):
                    continue

                if isinstance(child, TextNode):
                    # Collect content from direct TextBlock children
                    collected_content.append(child.get_content())
                else:
                    # Collect synthetic outputs from heading children
                    # that have them, and if not look in children
                    text: str | None = (
                        child.get_metadata_string_for_key(output_key)
                    )

                    if text:
                        collected_content.append(text)
                    else:  # recursion to headings down the tree
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
                new_hash = aggregate_hash(node, filter_func)
                if (
                    node.metadata
                    and output_key in node.metadata
                    and node.metadata[output_key]
                    and hash_key in node.metadata
                ):
                    if node.metadata[hash_key] == new_hash:
                        logger.info(
                            _node_property(
                                node,
                                TITLES_TEMP_KEY,
                                " skipped: text unchanged",
                            )
                        )
                        any_content_processed = True
                        return
            # If not hashed, check that output is already there
            else:
                if (
                    node.metadata
                    and output_key in node.metadata
                    and node.metadata[output_key]
                ):
                    any_content_processed = True
                    logger.info(
                        _node_property(
                            node,
                            TITLES_TEMP_KEY,
                            " skipped: " + output_key + " present",
                        )
                    )
                    return

            # the hash differs or the output is missing. we need to
            # recompute
            logger.info(
                "Aggregating " + _node_property(node, TITLES_TEMP_KEY)
            )
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
                # ignore: bound if hashed

            # Mark that we processed at least some content
            any_content_processed = True  # bool(collected_content)

    post_order_traversal(root_node, _process_node)

    # Warn if no content was processed (all nodes were filtered out,
    # or aggregate_func refused to compute aggregation)
    if not any_content_processed:
        heading_titles: str = _node_property(
            root_node, TITLES_TEMP_KEY, ": "
        )
        if root_node.is_root_node():
            logger.warning(
                heading_titles
                + "No aggregation was performed. This may indicate an "
                "overly restrictive filter, non-aggregable metadata, "
                "or an empty/small document.",
            )
        else:
            if len(root_node.get_text_children()) > 0:
                logger.warning(
                    heading_titles + "No aggregation was performed."
                )


def aggregate_hash(
    node: MarkdownNode,
    filter_func: Callable[[MarkdownNode], bool],
) -> str:
    """
    Create a hash from the text of the node, or of the descendants
    of the node. If the text is empty, an empty string is returned.

    Args:
        root_node: the node to compute the hash for
        filter_func a function to filter the nodes whose
            content should be hashed

    Returns:
        a string of 22 characters, or an empty string if there is
            no content in the tree.
    """

    if isinstance(node, TextNode):
        return (
            base_hash(node.get_content()) if filter_func(node) else ""
        )

    buffer: list[str] = []
    for child in node.children:
        if not filter_func(child):
            continue

        if isinstance(child, TextNode):
            buffer.append(child.get_content())
        else:
            buffer.append(aggregate_hash(child, filter_func))

    return base_hash("".join(buffer))
