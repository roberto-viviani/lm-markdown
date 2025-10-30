"""
Utilities for scan modules.

Important functions:
    post_order_hashed_aggregation
"""

from typing import Callable

from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    TextNode,
    post_order_traversal,
)
from .scan_keys import TXTHASH_KEY, FREEZE_KEY


def post_order_hashed_aggregation(
    root_node: MarkdownNode,
    aggregate_func: Callable[[str], str],
    output_key: str,
    hashed: bool = True,
    hash_key: str = TXTHASH_KEY,
    *,
    filter_func: Callable[[MarkdownNode], bool] = lambda x: True,
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
        aggregate_func is only called if there is content to aggregate.
        This avoids calls to llm's without content. aggregate_func iself
        may return empty for insufficient content.

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

    Note:
        If hashed is false, no re-computing of the output value takes
        place if there is already any. To recompute, use
        extract_content.
    """
    from lmm.utils.hash import base_hash

    delimiter: str = "\n\n"

    def _process_node(node: MarkdownNode) -> None:
        # Skip leaf nodes (they don't have children to synthetise)
        if isinstance(node, TextNode):
            return

        if not filter_func(node):
            return

        # do not repeat aggregation if the node is a parent of just
        # one parent node, as the content will be the same
        if isinstance(node, HeadingNode):
            if node.count_children() == 0:
                return
            if node.count_children() == 1 and isinstance(
                node.children[0], HeadingNode
            ):
                return

        # do not compute aggregation if there is a parent node
        # with a "frozen" property to prevent updates
        if node.fetch_metadata_for_key(FREEZE_KEY, True, False):
            return

        # For parent nodes, collect content from children
        collected_content: list[str] = []

        def _collect_text(node: MarkdownNode) -> None:
            for child in node.children:
                if not filter_func(child):
                    continue
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
                        if filter_func(child):
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
            # If not hashed, check that output is already there
            else:
                if (
                    node.metadata
                    and output_key in node.metadata
                    and node.metadata[output_key]
                ):
                    return

            # the hash differs or the output is missing. we need to
            # recompute
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
