"""
Utilities for scan modules.

Main functions:
    - preproc_for_markdown
    - post_order_hashed_aggregation

Behaviour:
    Exported functions in this module generally raise `ValueError` for invalid
    arguments or internal state errors. They also accept a `LoggerBase` object
    to log warnings and information about the aggregation process.
"""

from typing import TypeGuard
from collections.abc import Callable
import re

from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    post_order_traversal,
)
from lmm.utils.hash import base_hash
from lmm.utils.logging import LoggerBase, ConsoleLogger
from .scan_keys import TXTHASH_KEY, FREEZE_KEY, TITLES_TEMP_KEY


def preproc_for_markdown(response: str) -> str:
    """
    Pre-processes a string for markdown rendering, specifically
    handling LaTeX-style delimiters.

    Args:
        response: The string to be processed.

    Returns:
        The processed string with updated delimiters.
    """
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

    This function differs from tree.extract_content in that a
    hash is computed to verify that the content was changed before
    calling the aggregate function.

    Note:
        aggregate_func is only called if there is content to
        aggregate. This avoids calls to llm's without content. In
        addition, aggregate_func itself may autonomously return empty
        for insufficient content.

        If a heading child lacks a synthetic attribute as a result of
        this, the aggregation algorithm will descend into that child's
        subtree to find text to give more material to aggregate_func.

    Content collection strategy:
        Parent nodes collect content from their children as follows:
        - From text children: the raw text content is collected.
        - From heading children: if the child has a synthetic output
          (output_key in metadata), that output is collected. If not,
          the algorithm recurses into the child's subtree to collect
          raw text from deeper levels.
        This means that parent aggregation operates on children's
        synthetic outputs, not their raw text.

    Single-heading-child optimisation:
        When a non-root heading node has exactly one child which is
        also a heading node, the child's synthetic output is copied
        to the parent instead of calling aggregate_func (since the
        result would be identical). This copy cascades correctly
        in chains (H1->H2->H3): post-order processes H3 first,
        then H2 copies from H3, then H1 copies from H2. When a
        child is later added (no longer only-child), the node
        enters the normal aggregation path; the old copied output
        is invalidated by hash mismatch (hashed=True) or
        overwritten (hashed=False with output_key deleted).

    Manual edits:
        Manual edits to synthetic properties are overwritten on
        recomputation. Use ``frozen: true`` in metadata to preserve
        them.

    Args:
        root_node: The root node of the markdown tree
        aggregate_func: Function to process the collected content
            before storing. The collected content is provided as a
            string. The function may return an empty string if
            there is no/not enough material to synthetise, leaving
            it for synthesis at the next level. This implies that
            at the next level text will be recursively collected
            from all children nodes to attempt to compute the
            synthetic attribute.
        output_key: the key in the metadata where the synthetised
            attributes should be stored
        hashed: if true, stores a hash of the content used for
            aggregation, and if the content changes recomputes the
            aggregation. If false, the aggregation is computed only
            if the output key is missing from the metadata or its
            value is empty (see summary below)
        hash_key: the key in the metadata where the hash is read
            and stored.
        filter_func: a predicate function on the nodes to be
            aggregated. Only nodes where filter_func(node) is True
            will be aggregated. This means that nodes excluded by
            the filter_func will be excluded for both aggregation
            and production of synthetic attributes (the branch is
            completely pruned)
        logger: a logger object.

    Behaviour under different conditions

    `hashed = True` (default)
    - Computes a hash of the content of text nodes under each heading,
        ignoring synthetic outputs.
    - If the node already has both `output_key` and `hash_key` in
        metadata, and the stored hash matches the newly-computed hash,
        no new synthetic property is recomputed.
    - If hash differs, or `output_key` is missing, or `hash_key` is
        missing, recomputes the synthetic property and stores it in
        the metadata together with the new hash.
    Hence, when `hashed = True`, changes to raw text trigger
    recomputation, while changes to synthetic outputs of children do
    not (allowing manual editing of synthetic properties).

    `hashed = False`
    - If the node already has `output_key` in metadata with a truthy
        value, no recomputation takes place (the old property is
        retained).
    - If `output_key` is missing, the synthetic property is computed
        and stored in the metadata.
    - No hash is ever stored or checked.
    This is a "compute once" mode. To force recomputation, one must
    delete the `output_key` from the node's metadata manually, or
    use the `extract_content` function.

    `frozen: true` in metadata
    If a node has a `frozen` property set to true, no aggregation
    will take place on that node and all its descendants. This
    means that the aggregation process itself is frozen.

    Behaviour:
        Raises ValueError: If validation fails for any of the
        following:
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
    output_key = output_key.strip()

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
        if node.is_text_node():
            return

        if not _is_heading_node(node):
            # this does not defend against coding errors, it
            # just satisfies type checker
            raise ValueError(
                "Unreachable code reached: unexpected node type"
            )

        # do not compute aggregation if there is a parent node
        # with a "frozen" property to prevent updates
        if node.fetch_metadata_for_key(FREEZE_KEY, True, False):
            logger.info("Skipped (frozen)")
            return

        # no children: nothing to aggregate
        if node.count_children() == 0:
            return
        # single-heading-child optimisation: copy the child's
        # synthetic output instead of re-aggregating identical
        # content (root is exempt so it always aggregates)
        if (
            node != root_node
            and node.count_children() == 1
            and isinstance(node.children[0], HeadingNode)
        ):
            child = node.children[0]
            child_output = child.get_metadata_string_for_key(
                output_key, ""
            )
            if child_output:
                if not node.metadata:
                    node.metadata = {}
                node.metadata[output_key] = child_output
                if hashed:
                    node.metadata[hash_key] = (
                        aggregate_hash(node, filter_func)
                    )
                any_content_processed = True
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

                if child.is_text_node():
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
                            f" skipped: {output_key} present",
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
            any_content_processed = True

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
        node: the node to compute the hash for
        filter_func: a function to filter the nodes whose
            content should be hashed

    Returns:
        a string of 22 characters, or an empty string if there is
            no content in the tree.
    """

    if node.is_text_node():
        return (
            base_hash(node.get_content()) if filter_func(node) else ""
        )

    buffer: list[str] = []
    for child in node.children:
        if not filter_func(child):
            continue

        if child.is_text_node():
            buffer.append(child.get_content())
        else:
            buffer.append(aggregate_hash(child, filter_func))

    return base_hash("".join(buffer))
