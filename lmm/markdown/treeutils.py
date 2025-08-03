"""
Utility functions to extract information from markdown trees.

The functions may be divided in the following categories.

Aggregate: exchange information between child and parent nodes. These
functions revise the tree and have side effects on the tree.
    aggregate_content_in_parent_metadata
    inherit_metadata
    extract_property

Traversal: traverse the tree and extract information, allowing
inheritance of metadata directly of through a function. No side
effects.
    collect_text
    collect_headings
    collect_dictionaries
    collect_table_of_contents
    collect_textblocks

Fold: no side effects
    count_words

Select nodes: select nodes from tree
    get_nodes, get_text_nodes, get_heading_nodes
    get_nodes_with_metadata

"""

from typing import Any, Callable, cast, TypeVar

from .tree import (
    MarkdownNode,
    TextNode,
    HeadingNode,
    MarkdownTree,
    pre_order_traversal,
    post_order_traversal,
    traverse_tree,
    traverse_tree_nodetype,
    fold_tree,
)
from .parse_markdown import (
    Block,
    MetadataBlock,
    TextBlock,
)


# aggregation and inheritance---------------------------------------


def aggregate_content_in_parent_metadata(
    root_node: MarkdownNode,
    key: str,
    summarize_func: Callable[[str], str],
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> MarkdownNode:
    """
    Collects content from leaf node children and outputs from non-leaf
    children, processes them with a processing function, and stores
    the result in the parent's metadata in key.

    This function uses post-order traversal to ensure that child
    summaries are computed before parent summaries, enabling
    progressive accumulation of text content up the tree.

    Args:
        root_node: The root node of the markdown tree
        key: The metadata key of the heading nodes where to store the
            aggregate content, and read from to compute aggregation
        summarize_func: Function to process the collected content
            before storing
        filter_func: a filter function to select the nodes that
            should be processed
    """

    def process_node(node: MarkdownNode) -> None:
        # Skip leaf nodes (they don't have children to summarize)
        if not (isinstance(node, HeadingNode) and filter_func(node)):
            return

        # For parent nodes, collect content from children
        collected_content: list[str] = []

        for child in node.children:
            if child.is_text_node():
                # Collect content from direct TextBlock children
                collected_content.append(child.get_content())
            elif child.is_parent_node() and key in child.metadata:
                # Collect summaries from parent children
                collected_content.append(child.metadata[key])

        # If we collected any content, process it and store in
        # metadata
        if collected_content:
            joined_content = "\n\n".join(collected_content)
            summary = summarize_func(joined_content)

            # Initialize metadata dictionary if it doesn't exist
            if not node.metadata:
                node.metadata = {}

            # Store the summary in metadata
            node.metadata[key] = summary

    # ensure children are processed before parents
    post_order_traversal(root_node, process_node)
    return root_node


def inherit_metadata(
    node: MarkdownNode,
    exclude: list[str],
    inherit: bool = True,
    include_header: bool = False,
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> MarkdownNode:
    """Copy the metadata of headings into those of children nodes. If
    the metadata property is already defined on the child, no copy is
    made.

    Args:
        node: the root node of the branch to work on
        exclude: a list of keys that are not inherited.
        inherit: inheritance goes up the hierarchy to the first
            heading with metadata
        include_header: if inherit, whether to consider the header
            for inheritance
        filter_func: a filter for the children nodes to process.

    Returns:
        the modified branch

    Note: when inherit is true, the inherit search stops at the first
        parent heading that has any metadata.
    """

    def _meta_from_parent(n: MarkdownNode) -> None:
        if not filter_func(n):
            return

        meta = n.get_metadata()  # get_metadata() delivers copy
        parent = n.parent
        if parent:
            if inherit:
                parentmeta = parent.fetch_metadata(
                    None, include_header
                )
            else:
                parentmeta = parent.get_metadata()
            for k in parentmeta.keys():
                if k in exclude:
                    continue
                if k not in meta:
                    meta[k] = parentmeta[k]
        n.metadata = meta

    pre_order_traversal(node, _meta_from_parent)
    return node


def extract_property(
    node: MarkdownNode,
    key: str,
    add_type: bool = True,
    select: bool = False,
) -> MarkdownNode:
    """Extract a property from the metadata of a heading and transform
    it into a child text node with content given by that property.

    Args:
        node: the root or branch node to work on
        key: the property to be moved into a text node
        add_type: if True, the metadata of the added text child node
            will have a 'type' property with the value of the
            transferred property.
        select: if True, replaces all children of the heading node
            with the new node containing the property of the metadata.
            If the heading node has no such property, then all text
            children of the heading node are removed.

    Returns: the root node of the modified branch

    NOTE: This function changes the structure of the tree
    """

    def _extract_property(node: MarkdownNode):
        if isinstance(node, HeadingNode):
            if select:
                if key in node.metadata:
                    text = TextNode(
                        TextBlock(content=node.metadata.pop(key)),
                        parent=node,
                    )
                    if add_type:
                        text.metadata["type"] = key

                    heading_nodes = node.get_heading_children()
                    newchildren: list[MarkdownNode] = [text]
                    newchildren.extend(heading_nodes)
                    node.children = newchildren
                else:
                    node.children = cast(
                        list[MarkdownNode],
                        node.get_heading_children(),
                    )
            else:
                if key in node.metadata:
                    text = TextNode(
                        TextBlock(content=node.metadata.pop(key)),
                        parent=node,
                    )
                    if add_type:
                        text.metadata["type"] = key

                    node.children.insert(0, text)

    # process children before parents
    post_order_traversal(node, _extract_property)

    return node


# traverse ---------------------------------------------------------


def collect_text(
    root: MarkdownTree,
    sep: str = "\n\n",
    filter_func: Callable[[TextNode], bool] = lambda _: True,
) -> str:
    """
    Join all text in the text descendants of the node.

    Args:
        root: The root node of the tree, or a heading node
        sep: a separator string
        filter_func: an optional filter on the node

    Returns:
        The accumulated text.
    """
    import re

    if not root:
        return ""

    mapf: Callable[[TextNode], str] = lambda x: x.get_content()
    content = sep.join(
        traverse_tree_nodetype(root, mapf, TextNode, filter_func)
    )
    return re.sub(rf"{sep}({sep})+", sep, content).strip()


def collect_headings(
    root: MarkdownTree,
    sep: str = "\n\n",
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> str:
    """
    Join all heading text of the node and its descendants.

    Args:
        root: The root node of the tree, or a heading node
        sep: a separator string
        filter_func: an optional filter on the node

    Returns:
        The accumulated text.
    """

    if not root:
        return ""

    mapf: Callable[[HeadingNode], str] = lambda x: x.get_content()
    content = sep.join(
        traverse_tree_nodetype(root, mapf, HeadingNode, filter_func)
    )
    return content.strip()


def collect_dictionaries(
    root: MarkdownTree,
    map_func: Callable[
        [MarkdownNode], dict[str, Any]
    ] = lambda x: x.as_dict(True, False),
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> list[dict[str, Any]]:
    """Unfold a tree or branch into a list of dictionaries,
    containing the node content and its metadata, or a selection of
    metadata as specified by map_func, optionally filtered by filter
    func.

    Args:
        root: the root node of the tree or branch
        map_func (opt): a function mapping a MarkdownNode to a
            dictionary with text content and metadata fields. The
            default copies the content and the metadata of the node,
            inheriting from parents until a metadata is found,
            but excluding the header.
        filter_func (opt): a function that filters the nodes to which
            map_func is applied.

    Returns: a list of dictionaries with key 'content' (the text) and
        'metadata', or with the keys specified by map_func

    Note: Use map_func to collect information from the parents of
        the node hierarchically.
    """
    if not root:
        return []

    return traverse_tree(root.tree_copy(), map_func, filter_func)


def collect_table_of_contents(
    root: MarkdownNode,
) -> list[dict[str, int | str]]:
    """
    A specialized collect function to extract a table of contents
    from the markdown tree.

    Args:
        root: The root node of the tree

    Returns:
        A list of dictionaries representing the table of contents
    """

    def collect_headings(node: HeadingNode) -> dict[str, int | str]:
        return {
            'level': node.heading_level(),
            'content': node.get_content(),
        }

    return traverse_tree_nodetype(root, collect_headings, HeadingNode)


def collect_textblocks(
    root: MarkdownTree,
    inherit: bool = True,
    include_header: bool = False,
    filter_func: Callable[[TextNode], bool] = lambda _: True,
) -> list[Block]:
    """Unfold the tree to a block list, replacing headings with
    metadata blocks, such that text blocks are annotated by the
    metadata of the parent heading. If inherit is true, inherited
    metadata are sought up to the first parent heading with parent
    heading with metadata. If include_header is true, the header
    is considered as a source of metadata.

    Args:
        root: the tree
        inherit: metadata keys are inherited from parent if True
        include_header: if to inherit from header
        filter_func: a predicate to filter text nodes to include
            in the final block list.

    Returns: a block list

    Note: the metadata_block members, corresponding to the
        _private member of MetadataBlocks, are ignored.
    """

    if not root:
        return []

    dicts: list[dict[str, Any]] = traverse_tree_nodetype(
        root.tree_copy(),
        lambda x: x.as_dict(inherit, include_header),
        TextNode,
        filter_func,
    )
    blocks: list[Block] = []
    for d in dicts:
        if 'metadata' in d:
            blocks.append(MetadataBlock._from_dict(d['metadata']))  # type: ignore
        blocks.append(TextBlock(content=d['content']))
    return blocks


# fold ------------------------------------------------------------


def count_words(root: MarkdownTree) -> int:
    """
    Count the total number of words in the tree representation
     of the Markdown document.

    Args:
        root: The root node of the tree

    Returns:
        The total number of words in the document
    """

    if not root:
        return 0

    def count_words_in_node(node: MarkdownNode) -> int:
        if isinstance(node.block, TextBlock):
            return len(node.get_content().split())
        else:
            return 0

    return fold_tree(
        root, lambda acc, node: acc + count_words_in_node(node), 0
    )


# select nodes -----------------------------------------------------


from enum import Enum  # fmt: skip
class CopyOpts(Enum):
    NAKED_COPY = 2  # copy node and take it off tree
    NODE_COPY = 1  # copy node and keep links
    REFERENCE = 0  # reference to node (python way)


MN = TypeVar("MN", bound=MarkdownNode)  # fmt: skip
def get_nodes(
    root: MarkdownNode,
    opts: CopyOpts = CopyOpts.NAKED_COPY,
    node_type: type[MN] = MarkdownNode,
    filter_func: Callable[[MN], bool] = lambda _: True,
) -> list[MN]:
    """
    Find all nodes of node_type, or all such nodes that satisfy a
    predicate function

    Args:
        root: The root node of the tree
        opts: a CopyOpts value
        node_type: the type of node to select (default to all)
        filter_func: a predicate to select the nodes (defaults
            to all nodes)

    Returns:
        A list of the text nodes, or of those where filter_func is
        true

    See also: get_text_nodes, get_heading_nodes,
        get_nodes_with_metadata for examples of uses
    """
    f: Callable[[MN], MN]
    match opts:
        case CopyOpts.NAKED_COPY:
            f = lambda x: x.naked_copy()  # noqa
        case CopyOpts.NODE_COPY:
            f = lambda x: x.node_copy()  # noqa
        case CopyOpts.REFERENCE:
            f = lambda x: x  # noqa
        case _:
            raise RuntimeError(
                "Unreachable code reached due to "
                + "unrecognized CopyOpts enum value"
            )

    nodes: list[MN] = traverse_tree_nodetype(
        root, f, node_type, filter_func
    )
    return nodes


def get_nodes_with_metadata(
    root: MarkdownNode,
    metadata_key: str,
    node_type: type[MN] = MarkdownNode,
    opts: CopyOpts = CopyOpts.NAKED_COPY,
) -> list[MN]:
    """
    Find all nodes that have a specific metadata key.

    Args:
        root: The root node of the tree
        metadata_key: The metadata key to search for
        node_type: the node type, defaults to MarkdownNode
        opts: a CopyOpts value, defaults to NAKED_COPY

    Returns:
        A list of nodes that have the specified metadata key
    """

    return get_nodes(
        root,
        opts,
        node_type,
        lambda x: bool(x.metadata) and metadata_key in x.metadata,
    )


def get_textnodes(
    root: MarkdownNode,
    opts: CopyOpts = CopyOpts.NAKED_COPY,
    filter_func: Callable[[TextNode], bool] = lambda _: True,
) -> list[TextNode]:
    """
    Find all text nodes, or all text nodes that satisfy a predicate
    function

    Args:
        root: The root node of the tree
        opts: a CopyOpts value
        filter_func: a predicate to select the text nodes

    Returns:
        A list of the text nodes, or of those where filter_func is
        true
    """

    return get_nodes(root, opts, TextNode, filter_func)


def get_headingnodes(
    root: MarkdownNode,
    opts: CopyOpts = CopyOpts.NAKED_COPY,
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> list[HeadingNode]:
    """
    Find all heading nodes, or all heading nodes that satisfy a
    predicate function

    Args:
        root: The root node of the tree
        opts: a CopyOpts value
        filter_func: a predicate to select the heading nodes

    Returns:
        A list of the heading nodes, or of those where filter_func is
        true
    """

    return get_nodes(root, opts, HeadingNode, filter_func)
