"""
Utility functions to extract information from markdown trees.

The functions may be divided in the following categories.

Aggregate:
    exchange information between child and parent nodes. These
    functions revise the tree and have side effects on the tree.
    `summarize_content`
    `inherit_metadata`
    `extract_property`

Traversal:
    traverse the tree and extract information, allowing
    inheritance of metadata directly or through a function. No side
    effects.
    `collect_text`
    `collect_headings`
    `collect_dictionaries`
    `collect_table_of_contents`
    `collect_annotated_textblocks`

Fold:
    `count_words`

Select nodes:
    select nodes from tree
    `get_nodes`, `get_text_nodes`, `get_heading_nodes`
    `get_nodes_with_metadata`

"""

# protected members of Block used
# pyright: reportPrivateUsage=false

from typing import TypeVar
from collections.abc import Callable

from .tree import (
    MarkdownNode,
    TextNode,
    HeadingNode,
    MarkdownTree,
    NodeDict,
    pre_order_traversal,
    post_order_traversal,
    traverse_tree,
    traverse_tree_nodetype,
    fold_tree,
    propagate_content,
)
from .parse_markdown import (
    Block,
    MetadataBlock,
    TextBlock,
)


# inheritance------------------------------------------------------

def inherit_parent_properties(
    node: MarkdownNode,
    properties: list[str],
    destination_names: list[str] | None,
    include_header: bool = False,
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> MarkdownNode:
    """ Copy specified metadata properties from parent to the meta-
    data of its immediate children. 
    
    Args:
        node: the root node of the branch to work on
        properties: a list of metadata properties to copy
        destination_names: the names of the keys to copy the 
            properties into. If None, the same key names of the 
            parent are used
        include_header: also inherit properties of header
        filter_func: a filter for the children nodes to process.

    Behaviour:
        raises ValueError if destination_names is not Null and is
        not of the same length as properties.

    Note:
        Properties are inherited from the immediate parent only. This
        function does not propagate properties recursively from
        grandparents or higher ancestors in a single pass.
    """
    if destination_names is not None:
        if not len(destination_names) == len(properties):
            raise ValueError(
                f"inherit_parent_property: "
                "destination_names is of length "
                f"{len(destination_names)}, but properties"
                f" is of length {len(properties)}"
            )
    else:
        destination_names = properties

    def _add_parent_summary(n: MarkdownNode) -> None:
        if not filter_func(n):
            return
        if n.parent:
            for p, d in zip(properties, destination_names):
                property: str | None = n.parent.get_metadata_string_for_key(p)
                if property:
                    n.set_metadata_for_key(d, property)

    post_order_traversal(node, _add_parent_summary)
    return node


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

    Note:
        when inherit is true, the inherit search stops at the first
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


def bequeath_properties(
    node: HeadingNode,
    keys: list[str],
    new_keys: list[str] = [],
    inherit: bool = False,
    include_header: bool = False,
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> HeadingNode:
    """Extract a property from the metadata of a heading node and
    bequeeth it into the child text nodes.

    Args:
        node: the root or branch node to work on
        keys: the properties to be given to the text nodes
        new_keys: the new key names
        inherit: keys are bequeethed to grandchildren down
            the tree.
        include_header: includer header node in the pool of
            nodes that bequeath properties to children
        filter_func: a predicate to filter the heading nodes
            that bequeath properties.

    Returns:
        the root node of the modified branch

    Note:
        This function differs from inherit_metadata in that here
        the keys that are inherited are specified, as well as the
        new names they take.
    """
    from lmm.markdown.tree import post_order_traversal
    from lmm.markdown.parse_yaml import MetadataValue

    if not (new_keys):
        new_keys = keys

    def process_node(n: HeadingNode) -> HeadingNode:
        for child in n.get_text_children():
            for k, nk in zip(keys, new_keys):
                value: MetadataValue = (
                    child.fetch_metadata_for_key(k, include_header)
                    if inherit
                    else child.get_metadata_for_key(k)
                )
                if value:
                    child.set_metadata_for_key(nk, value)
        return n

    traverse_tree_nodetype(
        node,
        process_node,
        HeadingNode,
        filter_func,
        post_order_traversal,
    )
    return node


def propagate_property(
    node: HeadingNode,
    key: str,
    *,
    inherited_keys: list[str] = [],
    add_key_info: bool = True,
    select: bool = False,
) -> HeadingNode:
    """Extract a property from the metadata of a heading node and
    transform it into a child text node with content given by that
    property.

    Args:
        node: the root or branch node to work on
        key: the property to be moved into a text node
        inherited_keys: the keys that the new text node inherits
        add_key_info: if True, the metadata of the added text child
            node will have a 'type' property with the value of the
            transferred property.
        select: if True, replaces all children of the heading node
            with the new node containing the property of the metadata.
            If the heading node has no such property, then the text
            children are not altered.

    Returns:
        the root node of the modified branch

    Note:
        This function changes the structure of the tree.
    """

    def process_node(n: HeadingNode) -> TextNode:
        node: TextNode = TextNode.from_content(
            content=str(n.metadata.pop(key)),
            metadata={'type': key} if add_key_info else {},
        )
        for k in inherited_keys:
            if n.has_metadata_key(k):
                node.set_metadata_for_key(
                    k, n.get_metadata_for_key(k)
                )
        return node

    return propagate_content(
        node, process_node, select, lambda n: key in n.metadata.keys()
    )


# traverse ---------------------------------------------------------


def collect_text(
    root: MarkdownTree,
    filter_func: Callable[[TextNode], bool] = lambda _: True,
) -> list[str]:
    """
    Collect all text in the text node descendants of the node.

    Args:
        root: The root node of the tree, or a heading node
        filter_func: an optional filter on the node

    Returns:
        A list containing the accumulated text.
    """

    if not root:
        return []

    mapf: Callable[[TextNode], str] = lambda x: x.get_content()
    return traverse_tree_nodetype(root, mapf, TextNode, filter_func)


def collect_headings(
    root: MarkdownTree,
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> list[str]:
    """
    Collect all heading text of the node and its descendants.

    Args:
        root: The root node of the tree, or a heading node
        filter_func: an optional filter on the node

    Returns:
        A list of the accumulated text.
    """

    if not root:
        return []

    mapf: Callable[[HeadingNode], str] = lambda x: x.get_content()
    return traverse_tree_nodetype(
        root, mapf, HeadingNode, filter_func
    )


def collect_dictionaries(
    root: MarkdownTree,
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
    map_func: Callable[
        [MarkdownNode], NodeDict
    ] = lambda x: x.as_dict(True, False),
) -> list[NodeDict]:
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

    Note:
        Use `map_func` to collect information from the parents of
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


def collect_annotated_textblocks(
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

    Returns:
        a block list

    Note:
        the metadata_block members, corresponding to the
        `private_` member of MetadataBlocks, are ignored.
    """

    if not root:
        return []

    dicts: list[NodeDict] = traverse_tree_nodetype(
        root.tree_copy(),
        lambda x: x.as_dict(inherit, include_header),
        TextNode,
        filter_func,
    )
    blocks: list[Block] = []
    for d in dicts:
        if 'metadata' in d:
            blocks.append(MetadataBlock._from_dict(d['metadata']))
        blocks.append(TextBlock(content=str(d['content'])))
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


MN = TypeVar("MN", bound=MarkdownNode)  # fmt: skip


def get_nodes(
    root: MarkdownNode,
    naked_copy: bool = True,
    node_type: type[MN] = MarkdownNode,
    filter_func: Callable[[MN], bool] = lambda _: True,
) -> list[MN]:
    """
    Find all nodes of node_type, or all such nodes that satisfy a
    predicate function

    Args:
        root: The root node of the tree
        naked_copy: naked copy is a deep copy of a node taken off the
            tree (without parent or children); if False, gives a
            reference. Defaults to True.
        node_type: the type of node to select (default to all)
        filter_func: a predicate to select the nodes (defaults
            to all nodes)

    Returns:
        A list of the text nodes, or of those where filter_func is
        true

    See also:
        `get_text_nodes`, `get_heading_nodes`,
        `get_nodes_with_metadata` for examples of uses
    """
    nodes: list[MN]
    if naked_copy:
        nodes = traverse_tree_nodetype(
            root, lambda x: x.naked_copy(), node_type, filter_func
        )
    else:
        nodes = traverse_tree_nodetype(
            root, lambda x: x, node_type, filter_func
        )
    return nodes


def get_nodes_with_metadata(
    root: MarkdownNode,
    metadata_key: str,
    node_type: type[MN] = MarkdownNode,
    naked_copy: bool = True,
) -> list[MN]:
    """
    Find all nodes that have a specific metadata key.

    Args:
        root: The root node of the tree
        metadata_key: The metadata key to search for
        node_type: the node type, defaults to MarkdownNode
        naked_copy: if True, naked copy, otherwise reference (defaults
            to True)

    Returns:
        A list of nodes that have the specified metadata key
    """

    return get_nodes(
        root,
        naked_copy,
        node_type,
        lambda x: bool(x.metadata) and metadata_key in x.metadata,
    )


def get_textnodes(
    root: MarkdownNode,
    naked_copy: bool = True,
    filter_func: Callable[[TextNode], bool] = lambda _: True,
) -> list[TextNode]:
    """
    Find all text nodes, or all text nodes that satisfy a predicate
    function

    Args:
        root: The root node of the tree
        naked_copy: if True, naked copy, otherwise reference (defaults
            to True)
        filter_func: a predicate to select the text nodes

    Returns:
        A list of the text nodes, or of those where filter_func is
        true
    """

    return get_nodes(root, naked_copy, TextNode, filter_func)


def get_headingnodes(
    root: MarkdownNode,
    naked_copy: bool = True,
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> list[HeadingNode]:
    """
    Find all heading nodes, or all heading nodes that satisfy a
    predicate function

    Args:
        root: The root node of the tree
        naked_copy: if True, naked copy, otherwise reference (defaults
            to True)
        filter_func: a predicate to select the heading nodes

    Returns:
        A list of the heading nodes, or of those where filter_func is
        true
    """

    return get_nodes(root, naked_copy, HeadingNode, filter_func)


# map -------------------------------------------------------------
def pre_order_map_tree(
    node: MarkdownNode,
    map_func: Callable[[MarkdownNode], MarkdownNode],
) -> MarkdownNode:
    """
    Applies map_func in pre-order to the nodes of the tree.

    Args:
        node: The root node of the tree or subtree
        map_func: The function to apply to each node that returns a
            new node

    Returns:
        A new tree with the same structure, but transformed by
            map_func

    Related functions:
        pre_order_traversal: has the same purpose, but with a
            different parameter function signature
            'Callable[[MarkdownNode], None]` and return type None

    Note:
        Make a deep copy of the root node prior to calling this
        function to prevent side effects:
        'pre_order_map(node.tree_copy())'
    """

    mapped_node = map_func(node)
    mapped_node.children = [
        pre_order_map_tree(child, map_func) for child in node.children
    ]

    return mapped_node


def post_order_map_tree(
    node: MarkdownNode,
    map_func: Callable[[MarkdownNode], MarkdownNode],
) -> MarkdownNode:
    """
    Applies map_func in post-order to the nodes of the tree.

    Args:
        node: The root node of the tree or subtree
        map_func: The function to apply to each node that returns a
            new node

    Returns:
        A new tree with the same structure, but transformed by
            map_func

    Related functions:
        post_order_traversal: has the same purpose, but with a
            different parameter function signature
            'Callable[[MarkdownNode], None]` and return type None

    Note:
        Make a deep copy of the root node prior to calling this
        function to prevent side effects:
        `post_order_map(node.tree_copy())`
    """
    node.children = [
        post_order_map_tree(child, map_func)
        for child in node.children
    ]
    return map_func(node)


def prune_tree(
    node: MarkdownTree,
    filter_func: Callable[[MarkdownNode], bool],
) -> MarkdownTree:
    """Prune all nodes of the tree that do not satisfy the predicate
    filter_func."""
    if node is None:
        return None

    if not filter_func(node):
        return None

    def _visit_func(n: MarkdownNode) -> None:
        survivors: list[MarkdownNode] = []
        for child in n.children:
            if filter_func(child):
                survivors.append(child)
        n.children = survivors

    root = node.tree_copy()
    pre_order_traversal(root, _visit_func)
    return root


# utilities -------------------------------------------------------
def print_tree_info(node: MarkdownNode, indent: int = 0) -> None:
    """
    Print information about a node and its descendants, including its
    type, content, and metadata.

    Args:
        node: The node to print information for
        indent: The indentation level for pretty printing
    """
    indent_str = "  " * indent

    # Print node type and content
    if isinstance(node, HeadingNode):
        print(
            f"{indent_str}Heading (Level {node.heading_level()}): "
            + f"{node.get_content()}"
        )
    elif isinstance(node, TextNode):
        content = node.get_content()
        if len(content) > 50:
            content = content[:47] + "..."
        print(f"{indent_str}Text: {content}")
    else:
        print(f"{indent_str}Other: {type(node.block)}")

    # Print metadata
    if node.metadata:
        print(f"{indent_str}  Metadata: {node.metadata}")

    # Print title_ from effective metadata (if it exists)
    title_path = node.fetch_metadata_for_key('title')
    if title_path:
        print(f"{indent_str}  Effective title path: {title_path}")

    # Print children recursively
    for child in node.children:
        print_tree_info(child, indent + 1)
