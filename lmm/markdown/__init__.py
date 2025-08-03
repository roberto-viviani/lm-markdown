"""Handles markdown content programmatically, allowing loading and
parsing markdown files, creating a tree representation, and modifying
them."""

# flake8: noqa F401

from .parse_markdown import (
    # block types
    MetadataBlock,
    HeaderBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
    Block,
    # functionality on block lists
    parse_markdown_text,
    serialize_blocks,
    blocklist_errors,
    blocklist_haserrors,
    blocklist_get_info,
    blocklist_copy,
    blocklist_map,
    load_blocks,
)


from .blocks import (
    clear_metadata,
    merge_textblocks,
    pool_textblocks_if,
    pool_short_textblocks,
    pool_code_blocks,
    pool_equation_blocks,
)

from .tree import (
    # types
    MarkdownTree,
    MarkdownNode,
    HeadingNode,
    TextNode,
    TraversalFunc,
    # functionality
    blocks_to_tree,
    tree_to_blocks,
    tree_copy,
    pre_order_map,
    post_order_map,
    pre_order_traversal,
    post_order_traversal,
    traverse_tree,
    traverse_tree_nodetype,
    fold_tree,
)

from .treeutils import (
    aggregate_content_in_parent_metadata,
    inherit_metadata,
    extract_property,
    collect_text,
    collect_table_of_contents,
    collect_dictionaries,
    collect_headings,
    collect_textblocks,
    count_words,
    CopyOpts,
    get_nodes,
    get_headingnodes,
    get_nodes_with_metadata,
    get_textnodes,
)

from .ioutils import (
    load_markdown,
    save_markdown,
)
