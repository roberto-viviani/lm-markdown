"""Handles markdown content programmatically, allowing loading and
parsing markdown files, creating a tree representation, and modifying
them."""


from .parse_markdown import (  # noqa: F401
    # block types
    MetadataBlock as MetadataBlock,
    HeaderBlock as HeaderBlock,
    HeadingBlock as HeadingBlock,
    TextBlock as TextBlock,
    ErrorBlock as ErrorBlock,
    Block as Block,
    # functionality on block lists
    parse_markdown_text as parse_markdown_text,
    serialize_blocks as serialize_blocks,
    blocklist_errors as blocklist_errors,
    blocklist_haserrors as blocklist_haserrors,
    blocklist_get_info as blocklist_get_info,
    blocklist_copy as blocklist_copy,
    blocklist_map as blocklist_map,
    load_blocks as load_blocks,
)


from .blocks import (  # noqa: F401
    clear_metadata as clear_metadata,
    merge_textblocks as merge_textblocks,
    pool_textblocks_if as pool_textblocks_if,
    pool_short_textblocks as pool_short_textblocks,
    pool_code_blocks as pool_code_blocks,
    pool_equation_blocks as pool_equation_blocks,
)

from .tree import (  # noqa: F401
    # types
    MarkdownTree as MarkdownTree,
    MarkdownNode as MarkdownNode,
    HeadingNode as HeadingNode,
    TextNode as TextNode,
    TraversalFunc as TraversalFunc,
    # functionality
    blocks_to_tree as blocks_to_tree,
    tree_to_blocks as tree_to_blocks,
    tree_copy as tree_copy,
    pre_order_map as pre_order_map,
    post_order_map as post_order_map,
    pre_order_traversal as pre_order_traversal,
    post_order_traversal as post_order_traversal,
    traverse_tree as traverse_tree,
    traverse_tree_nodetype as traverse_tree_nodetype,
    fold_tree as fold_tree,
)

from .treeutils import (  # noqa: F401
    aggregate_content_in_parent_metadata as aggregate_content_in_parent_metadata,
    inherit_metadata as inherit_metadata,
    extract_property as extract_property,
    collect_text as collect_text,
    collect_table_of_contents as collect_table_of_contents,
    collect_dictionaries as collect_dictionaries,
    collect_headings as collect_headings,
    collect_textblocks as collect_textblocks,
    count_words as count_words,
    CopyOpts as CopyOpts,
    get_nodes as get_nodes,
    get_headingnodes as get_headingnodes,
    get_nodes_with_metadata as get_nodes_with_metadata,
    get_textnodes as get_textnodes,
)

from .ioutils import (  # noqa: F401
    load_markdown as load_markdown,
    save_markdown as save_markdown,
)
