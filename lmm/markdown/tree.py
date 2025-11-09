"""
This module provides functionality to represent a Markdown document
as a tree. The headings constitute the nodes of the tree
(`HeadingNode`) and the text blocks constitute the leafs (`TextNode`).
Metadata in the markdown document are converted to metadata properties
of nodes. The metadata blocks are interpreted as annotations for the
metadata of the block that follows, be it a heading or a text block.

In general, trees will be not constructed manually but will be
constructed from blocks of parsed markdown:

```python
blocks = load_markdown("my_markdown.md")
root = blocks_to_tree(blocks)
```

After working on the markdown, it may be retransformed into a block
list:

```python
blocks = tree_to_blocks(root)
save_markdown("my_markdown.md", blocks)
```

The first block in the list must be a header or a metadata block, from
which the root node of the tree is built. If no header block is
present at the beginning of the list, one is created with default
values. Other parent nodes are built from the heading blocks, and the
leaf nodes from the text blocks. Except for the first block, metadata
blocks are used annotate the nodes with properties saved as metadata.

```markdown
---  # This will become the metadata of the root node
title: The document  # This will also be the content of the root node
description: header
---

---  # This metadata block will annotate the heading that follows
summary: introductory words
---

# Introduction

This is the text of the introduction. The property 'summary' will by
default be applied to this text too, since it is a descendant of the
heading for which the metadata were defined.
```

Importantly, while in a blocklist there are blocks of header/metadata,
heading, and text types, a tree only has heading and text nodes. The
root node is a heading from the title property of the header metadata
block.

Note:
    the tree is a representation of the blocks in the blocklist, and
    not a copy of them. It is used to operate on the markdown data
    through side effects.
"""

# using protected members
# pyright: reportPrivateUsage=false

from abc import ABC, abstractmethod
from typing import TypeVar, Self
from collections.abc import Callable, Sequence
import copy
from pathlib import Path

from .parse_yaml import (
    MetadataDict,
    MetadataValue,
    is_metadata_primitive,
)
from .parse_markdown import (
    Block,
    MetadataBlock,
    HeaderBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
)
from .parse_markdown import serialize_blocks, load_blocks
from .ioutils import report_error_blocks
from lmm.utils.logging import LoggerBase, get_logger


from typing import TypedDict  # fmt: skip
class NodeDict(TypedDict):
    """A dictionary representation of a node.

    Fields:
        content (str): the string content of the node (the title for
            a heading node, the text of a text node)
        metadata (MetadataDict): the metadata
    """

    content: str
    metadata: MetadataDict


T = TypeVar("T")
U = TypeVar("U")


class MarkdownNode(ABC):
    """
    Represents a node in the markdown tree structure.

    Each node contains a markdown block (heading or text), a reference
    to its parent, a list of children, and associated metadata. Both
    heading and text blocks have textual content.

    Use accessor functions to read properties of the node.
    """

    def __init__(
        self, block: Block, parent: 'HeadingNode | None' = None
    ):
        """
        Initialize a new MarkdownNode.

        Args:
            block: The original block (heading or text)
            parent: The parent node (None for root)
        """
        self.block: Block = block  # Original block
        self.parent: 'HeadingNode | None' = parent  # Parent node
        self.children: list['MarkdownNode'] = []  # Child nodes
        self.metadata: MetadataDict = {}  # Associated metadata
        # Original metadata block. This is a private immutable data
        # member that is used at serialization to reconstitute the
        # original markdown. Changes in self.metadata will be
        # incorporated in the original metadata block when the block
        # is serialized.
        self.metadata_block: HeaderBlock | MetadataBlock | None = None

    # Copy
    @abstractmethod
    def naked_copy(self) -> Self:
        """Make a deep copy of this node and take it off the tree."""
        pass

    @abstractmethod
    def node_copy(self) -> Self:
        """Make a copy of this node, but keep links to children."""
        pass

    @abstractmethod
    def tree_copy(self) -> Self:
        """Make a deep copy of this node and its children
        (copy subtree)."""
        pass

    # Utility functions to retrieve basic properties
    def is_header_node(self) -> bool:
        """A node initialized from a header block."""
        return self.metadata_block is not None and isinstance(
            self.metadata_block, HeaderBlock
        )

    def is_root_node(self) -> bool:
        """A node w/o parents, not necessarily a header."""
        return self.parent is None

    @abstractmethod
    def get_text_children(self) -> list['TextNode']:
        pass

    @abstractmethod
    def get_heading_children(self) -> list['HeadingNode']:
        pass

    def get_parent(self) -> 'HeadingNode | None':
        return self.parent

    def count_children(self) -> int:
        return len(self.children)

    @abstractmethod
    def heading_level(self) -> int | None:
        pass

    # Content and metadata
    @abstractmethod
    def get_content(self) -> str:
        """Returns text of headings or of text nodes."""
        pass

    @abstractmethod
    def set_content(self, content: str) -> None:
        """Set text of headings or of text nodes."""
        pass

    def get_metadata(self, key: str | None = None) -> MetadataDict:
        """
        Get the metadata of the current node. For the header node,
        the document header is the metadata.

        Returns:
            a conformant dictionary.
        """
        if not key:
            return copy.deepcopy(self.metadata)
        elif self.metadata and key in self.metadata:
            return {key: self.metadata[key]}
        return {}

    def get_metadata_for_key(
        self, key: str, default: MetadataValue = None
    ) -> MetadataValue:
        """
        Get the key value in the metadata of the current node. For the
        root node, the header value for that key is returned.

        Args:
            key: the key for which the metadata is searched
            default: a default value if the key is absent

        Returns:
            the key value, or a default value if the key is not
                found (None if no default specified).
        """
        if key in self.metadata:
            return self.metadata[key]
        return default

    def get_metadata_string_for_key(
        self, key: str, default: str | None = None
    ) -> str | None:
        """
        Get the string representation of the key value in the
        metadata of the current node. If the value is a dict
        of list, return None. If the key is not present, return
        a default value. For the root node, the header value for
        that key is returned.

        Args:
            key: the key for which the metadata is searched
            default: a default value if the key is absent

        Returns:
            the key value, a default value if the key is not
                found (None if no default specified). If the value
                is not a primitive value of the int, float, str, or
                bool type, returns None.
        """
        if key in self.metadata:
            value: MetadataValue = self.metadata[key]
            if is_metadata_primitive(value):
                return str(value)
            else:
                return None
        return default

    def set_metadata_for_key(
        self, key: str, value: MetadataValue
    ) -> None:
        """Set the metadata value at key of the current node.

        Args:
            key: the key where the value should be set
            value: the metadata value
        """
        self.metadata[key] = value

    def fetch_metadata(
        self, key: str | None = None, include_header: bool = True
    ) -> MetadataDict:
        """
        Returns the effective metadata for this node by traversing up
        the tree if necessary to find inherited metadata. The metadata
        are those of the first node with metadata. If a key is given,
        only a dictionary with that key will be returned. If
        include_header is False, the header node is not considered.

        Args:
            key: the key for which the metadata is searched
            include_header: if the header metadata should be included
                in the search

        Returns:
            A dictionary giving the effective metadata for this node
        """

        if not key:
            if self.metadata:
                if not include_header and self.is_header_node():
                    return {}
                return self.metadata.copy()
            elif self.parent:
                return self.parent.fetch_metadata(
                    None,
                    include_header,
                )
            return {}
        else:
            value = self.fetch_metadata_for_key(key, include_header)
            return {key: value} if value else {}

    def fetch_metadata_for_key(
        self,
        key: str,
        include_header: bool = True,
        default: MetadataValue = None,
    ) -> MetadataValue:
        """
        Returns the value for a specific metadata key by traversing up
        the tree if necessary to find inherited metadata. If
        include_header is False, the header node is not considered.

        This function extends the concept of metadata inheritance to
        look for a specific key in the node's metadata or its
        ancestors' metadata.

        Args:
            key: The specific metadata key to look for
            include_header: If to include the header in the search
            default: the value to return if the key is not found

        Returns:
            The value for the specified key, or a default value if not
            found in the node or any of its ancestors (or None if no
            default was specified).
        """

        if not key:
            return default

        if self.metadata and key in self.metadata:
            if not include_header and self.is_header_node():
                return default
            return self.metadata[key]
        elif self.parent:
            return self.parent.fetch_metadata_for_key(
                key, include_header, default
            )
        return default

    def fetch_metadata_string_for_key(
        self,
        key: str,
        include_header: bool = True,
        default: str | None = None,
    ) -> str | None:
        """
        Returns the string representation of the value of a specific
        metadata key by traversing up the tree if necessary to find
        inherited metadata. If the key is not present, return a
        default value. If the key is a dict or list, return None. If
        include_header is False, the header node is not considered.

        This function extends the concept of metadata inheritance to
        look for a specific key in the node's metadata or its
        ancestors' metadata.

        Args:
            key: The specific metadata key to look for
            include_header: If to include the header in the search
            default: the value to return if the key is not found

        Returns:
            The value for the specified key, or a default value if not
            found in the node or any of its ancestors (or None if no
            default was specified). If the value is not a primitive
            value of the int, float, str, or bool type returns None.
        """

        if not key:
            return default

        if self.metadata and key in self.metadata:
            if not include_header and self.is_header_node():
                return default
            return self.get_metadata_string_for_key(key)
        elif self.parent:
            return self.parent.fetch_metadata_string_for_key(
                key, include_header, default
            )
        return default

    def as_dict(
        self, inherit: bool = False, include_header: bool = True
    ) -> NodeDict:
        """
        Return a dictionary representation of the node.

        Args:
            inherit: if True, the metadata of the first parent with
                metadata
            include_header: if False, the header is not considered as
                a metadata source.

        Returns: a dictionary with keys 'content' and
        'metadata'.
        """
        if not inherit:
            return {
                'content': self.get_content(),
                'metadata': self.get_metadata(),
            }
        else:
            return {
                'content': self.get_content(),
                'metadata': self.fetch_metadata(None, include_header),
            }

    @abstractmethod
    def get_info(self) -> str:
        """Return a string rpresentation of the node with info
        on its properties"""
        pass


# Define the node types
class HeadingNode(MarkdownNode):
    def __init__(
        self,
        block: HeadingBlock | HeaderBlock,
        parent: "HeadingNode | None" = None,
    ):
        # this can only happen if type checks were ignored
        assert isinstance(
            block, (HeadingBlock, HeaderBlock)
        ), f"Invalid block type: {type(block)}"
        # type checker complains here, but it will enforce the
        # type at any subsequent access to the block data member,
        # simulating covariance
        self.block: HeadingBlock | HeaderBlock  # type: ignore
        super().__init__(block, parent)

    # Copy
    def naked_copy(self) -> 'HeadingNode':
        """Make a deep copy of this node and take it off the tree"""
        block_copy: HeadingBlock | HeaderBlock = (
            self.block.model_copy(deep=True)
        )
        new_node = HeadingNode(block_copy)
        new_node.metadata = copy.deepcopy(self.metadata)
        if self.metadata_block:
            new_node.metadata_block = self.metadata_block.model_copy(
                deep=True
            )

        return new_node

    def node_copy(self) -> 'HeadingNode':
        """Make a deep copy of this node, maintaining links to
        children. This creates a new branch root with reference to
        all children, but detached from the upper tree.
        """

        newnode = self.naked_copy()
        newnode.children = self.children
        return newnode

    def tree_copy(self) -> 'HeadingNode':
        """Make a deep copy of this node and its children
        (copy subtree)"""

        new_node = self.naked_copy()

        for child in self.children:
            child_copy = child.tree_copy()
            child_copy.parent = new_node
            new_node.children.append(child_copy)

        return new_node

    # Utility functions to retrieve basic properties
    def get_text_children(self) -> list['TextNode']:
        return [n for n in self.children if isinstance(n, TextNode)]

    def get_heading_children(self) -> list['HeadingNode']:
        return [
            n for n in self.children if isinstance(n, HeadingNode)
        ]

    def heading_level(self) -> int:
        if isinstance(self.block, HeadingBlock):
            return self.block.level
        return 0  # Root level for HeaderBlock

    def get_content(self) -> str:
        """Returns the title of the heading represented by the node"""
        match self.block:
            case HeadingBlock() if self.is_header_node():
                return str(self.get_metadata_for_key('title'))
            case HeadingBlock():
                return self.block.get_content()
            case HeaderBlock():
                raise RuntimeError(
                    "Unreachable code reached: "
                    + "unrecognized block type in node"
                )
            case _:
                raise RuntimeError(
                    "Unreachable code reached: "
                    + "unrecognized block type in node"
                )

    def set_content(self, content: str) -> None:
        match self.block:
            case HeadingBlock() if self.is_header_node():
                self.set_metadata_for_key('title', content)
            case HeadingBlock():
                self.block.content = content
            case _:
                raise RuntimeError(
                    "Unreachable code reached: "
                    + "unrecognized block type in node"
                )

    def get_info(self, indent: int = 0) -> str:
        """
        Reports information about a node, including its type, content,
        and metadata.

        Args:
            indent: The indentation level for pretty printing
        """
        import yaml

        indent_str = "  " * indent

        # Collect block type and content
        info: str
        if self.is_header_node():
            info = "Header node:\n"
            info += f"{indent_str}Content: {self.get_content()}"
        else:
            info = "Heading node\n"
            info += (
                f"{indent_str}Heading (Level {self.heading_level()}):"
                + f" {self.get_content()}"
            )

        # Info on children
        if self.children:
            info += f"\nHas {self.count_children()} children, of "
            info += (
                f"which {len(self.get_text_children())} are "
                + "text children"
            )
        else:
            info += "\nNode with no children"

        # Collect metadata
        if self.metadata:
            info += (
                f"\n{indent_str}Metadata:"
                + f"\n{yaml.safe_dump(self.get_metadata())}"
            )

        if not info[-1] == '\n':
            info += "\n"
        return info

    def add_child(self, child_node: MarkdownNode) -> None:
        """
        Add a child node to this node.

        Args:
            child_node: The node to add as a child

        Note:
            one cannot add a heading node with a level equal or higher
            than that of the parent node. The level of the heading
            node is adjusted downwards automatically. Beyond level 6,
            a text node is added.

        TODO:
            Add testing
        """
        if isinstance(child_node, HeadingNode):
            if self.heading_level() == 6:
                self.children.append(
                    TextNode.from_content(
                        content="######## "
                        + child_node.get_content(),
                        metadata=child_node.metadata,
                        parent=self,
                    )
                )
                return
            elif child_node.heading_level() <= self.heading_level():
                new_node = HeadingNode(
                    block=HeadingBlock(
                        level=self.heading_level() + 1,
                        content=child_node.get_content(),
                    )
                )
                new_node.metadata = child_node.metadata
                new_node.metadata_block = MetadataBlock(
                    content=child_node.metadata
                )
                child_node = new_node

        child_node.parent = self
        self.children.append(child_node)


class TextNode(MarkdownNode):
    def __init__(
        self,
        block: TextBlock | ErrorBlock,
        parent: "HeadingNode | None" = None,
    ):
        # this can only happen if type checks were ignored
        assert isinstance(
            block, (TextBlock, ErrorBlock)
        ), f"Invalid block type: {type(block)}"
        # type checker complains here, but it will enforce the
        # type at any subsequent access to the block data member,
        # simulating covariance
        self.block: TextBlock | ErrorBlock  # type: ignore
        super().__init__(block, parent)

    @staticmethod
    def from_content(
        content: str,
        metadata: MetadataDict = {},
        parent: HeadingNode | None = None,
    ) -> 'TextNode':
        """Create a text node from content and metadata."""
        newnode = TextNode(
            block=TextBlock.from_text(content), parent=parent
        )
        if metadata:
            newnode.metadata = metadata
        return newnode

    # Copy
    def naked_copy(self) -> 'TextNode':
        """Make a deep copy of this node and take it off the tree"""
        block_copy: TextBlock | ErrorBlock = self.block.model_copy(
            deep=True
        )
        new_node = TextNode(block_copy)
        new_node.metadata = copy.deepcopy(self.metadata)
        if self.metadata_block:
            new_node.metadata_block = self.metadata_block.model_copy(
                deep=True
            )

        return new_node

    def node_copy(self) -> 'TextNode':
        """Make a deep copy of this node (same as naked_copy for
        TextNodes)."""

        return self.naked_copy()

    def tree_copy(self) -> 'TextNode':
        """Make a deep copy of this node"""

        # Copy self
        return self.naked_copy()

    # Utility functions to retrieve basic properties
    def get_text_children(self) -> list['TextNode']:
        """Always returns an empty list for text nodes."""
        return []  # Text nodes have no children

    def get_heading_children(self) -> list['HeadingNode']:
        """Always returns an empty list for text nodes."""
        return []  # Text nodes have no children

    def heading_level(self) -> None:
        """Return None for the level of text nodes."""
        return None  # Text nodes don't have levels

    def get_content(self) -> str:
        """Returns the content of the markdown text represented by
        the node."""
        return self.block.get_content()

    def set_content(self, content: str) -> None:
        """Set the content of the node"""
        self.block.content = content

    def get_info(self, indent: int = 0) -> str:
        """
        Reports information about a node, including its type, content,
        and metadata.

        Args:
            indent: The indentation level for pretty printing
        """
        import yaml

        indent_str = "  " * indent

        # Collect block type and content
        info = "Text node"
        info += "\n" if self.get_parent() else " (freestanding)\n"

        content = self.get_content()
        if len(content) > 50:
            content = content[:47] + "..."
        if len(content) > 0:
            info += f"{indent_str}Text: {content}"
        else:
            info += "Placeholder text block for metadata"

        # Collect metadata
        if self.metadata:
            info += (
                f"\n{indent_str}Metadata:"
                + f"\n{yaml.safe_dump(self.metadata)}"
            )

        if not info[-1] == '\n':
            info += "\n"
        return info


# represent tree with the root heading node while allowing an empty
# tree to maintain the isomorphism with a list
MarkdownTree = HeadingNode | None


def blocks_to_tree(
    blocks: list[Block], logger: LoggerBase = get_logger(__name__)
) -> MarkdownTree:
    """
    Builds a tree representation of a list of markdown blocks.

    Args:
        blocks: The list of blocks parsed from a markdown file

    Returns:
        A root node, or None for an empty block list.

    Note:
        conversion to tree of a non-empty block list adds a
        metadata blocks in front, if missing, and an empty text
        block after metadata blocks without following text or
        heading block, to the original list of markdown blocks.
        If the block list starts with a heading, adds a header
        with the content of the heading.

    Note:
        the nodes contain references to blocks. To avoid side
        effects, copy the blocks first:
        ```
        root = blocks_to_tree(blocklist_copy(blocks))
        ```
    """
    if not blocks:
        return None

    # Report error blocks in logger
    report_error_blocks(blocks, logger)

    # Enforce the first block being header
    header_block: HeaderBlock
    match blocks[0]:
        case HeaderBlock():
            header_block = blocks[0]
        case MetadataBlock() as bl:
            header_block = HeaderBlock._from_metadata_block(bl)
        case HeadingBlock() as bl:
            if bl.get_content():
                header_block = HeaderBlock(
                    content={'title': bl.get_content()}
                )
            else:
                header_block = HeaderBlock.from_default()
            blocks = [header_block] + blocks
        case TextBlock() | ErrorBlock():
            header_block = HeaderBlock.from_default()
            blocks = [header_block] + blocks

    # Create root node as containing a HeadingBlock with the
    # document title as content.
    root_title = str(header_block.content["title"])
    root_block = HeadingBlock(level=0, content=root_title)
    root_node = HeadingNode(root_block)
    root_node.metadata = header_block.content
    # Store the original header block for reconstitution
    root_node.metadata_block = header_block

    current_node = root_node
    current_metadata: MetadataDict | None = None

    # Process remaining blocks
    current_metadata_block: MetadataBlock | None = None

    def _find_appropriate_parent(
        cur_node: HeadingNode, new_heading_level: int
    ) -> HeadingNode:
        """
        Finds the appropriate parent node for a new heading based on
        its level.

        Args:
            cur_node: The current node in the tree
            new_heading_level: The level of the new heading

        Returns:
            The appropriate parent node for the new heading
        """
        # For HeadingNode, we can now safely access the level
        while (
            cur_node.parent
            and cur_node.heading_level() >= new_heading_level
        ):
            cur_node = cur_node.parent
        return cur_node

    for block in blocks[1:]:
        match block:
            case HeadingBlock():
                # Appropriate parent depending on level
                parent = _find_appropriate_parent(
                    current_node, block.level
                )

                new_node = HeadingNode(block)
                if current_metadata:
                    new_node.metadata = current_metadata
                    new_node.metadata_block = current_metadata_block
                    current_metadata = None
                    current_metadata_block = None

                parent.add_child(new_node)
                current_node = new_node

            case MetadataBlock():
                # Handle consecutive metadata blocks
                if current_metadata:
                    # Create empty text node with the metadata
                    empty_text_block = TextBlock(content="")
                    text_node = TextNode(empty_text_block)
                    text_node.metadata = current_metadata
                    text_node.metadata_block = current_metadata_block
                    current_node.add_child(text_node)

                current_metadata = block.content
                current_metadata_block = block

            case TextBlock():
                text_node = TextNode(block)
                if current_metadata:
                    text_node.metadata = current_metadata
                    text_node.metadata_block = current_metadata_block
                    current_metadata = None
                    current_metadata_block = None

                current_node.add_child(text_node)

            case ErrorBlock():
                # Text node that contains the offending text
                text_node = TextNode(block)

                # Add to current node
                current_node.add_child(text_node)

    # Handle any remaining metadata
    if current_metadata:
        empty_text_block = TextBlock(content="")
        text_node = TextNode(empty_text_block)
        text_node.metadata = current_metadata
        text_node.metadata_block = current_metadata_block
        current_node.add_child(text_node)

    return root_node


def tree_to_blocks(
    root_node: MarkdownNode | MarkdownTree,
) -> list[Block]:
    """
    Reconstitutes the original block list from the tree
        representation.

    Args:
        root_node: The root node of the tree

    Returns:
        The reconstituted list of blocks.

    Note:
        the blocks contain references to node components. To
        avoid side effects, copy the tree first:
        ```python
        tree_to_blocks(tree_copy(root_node))
        ```
    """
    if not root_node:
        return []

    blocks: list[Block] = []

    # Special handling for the root node
    # The root node is artificial (created from the header metadata)
    # So we only add the header metadata, not the heading itself
    if root_node.is_header_node():
        if root_node.metadata_block:
            newblock = root_node.metadata_block
            if root_node.metadata:
                newblock.content = root_node.metadata
            blocks.append(newblock)
        elif root_node.metadata:
            blocks.append(HeaderBlock(content=root_node.metadata))
        else:
            blocks.append(HeaderBlock.from_default())

    # Process all child nodes
    def process_node(node: MarkdownNode) -> None:
        # Skip the root node as it's handled separately
        if node == root_node:
            return

        match node:
            case HeadingNode():
                # For heading nodes, first add metadata if present
                if node.metadata_block:
                    newblock = node.metadata_block
                    if node.metadata:
                        newblock.content = node.metadata
                    blocks.append(newblock)
                elif node.metadata:
                    blocks.append(
                        MetadataBlock(content=node.metadata)
                    )
                else:
                    pass
                # Then add the heading block
                blocks.append(node.block)
            case TextNode():
                # For text nodes, first add metadata if present
                if node.metadata_block:
                    newblock = node.metadata_block
                    if node.metadata:
                        newblock.content = node.metadata
                    blocks.append(newblock)
                elif node.metadata:
                    blocks.append(
                        MetadataBlock(content=node.metadata)
                    )
                else:
                    pass
                # Then add the text block
                blocks.append(node.block)
            case _:
                raise RuntimeError(
                    "Unreachable code reached: unrecognized node type"
                )

    # Perform pre-order traversal
    pre_order_traversal(root_node, process_node)

    return blocks


# Note: all remaining functions work with subtrees or trees, but do
# not take empty trees by type.


def tree_copy(root: MarkdownNode) -> MarkdownNode:
    """Make a deep copy of a non-empty tree or subtree.
    Returns:
        a root node with a copy of the tree.
    """
    return root.tree_copy()


# traversal -------------------------------------------------------
def pre_order_traversal(
    node: MarkdownNode, visit_func: Callable[[MarkdownNode], None]
) -> None:
    """
    Performs a pre-order traversal of the tree, applying visit_func to
        each node.

    Args:
        node: The root node of the tree or subtree
        visit_func: The function to apply to each node

    Note:
        this function may be used with side effects on the tree
    """
    visit_func(node)
    for child in node.children:
        pre_order_traversal(child, visit_func)


def post_order_traversal(
    node: MarkdownNode, visit_func: Callable[[MarkdownNode], None]
) -> None:
    """
    Performs a post-order traversal of the tree, applying visit_func
        to each node.

    Args:
        node: The root node of the tree or subtree
        visit_func: The function to apply to each node

    Note:
        this function may be used with side effects on the tree
    """
    for child in node.children:
        post_order_traversal(child, visit_func)
    visit_func(node)


# type declaration of previous two functions
TraversalFunc = Callable[
    [MarkdownNode, Callable[[MarkdownNode], None]], None
]


def traverse_tree(
    node: MarkdownNode,
    map_func: Callable[[MarkdownNode], T],
    filter_func: Callable[[MarkdownNode], bool] = lambda _: True,
    traversal_func: TraversalFunc = pre_order_traversal,
) -> list[T]:
    """
    Applies map_func to each node in the tree using the specified
        traversal function, for nodes satisfying the predicate
        boolean_func(n). The traversal produces a list of the
        return type of the map function.

    Args:
        node: The root node of the tree or subtree
        map_func: The function to apply to each node
        filter_func: A predicate to select the nodes to which the
            map function will be applied and add to the list
        traversal_func: The traversal function to use
            (pre_order_traversal by default)

    Returns:
        A list containing the results of applying map_func to
            each node

    Example:
        ```python
        def collect_contents(root: MarkdownNode) -> list[str]:
            return traverse_tree(root, lambda n: n.get_content())
        ```
    """
    result: list[T] = []

    def collect_results(n: MarkdownNode) -> None:
        if filter_func(n):
            result.append(map_func(n))

    traversal_func(node, collect_results)
    return result


MN = TypeVar("MN", bound=MarkdownNode)  # fmt: skip


def traverse_tree_nodetype(
    node: MarkdownNode,
    map_func: Callable[[MN], T],
    node_type: type[MN],
    filter_func: Callable[[MN], bool] = lambda _: True,
    traversal_func: Callable[
        [MarkdownNode, Callable[[MarkdownNode], None]], None
    ] = pre_order_traversal,
) -> list[T]:
    """
    Applies map_func to each node in the tree using the specified
        traversal function, for nodes of the specified type. The type
        must be a subclass of MarkdownNode. The traversal produces a
        list of the return type of the map function.

    Args:
        node: The root node of the tree or subtree
        map_func: The function to apply to each node
        node_type: The type of nodes to apply map_func to and include
            in the output list
        filter_func: A predicate function to filter the traversed
            nodes (defaults to true)
        traversal_func: The traversal function to use
            (pre_order_traversal by default)

    Returns:
        A list containing the results of applying map_func to nodes of
            type node_type

    Example:
        ```python
        def collect_titles(root: MarkdownNode) -> list[str]:
            return traverse_tree_nodetype(root,
                                          lambda n: n.get_content()),
                                          HeadingNode)
        ```
    """
    result: list[T] = []

    def collect_results(n: MarkdownNode) -> None:
        if isinstance(n, node_type) and filter_func(n):
            # If n is of the correct type, apply map_func
            result.append(map_func(n))

    traversal_func(node, collect_results)
    return result


# fold ------------------------------------------------------------
def fold_tree(
    node: MarkdownNode,
    fold_func: Callable[[U, MarkdownNode], U],
    initial_value: U,
    traversal_func: TraversalFunc = post_order_traversal,
) -> U:
    """
    Applies fold_func to accumulate a value across the tree using the
        specified traversal function. The fold function has no
        access to the children and parent of the node.

    Args:
        node: The root node of the tree or subtree
        fold_func: The function to apply to accumulate values
        initial_value: The initial value for the accumulation
        traversal_func: The traversal function to use
            (post_order_traversal by default)

    Returns:
        The accumulated value
    """
    result = [initial_value]

    def accumulate(n: MarkdownNode) -> None:
        result[0] = fold_func(result[0], n.naked_copy())

    traversal_func(node, accumulate)
    return result[0]


# exchange information between metadata and content---------------
def extract_content(
    root_node: HeadingNode,
    output_key: str,
    extract_func: Callable[[Sequence[MarkdownNode]], MetadataValue],
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> HeadingNode:
    """Extracts information from children content, processes it, and
    saves it in the output_key of metadata of parents. The extraction
    proceeds in post-order traversal to aggregate information bottom-
    up.
    To collect information from lower levels, code extract_func to
    use the information stored in output_key at previous rounds of
    traversal.

    Args:
        root_node: the heading node from which the traversal starts
        output_key: the key of the heading metadata where information
            is stored
        extract_func: The function to extract functions. Args: a list
            of `MarkdownNode`s, returns: a valid metadata value
        filter_func: A predicate function to filter `HeadingNode`s to
            which `extract_func` is applied.

    Returns:
        the root node where the traversal was started.

    Note:
        this is a convenience function to process information in
        the tree, conceptually equivalent to the extraction of
        information from neighbours of a graph.

    See also:
        `propagate_content`: extract information from parents to
        children.

    Example:
        ```python
        def count_words(root: MarkdownNode) -> MarkdownNode
            KEY = "wordscount"

            def proc_sum(data: Sequence[MarkdownNode]) -> str:
                buff: list[str] = []
                for d in data:
                    match d:
                        case TextNode():
                            buff.append(d.get_content())
                        case HeadingNode():
                            value=str(d.get_metadata_for_key(KEY, ""))
                            if value:
                                buff.append(value)
                        case _:
                            raise RuntimeError("Unrecognized node")

                count: int = len((" ".join(buff)).split())
                return f"There are {count} words."

        return extract_content(root, KEY, proc_sum)
        ```
    """

    def process_node(node: MarkdownNode) -> None:
        match node:
            case TextNode():
                return
            case HeadingNode():
                if not filter_func(node):
                    return
            case _:
                raise RuntimeError(
                    "Unreachable code reached: "
                    + "unrecognized node type"
                )

        value: MetadataValue = extract_func(node.children)
        if not node.metadata:
            node.metadata = {}
        node.metadata[output_key] = value

    post_order_traversal(root_node, process_node)
    return root_node


def propagate_content(
    root_node: HeadingNode,
    collect_func: Callable[[HeadingNode], str | TextNode],
    select: bool,
    filter_func: Callable[[HeadingNode], bool] = lambda _: True,
) -> HeadingNode:
    """Use information from parent nodes to develop or replace
    children text nodes. The function traverses the tree top-down
    in pre-order.

    Args:
        root_node: the heading node from which the traversal starts
        collect_func: a function that creates text from a parent node,
            such as from the metadata of the parent node.
            Args: the parent node.
            Returns: a string or a text node. If a string, a text
                node will be created with the string as content. If
                a text node, it will be given the heading node as
                parent. Return a text node if you need to store
                information in the metadata of the text node. Return
                a simple string in all other cases. Example of
                returning a text node:
                    return TextNode.from_content("new content",
                                                 {'key': "value"})
        select: whether to replace all text children with a new
            text node child containing the text, or add the text
            node to the existing children.
        filter_func: A predicate function selecting heading nodes
            that will be processed.

    Returns:
        the root node where the traversal was started.

    Note:
        This function changes the structure of the children of the
        tree, but does not change the structure of the parent nodes.
        For more general operations on the tree structure, call
        pre_order_traversal directly.

    See also:
        `extract_content`: propagate information from children
        to parents; `propagate_property` (treeutils): example of use
    """

    def process_node(node: MarkdownNode) -> None:
        heading_node: HeadingNode
        match node:
            case HeadingNode():
                heading_node = node
            case TextNode():
                return
            case _:
                raise RuntimeError(
                    "Unreachable code reached: unrecognized node type"
                )

        if not filter_func(heading_node):
            return

        value: str | TextNode = collect_func(heading_node)
        if isinstance(value, TextNode):
            new_node = value
            new_node.parent = heading_node
        else:
            new_node: TextNode = TextNode.from_content(
                value, {}, heading_node
            )
        if select:
            new_children: list[MarkdownNode] = [new_node]
            for child in node.children:
                if isinstance(child, HeadingNode):
                    new_children.append(child)
            heading_node.children = new_children
        else:
            heading_node.children.insert(0, new_node)

    pre_order_traversal(root_node, process_node)
    return root_node


# utility to get a reference to node types ------------------------


def get_text_nodes(root: MarkdownNode) -> list[TextNode]:
    """Return (references to) text nodes in tree"""
    return traverse_tree_nodetype(root, lambda x: x, TextNode)


def get_heading_nodes(root: MarkdownNode) -> list[HeadingNode]:
    """Return (references to) heading nodes in tree"""
    return traverse_tree_nodetype(root, lambda x: x, HeadingNode)


# utilities to load and save to file ------------------------------
def load_tree(source: str | Path, logger: LoggerBase) -> MarkdownTree:
    """Load a pandoc markdown file or string into a tree.

    This function wraps blocks_to_tree and adds console logging for
    errors.

    Args:
        source: Path to a markdown file or a string containing
            markdown content. If a single-line string without newlines
            is provided, it's treated as a file path.

    Returns:
        The root object of the tree.
    """

    # Pure parsing function  (no exceptions raised)
    blocks = load_blocks(source, logger=logger)
    if not blocks:
        return None

    # Enforce constraint on first block
    if not isinstance(blocks[0], HeaderBlock):
        header = HeaderBlock.from_default(str(source))
        blocks = [header] + blocks

    return blocks_to_tree(blocks, logger)


def serialize_tree(node: MarkdownTree) -> str:
    """Serialize a markdown tree to a string.

    Args:
        node: the node with descendants to be serialized
    """
    blocks = tree_to_blocks(node)
    return serialize_blocks(blocks)


def get_tree_info(node: MarkdownTree) -> list[str]:
    if node is None:
        return []
    return traverse_tree(node, lambda x: x.get_info())


def save_tree(file_name: str | Path, tree: MarkdownTree) -> None:
    """Write a markdown tree to a markdown file.

    Args:
        file_name: Path to the output file (string or Path object)
        tree: the node with descendants to be serialized
    """
    content = serialize_tree(tree)
    from .ioutils import save_markdown
    from lmm.utils import logger

    save_markdown(file_name, content, logger)
