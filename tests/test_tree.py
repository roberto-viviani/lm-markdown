"""Test tree"""

# flake8: noqa
# pyright: basic
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false

import unittest

import logging
from typing import Sequence

# Attempt to import the functions to be tested
from lmm.markdown.parse_markdown import (
    parse_markdown_text,
    ErrorBlock,
    HeaderBlock,
    HeadingBlock,
    TextBlock,
    MetadataBlock,
    Block,
    MetadataDict,
)
from lmm.markdown.tree import (
    MarkdownTree,
    MarkdownNode,
    HeadingNode,
    TextNode,
    blocks_to_tree,
    tree_to_blocks,
    load_tree,
    traverse_tree_nodetype,
    traverse_tree,
    extract_content,
    post_order_traversal,
    pre_order_traversal,
    propagate_content,
    fold_tree,
)
from lmm.utils.logging import LoglistLogger


class TestNodeConstruction(unittest.TestCase):
    def test_build_headingnode(self):
        src = """---
first: 1
second: [1, 2, 3]
---"""
        data: dict[str, int | list[int] | str] = {
            'first': 1,
            'second': [1, 2, 3],
        }
        logger = LoglistLogger()
        node = load_tree(src, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if node is None:
            raise RuntimeError("Could not parse tree")
        self.assertTrue(node.is_header_node())
        self.assertTrue(node.is_root_node())
        self.assertEqual(node.count_children(), 0)
        self.assertEqual(node.heading_level(), 0)
        self.assertEqual(node.get_content(), "Title")
        meta = node.get_metadata()
        titleddata = data.copy()
        titleddata['title'] = "Title"
        self.assertDictEqual(meta, titleddata)
        meta['first'] = 2
        self.assertDictEqual(node.get_metadata('first'), {'first': 1})
        self.assertEqual(node.get_metadata_for_key('first'), 1)
        self.assertEqual(
            node.get_metadata_string_for_key('first'), "1"
        )
        self.assertDictEqual(node.fetch_metadata(), titleddata)
        self.assertDictEqual(
            node.fetch_metadata('first'), {'first': 1}
        )
        self.assertEqual(node.fetch_metadata_for_key('first'), 1)
        self.assertEqual(
            node.fetch_metadata_string_for_key('first'), "1"
        )
        nodedict = node.as_dict()
        self.assertEqual(nodedict['content'], "Title")
        self.assertEqual(nodedict['metadata'], titleddata)
        self.assertEqual(node.get_heading_children(), [])
        self.assertEqual(node.get_text_children(), [])

        heading = HeadingNode(
            HeadingBlock(level=1, content="Heading")
        )
        node.add_child(heading)
        self.assertEqual(node.count_children(), 1)
        self.assertIs(node.get_heading_children()[0], heading)
        self.assertEqual(heading.parent, node)
        self.assertFalse(heading.is_header_node())
        self.assertFalse(heading.is_root_node())
        self.assertEqual(heading.count_children(), 0)
        self.assertEqual(heading.heading_level(), 1)
        self.assertEqual(heading.get_content(), "Heading")

        text = TextNode(TextBlock.from_text("Text"))
        node.add_child(text)
        self.assertEqual(node.count_children(), 2)
        self.assertIs(node.get_text_children()[0], text)
        self.assertEqual(text.parent, node)
        self.assertFalse(text.is_header_node())
        self.assertFalse(text.is_root_node())
        self.assertEqual(text.count_children(), 0)
        self.assertIsNone(text.heading_level())
        self.assertEqual(text.get_content(), "Text")

        nodecopy = node.naked_copy()
        self.assertTrue(nodecopy.is_header_node())
        self.assertTrue(nodecopy.is_root_node())
        self.assertEqual(nodecopy.count_children(), 0)
        self.assertEqual(nodecopy.heading_level(), 0)
        self.assertEqual(nodecopy.get_content(), "Title")
        meta = nodecopy.get_metadata()
        titleddata = data.copy()
        titleddata['title'] = "Title"
        self.assertDictEqual(meta, titleddata)
        meta['first'] = 2
        self.assertDictEqual(
            nodecopy.get_metadata('first'), {'first': 1}
        )
        self.assertEqual(nodecopy.get_metadata_for_key('first'), 1)
        self.assertDictEqual(nodecopy.fetch_metadata(), titleddata)
        self.assertDictEqual(
            nodecopy.fetch_metadata('first'), {'first': 1}
        )
        self.assertEqual(nodecopy.fetch_metadata_for_key('first'), 1)
        nodedict = nodecopy.as_dict()
        self.assertEqual(nodedict['content'], "Title")
        self.assertEqual(nodedict['metadata'], titleddata)
        self.assertEqual(nodecopy.get_heading_children(), [])
        self.assertEqual(nodecopy.get_text_children(), [])

        nodecopy = node.node_copy()
        self.assertTrue(nodecopy.is_header_node())
        self.assertTrue(nodecopy.is_root_node())
        self.assertEqual(nodecopy.count_children(), 2)
        self.assertEqual(nodecopy.heading_level(), 0)
        self.assertEqual(nodecopy.get_content(), "Title")
        meta = nodecopy.get_metadata()
        titleddata = data.copy()
        titleddata['title'] = "Title"
        self.assertDictEqual(meta, titleddata)
        meta['first'] = 2
        self.assertDictEqual(
            nodecopy.get_metadata('first'), {'first': 1}
        )
        self.assertEqual(nodecopy.get_metadata_for_key('first'), 1)
        self.assertDictEqual(nodecopy.fetch_metadata(), titleddata)
        self.assertDictEqual(
            nodecopy.fetch_metadata('first'), {'first': 1}
        )
        self.assertEqual(nodecopy.fetch_metadata_for_key('first'), 1)
        nodedict = nodecopy.as_dict()
        self.assertEqual(nodedict['content'], "Title")
        self.assertEqual(nodedict['metadata'], titleddata)
        self.assertIs(nodecopy.get_heading_children()[0], heading)
        self.assertIs(nodecopy.get_text_children()[0], text)

        nodecopy.metadata['title'] = "A new title"
        self.assertEqual(node.get_content(), "Title")
        self.assertEqual(nodecopy.get_content(), "A new title")

        heading.add_child(text)
        headingcopy = heading.naked_copy()
        self.assertFalse(headingcopy.is_header_node())
        self.assertTrue(headingcopy.is_root_node())
        self.assertEqual(headingcopy.count_children(), 0)
        self.assertEqual(headingcopy.heading_level(), 1)
        self.assertEqual(headingcopy.get_content(), "Heading")

        headingcopy = heading.node_copy()
        self.assertFalse(headingcopy.is_header_node())
        self.assertTrue(headingcopy.is_root_node())
        self.assertIsNone(headingcopy.parent)
        self.assertEqual(headingcopy.count_children(), 1)
        self.assertEqual(headingcopy.heading_level(), 1)
        self.assertEqual(headingcopy.get_content(), "Heading")


class TestTreeConstruction(unittest.TestCase):
    def test_construction_text_empty(self):
        text = ""
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        self.assertIsNone(root)

    def test_construction_text_regular(self):
        text = "Content of text block\non two lines"
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_text_children()), 1)

    def test_construction_heading(self):
        text = "# A heading"
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_heading_children()), 1)

    def test_construction_heading_empty(self):
        text = "# "
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) > 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_text_children()), 1)

        blocks = tree_to_blocks(root)
        self.assertIsInstance(blocks[-1], ErrorBlock)

    def test_construction_metadata(self):
        data = "---\ntitle: Title\n---"
        logger = LoglistLogger()
        root = load_tree(data, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 0)

    def test_construction_metadata_empty(self):
        data = "---\n---"
        logger = LoglistLogger()
        root = load_tree(data, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) > 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_text_children()), 1)

        blocks = tree_to_blocks(root)
        self.assertIsInstance(blocks[-1], ErrorBlock)

    def test_construction_header_invalid(self):
        text = "---\n1: first line\n---"
        # expected behaviour: invalid dict is set to private_,
        # default dict created for header
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 0)

    def test_construction_metadata_invalid1(self):
        text = "---\n1: first block\n---\n\n---\n2: second block\n---"
        # expected behaviour: invalid dict is set to private,
        # empty dict in metadata replaced with default
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 0)

    def test_construction_metadata_invalid2(self):
        text = "---\n[1, 2, 3]\n---"
        # expected behaviour: invalid dict is set to private,
        # empty dict in metadata replaced with default
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 0)
        self.assertIsNotNone(root.metadata_block)
        self.assertIsNotNone(root.metadata_block.private_)  # type: ignore
        self.assertListEqual(root.metadata_block.private_, [1, 2, 3])  # type: ignore

    def test_construction_metadata_error(self):
        text = "---\nfirst\nsecond\n---"
        # expected behaviour: ErrorBlock instead of MetadataBlock
        logger = LoglistLogger()
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) > 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_text_children()), 1)

        blocks = tree_to_blocks(root)
        self.assertIsInstance(blocks[-1], ErrorBlock)

    def test_construction_header_from_nested_dict(self):
        data = (
            "---\ntitle: MyTitle\ndata:\n  token:\n    nested: 1\n---"
        )
        # expected behaviour: over-nested dict goes in private_
        logger = LoglistLogger()
        root = load_tree(data, logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) == 0)
        if root is None:
            raise RuntimeError("Could not parse tree")
        self.assertEqual(root.count_children(), 0)

        # Default metadata added
        self.assertEqual(root.get_content(), "Title")

    # def test_load_tree(self):
    #     root = load_tree("./tests/test_markdown.md")

    #     self.assertIsNotNone(root)


class TestEdgeTree(unittest.TestCase):
    def test_empty_list(self):
        root: MarkdownTree = blocks_to_tree([])
        blocks = tree_to_blocks(root)

        self.assertEqual(len(blocks), 0)

    def test_noheader_list(self):
        root: MarkdownTree = blocks_to_tree(
            [TextBlock(content="This")]
        )
        blocks = tree_to_blocks(root)

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)

    def test_heading_list(self):
        root: MarkdownTree = blocks_to_tree(
            [HeadingBlock(level=1, content="This")]
        )
        if root is None:
            raise ValueError("Could not form tree")
        blocks = tree_to_blocks(root)

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertEqual(blocks[0].get_key("title"), "This")
        self.assertIsInstance(blocks[1], HeadingBlock)

    def test_error_block(self):
        root: MarkdownTree = blocks_to_tree(
            [ErrorBlock(content="This is an error.")]
        )
        if root is None:
            raise ValueError("Could not form tree")
        self.assertEqual(root.count_children(), 1)
        self.assertEqual(len(root.get_text_children()), 1)

        blocks = tree_to_blocks(root)
        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertEqual(blocks[0].get_key("title"), "Title")
        self.assertIsInstance(blocks[1], ErrorBlock)


class TestParseMarkdown(unittest.TestCase):
    def test_parse(self):
        text = """---\ntitle: header\n---\nThis is text\n\n
            \n---\n- a: list\n- b: this\n---\n"""
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIsInstance(blocks[1], TextBlock)
        self.assertIsInstance(blocks[-1], MetadataBlock)
        root = blocks_to_tree(blocks)  # adds empty text at end
        newblocks = tree_to_blocks(root)
        self.assertIsInstance(newblocks[-1], TextBlock)
        self.assertListEqual(blocks, newblocks[:-1])

    def test_yaml_parse_error(self):
        text = """---\ntitle: header\n---\nThis is text\n\n
            \n---\n- a list\ntext: this\n---\n"""
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[-1], ErrorBlock)
        root = blocks_to_tree(blocks)
        newblocks = tree_to_blocks(root)
        self.assertListEqual(blocks, newblocks)

    def test_invalid_metadata(self):
        """Metadata with a literal"""
        text = "---\ntitle: this\n---\n\n---\nthis is text\n---"
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[-1], ErrorBlock)
        root = blocks_to_tree(blocks)
        newblocks = tree_to_blocks(root)
        self.assertListEqual(blocks, newblocks)

    def test_empty_heading(self):
        """Empty heading"""
        text = "### {class = any}\n"
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[-1], ErrorBlock)
        root = blocks_to_tree(blocks)  # adds header
        newblocks = tree_to_blocks(root)
        self.assertIsInstance(newblocks[0], HeaderBlock)
        self.assertListEqual(blocks, newblocks[1:])

    def test_empty_heading2(self):
        """Empty heading with metadata"""
        text = "---\nnote: 'metadata'\n...\n\n### {class = any}\n"
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertIsInstance(blocks[-1], ErrorBlock)
        root = blocks_to_tree(blocks)
        newblocks = tree_to_blocks(root)
        self.assertListEqual(blocks, newblocks)

    def test_unclosed_metadata(self):
        """Unclosed metadata"""
        text = "---\nnote: 'metadata'\n\n### {class = any}\n"
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[0], ErrorBlock)
        root = blocks_to_tree(blocks)  # adds header
        newblocks = tree_to_blocks(root)
        self.assertIsInstance(newblocks[0], HeaderBlock)
        self.assertListEqual(blocks, newblocks[1:])

    def test_noncomformant_metadata(self):
        """Metadata with non-string key"""
        text = """--- # non-comformant dictionary
(0, 1): a tuple
(1, 2): another tuple
---
text after non-conformant"""
        blocks = parse_markdown_text(text)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertEqual(len(blocks[0].content), 3)
        root = blocks_to_tree(blocks)
        newblocks = tree_to_blocks(root)
        self.assertListEqual(blocks, newblocks)


class TestTreeCopy(unittest.TestCase):
    """Test the tree_copy() methods for HeadingNode and TextNode"""

    def setUp(self):
        """Set up test fixtures with various tree structures"""
        # Create basic blocks for testing
        self.header_block = HeaderBlock(
            content={"title": "Test Document"}
        )
        self.heading_block1 = HeadingBlock(
            level=1, content="Chapter 1"
        )
        self.heading_block2 = HeadingBlock(
            level=2, content="Section 1.1"
        )
        self.text_block1 = TextBlock(
            content="This is some text content."
        )
        self.text_block2 = TextBlock(
            content="More text content here."
        )

        # Create metadata for testing
        self.test_metadata: MetadataDict = {
            "author": "Test Author",
            "tags": ["test", "markdown"],
            "nested": {"key": "value", "list": [1, 2, 3]},
        }
        self.metadata_block = MetadataBlock(
            content=self.test_metadata
        )

    def test_textnode_tree_copy_basic(self):
        """Test basic deep copy of TextNode"""
        # Create a text node
        original = TextNode(self.text_block1)

        # Copy it
        copy = original.tree_copy()

        # Verify they are different objects
        self.assertIsNot(original, copy)
        self.assertIsNot(original.block, copy.block)

        # Verify content is the same
        self.assertEqual(original.get_content(), copy.get_content())
        self.assertEqual(original.block.content, copy.block.content)

        # Verify modifying original doesn't affect copy
        original.block.content = "Modified content"
        self.assertNotEqual(
            original.get_content(), copy.get_content()
        )

    def test_textnode_tree_copy_with_metadata(self):
        """Test TextNode tree_copy with metadata"""
        # Create text node with metadata
        original = TextNode(self.text_block1)
        original.metadata = self.test_metadata.copy()
        original.metadata_block = self.metadata_block

        # Copy it
        copy = original.tree_copy()

        # Verify metadata is deeply copied
        self.assertIsNot(original.metadata, copy.metadata)
        self.assertEqual(original.metadata, copy.metadata)

        # Verify nested metadata is now truly independent (deep copy)
        original.metadata["nested"]["key"] = "modified"  # type: ignore
        # This assertion shows that nested metadata IS now independent
        self.assertNotEqual(
            original.metadata["nested"]["key"],  # type: ignore
            copy.metadata["nested"]["key"],  # type: ignore
        )

        # Verify metadata_block is copied (deep copy via model_copy)
        if original.metadata_block and copy.metadata_block:
            self.assertIsNot(
                original.metadata_block, copy.metadata_block
            )
            # metadata_block.content is deeply copied and independent
            # Note: Changes to node.metadata don't affect metadata_block.content
            # since they are separate objects
            self.assertEqual(
                original.metadata_block.content,
                copy.metadata_block.content,
            )

    def test_headingnode_tree_copy_no_children(self):
        """Test HeadingNode tree_copy with no children"""
        # Create heading node without children
        original = HeadingNode(self.heading_block1)

        # Copy it
        copy = original.tree_copy()

        # Verify they are different objects
        self.assertIsNot(original, copy)
        self.assertIsNot(original.block, copy.block)

        # Verify content is preserved
        self.assertEqual(original.get_content(), copy.get_content())
        self.assertEqual(
            original.heading_level(), copy.heading_level()
        )

        # Verify children list is separate but empty
        self.assertIsNot(original.children, copy.children)
        self.assertEqual(len(copy.children), 0)

    def test_headingnode_tree_copy_with_text_children(self):
        """Test HeadingNode tree_copy with TextNode children"""
        # Create heading with text children
        original = HeadingNode(self.heading_block1)
        text_child1 = TextNode(self.text_block1)
        text_child2 = TextNode(self.text_block2)

        original.add_child(text_child1)
        original.add_child(text_child2)

        # Copy the tree
        copy = original.tree_copy()

        # Verify structure
        self.assertEqual(len(copy.children), 2)
        self.assertIsInstance(copy.children[0], TextNode)
        self.assertIsInstance(copy.children[1], TextNode)

        # Verify children are different objects
        self.assertIsNot(original.children[0], copy.children[0])
        self.assertIsNot(original.children[1], copy.children[1])

        # Verify parent relationships are correct
        self.assertIs(copy.children[0].parent, copy)
        self.assertIs(copy.children[1].parent, copy)

        # Verify content is preserved
        self.assertEqual(
            original.children[0].get_content(),
            copy.children[0].get_content(),
        )
        self.assertEqual(
            original.children[1].get_content(),
            copy.children[1].get_content(),
        )

    def test_headingnode_tree_copy_with_heading_children(self):
        """Test HeadingNode tree_copy with HeadingNode children"""
        # Create nested heading structure
        root = HeadingNode(self.heading_block1)  # Level 1
        child_heading = HeadingNode(self.heading_block2)  # Level 2

        root.add_child(child_heading)

        # Copy the tree
        copy = root.tree_copy()

        # Verify structure
        self.assertEqual(len(copy.children), 1)
        self.assertIsInstance(copy.children[0], HeadingNode)

        # Verify it's a different object
        self.assertIsNot(root.children[0], copy.children[0])

        # Verify parent relationship
        self.assertIs(copy.children[0].parent, copy)

        # Verify content is preserved
        self.assertEqual(
            root.children[0].get_content(),
            copy.children[0].get_content(),
        )
        self.assertEqual(
            root.children[0].heading_level(),
            copy.children[0].heading_level(),
        )

    def test_headingnode_tree_copy_mixed_children(self):
        """Test HeadingNode tree_copy with mixed child types"""
        # Create complex tree structure
        root = HeadingNode(self.heading_block1)
        heading_child = HeadingNode(self.heading_block2)
        text_child1 = TextNode(self.text_block1)
        text_child2 = TextNode(self.text_block2)

        root.add_child(text_child1)
        root.add_child(heading_child)
        root.add_child(text_child2)

        # Copy the tree
        copy = root.tree_copy()

        # Verify structure
        self.assertEqual(len(copy.children), 3)
        self.assertIsInstance(copy.children[0], TextNode)
        self.assertIsInstance(copy.children[1], HeadingNode)
        self.assertIsInstance(copy.children[2], TextNode)

        # Verify all children are different objects
        for i in range(3):
            self.assertIsNot(root.children[i], copy.children[i])
            self.assertIs(copy.children[i].parent, copy)

    def test_headingnode_tree_copy_deep_nesting(self):
        """Test HeadingNode tree_copy with deep nesting"""
        # Create deeply nested structure
        level1 = HeadingNode(HeadingBlock(level=1, content="Level 1"))
        level2 = HeadingNode(HeadingBlock(level=2, content="Level 2"))
        level3 = HeadingNode(HeadingBlock(level=3, content="Level 3"))
        text_leaf = TextNode(TextBlock(content="Deep text"))

        level1.add_child(level2)
        level2.add_child(level3)
        level3.add_child(text_leaf)

        # Copy the tree
        copy = level1.tree_copy()

        # Navigate to the deepest level
        copy_level2 = copy.children[0]
        copy_level3 = copy_level2.children[0]
        copy_text = copy_level3.children[0]

        # Verify deep structure is preserved
        self.assertIsInstance(copy_level2, HeadingNode)
        self.assertIsInstance(copy_level3, HeadingNode)
        self.assertIsInstance(copy_text, TextNode)

        # Verify all are different objects
        self.assertIsNot(level2, copy_level2)
        self.assertIsNot(level3, copy_level3)
        self.assertIsNot(text_leaf, copy_text)

        # Verify parent relationships
        self.assertIs(copy_level2.parent, copy)
        self.assertIs(copy_level3.parent, copy_level2)
        self.assertIs(copy_text.parent, copy_level3)

        # Verify content
        self.assertEqual(copy_text.get_content(), "Deep text")

    def test_tree_copy_independence_modifications(self):
        """Test that modifications to original don't affect copy"""
        # Create tree with metadata
        original = HeadingNode(self.heading_block1)
        original.metadata = self.test_metadata.copy()

        text_child = TextNode(self.text_block1)
        text_child.metadata = {"child_meta": "value"}
        original.add_child(text_child)

        # Copy the tree
        copy = original.tree_copy()

        # Modify original tree
        if isinstance(original.block, HeadingBlock):
            original.block.content = "Modified heading"
        original.metadata["author"] = "Modified Author"
        if isinstance(original.children[0].block, TextBlock):
            original.children[0].block.content = "Modified text"
        original.children[0].metadata["child_meta"] = "modified"

        # Verify copy is unaffected
        self.assertEqual(copy.get_content(), "Chapter 1")
        self.assertEqual(copy.metadata["author"], "Test Author")
        self.assertEqual(
            copy.children[0].get_content(),
            "This is some text content.",
        )
        self.assertEqual(
            copy.children[0].metadata["child_meta"], "value"
        )

    def test_tree_copy_independence_reverse(self):
        """Test that modifications to copy don't affect original"""
        # Create tree
        original = HeadingNode(self.heading_block1)
        text_child = TextNode(self.text_block1)
        original.add_child(text_child)

        # Copy the tree
        copy = original.tree_copy()

        # Modify copy
        if isinstance(copy.block, HeadingBlock):
            copy.block.content = "Modified copy heading"
        if isinstance(copy.children[0].block, TextBlock):
            copy.children[0].block.content = "Modified copy text"

        # Verify original is unaffected
        self.assertEqual(original.get_content(), "Chapter 1")
        self.assertEqual(
            original.children[0].get_content(),
            "This is some text content.",
        )

    def test_tree_copy_with_error_blocks(self):
        """Test tree_copy with ErrorBlock nodes"""
        # Create text node with error block
        error_block = ErrorBlock(
            content="Test error", errormsg="Error message"
        )
        error_node = TextNode(error_block)

        # Copy it
        copy = error_node.tree_copy()

        # Verify error content is preserved
        self.assertEqual(copy.get_content(), "Test error")
        self.assertIsNot(error_node.block, copy.block)

        # Verify it's still an ErrorBlock
        self.assertIsInstance(copy.block, ErrorBlock)

    def test_tree_copy_preserves_all_attributes(self):
        """Test that tree_copy preserves all node attributes"""
        # Create heading with attributes
        heading_with_attrs = HeadingBlock(
            level=2, content="Test", attributes="class=test"
        )
        node = HeadingNode(heading_with_attrs)
        node.metadata = {"key": "value"}

        # Copy it
        copy = node.tree_copy()

        # Verify all attributes are preserved
        if isinstance(copy.block, HeadingBlock):
            self.assertEqual(copy.block.level, 2)
            self.assertEqual(copy.block.content, "Test")
            self.assertEqual(copy.block.attributes, "class=test")
        self.assertEqual(copy.metadata, {"key": "value"})

        # Verify they are separate objects
        self.assertIsNot(node.block, copy.block)
        self.assertIsNot(node.metadata, copy.metadata)


header = HeaderBlock(content={"title": "Test blocklist"})
metadata = MetadataBlock(
    content={
        "questions": "What is the nature of the test? - How can we fix it?",
        "~chat": "Some discussion",
        "summary": "The summary of the text.",
    }
)
heading = HeadingBlock(level=2, content="First title")
text = TextBlock(content="This is text following the heading")
blocklist: list[Block] = [header, metadata, heading, text]


class TestTraverseNodetype(unittest.TestCase):
    def test_text(self):
        root = blocks_to_tree(blocklist)
        if not root:
            raise (Exception("Invalid blocks"))

        def _text_to_dict(n: TextNode) -> dict[str, str]:
            return {"content": n.get_content()}

        dicts: list[dict[str, str]] = traverse_tree_nodetype(
            root, _text_to_dict, TextNode
        )
        self.assertEqual(len(dicts), 1)
        self.assertIsInstance(dicts, list)
        self.assertEqual(
            dicts[0]["content"], "This is text following the heading"
        )

    def test_nontext(self):
        root = blocks_to_tree([header, metadata, heading])
        if not root:
            raise (Exception("Invalid blocks"))

        def _text_to_dict(n: TextNode) -> dict[str, str]:
            return {"content": n.get_content()}

        dicts: list[dict[str, str]] = traverse_tree_nodetype(
            root, _text_to_dict, TextNode
        )
        self.assertEqual(len(dicts), 0)

    def test_heading(self):
        root = blocks_to_tree(blocklist)
        if not root:
            raise (Exception("Invalid blocks"))

        def _text_to_dict(n: HeadingNode) -> dict[str, str]:
            return {"content": n.get_content()}

        dicts: list[dict[str, str]] = traverse_tree_nodetype(
            root, _text_to_dict, HeadingNode
        )
        self.assertEqual(
            len(dicts), 2
        )  # Root heading + child heading
        self.assertIsInstance(dicts, list)
        self.assertEqual(
            dicts[1]["content"], "First title"
        )  # The child heading content


class TestExtractContent(unittest.TestCase):
    def test_extract_content(self):
        root = blocks_to_tree(
            [header, text, metadata, text, heading, text]
        )
        if root is None:
            raise ValueError("Could not form tree")
        KEY = "wordcount"

        def proc_sum(node: Sequence[MarkdownNode]) -> int:
            score: int = 0
            for n in node:
                if isinstance(n, TextNode):
                    score += len(n.get_content().split())
            return score

        root = extract_content(root, KEY, proc_sum)
        self.assertIsInstance(root.get_metadata_for_key(KEY), int)
        self.assertEqual(root.get_metadata_for_key(KEY), 12)

    def test_extract_content_upwards(self):
        root = blocks_to_tree(
            [header, text, metadata, text, heading, text]
        )
        if root is None:
            raise ValueError("Could not form tree")
        KEY = "wordcount"

        def proc_sum(node: Sequence[MarkdownNode]) -> int:
            score: int = 0
            for n in node:
                match n:
                    case TextNode():
                        score += len(n.get_content().split())
                    case HeadingNode() as n_:
                        value = n_.get_metadata_for_key(KEY)
                        if isinstance(value, int):
                            score += value
                    case _:
                        raise ValueError("Invalid node type")
            return score

        root = extract_content(root, KEY, proc_sum)
        self.assertIsInstance(root.get_metadata_for_key(KEY), int)
        self.assertEqual(root.get_metadata_for_key(KEY), 6 * 3)

    def test_extract_content_in_null(self):
        # check works in tree w/o Text nodes w/o error
        root = blocks_to_tree(
            [header, text, metadata, text, heading, text]
        )
        if root is None:
            raise ValueError("Could not form tree")
        KEY = "wordcount"

        def proc_sum(node: Sequence[MarkdownNode]) -> int:
            score: int = 0
            for n in node:
                if isinstance(n, TextNode):
                    score += len(n.get_content().split())
            return score

        # eliminate text nodes
        def del_text_nodes(node: MarkdownNode) -> None:
            if isinstance(node, HeadingNode):
                newkids: list[MarkdownNode] = []
                for k in node.children:
                    if isinstance(k, HeadingNode):
                        newkids.append(k)
                node.children = newkids

        post_order_traversal(root, del_text_nodes)

        root = extract_content(root, KEY, proc_sum)
        self.assertIsInstance(root.get_metadata_for_key(KEY), int)
        self.assertEqual(root.get_metadata_for_key(KEY), 0)

    def test_extract_content_strings(self):
        root = blocks_to_tree(
            [header, text, metadata, text, heading, text]
        )
        KEY = "summary"
        if root is None:
            raise ValueError("Could not form tree")

        def proc_sum(data: Sequence[MarkdownNode]) -> str:
            buff: list[str] = []
            for d in data:
                match d:
                    case TextNode():
                        buff.append(d.get_content())
                    case HeadingNode():
                        value = str(d.get_metadata_for_key(KEY, ""))
                        if value:
                            buff.append(value)
                    case _:
                        raise RuntimeError("Unrecognized node")

            count: int = len((" ".join(buff)).split())
            return f"There are {count} words."

        root = extract_content(root, KEY, proc_sum)
        self.assertIsInstance(
            root.get_metadata_for_key("summary"), str
        )
        self.assertEqual(
            root.get_metadata_for_key("summary"),
            "There are 16 words.",
        )

    def test_extract_content_strings_onlyheader(self):
        # Check works in tree with only header w/o error
        root = blocks_to_tree([header])
        KEY = "summary"
        if root is None:
            raise ValueError("Could not form tree")

        def proc_sum(data: Sequence[MarkdownNode]) -> str:
            buff: list[str] = []
            for d in data:
                match d:
                    case TextNode():
                        buff.append(d.get_content())
                    case HeadingNode():
                        value = str(d.get_metadata_for_key(KEY, ""))
                        if value:
                            buff.append(value)
                    case _:
                        raise RuntimeError("Unrecognized node")

            count: int = len((" ".join(buff)).split())
            return f"There are {count} words."

        root = extract_content(root, KEY, proc_sum)
        self.assertIsInstance(
            root.get_metadata_for_key("summary"), str
        )
        self.assertEqual(
            root.get_metadata_for_key("summary"),
            "There are 0 words.",
        )


class TestAddChild(unittest.TestCase):
    """Test the add_child() level adjustment functionality."""

    def test_add_child_normal_case(self):
        """Adding child with higher level - no adjustment needed."""
        parent = HeadingNode(HeadingBlock(level=1, content="Parent"))
        child = HeadingNode(HeadingBlock(level=3, content="Child"))
        child.metadata = {"key": "value"}

        parent.add_child(child)

        self.assertEqual(len(parent.children), 1)
        self.assertIs(parent.children[0], child)
        self.assertEqual(child.heading_level(), 3)
        self.assertEqual(child.get_content(), "Child")
        self.assertEqual(child.metadata, {"key": "value"})

    def test_add_child_level_adjustment_same_level(self):
        """Adding child with same level as parent - adjusted to parent+1."""
        parent = HeadingNode(HeadingBlock(level=2, content="Parent"))
        child = HeadingNode(HeadingBlock(level=2, content="Child"))
        child.metadata = {"key": "value"}

        parent.add_child(child)

        self.assertEqual(len(parent.children), 1)
        # Child should be adjusted to level 3
        self.assertIsInstance(parent.children[0], HeadingNode)
        self.assertEqual(parent.children[0].heading_level(), 3)
        self.assertEqual(parent.children[0].get_content(), "Child")
        # Metadata should be preserved
        self.assertEqual(parent.children[0].metadata, {"key": "value"})

    def test_add_child_level_adjustment_higher_level(self):
        """Adding child with level < parent level - adjusted downward."""
        parent = HeadingNode(HeadingBlock(level=3, content="Parent"))
        child = HeadingNode(HeadingBlock(level=1, content="Child"))
        child.metadata = {"author": "test", "tags": ["a", "b"]}

        parent.add_child(child)

        self.assertEqual(len(parent.children), 1)
        # Child should be adjusted to level 4
        self.assertIsInstance(parent.children[0], HeadingNode)
        self.assertEqual(parent.children[0].heading_level(), 4)
        # Content should be preserved
        self.assertEqual(parent.children[0].get_content(), "Child")
        # Metadata should be preserved
        self.assertEqual(
            parent.children[0].metadata,
            {"author": "test", "tags": ["a", "b"]},
        )

    def test_add_child_to_level_6_conversion_to_text(self):
        """Adding any heading to level 6 parent - converted to TextNode."""
        parent = HeadingNode(HeadingBlock(level=6, content="Level 6"))
        child = HeadingNode(HeadingBlock(level=7, content="Child"))
        child.metadata = {"key": "value"}

        parent.add_child(child)

        self.assertEqual(len(parent.children), 1)
        # Should be converted to TextNode
        self.assertIsInstance(parent.children[0], TextNode)
        # Content should include heading markers
        self.assertEqual(
            parent.children[0].get_content(), "######## Child"
        )
        # Metadata should be preserved
        self.assertEqual(parent.children[0].metadata, {"key": "value"})

    def test_add_child_metadata_preserved(self):
        """Verify complex metadata is preserved during adjustment."""
        parent = HeadingNode(HeadingBlock(level=2, content="Parent"))
        child = HeadingNode(HeadingBlock(level=1, content="Child"))
        child.metadata = {
            "simple": "value",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        parent.add_child(child)

        # All metadata should be preserved
        self.assertEqual(
            parent.children[0].metadata,
            {
                "simple": "value",
                "number": 42,
                "list": [1, 2, 3],
                "nested": {"key": "value"},
            },
        )

    def test_add_child_content_preserved(self):
        """Verify content is exactly preserved during adjustment."""
        parent = HeadingNode(HeadingBlock(level=4, content="Parent"))
        child = HeadingNode(
            HeadingBlock(level=2, content="Special #$% Content")
        )

        parent.add_child(child)

        # Content should be exactly the same
        self.assertEqual(
            parent.children[0].get_content(), "Special #$% Content"
        )


class TestPropagateContent(unittest.TestCase):
    """Test the propagate_content() function."""

    def setUp(self):
        """Set up a simple tree structure for testing."""
        header = HeaderBlock(content={"title": "Test"})
        heading1 = HeadingBlock(level=1, content="Chapter 1")
        text1 = TextBlock(content="Original text 1")
        heading2 = HeadingBlock(level=2, content="Section 1.1")
        text2 = TextBlock(content="Original text 2")

        blocks = [header, heading1, text1, heading2, text2]
        self.root = blocks_to_tree(blocks)
        if not self.root:
            raise ValueError("Could not build tree")

    def test_propagate_select_true_string(self):
        """select=True with string - should replace all text children."""

        def collect_func(node: HeadingNode) -> str:
            return f"Generated content for {node.get_content()}"

        propagate_content(
            self.root, collect_func, select=True, filter_func=lambda _: True
        )

        # Root should have new text child, heading child preserved
        root_children = self.root.children
        # Should have text + heading children
        text_children = [c for c in root_children if isinstance(c, TextNode)]
        heading_children = [
            c for c in root_children if isinstance(c, HeadingNode)
        ]

        # Only one text child (original replaced)
        self.assertEqual(len(text_children), 1)
        self.assertEqual(
            text_children[0].get_content(), "Generated content for Test"
        )
        # Heading children preserved
        self.assertGreater(len(heading_children), 0)

    def test_propagate_select_false_string(self):
        """select=False with string - should prepend text child."""
        # Get initial child count
        initial_count = self.root.count_children()

        def collect_func(node: HeadingNode) -> str:
            return f"Prepended text for {node.get_content()}"

        propagate_content(
            self.root,
            collect_func,
            select=False,
            filter_func=lambda _: True,
        )

        # Root should have one more child (prepended)
        self.assertEqual(self.root.count_children(), initial_count + 1)

        # First child should be the new text node
        first_child = self.root.children[0]
        self.assertIsInstance(first_child, TextNode)
        self.assertEqual(
            first_child.get_content(), "Prepended text for Test"
        )

    def test_propagate_collect_returns_textnode(self):
        """collect_func returns TextNode with metadata."""

        def collect_func(node: HeadingNode) -> TextNode:
            return TextNode.from_content(
                f"Custom node for {node.get_content()}",
                {"source": "propagate", "level": node.heading_level()},
            )

        propagate_content(
            self.root, collect_func, select=True, filter_func=lambda _: True
        )

        # Check that text children have metadata
        text_children = [
            c for c in self.root.children if isinstance(c, TextNode)
        ]
        self.assertEqual(len(text_children), 1)
        self.assertEqual(text_children[0].metadata["source"], "propagate")
        self.assertEqual(text_children[0].metadata["level"], 0)

    def test_propagate_with_filter(self):
        """filter_func selects specific headings."""
        # Only process headings at level 1
        def filter_func(node: HeadingNode) -> bool:
            return node.heading_level() == 1

        def collect_func(node: HeadingNode) -> str:
            return "Filtered content"

        propagate_content(
            self.root, collect_func, select=True, filter_func=filter_func
        )

        # Root (level 0) should not be filtered
        # Check level 1 heading
        level1_heading = [
            c for c in self.root.children if isinstance(c, HeadingNode)
        ][0]
        level1_texts = [
            c for c in level1_heading.children if isinstance(c, TextNode)
        ]

        # Level 1 heading should have filtered content
        self.assertEqual(len(level1_texts), 1)
        self.assertEqual(level1_texts[0].get_content(), "Filtered content")

    def test_propagate_preserves_heading_children(self):
        """Heading children are not removed when select=True."""

        def collect_func(node: HeadingNode) -> str:
            return "New text"

        # Count heading children before
        heading_children_before = len(self.root.get_heading_children())

        propagate_content(
            self.root, collect_func, select=True, filter_func=lambda _: True
        )

        # Heading children should still be there
        heading_children_after = len(self.root.get_heading_children())
        self.assertEqual(heading_children_before, heading_children_after)

    def test_propagate_nested_structure(self):
        """Works correctly on nested headings."""

        def collect_func(node: HeadingNode) -> str:
            return f"Level {node.heading_level()}"

        propagate_content(
            self.root, collect_func, select=True, filter_func=lambda _: True
        )

        # Check nested heading has correct content
        level1_heading = self.root.get_heading_children()[0]
        level2_heading = level1_heading.get_heading_children()[0]

        level2_texts = level2_heading.get_text_children()
        self.assertEqual(len(level2_texts), 1)
        self.assertEqual(level2_texts[0].get_content(), "Level 2")


class TestFoldTree(unittest.TestCase):
    """Test the fold_tree() function."""

    def setUp(self):
        """Set up a simple tree for testing."""
        header = HeaderBlock(content={"title": "Test"})
        heading1 = HeadingBlock(level=1, content="Chapter")
        text1 = TextBlock(content="First text")
        text2 = TextBlock(content="Second text")

        blocks = [header, heading1, text1, text2]
        self.root = blocks_to_tree(blocks)
        if not self.root:
            raise ValueError("Could not build tree")

    def test_fold_count_nodes(self):
        """Use fold to count total nodes."""

        def count_func(acc: int, node: MarkdownNode) -> int:
            return acc + 1

        total = fold_tree(self.root, count_func, 0)

        # Root + heading + 2 texts = 4 nodes
        self.assertEqual(total, 4)

    def test_fold_concatenate_content(self):
        """Use fold to concatenate all content."""

        def concat_func(acc: str, node: MarkdownNode) -> str:
            content = node.get_content()
            if acc:
                return acc + " | " + content
            return content

        result = fold_tree(self.root, concat_func, "")

        # Should have all content concatenated
        self.assertIn("Test", result)
        self.assertIn("Chapter", result)
        self.assertIn("First text", result)
        self.assertIn("Second text", result)

    def test_fold_with_pre_order(self):
        """Test with pre_order_traversal."""

        def list_append_func(
            acc: list[str], node: MarkdownNode
        ) -> list[str]:
            acc.append(node.get_content())
            return acc

        result = fold_tree(
            self.root, list_append_func, [], pre_order_traversal
        )

        # Pre-order: root first
        self.assertEqual(result[0], "Test")
        self.assertEqual(len(result), 4)

    def test_fold_with_post_order(self):
        """Test with post_order_traversal (default)."""

        def list_append_func(
            acc: list[str], node: MarkdownNode
        ) -> list[str]:
            acc.append(node.get_content())
            return acc

        result = fold_tree(self.root, list_append_func, [])

        # Post-order: root last
        self.assertEqual(result[-1], "Test")
        self.assertEqual(len(result), 4)


class TestTraverseTree(unittest.TestCase):
    """Test traverse_tree() without type filtering."""

    def setUp(self):
        """Set up a tree for testing."""
        header = HeaderBlock(content={"title": "Root"})
        heading1 = HeadingBlock(level=1, content="H1")
        text1 = TextBlock(content="T1")

        blocks = [header, heading1, text1]
        self.root = blocks_to_tree(blocks)
        if not self.root:
            raise ValueError("Could not build tree")

    def test_traverse_all_nodes(self):
        """Traverse all nodes without filter."""
        contents = traverse_tree(self.root, lambda n: n.get_content())

        # Should have all 3 nodes
        self.assertEqual(len(contents), 3)
        self.assertIn("Root", contents)
        self.assertIn("H1", contents)
        self.assertIn("T1", contents)

    def test_traverse_with_filter(self):
        """Use filter_func to select specific nodes."""
        # Only nodes with content starting with "T"
        filtered = traverse_tree(
            self.root,
            lambda n: n.get_content(),
            filter_func=lambda n: n.get_content().startswith("T"),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], "T1")

    def test_traverse_preorder_vs_postorder(self):
        """Verify order differences."""
        pre_order_result = traverse_tree(
            self.root,
            lambda n: n.get_content(),
            traversal_func=pre_order_traversal,
        )
        post_order_result = traverse_tree(
            self.root,
            lambda n: n.get_content(),
            traversal_func=post_order_traversal,
        )

        # Pre-order: root first
        self.assertEqual(pre_order_result[0], "Root")
        # Post-order: root last
        self.assertEqual(post_order_result[-1], "Root")


class TestTraversalDirect(unittest.TestCase):
    """Test pre_order_traversal and post_order_traversal directly."""

    def test_pre_order_traversal_order(self):
        """Verify pre-order visit sequence."""
        header = HeaderBlock(content={"title": "Root"})
        h1 = HeadingBlock(level=1, content="H1")
        t1 = TextBlock(content="T1")
        h2 = HeadingBlock(level=2, content="H2")
        t2 = TextBlock(content="T2")

        blocks = [header, h1, t1, h2, t2]
        root = blocks_to_tree(blocks)
        if not root:
            raise ValueError("Could not build tree")

        visited: list[str] = []

        def visit(node: MarkdownNode) -> None:
            visited.append(node.get_content())

        pre_order_traversal(root, visit)

        # Pre-order: parent before children
        # Root -> H1 -> T1 -> H2 -> T2
        self.assertEqual(visited[0], "Root")
        self.assertEqual(visited[1], "H1")
        # T1 comes before H2 (siblings in order)
        t1_index = visited.index("T1")
        h2_index = visited.index("H2")
        self.assertLess(t1_index, h2_index)

    def test_post_order_traversal_order(self):
        """Verify post-order visit sequence."""
        header = HeaderBlock(content={"title": "Root"})
        h1 = HeadingBlock(level=1, content="H1")
        t1 = TextBlock(content="T1")

        blocks = [header, h1, t1]
        root = blocks_to_tree(blocks)
        if not root:
            raise ValueError("Could not build tree")

        visited: list[str] = []

        def visit(node: MarkdownNode) -> None:
            visited.append(node.get_content())

        post_order_traversal(root, visit)

        # Post-order: children before parent
        # T1 -> H1 -> Root
        self.assertEqual(visited[-1], "Root")
        # H1 comes after T1
        self.assertEqual(visited[-2], "H1")


class TestHelperFunctions(unittest.TestCase):
    """Test utility helper functions."""

    def setUp(self):
        """Set up a tree with mixed nodes."""
        header = HeaderBlock(content={"title": "Root"})
        h1 = HeadingBlock(level=1, content="H1")
        t1 = TextBlock(content="T1")
        h2 = HeadingBlock(level=2, content="H2")
        t2 = TextBlock(content="T2")

        blocks = [header, h1, t1, h2, t2]
        self.root = blocks_to_tree(blocks)
        if not self.root:
            raise ValueError("Could not build tree")

    def test_get_text_nodes(self):
        """Collect all text nodes from tree."""
        from lmm.markdown.tree import get_text_nodes

        text_nodes = get_text_nodes(self.root)

        # Should have 2 text nodes
        self.assertEqual(len(text_nodes), 2)
        contents = [n.get_content() for n in text_nodes]
        self.assertIn("T1", contents)
        self.assertIn("T2", contents)

    def test_get_heading_nodes(self):
        """Collect all heading nodes from tree."""
        from lmm.markdown.tree import get_heading_nodes

        heading_nodes = get_heading_nodes(self.root)

        # Should have 3 heading nodes (Root, H1, H2)
        self.assertEqual(len(heading_nodes), 3)
        contents = [n.get_content() for n in heading_nodes]
        self.assertIn("Root", contents)
        self.assertIn("H1", contents)
        self.assertIn("H2", contents)

    def test_get_tree_info(self):
        """Generate info strings for all nodes."""
        from lmm.markdown.tree import get_tree_info

        info_list = get_tree_info(self.root)

        # Should have info for all nodes
        self.assertEqual(len(info_list), 5)
        # All should be strings
        for info in info_list:
            self.assertIsInstance(info, str)
        # Info should contain node type information
        info_str = "\n".join(info_list)
        self.assertIn("Header node", info_str)
        self.assertIn("Heading node", info_str)
        self.assertIn("Text node", info_str)


class TestMetadataInheritance(unittest.TestCase):
    """Test TextNode metadata inheritance behavior."""

    def test_textnode_metadata_from_parent(self):
        """Test automatic parent inheritance in get_metadata_for_key()."""
        header = HeaderBlock(content={"title": "Test"})
        heading = HeadingBlock(level=1, content="Chapter")
        text = TextBlock(content="Text content")

        root = blocks_to_tree([header, heading, text])
        if not root:
            raise ValueError("Could not build tree")

        # Add metadata to heading
        heading_node = root.get_heading_children()[0]
        heading_node.metadata = {"author": "Test Author", "version": 1}

        # Get text node
        text_node = heading_node.get_text_children()[0]

        # text_node.get_metadata_for_key() should inherit from parent
        self.assertEqual(
            text_node.get_metadata_for_key("author"), "Test Author"
        )
        self.assertEqual(text_node.get_metadata_for_key("version"), 1)

    def test_textnode_metadata_vs_fetch(self):
        """Show difference between get_metadata_for_key() and fetch_metadata_for_key()."""
        header = HeaderBlock(content={"title": "Test"})
        heading = HeadingBlock(level=1, content="Chapter")
        text = TextBlock(content="Text")

        root = blocks_to_tree([header, heading, text])
        if not root:
            raise ValueError("Could not build tree")

        heading_node = root.get_heading_children()[0]
        heading_node.metadata = {"key": "heading_value"}

        text_node = heading_node.get_text_children()[0]
        text_node.metadata = {}

        # get_metadata_for_key() automatically checks parent
        self.assertEqual(
            text_node.get_metadata_for_key("key"), "heading_value"
        )

        # TextNode.get_metadata_for_key() implements automatic parent
        # inheritance directly, unlike MarkdownNode base class


class TestHasMetadataKey(unittest.TestCase):
    """Test has_metadata_key() with inherit parameter."""

    def setUp(self):
        """Set up a tree with metadata."""
        header = HeaderBlock(content={"title": "Test", "doc_meta": "value"})
        heading = HeadingBlock(level=1, content="Chapter")
        text = TextBlock(content="Text")

        self.root = blocks_to_tree([header, heading, text])
        if not self.root:
            raise ValueError("Could not build tree")

        # Add metadata to heading
        heading_node = self.root.get_heading_children()[0]
        heading_node.metadata = {"heading_meta": "value"}

        self.text_node = heading_node.get_text_children()[0]

    def test_has_metadata_key_inherit_true(self):
        """Test inheritance traversal with inherit=True."""
        # Text node should find metadata in parent
        self.assertTrue(
            self.text_node.has_metadata_key("heading_meta", inherit=True)
        )
        # And also in root (needs include_header=True to reach header metadata)
        self.assertTrue(
            self.text_node.has_metadata_key(
                "doc_meta", inherit=True, include_header=True
            )
        )


    def test_has_metadata_key_inherit_false(self):
        """Test non-inheriting behavior with inherit=False."""
        # Text node has no metadata of its own
        self.assertFalse(
            self.text_node.has_metadata_key("heading_meta", inherit=False)
        )
        self.assertFalse(
            self.text_node.has_metadata_key("doc_meta", inherit=False)
        )


if __name__ == "__main__":
    unittest.main(
        argv=["first-arg-is-ignored"], exit=False
    )  # Using exit=False for
    # environments like notebooks
