""" Test tree """
# flake8: noqa

import unittest

# Attempt to import the functions to be tested
from lmm.markdown import (
    parse_markdown_text,
    ErrorBlock,
    HeaderBlock,
    TextBlock,
    MetadataBlock,
)
from lmm.markdown.tree import *

class TestTreeConstruction(unittest.TestCase):

    def test_load_tree(self):
        root = load_tree("./tests/test_markdown.md")

        self.assertIsNotNone(root)


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
        blocks = tree_to_blocks(root)

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertEqual(blocks[0].get_key("title"), "This")  # type: ignore
        self.assertIsInstance(blocks[1], HeadingBlock)


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
        self.test_metadata = {
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
        original.metadata["nested"]["key"] = "modified"
        # This assertion shows that nested metadata IS now independent
        self.assertNotEqual(
            original.metadata["nested"]["key"],
            copy.metadata["nested"]["key"],
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


if __name__ == "__main__":
    unittest.main(
        argv=["first-arg-is-ignored"], exit=False
    )  # Using exit=False for
    # environments like notebooks
