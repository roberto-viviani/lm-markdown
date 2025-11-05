"""Test scanutils module"""

# flake8: noqa
# pyright: basic
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false

import unittest
from typing import Sequence

from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    TextNode,
    load_tree,
)
from lmm.scan.scanutils import (
    post_order_hashed_aggregation,
    aggregate_hash,
)
from lmm.scan.scan_keys import TXTHASH_KEY
from lmm.utils.logging import LoglistLogger


def word_count_aggregate(content: str) -> str:
    """
    Aggregate function that counts words in the input string.

    Args:
        content: The string content to count words in

    Returns:
        A string like "There are xxx words"
    """
    if not content or not content.strip():
        return ""

    word_count = len(content.split())
    return f"There are {word_count} words"


class TestPostOrderHashedAggregationBasic(unittest.TestCase):
    """Test basic functionality of post_order_hashed_aggregation"""

    def test_simple_tree_with_text(self):
        """Test aggregation on simple tree with header, heading, and text"""
        markdown = """---
title: Test Document
---

# Introduction

This is some sample text with multiple words here.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # Apply aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root has only one child (the Introduction heading), so root is skipped
        # The aggregation should be on the Introduction heading instead
        heading_node = root.get_heading_children()[0]
        self.assertIsNotNone(heading_node)
        self.assertIn(OUTPUT_KEY, heading_node.metadata)
        # "This is some sample text with multiple words here." = 9 words
        self.assertEqual(
            heading_node.metadata[OUTPUT_KEY], "There are 9 words"
        )

        # Check that hash was stored on the heading
        self.assertIn(TXTHASH_KEY, heading_node.metadata)

    def test_nested_headings_with_text(self):
        """Test aggregation with nested heading structure"""
        markdown = """---
title: Document
---

# Chapter One

First paragraph with some words.

## Section One Point One

Second paragraph here too.

# Chapter Two

Third paragraph content.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root should aggregate all content
        self.assertIn(OUTPUT_KEY, root.metadata)
        # The function aggregates synthesized outputs from children, not raw text
        # Chapter One includes its text + Section output: "First paragraph with some words.\n\nThere are 4 words" = 9 words
        # Chapter Two: "There are 3 words"
        # Root aggregates these outputs: "There are 9 words\n\nThere are 3 words" = 8 words
        self.assertEqual(
            root.metadata[OUTPUT_KEY], "There are 8 words"
        )


class TestPostOrderHashedAggregationEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def test_only_header(self):
        """Test with markdown containing only a header"""
        markdown = """---
title: Only Header
description: A document with just metadata
---"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # Should not crash
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Since there's no text content, output_key should not be set
        # (post_order aggregation does not invoke aggregation function
        # when there is no input)
        self.assertNotIn(OUTPUT_KEY, root.metadata)

    def test_header_and_heading_no_text(self):
        """Test with header and heading but no text content"""
        markdown = """---
title: Document
---

# Empty Chapter

## Empty Section
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # Should not crash
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Since there's no text content, output_key should not be set
        self.assertNotIn(OUTPUT_KEY, root.metadata)


class TestPostOrderHashedAggregationHashing(unittest.TestCase):
    """Test hashing behavior"""

    def test_hashed_caching_same_content(self):
        """Test that with hashing enabled, same content doesn't recompute"""
        markdown = """---
title: Test
---

# Heading

Some text content here.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # First aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]
        first_result = heading_node.metadata.get(OUTPUT_KEY)
        first_hash = heading_node.metadata.get(TXTHASH_KEY)

        self.assertEqual(first_result, "There are 4 words")
        self.assertIsNotNone(first_hash)

        # Second aggregation with same content should use cache
        # To verify, we'll use a function that would produce different output
        def different_aggregate(content: str) -> str:
            return "DIFFERENT OUTPUT"

        post_order_hashed_aggregation(
            root,
            different_aggregate,
            OUTPUT_KEY,
            hashed=True,
            hash_key=TXTHASH_KEY,
        )

        # Result should be unchanged (cached)
        self.assertEqual(
            heading_node.metadata.get(OUTPUT_KEY), first_result
        )
        self.assertEqual(
            heading_node.metadata.get(TXTHASH_KEY), first_hash
        )

    def test_hashed_recomputes_on_content_change(self):
        """Test that changing content triggers recomputation with hashing"""
        markdown = """---
title: Test
---

# Heading

Original text content.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # First aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]
        first_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(first_result, "There are 3 words")

        # Modify the text content
        text_node = (
            root.get_text_children()[0]
            if root.get_text_children()
            else None
        )
        if text_node is None:
            # Navigate through children to find text node
            for child in root.children:
                if isinstance(child, HeadingNode):
                    text_nodes = child.get_text_children()
                    if text_nodes:
                        text_node = text_nodes[0]
                        break

        self.assertIsNotNone(text_node)
        if text_node:
            text_node.set_content(
                "Modified text with many more words now."
            )

        # Rerun aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Result should reflect new content
        second_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(second_result, "There are 7 words")
        self.assertNotEqual(first_result, second_result)

    def test_hashed_recomputes_on_deleted_metadata(self):
        """Test that deleted metadata triggers recomputation"""
        markdown = """---
title: Test
---

# Heading

Original text content.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # First aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=False
        )

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]
        first_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(first_result, "There are 3 words")

        # Modify the text content
        text_node = (
            root.get_text_children()[0]
            if root.get_text_children()
            else None
        )
        if text_node is None:
            # Navigate through children to find text node
            for child in root.children:
                if isinstance(child, HeadingNode):
                    text_nodes = child.get_text_children()
                    if text_nodes:
                        text_node = text_nodes[0]
                        break

        self.assertIsNotNone(text_node)
        if text_node:
            text_node.set_content(
                "Modified text with many more words now."
            )

        # Clear the previous aggregation result to force recomputation
        if OUTPUT_KEY in heading_node.metadata:
            del heading_node.metadata[OUTPUT_KEY]
        if TXTHASH_KEY in heading_node.metadata:
            del heading_node.metadata[TXTHASH_KEY]

        # Rerun aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=False
        )

        # Result should reflect new content
        second_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(second_result, "There are 7 words")
        self.assertNotEqual(first_result, second_result)

    def test_hashed_no_recomputes_on_frozen(self):
        """Test that changing content does not trigger recomputation
        with hashing, if the node is frozen"""
        markdown = """---
title: Test
---

# Heading

Original text content.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # First aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]
        first_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(first_result, "There are 3 words")

        # Modify the text content
        text_node = (
            root.get_text_children()[0]
            if root.get_text_children()
            else None
        )
        if text_node is None:
            # Navigate through children to find text node
            for child in root.children:
                if isinstance(child, HeadingNode):
                    text_nodes = child.get_text_children()
                    if text_nodes:
                        text_node = text_nodes[0]
                        break

        self.assertIsNotNone(text_node)
        if text_node:
            text_node.set_content(
                "Modified text with many more words now."
            )

        from lmm.scan.scan_keys import FREEZE_KEY  # fmat: skip

        text_node.get_parent().metadata[FREEZE_KEY] = True

        # Rerun aggregation
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Result should reflect new content
        second_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(second_result, first_result)

    def test_non_hashed_only_computes_when_missing(self):
        """Test that with hashed=False, only computes when output is missing"""
        markdown = """---
title: Test
---

# Heading

Some text here.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # First aggregation with hashed=False
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=False
        )

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]
        first_result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(first_result, "There are 3 words")

        # Hash should not be stored when hashed=False
        self.assertNotIn(TXTHASH_KEY, heading_node.metadata)

        # Second aggregation with different function should NOT recompute
        def different_aggregate(content: str) -> str:
            return "DIFFERENT OUTPUT"

        post_order_hashed_aggregation(
            root, different_aggregate, OUTPUT_KEY, hashed=False
        )

        # Result should be unchanged (not recomputed)
        self.assertEqual(
            heading_node.metadata.get(OUTPUT_KEY), first_result
        )

    def test_non_hashed_computes_when_empty(self):
        """Test that with hashed=False, computes if output value is empty"""
        markdown = """---
title: Test
---

# Heading

Some text here.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # Root has only one child, so heading gets the aggregation
        heading_node = root.get_heading_children()[0]

        # Pre-set an empty value on the heading node
        heading_node.metadata[OUTPUT_KEY] = ""

        # Aggregation should compute since value is empty
        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=False
        )

        result = heading_node.metadata.get(OUTPUT_KEY)
        self.assertEqual(result, "There are 3 words")


class TestPostOrderHashedAggregationComplexCases(unittest.TestCase):
    """Test more complex scenarios"""

    def test_multiple_text_nodes_under_heading(self):
        """Test aggregation with multiple text nodes under a heading"""
        markdown = """---
title: Test
---

# Chapter

First text block here.

Second text block here.

Third text block.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # Root has only one child (Chapter heading), so aggregation is on the heading
        heading_node = root.get_heading_children()[0]
        # Should aggregate all text blocks
        # "First text block here" (4) + "Second text block here" (4) +
        # "Third text block" (3) = 11 words
        self.assertEqual(
            heading_node.metadata[OUTPUT_KEY], "There are 11 words"
        )

    def test_deeply_nested_structure(self):
        """Test aggregation with deeply nested heading structure"""
        markdown = """---
title: Book
---

# Part One

## Chapter One

### Section A

Content in section A.

### Section B

Content in section B.

## Chapter Two

Content in chapter two.

# Part Two

Content in part two.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # The function aggregates synthesized outputs from children
        # Each leaf produces "There are 4 words"
        # These get aggregated up the tree, counting words in the synthetic outputs
        # Root aggregates: "There are 4 words\n\nThere are 4 words\n\nThere are 4 words\n\nThere are 4 words"
        # But actually the aggregation is more complex with nested structure
        # Based on actual behavior: 8 words
        self.assertEqual(
            root.metadata[OUTPUT_KEY], "There are 8 words"
        )

    def test_custom_hash_key(self):
        """Test using a custom hash key"""
        markdown = """---
title: Test
---

# Heading

Some text content.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"
        CUSTOM_HASH_KEY = "custom_hash"

        post_order_hashed_aggregation(
            root,
            word_count_aggregate,
            OUTPUT_KEY,
            hashed=True,
            hash_key=CUSTOM_HASH_KEY,
        )

        # Root has only one child, so aggregation is on the heading
        heading_node = root.get_heading_children()[0]
        # Should use custom hash key
        self.assertIn(CUSTOM_HASH_KEY, heading_node.metadata)
        self.assertNotIn(TXTHASH_KEY, heading_node.metadata)
        self.assertEqual(
            heading_node.metadata[OUTPUT_KEY], "There are 3 words"
        )

    def test_empty_aggregate_result(self):
        """Test when aggregate function returns empty string"""
        markdown = """---
title: Test
---

# Heading

Some text.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        # Aggregate function that always returns empty
        def empty_aggregate(content: str) -> str:
            return ""

        post_order_hashed_aggregation(
            root, empty_aggregate, OUTPUT_KEY, hashed=True
        )

        # When aggregate returns empty, output_key should not be set
        self.assertNotIn(OUTPUT_KEY, root.metadata)


class TestPostOrderHashedAggregationSingleParent(unittest.TestCase):
    """Test the special case where heading has only one heading child"""

    def test_heading_with_single_heading_child(self):
        """Test that heading with single heading child doesn't aggregate"""
        markdown = """---
title: Document
---

# Chapter

## Section

Text content here.
"""
        logger = LoglistLogger()
        root = load_tree(markdown, logger)
        self.assertIsNotNone(root)
        if root is None:
            return

        OUTPUT_KEY = "word_summary"

        post_order_hashed_aggregation(
            root, word_count_aggregate, OUTPUT_KEY, hashed=True
        )

        # The "Chapter" heading should NOT have aggregation
        # because it has only one child which is also a heading
        chapter_node = (
            root.get_heading_children()[0]
            if root.get_heading_children()
            else None
        )
        self.assertIsNotNone(chapter_node)
        if chapter_node:
            # Chapter node should not have the output key
            self.assertNotIn(OUTPUT_KEY, chapter_node.metadata)

        # Root also should not have aggregation because it has only one heading child
        # The Section (child of Chapter) will have the aggregation
        self.assertNotIn(OUTPUT_KEY, root.metadata)

        # Find the Section node which should have the aggregation
        if chapter_node:
            section_node = (
                chapter_node.get_heading_children()[0]
                if chapter_node.get_heading_children()
                else None
            )
            if section_node:
                self.assertIn(OUTPUT_KEY, section_node.metadata)
                self.assertEqual(
                    section_node.metadata[OUTPUT_KEY],
                    "There are 3 words",
                )


title_block = "---\ntitle: document\n---\n" ""
heading_block = "# first heading\n\n"
text_block = "Text block.\n\n"
skip_metadata = "---\nskip: True\n---\n"

from lmm.markdown.parse_markdown import blocklist_copy
from lmm.markdown.tree import tree_to_blocks, serialize_tree


class TestSkippedAggregation(unittest.TestCase):

    def test_skipped_text(self):
        # Test that post_order_hashed_aggregation skips aggregating
        # from text nodes that do not satify a predicate function
        # passed in in the filter_func
        markdown = (
            title_block
            + heading_block
            + "First block.\n\n"
            + skip_metadata
            + "Second block.\n\n"
            + "Third block.\n\n"
        )

        root = load_tree(markdown, LoglistLogger())
        self.assertFalse(
            root.children[0]
            .children[0]
            .get_metadata_for_key('skip', False)
        )
        self.assertTrue(
            root.children[0]
            .children[1]
            .get_metadata_for_key('skip', False)
        )
        self.assertFalse(
            root.children[0]
            .children[2]
            .get_metadata_for_key('skip', False)
        )

        OUTPUT_KEY = "text_aggregate"

        post_order_hashed_aggregation(
            root,
            lambda x: x,
            OUTPUT_KEY,
            False,
            filter_func=lambda x: not x.get_metadata_for_key(
                "skip", False
            ),
        )

        aggregate = root.children[0].get_metadata_for_key(OUTPUT_KEY)
        self.assertIn("First block", aggregate)
        self.assertIn("Third block", aggregate)
        self.assertNotIn("Second block", aggregate)

    def test_skipped_heading(self):
        # test that aggregation is not computed on a hading with
        # a skipped predicate

        markdown = (
            title_block
            + skip_metadata
            + heading_block
            + "First block.\n\n"
            + "Second block.\n\n"
            + "Third block.\n\n"
            + "# Second heading\n\n"
            + "First block.\n\n"
            + "Second block.\n\n"
            + "Third block.\n\n"
        )
        root = load_tree(markdown, LoglistLogger())

        OUTPUT_KEY = "text_aggregate"

        post_order_hashed_aggregation(
            root,
            lambda x: x,
            OUTPUT_KEY,
            False,
            filter_func=lambda x: not x.get_metadata_for_key(
                "skip", False
            ),
        )

        aggregate = root.children[0].get_metadata_for_key(
            OUTPUT_KEY, None
        )
        self.assertIsNone(aggregate)
        aggregate = root.children[1].get_metadata_for_key(
            OUTPUT_KEY, None
        )
        self.assertTrue(aggregate)

        # repeat for other order
        markdown = (
            title_block
            + heading_block
            + "First block.\n\n"
            + "Second block.\n\n"
            + "Third block.\n\n"
            + skip_metadata
            + "# Second heading\n\n"
            + "First block.\n\n"
            + "Second block.\n\n"
            + "Third block.\n\n"
        )
        root = load_tree(markdown, LoglistLogger())

        post_order_hashed_aggregation(
            root,
            lambda x: x,
            OUTPUT_KEY,
            False,
            filter_func=lambda x: not x.get_metadata_for_key(
                "skip", False
            ),
        )

        aggregate = root.children[0].get_metadata_for_key(
            OUTPUT_KEY, None
        )
        self.assertTrue(aggregate)
        aggregate = root.children[1].get_metadata_for_key(
            OUTPUT_KEY, None
        )
        self.assertIsNone(aggregate)


from lmm.utils.hash import base_hash


class TestAggregateHash(unittest.TestCase):

    def test_empty_hash(self):
        logger = LoglistLogger()

        root = load_tree("# A title", logger)
        self.assertEqual(logger.count_logs(), 0)
        hash = aggregate_hash(root)
        self.assertEqual(hash, "")

    def test_hash_textnode(self):
        node = TextNode.from_content("test")
        hash = aggregate_hash(node)
        self.assertEqual(hash, base_hash("test"))

    def test_hash(self):
        # garden path
        logger = LoglistLogger()
        text = """
---
title: document
---

# First title

This is the first piece of text

## Second title

More text.
"""

        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        first_heading = root.get_heading_children()[0]
        second_heading = first_heading.get_heading_children()[0]

        hash_leaf = aggregate_hash(second_heading)
        self.assertEqual(hash_leaf, base_hash("More text."))

        hash_head = aggregate_hash(first_heading)
        self.assertEqual(
            hash_head,
            base_hash("This is the first piece of text" + hash_leaf),
        )

    def test_hash_filtered(self):
        logger = LoglistLogger()
        text = """
---
title: document
---

# First title

This is the first piece of text

---
skip: True
---
## Second title

More text.
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_head = aggregate_hash(
            root.children[0],
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )
        self.assertEqual(
            hash_head, base_hash("This is the first piece of text")
        )

    def test_hash_filtered_textnode(self):
        logger = LoglistLogger()
        text = """
---
title: document
---

# First title

This is the first piece of text

## Second title

---
skip: True
---
More text
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_head = aggregate_hash(
            root.children[0],
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )
        self.assertEqual(
            hash_head, base_hash("This is the first piece of text")
        )

    def test_hash_filtered_texts(self):
        logger = LoglistLogger()
        text = """
---
title: document
---

# First title

This is the first piece of text

## Second title

First text block.

---
skip: True
---
More text.

Third text block.
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_head = aggregate_hash(
            root.children[0],
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )

        text = """
---
title: document
---

# First title

This is the first piece of text

## Second title

First text block.

---
skip: True
---
More text was changed.

Third text block.
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_changed = aggregate_hash(
            root.children[0],
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )
        self.assertEqual(hash_head, hash_changed)

    def test_hash_skipall(self):
        logger = LoglistLogger()
        text = """
---
title: document
skip: True
---

# First title

This is the first piece of text

---
skip: True
---
## Second title

More text.
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_head = aggregate_hash(
            root,
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )
        self.assertEqual(hash_head, "")

    def test_hash_skipall2(self):
        logger = LoglistLogger()
        text = """
---
title: document
---

---
skip: True
---
# First title

This is the first piece of text

---
skip: True
---
## Second title

More text.
"""
        root = load_tree(text, logger)
        self.assertTrue(logger.count_logs() == 0)

        hash_head = aggregate_hash(
            root,
            lambda x: not x.fetch_metadata_for_key(
                'skip', True, False
            ),
        )
        self.assertEqual(hash_head, "")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
