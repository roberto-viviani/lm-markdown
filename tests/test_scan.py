"""Test scan.py"""

import unittest

from lmm.scan import scan, markdown_scan
from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    parse_markdown_text,
)

test_md_only_heading = """# This is my title"""
test_md_no_header_text = (
    """This is a simple file with just text: no header, no heading."""
)
test_md_no_header_heading = """
# A main heading {class = any}

This is a markdown file with a heading, no header.
"""
test_md_no_title = """
---
author: john
...

This is a markdown file with a starting metadata block that has no title tag.
"""


class TestHeaders(unittest.TestCase):

    def test_empty(self):
        blocks: list[Block] = scan([])

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore

    def test_empty2(self):
        blocks: list[Block] = parse_markdown_text("")
        blocks = scan(blocks)

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore

    def test_only_heading(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_only_heading
        )
        blocks = scan(blocks)

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "This is my title")  # type: ignore

    def test_no_header_text(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_no_header_text
        )
        blocks = scan(blocks)

        self.assertTrue(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore
        self.assertEqual(
            blocks[1].get_content(), test_md_no_header_text
        )

    def test_md_no_header_heading(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_no_header_heading
        )
        blocks = scan(blocks)

        self.assertTrue(len(blocks), 3)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "A main heading")  # type: ignore
        self.assertEqual(
            blocks[-1].get_content(),
            "This is a markdown file with a heading, no header.",
        )

    def test_md_no_title(self):
        blocks: list[Block] = parse_markdown_text(test_md_no_title)
        blocks = scan(blocks)

        self.assertTrue(len(blocks), 3)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore
        self.assertEqual(blocks[0].content['author'], "john")  # type: ignore
        self.assertEqual(
            blocks[-1].get_content(),
            "This is a markdown file with a starting metadata block that has no title tag.",
        )

    def test_invalid_markdown(self):
        blocks: list[Block] = markdown_scan("nonexistent.md")
        self.assertEqual(len(blocks), 0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        unittest.main()
