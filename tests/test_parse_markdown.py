# flake8: noqa

"""Unit tests for parse_markdown_text function from
lmm.markdown.parse_markdown module."""

import unittest
from lmm.markdown.parse_markdown import (
    parse_markdown_text,
    serialize_blocks,
    blocklist_haserrors,
    blocklist_errors,
    blocklist_copy,
    blocklist_map,
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
)


class TestMardownBlocks(unittest.TestCase):

    def test_construction_text_empty(self):
        text = ""
        block = TextBlock(content=text)
        self.assertEqual(block.get_content(), text)
        self.assertEqual(block.get_word_count(), 0)
        self.assertTrue(block.is_empty())
        parse = block.serialize()
        self.assertEqual(
            text,
            serialize_blocks(parse_markdown_text(parse)),
        )
        block.extend("Part of new")
        self.assertEqual(block.get_content(), "Part of new")

    def test_construction_text_regular(self):
        text = "Content of text block\non two lines"
        block = TextBlock(content=text)
        self.assertEqual(block.get_content(), text)
        self.assertEqual(block.get_word_count(), len(text.split()))
        self.assertFalse(block.is_empty())
        parse = block.serialize()
        self.assertEqual(
            text,
            serialize_blocks(parse_markdown_text(parse)),
        )
        block.extend("Part of new")
        self.assertEqual(
            block.get_content(), "\n\n".join([text, "Part of new"])
        )

    def test_construction_heading(self):
        text = "A heading"
        block = HeadingBlock(content=text, level=2)
        self.assertEqual(block.get_content(), text)
        parse = block.serialize()
        self.assertEqual(
            "## " + text + "\n",
            serialize_blocks(parse_markdown_text(parse)),
        )

    def test_construction_heading_empty(self):
        text = ""
        block = HeadingBlock(content=text, level=2)
        self.assertEqual(block.get_content(), text)
        parse = block.serialize()
        self.assertEqual(
            "## " + text + "\n",
            parse,
        )
        self.assertIsInstance(
            parse_markdown_text(parse)[0], ErrorBlock
        )

    def test_construction_metadata(self):
        data = {'title': "Title"}
        block = MetadataBlock(content=data)
        self.assertEqual(block.get_key('title'), "Title")
        self.assertEqual(block.get_key('titles'), "")
        self.assertDictEqual(block.get_content(), data)
        parse = block.serialize()
        self.assertEqual(
            parse,
            serialize_blocks(parse_markdown_text(parse)),
        )

    def test_construction_metadata_empty(self):
        data = {}
        block = MetadataBlock(content=data)
        self.assertDictEqual(block.get_content(), data)
        parse = block.serialize()
        self.assertEqual(parse, "---\n---\n")
        self.assertIsInstance(
            parse_markdown_text(parse)[0], ErrorBlock
        )

    def test_construction_header_invalid(self):
        text = "---\n1: first line\n---"
        # expected behaviour: invalid dict is set to private_,
        # default dict created for header
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertIsInstance(block, HeaderBlock)
        self.assertDictEqual(block.get_content(), {'title': "Title"})
        self.assertDictEqual(block.private_[0], {1: "first line"})

    def test_construction_metadata_invalid(self):
        text = "---\n1: first block\n---\n\n---\n2: second block\n---"
        # expected behaviour: invalid dict is set to private,
        # empty dict in metadata
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 2)
        block = blocks[1]
        self.assertIsInstance(block, MetadataBlock)
        self.assertDictEqual(block.get_content(), {})
        self.assertDictEqual(block.private_[0], {2: "second block"})


class TestBlocklist(unittest.TestCase):

    def test_construction_error(self):
        text = "---\na simple literal\n---"
        # expected behaviour: metadata of literal not allowed by
        # LM markdown parser, even if legal for yaml
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertIsInstance(block, ErrorBlock)
        self.assertTrue(block.serialize().startswith("** ERROR:"))
        newtext = serialize_blocks(blocks).replace(
            "a simple literal", "lit: a simple literal"
        )
        # expected behaviour: after correction, error message removed
        # when parsing, correct object created
        newblocks = parse_markdown_text(newtext)
        self.assertIsInstance(newblocks[0], HeaderBlock)
        self.assertEqual(
            newblocks[0].get_key('lit'), "a simple literal"
        )

    def test_blocklist_error(self):
        text = "---\na simple literal\n---"
        # expected behaviour: metadata of literal not allowed by
        # LM markdown parser, even if legal for yaml
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 1)
        # reports errors
        self.assertTrue(blocklist_haserrors(blocks))
        self.assertEqual(len(blocklist_errors(blocks)), 1)

    def test_blocklist_copy(self):
        text = "---\ntitle: Title\n---\nsome text."
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIsInstance(blocks[1], TextBlock)
        blockscopy = blocklist_copy(blocks)
        blockscopy[1].content = "new content"
        self.assertNotEqual(
            blocks[1].get_content(), blockscopy[1].get_content()
        )

    def test_blocklist_map(self):
        text = "---\ntitle: Title\n---\nsome text"
        blocks = parse_markdown_text(text)
        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIsInstance(blocks[1], TextBlock)
        newblocks = blocklist_map(
            blocks,
            lambda x: TextBlock.from_text(
                x.get_content() + ", really!"
            ),
            lambda x: isinstance(x, TextBlock),
        )
        self.assertEqual(len(blocks), len(newblocks))
        self.assertEqual(
            newblocks[1].get_content(), "some text, really!"
        )
        self.assertNotEqual(
            blocks[1].get_content(), newblocks[1].get_content()
        )

    def test_blocklist_map_example(self):
        blocks = [
            MetadataBlock._from_dict({'title': "Title"}),
            TextBlock.from_text("A text block"),
        ]
        newblocks = blocklist_map(
            blocks,
            lambda x: TextBlock.from_text(
                x.get_content() + ", really!"
            ),
            lambda x: isinstance(x, TextBlock),
        )
        text = serialize_blocks(newblocks)
        self.assertNotEqual(text, serialize_blocks(blocks))


class TestParseMarkdownText(unittest.TestCase):
    """Test cases for the parse_markdown_text function."""

    def test_empty_content(self):
        """Test parsing empty content returns empty list."""
        result = parse_markdown_text("")
        self.assertEqual(result, [])

    def test_whitespace_only_content(self):
        """Test parsing whitespace-only content returns empty list."""
        result = parse_markdown_text("   \n\n  \t  \n")
        self.assertEqual(result, [])

    def test_simple_text_block(self):
        """Test parsing simple text content."""
        content = "This is a simple text block."
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextBlock)
        self.assertEqual(
            result[0].content, "This is a simple text block."
        )
        self.assertEqual(content, serialize_blocks(result))

    def test_multiple_text_blocks(self):
        """Test parsing multiple text blocks separated by blank lines."""
        content = """First paragraph.

Second paragraph.

Third paragraph."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 3)
        for block in result:
            self.assertIsInstance(block, TextBlock)
        self.assertEqual(result[0].content, "First paragraph.")
        self.assertEqual(result[1].content, "Second paragraph.")
        self.assertEqual(result[2].content, "Third paragraph.")
        self.assertEqual(content, serialize_blocks(result))

    def test_simple_heading(self):
        """Test parsing simple heading blocks."""
        content = "# First level heading"
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeadingBlock)
        if isinstance(result[0], HeadingBlock):
            self.assertEqual(result[0].level, 1)
            self.assertEqual(result[0].content, "First level heading")
        self.assertEqual(content, serialize_blocks(result).strip())

    def test_multiple_heading_levels(self):
        """Test parsing headings at different levels."""
        content = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 6)
        for i, block in enumerate(result):
            self.assertIsInstance(block, HeadingBlock)
            if isinstance(block, HeadingBlock):
                self.assertEqual(block.level, i + 1)
                self.assertEqual(block.content, f"Level {i + 1}")
        self.assertEqual(
            content,
            serialize_blocks(result).replace("\n\n", "\n").strip(),
        )

    def test_heading_with_attributes(self):
        """Test parsing heading with attributes."""
        content = '## Second level heading {class = "elicits a warning in scan_rag"}'
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeadingBlock)
        self.assertEqual(result[0].level, 2)
        self.assertEqual(result[0].content, "Second level heading")
        self.assertEqual(
            result[0].attributes,
            'class = "elicits a warning in scan_rag"',
        )
        self.assertEqual(content, serialize_blocks(result).strip())

    def test_empty_heading_error(self):
        """Test that empty headings generate error blocks."""
        content = "## "
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertIn("Empty heading content", result[0].content)

    def test_heading_with_attributes_but_no_text_error(self):
        """Test heading with attributes but no text generates error."""
        content = "## {class = test}"
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertIn(
            "attributes, but there is no heading text",
            result[0].content,
        )

    def test_header_block_first_metadata(self):
        """Test that first metadata block becomes header block."""
        content = """---
title: My Document
author: John Doe
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertEqual(result[0].content["title"], "My Document")
        self.assertEqual(result[0].content["author"], "John Doe")

    def test_header_block_with_comment(self):
        """Test header block with comment."""
        content = """---  # This is a comment
title: My Document
author: John Doe
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertEqual(result[0].comment, "This is a comment")
        self.assertEqual(result[0].content["title"], "My Document")

    def test_metadata_block_not_first(self):
        """Test that metadata blocks after the first are MetadataBlock, not HeaderBlock."""
        content = """---
title: My Document
---

Some text.

--- # questions already in original
~questions: The question of the day
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertIsInstance(result[1], TextBlock)
        self.assertIsInstance(result[2], MetadataBlock)
        self.assertEqual(
            result[2].comment, "questions already in original"
        )
        self.assertEqual(content, serialize_blocks(result).strip())

    def test_metadata_block_with_ellipsis_end(self):
        """Test metadata block ending with ellipsis."""
        content = """---
first: 1
second:
  word: my word
  number: 1
..."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(
            result[0], HeaderBlock
        )  # First block becomes header
        self.assertEqual(result[0].content["first"], 1)
        self.assertDictEqual(
            result[0].content["second"],
            {"word": "my word", "number": 1},
        )

    def test_unclosed_metadata_block_error(self):
        """Test that unclosed metadata blocks generate errors."""
        content = """---
title: Unclosed
author: Test"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertIn("Unclosed metadata block", result[0].content)
        self.assertEqual(result[0].origin.strip(), content)

    def test_invalid_yaml_in_metadata_error(self):
        """Test that invalid YAML in metadata generates error."""
        content = """---
title: Test
invalid: [unclosed list
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertIn("YAML parse error", result[0].content)
        self.assertEqual(result[0].origin.strip(), content)

    def test_non_conformant_dictionary_error(self):
        """Test non-conformant dictionary"""
        content = """
--- # non-conformant dictionary
1: integer key
2: another integer
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertEqual(result[0].get_key("title", ""), "Title")
        self.assertDictEqual(
            result[0].private_[0],
            {1: "integer key", 2: "another integer"},
        )

    def test_mixed_content_types(self):
        """Test parsing mixed content with headers, metadata, headings, and text."""
        content = """---
title: Mixed Content Test
---

This is some text after the header.

# First Level Heading

Text under heading.

---
~textid: test123
~questions: What is this?
---

More text after metadata.

## Second Level Heading

Final text block."""
        result = parse_markdown_text(content)

        # Should have: HeaderBlock, TextBlock, HeadingBlock, TextBlock, MetadataBlock, TextBlock, HeadingBlock, TextBlock
        self.assertEqual(len(result), 8)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertIsInstance(result[1], TextBlock)
        self.assertIsInstance(result[2], HeadingBlock)
        self.assertIsInstance(result[3], TextBlock)
        self.assertIsInstance(result[4], MetadataBlock)
        self.assertIsInstance(result[5], TextBlock)
        self.assertIsInstance(result[6], HeadingBlock)
        self.assertIsInstance(result[7], TextBlock)

    def test_mapped_keys(self):
        """Test mapped keys in metadata."""
        content = """
---
title: Mapped keys
?: test
---

This is some text after the header.

---
?: test123
questions: What is this?
---
"""
        result = parse_markdown_text(content, {'?': "query"})

        # Should have: HeaderBlock, TextBlock, HeadingBlock
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], HeaderBlock)
        # mapped key ignored for header
        self.assertEqual(
            result[0].get_key('title', ""), "Mapped keys"
        )
        self.assertEqual(result[0].get_key('?', ""), "test")
        self.assertIsInstance(result[1], TextBlock)
        self.assertIsInstance(result[2], MetadataBlock)
        # mapped key replaced in metadata block
        self.assertEqual(result[2].get_key('query', ""), "test123")
        self.assertEqual(
            result[2].get_key('questions', ""), "What is this?"
        )

    def test_heading_not_recognized_after_text_without_blank_line(
        self,
    ):
        """Test that headings after text without blank line are treated as text."""
        content = """Some text followed by a heading without blank separation.
## This should not be recognized as a heading."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextBlock)
        self.assertIn(
            "## This should not be recognized as a heading.",
            result[0].content,
        )

    def test_seven_hash_heading_treated_as_text(self):
        """Test that 7+ hash marks are treated as text, not headings."""
        content = "####### This is text with 7 '#', should be recognized as simple text."
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextBlock)
        self.assertEqual(
            result[0].content,
            "####### This is text with 7 '#', should be recognized as simple text.",
        )

    def test_metadata_with_list_content(self):
        """Test metadata block containing list content."""
        content = """---
- start: the start
- second: 2
  another: item
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(
            result[0], HeaderBlock
        )  # First block becomes header
        self.assertDictEqual(
            result[0].content,
            {"start": "the start", 'title': "Title"},
        )
        self.assertDictEqual(
            result[0].private_[0], {"second": 2, "another": "item"}
        )

    def test_metadata_with_multiline_string(self):
        """Test metadata with multiline string using | and > syntax."""
        content = """---
~summary: |
  Describes the 'difference' between: modelling observational and experimental data,
  and the issues that "arise" when interpreting the output of linear models.
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertIn("~summary", result[0].content)

    def test_error_block_with_literal_content(self):
        """Test that literal content in metadata generates error."""
        content = """---
literal string
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertEqual(result[0].origin.strip(), content)

    def test_error_block_with_multiple_literals(self):
        """Test that multiple literals in metadata generate error."""
        content = """---
first literal
second literal
---"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertEqual(result[0].origin.strip(), content)

    def test_carriage_return_normalization(self):
        """Test that different line ending formats are normalized."""
        content_crlf = "First line\r\nSecond line"
        content_cr = "First line\rSecond line"
        content_lf = "First line\nSecond line"

        result_crlf = parse_markdown_text(content_crlf)
        result_cr = parse_markdown_text(content_cr)
        result_lf = parse_markdown_text(content_lf)

        # All should produce the same result
        self.assertEqual(len(result_crlf), 1)
        self.assertEqual(len(result_cr), 1)
        self.assertEqual(len(result_lf), 1)

        for result in [result_crlf, result_cr, result_lf]:
            self.assertIsInstance(result[0], TextBlock)
            self.assertEqual(
                result[0].content, "First line\nSecond line"
            )

    def test_blank_lines_handling(self):
        """Test that multiple blank lines are handled correctly."""
        content = """First paragraph.



Second paragraph.


Third paragraph."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 3)
        for block in result:
            self.assertIsInstance(block, TextBlock)

    def test_text_immediately_after_heading(self):
        """Test text immediately following heading without blank line."""
        content = """# Heading
Text immediately after heading."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], HeadingBlock)
        self.assertIsInstance(result[1], TextBlock)
        self.assertEqual(
            result[1].content, "Text immediately after heading."
        )

    def test_consecutive_headings(self):
        """Test consecutive headings without blank lines."""
        content = """## First heading
## Second heading"""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], HeadingBlock)
        self.assertIsInstance(result[1], HeadingBlock)
        self.assertEqual(result[0].content, "First heading")
        self.assertEqual(result[1].content, "Second heading")

    def test_text_immediately_after_metadata(self):
        """Test text immediately following metadata block."""
        content = """---
~textid: test
---
This is some text immediately following a metadata block."""
        result = parse_markdown_text(content)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertIsInstance(result[1], TextBlock)
        self.assertEqual(
            result[1].content,
            "This is some text immediately following a metadata block.",
        )

    def test_complex_test_markdown_file_structure(self):
        """Test parsing the complex structure from test_markdown.md file."""
        # Read the test markdown file content
        with open(
            "tests/test_markdown.md", "r", encoding="utf-8"
        ) as f:
            content = f.read()

        result = parse_markdown_text(content)

        # Should have multiple blocks of different types
        self.assertGreater(len(result), 10)

        # First block should be HeaderBlock
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertEqual(result[0].content["title"], "My Document")
        self.assertEqual(result[0].content["author"], "John Doe")

        # Should contain various block types
        block_types = [type(block).__name__ for block in result]
        self.assertIn("HeaderBlock", block_types)
        self.assertIn("MetadataBlock", block_types)
        self.assertIn("HeadingBlock", block_types)
        self.assertIn("TextBlock", block_types)
        self.assertIn(
            "ErrorBlock", block_types
        )  # Due to non-conformant content


if __name__ == "__main__":
    unittest.main()
