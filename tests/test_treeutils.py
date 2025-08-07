"""test treeutils"""

# flake8: noqa

import trace
import unittest

from lmm.markdown import blocklist_copy
from lmm.markdown.tree import *
from lmm.markdown.treeutils import *

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
blocklist = [header, metadata, heading, text]


class TestTreeToTextBlock(unittest.TestCase):

    document = "---\ntitle: 'The title'\n---\n\nSome text here.\n"
    root = load_tree(document)

    def test_header_plus_text(self):
        """A simple conformant document"""
        blocks = collect_textblocks(self.root, inherit=True)

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertIsInstance(blocks[1], TextBlock)

    def test_header_plus_text_inherit(self):
        """Inherit from header"""
        document = (
            "---\ntitle: 'The title'\n---\n\n# Level 1\n"
            + "\n## Level 2\n\nSome text here.\n"
        )
        blocks = collect_textblocks(
            self.root, inherit=True, include_header=True
        )

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertEqual(blocks[0].get_key('title'), "The title")  # type: ignore
        self.assertIsInstance(blocks[1], TextBlock)

    def test_header_plus_text_inherit_exclusion(self):
        """Inherit but not from header"""
        document = (
            "---\ntitle: 'The title'\n---\n\n# Level 1\n"
            + "\n## Level 2\n\nSome text here.\n"
        )
        blocks = collect_textblocks(
            self.root, inherit=True, include_header=False
        )

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertEqual(blocks[0].get_key('title'), "")  # type: ignore
        self.assertIsInstance(blocks[1], TextBlock)


class TestExtractProperty(unittest.TestCase):

    def test_heading_with_property(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[1].content[key] = "Text to be shifted"  # type: ignore
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = extract_property(root, key)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 2)
        self.assertFalse(key in heading.metadata)
        self.assertEqual(
            heading.get_text_children()[0].get_content(),
            "Text to be shifted",
        )

    def test_heading_without_property(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = extract_property(root, key)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 1)
        self.assertEqual(
            heading.get_text_children()[0].get_content(),
            blocks[-1].get_content(),
        )

    def test_heading_with_property_type(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[1].content[key] = "Text to be shifted"  # type: ignore
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = extract_property(root, key, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 2)
        self.assertFalse(key in heading.metadata)
        self.assertEqual(
            heading.get_text_children()[0].get_content(),
            "Text to be shifted",
        )
        self.assertTrue(
            "type" in heading.get_text_children()[0].metadata
        )
        self.assertEqual(
            heading.get_text_children()[0].metadata["type"], key
        )

    def test_heading_with_property_select(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[1].content[key] = "Text to be shifted"  # type: ignore
        blocks.append(
            HeaderBlock(content={key: "Some content of property"})
        )
        blocks.append(
            HeadingBlock(level=3, content="A subtitle of level 3")
        )
        blocks.append(TextBlock(content="Text for subheading"))
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        heading: HeadingNode = root.get_heading_children()[0]
        self.assertEqual(len(heading.get_heading_children()), 1)
        self.assertEqual(
            len(
                heading.get_heading_children()[0].get_text_children()
            ),
            1,
        )
        rootnode = extract_property(root, key, False, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 1)
        self.assertFalse(key in heading.metadata)
        self.assertEqual(
            heading.get_text_children()[0].get_content(),
            "Text to be shifted",
        )
        self.assertFalse(
            "type" in heading.get_text_children()[0].metadata
        )
        self.assertEqual(len(heading.get_heading_children()), 1)
        self.assertEqual(
            len(
                heading.get_heading_children()[0].get_text_children()
            ),
            1,
        )

    def test_heading_with_property_select3(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        # a negative
        # blocks[1].content[key] = "Text to be shifted" # type: ignore
        blocks.append(
            HeaderBlock(content={key: "Some content of property"})
        )
        blocks.insert(
            -1, HeadingBlock(level=3, content="A subtitle of level 3")
        )
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        heading: HeadingNode = root.get_heading_children()[0]
        self.assertEqual(len(heading.get_heading_children()), 1)
        self.assertEqual(
            heading.get_heading_children()[0].get_content(),
            "A subtitle of level 3",
        )
        self.assertEqual(
            len(
                heading.get_heading_children()[0].get_text_children()
            ),
            1,
        )
        rootnode = extract_property(root, key, False, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 0)
        self.assertEqual(len(heading.get_heading_children()), 1)
        self.assertEqual(
            heading.get_heading_children()[0].get_content(),
            "A subtitle of level 3",
        )
        self.assertEqual(
            len(
                heading.get_heading_children()[0].get_text_children()
            ),
            0,
        )

    def test_heading_with_property_select2(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        # a negative
        # blocks[1].content[key] = "Text to be shifted" # type: ignore
        blocks.insert(
            -1, HeadingBlock(level=3, content="A subtitle of level 3")
        )
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        heading: HeadingNode = root.get_heading_children()[0]
        self.assertEqual(len(heading.get_heading_children()), 1)
        self.assertEqual(
            heading.get_heading_children()[0].get_content(),
            "A subtitle of level 3",
        )
        self.assertEqual(
            len(
                heading.get_heading_children()[0].get_text_children()
            ),
            1,
        )
        rootnode = extract_property(root, key, False, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 0)
        self.assertEqual(len(heading.get_heading_children()), 1)

    def test_heading_with_header(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[0].content[key] = "Text to be shifted"  # type: ignore
        root: MarkdownTree = blocks_to_tree([blocks[0]])
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = extract_property(root, key, False, False)
        self.assertEqual(rootnode.count_children(), 1)
        self.assertEqual(
            rootnode.get_text_children()[0].get_content(),
            "Text to be shifted",
        )


class TestInheritedProperty(unittest.TestCase):

    def test_inherit(self):
        blocks = blocklist_copy(blocklist)
        root = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        # excludes ~chat, inherit rest
        node = inherit_metadata(root, ["~chat"])
        textkids: list[TextNode] = node.get_heading_children()[
            0
        ].get_text_children()
        for t in textkids:
            self.assertFalse("~chat" in t.metadata)
            self.assertTrue("questions" in t.metadata)
            self.assertEqual(
                t.metadata["questions"], metadata.content["questions"]
            )
            self.assertTrue("summary" in t.metadata)
            self.assertEqual(
                t.metadata["summary"], metadata.content["summary"]
            )

    def test_override(self):
        blocks = blocklist_copy(blocklist)
        root = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        textkids: list[TextNode] = root.get_heading_children()[
            0
        ].get_text_children()
        textkids[0].metadata["summary"] = "This is a new summary"
        # excludes ~chat, inherit rest
        node = inherit_metadata(root, ["~chat"])
        textkids: list[TextNode] = node.get_heading_children()[
            0
        ].get_text_children()
        for t in textkids:
            self.assertFalse("~chat" in t.metadata)
            self.assertTrue("questions" in t.metadata)
            self.assertEqual(
                t.metadata["questions"], metadata.content["questions"]
            )
            self.assertTrue("summary" in t.metadata)
            self.assertEqual(
                t.metadata["summary"], "This is a new summary"
            )

    def test_heading_wo_metadata(self):
        blocks = blocklist_copy(blocklist)
        root = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        root.get_heading_children()[0].metadata = {}
        textkids: list[TextNode] = root.get_heading_children()[
            0
        ].get_text_children()
        textkids[0].metadata["summary"] = "This is a new summary"
        # excludes ~chat, although it is not there
        node = inherit_metadata(root, ["~chat"])
        textkids: list[TextNode] = node.get_heading_children()[
            0
        ].get_text_children()
        for t in textkids:
            self.assertFalse("~chat" in t.metadata)
            self.assertFalse("questions" in t.metadata)
            self.assertTrue("summary" in t.metadata)
            self.assertEqual(
                t.metadata["summary"], "This is a new summary"
            )

    def test_wo_metadata(self):
        blocks = blocklist_copy(blocklist)
        root = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        root.get_heading_children()[0].metadata = {}
        # excludes ~chat, although not there
        node = inherit_metadata(root, ["~chat"])
        textkids: list[TextNode] = node.get_heading_children()[
            0
        ].get_text_children()
        for t in textkids:
            self.assertFalse("~chat" in t.metadata)
            self.assertFalse("questions" in t.metadata)
            self.assertFalse("summary" in t.metadata)


class TestGetTextnodes(unittest.TestCase):
    def test_textnodes(self):
        root = blocks_to_tree(blocklist)
        if not root:
            raise (Exception("Invalid blocks"))

        textnodes: list[TextNode] = get_textnodes(root)
        self.assertTrue(len(textnodes) > 0)
        for n in textnodes:
            self.assertTrue(isinstance(n, TextNode))


class TestTransduceNodetype(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
