"""test treeutils"""

# flake8: noqa
# pyright: basic
# pyright: reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false

import unittest

from lmm.markdown.parse_markdown import blocklist_copy
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
blocklist: list[Block] = [header, metadata, heading, text]


class TestCollectTextBlocks(unittest.TestCase):
    def test_header_plus_text(self):
        """A simple conformant document"""
        document = "---\ntitle: 'The title'\n---\n\nSome text here.\n"
        root = load_tree(document)
        blocks = collect_annotated_textblocks(root, inherit=True)

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertIsInstance(blocks[1], TextBlock)

    def test_header_plus_text_inherit(self):
        """Inherit from header"""
        document = (
            "---\ntitle: 'The title'\n---\n\n# Level 1\n"
            + "\n## Level 2\n\nSome text here.\n"
        )
        root = load_tree(document)
        if root is None:
            raise ValueError("Could not form tree")
        blocks = collect_annotated_textblocks(
            root, inherit=True, include_header=True
        )

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertEqual(blocks[0].get_key('title'), "The title")
        self.assertIsInstance(blocks[1], TextBlock)

    def test_header_plus_text_inherit_exclusion(self):
        """Inherit but not from header"""
        document = (
            "---\ntitle: 'The title'\n---\n\n# Level 1\n"
            + "\n## Level 2\n\nSome text here.\n"
        )
        root = load_tree(document)
        if root is None:
            raise ValueError("Could not form tree")
        blocks = collect_annotated_textblocks(
            root, inherit=True, include_header=False
        )

        self.assertEqual(len(blocks), 2)
        self.assertIsInstance(blocks[0], MetadataBlock)
        self.assertEqual(blocks[0].get_key('title'), "")
        self.assertIsInstance(blocks[1], TextBlock)


class TestCollectThings(unittest.TestCase):
    def test_collect_nulltext(self):
        root: MarkdownTree = blocks_to_tree(blocklist[:-1])
        texts: list[str] = collect_text(root)
        self.assertEqual(len(texts), 0)

    def test_collect_text(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        texts: list[str] = collect_text(root)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], text.get_content())

    def test_collect_text_filter(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        texts: list[str] = collect_text(
            root, lambda x: x.get_content().startswith("This is text")
        )
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], text.get_content())

    def test_collect_headings(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        texts: list[str] = collect_headings(root)
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Test blocklist")

    def test_collect_headings_filtered(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        texts: list[str] = collect_headings(
            root, lambda n: n.get_content() == "First title"
        )
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], "First title")

    def test_collect_dictionaries(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        if root is None:
            raise RuntimeError("Could not parse tree")
        dicts: list[NodeDict] = collect_dictionaries(root)
        self.assertEqual(len(dicts), 3)
        self.assertEqual(root.get_content(), dicts[0]['content'])
        self.assertEqual(
            root.get_heading_children()[0].get_content(),
            dicts[1]['content'],
        )
        self.assertDictEqual(
            root.get_heading_children()[0].get_metadata(),
            dicts[1]['metadata'],
        )

    def test_collect_headings_filter(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        if root is None:
            raise RuntimeError("Could not parse tree")
        dicts: list[NodeDict] = collect_dictionaries(
            root, lambda n: n.get_content() == "First title"
        )
        self.assertEqual(len(dicts), 1)
        self.assertEqual(
            root.get_heading_children()[0].get_content(),
            dicts[0]['content'],
        )
        self.assertDictEqual(
            root.get_heading_children()[0].get_metadata(),
            dicts[0]['metadata'],
        )

    def test_collect_table_of_contents(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        if root is None:
            raise RuntimeError("Could not parse tree")
        contents: list[dict[str, int | str]] = (
            collect_table_of_contents(root)
        )
        self.assertEqual(len(contents), 2)
        self.assertDictEqual(
            contents[0], {'level': 0, 'content': "Test blocklist"}
        )


class TestCountWords(unittest.TestCase):
    def test_count_words(self):
        root: MarkdownTree = blocks_to_tree(blocklist)
        count: int = count_words(root)
        self.assertEqual(count, 6)


class TestPropagateProperty(unittest.TestCase):
    def test_heading_with_property(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[1].content[key] = "Text to be shifted"
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = propagate_property(root, key)
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
        rootnode = propagate_property(root, key)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 1)
        self.assertEqual(
            heading.get_text_children()[0].get_content(),
            blocks[-1].get_content(),
        )

    def test_heading_with_property_type(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[1].content[key] = "Text to be shifted"
        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = propagate_property(root, key, True)
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
        blocks[1].content[key] = "Text to be shifted"
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
        rootnode = propagate_property(root, key, False, True)
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
        # a negative, i.e. test for when the property does not exist
        # blocks[1].content[key] = "Text to be shifted"
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
        rootnode = propagate_property(root, key, False, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 1)
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

    def test_heading_with_property_select2(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        # a negative, i.e. test for when the propoerty does not exist
        # blocks[1].content[key] = "Text to be shifted"
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
        rootnode = propagate_property(root, key, False, True)
        heading: HeadingNode = rootnode.get_heading_children()[0]
        self.assertEqual(len(heading.get_text_children()), 0)
        self.assertEqual(len(heading.get_heading_children()), 1)

    def test_heading_with_header(self):
        key: str = "new_property"
        blocks = blocklist_copy(blocklist)
        blocks[0].content[key] = "Text to be shifted"
        root: MarkdownTree = blocks_to_tree([blocks[0]])
        if not root:
            raise (Exception("Invalid blocks"))
        rootnode = propagate_property(root, key, False, False)
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


class TestMapTree(unittest.TestCase):
    def test_pre_order_map(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        def map_func(node: MarkdownNode) -> MarkdownNode:
            if isinstance(node, TextNode):
                node.set_content("New text!")
            return node

        root = pre_order_map_tree(root, map_func)

        blocks = tree_to_blocks(root)
        for b in blocks:
            if isinstance(b, TextBlock):
                self.assertEqual(b.get_content(), "New text!")

    def test_post_order_map(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        def map_func(node: MarkdownNode) -> MarkdownNode:
            if isinstance(node, TextNode):
                node.set_content("New text!")
            return node

        root = post_order_map_tree(root, map_func)

        blocks = tree_to_blocks(root)
        for b in blocks:
            if isinstance(b, TextBlock):
                self.assertEqual(b.get_content(), "New text!")


class TestGetTextnodes(unittest.TestCase):
    def test_textnodes(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        NAKED_COPY = True
        textnodes: list[TextNode] = get_textnodes(root, NAKED_COPY)
        self.assertTrue(len(textnodes) > 0)
        for n in textnodes:
            self.assertIsNone(n.parent)

        textblocks = [b for b in blist if isinstance(b, TextBlock)]
        textnodes[0].block.content = "New content"
        self.assertNotEqual(
            textblocks[0].get_content(), "New content"
        )

    def test_textnodes_ref(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        NAKED_COPY = False
        textnodes: list[TextNode] = get_textnodes(root, NAKED_COPY)
        self.assertTrue(len(textnodes) > 0)
        for n in textnodes:
            self.assertTrue(bool(n.parent))

        textblocks = [b for b in blist if isinstance(b, TextBlock)]
        textnodes[0].block.content = "New content"
        self.assertEqual(textblocks[0].get_content(), "New content")

    def test_textnodes_filter(self):
        root = blocks_to_tree(
            blocklist_copy(blocklist)
            + [TextBlock(content="That is new")]
        )
        if not root:
            raise (Exception("Invalid blocks"))

        NAKED_COPY = True
        textnodes: list[TextNode] = get_textnodes(
            root,
            NAKED_COPY,
            lambda tn: tn.get_content().startswith("That"),
        )
        self.assertTrue(len(textnodes) == 1)
        for n in textnodes:
            self.assertIsNone(n.parent)

        self.assertEqual(textnodes[0].get_content(), "That is new")


class TestGetHeadingnodes(unittest.TestCase):
    def test_headingnodes(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        NAKED_COPY = True
        hnodes: list[HeadingNode] = get_headingnodes(root, NAKED_COPY)
        self.assertTrue(len(hnodes) > 0)
        for n in hnodes:
            self.assertIsNone(n.parent)

        hblocks = [b for b in blist if isinstance(b, HeadingBlock)]
        hnodes[0].block.content = "New content"
        self.assertNotEqual(hblocks[0].get_content(), "New content")

    def test_headingnodes_ref(self):
        blist = blocklist_copy(blocklist)
        root = blocks_to_tree(blist)
        if not root:
            raise (Exception("Invalid blocks"))

        NAKED_COPY = False
        hnodes: list[HeadingNode] = get_headingnodes(root, NAKED_COPY)
        self.assertTrue(len(hnodes) > 0)
        # note: the first heading is the root, and has no parent
        for n in hnodes[1:]:
            self.assertIsNotNone(n.parent)

        hblocks = [b for b in blist if isinstance(b, HeadingBlock)]
        hnodes[1].block.content = "New content"
        self.assertEqual(hblocks[0].get_content(), "New content")


if __name__ == "__main__":
    unittest.main()
