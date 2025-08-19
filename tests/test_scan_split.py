"""Test scan_split"""

import unittest

from typing import Any

from lmm.scan.scan_keys import UUID_KEY, TEXTID_KEY
from lmm.scan.scan_split import scan_split, NullTextSplitter
from lmm.markdown.parse_markdown import (
    MetadataBlock,
    HeaderBlock,
    parse_markdown_text,
)
from lmm.markdown.tree import (
    blocks_to_tree,
    MarkdownTree,
    TextNode,
    traverse_tree_nodetype,
)
from lmm.markdown.treeutils import get_textnodes


header_block = """
---
title: my text
...

"""

metadata_block = """
---
questions: What is logistic regression?
...

"""

text_block = """
Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see. There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant". The second use of linear models is to predict the outcome given certain values of the predictors.[^1] This use of linear models is the same as in machine learning.[^2] In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome. When we look at the significance of the association between predictors and outcome, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past and we therefore look at their consequences in a sample from the population. Studies of this type are called *observational*. An important issue in observational studies is that predictors may be *confounded* by other factors affecting the outcome. For example, it is conceivable that traumas may occur more frequently in adverse socioeconomic conditions, and these latter may in turn adversely affect the predisposition to psychopathology. When we assess the association of traumas and psychopathology, it may include the effects of socioeconomic conditions, as traumatized individuals are also those that are most disadvantaged socioeconomically. In the second setting, the predictor of interest is a variable representing an experimental manipulation, such as treatment, changes in aspects of a cognitive task, etc. A key aspect of *experimental* studies is that the value of the predictor of interest, being determined by you, the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

"""

document = header_block + metadata_block + text_block


class TestNulls(unittest.TestCase):

    def test_null_document(self):
        blocks = scan_split([])

        self.assertEqual(len(blocks), 0)

    def test_empty_document(self):
        blocks = parse_markdown_text(header_block)
        cb = len(blocks)
        blocks = scan_split(blocks)

        self.assertEqual(len(blocks), cb)

    def test_empty_document2(self):
        blocks = parse_markdown_text(header_block + "\n#A title\n")
        cb = len(blocks)
        blocks = scan_split(blocks)

        self.assertEqual(len(blocks), cb)


class TestSplits(unittest.TestCase):

    def test_null_split(self):
        blocks = parse_markdown_text(document)
        cb = len(blocks)
        blocks = scan_split(blocks, NullTextSplitter())

        self.assertEqual(len(blocks), cb)

    def test_default_split(self):
        blocks = parse_markdown_text(document)
        cb = len(blocks)
        meta: dict[str, Any] = (
            [
                b.content
                for b in blocks[1:]
                if isinstance(b, MetadataBlock)
            ]
        )[0]
        blocks = scan_split(blocks)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertTrue(len(blocks) > cb)

        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid doc in test block")

        nodes: list[TextNode] = get_textnodes(root)
        for n in nodes:
            self.assertDictEqual(meta, n.get_metadata())

    def test_split_UUID(self):
        uuid: str = (
            UUID_KEY + ": ff8c11c3-1dfe-5746-9df8-46a57ace5ad9"
        )
        newdoc = (
            header_block + "\n---\n" + uuid + "\n...\n" + text_block
        )
        blocks = parse_markdown_text(newdoc)
        cb = len(blocks)
        blocks = scan_split(blocks)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertTrue(len(blocks) > cb)

        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid doc in test block")

        nodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertGreater(len(nodes), 1)
        self.assertDictEqual(
            nodes[0].get_metadata(),
            {UUID_KEY: "ff8c11c3-1dfe-5746-9df8-46a57ace5ad9"},
        )
        for n in nodes[1:]:
            self.assertDictEqual(n.get_metadata(), {})

    def test_split_textid(self):
        uuid: str = TEXTID_KEY + ": Ch1.1"
        newdoc = (
            header_block + "\n---\n" + uuid + "\n...\n" + text_block
        )
        blocks = parse_markdown_text(newdoc)
        cb = len(blocks)
        blocks = scan_split(blocks)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertTrue(len(blocks) > cb)

        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid doc in test block")

        nodes: list[TextNode] = get_textnodes(root)
        self.assertGreater(len(nodes), 1)
        self.assertDictEqual(
            nodes[0].get_metadata(), {TEXTID_KEY: "Ch1.1"}
        )
        for n in nodes[1:]:
            self.assertDictEqual(n.get_metadata(), {})

    def test_split_metawithUUID(self):
        uuid: str = (
            UUID_KEY + ": ff8c11c3-1dfe-5746-9df8-46a57ace5ad9"
        )
        newdoc = (
            header_block
            + "\n---\n"
            + uuid
            + "\nquestions: about regression\n...\n"
            + text_block
        )
        blocks = parse_markdown_text(newdoc)
        cb = len(blocks)
        blocks = scan_split(blocks)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertTrue(len(blocks) > cb)

        root: MarkdownTree = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid doc in test block")

        nodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertGreater(len(nodes), 1)
        self.assertDictEqual(
            nodes[0].get_metadata(),
            {
                UUID_KEY: "ff8c11c3-1dfe-5746-9df8-46a57ace5ad9",
                'questions': "about regression",
            },
        )
        for n in nodes[1:]:
            self.assertDictEqual(
                n.get_metadata(), {'questions': "about regression"}
            )


if __name__ == "__main__":
    unittest.main()
