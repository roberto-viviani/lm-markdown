import unittest

from pydantic import ValidationError

from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    parse_markdown_text,
)
from lmm.markdown.tree import (
    blocks_to_tree,
    tree_to_blocks,
    MarkdownNode,
    HeadingNode,
    TextNode,
    pre_order_traversal,
    traverse_tree_nodetype,
)
from lmm.markdown.treeutils import (
    get_nodes_with_metadata,
    count_words,
)
from lmm.config.config import Settings, export_settings
from lmm.scan.scan_keys import (
    DOCID_KEY,
    TEXTID_KEY,
    HEADINGID_KEY,
    SOURCE_KEY,
    TXTHASH_KEY,
    TITLES_KEY,
    SKIP_KEY,
    UUID_KEY,
    SUMMARY_KEY,
    QUESTIONS_KEY,
)
from lmm.scan.scan_rag import scan_rag, ScanOpts


header_block = """
---
author: Roberto Viviani
date: '2025-05-04'
output:
  html_document: default
  md_document: default
  word_document: default
title: Chapter 1
docid: Chapter 1
---

"""

first_heeading_group = """
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see.

There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant".

The second use of linear models is to predict the outcome given certain values of the predictors.[^1] This use of linear models is the same as in machine learning.[^2] In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome.

[^1]: Predictor means the same as independent variable.

[^2]: Machine learning is the field of artificial intelligence that is concerned with training programs to accomplish tasks based on a training set.

"""
second_heading_group = """
---
note: revise if possible
...
## Observational and experimental studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcome, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past and we therefore look at their consequences in a sample from the population. Studies of this type are called *observational*. An important issue in observational studies is that predictors may be *confounded* by other factors affecting the outcome. For example, it is conceivable that traumas may occur more frequently in adverse socioeconomic conditions, and these latter may in turn adversely affect the predisposition to psychopathology. When we assess the association of traumas and psychopathology, it may include the effects of socioeconomic conditions, as traumatized individuals are also those that are most disadvantaged socioeconomically.

In the second setting, the predictor of interest is a variable representing an experimental manipulation, such as treatment, changes in aspects of a cognitive task, etc. A key aspect of *experimental* studies is that the value of the predictor of interest, being determined by you, the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

Importantly, the way in which we estimate models in observational and experimental studies is exactly the same. However, the conclusions that we may draw from establishing the likely existence of an association between predictor and outcome differ in these two cases. In observational studies, we can only infer the mere association, unless a sometimes fairly extensive efforts are made on being able to contain the effect of confounders. In experimental studies, we can infer that the treatment variable causes the effect measured by the outcome variable. The term *effect* is sometimes reserved to the causal association that may be established in experimental studies, but is often used more loosely to mean any association between predictors and outcomes. This loose usage may mask the very different nature of the associations detected in these two types of studies.

### Using models for prediction

When using linear models solely for prediction, the concerns regarding inference in observational studies do not apply. While confounding in predictor variables can influence the interpretation of the model coefficients, it may not significantly affect predictive performance. Therefore, if our primary focus is on achieving accurate predictions of the outcome, we can simply include all predictors in the model that contribute to reduce the prediction error.
"""

composite_text_group = """
---
summary: This section introduces the model equation, emphasizing its importance in understanding the linear model.
...
## The model equation

It is now time to see a linear model in action. Let us assume we measured depressive symptoms (the outcome) with a self-rating scale and we are interested in exploring their association with sex. We may formulate the model as follows:

depression = baseline + female + error

Here, $female$ encodes sex by taking the value 1 for women and 0 for men. The outcome we measured (depressive symptoms) is decomposed by the model into the sum of three terms: the baseline depressiveness, i.e. the average depressive symptoms of males; the difference in depressive symptoms observed on average in females relative to males, and a final term accounting for the difference in depressive symptoms measured in each individuals relative to these averages. This expression applies for each observation. If we index an observation with $i$, we may write

$$depression_i = baseline + female_i * female_coefficient + error_i$$

This equation is called the _model equation_. At this point, you might be tempted to skip over the model equation and think that you may deal with it later or avoid using equations altogether. Don't! The model equation is crucial to understand linear models. Once mastered, it becomes your most helpful aid in understanding the model. It is therefore crucial to become familiar with it. 
This model equation means that in individual $i$, the measured depressive symptoms $depression_i$ are the sum of  $baseline$, which is the average depressive symptoms in male individuals, of the average difference in depressive symptoms in females $female_coefficient$, and of the errors $error_i$. The $female_coefficient$ is applied to all all data, but here in males it has no influence as it is multiplied by $female$, which is zero in males. In females, the $female_coefficient$ is multiplied by one and remains equal to itself. The errors are the difference between the observed depressiveness in each individual and these estimated averages. Note that there is one baseline and one female coefficient for the whole dataset. These are the _coefficients_ of the model. The baseline coefficient is sometimes called _constant term_ or _intercept_.

When the model is _fitted_, we obtain the estimates of the coefficients. To do this, we use the following line in the R console:

```{r}
fit <- lm(depression ~ female, data = depr_sample)
```

where depr_sample is the data frame containing the columns `depression` and `female`. The expression `depression ~ female` is the model equation, except for the baseline term, which is added automatically by R. We could also have written this term explicitly in the model equation as follows,

```{r}
fit <- lm(depression ~ 1 + female, data = depr_sample)
```

obtaining the same model. We can now inspect the coefficients of the model fit:

```{r}
summary(fit)
```

To understand what these coefficients mean, let us put them in the model equation. We obtain

$$depression_i = 24 + female_i * 4 + error_i$$

To calculate the predicted depressiveness, we set the value of the female predictor to one or zero, depending on the sex of the individual we are predicting. It follows that 24 is the predicted depressive symptom score in males, and 4 is the predicted difference in these scores from males to females. It also follows that the predicted depression score in females is 24 + 4 = 28 (in many samples of self-reported depressiveness, one often finds higher scores in females, although this finding may fail to be confirmed by other measures of depressiveness).

While this is a linear model, it is also a two-sample _t_ test. In a two-sample _t_ test we test the difference between means. Here, these means are 24 and 28. The linear model is a very general formulation, relative to which other models are special cases. 

Another case of a linear model that you might not have suspected to be one is the mean:

$$depression_i = constant_term + error_i$$

where we used the name _constant term_ for the baseline, estimated as the average of all observed depression scores in the whole sample (which will differ from the estimate of the male average of the previous model). This is the simplest possible linear model, and we will come back to it later to help intuition about the properties of the fitted model.
"""

document = (
    header_block
    + first_heeading_group
    + second_heading_group
    + composite_text_group
)


class TestValidations(unittest.TestCase):

    def test_empty(self):
        # test valid blocklist produced from empty lists
        blocks = scan_rag([])

        self.assertEqual(len(blocks), 1)
        self.assertTrue(isinstance(blocks[0], HeaderBlock))
        header: HeaderBlock = blocks[0]  # type: ignore
        self.assertEqual(header.get_key('title'), "Title")
        self.assertTrue(header.get_key(DOCID_KEY))

    def test_heading(self):
        # test valid blocklist produced from just heading
        blocks = scan_rag(parse_markdown_text("# Heading 1"))

        self.assertEqual(len(blocks), 3)
        self.assertTrue(isinstance(blocks[0], HeaderBlock))
        header: HeaderBlock = blocks[0]  # type: ignore
        self.assertEqual(header.get_key('title'), "Heading 1")
        self.assertTrue(header.get_key(DOCID_KEY))
        self.assertTrue(isinstance(blocks[1], MetadataBlock))
        self.assertTrue(isinstance(blocks[2], HeadingBlock))

    def test_text(self):
        # test valid blocklist produced from just text
        blocks = scan_rag(parse_markdown_text("Text without titles"))

        self.assertEqual(len(blocks), 2)
        self.assertTrue(isinstance(blocks[0], HeaderBlock))
        header: HeaderBlock = blocks[0]  # type: ignore
        self.assertEqual(header.get_key('title'), "Title")
        self.assertTrue(header.get_key(DOCID_KEY))
        self.assertTrue(isinstance(blocks[1], TextBlock))
        self.assertEqual(
            blocks[1].get_content(), "Text without titles"
        )

    def test_text_textid(self):
        blocks = scan_rag(
            parse_markdown_text("Text without titles"),
            ScanOpts(textid=True),
        )

        self.assertEqual(len(blocks), 3)
        self.assertTrue(isinstance(blocks[0], HeaderBlock))
        header: HeaderBlock = blocks[0]  # type: ignore
        self.assertEqual(header.get_key('title'), "Title")
        self.assertTrue(header.get_key(DOCID_KEY))
        self.assertTrue(isinstance(blocks[1], MetadataBlock))
        self.assertIn(TEXTID_KEY, blocks[1].content)  # type: ignore
        self.assertTrue(isinstance(blocks[2], TextBlock))
        self.assertEqual(
            blocks[2].get_content(), "Text without titles"
        )


class TestBuilds(unittest.TestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={'model': "Debug/debug"},
            minor={'model': "Debug/debug"},
            aux={'model': "Debug/debug"},
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_setup(self):
        settings = Settings()
        self.assertEqual(settings.major.get_model_source(), "Debug")
        self.assertEqual(settings.minor.get_model_source(), "Debug")
        self.assertEqual(settings.aux.get_model_source(), "Debug")
        self.assertEqual(settings.major.get_model_name(), "debug")
        self.assertEqual(settings.minor.get_model_name(), "debug")
        self.assertEqual(settings.aux.get_model_name(), "debug")

    def test_build_summaries(self):
        # test all heading nodes and dependant text nodes have summary
        # NOTE: this should fail if text blocks are too short, modify
        blocks_raw: list[Block] = parse_markdown_text(document)

        root = blocks_to_tree(blocks_raw)
        if not root:
            raise Exception("Invalid text in test")

        # mark nodes that already have summaries, and were manually
        # added (without hash)
        def _mark_node(n: MarkdownNode) -> None:
            if (
                SUMMARY_KEY in n.metadata
                and TXTHASH_KEY not in n.metadata
            ):
                n.metadata['__MARKER__'] = "set"

        pre_order_traversal(root, _mark_node)
        blocks = tree_to_blocks(root)

        # now apply scan rag
        with self.assertRaises(ValidationError):
            # summary threshold at least 20
            blocks = scan_rag(
                blocks, ScanOpts(summaries=True, summary_threshold=15)
            )
        blocks = scan_rag(
            blocks, ScanOpts(summaries=True, summary_threshold=21)
        )

        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, SUMMARY_KEY)
        self.assertTrue(len(nodes) > 0)

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root, lambda x: x, HeadingNode
        )
        self.assertTrue(len(headingnodes) > 0)
        # text too short if not using a language model
        # self.assertIn(SUMMARY_KEY, headingnodes[0].metadata)
        for n in headingnodes[1:]:
            self.assertIn(SUMMARY_KEY, n.metadata)
            # test new summaries have hashes
            if '__MARKER__' not in n.metadata:
                self.assertIn(TXTHASH_KEY, n.metadata)

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertTrue(len(textnodes) > 0)
        self.assertTrue(
            textnodes[0].fetch_metadata_for_key(SUMMARY_KEY)
        )
        for n in textnodes:
            self.assertTrue(n.fetch_metadata_for_key(SUMMARY_KEY))

    def test_build_summaries_tresholded(self):
        # test all heading nodes and dependant text nodes have summary
        # NOTE: this should fail if text blocks are too short, modify
        new_doc = (
            header_block
            + first_heeading_group
            + """
### title 3
A few words.

### title 4
Another few words.
"""
        )
        blocks_raw: list[Block] = parse_markdown_text(new_doc)

        root = blocks_to_tree(blocks_raw)
        if not root:
            raise Exception("Invalid text in test")

        # mark nodes that already have summaries, and were manually
        # added (without hash)
        def _mark_node(n: MarkdownNode) -> None:
            if (
                SUMMARY_KEY in n.metadata
                and TXTHASH_KEY not in n.metadata
            ):
                n.metadata['__MARKER__'] = "set"

        pre_order_traversal(root, _mark_node)
        blocks = tree_to_blocks(root)

        # now apply scan rag
        THRESHOLD = 150
        blocks = scan_rag(
            blocks,
            ScanOpts(summaries=True, summary_threshold=THRESHOLD),
        )
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, SUMMARY_KEY)
        self.assertTrue(len(nodes) > 0)

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root, lambda x: x, HeadingNode
        )
        self.assertTrue(len(headingnodes) > 0)
        # TODO: this may not work here, the header gets few words
        # self.assertIn(SUMMARY_KEY, headingnodes[0].metadata)
        for n in headingnodes[1:]:
            if count_words(n) > THRESHOLD:
                self.assertIn(SUMMARY_KEY, n.metadata)
                # test new summaries have hashes
                if '__MARKER__' not in n.metadata:
                    self.assertIn(TXTHASH_KEY, n.metadata)
            else:
                self.assertTrue(SUMMARY_KEY not in n.metadata)

    def test_build_questions(self):
        # Test all headings have questions, except the root node
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(blocks_raw, ScanOpts(questions=True))
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, QUESTIONS_KEY)
        self.assertTrue(len(nodes) > 0)

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root,
            lambda x: x.naked_copy(),
            HeadingNode,
            lambda x: x.heading_level() > 0,
        )
        self.assertTrue(len(headingnodes) > 0)
        self.assertIn(QUESTIONS_KEY, headingnodes[0].metadata)
        for n in headingnodes:
            self.assertIn(QUESTIONS_KEY, n.metadata)

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertTrue(len(textnodes) > 0)
        self.assertTrue(
            textnodes[0].fetch_metadata_for_key(QUESTIONS_KEY)
        )
        for n in textnodes:
            self.assertTrue(n.fetch_metadata_for_key(QUESTIONS_KEY))

    def test_build_questions_language_model(self):
        # overrides config.toml
        from lmm.config.config import LanguageModelSettings

        # Test all headings have questions, except the root note
        blocks_raw: list[Block] = parse_markdown_text(document)
        opts = ScanOpts(
            questions=True,  # add questions
            language_model_settings=LanguageModelSettings(
                model="Debug/debug"
            ),
        )
        blocks = scan_rag(blocks_raw, opts)
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, QUESTIONS_KEY)
        self.assertTrue(len(nodes) > 0)

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root,
            lambda x: x.naked_copy(),
            HeadingNode,
            lambda x: not x.is_header_node(),
        )
        self.assertTrue(len(headingnodes) > 0)
        self.assertIn(QUESTIONS_KEY, headingnodes[0].metadata)
        for n in headingnodes:
            self.assertIn(QUESTIONS_KEY, n.metadata)

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertTrue(len(textnodes) > 0)
        self.assertTrue(
            textnodes[0].fetch_metadata_for_key(QUESTIONS_KEY)
        )
        for n in textnodes:
            self.assertTrue(n.fetch_metadata_for_key(QUESTIONS_KEY))

    def test_build_titles(self):
        # Test all headings have questions
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(blocks_raw, ScanOpts(titles=True))
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, TITLES_KEY)
        self.assertTrue(len(nodes) > 0)

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root, lambda x: x.naked_copy(), HeadingNode
        )
        self.assertTrue(len(headingnodes) > 0)
        self.assertIn(TITLES_KEY, headingnodes[0].metadata)
        for n in headingnodes:
            self.assertIn(TITLES_KEY, n.metadata)

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertTrue(len(textnodes) > 0)
        self.assertTrue(
            textnodes[0].fetch_metadata_for_key(TITLES_KEY)
        )
        for n in textnodes:
            self.assertTrue(n.fetch_metadata_for_key(TITLES_KEY))

    def test_build_headingid(self):
        # Test all headings have headingid
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(blocks_raw, ScanOpts(headingid=True))
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, HEADINGID_KEY)
        self.assertTrue(len(nodes) > 0)
        ids: list[str] = []
        for n in nodes:
            self.assertTrue(isinstance(n, HeadingNode))
            ids.append(n.get_metadata_for_key(HEADINGID_KEY))

        self.assertEqual(len(nodes), len(set(ids)))

        headingnodes: list[HeadingNode] = traverse_tree_nodetype(
            root, lambda x: x.naked_copy(), HeadingNode
        )
        self.assertEqual(len(nodes), len(headingnodes))

    def test_build_textid(self):
        # test all text nodes have textid
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(blocks_raw, ScanOpts(textid=True))
        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, TEXTID_KEY)
        self.assertTrue(len(nodes) > 0)
        ids: list[str] = []
        for n in nodes:
            self.assertTrue(isinstance(n, TextNode))
            ids.append(n.get_metadata_for_key(TEXTID_KEY))

        self.assertEqual(len(nodes), len(set(ids)))

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertEqual(len(nodes), len(textnodes))

    def test_build_UUID(self):
        # test all text nodes have textid
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(
            blocks_raw, ScanOpts(textid=True, UUID=True)
        )

        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, UUID_KEY)
        self.assertTrue(len(nodes) > 0)
        ids: list[str] = []
        for n in nodes:
            self.assertTrue(isinstance(n, TextNode))
            ids.append(n.get_metadata_for_key(UUID_KEY))

        self.assertEqual(len(nodes), len(set(ids)))

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertEqual(len(nodes), len(textnodes))

    def test_build_UUID2(self):
        # test all text nodes have textid
        blocks_raw: list[Block] = parse_markdown_text(document)
        blocks = scan_rag(
            blocks_raw, ScanOpts(textid=False, UUID=True)  # ignored
        )

        root = blocks_to_tree(blocks)
        if not root:
            raise Exception("Invalid text in test")

        nodes = get_nodes_with_metadata(root, UUID_KEY)
        self.assertTrue(len(nodes) > 0)
        ids: list[str] = []
        for n in nodes:
            self.assertTrue(isinstance(n, TextNode))
            ids.append(n.get_metadata_for_key(UUID_KEY))

        self.assertEqual(len(nodes), len(set(ids)))

        textnodes: list[TextNode] = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )
        self.assertEqual(len(nodes), len(textnodes))


class TestSkippedNodes(unittest.TestCase):
    # setup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={'model': "Debug/debug"},
            minor={'model': "Debug/debug"},
            aux={'model': "Debug/debug"},
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def setUp(self):
        self.doc_with_skip = """
---
title: Document with skips
---

# Heading 1 (Process)

Text under heading 1.

---
skip: True
...
## Heading 2 (Skip)

Text under heading 2 that should be skipped.

### Heading 3 (Also Skip)

Text under heading 3.

---
skip: True
...
This is a skipped text block.

# Heading 4 (Process)

Final text block.
"""
        self.blocks = parse_markdown_text(self.doc_with_skip)

    def test_skip_textid(self):
        """Test that textid is not added to skipped text blocks."""
        blocks = scan_rag(self.blocks, ScanOpts(textid=True))
        root = blocks_to_tree(blocks)
        self.assertIsNotNone(root)

        text_nodes = traverse_tree_nodetype(
            root, lambda x: x, TextNode
        )

        self.assertEqual(len(text_nodes), 5)

        processed_text_nodes = [
            n for n in text_nodes if TEXTID_KEY in n.metadata
        ]
        skipped_text_nodes = [
            n for n in text_nodes if TEXTID_KEY not in n.metadata
        ]

        # 4 text blocks should have textid, 1 should be skipped
        self.assertEqual(len(processed_text_nodes), 4)
        self.assertEqual(len(skipped_text_nodes), 1)

        self.assertEqual(
            skipped_text_nodes[0].get_content(),
            "This is a skipped text block.",
        )
        self.assertTrue(
            skipped_text_nodes[0].get_metadata_for_key(SKIP_KEY)
        )

    def test_skip_headingid(self):
        """Test that headingid is not added to skipped heading blocks."""
        blocks = scan_rag(self.blocks, ScanOpts(headingid=True))
        root = blocks_to_tree(blocks)
        self.assertIsNotNone(root)

        heading_nodes = traverse_tree_nodetype(
            root, lambda x: x, HeadingNode
        )

        # Root + 4 headings = 5
        self.assertEqual(len(heading_nodes), 5)

        processed_heading_nodes = [
            n for n in heading_nodes if HEADINGID_KEY in n.metadata
        ]
        skipped_heading_nodes = [
            n
            for n in heading_nodes
            if HEADINGID_KEY not in n.metadata
        ]

        # Root, H1, H4 should be processed. H2 and H3 should be skipped.
        self.assertEqual(len(processed_heading_nodes), 3)
        self.assertEqual(len(skipped_heading_nodes), 2)

        skipped_contents = {
            n.get_content() for n in skipped_heading_nodes
        }
        self.assertIn("Heading 2 (Skip)", skipped_contents)
        self.assertIn("Heading 3 (Also Skip)", skipped_contents)

    def test_skip_questions(self):
        """Test that questions are not added to skipped headings."""
        blocks = scan_rag(
            self.blocks,
            ScanOpts(questions=True, questions_threshold=1),
        )
        root = blocks_to_tree(blocks)
        self.assertIsNotNone(root)

        # Find all nodes that have questions
        nodes_with_questions = get_nodes_with_metadata(
            root, QUESTIONS_KEY
        )

        # Questions are added to Heading 1 and Heading 4.
        # The root node does not get questions aggregated from its children.
        self.assertEqual(len(nodes_with_questions), 2)

        for node in nodes_with_questions:
            self.assertFalse(
                node.get_metadata_for_key(SKIP_KEY, False)
            )

    def test_skip_summaries(self):
        """Test that summaries are not added to skipped headings."""
        blocks = scan_rag(
            self.blocks, ScanOpts(summaries=True, summary_threshold=1)
        )
        root = blocks_to_tree(blocks)
        self.assertIsNotNone(root)

        # Find all nodes that have summaries
        nodes_with_summaries = get_nodes_with_metadata(
            root, SUMMARY_KEY
        )

        # Summaries are added to Heading 1 and Heading 4.
        # The root node will also get a summary aggregated from its children.
        self.assertEqual(len(nodes_with_summaries), 3)

        for node in nodes_with_summaries:
            self.assertFalse(
                node.get_metadata_for_key(SKIP_KEY, False)
            )

    def test_skip_source(self):
        """Test that source is not added to skipped heading blocks."""
        blocks = scan_rag(self.blocks)
        root = blocks_to_tree(blocks)
        self.assertIsNotNone(root)

        heading_nodes = traverse_tree_nodetype(
            root, lambda x: x, HeadingNode
        )

        # Root + 4 headings = 5
        self.assertEqual(len(heading_nodes), 5)

        processed_heading_nodes = [
            n for n in heading_nodes if SOURCE_KEY in n.metadata
        ]
        skipped_heading_nodes = [
            n for n in heading_nodes if SOURCE_KEY not in n.metadata
        ]

        # H1, H4 should be processed. Header, H2 and H3 should be skipped.
        self.assertEqual(len(processed_heading_nodes), 2)
        self.assertEqual(len(skipped_heading_nodes), 3)

        skipped_contents = {
            n.get_content() for n in skipped_heading_nodes
        }
        self.assertIn("Heading 2 (Skip)", skipped_contents)
        self.assertIn("Heading 3 (Also Skip)", skipped_contents)


if __name__ == '__main__':
    unittest.main()
