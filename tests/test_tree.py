""" Test tree """
import unittest
from lmm.markdown.tree import load_tree

class TestTreeConstruction(unittest.TestCase):

    def test_load_tree(self):
        root = load_tree("./tests/test_markdown.md")

        self.assertIsNotNone(root)


if __name__ == "__main__":
    unittest.main()
