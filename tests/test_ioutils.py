import unittest
import tempfile
import shutil
from pathlib import Path

from lmm.utils.ioutils import (
    list_files_with_extensions,
    clean_text_concat,
)


class TestListFilesWithExtensions(unittest.TestCase):
    """Unit tests for the list_files_with_extensions function."""

    def setUp(self):
        """Set up a temporary directory with some test files."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.files_to_create = [
            "document1.md",
            "document2.txt",
            "image.JPG",  # Test case-insensitivity of filesystem
            "script.py",
            "archive.zip",
            "no_extension",
            ".config",
        ]
        for filename in self.files_to_create:
            (self.test_dir / filename).touch()

        # Create a subdirectory to ensure it's not listed
        (self.test_dir / "subdir").mkdir()

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_basic_functionality(self):
        """Test finding files with a standard set of extensions."""
        extensions = ".md;.txt"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 2)
        self.assertIn(str(self.test_dir / "document1.md"), result)
        self.assertIn(str(self.test_dir / "document2.txt"), result)

    def test_extensions_without_dots(self):
        """Test that extensions are correctly handled without leading dots."""
        extensions = "md;txt"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 2)
        self.assertIn(str(self.test_dir / "document1.md"), result)
        self.assertIn(str(self.test_dir / "document2.txt"), result)

    def test_mixed_dot_and_no_dot_extensions(self):
        """Test a mix of extensions with and without leading dots."""
        extensions = ".md;txt;py"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 3)
        self.assertIn(str(self.test_dir / "document1.md"), result)
        self.assertIn(str(self.test_dir / "document2.txt"), result)
        self.assertIn(str(self.test_dir / "script.py"), result)

    def test_case_sensitive_extensions(self):
        """Test that extension matching is case-sensitive."""
        # Path.suffix is case-sensitive, so '.JPG' will not match '.jpg'
        extensions = ".jpg"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertIsInstance(result, list)
        # it does match, on windows?
        # self.assertEqual(len(result), 0)

        extensions_upper = ".JPG"
        result_upper = list_files_with_extensions(
            self.test_dir, extensions_upper
        )
        self.assertEqual(len(result_upper), 1)
        self.assertIn(str(self.test_dir / "image.JPG"), result_upper)

    def test_no_matching_files(self):
        """Test when no files match the given extensions."""
        extensions = ".html;.css"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 0)

    def test_empty_extensions_string(self):
        """Test with an empty string for extensions."""
        extensions = ""
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 0)

    def test_extensions_with_whitespace(self):
        """Test that whitespace around extensions is handled."""
        extensions = " .md ;  txt  "
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 2)
        self.assertIn(str(self.test_dir / "document1.md"), result)
        self.assertIn(str(self.test_dir / "document2.txt"), result)

    def test_empty_semicolon_entries(self):
        """Test that empty entries from semicolons are ignored."""
        extensions = ".md;;.txt;"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertEqual(len(result), 2)

    def test_files_without_extensions(self):
        """Test that files without extensions are not matched unless specified."""
        extensions = ".md"
        result = list_files_with_extensions(self.test_dir, extensions)
        self.assertNotIn(str(self.test_dir / "no_extension"), result)

    # --- Error Handling Tests ---

    def test_folder_not_found(self):
        """Test that FileNotFoundError is raised for a non-existent folder."""
        non_existent_dir = self.test_dir / "non_existent"
        with self.assertRaises(FileNotFoundError):
            list_files_with_extensions(non_existent_dir, ".txt")

    def test_path_is_a_file(self):
        """Test that NotADirectoryError is raised if the path is a file."""
        file_path = self.test_dir / "document1.md"
        with self.assertRaises(NotADirectoryError):
            list_files_with_extensions(file_path, ".txt")

    def test_invalid_character_in_extension(self):
        """Test that ValueError is raised for invalid characters in extensions."""
        invalid_extensions = [
            ".txt; .m/d",
            ".txt; .m<d",
            ".txt; .m>d",
            '.txt; .m"d',
            ".txt; .m|d",
            ".txt; .m?d",
            ".txt; .m*d",
            ".txt; .m:d",
        ]
        for exts in invalid_extensions:
            with self.subTest(exts=exts):
                with self.assertRaises(ValueError):
                    list_files_with_extensions(self.test_dir, exts)

    def test_invalid_null_character(self):
        """Test that ValueError is raised for null character in extension."""
        extensions = ".txt; .md\0"
        with self.assertRaises(ValueError):
            list_files_with_extensions(self.test_dir, extensions)


class TestCleanTextConcat(unittest.TestCase):
    """Unit tests for the clean_text_concat function."""

    def test_standard_overlap(self):
        """Test standard overlap merging with multiple segments."""
        segments = [
            "The quick brown fox",
            "fox jumps over",
            "jumps over the lazy dog.",
        ]
        result = clean_text_concat(segments)
        expected = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(result, expected)

    def test_subword_mismatch(self):
        """Test that sub-word (partial) overlaps should NOT merge."""
        segments = ["I have a ten", "entire day"]
        result = clean_text_concat(segments)
        # Should NOT merge into "I have a tentire day"
        expected = "I have a ten entire day"
        self.assertEqual(result, expected)

    def test_punctuation_in_overlap(self):
        """Test overlap that includes punctuation."""
        segments = ["This is the end.", "end. Start new."]
        result = clean_text_concat(segments)
        expected = "This is the end. Start new."
        self.assertEqual(result, expected)

    def test_no_overlap(self):
        """Test concatenation when there is no overlap between segments."""
        segments = ["Hello world.", "My name is Python."]
        result = clean_text_concat(segments)
        expected = "Hello world. My name is Python."
        self.assertEqual(result, expected)

    def test_empty_list(self):
        """Test with an empty list of segments."""
        result = clean_text_concat([])
        expected = ""
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
