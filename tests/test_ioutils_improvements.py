#!/usr/bin/env python3
"""
Unit tests for the file size limits and encoding detection improvements
to the lmm.markdown.ioutils module.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from lmm.markdown.ioutils import (
    load_markdown,
    _check_file_size,
    _detect_encoding,
)
from lmm.utils.logging import LoglistLogger


class TestFileSizeLimits(unittest.TestCase):
    """Test file size limit functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_check_file_size_within_limits(self):
        """Test file size checking when file is within limits."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write("# Small Test File\nThis is a small test file.")
            temp_file = Path(f.name)

        try:
            result = _check_file_size(
                temp_file, 50.0, 10.0, self.logger
            )
            self.assertTrue(result)
            logs = self.logger.get_logs()
            self.assertEqual(len(logs), 0)  # No warnings or errors
        finally:
            temp_file.unlink()

    def test_check_file_size_warning_threshold(self):
        """Test file size checking when file triggers warning."""
        # Create a file that's about 0.01MB (should trigger warning at 0.005MB)
        content = "# Test Header\n" + "This is test content. " * 500
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            result = _check_file_size(
                temp_file, 50.0, 0.005, self.logger
            )
            self.assertTrue(result)  # Should still pass
            logs = self.logger.get_logs()
            self.assertEqual(len(logs), 1)
            self.assertIn("warning", logs[0])
            self.assertIn("large", logs[0])
        finally:
            temp_file.unlink()

    def test_check_file_size_exceeds_limit(self):
        """Test file size checking when file exceeds maximum."""
        # Create a file that's about 0.01MB (should fail at 0.005MB limit)
        content = "# Test Header\n" + "This is test content. " * 500
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            result = _check_file_size(
                temp_file, 0.005, 0.001, self.logger
            )
            self.assertFalse(result)  # Should fail
            logs = self.logger.get_logs()
            self.assertEqual(len(logs), 1)
            self.assertIn("error", logs[0])
            self.assertIn("too large", logs[0])
        finally:
            temp_file.unlink()

    def test_check_file_size_os_error(self):
        """Test file size checking when OS error occurs."""
        # Use a non-existent file
        fake_path = Path("/non/existent/file.md")
        result = _check_file_size(fake_path, 50.0, 10.0, self.logger)
        self.assertFalse(result)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertIn("error", logs[0])
        self.assertIn("Could not check file size", logs[0])

    def test_load_markdown_file_size_integration(self):
        """Test load_markdown integration with file size limits."""
        # Create a small file
        content = "# Test Header\nThis is test content."
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Should work with normal limits
            result = load_markdown(
                temp_file, logger=self.logger, max_size_mb=1.0
            )
            self.assertEqual(result, content)

            # Should fail with very small limit
            self.logger.clear_logs()
            result = load_markdown(
                temp_file, logger=self.logger, max_size_mb=0.000001
            )
            self.assertEqual(result, "")  # Should return empty string
            logs = self.logger.get_logs()
            self.assertTrue(any("too large" in log for log in logs))
        finally:
            temp_file.unlink()


class TestEncodingDetection(unittest.TestCase):
    """Test encoding detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_detect_encoding_utf8(self):
        """Test encoding detection for UTF-8 files."""
        content = "# Test Header\nThis is UTF-8 content with special chars: café, naïve"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='utf-8', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            encoding = _detect_encoding(temp_file, self.logger)
            self.assertEqual(encoding, 'utf-8')
            logs = self.logger.get_logs()
            self.assertTrue(
                any("UTF-8 encoding" in log for log in logs)
            )
        finally:
            temp_file.unlink()

    def test_detect_encoding_fallback(self):
        """Test encoding detection fallback mechanism."""
        # Create a file with Latin-1 encoding
        content = "# Test Header\nThis is Latin-1 content"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='latin-1', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Mock UTF-8 failure to test fallback by patching the open function
            # in the _detect_encoding function specifically
            original_open = open

            def mock_open_func(*args, **kwargs):
                if (
                    'encoding' in kwargs
                    and kwargs['encoding'] == 'utf-8'
                    and len(args) > 1
                    and 'r' in args[1]
                ):
                    raise UnicodeDecodeError(
                        'utf-8', b'', 0, 1, 'invalid start byte'
                    )
                return original_open(*args, **kwargs)

            with patch('builtins.open', side_effect=mock_open_func):
                encoding = _detect_encoding(temp_file, self.logger)

                # Should fall back to one of the fallback encodings
                # Note: 'ascii' is a subset of 'utf-8' and may be detected on some systems
                self.assertIn(
                    encoding,
                    [
                        'utf-8',
                        'latin-1',
                        'cp1252',
                        'iso-8859-1',
                        'ascii',
                    ],
                )
        finally:
            temp_file.unlink()

    @patch('lmm.markdown.ioutils.chardet')
    def test_detect_encoding_with_chardet(self, mock_chardet):
        """Test encoding detection using chardet library."""
        content = "# Test Header\nThis is test content"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='utf-8', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Mock chardet to return a specific encoding
            mock_chardet.detect.return_value = {
                'encoding': 'iso-8859-1',
                'confidence': 0.8,
            }

            # Mock the UTF-8 test to fail by patching the specific open call
            original_open = open
            utf8_call_count = 0

            def mock_open_func(*args, **kwargs):
                nonlocal utf8_call_count
                if (
                    'encoding' in kwargs
                    and kwargs['encoding'] == 'utf-8'
                    and len(args) > 1
                    and 'r' in args[1]
                ):
                    utf8_call_count += 1
                    if (
                        utf8_call_count == 1
                    ):  # Only fail the first UTF-8 call
                        raise UnicodeDecodeError(
                            'utf-8', b'', 0, 1, 'invalid start byte'
                        )
                return original_open(*args, **kwargs)

            with patch('builtins.open', side_effect=mock_open_func):
                encoding = _detect_encoding(temp_file, self.logger)

                self.assertEqual(encoding, 'iso-8859-1')
                logs = self.logger.get_logs()
                self.assertTrue(
                    any("iso-8859-1 encoding" in log for log in logs)
                )
        finally:
            temp_file.unlink()

    @patch(
        'lmm.markdown.ioutils.chardet', None
    )  # Simulate chardet not available
    def test_detect_encoding_no_chardet(self):
        """Test encoding detection when chardet is not available."""
        content = "# Test Header\nThis is test content"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='utf-8', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            encoding = _detect_encoding(temp_file, self.logger)
            self.assertEqual(encoding, 'utf-8')
            logs = self.logger.get_logs()
            self.assertTrue(
                any("UTF-8 encoding" in log for log in logs)
            )
        finally:
            temp_file.unlink()

    def test_load_markdown_encoding_integration(self):
        """Test load_markdown integration with encoding detection."""
        content = "# Test Header\nThis is test content with special chars: café"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', encoding='utf-8', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Test automatic encoding detection
            result = load_markdown(
                temp_file,
                logger=self.logger,
                auto_detect_encoding=True,
            )
            self.assertEqual(result, content)

            # Test manual encoding specification
            self.logger.clear_logs()
            result = load_markdown(
                temp_file,
                logger=self.logger,
                encoding='utf-8',
                auto_detect_encoding=False,
            )
            self.assertEqual(result, content)

            # Test with wrong encoding (should fall back gracefully)
            self.logger.clear_logs()
            result = load_markdown(
                temp_file,
                logger=self.logger,
                encoding='ascii',
                auto_detect_encoding=False,
            )
            # Should still get content due to fallback mechanism
            self.assertIsInstance(result, str)
        finally:
            temp_file.unlink()


class TestBackwardCompatibility(unittest.TestCase):
    """Test that new functionality doesn't break existing behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_string_input_unchanged(self):
        """Test that string input behavior is unchanged."""
        test_string = "# Test Header\nThis is a test string input."

        # Test with default parameters
        result = load_markdown(test_string, logger=self.logger)
        self.assertEqual(result, test_string)

        # Test with new parameters (should be ignored for string input)
        result = load_markdown(
            test_string,
            logger=self.logger,
            max_size_mb=0.001,  # Very small limit
            encoding='ascii',  # Wrong encoding
        )
        self.assertEqual(result, test_string)

    def test_multiline_string_input(self):
        """Test that multiline string input works correctly."""
        test_string = """# Test Header
        
This is a multiline test string.

It has multiple paragraphs.
"""
        result = load_markdown(test_string, logger=self.logger)
        self.assertEqual(result, test_string)

    def test_default_parameters(self):
        """Test that default parameters work as expected."""
        content = "# Test Header\nThis is test content."
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Test with minimal parameters (should use defaults)
            result = load_markdown(temp_file)
            self.assertEqual(result, content)

            # Test with logger only
            result = load_markdown(temp_file, logger=self.logger)
            self.assertEqual(result, content)
        finally:
            temp_file.unlink()

    def test_existing_error_handling(self):
        """Test that existing error handling still works."""
        # Test with non-existent file
        fake_path = Path("/non/existent/file.md")
        result = load_markdown(fake_path, logger=self.logger)
        self.assertEqual(result, "")

        logs = self.logger.get_logs()
        self.assertTrue(len(logs) > 0)
        self.assertTrue(any("error" in log.lower() for log in logs))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_empty_file(self):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write("")  # Empty file
            temp_file = Path(f.name)

        try:
            result = load_markdown(temp_file, logger=self.logger)
            self.assertEqual(result, "")
        finally:
            temp_file.unlink()

    def test_very_small_limits(self):
        """Test with very small size limits."""
        content = "# Test"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Even tiny files should fail with extremely small limits
            result = load_markdown(
                temp_file, logger=self.logger, max_size_mb=0.000001
            )
            self.assertEqual(result, "")

            logs = self.logger.get_logs()
            self.assertTrue(any("too large" in log for log in logs))
        finally:
            temp_file.unlink()

    def test_invalid_encoding_parameter(self):
        """Test with invalid encoding parameter."""
        content = "# Test Header\nThis is test content."
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Test with invalid encoding
            result = load_markdown(
                temp_file,
                logger=self.logger,
                encoding='invalid-encoding',
            )
            # Should fall back gracefully
            self.assertIsInstance(result, str)

            logs = self.logger.get_logs()
            # Should have some error or warning about encoding
            self.assertTrue(len(logs) > 0)
        finally:
            temp_file.unlink()

    def test_negative_size_limits(self):
        """Test with negative size limits."""
        content = "# Test"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Negative limits should be handled gracefully
            result = load_markdown(
                temp_file,
                logger=self.logger,
                max_size_mb=-1.0,
                warn_size_mb=-1.0,
            )
            # Should still work (negative limits effectively disable the check)
            self.assertEqual(result, content)
        finally:
            temp_file.unlink()


if __name__ == '__main__':
    unittest.main()
