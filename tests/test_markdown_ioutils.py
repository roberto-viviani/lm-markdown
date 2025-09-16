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
    save_markdown,
    report_error_blocks,
)
from lmm.markdown.parse_markdown import (
    ErrorBlock,
    TextBlock,
    HeaderBlock,
    HeadingBlock,
)
from lmm.utils.logging import (
    LoglistLogger,
    ConsoleLogger,
    ExceptionConsoleLogger,
    FileLogger,
)


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
            result = load_markdown(temp_file, logger=self.logger)
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


class TestPermissionErrors(unittest.TestCase):
    """Test handling of permission-related errors."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_permission_denied_read(self):
        """Test handling of permission denied when reading files."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write("# Test Content")
            temp_file = Path(f.name)

        try:
            # Mock permission error
            with patch('pathlib.Path.read_text') as mock_read:
                mock_read.side_effect = PermissionError(
                    "Permission denied"
                )

                result = load_markdown(temp_file, logger=self.logger)

                self.assertEqual(result, "")
                logs = self.logger.get_logs()
                self.assertTrue(
                    any(
                        "I/O error reading file" in log
                        for log in logs
                    )
                )
                self.assertTrue(
                    any("Permission denied" in log for log in logs)
                )
        finally:
            temp_file.unlink()

    def test_permission_denied_write(self):
        """Test handling of permission denied when writing files."""
        # Try to write to a read-only directory (simulate with mock)
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError(
                "Permission denied"
            )

            result = save_markdown(
                "/readonly/test.md", "# Test", logger=self.logger
            )

            self.assertFalse(result)
            logs = self.logger.get_logs()
            self.assertTrue(
                any(
                    "I/O error saving markdown" in log for log in logs
                )
            )
            self.assertTrue(
                any("Permission denied" in log for log in logs)
            )

    def test_file_in_use_error(self):
        """Test handling of file in use errors."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write("# Test Content")
            temp_file = Path(f.name)

        try:
            # Mock file in use error
            with patch('pathlib.Path.read_text') as mock_read:
                mock_read.side_effect = OSError(
                    "File is being used by another process"
                )

                result = load_markdown(temp_file, logger=self.logger)

                self.assertEqual(result, "")
                logs = self.logger.get_logs()
                self.assertTrue(
                    any(
                        "I/O error reading file" in log
                        for log in logs
                    )
                )
        finally:
            temp_file.unlink()


class TestInvalidPaths(unittest.TestCase):
    """Test handling of invalid file paths."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_nonexistent_file(self):
        """Test loading from non-existent file."""
        fake_path = Path("/this/path/does/not/exist.md")
        result = load_markdown(fake_path, logger=self.logger)

        self.assertEqual(result, "")
        logs = self.logger.get_logs()
        self.assertTrue(any("error" in log.lower() for log in logs))

    def test_directory_instead_of_file(self):
        """Test loading from a directory path instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            result = load_markdown(dir_path, logger=self.logger)

            self.assertEqual(result, "")
            logs = self.logger.get_logs()
            self.assertTrue(any("Not a file" in log for log in logs))

    def test_invalid_characters_in_path(self):
        """Test handling of paths with invalid characters."""
        # This test is platform-specific, so we'll mock the behavior
        invalid_path = Path("invalid\x00path.md")

        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = OSError("Invalid path")

            result = load_markdown(invalid_path, logger=self.logger)

            self.assertEqual(result, "")
            logs = self.logger.get_logs()
            self.assertTrue(
                any("error" in log.lower() for log in logs)
            )

    def test_save_to_nonexistent_directory(self):
        """Test saving to a non-existent directory (should create it)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = (
                Path(temp_dir) / "new_dir" / "subdir" / "test.md"
            )

            result = save_markdown(
                nested_path, "# Test Content", logger=self.logger
            )

            self.assertTrue(result)
            self.assertTrue(nested_path.exists())
            self.assertEqual(
                nested_path.read_text(), "# Test Content"
            )

    def test_save_to_invalid_path(self):
        """Test saving to an invalid path."""
        # Mock an invalid path scenario
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Invalid path")

            result = save_markdown(
                "/invalid\x00/path.md", "# Test", logger=self.logger
            )

            self.assertFalse(result)
            logs = self.logger.get_logs()
            self.assertTrue(
                any(
                    "I/O error saving markdown" in log for log in logs
                )
            )


class TestLoggerBehavior(unittest.TestCase):
    """Test behavior with different logger implementations."""

    def test_console_logger_behavior(self):
        """Test behavior with ConsoleLogger."""
        logger = ConsoleLogger(__name__)

        # Test with non-existent file - string_to_path_or_string returns the string if file doesn't exist
        result = load_markdown("/nonexistent/file.md", logger=logger)
        self.assertEqual(
            result, "/nonexistent/file.md"
        )  # Returns the string itself
        # ConsoleLogger outputs to console, so we can't easily capture it
        # but we can verify it doesn't crash

    def test_exception_logger_behavior(self):
        """Test behavior with ExceptionConsoleLogger."""
        logger = ExceptionConsoleLogger(__name__)

        # Test with non-existent file - string_to_path_or_string returns the string if file doesn't exist
        result = load_markdown("/nonexistent/file.md", logger=logger)
        self.assertEqual(
            result, "/nonexistent/file.md"
        )  # Returns the string itself

    def test_file_logger_behavior(self):
        """Test behavior with FileLogger."""
        with tempfile.NamedTemporaryFile(
            suffix='.log', delete=False
        ) as log_file:
            log_path = Path(log_file.name)

        try:
            logger = FileLogger(__name__, log_path)

            # Test with non-existent file - string_to_path_or_string returns the string if file doesn't exist
            result = load_markdown(
                "/nonexistent/file.md", logger=logger
            )
            self.assertEqual(
                result, "/nonexistent/file.md"
            )  # Returns the string itself

            # Verify log was written (though no error should be logged for string input)
            self.assertTrue(log_path.exists())
        finally:
            if log_path.exists():
                try:
                    log_path.unlink()
                except PermissionError:
                    # On Windows, the file might still be in use by the logger
                    pass

    def test_loglist_logger_filtering(self):
        """Test LoglistLogger with different filtering levels."""
        logger = LoglistLogger()

        # Generate different types of log messages
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Test different filter levels
        all_logs = logger.get_logs(0)  # All messages
        self.assertEqual(len(all_logs), 4)

        no_info = logger.get_logs(1)  # No info messages
        self.assertEqual(len(no_info), 3)
        self.assertFalse(
            any("Info message" in log for log in no_info)
        )

        errors_only = logger.get_logs(3)  # Only errors and critical
        self.assertEqual(len(errors_only), 2)
        self.assertTrue(
            any("Error message" in log for log in errors_only)
        )
        self.assertTrue(
            any("Critical message" in log for log in errors_only)
        )


class TestErrorBlockReporting(unittest.TestCase):
    """Test error block reporting logic with various scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = LoglistLogger()

    def test_no_error_blocks(self):
        """Test reporting with no error blocks."""
        blocks = [
            HeaderBlock(content={'title': 'Test'}),
            HeadingBlock(level=1, content='Heading'),
            TextBlock(content='Text content'),
        ]

        result = report_error_blocks(blocks, self.logger)

        self.assertEqual(result, blocks)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 0)

    def test_single_error_block_file_loading(self):
        """Test reporting with single error block (file loading failure)."""
        error_block = ErrorBlock(
            content="Could not load file",
            errormsg="File not found",
            origin="test.md",
        )
        blocks = [error_block]

        result = report_error_blocks(blocks, self.logger)

        self.assertEqual(result, [])
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertIn("error", logs[0])
        self.assertIn("Could not load file", logs[0])
        self.assertIn("File not found", logs[0])

    def test_multiple_error_blocks_parsing(self):
        """Test reporting with multiple error blocks (parsing failures)."""
        error1 = ErrorBlock(
            content="Invalid YAML",
            errormsg="Syntax error at line 5",
            origin="---\ninvalid: yaml: content\n---",
        )
        error2 = ErrorBlock(
            content="Invalid heading",
            errormsg="Empty heading content",
            origin="###   ",
        )
        text_block = TextBlock(content="Valid text")

        blocks = [error1, text_block, error2]

        result = report_error_blocks(blocks, self.logger)

        # Should return only non-error blocks
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextBlock)

        # Should log warnings for each error
        logs = self.logger.get_logs()
        warning_logs = [log for log in logs if "warning" in log]
        self.assertEqual(len(warning_logs), 2)

        # Check content of warnings - the error block content is not included, only error messages and origins
        all_warnings = "\n".join(warning_logs)
        self.assertIn("Syntax error at line 5", all_warnings)
        self.assertIn("Empty heading content", all_warnings)
        self.assertIn(
            "---\ninvalid: yaml: content\n---", all_warnings
        )
        self.assertIn("###   ", all_warnings)

    def test_mixed_blocks_with_errors(self):
        """Test reporting with mixed block types including errors."""
        blocks = [
            HeaderBlock(content={'title': 'Test Document'}),
            HeadingBlock(level=1, content='Section 1'),
            ErrorBlock(
                content="Parse error", errormsg="Invalid syntax"
            ),
            TextBlock(content="Valid content"),
            ErrorBlock(
                content="Another error", origin="problematic content"
            ),
            HeadingBlock(level=2, content='Subsection'),
        ]

        result = report_error_blocks(blocks, self.logger)

        # Should return only non-error blocks
        self.assertEqual(len(result), 4)
        error_blocks_in_result = [
            b for b in result if isinstance(b, ErrorBlock)
        ]
        self.assertEqual(len(error_blocks_in_result), 0)

        # Should log warnings for errors
        logs = self.logger.get_logs()
        warning_logs = [log for log in logs if "warning" in log]
        self.assertEqual(len(warning_logs), 2)

    def test_error_block_without_message(self):
        """Test error block reporting without error message."""
        error_block = ErrorBlock(
            content="Generic error", origin="some problematic content"
        )
        blocks = [TextBlock(content="Text"), error_block]

        result = report_error_blocks(blocks, self.logger)

        self.assertEqual(len(result), 1)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 1)
        # The error block content is not included in the warning message, only the origin
        self.assertIn("some problematic content", logs[0])

    def test_error_block_without_origin(self):
        """Test error block reporting without origin content."""
        error_block = ErrorBlock(
            content="Error without origin",
            errormsg="Something went wrong",
        )
        blocks = [error_block, TextBlock(content="Text")]

        result = report_error_blocks(blocks, self.logger)

        self.assertEqual(len(result), 1)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 1)
        # The error block content is not included in the warning message, only the error message
        self.assertIn("Something went wrong", logs[0])
        # Should not contain offending content section
        self.assertNotIn("Offending content", logs[0])

    def test_empty_block_list(self):
        """Test error block reporting with empty block list."""
        result = report_error_blocks([], self.logger)

        self.assertEqual(result, [])
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 0)


class TestSaveMarkdown(unittest.TestCase):

    def test_save_markdown(self):
        from lmm.markdown.parse_markdown import (
            Block,
            parse_markdown_text,
        )
        import io

        text = """
---
title: my document
---

This is some text in a markdown document.
"""
        blocks: list[Block] = parse_markdown_text(text)
        string_stream = io.StringIO()
        logger = LoglistLogger()
        save_markdown(string_stream, blocks, logger)
        self.assertTrue(logger.count_logs(1) == 0)
        self.assertEqual(string_stream.getvalue(), text.strip())


if __name__ == '__main__':
    unittest.main()
