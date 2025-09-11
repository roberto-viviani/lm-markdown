#!/usr/bin/env python3
"""
Additional edge case tests for the lmm.markdown.ioutils module.
These tests cover permission errors, invalid paths, and logger behavior.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from lmm.markdown.ioutils import (
    load_markdown,
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


if __name__ == '__main__':
    unittest.main()
