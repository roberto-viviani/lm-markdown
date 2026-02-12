"""Test scan.py"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from lmm.scan.scan import blocklist_scan, markdown_scan, scan, save_scan
from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    ErrorBlock,
    parse_markdown_text,
)
from lmm.utils.logging import LoggerBase


class MockLogger(LoggerBase):
    """Mock logger for testing that inherits from LoggerBase"""
    
    def __init__(self):
        self._warning_mock = Mock()
        self._error_mock = Mock()
        self._info_mock = Mock()
        self._debug_mock = Mock()
        self._critical_mock = Mock()
        self._level = 0
    
    def set_level(self, level: int) -> None:
        self._level = level
    
    def get_level(self) -> int:
        return self._level
    
    def warning(self, message: str) -> None:
        self._warning_mock(message)
    
    def error(self, message: str) -> None:
        self._error_mock(message)
    
    def info(self, message: str) -> None:
        self._info_mock(message)
    
    def debug(self, message: str) -> None:
        self._debug_mock(message)
    
    def critical(self, message: str) -> None:
        self._critical_mock(message)


test_md_only_heading = """# This is my title"""
test_md_no_header_text = (
    """This is a simple file with just text: no header, no heading."""
)
test_md_no_header_heading = """
# A main heading {class = any}

This is a markdown file with a heading, no header.
"""
test_md_no_title = """
---
author: john
...

This is a markdown file with a starting metadata block that has no title tag.
"""


class TestHeaders(unittest.TestCase):

    def test_empty(self):
        blocks: list[Block] = blocklist_scan([])

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore

    def test_empty2(self):
        blocks: list[Block] = parse_markdown_text("")
        blocks = blocklist_scan(blocks)

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore

    def test_only_heading(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_only_heading
        )
        blocks = blocklist_scan(blocks)

        self.assertTrue(len(blocks), 1)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "This is my title")  # type: ignore

    def test_no_header_text(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_no_header_text
        )
        blocks = blocklist_scan(blocks)

        self.assertTrue(len(blocks), 2)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore
        self.assertEqual(
            blocks[1].get_content(), test_md_no_header_text
        )

    def test_md_no_header_heading(self):
        blocks: list[Block] = parse_markdown_text(
            test_md_no_header_heading
        )
        blocks = blocklist_scan(blocks)

        self.assertTrue(len(blocks), 3)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "A main heading")  # type: ignore
        self.assertEqual(
            blocks[-1].get_content(),
            "This is a markdown file with a heading, no header.",
        )

    def test_md_no_title(self):
        blocks: list[Block] = parse_markdown_text(test_md_no_title)
        blocks = blocklist_scan(blocks)

        self.assertTrue(len(blocks), 3)
        self.assertIsInstance(blocks[0], HeaderBlock)
        self.assertIn('title', blocks[0].content)
        self.assertEqual(blocks[0].content['title'], "Title")  # type: ignore
        self.assertEqual(blocks[0].content['author'], "john")  # type: ignore
        self.assertEqual(
            blocks[-1].get_content(),
            "This is a markdown file with a starting metadata block that has no title tag.",
        )

    def test_invalid_markdown(self):
        blocks: list[Block] = markdown_scan("nonexistent.md")
        self.assertEqual(len(blocks), 0)


class TestBlocklistEdgeCases(unittest.TestCase):
    """Test edge cases for blocklist_scan"""

    def test_blocklist_with_only_errors(self):
        """Test that blocklist_scan returns ErrorBlock-only list as-is"""
        error_block = ErrorBlock(content="Parse error", comment="Invalid")
        blocks: list[Block] = [error_block]
        
        result = blocklist_scan(blocks)
        
        # Should return list as-is without adding HeaderBlock
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ErrorBlock)
        self.assertEqual(result[0].content, "Parse error")

    def test_blocklist_custom_default_title(self):
        """Test using custom default title"""
        blocks: list[Block] = parse_markdown_text(test_md_no_header_text)
        result = blocklist_scan(blocks, default_title="CustomTitle")
        
        self.assertIsInstance(result[0], HeaderBlock)
        self.assertEqual(result[0].content['title'], "CustomTitle")  # type: ignore


class TestMarkdownScan(unittest.TestCase):
    """Test markdown_scan function with file I/O"""

    def test_markdown_scan_title_from_stem(self):
        """Test that markdown_scan uses filename stem as title"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "my_document.md"
            test_file.write_text(test_md_no_header_text, encoding='utf-8')
            
            blocks = markdown_scan(test_file, save=False)
            
            self.assertGreater(len(blocks), 0)
            self.assertIsInstance(blocks[0], HeaderBlock)
            self.assertEqual(blocks[0].content['title'], "my_document")  # type: ignore

    def test_markdown_scan_save_to_original(self):
        """Test save=True behavior"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text(test_md_no_header_text, encoding='utf-8')
            
            blocks = markdown_scan(test_file, save=True)
            
            # Verify blocks were returned and file was modified
            self.assertGreater(len(blocks), 0)
            content = test_file.read_text(encoding='utf-8')
            self.assertIn("---", content)
            self.assertIn("title:", content)

    def test_markdown_scan_save_to_different_file(self):
        """Test save='path' behavior"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.md"
            target_file = Path(tmpdir) / "target.md"
            source_file.write_text(test_md_no_header_text, encoding='utf-8')
            
            blocks = markdown_scan(source_file, save=target_file)
            
            # Verify blocks were returned and target file was created
            self.assertGreater(len(blocks), 0)
            self.assertTrue(target_file.exists())
            content = target_file.read_text(encoding='utf-8')
            self.assertIn("---", content)
            self.assertIn("title:", content)

    def test_markdown_scan_no_save(self):
        """Test save=False behavior"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            original_content = test_md_no_header_text
            test_file.write_text(original_content, encoding='utf-8')
            
            blocks = markdown_scan(test_file, save=False)
            
            # Verify blocks were returned but file was NOT modified
            self.assertGreater(len(blocks), 0)
            self.assertEqual(
                test_file.read_text(encoding='utf-8'), 
                original_content
            )


class TestScanWrapper(unittest.TestCase):
    """Test the scan() wrapper function"""

    def test_scan_normal_operation(self):
        """Test scan wrapper with valid file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text(test_md_only_heading, encoding='utf-8')
            
            # Should not raise exception
            scan(test_file, save=False)

    def test_scan_exception_handling(self):
        """Test that exceptions are caught and logged"""
        mock_logger = MockLogger()
        
        # This should trigger an error (nonexistent file)
        scan("nonexistent_file.md", save=False, logger=mock_logger)
        
        # Verify logger.error was called
        mock_logger._error_mock.assert_called()
        call_args = str(mock_logger._error_mock.call_args)
        self.assertIn("nonexistent_file.md", call_args)


class TestLoggerUsage(unittest.TestCase):
    """Test that logger is used correctly"""

    def test_logger_warning_on_empty(self):
        """Test logger.warning is called when no blocks found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "empty.md"
            test_file.write_text("", encoding='utf-8')
            
            mock_logger = MockLogger()
            blocks = markdown_scan(test_file, save=False, logger=mock_logger)
            
            # Should log warning about no blocks
            mock_logger._warning_mock.assert_called()
            self.assertEqual(len(blocks), 0)

    @patch('lmm.scan.scan.mkd.blocklist_haserrors')
    def test_logger_warning_on_errors(self, mock_haserrors):
        """Test logger.warning is called when errors found"""
        # Mock blocklist_haserrors to return True
        mock_haserrors.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text(test_md_only_heading, encoding='utf-8')
            
            mock_logger = MockLogger()
            blocks = markdown_scan(test_file, save=False, logger=mock_logger)
            
            # Verify blocks were returned and warning was logged
            self.assertGreater(len(blocks), 0)
            warning_calls = [str(call) for call in mock_logger._warning_mock.call_args_list]
            error_warning_found = any("Errors found" in str(call) for call in warning_calls)
            self.assertTrue(error_warning_found)


class TestSaveScan(unittest.TestCase):
    """Test save_scan function with timestamp verification"""

    def test_save_scan_basic(self):
        """Test basic save to new file"""
        from lmm.scan.scan_keys import LAST_MODIFIED_KEY
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create blocks
            blocks = parse_markdown_text(test_md_only_heading)
            blocks = blocklist_scan(blocks)
            
            # Save to new file
            output_file = Path(tmpdir) / "output.md"
            mock_logger = MockLogger()
            success = save_scan(output_file, blocks, logger=mock_logger)
            
            # Verify success
            self.assertTrue(success)
            self.assertTrue(output_file.exists())
            
            # Verify timestamp was added
            self.assertIn(LAST_MODIFIED_KEY, blocks[0].content)

    def test_save_scan_updates_timestamp(self):
        """Test that timestamp is added/updated on save"""
        from lmm.scan.scan_keys import LAST_MODIFIED_KEY
        
        with tempfile.TemporaryDirectory() as tmpdir:
            blocks = parse_markdown_text(test_md_only_heading)
            blocks = blocklist_scan(blocks)
            
            # First save
            output_file = Path(tmpdir) / "test.md"
            mock_logger = MockLogger()
            save_scan(output_file, blocks, logger=mock_logger)
            first_timestamp = blocks[0].content[LAST_MODIFIED_KEY]
            
            # Second save - timestamp should update
            import time
            time.sleep(0.01)  # Small delay to ensure different timestamp
            save_scan(output_file, blocks, logger=mock_logger)
            second_timestamp = blocks[0].content[LAST_MODIFIED_KEY]
            
            # Timestamps should be different
            self.assertNotEqual(first_timestamp, second_timestamp)

    def test_save_scan_verify_success(self):
        """Test load, modify, save with matching timestamp"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save initial file
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text(test_md_only_heading, encoding='utf-8')
            
            # Load with markdown_scan (adds timestamp)
            blocks = markdown_scan(test_file, save=True)
            
            # Modify blocks
            blocks[0].content['author'] = 'Test Author'
            
            # Save with verification should succeed
            mock_logger = MockLogger()
            success = save_scan(test_file, blocks, verify_unchanged=True, logger=mock_logger)
            
            self.assertTrue(success)

    def test_save_scan_verify_failure(self):
        """Test timestamp mismatch detection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file and save with save_scan so it has a timestamp
            test_file = Path(tmpdir) / "test.md"
            blocks = parse_markdown_text(test_md_only_heading)
            blocks = blocklist_scan(blocks)
            save_scan(test_file, blocks, verify_unchanged=False)
            
            # Simulate external modification by loading and saving again with save_scan
            # This updates the file's timestamp
            import time
            time.sleep(0.01)
            blocks_new = markdown_scan(test_file, save=False)
            blocks_new[0].content['modified'] = 'by external process'
            save_scan(test_file, blocks_new, verify_unchanged=False)
            
            # Try to save original blocks with old timestamp - should fail
            mock_logger = MockLogger()
            success = save_scan(test_file, blocks, verify_unchanged=True, logger=mock_logger)
            
            # Should fail due to timestamp mismatch
            self.assertFalse(success)
            # Should have logged warning
            mock_logger._warning_mock.assert_called()

    def test_save_scan_no_verification(self):
        """Test save with verification disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text(test_md_only_heading, encoding='utf-8')
            
            # Load and save to update timestamp
            blocks = markdown_scan(test_file, save=True)
            
            # Modify file externally
            markdown_scan(test_file, save=True)
            
            # Save original blocks without verification should succeed
            mock_logger = MockLogger()
            success = save_scan(test_file, blocks, verify_unchanged=False, logger=mock_logger)
            
            self.assertTrue(success)

    def test_save_scan_missing_timestamp(self):
        """Test handling of blocks without timestamp"""
        from lmm.scan.scan_keys import LAST_MODIFIED_KEY
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create blocks without timestamp
            blocks = parse_markdown_text(test_md_only_heading)
            blocks = blocklist_scan(blocks)
            
            # Remove timestamp if it exists
            if LAST_MODIFIED_KEY in blocks[0].content:
                del blocks[0].content[LAST_MODIFIED_KEY]
            
            # Save should succeed and add timestamp
            output_file = Path(tmpdir) / "test.md"
            mock_logger = MockLogger()
            success = save_scan(output_file, blocks, logger=mock_logger)
            
            self.assertTrue(success)
            self.assertIn(LAST_MODIFIED_KEY, blocks[0].content)

    def test_save_scan_empty_blocks(self):
        """Test rejection of empty block list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.md"
            mock_logger = MockLogger()
            
            # Try to save empty list
            success = save_scan(output_file, [], logger=mock_logger)
            
            self.assertFalse(success)
            mock_logger._error_mock.assert_called()

    def test_save_scan_invalid_path(self):
        """Test handling of invalid save paths"""
        # Create blocks
        blocks = parse_markdown_text(test_md_only_heading)
        blocks = blocklist_scan(blocks)
        
        # Try to save to invalid path
        invalid_path = Path("Z:\\nonexistent\\path\\test.md")
        mock_logger = MockLogger()
        success = save_scan(invalid_path, blocks, logger=mock_logger)
        
        # Should fail gracefully
        self.assertFalse(success)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        unittest.main()

