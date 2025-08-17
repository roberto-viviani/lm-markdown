"""
Test scan.py.

Usage:
    test_scan_extended
    test_scan_extended  file_to_scan.md output_scan_file.md
"""

from lmm.scan.scan import markdown_scan

if __name__ == "__main__":
    import sys

    # Check if first command line parameter was given
    if len(sys.argv) > 1:
        sourcefile = sys.argv[1]
    else:
        print("Test markdown scan with following test file:")
        print(
            "test_scan_extended.md    (invalid metadata and headings)"
        )
        print("Add a second command line argument as output file")
        print("")
        sourcefile = ""

    # Check if second command line parameter was given
    if len(sys.argv) > 2:
        save = sys.argv[2]
    else:
        save = False

    # Call the scan function with the parameters
    markdown_scan(sourcefile, save)
