"""
Utilities to read/write to/from disc and print errors to console.
Errors are not propagated, but functions return null value.
"""

from pathlib import Path
from collections.abc import Callable

from pydantic import validate_call

from lmm.markdown.parse_markdown import Block

# Set up default logger
from .logging import get_logger, LoggerBase  # fmt: skip
logger: LoggerBase = get_logger(__name__)


def string_to_path_or_string(input_string: str) -> Path | str:
    """
    Takes a string as argument. If the string is one line, checks
    that the string codes for an existing file. If so, it returns a
    Path object for that file. Otherwise, it returns the string.

    A string is considered one line if it contains no newlines, or if
    it only has a single trailing newline character.

    Args:
        input_string: The input string to check

    Returns:
        Path object if the string represents an existing file,
        otherwise the original string
    """
    # Check if string is a single line (allowing for trailing \n)
    stripped_string = input_string.rstrip('\n\r')
    if '\n' in stripped_string or '\r' in stripped_string:
        return input_string

    # Try to create a Path object and check if it exists as a file
    try:
        potential_path = Path(stripped_string.strip())
        if potential_path.exists() and potential_path.is_file():
            return potential_path
    except (OSError, ValueError):
        # Invalid path characters or other path-related errors
        pass

    # Return original string if not a valid existing file
    return input_string


# Validate file name
def validate_file(
    source: str | Path, logger: LoggerBase = logger
) -> Path | None:
    """Returns: None for failure, Path object otherwise"""
    if not source:
        logger.warning("No file given")
        return None
    try:
        source_path = Path(source)
        if not source_path.exists():
            logger.error(f"File does not exist: {source}")
            return None
        if not source_path.is_file():
            logger.error(f"Not a file: {source}")
            return None
        if source_path.stat().st_size == 0:
            logger.warning(f"File is empty: {source}")
            return None
    except Exception as e:
        logger.error(f"Error accessing file {source}: {str(e)}")
        return None

    return source_path


# Interactive call of file -> file


@validate_call
def create_interface(
    f: Callable[[str, str], list[Block] | None], argv: list[str]
) -> None:
    """Waits for Enter key presses and handles Ctrl-C to
    enable interactive execution of the function f and for debugging.
    The first command-line argument is the markdown file on
    which the module acts. An optional second command-line
    argument is the file to which changes are saved. A third
    command line argument, if True, creates a loop for interactive
    editing.
    """
    if len(argv) > 1:
        filename = argv[1]
    else:
        print("Usage: first command line arg is source file")
        print("       second command line arg is save file (opt)")
        print("       third command line 'True' enters loop")
        return
    if len(argv) > 2:
        target = argv[2]
    else:
        target = filename

    if not validate_file(filename):
        return

    if len(argv) > 3:
        interactive = argv[3] == "True"
    else:
        interactive = False

    if not interactive:
        f(filename, target)
        return

    print(f"Press 'Enter' to execute the function on '{filename}'.")
    print("Press 'Ctrl-C' to exit.")

    try:
        input()
        while True:
            f(filename, target)
            # Waits for the user to press Enter
            input("Press 'Enter' to continue, 'Ctrl-C' to exit")
    except KeyboardInterrupt:
        print("\nCtrl-C detected. Exiting program.")
    except Exception as e:
        print("An unexpected error occurred: " + str(e))
    finally:
        print("Program gracefully terminated.")


def process_string_quotes(input_string: str) -> str:
    """
    Processes a string to ensure consistent internal quoting.

    Rules:
    - If the string contains the character ", except for the first
    and last character, replace it with ' and make sure the string
    starts and ends with ".
    - If the string contains the character ', make sure the string
    starts and ends with ".

    In short, the quote should create a string that can internally
    quote text with a consistent approach, starting from a string
    that may do so using different ways.

    Args:
        input_string: The string to be processed.

    Returns:
        The processed string with consistent quoting.
    """

    # Step 1: Remove any existing outer quotes to get the core content
    core_content = input_string

    # Check if the string starts and ends with double quotes
    if (
        len(core_content) >= 2
        and core_content.startswith('"')
        and core_content.endswith('"')
    ):
        core_content = core_content[1:-1]
    # Check if the string starts and ends with single quotes
    elif (
        len(core_content) >= 2
        and core_content.startswith("'")
        and core_content.endswith("'")
    ):
        core_content = core_content[1:-1]

    # Step 2: Handle internal double quotes
    # If the core content contains double quotes, replace them all
    # with single quotes
    # This ensures that internal quoting consistently uses single
    # quotes when the outer is double.
    if '"' in core_content:
        processed_internal_content = core_content.replace('"', "'")
    else:
        processed_internal_content = core_content

    # Step 3: Ensure the final string starts and ends with double
    # quotes
    # This applies to both cases: if it originally had internal
    # double quotes (now replaced with single), or if it had internal
    # single quotes, or no quotes.
    if "'" in processed_internal_content:
        final_string = '"' + processed_internal_content + '"'
    else:
        final_string = processed_internal_content

    return final_string


def append_postfix_to_filename(filename: str, postfix: str) -> str:
    """
    Appends a postfix string to the name of a file.

    Args:
        filename (str): The original name of the file (e.g.,
            "my_document.txt").
        postfix (str): The string to append (e.g., "_new").

    Returns:
        str: The new filename with the postfix appended.
    """
    import os

    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}{postfix}{extension}"
    return new_filename


def parse_external_boolean(value: object) -> bool:
    """Sanitize externally given boolean"""
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'no', ''):
            return False
        # Handle other string interpretations as needed
    # Fallback to Python's default truthiness for other types
    return bool(value)


def list_files_with_extensions(
    folder_path: str | Path, extensions: str | list[str]
) -> list[str]:
    """
    Lists all files in a given folder that match a set of specified extensions.

    Args:
        folder_path (str | Path): The full path to the folder to search.
        extensions (str | list[str]): A single semicolon-separated string of 
            file extensions (e.g., ".txt;.md;py") OR a standard list of strings 
            (e.g., ['.txt', 'md']). Extensions may or may not start with a dot.

    Returns:
        A list of full paths (as strings) for all matching files. Returns an
        empty list if no files are found.

    Raises:
        FileNotFoundError: If the specified folder_path does not exist.
        NotADirectoryError: If the specified folder_path is not a directory.
        ValueError: If the extensions string contains invalid characters for
            a filename.
    """
    # --- 1. Validate folder path ---
    p_folder = Path(folder_path)
    if not p_folder.exists():
        raise FileNotFoundError(
            f"The folder does not exist: '{folder_path}'"
        )
    if not p_folder.is_dir():
        raise NotADirectoryError(
            f"The specified path is not a directory: '{folder_path}'"
        )

    # --- 2. Process and Normalize Extensions ---
    raw_extensions: list[str] = []
    
    if isinstance(extensions, str):
        # Handle the semicolon-separated string input
        if not extensions:
            return []
        raw_extensions = extensions.split(';')
    elif isinstance(extensions, list):  # type: ignore (always met)
        # Handle the standard list input
        raw_extensions = extensions
    else:
        # Catch unexpected types
        raise TypeError(
            "Unreacheable code reached. Extensions supposed to be " 
            "a string (semicolon-separated) or a list of strings."
        )

    # Define invalid characters for filenames
    # This remains critical for security and robustness.
    invalid_chars = r'<>:"/\|?*' + "".join(map(chr, range(32)))

    processed_extensions: set[str] = set()
    for ext in raw_extensions:
        ext = str(ext).strip() # Ensure it's a string and strip whitespace
        if not ext:
            continue

        # Check for invalid characters
        if any(char in invalid_chars for char in ext):
            raise ValueError(
                f"Invalid character found in extension '{ext}'. Extensions cannot "
                f"contain any of the following: {invalid_chars}"
            )

        # Prepend dot if missing and store in the set
        if not ext.startswith('.'):
            processed_extensions.add('.' + ext.lower()) # Added .lower() for case-insensitivity
        else:
            processed_extensions.add(ext.lower()) # Added .lower() for case-insensitivity
            
    if not processed_extensions:
        return []

    # --- 3. Find matching files ---
    # Note: Using Path.suffix is case-sensitive, so we lower-case it here 
    # to match the lower-cased processed_extensions set.
    matching_files: list[str] = [
        str(file_path)
        for file_path in p_folder.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in processed_extensions
    ]

    return matching_files


def check_allowed_content(
    input_string: str, allowed_list: list[str]
) -> bool:
    """
    Extracts strings delimited by single quotes from input_string and checks
    if any of them are in the allowed_list.

    Args:
        input_string: The string to extract quoted content from.
        allowed_list: List of strings to check against.

    Returns:
        True if any extracted string is in allowed_list, False otherwise.
    """
    import re

    # Firth just check is in list
    if input_string in allowed_list:
        return True

    # Fallback, extract all strings delimited by single quotes
    pattern = r"'([^']*)'"
    extracted_strings = re.findall(pattern, input_string)

    # Check if any extracted string is in the allowed list
    for extracted in extracted_strings:
        if extracted in allowed_list:
            return True

    return False


def clean_text_concat(text_segments: list[str]) -> str:
    """
    Concatenates a list of strings, merging overlapping tails/heads
    if the overlap constitutes at least one whole word.

    The merge condition requires:
    1. The tail of text A matches the head of text B.
    2. The match represents a complete word boundary on both sides:
       - The character preceding the overlap in A must not be alphanumeric (or A starts with the overlap).
       - The character following the overlap in B must not be alphanumeric (or B ends with the overlap).
    3. The overlap contains at least one alphanumeric character (to ensure it's "at least a word"
       and not just whitespace/punctuation).

    Args:
        text_segments: A list of strings to concatenate.

    Returns:
        A single concatenated string with overlaps merged.
    """
    if not text_segments:
        return ""

    # Initialize with the first segment
    result_text = text_segments[0]

    for next_segment in text_segments[1:]:
        result_text = _merge_segments(result_text, next_segment)

    return result_text


def _merge_segments(left: str, right: str) -> str:
    """
    Helper function to merge two strings handling strict word-boundary overlap.
    """
    # Optimization: We only care about the end of 'left' and start of 'right'
    # We limit the check to the smaller of the two lengths.
    max_overlap = min(len(left), len(right))

    # Iterate from largest possible overlap down to 1
    for i in range(max_overlap, 0, -1):
        # 1. Check strict string equality
        candidate = left[-i:]
        if right.startswith(candidate):

            # 2. Check "At least a word" content
            # The overlap must contain at least one alphanumeric character.
            # Otherwise, we might merge on just ". " or " ".
            if not any(char.isalnum() for char in candidate):
                continue

            # 3. Check Left Boundary (in 'left' string)
            # If the character immediately before the overlap is alphanumeric,
            # we are slicing a word in half. (e.g. "content" -> overlap "tent" -> 'n' is alnum)
            left_boundary_ok = True
            if len(left) > i:
                char_before = left[-(i + 1)]
                if char_before.isalnum():
                    left_boundary_ok = False

            # 4. Check Right Boundary (in 'right' string)
            # If the character immediately after the overlap is alphanumeric,
            # we are slicing a word in half.
            right_boundary_ok = True
            if len(right) > i:
                char_after = right[i]
                if char_after.isalnum():
                    right_boundary_ok = False

            if left_boundary_ok and right_boundary_ok:
                # Valid overlap found. Merge and return.
                # We take 'left' as is, and append the non-overlapping part of 'right'.
                return left + right[i:]

    # If no valid overlap found, strictly concatenate with a separator if needed.
    # Logic: If left ends with space or right starts with space, just concat.
    # Otherwise add a space.
    if left.endswith(" ") or right.startswith(" "):
        return left + right
    return left + " " + right


if __name__ == "__main__":
    # Test Cases to verify edge cases discussed

    # 1. Standard Overlap
    segments1 = [
        "The quick brown fox",
        "fox jumps over",
        "jumps over the lazy dog.",
    ]
    print(f"Test 1 (Standard): {clean_text_concat(segments1)}")
    # Expected: "The quick brown fox jumps over the lazy dog."

    # 2. Sub-word (Partial) Overlap - Should NOT merge
    segments2 = ["I have a ten", "entire day"]
    print(
        f"Test 2 (Sub-word mismatch): {clean_text_concat(segments2)}"
    )
    # Expected: "I have a ten entire day" (Not "I have a tentire day")

    # 3. Punctuation included in overlap
    segments3 = ["This is the end.", "end. Start new."]
    print(f"Test 3 (Punctuation): {clean_text_concat(segments3)}")
    # Expected: "This is the end. Start new."

    # 4. No Overlap
    segments4 = ["Hello world.", "My name is Python."]
    print(f"Test 4 (No Overlap): {clean_text_concat(segments4)}")
    # Expected: "Hello world. My name is Python."

    # 5. Empty list
    print(f"Test 5 (Empty): '{clean_text_concat([])}'")
