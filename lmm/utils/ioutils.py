"""
Utilities to read/write to/from disc and print errors to console.
Errors are not propagated, but functions return null value.
"""

from pathlib import Path

# Set up default logger
from .logging import get_logger, ILogger  # fmt: skip
logger: ILogger = get_logger(__name__)


def string_to_path_or_string(input_string: str) -> Path | str:
    """
    Takes a string as argument. If the string is one line, checks that the
    string codes for an existing file. If so, it returns a Path object for
    that file. Otherwise, it returns the string.

    A string is considered one line if it contains no newlines, or if it
    only has a single trailing newline character.

    Args:
        input_string: The input string to check

    Returns:
        Path object if the string represents an existing file, otherwise the original string
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
    source: str | Path, logger: ILogger = logger
) -> None | Path:
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
from typing import Callable
from lmm.markdown import Block

from pydantic import validate_call


@validate_call
def create_interface(
    f: Callable[[str, str], list[Block]], argv: list[str]
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
