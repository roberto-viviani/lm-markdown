from pathlib import Path

try:
    import pdfplumber

    # Import LAParams explicitly to prevent buffer initialization errors
    from pdfminer.layout import LAParams
except ImportError:
    print(
        "Could not load pdfplumber. Please install it with 'pip install pdfplumber'."
    )
    exit()

# --- Configuration ---
# The core model for text extraction is based on detecting structured elements
# and then applying Markdown rules.


def apply_markdown_heuristics(page_text: str) -> str:
    """
    Applies simple heuristics to convert extracted raw text into basic Markdown format.

    This function attempts to:
    1. Clean up excessive whitespace.
    2. Ensure proper paragraph separation (Markdown requires two newlines).
    3. (Placeholder for advanced logic) Detect headings or lists based on patterns.
    """
    # 1. Normalize line endings and cleanup extra spaces
    lines = page_text.strip().split('\n')

    markdown_lines: list[str] = []

    # Simple logic: assume lines separated by only one newline are part of the
    # same paragraph, and lines separated by blank lines are new paragraphs.
    current_paragraph: list[str] = []

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            # End of a paragraph block, join and add to markdown_lines
            if current_paragraph:
                markdown_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            # Add an extra newline for Markdown paragraph separation
            markdown_lines.append("")
        else:
            # Simple list/heading detection placeholder
            if stripped_line.startswith(('1.', 'a.', '*', '-')):
                # If it looks like a list item, treat it as a new line item
                if current_paragraph:
                    markdown_lines.append(" ".join(current_paragraph))
                    current_paragraph = []
                markdown_lines.append(stripped_line)
            else:
                # Part of the current paragraph
                current_paragraph.append(stripped_line)

    # Add the last pending paragraph
    if current_paragraph:
        markdown_lines.append(" ".join(current_paragraph))

    return "\n".join(markdown_lines).strip()


def convert_pdf_to_md(pdf_path: Path, output_dir: Path) -> None:
    """
    Converts a single PDF file into a Markdown file.

    Args:
        pdf_path: Path object to the input PDF file.
        output_dir: Path object for the output directory.
    """
    print(f"Processing: {pdf_path.name}")

    markdown_content: list[str] = []
    output_filename = pdf_path.stem + ".md"
    output_path = output_dir / output_filename

    # Initialize default LAParams to fix the "unpack requires a buffer..." error
    # By explicitly passing this object, we prevent pdfminer.six from performing
    # an internal initialization step that fails on some PDFs.
    default_laparams = LAParams()

    try:
        # Pass the initialized default_laparams to pdfplumber.open()
        with pdfplumber.open(
            pdf_path,
            laparams=default_laparams.__dict__,
            repair=True,
            repair_setting="default",
            gs_path="C:/Program Files/gs/gs10.06.0/bin/gswin64c.exe",
        ) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                # Extract text retaining layout structure (via 'layout' argument)
                raw_text = page.extract_text(
                    x_tolerance=2, y_tolerance=2, layout=True
                )

                if raw_text:
                    # Apply markdown formatting heuristics
                    formatted_text = apply_markdown_heuristics(
                        raw_text
                    )
                    markdown_content.append(formatted_text)

                # Insert a metadata block between pages to track content
                if i < total_pages - 1:
                    markdown_content.append(
                        f"\n\n---\npage: {i + 1}\n"
                        f"total_pages: {total_pages}\n---\n\n"
                    )

        # Write the final content to the Markdown file
        output_path.write_text(
            "\n".join(markdown_content), encoding="utf-8"
        )
        print(f"Successfully converted to: {output_path}")

    except Exception as e:
        print(
            f"ERROR: Failed to process {pdf_path.name}. Reason: {e}"
        )


def convert_folder_to_markdown(input_dir: str, output_dir: str):
    """
    Reads all PDF files from an input directory and converts them to Markdown
    in an output directory.

    Args:
        input_dir: The path to the folder containing PDF files.
        output_dir: The path where the Markdown files will be saved.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # Create the output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ensured: {output_path}")

    # Find all PDF files in the input directory
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in pdf_files:
        convert_pdf_to_md(pdf_file, output_path)

    print("\nProcessing complete.")


if __name__ == "__main__":
    # --- Example Usage ---
    # NOTE: To run this code, ensure you have a folder named 'pdfs_to_process'
    # containing some PDF files, and have the 'pdfplumber' library installed.

    INPUT_FOLDER = "pdfimport"
    OUTPUT_FOLDER = "markdowndocs"

    # 1. Create dummy directories for testing
    Path(INPUT_FOLDER).mkdir(exist_ok=True)

    # 2. IMPORTANT: Please add some PDF files to the 'pdfs_to_process' folder
    # to test the script.

    # 3. Run the conversion
    print("--- Starting PDF Conversion Process ---")
    convert_folder_to_markdown(INPUT_FOLDER, OUTPUT_FOLDER)
    print("--- Process Finished ---")

# You will find the converted .md files in the 'markdown_output' folder.
