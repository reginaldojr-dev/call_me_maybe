"""File loading utilities with graceful I/O error handling."""

from pathlib import Path

from src.errors.input_errors import InputFileError


def load_file(file_path: Path) -> str:
    """Read and return the UTF-8 text content of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as error:
        raise InputFileError(f"File not found: {file_path}") from error
    except OSError as error:
        raise InputFileError(
            f"Error reading file {file_path}: {error}"
        ) from error
