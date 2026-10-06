from pathlib import Path
from src.errors.input_errors import InputFileError


def load_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    except OSError as error:
        raise InputFileError(f"Error reading file "
                             f"'{path}': {error}") from error
