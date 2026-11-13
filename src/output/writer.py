"""JSON output writer with automatic directory creation and error handling."""

import json
from pathlib import Path
from typing import Any

from src.errors.output_errors import OutputError


def write_output(
    results: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write validated function call results to a UTF-8 JSON file."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
    except OSError as error:
        raise OutputError(
            f"Error writing output to {output_path}: {error}"
        ) from error
