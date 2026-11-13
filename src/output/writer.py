import json
from typing import Any
from pathlib import Path

from src.errors.output_errors import OutputError


def write_output(
        results: list[dict[str, Any]],
        output_path: str
) -> None:

    try:
        Path(output_path).parent.mkdir(
            parents=True, exist_ok=True
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results,
                      f,
                      indent=2,
                      ensure_ascii=False
                      )
    except OSError as e:
        raise OutputError(
            f"Error writing output to {output_path}: {e}"
            ) from e
