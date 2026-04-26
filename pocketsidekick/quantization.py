"""Simple quantization export utilities for MVP workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def export_quantization_config(
    output_path: str,
    method: str = "int8",
    bits: int = 8,
    group_size: int = 128,
    symmetric: bool = True,
    extra: Dict[str, Any] | None = None,
) -> str:
    """Write a portable quantization config JSON file."""
    data = {
        "format_version": 1,
        "method": method,
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
    }
    if extra:
        data["extra"] = extra

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(destination)
