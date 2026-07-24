#!/usr/bin/env python3
"""Read Spectre transient results written in Nutmeg ASCII format.

This is the Python 3 counterpart of ``import_data.m``.  The Spectre driver in
``main_stdp.py`` requests ``-format nutascii`` so that results can be consumed
without proprietary Python bindings.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def _result_file(path: Path) -> Path:
    """Resolve a raw path to the text file that contains transient values."""
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Spectre result does not exist: {path}")

    candidates = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".raw", ".tran", ".ascii"}
    )
    if not candidates:
        candidates = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not candidates:
        raise ValueError(f"No result file was found below {path}")
    return candidates[0]


def _real_value(token: str) -> float:
    """Parse a real or Nutmeg complex token, returning its real component."""
    token = token.strip().rstrip(",")
    if token.startswith("(") and token.endswith(")"):
        token = token[1:-1].split(",", 1)[0]
    return float(token)


def _canonical_signal(name: str) -> str:
    """Normalize Spectre/Nutmeg voltage names for case-insensitive matching."""
    normalized = name.strip().strip('"').replace(":", ".")
    if normalized.lower().startswith("v(") and normalized.endswith(")"):
        normalized = normalized[2:-1]
    return normalized.casefold()


def _parse_nutmeg_ascii(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    try:
        variables_at = next(
            i for i, line in enumerate(lines) if line.strip().lower().startswith("variables:")
        )
        values_at = next(
            i for i, line in enumerate(lines) if line.strip().lower().startswith("values:")
        )
    except StopIteration as exc:
        raise ValueError(
            f"{path} is not a Nutmeg ASCII result. Run Spectre with '-format nutascii'."
        ) from exc

    metadata: dict[str, str] = {}
    for line in lines[:variables_at]:
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()

    try:
        variable_count = int(metadata["no. variables"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid 'No. Variables' in {path}") from exc

    names: list[str] = []
    first_variable = lines[variables_at].split(":", 1)[1]
    variable_lines = [first_variable, *lines[variables_at + 1 : values_at]]
    for line in variable_lines:
        fields = line.split()
        if len(fields) >= 3 and _INTEGER_RE.match(fields[0]):
            names.append(fields[1].strip('"'))
    if len(names) != variable_count:
        raise ValueError(
            f"Expected {variable_count} variables in {path}, but found {len(names)} definitions"
        )

    # Cadence's Nutmeg ASCII writer packs several values on each physical line,
    # whereas classic Nutmeg commonly emits one value per continuation line.
    # Both are the same logical token stream: point index, then N values.
    first_values = lines[values_at].split(":", 1)[1]
    value_tokens = " ".join([first_values, *lines[values_at + 1 :]]).split()
    stride = variable_count + 1
    if len(value_tokens) % stride:
        raise ValueError(
            f"Incomplete Nutmeg value stream in {path}: {len(value_tokens)} tokens "
            f"is not divisible by point stride {stride}"
        )

    rows: list[list[float]] = []
    for offset in range(0, len(value_tokens), stride):
        point_token = value_tokens[offset]
        if not _INTEGER_RE.match(point_token):
            raise ValueError(f"Invalid point index '{point_token}' in {path}")
        point_index = int(point_token)
        if point_index != len(rows):
            raise ValueError(
                f"Expected point index {len(rows)} in {path}, found {point_index}"
            )
        rows.append(
            [_real_value(token) for token in value_tokens[offset + 1 : offset + stride]]
        )

    if not rows:
        raise ValueError(f"No transient data points were found in {path}")

    point_count_text = metadata.get("no. points")
    if point_count_text is not None and len(rows) != int(point_count_text):
        raise ValueError(
            f"Expected {point_count_text} points in {path}, but parsed {len(rows)}"
        )
    return np.asarray(rows, dtype=float), names


def import_data(
    file: str | Path,
    signal_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(data, names)`` from a Spectre Nutmeg ASCII result.

    ``data[:, 0]`` is time.  When ``signal_order`` is supplied, all remaining
    columns are rearranged to that exact order.  Signal matching ignores case
    and accepts both ``v(X1.node)`` and ``X1.node`` spellings.
    """
    path = _result_file(Path(file).expanduser().resolve())
    data, names = _parse_nutmeg_ascii(path)

    if signal_order is None:
        return data, names

    lookup = {_canonical_signal(name): index for index, name in enumerate(names)}
    try:
        time_index = next(
            index
            for index, name in enumerate(names)
            if _canonical_signal(name) in {"time", "frequency"}
        )
    except StopIteration:
        time_index = 0

    indices = [time_index]
    ordered_names = [names[time_index]]
    missing: list[str] = []
    for requested in signal_order:
        index = lookup.get(_canonical_signal(requested))
        if index is None:
            missing.append(requested)
        else:
            indices.append(index)
            ordered_names.append(requested)

    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise KeyError(f"Missing {len(missing)} requested signal(s): {preview}{suffix}")

    return data[:, indices], ordered_names


def write_csv(path: str | Path, data: np.ndarray, names: Iterable[str]) -> None:
    """Write parsed results to a conventional CSV file."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Spectre Nutmeg ASCII result file")
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data, names = import_data(args.result)
    print(f"Read {data.shape[0]} points and {data.shape[1]} variables from {args.result}")
    if args.csv:
        write_csv(args.csv, data, names)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
