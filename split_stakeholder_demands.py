import argparse
import os
import re
import sys
from typing import Optional

import pandas as pd


def normalize_header(text: str) -> str:
    """
    Normalize header text by collapsing whitespace (including \r and \n)
    and trimming. Used for robust column matching when headers contain
    accidental line breaks or trailing spaces.
    """
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def find_target_column(columns, preferred_name: str, fallback_index_q_based: int = 16) -> str:
    """
    Attempt to find the target column by:
    1) Exact match
    2) Case-insensitive match after whitespace normalization
    3) Fallback to the 17th column (Excel column Q) if present
    """
    # 1) Exact match first
    for col in columns:
        if str(col) == preferred_name:
            return col

    # 2) Case-insensitive normalized match
    normalized_target = normalize_header(preferred_name).lower()
    normalized_map = {col: normalize_header(col).lower() for col in columns}
    for original_col, norm in normalized_map.items():
        if norm == normalized_target:
            return original_col

    # 3) Fallback to index for column Q (0-based index 16)
    if len(columns) > fallback_index_q_based:
        return columns[fallback_index_q_based]

    raise KeyError(
        f"Target column '{preferred_name}' not found and fallback index {fallback_index_q_based} is out of range."
    )


def split_into_lines(cell_value: Optional[str]) -> list:
    """
    Split a cell's text into logical lines using any line break variant.
    Strips whitespace and removes empty results.
    Also trims leading bullet characters or dashes commonly used in lists.
    """
    if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
        return []

    text = str(cell_value)
    # Split on CRLF / LF / CR
    parts = re.split(r"(?:\r\n|\n|\r)+", text)

    cleaned = []
    for part in parts:
        # Remove common bullet prefixes and surrounding whitespace
        item = re.sub(r"^[\s\-\u2022\u2023\u25E6\u2043\u2219\*\•·–—]+", "", part).strip()
        if item:
            cleaned.append(item)
    return cleaned


def explode_demands(input_path: str, output_path: str, column_name_hint: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read all as objects to preserve textual content; Excel engine inferred
    df = pd.read_excel(input_path, dtype=object)

    target_col = find_target_column(df.columns, column_name_hint)

    # Build list-of-lines per row then explode
    lines_series = df[target_col].apply(split_into_lines)
    exploded = df.copy()
    exploded["__line"] = lines_series
    exploded = exploded.explode("__line").reset_index(drop=True)

    # Drop rows where no line was present after splitting
    exploded = exploded[exploded["__line"].notna()]

    # Replace original column with the split line content
    exploded[target_col] = exploded["__line"].astype(str)
    exploded = exploded.drop(columns=["__line"])

    # Write to Excel
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    exploded.to_excel(output_path, index=False)

    print(
        f"Wrote {len(exploded)} rows to {output_path} (from {len(df)} input rows).\n"
        f"Column exploded: '{target_col}'."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Explode multiline demands into separate rows.")
    parser.add_argument(
        "--input",
        default=os.path.join("input", "ConsultationWorkshopsExtractedData.xlsx"),
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("input", "ConsultationWorkshopsExtractedData_demands_exploded.xlsx"),
        help="Path to output Excel file",
    )
    parser.add_argument(
        "--column",
        default="Stakeholder Groups specific demands",
        help=(
            "Column name hint. Matching is robust to trailing carriage returns and whitespace; "
            "fallback is Excel column Q."
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        explode_demands(args.input, args.output, args.column)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

