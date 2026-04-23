"""
Excel bank statement loader for ArthaSense.

Reads .xlsx / .xls files exported by Indian banks and returns a raw DataFrame
with original column names, ready for BankStatementPreprocessor.clean().

Strategy:
  1. Peek at the first 20 rows without headers to score each row for column-header signals.
  2. Re-read with the detected header row so pandas assigns correct column names.
  3. Strip trailing/leading whitespace from all string cells.
  4. Drop fully-empty rows and columns.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import IO, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Keywords that strongly suggest a row is a column-header row
_HEADER_KW = {
    "date", "narration", "description", "particulars", "transaction",
    "amount", "debit", "credit", "withdrawal", "deposit", "balance",
    "txn", "ref", "cheque", "chq", "remarks", "details", "dr", "cr",
}


def _header_score(row: pd.Series) -> int:
    """Count how many header keywords appear in this row."""
    text = " ".join(
        str(v).lower().strip() for v in row if pd.notna(v) and str(v).strip()
    )
    return sum(1 for kw in _HEADER_KW if re.search(r"\b" + kw + r"\b", text))


def _detect_header_row(df_peek: pd.DataFrame, max_scan: int = 20) -> int:
    """
    Scan up to `max_scan` rows and return the index of the most header-like row.
    Returns 0 if no clear candidate is found.
    """
    best_row, best_score = 0, 1       # require at least 2 matches
    for i in range(min(max_scan, len(df_peek))):
        score = _header_score(df_peek.iloc[i])
        if score > best_score:
            best_score, best_row = score, i
    logger.debug("Header detection: row %d scored %d", best_row, best_score)
    return best_row


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names:
      - Strip whitespace and newlines
      - Replace internal whitespace runs with single space
      - Rename "Unnamed: N" columns that result from merged/empty header cells
    """
    new_cols = []
    for i, c in enumerate(df.columns):
        name = re.sub(r"\s+", " ", str(c).strip())
        if re.match(r"^Unnamed:\s*\d+$", name, re.I):
            name = f"_col{i}"
        new_cols.append(name)
    df.columns = new_cols
    return df


def load_excel_raw(file_source: Union[str, Path, IO]) -> pd.DataFrame:
    """
    Read an Excel bank statement and return a raw DataFrame.

    The result should be passed to BankStatementPreprocessor.clean() to produce
    the final standardised schema.

    Parameters
    ----------
    file_source : str | Path | file-like
        .xlsx or .xls file path or Streamlit UploadedFile.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with original column names (cleaned of whitespace).

    Raises
    ------
    ValueError
        If the file cannot be read or contains no data rows.
    """
    # Determine engine from extension (if available)
    engine = None
    if hasattr(file_source, "name"):
        ext = Path(file_source.name).suffix.lower()
        engine = "xlrd" if ext == ".xls" else "openpyxl"
    elif isinstance(file_source, (str, Path)):
        ext = Path(file_source).suffix.lower()
        engine = "xlrd" if ext == ".xls" else "openpyxl"

    # ── Peek pass: read first 20 rows without headers ─────────────────────
    try:
        df_peek = pd.read_excel(
            file_source,
            header=None,
            nrows=20,
            engine=engine,
        )
    except Exception as exc:
        raise ValueError(f"Cannot open Excel file: {exc}") from exc

    header_row_idx = _detect_header_row(df_peek)
    logger.info("Excel header detected at row %d", header_row_idx)

    # ── Rewind file pointer before second read ─────────────────────────────
    if hasattr(file_source, "seek"):
        file_source.seek(0)

    # ── Full read with correct header row ─────────────────────────────────
    try:
        df = pd.read_excel(
            file_source,
            header=header_row_idx,
            engine=engine,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse Excel with header at row {header_row_idx}: {exc}"
        ) from exc

    # ── Drop fully empty rows / columns ───────────────────────────────────
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Excel file contains no data rows after the header. "
            "Please check that the correct sheet is exported."
        )

    # ── Clean column names ─────────────────────────────────────────────────
    df = _clean_columns(df)

    # ── Strip whitespace from all string-valued cells ──────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)

    logger.info("Excel loaded: %d rows × %d cols — header at row %d",
                len(df), len(df.columns), header_row_idx)
    return df
