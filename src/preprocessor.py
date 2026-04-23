"""
Bank Statement CSV Preprocessor for Indian bank formats.

Handles various CSV formats from Indian banks including SBI, HDFC, ICICI, Axis, etc.
Supports multiple date formats, encodings, and amount representations including
the Indian lakh numbering system.
"""

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Date formats commonly used in Indian bank statements
DATE_FORMATS: List[str] = [
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d-%b-%Y",
    "%d/%b/%Y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%B-%Y",
]

# Encodings commonly encountered in Indian bank CSV exports
ENCODINGS: List[str] = ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]

# Values treated as NaN / missing in Indian bank statements
NAN_REPRESENTATIONS: List[str] = [
    "",
    "-",
    "--",
    "NIL",
    "nil",
    "Nil",
    "N/A",
    "n/a",
    "NA",
    "na",
    "NaN",
    "nan",
    "NULL",
    "null",
    ".",
    "0.00",
    "0",
    "00",
    "0.0",
]

# Common column name mappings to standardized names
COLUMN_ALIASES: Dict[str, str] = {
    "txn date": "date",
    "txn_date": "date",
    "transaction date": "date",
    "transaction_date": "date",
    "trans date": "date",
    "trans_date": "date",
    "value date": "date",
    "value_date": "date",
    "posting date": "date",
    "posting_date": "date",
    "dt": "date",
    "narration": "description",
    "particulars": "description",
    "details": "description",
    "transaction details": "description",
    "transaction_details": "description",
    "remark": "description",
    "remarks": "description",
    "transaction remarks": "description",
    "description": "description",
    "desc": "description",
    "ref no": "reference",
    "ref_no": "reference",
    "reference no": "reference",
    "reference_no": "reference",
    "reference number": "reference",
    "chq no": "reference",
    "cheque no": "reference",
    "cheque_no": "reference",
    "chq/ref no": "reference",
    "withdrawal": "debit",
    "withdrawal amt": "debit",
    "withdrawal_amt": "debit",
    "debit amount": "debit",
    "debit_amount": "debit",
    "debit amt": "debit",
    "debit_amt": "debit",
    "dr": "debit",
    "dr amount": "debit",
    "dr_amount": "debit",
    "deposit": "credit",
    "deposit amt": "credit",
    "deposit_amt": "credit",
    "credit amount": "credit",
    "credit_amount": "credit",
    "credit amt": "credit",
    "credit_amt": "credit",
    "cr": "credit",
    "cr amount": "credit",
    "cr_amount": "credit",
    "balance": "balance",
    "closing balance": "balance",
    "closing_balance": "balance",
    "running balance": "balance",
    "available balance": "balance",
    "amount": "amount_raw",
}


def format_indian_currency(amount: float) -> str:
    """
    Format a number into Indian currency format with the Rupee symbol.

    Indian numbering groups the last three digits, then groups of two thereafter.
    Example: 123456789.50 -> "₹12,34,56,789"

    Args:
        amount: The numeric amount to format.

    Returns:
        Formatted string like "₹1,23,456".
    """
    is_negative = amount < 0
    amount = abs(amount)
    integer_part = int(amount)
    integer_str = str(integer_part)

    if len(integer_str) <= 3:
        formatted = integer_str
    else:
        last_three = integer_str[-3:]
        remaining = integer_str[:-3]
        # Group remaining digits in pairs from right to left
        groups: List[str] = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        groups.reverse()
        formatted = ",".join(groups) + "," + last_three

    result = f"₹{formatted}"
    if is_negative:
        result = f"-{result}"
    return result


def _clean_amount_string(value: Any) -> Optional[float]:
    """
    Clean an amount string from Indian bank statements and convert to float.

    Handles:
    - Rupee symbol removal
    - Comma removal (including Indian lakh format like 1,23,456.78)
    - Whitespace stripping
    - Dr/Cr suffix detection
    - Various NaN representations

    Args:
        value: Raw value from CSV cell.

    Returns:
        Cleaned float value, or None if the value represents missing data.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()

    # Check against known NaN representations
    if s in NAN_REPRESENTATIONS:
        return None

    # Remove rupee symbol and whitespace
    s = s.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("INR", "").strip()

    # Remove all spaces within the number
    s = s.replace(" ", "")

    # Detect Dr/Cr suffix (some banks use a single Amount column with Dr/Cr)
    dr_cr_flag: Optional[str] = None
    dr_match = re.search(r"(Dr\.?|DR\.?)$", s)
    cr_match = re.search(r"(Cr\.?|CR\.?)$", s)
    if dr_match:
        dr_cr_flag = "debit"
        s = s[: dr_match.start()].strip()
    elif cr_match:
        dr_cr_flag = "credit"
        s = s[: cr_match.start()].strip()

    # Remove commas (handles Indian lakh format: 1,23,456.78)
    s = s.replace(",", "")

    # Handle parentheses as negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    # Remove trailing minus
    if s.endswith("-"):
        s = "-" + s[:-1]

    try:
        result = float(s)
    except (ValueError, TypeError):
        logger.warning("Could not parse amount value: %r", value)
        return None

    return result


def _parse_date(value: Any, formats: Optional[List[str]] = None) -> Optional[pd.Timestamp]:
    """
    Attempt to parse a date value using multiple Indian bank date formats.

    Args:
        value: Raw date value from CSV.
        formats: List of strftime formats to try. Defaults to DATE_FORMATS.

    Returns:
        Parsed Timestamp or None if parsing fails.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, pd.Timestamp):
        return value

    s = str(value).strip()
    if not s or s in NAN_REPRESENTATIONS:
        return None

    if formats is None:
        formats = DATE_FORMATS

    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue

    # Fallback: let pandas infer
    try:
        return pd.to_datetime(s, dayfirst=True)
    except (ValueError, TypeError):
        logger.warning("Could not parse date value: %r", value)
        return None


def _normalize_merchant_name(name: Any) -> str:
    """
    Normalize a merchant / description string.

    Steps:
    - Strip leading/trailing whitespace
    - Convert to title case
    - Remove trailing numbers, hashes, and reference codes

    Args:
        name: Raw description string.

    Returns:
        Cleaned merchant name.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""

    s = str(name).strip()

    if not s:
        return ""

    # Remove trailing reference numbers, hashes, and IDs
    # Patterns like: "Amazon #12345", "Swiggy 987654321", "UPI/1234567890"
    s = re.sub(r"[#/]\s*[\d]+\s*$", "", s)
    s = re.sub(r"\s+\d{4,}\s*$", "", s)
    s = re.sub(r"\s*-\s*\d{4,}\s*$", "", s)

    # Strip again after removal
    s = s.strip()

    # Title case
    s = s.title()

    return s


def _detect_header_row(raw_text: str, max_rows: int = 20) -> int:
    """
    Detect the actual header row in a CSV that may have extra header/info rows.

    Looks for the first row that contains date-like or amount-like column headers.

    Args:
        raw_text: Raw CSV text content.
        max_rows: Maximum rows to scan.

    Returns:
        0-based index of the header row.
    """
    header_keywords = {
        "date", "txn", "transaction", "narration", "particulars",
        "debit", "credit", "amount", "withdrawal", "deposit",
        "balance", "description", "details", "dr", "cr",
    }

    lines = raw_text.strip().split("\n")
    for i, line in enumerate(lines[:max_rows]):
        cells = [c.strip().strip('"').strip("'").lower() for c in line.split(",")]
        matches = sum(1 for c in cells if any(kw in c for kw in header_keywords))
        if matches >= 2:
            return i

    return 0


class BankStatementPreprocessor:
    """
    Preprocessor for Indian bank statement CSVs.

    Handles various bank-specific formats, date representations, encoding issues,
    and amount formatting (including the Indian lakh numbering system).

    Usage:
        preprocessor = BankStatementPreprocessor()
        df = preprocessor.load_csv("statement.csv")
        cleaned = preprocessor.clean(df)
        stats = preprocessor.get_summary_stats(cleaned)
    """

    def __init__(self) -> None:
        self._raw_df: Optional[pd.DataFrame] = None
        self._cleaned_df: Optional[pd.DataFrame] = None

    def load_csv(
        self,
        source: Union[str, Path, Any],
        encoding: Optional[str] = None,
        date_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a bank statement CSV from a file path or Streamlit UploadedFile.

        Tries multiple encodings and detects the header row automatically.

        Args:
            source: File path (str or Path) or Streamlit UploadedFile object.
            encoding: Force a specific encoding. If None, tries multiple encodings.
            date_format: Hint for date format. If None, auto-detects.

        Returns:
            Raw DataFrame loaded from the CSV.

        Raises:
            ValueError: If the CSV cannot be read with any encoding.
        """
        raw_text: Optional[str] = None
        encodings_to_try = [encoding] if encoding else ENCODINGS

        # Handle Streamlit UploadedFile (has .read() / .getvalue() methods)
        if hasattr(source, "read") or hasattr(source, "getvalue"):
            logger.info("Loading from uploaded file object")
            if hasattr(source, "getvalue"):
                raw_bytes = source.getvalue()
            else:
                raw_bytes = source.read()
                # Reset the file pointer if possible
                if hasattr(source, "seek"):
                    source.seek(0)

            if isinstance(raw_bytes, str):
                raw_text = raw_bytes
            else:
                for enc in encodings_to_try:
                    try:
                        raw_text = raw_bytes.decode(enc)
                        logger.info("Decoded uploaded file with encoding: %s", enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

            if raw_text is None:
                raise ValueError("Could not decode the uploaded file with any supported encoding")

        else:
            # Handle file path
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info("Loading from file path: %s", file_path)

            for enc in encodings_to_try:
                try:
                    raw_text = file_path.read_text(encoding=enc)
                    logger.info("Read file with encoding: %s", enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue

            if raw_text is None:
                raise ValueError(
                    f"Could not read {file_path} with any supported encoding: {encodings_to_try}"
                )

        # Detect the header row
        header_row = _detect_header_row(raw_text)
        if header_row > 0:
            logger.info("Detected header at row %d, skipping %d rows", header_row, header_row)

        try:
            df = pd.read_csv(
                StringIO(raw_text),
                skiprows=header_row,
                na_values=NAN_REPRESENTATIONS,
                keep_default_na=True,
                skipinitialspace=True,
                on_bad_lines="warn",
            )
        except Exception as exc:
            logger.error("Failed to parse CSV: %s", exc)
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

        # Drop completely empty rows and columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        logger.info("Loaded CSV with shape: %s", df.shape)
        self._raw_df = df
        return df

    # ------------------------------------------------------------------
    # Universal file loader (CSV + PDF + Excel)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ext(source: Any) -> str:
        """Return the lowercase file extension of *source* (e.g. '.pdf')."""
        if hasattr(source, "name"):
            return Path(source.name).suffix.lower()
        if isinstance(source, (str, Path)):
            return Path(source).suffix.lower()
        return ".csv"

    def load_file(self, source: Any) -> pd.DataFrame:
        """
        Auto-detect file type and return a fully standardised, cleaned DataFrame
        ready for the ML pipeline — equivalent to ``clean(load_csv(source))``
        for CSV inputs.

        Supported extensions
        --------------------
        .csv              → existing CSV pipeline (load_csv + clean)
        .pdf              → PDF parser (pdfplumber, text-based PDFs only)
        .xlsx / .xls      → Excel parser (openpyxl / xlrd) then clean

        Parameters
        ----------
        source : file path, str, Path, or Streamlit UploadedFile

        Returns
        -------
        pd.DataFrame
            Standardised DataFrame with columns:
            date, description, debit, credit, balance, amount,
            transaction_type, month_year

        Raises
        ------
        ImportError        if pdfplumber / openpyxl is not installed
        ScannedPDFError    if PDF is image-only
        PDFParseError      if no transactions found in PDF
        ValueError         if file cannot be read / parsed
        """
        ext = self._get_ext(source)

        if ext == ".pdf":
            from src.pdf_parser import parse_pdf
            return parse_pdf(source)

        elif ext in (".xlsx", ".xls"):
            from src.excel_parser import load_excel_raw
            raw = load_excel_raw(source)
            # Rewind if the loader consumed the file pointer
            if hasattr(source, "seek"):
                source.seek(0)
            return self.clean(raw)

        else:
            # Default: CSV (or unknown extension treated as CSV)
            raw = self.load_csv(source)
            return self.clean(raw)

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and standardize a raw bank statement DataFrame.

        Steps:
        1. Standardize column names
        2. Parse dates
        3. Clean amounts (remove symbols, handle lakh format)
        4. Create unified amount and transaction_type columns
        5. Normalize merchant/description names
        6. Sort by date, deduplicate, add month_year

        Args:
            df: Raw DataFrame. If None, uses the last loaded DataFrame.

        Returns:
            Cleaned DataFrame.

        Raises:
            ValueError: If no DataFrame is available to clean.
        """
        if df is None:
            df = self._raw_df
        if df is None:
            raise ValueError("No DataFrame to clean. Call load_csv() first or pass a DataFrame.")

        df = df.copy()

        # --- Step 1: Standardize column names ---
        df.columns = [str(c).strip().lower() for c in df.columns]
        logger.info("Original columns: %s", list(df.columns))

        rename_map: Dict[str, str] = {}
        for col in df.columns:
            normalized = col.strip().lower()
            if normalized in COLUMN_ALIASES:
                target = COLUMN_ALIASES[normalized]
                # Avoid overwriting if we already mapped a column to this target
                if target not in rename_map.values():
                    rename_map[col] = target
        df.rename(columns=rename_map, inplace=True)
        logger.info("Columns after renaming: %s", list(df.columns))

        # --- Handle single 'amount_raw' column with Dr/Cr suffix ---
        if "amount_raw" in df.columns and "debit" not in df.columns and "credit" not in df.columns:
            logger.info("Detected single amount column with potential Dr/Cr suffixes")
            df["debit"] = 0.0
            df["credit"] = 0.0
            for idx, val in df["amount_raw"].items():
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                s = str(val).strip()
                is_debit = bool(re.search(r"(Dr\.?|DR\.?)$", s))
                is_credit = bool(re.search(r"(Cr\.?|CR\.?)$", s))
                cleaned = _clean_amount_string(val)
                if cleaned is None:
                    continue
                cleaned = abs(cleaned)
                if is_debit:
                    df.at[idx, "debit"] = cleaned
                elif is_credit:
                    df.at[idx, "credit"] = cleaned
                else:
                    # Default: positive = debit, negative = credit
                    if cleaned >= 0:
                        df.at[idx, "debit"] = cleaned
                    else:
                        df.at[idx, "credit"] = abs(cleaned)
            df.drop(columns=["amount_raw"], inplace=True)

        # Deduplicate column names that may arise from renaming (e.g. both 'Date'
        # and 'Value Date' becoming 'date'). Keep the first occurrence.
        if df.columns.duplicated().any():
            logger.warning("Duplicate column names detected after renaming; keeping first occurrence")
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # --- Step 2: Parse dates ---
        if "date" in df.columns:
            df["date"] = df["date"].apply(_parse_date)
            unparsed = int(df["date"].isna().sum())
            if unparsed > 0:
                logger.warning("Could not parse %d date values", unparsed)
            df.dropna(subset=["date"], inplace=True)
            df["date"] = pd.to_datetime(df["date"])
        else:
            logger.warning("No date column found in the DataFrame")

        # --- Step 3: Clean amounts ---
        for col in ["debit", "credit"]:
            if col in df.columns:
                df[col] = df[col].apply(_clean_amount_string)
                df[col] = df[col].fillna(0.0).astype(float)
            else:
                df[col] = 0.0

        if "balance" in df.columns:
            df["balance"] = df["balance"].apply(_clean_amount_string)
            df["balance"] = df["balance"].astype(float)

        # --- Step 4: Create unified 'amount' and 'transaction_type' columns ---
        # Convention: debit (money going out) is positive, credit (income) is negative
        df["amount"] = df.apply(
            lambda row: row["debit"] if row["debit"] > 0 else -row["credit"],
            axis=1,
        )
        df["transaction_type"] = df.apply(
            lambda row: "debit" if row["debit"] > 0 else "credit",
            axis=1,
        )

        # --- Step 5: Normalize merchant / description names ---
        if "description" in df.columns:
            df["description"] = df["description"].apply(_normalize_merchant_name)

        # --- Step 6: Sort by date ascending ---
        if "date" in df.columns:
            df.sort_values("date", ascending=True, inplace=True)

        # --- Step 7: Drop duplicates ---
        dedup_cols = []
        for c in ["date", "description", "debit", "credit"]:
            if c in df.columns:
                dedup_cols.append(c)
        if dedup_cols:
            before = len(df)
            df.drop_duplicates(subset=dedup_cols, keep="first", inplace=True)
            dropped = before - len(df)
            if dropped > 0:
                logger.info("Dropped %d duplicate rows", dropped)

        # --- Step 8: Add month_year column ---
        if "date" in df.columns:
            df["month_year"] = df["date"].dt.strftime("%b %Y")

        # Reset index
        df.reset_index(drop=True, inplace=True)

        logger.info("Cleaned DataFrame shape: %s", df.shape)
        self._cleaned_df = df
        return df

    def get_summary_stats(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compute summary statistics from a cleaned bank statement DataFrame.

        Args:
            df: Cleaned DataFrame. If None, uses the last cleaned DataFrame.

        Returns:
            Dictionary containing:
                - total_debits: Sum of all debit transactions
                - total_credits: Sum of all credit transactions
                - net_cashflow: total_credits - total_debits
                - transaction_count: Total number of transactions
                - date_range: Tuple of (earliest_date, latest_date)
                - avg_monthly_spend: Average monthly debit spending

        Raises:
            ValueError: If no DataFrame is available.
        """
        if df is None:
            df = self._cleaned_df
        if df is None:
            raise ValueError("No cleaned DataFrame available. Call clean() first or pass a DataFrame.")

        total_debits = float(df.loc[df["transaction_type"] == "debit", "debit"].sum())
        total_credits = float(df.loc[df["transaction_type"] == "credit", "credit"].sum())
        net_cashflow = total_credits - total_debits
        transaction_count = len(df)

        date_range: Optional[tuple] = None
        if "date" in df.columns and not df["date"].isna().all():
            date_range = (df["date"].min(), df["date"].max())

        # Average monthly spend (debits only)
        avg_monthly_spend = 0.0
        if "date" in df.columns and transaction_count > 0:
            debit_df = df[df["transaction_type"] == "debit"]
            if not debit_df.empty:
                monthly_spend = debit_df.groupby(debit_df["date"].dt.to_period("M"))["debit"].sum()
                avg_monthly_spend = float(monthly_spend.mean()) if len(monthly_spend) > 0 else 0.0

        stats: Dict[str, Any] = {
            "total_debits": total_debits,
            "total_credits": total_credits,
            "net_cashflow": net_cashflow,
            "transaction_count": transaction_count,
            "date_range": date_range,
            "avg_monthly_spend": avg_monthly_spend,
        }

        logger.info(
            "Summary: %d transactions, debits=%s, credits=%s, net=%s",
            transaction_count,
            format_indian_currency(total_debits),
            format_indian_currency(total_credits),
            format_indian_currency(net_cashflow),
        )

        return stats
