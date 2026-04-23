"""
PDF bank statement parser for ArthaSense — multi-bank edition.

Supported banks / apps (auto-detected):
  UPI Apps   : GPay, PhonePe, Paytm
  Private    : HDFC Bank, HDFC Credit Card, ICICI Bank, ICICI CC,
               Axis Bank, Axis CC, Kotak Mahindra Bank, IndusInd
  Public     : SBI, PNB, Bank of Baroda, Canara Bank
  Credit Card: SBI Card, AmEx

Architecture
------------
  detect_bank(pdf_text)         → bank name string
  extract_text(pdf_file)        → full PDF text
  parse_transactions(text, bank, rows) → normalized records
  parse_pdf_statement(pdf_file) → 5-column normalized DataFrame
      columns: date, description, amount, transaction_type, bank_source

  parse_pdf(pdf_file)           → 8-column pipeline DataFrame (backward compat)
      columns: date, description, debit, credit, balance, amount,
               transaction_type, month_year

Amount convention in parse_pdf_statement():
  amount   always positive
  transaction_type  "debit" | "credit"
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ScannedPDFError(ValueError):
    """Raised when the PDF has no extractable text (image / scanned)."""


class PDFParseError(ValueError):
    """Raised when no transactions can be extracted from the PDF."""


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_DATE_FMTS = [
    "%d %b %Y %H:%M:%S",
    "%d %b %Y %I:%M:%S %p",
    "%d %b %Y %H:%M",
    "%d %b %Y %I:%M %p",
    "%d %b %Y",
    "%b %d, %Y",           # Jan 15, 2024  (GPay)
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d %B %Y",
    "%d-%B-%Y",
    "%d %b '%y",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%d.%m.%y",
]

_DAY_PREFIX = re.compile(r"^[A-Za-z]+,\s*")
_MULTI_WS   = re.compile(r"\s{2,}")


def _parse_date(raw: Any) -> Optional[pd.Timestamp]:
    if raw is None:
        return None
    if isinstance(raw, pd.Timestamp):
        return raw
    s = _DAY_PREFIX.sub("", str(raw).strip())
    s = _MULTI_WS.sub(" ", s).strip()
    if not s:
        return None
    for fmt in _DATE_FMTS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

_AMT_SIGNED = re.compile(
    r"^(?P<sign>[+\-])\s*(?:Rs\.?|₹|INR)\s*(?P<num>[\d,]+(?:\.\d+)?)$", re.I
)
_AMT_PLAIN = re.compile(
    r"^(?:Rs\.?|₹|INR)\s*(?P<num>[\d,]+(?:\.\d+)?)$", re.I
)
_AMT_CRDR = re.compile(
    r"^(?:Rs\.?|₹|INR)?\s*(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<suf>CR|DR|C|D)\.?$", re.I
)
_AMT_PARENS = re.compile(
    r"^\((?:Rs\.?|₹|INR)?\s*(?P<num>[\d,]+(?:\.\d+)?)\)$", re.I
)
_AMT_NUM = re.compile(r"^(?P<num>[\d,]+(?:\.\d+)?)$")
_EMPTY_VALS = frozenset({"", "-", "--", "nil", "n/a", "na", "nan", "null", "—", "–"})


def _parse_amount_signed(raw: Any) -> Optional[float]:
    """Parse amount → signed float.  positive=credit, negative=debit.  None if not an amount."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s.lower() in _EMPTY_VALS:
        return None

    m = _AMT_SIGNED.match(s)
    if m:
        num = float(m.group("num").replace(",", ""))
        return num if m.group("sign") == "+" else -num

    m = _AMT_CRDR.match(s)
    if m:
        num = float(m.group("num").replace(",", ""))
        return num if m.group("suf").upper() in ("CR", "C") else -num

    m = _AMT_PARENS.match(s)
    if m:
        return -float(m.group("num").replace(",", ""))

    m = _AMT_PLAIN.match(s)
    if m:
        return -float(m.group("num").replace(",", ""))  # unsigned → debit

    # Plain number — try to parse, treat as debit
    s_clean = re.sub(r"^(?:Rs\.?|₹|INR)\s*", "", s, flags=re.I).replace(",", "").strip()
    try:
        return -abs(float(s_clean))
    except ValueError:
        return None


def _parse_amount_plain(raw: Any) -> float:
    """Parse non-negative amount string (separate debit/credit columns)."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return abs(float(raw))
    s = str(raw).strip()
    if s.lower() in _EMPTY_VALS:
        return 0.0
    s = re.sub(r"^(?:Rs\.?|₹|INR)\s*", "", s, flags=re.I)
    s = re.sub(r"\s*(CR|DR|C|D)\.?$", "", s, flags=re.I)
    s = s.replace(",", "").strip()
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Column-name normalisation helpers
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


_DATE_COLS    = {"date", "txn date", "transaction date", "value date", "date & time",
                 "posting date", "tran date", "trans date", "txndate"}
_DESC_COLS    = {"narration", "description", "particulars", "transaction details",
                 "details", "remarks", "remark", "transaction remarks",
                 "transaction description", "particulars of transaction"}
_DEBIT_COLS   = {"debit", "withdrawal", "withdrawal amt", "dr", "debit amount",
                 "debit amt", "withdrawal amount", "debit (inr)", "tran amount dr",
                 "money out"}
_CREDIT_COLS  = {"credit", "deposit", "deposit amt", "cr", "credit amount",
                 "credit amt", "deposit amount", "credit (inr)", "tran amount cr",
                 "money in"}
_AMOUNT_COLS  = {"amount", "transaction amount", "net amount", "tran amount",
                 "transaction amt"}
_DRCR_COLS    = {"dr/cr", "type", "txn type", "transaction type", "cr/dr"}
_BALANCE_COLS = {"balance", "closing balance", "available balance", "bal",
                 "balance (inr)"}

_PAYTM_SIGNALS = {"date & time", "transaction details", "notes & tags", "your account"}

_HEADER_SIGNALS = (
    _DATE_COLS | _DESC_COLS | _DEBIT_COLS | _CREDIT_COLS |
    _AMOUNT_COLS | _BALANCE_COLS | _DRCR_COLS | {"paytm", "note", "ref"}
)

# Row patterns to skip (non-transaction rows)
_SKIP_RE = re.compile(
    r"^(note[:\s]|this payment|self.?transfer|page\s*\d|total\s|opening\s+balance|"
    r"closing\s+balance|statement\s+period|account\s+no|account\s+number|"
    r"ifsc|branch|customer\s+id|mobile|email|address|dear\s|powered\s+by|"
    r"for\s+any|contact\s+us|helpdesk|toll.?free)",
    re.I,
)


def _find_col(headers: list[str], candidates: set[str]) -> int:
    for i, h in enumerate(headers):
        if h in candidates:
            return i
    for i, h in enumerate(headers):
        for c in candidates:
            if c in h:
                return i
    return -1


def _find_header_row(rows: list[list], max_scan: int = 40) -> Optional[int]:
    best_idx, best_score = None, 1
    for i, row in enumerate(rows[:max_scan]):
        norm = {_norm(c) for c in (row or []) if c}
        score = len(norm & _HEADER_SIGNALS)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


# ---------------------------------------------------------------------------
# Bank detection fingerprints
# ---------------------------------------------------------------------------
# Order matters: more specific entries (CC variants) before generic bank names.

_BANK_FINGERPRINTS: list[tuple[str, list[str]]] = [
    # UPI apps — PhonePe before GPay: PhonePe signals are more unique;
    # generic UPI language ("Google Pay", "Transaction statement") could appear
    # in other contexts, so GPay uses a score-based check (see detect_bank()).
    ("PhonePe",     ["PhonePe", "Phone Pe Transaction", "phonepe.com"]),
    ("Paytm",       ["Paytm Payments Bank", "PAYTM", "paytm.com",
                     "Transaction Details\nNotes & Tags"]),
    # GPay uses score-based detection — handled separately in detect_bank()
    # The entry below is kept as a fallback single-signal anchor only.
    ("GPay",        ["Google Pay", "Google Payment", "g.co/pay", "Google LLC"]),
    # Credit Cards — before their parent banks
    ("HDFC CC",     ["HDFC Bank Credit Card", "HDFC Credit Card Statement",
                     "HDFC Bank Ltd. Credit Card"]),
    ("ICICI CC",    ["ICICI Bank Credit Card", "ICICI Credit Card Statement"]),
    ("Axis CC",     ["Axis Bank Credit Card", "Axis Credit Card Statement"]),
    ("SBI Card",    ["SBI Card", "SBI Credit Card", "sbicard.com",
                     "SBI Cards and Payment Services"]),
    ("AmEx",        ["American Express", "Amex", "americanexpress.com"]),
    # Private Banks
    ("HDFC",        ["HDFC Bank Limited", "HDFC BANK", "hdfcbank.com",
                     "HDFC Bank Ltd"]),
    ("ICICI",       ["ICICI Bank", "iMobile", "icicibank.com",
                     "ICICI Bank Limited"]),
    ("Axis",        ["Axis Bank", "axisbank.com", "Axis Bank Limited"]),
    ("Kotak",       ["Kotak Mahindra Bank", "kotakbank.com",
                     "Kotak Bank"]),
    ("IndusInd",    ["IndusInd Bank", "indusind.com", "IndusInd Bank Limited"]),
    # Public Banks
    ("SBI",         ["State Bank of India", "onlinesbi.com",
                     "State Bank Of India"]),
    ("PNB",         ["Punjab National Bank", "PNB", "pnbindia.in"]),
    ("BOB",         ["Bank of Baroda", "bankofbaroda.in", "Bank Of Baroda"]),
    ("Canara",      ["Canara Bank", "canarabank.in"]),
]


_GPAY_SIGNALS: list[tuple[str, str]] = [
    # (signal_string,  portion_of_text_to_search)
    # Each tuple: signal to find, which slice of pdf_text to check.
    # At least 2 must match to confirm GPay — prevents false positives from
    # other apps that use UPI language but are not GPay.
    ("Google Pay",           "first_500"),
    ("Transaction statement", "first_200"),
    ("UPITransactionID:",    "full"),   # pdfplumber strips spaces
    ("UPI Transaction ID:",  "full"),   # PyPDF2 / whitespace-preserving extractor
    ("PaidbyIndianBank",     "full"),   # space-stripped (pdfplumber)
    ("Paid by Indian Bank",  "full"),   # space-preserving extractor
    ("PaidtoIndianBank",     "full"),   # credit indicator (space-stripped)
    ("Paid to Indian Bank",  "full"),   # credit indicator (space-preserving)
    ("Powered by UPI",       "full"),   # footer line present on every page
]


def _gpay_score(pdf_text: str) -> int:
    """Return how many GPay fingerprint signals match in *pdf_text*."""
    first_200 = pdf_text[:200]
    first_500 = pdf_text[:500]
    score = 0
    for sig, scope in _GPAY_SIGNALS:
        search_in = first_200 if scope == "first_200" else (
            first_500 if scope == "first_500" else pdf_text
        )
        if sig in search_in:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Paytm score-based detection signals
# At least 2 must match to confirm Paytm — prevents false positives.
# ---------------------------------------------------------------------------

_PAYTM_DETECT_SIGNALS: list[tuple[str, str]] = [
    # (signal_string, portion_of_text_to_search)
    ("Paytm Payments Bank",       "first_2000"),
    ("paytm.com",                 "first_2000"),
    ("Passbook Payments History", "full"),
    ("UPI Ref No:",               "full"),
    ("Notes & Tags",              "first_3000"),
    ("PAYTM",                     "first_500"),
]


def _paytm_detect_score(pdf_text: str) -> int:
    """Return how many Paytm detection signals match in *pdf_text*."""
    first_500  = pdf_text[:500]
    first_2000 = pdf_text[:2000]
    first_3000 = pdf_text[:3000]
    score = 0
    for sig, scope in _PAYTM_DETECT_SIGNALS:
        search_in = (first_500  if scope == "first_500"  else
                     first_2000 if scope == "first_2000" else
                     first_3000 if scope == "first_3000" else pdf_text)
        if sig in search_in:
            score += 1
    return score


def detect_bank(pdf_text: str) -> str:
    """
    Detect the bank / app that issued the PDF statement.

    GPay uses score-based detection (≥2 signals must match) to prevent
    false positives from other UPI apps that share generic language.
    PhonePe is checked before GPay because its header strings are unique.
    All other banks use single-signal fingerprinting ordered from most
    specific (CC variants) to least specific (parent bank names).

    Parameters
    ----------
    pdf_text : str
        Full text extracted from the PDF.

    Returns
    -------
    str
        Bank identifier e.g. "HDFC", "GPay", "Paytm", "SBI Card".
        Returns "Unknown" if no fingerprint matched.
    """
    # --- GPay: score-based (2+ signals required) --------------------------
    # Checked first because GPay fingerprints overlap with generic UPI text.
    # PhonePe check below will correctly short-circuit for PhonePe PDFs.
    _gscore = _gpay_score(pdf_text)
    logger.debug("GPay signal score: %d", _gscore)

    sample = pdf_text[:4000]

    for bank, signals in _BANK_FINGERPRINTS:
        if bank == "GPay":
            # Only return GPay if score ≥ 2 AND no PhonePe signal present
            if _gscore >= 2 and "PhonePe" not in pdf_text and "phonepe.com" not in pdf_text:
                logger.info("Bank detected: GPay (score=%d)", _gscore)
                return "GPay"
            continue  # skip the single-signal fallback entry for GPay

        if bank == "Paytm":
            # Score-based detection: ≥2 signals must match to confirm Paytm
            _pscore = _paytm_detect_score(pdf_text)
            logger.debug("Paytm signal score: %d", _pscore)
            if _pscore >= 2:
                logger.info("Bank detected: Paytm (score=%d)", _pscore)
                return "Paytm"
            continue  # skip the single-signal fallback entry for Paytm

        for sig in signals:
            if sig in sample:
                logger.info("Bank detected: %s (signal: %r)", bank, sig)
                return bank

    logger.warning("Bank not detected — returning 'Unknown'")
    return "Unknown"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_file: Any) -> str:
    """
    Extract all text from a PDF using pdfplumber (primary) or PyPDF2 (fallback).

    Parameters
    ----------
    pdf_file : str | Path | file-like
        PDF source.

    Returns
    -------
    str
        Concatenated text from all pages.

    Raises
    ------
    ScannedPDFError
        If extracted text is too short to be a text-based PDF.
    """
    raw_bytes = _to_bytes(pdf_file)

    text = _extract_via_pdfplumber(raw_bytes)
    if text is None:
        text = _extract_via_pypdf2(raw_bytes)

    if text is None:
        raise ScannedPDFError(
            "Could not extract text from PDF. "
            "Try uploading a text-based PDF, not a scanned image."
        )

    return text


def _to_bytes(pdf_file: Any) -> bytes:
    if isinstance(pdf_file, (str, Path)):
        return Path(pdf_file).read_bytes()
    if hasattr(pdf_file, "getvalue"):
        b = pdf_file.getvalue()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        return b
    if hasattr(pdf_file, "read"):
        b = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        return b
    raise TypeError(f"Cannot read PDF from {type(pdf_file)}")


def _extract_via_pdfplumber(raw_bytes: bytes) -> Optional[str]:
    try:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                pages_text.append(t)
        return "\n".join(pages_text)
    except Exception as e:
        logger.debug("pdfplumber extraction failed: %s", e)
        return None


def _extract_via_pypdf2(raw_bytes: bytes) -> Optional[str]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(raw_bytes))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
        return "\n".join(pages_text)
    except Exception as e:
        logger.debug("PyPDF2 extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Table extraction helper
# ---------------------------------------------------------------------------

_TBL_LINE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "min_words_vertical": 1,
}
_TBL_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "join_tolerance": 4,
}


def _extract_all_rows(raw_bytes: bytes) -> tuple[list[list], str, int]:
    """
    Extract all table rows and full text from every PDF page.

    Returns
    -------
    (all_rows, full_text, total_chars)
    """
    import pdfplumber
    all_rows: list[list] = []
    page_texts: list[str] = []
    total_chars = 0

    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            total_chars += len(text)
            page_texts.append(text)

            tables = page.extract_tables(_TBL_LINE) or []
            if not tables:
                tables = page.extract_tables(_TBL_TEXT) or []
            for tbl in tables:
                for row in (tbl or []):
                    all_rows.append(row)

    return all_rows, "\n".join(page_texts), total_chars


# ---------------------------------------------------------------------------
# Row-level parsers
# ---------------------------------------------------------------------------

def _norm_cell(c: Any) -> str:
    if c is None:
        return ""
    return re.sub(r"[ \t]{2,}", " ", str(c).replace("\n", " ").strip())


def _parse_paytm_rows(rows: list[list]) -> list[dict]:
    """Paytm UPI format: Date & Time | Details | Notes | Account | Amount."""
    records: list[dict] = []
    current: Optional[dict] = None

    for row in rows:
        cells = [_norm_cell(c) for c in row]
        if not any(cells):
            continue

        row_text = " ".join(cells)
        row_low  = row_text.lower()

        # Skip repeated header rows
        if ("date & time" in row_low or "transaction details" in row_low) and "amount" in row_low:
            continue

        # Sub-row (Date cell empty) — append UPI metadata to current record
        if not cells[0]:
            if current is not None:
                extra = " ".join(c for c in cells[1:] if c)
                if extra and any(kw in extra.lower() for kw in
                                 ("upi ref", "upi id", "order id", "bank ref",
                                  "transaction id", "ref no")):
                    current["meta"] = current.get("meta", "") + " | " + extra
            continue

        if _SKIP_RE.search(row_text.strip()):
            if current:
                records.append(current)
                current = None
            continue

        amount_cell = cells[-1]
        raw_amt = _parse_amount_signed(amount_cell)
        if raw_amt is None:
            if current:
                records.append(current)
                current = None
            continue

        date = _parse_date(cells[0])
        if date is None:
            continue

        if current:
            records.append(current)

        current = {
            "date":        date,
            "description": cells[1] if len(cells) > 1 else "",
            "notes":       cells[2] if len(cells) > 2 else "",
            "account":     cells[3] if len(cells) > 3 else "",
            "raw_amount":  raw_amt,
            "meta":        "",
        }

    if current:
        records.append(current)
    logger.info("Paytm parser: %d records", len(records))
    return records


def _parse_split_rows(
    rows: list[list],
    col_date: int,
    col_desc: int,
    col_debit: int,
    col_credit: int,
    col_balance: int,
    desc_cleaner: Any = None,
) -> list[dict]:
    """Generic parser for statements with separate Debit / Credit columns."""
    records: list[dict] = []
    for row in rows:
        cells = [_norm_cell(c) for c in row]
        if not any(cells):
            continue
        if _SKIP_RE.search(" ".join(cells)):
            continue
        if col_date >= len(cells):
            continue

        date = _parse_date(cells[col_date])
        if date is None:
            continue

        debit  = _parse_amount_plain(cells[col_debit])  if col_debit  < len(cells) else 0.0
        credit = _parse_amount_plain(cells[col_credit]) if col_credit < len(cells) else 0.0
        if debit == 0.0 and credit == 0.0:
            continue

        bal = _parse_amount_plain(cells[col_balance]) if 0 <= col_balance < len(cells) else 0.0

        desc = cells[col_desc] if col_desc < len(cells) else ""
        if desc_cleaner:
            desc = desc_cleaner(desc)

        # raw_amount convention: positive = credit (money in), negative = debit (money out)
        raw_amt = credit - debit

        records.append({
            "date":       date,
            "description": desc,
            "raw_amount": raw_amt,
            "balance":    bal,
        })
    logger.info("Split-column parser: %d records", len(records))
    return records


def _parse_single_amount_rows(
    rows: list[list],
    col_date: int,
    col_desc: int,
    col_amount: int,
    col_balance: int,
    desc_cleaner: Any = None,
) -> list[dict]:
    """Generic parser for single-amount-column statements (Dr/Cr suffix or signed)."""
    records: list[dict] = []
    for row in rows:
        cells = [_norm_cell(c) for c in row]
        if not any(cells):
            continue
        if _SKIP_RE.search(" ".join(cells)):
            continue
        if col_date >= len(cells):
            continue

        date = _parse_date(cells[col_date])
        if date is None:
            continue

        raw_amt = _parse_amount_signed(cells[col_amount]) if col_amount < len(cells) else None
        if raw_amt is None:
            continue

        bal = _parse_amount_plain(cells[col_balance]) if 0 <= col_balance < len(cells) else 0.0

        desc = cells[col_desc] if col_desc < len(cells) else ""
        if desc_cleaner:
            desc = desc_cleaner(desc)

        records.append({
            "date":        date,
            "description": desc,
            "raw_amount":  raw_amt,
            "balance":     bal,
        })
    logger.info("Single-amount parser: %d records", len(records))
    return records


def _parse_axis_rows(
    rows: list[list],
    col_date: int,
    col_desc: int,
    col_drcr: int,
    col_amount: int,
    col_balance: int,
) -> list[dict]:
    """Axis Bank format: has separate DR/CR indicator column + single amount column."""
    records: list[dict] = []
    for row in rows:
        cells = [_norm_cell(c) for c in row]
        if not any(cells):
            continue
        if _SKIP_RE.search(" ".join(cells)):
            continue
        if col_date >= len(cells):
            continue

        date = _parse_date(cells[col_date])
        if date is None:
            continue

        amt_str = cells[col_amount] if col_amount < len(cells) else ""
        amt = _parse_amount_plain(amt_str)
        if amt == 0.0:
            continue

        drcr_str = cells[col_drcr].upper() if col_drcr < len(cells) else "DR"
        is_credit = "CR" in drcr_str

        raw_amt = amt if is_credit else -amt  # pos=credit, neg=debit
        bal = _parse_amount_plain(cells[col_balance]) if 0 <= col_balance < len(cells) else 0.0

        records.append({
            "date":        date,
            "description": cells[col_desc] if col_desc < len(cells) else "",
            "raw_amount":  raw_amt,
            "balance":     bal,
        })
    logger.info("Axis parser: %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# GPay text-based parser (replaces the old table-based parser)
# ---------------------------------------------------------------------------
#
# GPay PDFs do NOT use extractable tables — pdfplumber merges all text into
# a single column with no reliable table structure.  The actual format
# (verified from a real statement) is line-based with no spaces between words:
#
#   Line 0: "{DD}{Mon},{YYYY} Paid[to|from]{RecipientName} ₹{amount}"
#   Line 1: "{HH:MM}{AM|PM} UPITransactionID:{numeric_id}"
#   Line 2: "Paid[by|to]IndianBank{last4}"
#
# Each page has a header ("Transaction statement\n{phone}, {email}\n…") and a
# footer ("Note:This statement…\nPage{N}of{M}") that must be stripped first.

_GPAY_PAGE_HEADER = re.compile(
    r"Transaction statement\n"
    r"[\d]+,\s*\S+@\S+\n"
    r"(?:Transactionstatementperiod[^\n]+\n[^\n]+\n)?"   # page-1 summary row
    r"Date&time\s+Transactiondetails\s+Amount\n",
)
_GPAY_PAGE_FOOTER = re.compile(
    r"Note:Thisstatement[^\n]+\n[^\n]+\nPage\d+of\d+",
    re.DOTALL,
)

# Main transaction line:
#   "03Oct,2025 PaidtoATULPRATAPSINGH ₹40"
#   "07Oct,2025 ReceivedfromAdityaArunDeshmukh ₹60"
_GPAY_TX = re.compile(
    r"^(\d{2}\w{3},\d{4})\s+"          # date:   "03Oct,2025"
    r"(Paidto|Receivedfrom)"            # action: "Paidto" / "Receivedfrom"
    r"(.+?)\s+₹([\d,]+\.?\d*)$"        # name + amount (₹ always present)
)
_GPAY_DATE = re.compile(r"^(\d{2})(\w{3}),(\d{4})$")

# CamelCase splitter — inserts spaces at word boundaries for mixed-case names
# so "AdityaArunDeshmukh" → "Aditya Arun Deshmukh" (helps Layer-0 person detection).
# Uses capture-group substitution to avoid variable-width lookbehinds (Python re
# does not support them).
_CC_LC_UC  = re.compile(r"([a-z])([A-Z])")       # lowercase → uppercase
_CC_UC_RUN = re.compile(r"([A-Z]+)([A-Z][a-z])") # "LLPKanpur" → "LLP Kanpur"

# ALL-CAPS boundary detector: fixed-width {2} lookbehind is valid in Python re.
# Fires at transitions like "ABCDEFGHijkl" → spaces before the mixed part.
_ALLCAPS_BOUNDARY_RE = re.compile(
    r"(?<=[A-Z]{2})(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])"
)

# Correction map for known mangled GPay names.
# pdfplumber strips spaces from ALL-CAPS merchant names, producing
# concatenated blobs like "RAWATENTERPRISES".  This map is applied
# BEFORE any regex-based splitting.
# Keys are stored UPPERCASE — the lookup normalises via raw.upper()
# so "Familyrestaurant", "FAMILYRESTAURANT", and "familyrestaurant" all match.
_GPAY_NAME_MAP: dict[str, str] = {
    "RAWATENTERPRISES":               "Rawat Enterprises",
    "RAWAT ENTERPRISES":              "Rawat Enterprises",
    "INDIANINSTITUTEOFTECHNOLOGYKANPURHALLACCOUNT":
                                      "IIT Kanpur Hall Account",
    "INDIAN INSTITUTE OF TECHNOLOGY KANPUR HALL ACCOUNT":
                                      "IIT Kanpur Hall Account",
    "ROOPSAGARFOODWORKSANDTELLPKANPAC": "Roop Sagar Foodworks",
    "SHAKILAHAMAD":                   "Shakil Ahamad",
    "MYJIO":                          "MyJio",
    "MYJIOMYJIO":                     "MyJio",
    "FAMILYRESTAURANT":               "Family Restaurant",
    "KISHORECYCLESTORE":              "Kishore Cycle Store",
    # Common person names that appear concatenated (13+ occurrences in real data)
    "ATULPRATAPSINGH":                "Atul Pratap Singh",
}


def _split_camel(name: str) -> str:
    """Insert spaces at CamelCase boundaries in a recipient name."""
    if not name:
        return ""
    # All-uppercase: word boundaries are unknown — return as-is
    if name.isupper():
        return name
    s = _CC_LC_UC.sub(r"\1 \2", name)
    s = _CC_UC_RUN.sub(r"\1 \2", s)
    return s.strip()


def _fix_gpay_name(raw: str, is_credit: bool) -> str:
    """
    Clean a GPay recipient / sender name.

    Priority
    --------
    1. Credits (Received from …) are already properly CamelCased —
       just run the standard splitter; never apply ALL-CAPS fixes.
    2. Exact lookup in _GPAY_NAME_MAP — handles well-known mangled names.
    3. Mixed-case strings — CamelCase splitter.
    4. ALL-CAPS with embedded spaces — title-case only.
    5. ALL-CAPS without spaces, len > 15 — regex boundary insertion + title-case.
    6. ALL-CAPS short (len ≤ 15) — title-case only.
    """
    if not raw:
        return raw

    # Priority 1 — credit names keep their CamelCase
    if is_credit:
        return _split_camel(raw)

    # Priority 2 — correction map (case-insensitive: keys stored as UPPERCASE)
    if raw.upper() in _GPAY_NAME_MAP:
        return _GPAY_NAME_MAP[raw.upper()]

    # Determine if purely ALL-CAPS (spaces allowed, but only alpha + space)
    alpha_only = raw.replace(" ", "")
    is_allcaps = alpha_only.isalpha() and alpha_only.isupper()

    if not is_allcaps:
        # Priority 3 — mixed case: standard CamelCase splitter
        return _split_camel(raw)

    # Priority 4 — ALL-CAPS with existing spaces: title-case each word
    if " " in raw:
        return raw.title()

    # Priority 5 — ALL-CAPS blob, len > 15: try regex boundary + title-case
    if len(raw) > 15:
        split = _ALLCAPS_BOUNDARY_RE.sub(" ", raw)
        return split.title()

    # Priority 6 — short ALL-CAPS blob: just title-case
    return raw.title()


def _parse_gpay_date(raw: str) -> Optional[pd.Timestamp]:
    m = _GPAY_DATE.match(raw)
    if m:
        try:
            return pd.Timestamp(
                datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y")
            )
        except ValueError:
            pass
    return None


def parse_gpay_text(raw_text: str) -> pd.DataFrame:
    """
    Parse a GPay transaction statement PDF from its extracted text.

    GPay does not produce tables — this parser works entirely on the
    line-structured text returned by pdfplumber's extract_text().

    Name cleaning (priority order):
      1. Correction map (_GPAY_NAME_MAP) for known mangled ALL-CAPS blobs.
      2. Mixed-case names: CamelCase splitter.
      3. ALL-CAPS with spaces: title-case.
      4. ALL-CAPS blob len > 15: regex boundary insertion + title-case.
      5. Short ALL-CAPS blob: title-case.
    Credit names (Received from …) are handled by the standard CamelCase
    splitter only — their casing is already correct.

    Verified against a real 23-page GPay statement:
      225 transactions | Sent ₹64,498.65 | Received ₹1,738.00

    Parameters
    ----------
    raw_text : str
        Full text extracted from all PDF pages (pages joined by newlines).

    Returns
    -------
    pd.DataFrame  with columns:
        date, description, amount (always positive), transaction_type,
        bank_source
    """
    # ── Step 1: Strip per-page headers and footers ─────────────────────────
    text = _GPAY_PAGE_HEADER.sub("", raw_text)
    text = _GPAY_PAGE_FOOTER.sub("", text)

    # ── Step 2: Parse line by line ─────────────────────────────────────────
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    records: list[dict] = []
    failed: list[str]   = []
    i = 0
    while i < len(lines):
        m = _GPAY_TX.match(lines[i])
        if m:
            date      = _parse_gpay_date(m.group(1))
            action    = m.group(2)              # "Paidto" | "Receivedfrom"
            raw_name  = m.group(3).strip()
            is_credit = (action == "Receivedfrom")

            # Use _fix_gpay_name: applies correction map + ALL-CAPS handling
            name   = _fix_gpay_name(raw_name, is_credit)
            amount = float(m.group(4).replace(",", ""))

            records.append({
                "date":             date,
                "description":      name,
                "amount":           amount,
                "transaction_type": "credit" if is_credit else "debit",
                "bank_source":      "GPay",
            })
            i += 3   # skip the time+UPI-ID line and the bank-direction line
        else:
            # Date-like line that didn't match full tx pattern → flag for report
            if re.match(r"^\d{2}\w{3},\d{4}", lines[i]):
                failed.append(lines[i])
            i += 1

    logger.info(
        "GPay text parser: %d transactions (%d failed blocks)",
        len(records), len(failed),
    )

    if failed:
        logger.warning("GPay: %d lines matched date but not full tx pattern:", len(failed))
        for fl in failed[:5]:
            logger.warning("  %r", fl)

    # ── Step 3: Build DataFrame ────────────────────────────────────────────
    if not records:
        return pd.DataFrame(columns=["date", "description", "amount",
                                     "transaction_type", "bank_source"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


# ---------------------------------------------------------------------------
# Paytm text-based parser
# ---------------------------------------------------------------------------
#
# Paytm UPI Statement PDFs do NOT have reliable extractable tables.
# Actual format (verified from real 74-page statement):
#
#   Line 0:  "{day} {Mon} {Paid to|Received from|...} {Name} [Tag: {bank}] [+/-] Rs.{amount}"
#   Line 1:  "{HH:MM AM|PM}"
#   Line 2:  (optional name continuation OR "UPI ID: ..." or "A/c No: ...")
#   Line N:  "UPI Ref No: {number}"
#
# Transactions verified:
#   486 total | Paid Rs.206,878.02 (426) | Received Rs.166,733 (60)

_PAYTM_TX_START = re.compile(
    r'^(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
    r'(.+?)\s+([+-])\s*Rs\.([\d,]+(?:\.\d+)?)$',
    re.IGNORECASE,
)
_PAYTM_PERIOD_HDR = re.compile(
    r'(\d{1,2})\s+(\w{3,})\'(\d{2})\s*-\s*(\d{1,2})\s+(\w{3,})\'(\d{2})',
    re.IGNORECASE,
)
_PAYTM_TIME_LINE = re.compile(r'^\d{1,2}:\d{2}\s*(AM|PM)$', re.IGNORECASE)
_PAYTM_UPI_LINE  = re.compile(
    r'^(UPI\s*(ID|Ref\s*No|Transaction\s*ID|Ref)[:\s]|A/c\s*No[:\s]|'
    r'Page\s*\d|For\s+any|Contact\s*Us)',
    re.IGNORECASE,
)
_PAYTM_ACTION_PREFIX = re.compile(
    r'^(Paid\s+to\s+|Money\s+sent\s+to\s+|Received\s+from\s+|'
    r'Transferred\s+to\s+|Money\s+transferred\s+to\s+|'
    r'Add\s+Money\s+to\s+|Merchant\s+Payment\s+to\s+)',
    re.IGNORECASE,
)
_PAYTM_TAG_STRIP  = re.compile(r'\s+Tag:\s+.+$',  re.IGNORECASE)
_PAYTM_NOTE_STRIP = re.compile(r'\s+Note:\s+.+$', re.IGNORECASE)
_PAYTM_HASHTAG    = re.compile(r'#\s*(\S+)')
_PAYTM_EXCLUDE_RE = re.compile(
    # Only skip true internal account-to-account transfers (double-counted in net).
    # NOTE: UPI Lite, Gold Coins, Loyalty Points, Vouchers are all real spend events.
    r'(Self.?Transfer|Wallet\s*to\s*Bank\s*Transfer)',
    re.IGNORECASE,
)

_MONTH_NUMS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _paytm_infer_years(text: str) -> dict:
    """
    Extract the year map from the Paytm statement period header.

    Expects a header like "24 MAR'25 - 23 MAR'26".
    Returns a dict with start/end year, month, and day so that
    _paytm_year_for() can assign the correct year to each transaction.
    """
    m = _PAYTM_PERIOD_HDR.search(text[:3000])
    if not m:
        # Fallback: extract any 4-digit year from the first 500 chars
        yr_m = re.search(r"20(\d{2})", text[:500])
        yr = 2000 + int(yr_m.group(1)) if yr_m else 2025
        return {
            "start_year": yr, "end_year": yr + 1,
            "start_month": 1, "start_day": 1,
            "end_month": 12, "end_day": 31,
        }

    start_day = int(m.group(1))
    start_mon = m.group(2).lower()[:3]
    start_yr  = 2000 + int(m.group(3))
    end_day   = int(m.group(4))
    end_mon   = m.group(5).lower()[:3]
    end_yr    = 2000 + int(m.group(6))

    return {
        "start_year":  start_yr,
        "end_year":    end_yr,
        "start_month": _MONTH_NUMS.get(start_mon, 3),
        "start_day":   start_day,
        "end_month":   _MONTH_NUMS.get(end_mon, 3),
        "end_day":     end_day,
    }


def _paytm_year_for(day: int, month_num: int, year_map: dict) -> int:
    """
    Infer the year for a transaction given its day and month number.

    For statements spanning two calendar years (e.g. MAR'25 – MAR'26):
      - month > start_month  → start_year (e.g. Apr-Dec → 2025)
      - month < start_month  → end_year   (e.g. Jan-Feb → 2026)
      - month == start_month → start_year if day >= start_day else end_year
    """
    start_yr  = year_map.get("start_year",  2025)
    end_yr    = year_map.get("end_year",    start_yr)
    start_mon = year_map.get("start_month", 1)
    start_day = year_map.get("start_day",   1)

    if start_yr == end_yr:
        return start_yr
    if month_num > start_mon:
        return start_yr
    if month_num < start_mon:
        return end_yr
    # month_num == start_mon
    return start_yr if day >= start_day else end_yr


def parse_paytm_text(raw_text: str) -> pd.DataFrame:
    """
    Parse a Paytm UPI Statement PDF from its extracted text.

    Paytm does not produce reliable tables — this parser works entirely on the
    line-structured text returned by pdfplumber's extract_text().

    Verified against a real 74-page Paytm statement:
      486 transactions | Paid Rs.206,878.02 (426) | Received Rs.166,733 (60)

    Parameters
    ----------
    raw_text : str
        Full text extracted from all PDF pages (pages joined by newlines).

    Returns
    -------
    pd.DataFrame  with columns:
        date, description, amount (always positive), transaction_type,
        bank_source, paytm_tag
    """
    year_map = _paytm_infer_years(raw_text)
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]

    records: list[dict] = []
    i = 0
    while i < len(lines):
        m = _PAYTM_TX_START.match(lines[i])
        if not m:
            i += 1
            continue

        day_s, mon_s, middle, sign, amt_s = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        )

        # Skip known noise transactions
        if _PAYTM_EXCLUDE_RE.search(middle):
            i += 1
            continue

        day       = int(day_s)
        month_num = _MONTH_NUMS.get(mon_s.lower()[:3], 1)
        year      = _paytm_year_for(day, month_num, year_map)

        try:
            date = pd.Timestamp(year=year, month=month_num, day=day)
        except Exception:
            i += 1
            continue

        amount  = float(amt_s.replace(",", ""))
        tx_type = "credit" if sign == "+" else "debit"

        # ── Clean description from the middle group ────────────────────────
        desc = middle.strip()
        desc = _PAYTM_TAG_STRIP.sub("",  desc)   # remove "Tag: Bank Name"
        desc = _PAYTM_NOTE_STRIP.sub("", desc)   # remove "Note: ..."
        desc = _PAYTM_ACTION_PREFIX.sub("", desc) # remove "Paid to" / "Received from" etc.
        desc = desc.strip()

        # ── Extract Paytm category tag (#HashtagName) from transaction line ─
        paytm_tag = ""
        tag_m = _PAYTM_HASHTAG.search(middle)
        if tag_m:
            paytm_tag = tag_m.group(1)

        # ── Look ahead for multi-line name continuation ────────────────────
        # Structure: line i (tx), line i+1 (time HH:MM AM/PM),
        #            line i+2 (optional: name continuation / UPI ID / A/c No)
        j = i + 1
        if j < len(lines) and _PAYTM_TIME_LINE.match(lines[j]):
            j += 1  # step past the time line
            if j < len(lines) and not _PAYTM_UPI_LINE.match(lines[j]):
                cont = lines[j]
                # Ignore if it's another transaction start
                if not _PAYTM_TX_START.match(cont):
                    # Extract category tag from continuation if not yet found
                    if not paytm_tag:
                        ctag_m = _PAYTM_HASHTAG.search(cont)
                        if ctag_m:
                            paytm_tag = ctag_m.group(1)
                    # Append name portion (everything before the first #)
                    cont_name = re.sub(r'\s*#.*$', '', cont).strip()
                    if cont_name and not re.match(r'^\d+$', cont_name):
                        desc = (desc + " " + cont_name).strip()

        records.append({
            "date":             date,
            "description":      desc or "Paytm Transaction",
            "amount":           amount,
            "transaction_type": tx_type,
            "bank_source":      "Paytm",
            "paytm_tag":        paytm_tag,
        })
        i += 1

    debits  = sum(1 for r in records if r["transaction_type"] == "debit")
    credits = sum(1 for r in records if r["transaction_type"] == "credit")
    debit_total  = sum(r["amount"] for r in records if r["transaction_type"] == "debit")
    credit_total = sum(r["amount"] for r in records if r["transaction_type"] == "credit")

    logger.info(
        "Paytm text parser: %d transactions "
        "(%d debits = Rs.%.2f | %d credits = Rs.%.2f)",
        len(records), debits, debit_total, credits, credit_total,
    )

    if not records:
        return pd.DataFrame(columns=["date", "description", "amount",
                                     "transaction_type", "bank_source"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _parse_phonepe_rows(rows: list[list], col_date: int, col_desc: int,
                        col_type: int, col_amount: int) -> list[dict]:
    """PhonePe: Date | Description | Type (Debit/Credit) | Amount (always positive)."""
    records: list[dict] = []
    for row in rows:
        cells = [_norm_cell(c) for c in row]
        if not any(cells):
            continue
        if _SKIP_RE.search(" ".join(cells)):
            continue
        if col_date >= len(cells):
            continue

        date = _parse_date(cells[col_date])
        if date is None:
            continue

        amt = _parse_amount_plain(cells[col_amount]) if col_amount < len(cells) else 0.0
        if amt == 0.0:
            continue

        txn_type_str = cells[col_type].lower() if col_type < len(cells) else "debit"
        is_credit = "credit" in txn_type_str or "in" == txn_type_str
        raw_amt = amt if is_credit else -amt

        records.append({
            "date":        date,
            "description": cells[col_desc] if col_desc < len(cells) else "",
            "raw_amount":  raw_amt,
            "balance":     0.0,
        })
    logger.info("PhonePe parser: %d records", len(records))
    return records


# ---------------------------------------------------------------------------
# Description cleaners (bank-specific)
# ---------------------------------------------------------------------------

_SBI_PREFIX = re.compile(
    r"^(BY TRANSFER[\-\s]|TO TRANSFER[\-\s]|INB\s+|NEFT CR[\-\s]+|"
    r"NEFT DR[\-\s]+|RTGS CR[\-\s]+|RTGS DR[\-\s]+|UPI[\-/])",
    re.I,
)
_REF_SUFFIX = re.compile(r"\s+\d{6,}\s*$")
_EXCESS_WS  = re.compile(r"\s{2,}")


def _sbi_desc_cleaner(desc: str) -> str:
    desc = _SBI_PREFIX.sub("", desc).strip()
    desc = _REF_SUFFIX.sub("", desc).strip()
    desc = _EXCESS_WS.sub(" ", desc)
    return desc


def _hdfc_desc_cleaner(desc: str) -> str:
    desc = re.sub(r"[\-\s]+\d{6,}\s*$", "", desc).strip()
    desc = _EXCESS_WS.sub(" ", desc)
    return desc


def _icici_desc_cleaner(desc: str) -> str:
    desc = re.sub(r"\s+(REF\s*NO|REF)[\.\s]*[\dA-Z]{6,}\s*$", "", desc, flags=re.I)
    desc = _EXCESS_WS.sub(" ", desc).strip()
    return desc


# ---------------------------------------------------------------------------
# Generic / scoring-based fallback parser
# ---------------------------------------------------------------------------

def _score_table(rows: list[list]) -> int:
    """Score a table by how many header signals it contains in its first row."""
    if not rows:
        return 0
    header = {_norm(c) for c in (rows[0] or []) if c}
    return len(header & _HEADER_SIGNALS)


def _parse_generic_rows(all_rows: list[list]) -> list[dict]:
    """
    Fallback parser for PNB, BOB, Canara, AmEx, and unknown banks.

    Scores all candidate header rows, picks the best table, auto-detects
    column indices, and routes to split or single-amount parser.
    """
    if not all_rows:
        return []

    # Try to find header row among all rows
    header_idx = _find_header_row(all_rows)
    if header_idx is None:
        logger.warning("Generic parser: no header row found")
        return []

    header_row = all_rows[header_idx]
    data_rows  = all_rows[header_idx + 1:]
    norm_hdrs  = [_norm(c) for c in header_row]

    logger.info("Generic parser: header=%s", norm_hdrs)

    # Detect format
    has_split  = bool(set(norm_hdrs) & _DEBIT_COLS) and bool(set(norm_hdrs) & _CREDIT_COLS)
    has_amount = bool(set(norm_hdrs) & _AMOUNT_COLS)

    ci_date    = _find_col(norm_hdrs, _DATE_COLS)
    ci_desc    = _find_col(norm_hdrs, _DESC_COLS)
    ci_balance = _find_col(norm_hdrs, _BALANCE_COLS)
    if ci_desc < 0:
        ci_desc = max(1, (ci_date + 1) if ci_date >= 0 else 1)

    if has_split:
        ci_debit  = _find_col(norm_hdrs, _DEBIT_COLS)
        ci_credit = _find_col(norm_hdrs, _CREDIT_COLS)
        if ci_date < 0 or ci_debit < 0 or ci_credit < 0:
            return []
        return _parse_split_rows(data_rows, ci_date, ci_desc, ci_debit, ci_credit, ci_balance)

    elif has_amount:
        ci_amount = _find_col(norm_hdrs, _AMOUNT_COLS)
        if ci_date < 0 or ci_amount < 0:
            return []
        return _parse_single_amount_rows(data_rows, ci_date, ci_desc, ci_amount, ci_balance)

    else:
        # Last resort: try Paytm-style heuristic
        logger.warning("Generic parser: unknown format, attempting Paytm-style heuristic")
        return _parse_paytm_rows(all_rows)


# ---------------------------------------------------------------------------
# Universal text-line parser (last-resort fallback for any unknown format)
# ---------------------------------------------------------------------------

# Matches most date formats found in bank statements worldwide
_UNIV_DATE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\d{4}[\/\-]\d{2}[\/\-]\d{2}"                                          # 2024-01-15
    r"|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}"                                 # 15/01/2024
    r"|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2}(?!\d)"                           # 15/01/24
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2},?\s*\d{4}"
    r")(?!\d)",
    re.I,
)

# Matches monetary amounts with currency symbol, CR/DR suffix, or signed prefix
_UNIV_AMT_RE = re.compile(
    r"(?:"
    r"(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?"          # ₹1,234.56 / Rs.500
    r"|[\d,]+(?:\.\d{1,2})?\s*(?:CR|DR|Cr|Dr)\b"   # 1234.56 CR
    r"|[+\-]\s*(?:Rs\.?|₹|INR)?\s*[\d,]+(?:\.\d+)?" # +1234.56 / -₹500
    r")",
    re.I,
)

# Strips currency/sign/suffix from an amount match to get the raw number
_UNIV_AMT_NUM = re.compile(r"[\d,]+(?:\.\d+)?")
_UNIV_CRDR    = re.compile(r"\b(CR|Cr)\b", re.I)
_UNIV_SIGN    = re.compile(r"^[+\-]")


def _univ_parse_amount(raw: str) -> Optional[float]:
    """Parse a raw amount string from _UNIV_AMT_RE → signed float."""
    s = raw.strip()
    is_credit = bool(_UNIV_CRDR.search(s))
    sign = -1.0
    if _UNIV_SIGN.match(s):
        sign = 1.0 if s[0] == "+" else -1.0
    elif is_credit:
        sign = 1.0
    m = _UNIV_AMT_NUM.search(s)
    if not m:
        return None
    try:
        return sign * float(m.group().replace(",", ""))
    except ValueError:
        return None


def _parse_universal_text(full_text: str) -> list[dict]:
    """
    Last-resort parser: scan every line for date + amount patterns.

    Works with ANY bank statement format — does not require known column headers.
    Returns records with keys: date, description, raw_amount.
    """
    records: list[dict] = []
    lines = full_text.splitlines()

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or _SKIP_RE.match(line_stripped):
            continue

        date_m = _UNIV_DATE_RE.search(line_stripped)
        if not date_m:
            continue

        # Try to find all amount candidates in the same line
        amt_matches = list(_UNIV_AMT_RE.finditer(line_stripped))
        if not amt_matches:
            # Check continuation line
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                amt_matches = list(_UNIV_AMT_RE.finditer(nxt))
            if not amt_matches:
                continue

        # Use the LAST amount match (usually the transaction amount, not balance)
        amt_raw = amt_matches[-1].group()
        amount  = _univ_parse_amount(amt_raw)
        if amount is None:
            continue

        # Description: part of line between date and amount (or after date)
        date_end  = date_m.end()
        amt_start = amt_matches[-1].start()
        if amt_start > date_end:
            desc = line_stripped[date_end:amt_start].strip(" ,|-")
        else:
            desc = line_stripped[date_m.end():].strip(" ,|-")
        # Strip the amount itself from description if it leaked in
        desc = _UNIV_AMT_RE.sub("", desc).strip(" ,|-")
        if not desc:
            desc = line_stripped[: date_m.start()].strip(" ,|-")
        if not desc:
            desc = "Transaction"

        ts = _parse_date(date_m.group())
        if ts is None:
            continue

        records.append({
            "date":        ts,
            "description": desc,
            "raw_amount":  amount,
            "balance":     0.0,
        })

    logger.info("Universal text parser: %d records found", len(records))
    return records


# ---------------------------------------------------------------------------
# Positional column detection (no keyword headers needed)
# ---------------------------------------------------------------------------

def _parse_positional_rows(all_rows: list[list]) -> list[dict]:
    """
    Detect date/amount columns by position, handling split debit/credit formats.

    Strategy
    --------
    1. Score every column for date-likeness and amount-likeness.
    2. Exclude the last dense numeric column (running balance is always rightmost).
    3. If two mutually-exclusive amount columns remain → split debit/credit.
    4. Otherwise → single signed-amount column.
    """
    if not all_rows:
        return []

    # Find the first row that contains a parseable date → data start
    data_start = 0
    for i, row in enumerate(all_rows[:40]):
        for cell in (row or []):
            if cell and _parse_date(str(cell)):
                data_start = i
                break
        if data_start == i:
            break

    data_rows = [r for r in all_rows[data_start:]
                 if r and sum(c is not None for c in r) >= 2]
    if not data_rows:
        return []

    sample = data_rows[:30]
    ncols  = max(len(r) for r in sample)

    date_col_scores   = [0] * ncols
    amount_col_scores = [0] * ncols

    for row in sample:
        for j, cell in enumerate(row or []):
            if cell is None:
                continue
            s = str(cell).strip()
            if _parse_date(s):
                date_col_scores[j] += 1
            if _parse_amount_signed(s) is not None:
                amount_col_scores[j] += 1

    ci_date = int(max(range(ncols), key=lambda j: date_col_scores[j]))
    if date_col_scores[ci_date] == 0:
        return []

    # Candidate amount columns (not the date col, score > 0)
    amt_candidates = sorted(
        [j for j in range(ncols) if j != ci_date and amount_col_scores[j] > 0],
        key=lambda j: amount_col_scores[j],
        reverse=True,
    )
    if not amt_candidates:
        return []

    # Exclude the last column if it is dense (≥60% filled) — that is the balance
    last_col = ncols - 1
    ci_balance = -1
    if last_col in amt_candidates and amount_col_scores[last_col] >= len(sample) * 0.6:
        ci_balance = last_col
        amt_candidates = [j for j in amt_candidates if j != last_col]
    if not amt_candidates:
        return []

    # Detect split debit/credit: two columns that are mutually exclusive
    has_split  = False
    ci_debit   = ci_credit = -1
    ci_amount  = amt_candidates[0]

    if len(amt_candidates) >= 2:
        c1, c2    = amt_candidates[0], amt_candidates[1]
        exclusive = 0
        for row in sample:
            v1 = row[c1] if c1 < len(row) else None
            v2 = row[c2] if c2 < len(row) else None
            p1 = _parse_amount_signed(str(v1)) is not None if v1 else False
            p2 = _parse_amount_signed(str(v2)) is not None if v2 else False
            if bool(p1) != bool(p2):   # exactly one filled → mutually exclusive
                exclusive += 1
        if exclusive >= max(2, len(sample) * 0.3):
            has_split  = True
            ci_debit   = min(c1, c2)   # earlier column = debit (convention)
            ci_credit  = max(c1, c2)

    # Description: column with longest average text, excluding amount/date/balance cols
    skip_cols = {ci_date, ci_balance}
    skip_cols |= ({ci_debit, ci_credit} if has_split else {ci_amount})
    text_lens = [0.0] * ncols
    for row in sample:
        for j, cell in enumerate(row or []):
            if cell and j not in skip_cols:
                text_lens[j] += len(str(cell))
    ci_desc = int(max(
        [j for j in range(ncols) if j not in skip_cols],
        key=lambda j: text_lens[j],
        default=-1,
    ))

    records: list[dict] = []
    for row in data_rows:
        if not row:
            continue
        raw_date = row[ci_date] if ci_date < len(row) else None
        ts = _parse_date(raw_date)
        if ts is None:
            continue

        raw_desc = str(row[ci_desc] or "").strip() if ci_desc >= 0 and ci_desc < len(row) else ""

        if has_split:
            dr_raw = row[ci_debit]  if ci_debit  < len(row) else None
            cr_raw = row[ci_credit] if ci_credit < len(row) else None
            dr = _parse_amount_plain(dr_raw)
            cr = _parse_amount_plain(cr_raw)
            if dr == 0.0 and cr == 0.0:
                continue
            raw_amount = cr - dr          # positive = credit, negative = debit
        else:
            raw_val    = row[ci_amount] if ci_amount < len(row) else None
            raw_amount = _parse_amount_signed(raw_val)
            if raw_amount is None:
                continue

        records.append({
            "date":        ts,
            "description": raw_desc,
            "raw_amount":  raw_amount,
            "balance":     0.0,
        })

    logger.info(
        "Positional parser: %d records (split=%s, ci_date=%d, ci_debit=%s, "
        "ci_credit=%s, ci_amount=%s, ci_balance=%s)",
        len(records), has_split, ci_date,
        ci_debit if has_split else "-",
        ci_credit if has_split else "-",
        ci_amount if not has_split else "-",
        ci_balance,
    )
    return records


# ---------------------------------------------------------------------------
# Route to bank-specific parser
# ---------------------------------------------------------------------------

def parse_transactions(
    bank: str,
    all_rows: list[list],
    full_text: str,
) -> list[dict]:
    """
    Route to the appropriate bank-specific parser and return raw records.

    Parameters
    ----------
    bank      : detected bank name (from detect_bank())
    all_rows  : all table rows extracted from the PDF
    full_text : full extracted text (for text-based parsers like Paytm)

    Returns
    -------
    list[dict]  — raw records with keys: date, description, raw_amount, balance
    """
    if not all_rows:
        return []

    header_idx = _find_header_row(all_rows)
    if header_idx is None:
        if bank == "Paytm":
            return _parse_paytm_rows(all_rows)
        return _parse_generic_rows(all_rows)

    header_row = all_rows[header_idx]
    data_rows  = all_rows[header_idx + 1:]
    norm_hdrs  = [_norm(c) for c in header_row]
    norm_set   = set(norm_hdrs)

    # ── Paytm ──────────────────────────────────────────────────────────────
    if bank == "Paytm" or bool(_PAYTM_SIGNALS & norm_set):
        return _parse_paytm_rows(data_rows)

    # ── GPay ───────────────────────────────────────────────────────────────
    # GPay PDFs have no table structure; they are handled by parse_gpay_text()
    # which is called directly from parse_pdf_statement() before reaching here.
    # If we somehow land here with bank="GPay", fall back to generic.
    if bank == "GPay":
        logger.warning("GPay reached parse_transactions(); using generic fallback")
        return _parse_generic_rows(all_rows)

    # ── PhonePe ────────────────────────────────────────────────────────────
    if bank == "PhonePe":
        ci_date   = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc   = _find_col(norm_hdrs, _DESC_COLS)
        ci_type   = _find_col(norm_hdrs, _DRCR_COLS)
        ci_amount = _find_col(norm_hdrs, _AMOUNT_COLS)
        if ci_date < 0 or ci_amount < 0:
            return _parse_generic_rows(all_rows)
        if ci_desc < 0:
            ci_desc = 1
        if ci_type < 0:
            ci_type = 2
        return _parse_phonepe_rows(data_rows, ci_date, ci_desc, ci_type, ci_amount)

    # ── Axis Bank (has DR/CR column) ────────────────────────────────────────
    if bank in ("Axis", "Axis CC", "IndusInd"):
        ci_date   = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc   = _find_col(norm_hdrs, _DESC_COLS)
        ci_drcr   = _find_col(norm_hdrs, _DRCR_COLS)
        ci_amount = _find_col(norm_hdrs, _AMOUNT_COLS)
        ci_bal    = _find_col(norm_hdrs, _BALANCE_COLS)
        if ci_date >= 0 and ci_drcr >= 0 and ci_amount >= 0:
            if ci_desc < 0:
                ci_desc = 1
            return _parse_axis_rows(data_rows, ci_date, ci_desc, ci_drcr, ci_amount, ci_bal)
        # Fall through to split/single detection

    # ── HDFC Bank (split columns) ────────────────────────────────────────
    if bank == "HDFC":
        ci_date    = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc    = _find_col(norm_hdrs, {"narration", "description", "particulars"})
        ci_debit   = _find_col(norm_hdrs, _DEBIT_COLS)
        ci_credit  = _find_col(norm_hdrs, _CREDIT_COLS)
        ci_balance = _find_col(norm_hdrs, _BALANCE_COLS)
        if ci_date >= 0 and ci_debit >= 0 and ci_credit >= 0:
            if ci_desc < 0:
                ci_desc = 1
            return _parse_split_rows(data_rows, ci_date, ci_desc, ci_debit, ci_credit,
                                     ci_balance, _hdfc_desc_cleaner)

    # ── HDFC CC / SBI Card / ICICI CC (single amount) ───────────────────
    if bank in ("HDFC CC", "SBI Card", "ICICI CC", "AmEx"):
        ci_date   = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc   = _find_col(norm_hdrs, _DESC_COLS)
        ci_amount = _find_col(norm_hdrs, _AMOUNT_COLS)
        ci_bal    = _find_col(norm_hdrs, _BALANCE_COLS)
        if ci_date >= 0 and ci_amount >= 0:
            if ci_desc < 0:
                ci_desc = 1
            return _parse_single_amount_rows(data_rows, ci_date, ci_desc, ci_amount, ci_bal)

    # ── ICICI Bank / SBI / Kotak / PNB / BOB / Canara (split columns) ────
    if bank in ("ICICI", "SBI", "Kotak", "PNB", "BOB", "Canara"):
        ci_date    = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc    = _find_col(norm_hdrs, _DESC_COLS)
        ci_debit   = _find_col(norm_hdrs, _DEBIT_COLS)
        ci_credit  = _find_col(norm_hdrs, _CREDIT_COLS)
        ci_balance = _find_col(norm_hdrs, _BALANCE_COLS)
        cleaner = _sbi_desc_cleaner if bank == "SBI" else (
            _icici_desc_cleaner if bank == "ICICI" else None
        )
        if ci_date >= 0 and ci_debit >= 0 and ci_credit >= 0:
            if ci_desc < 0:
                ci_desc = 1
            return _parse_split_rows(data_rows, ci_date, ci_desc, ci_debit, ci_credit,
                                     ci_balance, cleaner)

    # ── SBI Card (can also be split) ─────────────────────────────────────
    if bank == "SBI Card":
        ci_date    = _find_col(norm_hdrs, _DATE_COLS)
        ci_desc    = _find_col(norm_hdrs, _DESC_COLS)
        ci_debit   = _find_col(norm_hdrs, _DEBIT_COLS)
        ci_credit  = _find_col(norm_hdrs, _CREDIT_COLS)
        ci_amount  = _find_col(norm_hdrs, _AMOUNT_COLS)
        ci_balance = _find_col(norm_hdrs, _BALANCE_COLS)
        if ci_desc < 0:
            ci_desc = 1
        if ci_debit >= 0 and ci_credit >= 0:
            return _parse_split_rows(data_rows, ci_date, ci_desc, ci_debit, ci_credit, ci_balance)
        if ci_amount >= 0:
            return _parse_single_amount_rows(data_rows, ci_date, ci_desc, ci_amount, ci_balance)

    # ── Generic fallback ─────────────────────────────────────────────────
    logger.warning("No bank-specific parser for %r — using generic fallback", bank)
    return _parse_generic_rows(all_rows)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_PURE_NUMERIC = re.compile(r"^\d+$")
_REF_TOKENS   = re.compile(r"\b\d{6,}\b")


def _clean_description(desc: str) -> str:
    if not desc:
        return ""
    desc = desc.strip()
    # Remove purely numeric tokens (ref/cheque numbers)
    tokens = desc.split()
    tokens = [t for t in tokens if not _PURE_NUMERIC.match(t)]
    desc   = " ".join(tokens)
    # Remove long ref numbers embedded in text
    desc   = _REF_TOKENS.sub("", desc).strip()
    desc   = _EXCESS_WS.sub(" ", desc)
    return desc[:200]


def _post_process(df: pd.DataFrame, bank_source: str) -> pd.DataFrame:
    """
    Apply post-processing cleaning to the raw parsed DataFrame.

    Steps:
      1. Date normalisation — strip time, drop nulls
      2. Amount cleaning  — ensure positive, drop zeros / nulls
      3. Description cleaning — remove ref numbers, truncate to 200 chars
      4. Duplicate flagging — flag (date + amount + description) duplicates
      5. Add bank_source column
    """
    df = df.copy()

    # 1. Date normalisation
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # 2. Amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df = df[df["amount"] > 0].dropna(subset=["amount"])

    # 3. Description
    df["description"] = df["description"].apply(
        lambda x: _clean_description(str(x)) if pd.notna(x) else ""
    )

    # 4. Duplicate flagging (keep first, flag subsequent)
    dup_mask = df.duplicated(subset=["date", "amount", "description"], keep="first")
    df["is_duplicate"] = dup_mask
    logger.info("Flagged %d duplicate rows (kept, not dropped)", int(dup_mask.sum()))

    # 5. bank_source
    df["bank_source"] = bank_source

    return df.reset_index(drop=True)


def _validate_output(df: pd.DataFrame) -> None:
    required = {"date", "description", "amount", "transaction_type", "bank_source"}
    missing = required - set(df.columns)
    if missing:
        raise PDFParseError(f"Normalized output missing columns: {missing}")

    null_counts = df[["date", "amount", "transaction_type"]].isnull().sum()
    if null_counts.any():
        raise PDFParseError(
            f"Normalized output has nulls: {null_counts[null_counts > 0].to_dict()}"
        )

    bad_types = ~df["transaction_type"].isin({"debit", "credit"})
    if bad_types.any():
        raise PDFParseError(
            f"transaction_type contains values other than 'debit'/'credit': "
            f"{df.loc[bad_types, 'transaction_type'].unique().tolist()}"
        )


# ---------------------------------------------------------------------------
# Records → normalized DataFrame
# ---------------------------------------------------------------------------

def _records_to_normalized_df(records: list[dict], bank_source: str) -> pd.DataFrame:
    """
    Convert raw parser records to the 5-column normalized output.

    raw_amount convention: positive = credit (money in), negative = debit (money out)
    Output amount: always positive — transaction_type indicates direction.
    """
    rows = []
    for r in records:
        raw_amt = float(r.get("raw_amount", 0) or 0)
        is_credit = raw_amt > 0

        desc  = str(r.get("description", "")).strip()
        notes = str(r.get("notes",       "")).strip()
        meta  = str(r.get("meta",        "")).strip()
        extras = " | ".join(x for x in (notes, meta) if x and x != desc)
        if extras:
            desc = f"{desc} | {extras}" if desc else extras

        date_val = r.get("date")
        rows.append({
            "date":             date_val,
            "description":      desc,
            "amount":           abs(raw_amt),
            "transaction_type": "credit" if is_credit else "debit",
            "bank_source":      bank_source,
        })

    if not rows:
        return pd.DataFrame(columns=["date", "description", "amount",
                                     "transaction_type", "bank_source"])

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Validation report (console)
# ---------------------------------------------------------------------------

def _print_routing_table(bank: str, parser_used: str, columns_found: list[str], row_count: int) -> None:
    print(f"\n{'='*60}")
    print("PDF PARSER ROUTING TABLE")
    print(f"{'='*60}")
    print(f"  Bank detected  : {bank}")
    print(f"  Parser used    : {parser_used}")
    print(f"  Columns found  : {columns_found}")
    print(f"  Transactions   : {row_count}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Master function — parse_pdf_statement()
# ---------------------------------------------------------------------------

def parse_pdf_statement(pdf_file: Any) -> pd.DataFrame:
    """
    Parse any supported PDF bank statement into a normalized DataFrame.

    Steps
    -----
    1. extract_text()       — pdfplumber primary, PyPDF2 fallback
    2. detect_bank()        — fingerprint matching
    3. parse_transactions() — route to bank-specific parser
    4. _post_process()      — clean dates, amounts, descriptions; flag dupes
    5. _validate_output()   — assert column contract

    Parameters
    ----------
    pdf_file : str | Path | file-like (Streamlit UploadedFile)

    Returns
    -------
    pd.DataFrame with columns:
        date             datetime64[ns], no time component
        description      str, cleaned
        amount           float, always positive
        transaction_type "debit" | "credit"
        bank_source      str, detected bank name
        is_duplicate     bool, flagged (not dropped)

    Raises
    ------
    ScannedPDFError  if PDF is image-only
    PDFParseError    if no transactions found
    """
    raw_bytes = _to_bytes(pdf_file)

    # Step 1 — Extract text
    full_text = _extract_via_pdfplumber(raw_bytes) or _extract_via_pypdf2(raw_bytes)
    if full_text is None:
        raise ScannedPDFError(
            "Could not extract text from PDF. "
            "Try uploading a text-based PDF, not a scanned image."
        )

    if len(full_text.strip()) < 150:
        raise ScannedPDFError(
            "This appears to be a scanned PDF — please use a text-based export "
            "from your bank's internet banking portal."
        )

    # Step 2 — Detect bank
    bank = detect_bank(full_text)

    # Step 3a — GPay: prefer table extraction (preserves original casing);
    #           fall back to regex text parser if tables yield < 3 tx rows.
    if bank == "GPay":
        # Priority 1: attempt table extraction from raw bytes
        _gpay_from_table = False
        try:
            import pdfplumber
            from io import BytesIO as _BytesIO
            _gpay_tx_rows: list[list] = []
            with pdfplumber.open(_BytesIO(raw_bytes)) as _gpdf:
                for _pg in _gpdf.pages:
                    for _tbl in (_pg.extract_tables() or []):
                        for _row in (_tbl or []):
                            # A real transaction row has ≥ 3 non-None cells
                            if _row and sum(c is not None for c in _row) >= 3:
                                _gpay_tx_rows.append(_row)
            if len(_gpay_tx_rows) >= 3:
                logger.info(
                    "GPay: table extraction found %d rows — using table parser",
                    len(_gpay_tx_rows),
                )
                _gpay_from_table = True
                # Table-based parsing path (future: parse _gpay_tx_rows here)
                # Currently GPay tables do not carry transaction data in known
                # format, so we log and fall through to the text parser.
                logger.warning(
                    "GPay table parser not yet implemented — "
                    "falling back to text parser"
                )
                _gpay_from_table = False
        except Exception as _te:
            logger.debug("GPay table extraction attempt failed: %s", _te)

        if not _gpay_from_table:
            logger.info("GPay: table rows < 3 — using regex text parser")

        df = parse_gpay_text(full_text)
        if len(df) == 0:
            raise PDFParseError(
                "No transactions could be extracted from this GPay statement. "
                "Ensure the file is the text-based PDF downloaded from Google Pay, "
                "not a screenshot."
            )
        df = _post_process(df, "GPay")
        if len(df) == 0:
            raise PDFParseError("All GPay rows were filtered out during cleaning.")
        _validate_output(df)
        _print_routing_table(
            bank="GPay",
            parser_used="parse_gpay_text (regex, line-based)",
            columns_found=list(df.columns),
            row_count=len(df),
        )
        logger.info("parse_pdf_statement: bank=GPay rows=%d", len(df))
        return df

    # Step 3b — Paytm: text-based parser (no reliable tables in Paytm PDFs)
    if bank == "Paytm":
        df = parse_paytm_text(full_text)
        if len(df) == 0:
            raise PDFParseError(
                "No transactions could be extracted from this Paytm statement. "
                "Ensure the file is the text-based PDF downloaded from the Paytm app, "
                "not a screenshot."
            )
        df = _post_process(df, "Paytm")
        if len(df) == 0:
            raise PDFParseError("All Paytm rows were filtered out during cleaning.")
        _validate_output(df)
        _print_routing_table(
            bank="Paytm",
            parser_used="parse_paytm_text (regex, line-based)",
            columns_found=list(df.columns),
            row_count=len(df),
        )
        logger.info("parse_pdf_statement: bank=Paytm rows=%d", len(df))
        return df

    # Step 3c — All other banks: table-based parsers
    try:
        all_rows, _, total_chars = _extract_all_rows(raw_bytes)
    except Exception as e:
        raise PDFParseError(f"Could not extract tables from PDF: {e}") from e

    if not all_rows:
        raise PDFParseError(
            "No tables could be extracted from the PDF. "
            "Ensure the file is a text-based bank statement, not a screenshot."
        )

    records = parse_transactions(bank, all_rows, full_text)

    # Fallback 1: positional column detection (no keyword headers needed)
    if not records:
        logger.info("Table parser returned no records — trying positional column detector")
        records = _parse_positional_rows(all_rows)

    # Fallback 2: universal line-by-line text scanner
    if not records:
        logger.info("Positional parser returned no records — trying universal text parser")
        records = _parse_universal_text(full_text)

    if not records:
        raise PDFParseError(
            "No transactions could be extracted from this PDF. "
            "The file appears to be a text-based PDF, but no transaction rows "
            "were found. Try exporting the statement as CSV from your bank's "
            "portal and uploading that instead."
        )

    # Step 4 — Convert records to DataFrame
    bank_label = bank if bank != "Unknown" else "Bank Statement"
    df = _records_to_normalized_df(records, bank_label)

    # Step 5 — Post-processing
    df = _post_process(df, bank)

    if len(df) == 0:
        raise PDFParseError(
            "All extracted rows were filtered out during cleaning. "
            "The PDF may contain non-standard formatting."
        )

    # Step 6 — Validate
    _validate_output(df)

    # Console routing table
    _print_routing_table(
        bank=bank,
        parser_used=f"{'split-column' if bank in ('HDFC','ICICI','SBI','Kotak','PNB','BOB','Canara') else bank.lower()}",
        columns_found=list(df.columns),
        row_count=len(df),
    )

    logger.info("parse_pdf_statement: bank=%s rows=%d", bank, len(df))
    return df


# ---------------------------------------------------------------------------
# Backward-compatible parse_pdf() — returns full 8-column pipeline schema
# ---------------------------------------------------------------------------

def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """
    Legacy conversion: raw records → 8-column pipeline schema.
    raw_amount: positive = credit, negative = debit
    Pipeline convention: amount positive = debit, negative = credit.
    """
    rows = []
    for r in records:
        raw_amt   = float(r.get("raw_amount", 0) or 0)
        is_credit = raw_amt > 0
        debit_val  = 0.0 if is_credit else abs(raw_amt)
        credit_val = abs(raw_amt) if is_credit else 0.0
        pipeline_amount = debit_val if not is_credit else -credit_val

        desc   = str(r.get("description", "")).strip()
        notes  = str(r.get("notes",       "")).strip()
        meta   = str(r.get("meta",        "")).strip()
        extras = " | ".join(x for x in (notes, meta) if x and x != desc)
        if extras:
            desc = f"{desc} | {extras}" if desc else extras

        date_val = r.get("date")
        my_str   = (
            date_val.strftime("%b %Y")
            if isinstance(date_val, (pd.Timestamp, datetime)) and pd.notna(date_val)
            else ""
        )

        rows.append({
            "date":             date_val,
            "description":      desc,
            "debit":            round(debit_val,  2),
            "credit":           round(credit_val, 2),
            "balance":          round(float(r.get("balance", 0) or 0), 2),
            "amount":           round(pipeline_amount, 2),
            "transaction_type": "credit" if is_credit else "debit",
            "month_year":       my_str,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "description", "debit", "credit", "balance",
            "amount", "transaction_type", "month_year",
        ])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def parse_pdf(file_source: Union[str, Path, IO]) -> pd.DataFrame:
    """
    Parse a PDF bank statement and return the 8-column pipeline schema.

    This is the backward-compatible entry point used by
    BankStatementPreprocessor.load_file().

    Internally calls parse_pdf_statement() and expands the output.

    Returns
    -------
    pd.DataFrame with columns:
        date, description, debit, credit, balance, amount,
        transaction_type, month_year
    """
    normalized = parse_pdf_statement(file_source)

    # Expand normalized 5-col → full 8-col pipeline schema
    is_credit = normalized["transaction_type"] == "credit"
    debit_col  = normalized["amount"].where(~is_credit, 0.0)
    credit_col = normalized["amount"].where(is_credit,  0.0)
    # Pipeline sign: debit positive, credit negative
    pipeline_amount = normalized["amount"].where(~is_credit, 0) - \
                      normalized["amount"].where(is_credit,  0)

    out = pd.DataFrame({
        "date":             normalized["date"],
        "description":      normalized["description"],
        "debit":            debit_col,
        "credit":           credit_col,
        "balance":          0.0,
        "amount":           pipeline_amount,
        "transaction_type": normalized["transaction_type"],
        "month_year":       normalized["date"].dt.strftime("%b %Y"),
    })

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out
