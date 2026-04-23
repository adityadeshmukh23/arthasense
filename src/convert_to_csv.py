"""
convert_to_csv.py — ArthaSense multi-format statement converter.

Public API
----------
convert_to_arthasense_csv(uploaded_file) -> pd.DataFrame
    Parse a PDF or Excel bank statement and return a clean DataFrame with
    columns: [date, description, amount, transaction_type, account]

    amount sign convention (caller-facing):
        positive  → credit  (money received / refund)
        negative  → debit   (money sent / payment)

to_pipeline_df(df) -> pd.DataFrame
    Bridge the 5-column output above to the full ArthaSense pipeline schema:
    [date, description, debit, credit, balance, amount, transaction_type, month_year]

    Pipeline amount sign convention (reversed):
        positive  → debit
        negative  → credit

Supported formats
-----------------
PDF  : Paytm UPI statement (primary), generic split-column / single-amount fallback
Excel: .xlsx / .xls with auto-detected header row
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Amount helpers ────────────────────────────────────────────────────────────

# "+ Rs.8,000"  "- Rs.1,23,456.00"  "+₹ 500"  "-INR2,000"
_AMT_SIGNED = re.compile(
    r"(?P<sign>[+\-])\s*(?:Rs\.?|₹|INR)\s*(?P<num>[\d,]+(?:\.\d+)?)", re.I
)
# "Rs.8,000"  "₹553"  (no sign → debit)
_AMT_PLAIN = re.compile(
    r"(?:Rs\.?|₹|INR)\s*(?P<num>[\d,]+(?:\.\d+)?)", re.I
)
# "8,000.00 Cr"  "553 Dr"
_AMT_CRDR = re.compile(
    r"(?:Rs\.?|₹|INR)?\s*(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<suf>CR|DR|C|D)\b", re.I
)
_EMPTY = frozenset({"", "-", "--", "nil", "n/a", "na", "nan", "null", "—", "–"})


def _parse_amount(raw: str) -> float | None:
    """
    Parse a bank amount cell → signed float.
      positive = credit (money in)
      negative = debit  (money out)
    Returns None when the string is not a recognisable amount.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.lower() in _EMPTY:
        return None

    m = _AMT_SIGNED.search(s)
    if m:
        num = float(m.group("num").replace(",", ""))
        return num if m.group("sign") == "+" else -num

    m = _AMT_CRDR.search(s)
    if m:
        num = float(m.group("num").replace(",", ""))
        return num if m.group("suf").upper() in ("CR", "C") else -num

    m = _AMT_PLAIN.search(s)
    if m:
        return -float(m.group("num").replace(",", ""))  # unsigned → debit

    return None


# ─── Date helpers ──────────────────────────────────────────────────────────────

# Matches "22 Mar", "03 Dec", "1 Jan" (day + abbreviated month, no year)
_DATE_DAYMON = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.I,
)
# Matches time like "11:39 PM", "23:59", "09:05 AM"
_TIME_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})(?:\s*(?P<ampm>AM|PM))?", re.I
)
# 4-digit year  e.g. 2025
_YEAR_4DIG_RE = re.compile(r"\b(20\d{2})\b")
# 2-digit year after month name or apostrophe: "DEC'25", "MAR'26", "Dec 25"
_YEAR_2DIG_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\s\-]+(\d{2})\b",
    re.I,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_years_from_text(text: str) -> list[int]:
    """
    Extract calendar years from header text.
    Handles both 4-digit (2025) and 2-digit apostrophe formats (DEC'25 → 2025).
    """
    years: set[int] = set()
    for y in _YEAR_4DIG_RE.findall(text):
        years.add(int(y))
    for m in _YEAR_2DIG_RE.finditer(text):
        yy = int(m.group(1))
        if 20 <= yy <= 99:          # only reasonable 21st-century years
            years.add(2000 + yy)
    return sorted(years)


def _parse_day_mon_time(cell_text: str) -> tuple[int, int, int, int, int] | None:
    """
    Parse a Paytm date cell like "22 Mar\\n11:39 PM".
    Returns (month, day, hour24, minute, second) or None.
    """
    if not cell_text:
        return None
    text = cell_text.replace("\n", " ").strip()

    dm = _DATE_DAYMON.search(text)
    if not dm:
        return None
    day   = int(dm.group("day"))
    month = _MONTH_MAP.get(dm.group("mon").lower())
    if not month:
        return None

    hour, minute = 0, 0
    tm = _TIME_RE.search(text)
    if tm:
        hour   = int(tm.group("h"))
        minute = int(tm.group("m"))
        ampm   = (tm.group("ampm") or "").upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0

    return (month, day, hour, minute, 0)


def _safe_timestamp(year: int, month: int, day: int,
                    h: int, m: int, s: int) -> pd.Timestamp:
    try:
        return pd.Timestamp(year=year, month=month, day=day,
                            hour=h, minute=m, second=s)
    except ValueError:
        return pd.NaT


def _assign_years(
    parsed: list[tuple[int, int, int, int, int]],
    min_year: int,
    max_year: int | None = None,
) -> list[pd.Timestamp]:
    """
    Assign calendar years to (month, day, H, M, S) tuples.

    Automatically detects whether the statement is chronological (oldest first)
    or reverse-chronological (newest first — typical for Paytm exports) and
    assigns years accordingly so that year boundaries (e.g. Dec→Jan) are handled
    correctly in both directions.

    Parameters
    ----------
    parsed   : list of (month, day, hour, minute, second) tuples
    min_year : the earlier year seen in the statement header (e.g. 2025)
    max_year : the later year seen in the statement header (e.g. 2026)
    """
    if not parsed:
        return []
    if max_year is None:
        max_year = min_year + 1

    # ── Detect direction from the first 30 month values ───────────────────
    # Count how many consecutive-pair transitions go month-forward vs month-back.
    # Ignore large jumps (>6 months) which are year boundaries and not direction signals.
    months_sample = [p[0] for p in parsed[:30]]
    forward_count = 0
    backward_count = 0
    for i in range(1, len(months_sample)):
        diff = months_sample[i] - months_sample[i - 1]
        if abs(diff) <= 6:          # not a year-boundary jump
            if diff >= 0:
                forward_count += 1
            else:
                backward_count += 1

    oldest_first = forward_count >= backward_count
    logger.info(
        "Year assignment: forward=%d backward=%d → %s",
        forward_count, backward_count,
        "oldest-first (ascending)" if oldest_first else "newest-first (descending)",
    )

    timestamps: list[pd.Timestamp] = []
    prev_month = parsed[0][0]

    if oldest_first:
        # ── Chronological: Dec 2025 → Jan 2026 → Mar 2026 ─────────────────
        current_year = min_year
        for month, day, h, m, s in parsed:
            # Month rolled significantly backward → new year started
            if month < prev_month - 2:
                current_year += 1
            prev_month = month
            timestamps.append(_safe_timestamp(current_year, month, day, h, m, s))
    else:
        # ── Reverse-chronological: Mar 2026 → Jan 2026 → Dec 2025 ─────────
        current_year = max_year
        for month, day, h, m, s in parsed:
            # Month jumped significantly forward → crossed a year boundary backward
            if month > prev_month + 2:
                current_year -= 1
            prev_month = month
            timestamps.append(_safe_timestamp(current_year, month, day, h, m, s))

    return timestamps


# ─── Statement-period helpers (month → year direct lookup) ────────────────────

# Matches "23 DEC'25" or "22 MAR'26" inside a statement period header
_PERIOD_RE = re.compile(
    r"\b\d{1,2}\s+(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"['\s\-]+(?P<yr>\d{2})\b",
    re.I,
)


def _extract_period(text: str) -> tuple[int, int, int, int] | None:
    """
    Extract (start_month, start_year, end_month, end_year) from header text.
    E.g. "23 DEC'25 - 22 MAR'26" → (12, 2025, 3, 2026).
    Returns None if fewer than 2 date matches are found.
    """
    matches = list(_PERIOD_RE.finditer(text))
    if len(matches) < 2:
        return None
    results = []
    for m in matches[:2]:
        mon = _MONTH_MAP.get(m.group("mon").lower())
        yr = 2000 + int(m.group("yr"))
        if mon:
            results.append((mon, yr))
    if len(results) < 2:
        return None
    (mon1, yr1), (mon2, yr2) = results
    if yr1 < yr2 or (yr1 == yr2 and mon1 <= mon2):
        return (mon1, yr1, mon2, yr2)
    return (mon2, yr2, mon1, yr1)


def _build_month_year_map(
    start_month: int, start_year: int,
    end_month: int, end_year: int,
) -> dict[int, int]:
    """
    Build {month: year} lookup covering the full statement period.
    E.g. start=(12, 2025) end=(3, 2026) → {12: 2025, 1: 2026, 2: 2026, 3: 2026}.
    """
    result: dict[int, int] = {}
    y, m = start_year, start_month
    for _ in range(14):              # safety: at most 14 months
        result[m] = y
        if y == end_year and m == end_month:
            break
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return result


# ─── Description / account cleaners ───────────────────────────────────────────

# Lines that are UPI metadata — strip them from description
_UPI_META = re.compile(
    r"^(UPI\s*(Ref|ID|Ref\s*No|Transaction\s*ID)|Order\s*ID|Bank\s*Ref"
    r"|Mandate\s*ID|Transaction\s*Ref|Ref\s*No\.?)\b",
    re.I,
)

# Header row in the table (column names row)
_HEADER_ROW = re.compile(
    r"(date\s*(?:&|and|/|\|)\s*time|transaction\s+details|your\s+account)",
    re.I,
)

# Page-level noise rows that are NOT transactions.
# Deliberately narrow to avoid accidentally dropping legitimate merchant names.
_SKIP_ROW = re.compile(
    r"(?:"
    r"page\s+\d+\s+of\s+\d+"               # "Page 5 of 30"
    r"|powered\s+by"                         # "Powered by Paytm"
    r"|for\s+any\s+(?:queries|support|help)" # "For any queries..."
    r"|^dear\s+\w"                           # "Dear Customer"
    r"|helpdesk@"                            # Paytm helpdesk email
    r"|toll[\s\-]?free[\s:]+\d"             # "Toll-free: 1800..."
    r"|this\s+(?:statement|document)\s+is"  # "This statement is generated..."
    r"|statement\s+summary\s*$"             # standalone section header
    r"|total\s+debits?\s*:"                 # "Total Debits: ₹X"
    r"|total\s+credits?\s*:"               # "Total Credits: ₹X"
    r"|opening\s+balance\s*[:\-]"          # "Opening Balance:"
    r"|closing\s+balance\s*[:\-]"          # "Closing Balance:"
    r")",
    re.I,
)


def _clean_description(raw: str) -> str:
    """
    Keep only the merchant / payee name (first non-empty, non-metadata line).
    Strip all subsequent lines which are UPI IDs, ref numbers, etc.
    """
    if not raw:
        return ""
    lines = [ln.strip() for ln in str(raw).replace("\r", "\n").split("\n")]
    for line in lines:
        if line and not _UPI_META.match(line):
            return line
    return lines[0] if lines else ""


def _clean_account(raw: str) -> str:
    """
    "State Bank Of India - 24"   → "State Bank Of India"
    "ICICI Bank Credit Card - 1" → "ICICI Bank Credit Card"
    "UPI Lite"                   → "UPI Lite"
    Strips trailing " - <digits>" suffixes (masked account numbers).
    """
    if not raw:
        return ""
    cleaned = re.sub(r"\s*[-–]\s*\d+\s*$", "", str(raw).strip())
    cleaned = cleaned.split("\n")[0].strip()
    return cleaned


def _is_self_transfer(description: str) -> bool:
    return bool(re.search(r"self\s*transfer", description, re.I))


# ─── Paytm PDF parser ─────────────────────────────────────────────────────────

def _norm_cell(c: Any) -> str:
    """Normalise a raw pdfplumber table cell to a clean string."""
    if c is None:
        return ""
    return re.sub(r"[ \t]{2,}", " ", str(c).strip())


def _parse_paytm_pdf(pdf_source: Any) -> pd.DataFrame:
    """
    Parse a Paytm UPI statement PDF using extract_text() line-by-line.

    Paytm's PDF uses variable-width columns that pdfplumber's extract_tables()
    only partially captures (~50% hit rate).  extract_text() returns all
    transactions reliably in a consistent layout:

      Line 0: "{D} {Mon}  {description}  [Note/Tag: ...]  {account_frag}  {+/-} Rs.{amount}"
      Line 1: "{HH:MM AM/PM}"
      Lines 2+: UPI ID / Ref No / Order ID  (skipped for description)

    Year assignment uses the statement period header ("23 DEC'25 - 22 MAR'26")
    for a direct month→year lookup, avoiding direction-detection pitfalls when
    many consecutive transactions fall in the same month.
    """
    import pdfplumber

    # ── Read source bytes ─────────────────────────────────────────────────
    raw_bytes = None
    if hasattr(pdf_source, "getvalue"):
        raw_bytes = pdf_source.getvalue()
    elif hasattr(pdf_source, "read"):
        raw_bytes = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
    src = BytesIO(raw_bytes) if raw_bytes is not None else pdf_source

    # Lines that are page chrome — discard after header is found
    _SKIP_LINE = re.compile(
        r"^(?:"
        r"Page\s+\d+\s+of\s+\d+"
        r"|For\s+any\s+queries"
        r"|Contact\s+Us"
        r"|Passbook\s+Payments\s+History"
        r"|All\s+payments\s+done\s+by"
        r"|Date\s*&"
        r"|Time$"
        r"|Transaction\s+Details"
        r"|Your\s+Account"
        r"|Amount$"
        r")",
        re.I,
    )

    # A new transaction starts when a line begins with "D+ Mon"
    _TX_LINE = re.compile(
        r"^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
        re.I,
    )

    # ── Extract text lines from every page ────────────────────────────────
    header_text = ""
    all_lines: list[str] = []

    with pdfplumber.open(src) as pdf:
        n_pages = len(pdf.pages)
        logger.info("Paytm PDF: %d pages", n_pages)

        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_idx < 3:
                header_text += page_text

            past_header = False          # reset for each page
            for raw_line in page_text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                # Skip everything on this page until the column-header row
                if not past_header:
                    if re.search(r"Transaction\s+Details", line, re.I):
                        past_header = True
                    continue
                # Skip page chrome lines
                if _SKIP_LINE.match(line):
                    continue
                all_lines.append(line)

    logger.info("Content lines after filtering: %d", len(all_lines))

    if not all_lines:
        raise ValueError(
            "No transaction text found in PDF. "
            "Ensure the file is a text-based Paytm statement, not a screenshot."
        )

    # ── Year / period extraction ──────────────────────────────────────────
    years = _extract_years_from_text(header_text)
    min_year = years[0] if years else datetime.now().year - 1
    max_year = years[-1] if len(years) > 1 else min_year + 1

    period = _extract_period(header_text)
    if period:
        start_mon, start_yr, end_mon, end_yr = period
        month_to_year = _build_month_year_map(start_mon, start_yr, end_mon, end_yr)
        logger.info(
            "Period: %d/%d – %d/%d  month_to_year=%s",
            start_mon, start_yr, end_mon, end_yr, month_to_year,
        )
    else:
        month_to_year = {}
        logger.warning("Could not extract statement period; will use direction detection")

    # ── Group lines into transaction blocks ───────────────────────────────
    blocks: list[list[str]] = []
    current_block: list[str] | None = None

    for line in all_lines:
        if _TX_LINE.match(line):
            if current_block is not None:
                blocks.append(current_block)
            current_block = [line]
        elif current_block is not None:
            current_block.append(line)
        # else: stray line before first transaction — discard

    if current_block is not None:
        blocks.append(current_block)

    logger.info("Transaction blocks: %d", len(blocks))

    # ── Parse each block ──────────────────────────────────────────────────
    raw_parsed: list[tuple] = []
    stats_skip = {"no_amount": 0, "bad_date": 0, "self_transfer": 0}

    for block in blocks:
        if not block:
            continue
        line0 = block[0]

        # Date
        dm = _DATE_DAYMON.search(line0)
        if not dm:
            stats_skip["bad_date"] += 1
            continue
        day = int(dm.group("day"))
        month = _MONTH_MAP.get(dm.group("mon").lower())
        if not month:
            stats_skip["bad_date"] += 1
            continue

        # Time: check line0 first (rare), then line1 (typical)
        hour, minute = 0, 0
        for tline in ([line0] + block[1:3]):
            tm = _TIME_RE.search(tline)
            if tm:
                hour = int(tm.group("h"))
                minute = int(tm.group("m"))
                ampm = (tm.group("ampm") or "").upper()
                if ampm == "PM" and hour != 12:
                    hour += 12
                elif ampm == "AM" and hour == 12:
                    hour = 0
                break

        # Amount: only signed (+ or -) amounts; work on text after the date
        text_after_date = line0[dm.end():].strip()
        amt_m = None
        for m in _AMT_SIGNED.finditer(text_after_date):
            amt_m = m                                    # keep the LAST match
        if amt_m is None:
            stats_skip["no_amount"] += 1
            logger.debug("No signed amount: %r", line0)
            continue
        num = float(amt_m.group("num").replace(",", ""))
        amount = num if amt_m.group("sign") == "+" else -num

        # Description: cut at first of (Note: / Tag: / amount sign)
        cut = amt_m.start()
        while cut > 0 and text_after_date[cut - 1] == " ":
            cut -= 1
        for marker in (" Note:", " Tag:", " note:", " tag:"):
            pos = text_after_date.find(marker)
            if 0 <= pos < cut:
                cut = pos
        desc = text_after_date[:cut].strip()
        if not desc:
            desc = text_after_date.split(" Note:")[0].split(" Tag:")[0].strip()

        if _is_self_transfer(desc):
            stats_skip["self_transfer"] += 1
            continue

        # Account (best-effort from merged lines)
        account = ""
        for bline in block:
            if re.search(r"State\s+Bank", bline, re.I):
                account = "State Bank Of India"
                break
            if re.search(r"ICICI\s+Bank", bline, re.I):
                account = "ICICI Bank"
                break
            if re.search(r"UPI\s+Lite", bline, re.I):
                account = "UPI Lite"
                break
            if re.search(r"Gold\s+Coin", bline, re.I):
                account = "Gold Coins"
                break

        raw_parsed.append((month, day, hour, minute, 0, desc, account, amount))

    logger.info(
        "Parse summary: accepted=%d | no_amount=%d bad_date=%d self_transfer=%d",
        len(raw_parsed),
        stats_skip["no_amount"], stats_skip["bad_date"], stats_skip["self_transfer"],
    )

    if not raw_parsed:
        raise ValueError(
            "No transactions could be extracted from the Paytm PDF. "
            "Check that the file is a text-based statement export."
        )

    # ── Assign years ──────────────────────────────────────────────────────
    timestamps: list[pd.Timestamp] = []
    if month_to_year:
        # Direct lookup from period header — most reliable
        for month, day, h, m, s, *_ in raw_parsed:
            year = month_to_year.get(month, min_year)
            timestamps.append(_safe_timestamp(year, month, day, h, m, s))
    else:
        # Fallback: direction detection
        partial_dates = [(r[0], r[1], r[2], r[3], r[4]) for r in raw_parsed]
        timestamps = _assign_years(partial_dates, min_year, max_year)

    # ── Build DataFrame ───────────────────────────────────────────────────
    rows = []
    for ts, record in zip(timestamps, raw_parsed):
        _, _, _, _, _, desc, account, amount = record
        if pd.isna(ts):
            continue
        rows.append({
            "date":             ts,
            "description":      desc,
            "amount":           amount,
            "transaction_type": "credit" if amount > 0 else "debit",
            "account":          account,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    n_debit  = (df["transaction_type"] == "debit").sum()
    n_credit = (df["transaction_type"] == "credit").sum()
    logger.info(
        "Paytm PDF parsed: %d transactions (%d debit, %d credit)",
        len(df), n_debit, n_credit,
    )
    print(
        f"[ArthaSense] PDF parsed: {len(df)} transactions "
        f"({n_debit} debits + {n_credit} credits)"
    )
    return df


# ─── Excel parser ──────────────────────────────────────────────────────────────

_HEADER_KW = {
    "date", "narration", "description", "particulars", "transaction",
    "amount", "debit", "credit", "withdrawal", "deposit", "balance",
    "txn", "ref", "cheque",
}


def _score_header(row: pd.Series) -> int:
    text = " ".join(str(v).lower() for v in row if pd.notna(v))
    return sum(1 for kw in _HEADER_KW if re.search(r"\b" + kw + r"\b", text))


def _find_excel_header(df_peek: pd.DataFrame) -> int:
    best_row, best_score = 0, 1
    for i in range(min(20, len(df_peek))):
        s = _score_header(df_peek.iloc[i])
        if s > best_score:
            best_score, best_row = s, i
    return best_row


def _parse_excel(file_source: Any) -> pd.DataFrame:
    """
    Read an Excel bank statement and produce the 5-column output schema.
    Uses the existing preprocessor for column normalisation.
    """
    engine = None
    if hasattr(file_source, "name"):
        engine = "xlrd" if Path(file_source.name).suffix.lower() == ".xls" else "openpyxl"

    # Peek to find header row
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    df_peek = pd.read_excel(file_source, header=None, nrows=20, engine=engine)
    hdr = _find_excel_header(df_peek)
    logger.info("Excel header at row %d", hdr)

    if hasattr(file_source, "seek"):
        file_source.seek(0)
    df = pd.read_excel(file_source, header=hdr, engine=engine)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]

    if df.empty:
        raise ValueError("Excel file contains no data rows after the header.")

    # Delegate to preprocessor for clean schema, then re-shape to 5 cols
    from src.preprocessor import BankStatementPreprocessor
    pp    = BankStatementPreprocessor()
    clean = pp.clean(df)

    # Convert to the 5-column caller schema
    out = pd.DataFrame({
        "date":             clean["date"],
        "description":      clean["description"],
        # flip pipeline sign: pipeline amount>0=debit; we want debit=negative
        "amount":           clean["amount"].apply(lambda x: -x),
        "transaction_type": clean["transaction_type"],
        "account":          "Excel Import",
    })
    return out


# ─── Public API ────────────────────────────────────────────────────────────────

def convert_to_arthasense_csv(
    uploaded_file: Any,
) -> pd.DataFrame:
    """
    Convert a PDF or Excel bank statement into a clean, pipeline-ready DataFrame.

    Parameters
    ----------
    uploaded_file : str | Path | Streamlit UploadedFile
        The uploaded bank statement file.  Extension determines the parser:
        .pdf  → multi-bank PDF parser (GPay, PhonePe, Paytm, HDFC, ICICI,
                SBI, Axis, Kotak, PNB, BOB, Canara, AmEx, SBI Card, etc.)
        .xlsx / .xls → Excel parser via openpyxl / xlrd

    Returns
    -------
    pd.DataFrame with columns:
        date             (datetime64[ns])
        description      (str)   — merchant / payee name only
        amount           (float) — positive = credit, negative = debit
        transaction_type (str)   — "credit" | "debit"
        bank_source      (str)   — detected bank / app name
        account          (str)   — empty string (compat placeholder)

    Raises
    ------
    ValueError     if no transactions can be extracted
    ImportError    if pdfplumber / openpyxl is missing
    """
    ext = (
        Path(uploaded_file.name).suffix.lower()
        if hasattr(uploaded_file, "name")
        else Path(str(uploaded_file)).suffix.lower()
    )

    if ext == ".pdf":
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF parsing. "
                "Run: pip install pdfplumber"
            )
        from src.pdf_parser import parse_pdf_statement
        df = parse_pdf_statement(uploaded_file)
        # Adapt sign convention: parse_pdf_statement returns amount always positive.
        # to_pipeline_df() expects positive = credit, negative = debit.
        is_debit = df["transaction_type"] == "debit"
        df["amount"] = df["amount"].where(~is_debit, -df["amount"])
        df["account"] = ""
        return df[["date", "description", "amount", "transaction_type",
                   "bank_source", "account"]]

    elif ext in (".xlsx", ".xls"):
        df = _parse_excel(uploaded_file)
        df["bank_source"] = "Excel"
        return df

    elif ext == ".csv":
        from src.preprocessor import BankStatementPreprocessor
        pp    = BankStatementPreprocessor()
        raw   = pp.load_csv(uploaded_file)
        clean = pp.clean(raw)
        out = pd.DataFrame({
            "date":             clean["date"],
            "description":      clean["description"],
            "amount":           clean["amount"].apply(lambda x: -x),
            "transaction_type": clean["transaction_type"],
            "account":          "CSV Import",
            "bank_source":      "CSV",
        })
        return out

    else:
        # Generic fallback: try CSV parsing for any unknown extension
        try:
            from src.preprocessor import BankStatementPreprocessor
            pp    = BankStatementPreprocessor()
            raw   = pp.load_csv(uploaded_file)
            clean = pp.clean(raw)
            out = pd.DataFrame({
                "date":             clean["date"],
                "description":      clean["description"],
                "amount":           clean["amount"].apply(lambda x: -x),
                "transaction_type": clean["transaction_type"],
                "account":          "CSV Import",
                "bank_source":      "CSV",
            })
            logger.info("Unknown extension '%s' parsed as CSV successfully", ext)
            return out
        except Exception as _csv_err:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                "Please upload a PDF, Excel (.xlsx/.xls), or CSV file."
            ) from _csv_err


def to_pipeline_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 5-column convert_to_arthasense_csv() output into the full
    ArthaSense pipeline schema expected by run_pipeline().

    Input amount sign:  positive = credit, negative = debit
    Pipeline amount sign: positive = debit, negative = credit  (reversed)

    Output columns:
        date, description, debit, credit, balance, amount, transaction_type, month_year
    """
    df = df.copy()

    is_credit = df["transaction_type"] == "credit"
    signed    = df["amount"]                       # caller sign: + credit / - debit

    debit_col  = signed.where(~is_credit, 0).abs()
    credit_col = signed.where(is_credit,  0).abs()

    # Pipeline convention: amount>0 = debit, amount<0 = credit → flip sign
    pipeline_amount = signed.apply(lambda x: abs(x) if x < 0 else -abs(x))

    out = pd.DataFrame({
        "date":             df["date"],
        "description":      df["description"],
        "debit":            debit_col,
        "credit":           credit_col,
        "balance":          0.0,
        "amount":           pipeline_amount,
        "transaction_type": df["transaction_type"],
        "account":          df.get("account", pd.Series([""] * len(df))),
        "bank_source":      df.get("bank_source", pd.Series([""] * len(df))),
        "month_year":       df["date"].dt.strftime("%b %Y"),
    })

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


# ─── CLI test helper ───────────────────────────────────────────────────────────

def _debug_pdf(path: str, n_pages: int = 3) -> None:
    """
    Print raw pdfplumber output for the first n_pages of a Paytm PDF.
    Shows extract_text(), extract_tables() results, and the raw rows list
    that the current parser would receive.
    """
    import pdfplumber

    _LINE_SETTINGS = dict(
        vertical_strategy="lines", horizontal_strategy="lines",
        snap_tolerance=4, join_tolerance=4, min_words_vertical=1,
    )
    _TEXT_SETTINGS = dict(
        vertical_strategy="text", horizontal_strategy="text",
        snap_tolerance=4, join_tolerance=4,
    )

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        print(f"\n{'='*70}")
        print(f"PDF: {path}")
        print(f"Total pages: {total}")
        print(f"{'='*70}")

        for i in range(min(n_pages, total)):
            page = pdf.pages[i]
            print(f"\n{'─'*70}")
            print(f"PAGE {i+1} — extract_text()")
            print(f"{'─'*70}")
            raw_text = page.extract_text() or "(empty)"
            print(raw_text)

            print(f"\n{'─'*70}")
            print(f"PAGE {i+1} — extract_tables() with LINE strategy")
            print(f"{'─'*70}")
            line_tables = page.extract_tables(_LINE_SETTINGS) or []
            print(f"  Tables found: {len(line_tables)}")
            for ti, tbl in enumerate(line_tables):
                print(f"  Table {ti+1}: {len(tbl)} rows")
                for ri, row in enumerate(tbl[:5]):
                    print(f"    row[{ri}]: {row}")
                if len(tbl) > 5:
                    print(f"    ... ({len(tbl)-5} more rows)")

            print(f"\n{'─'*70}")
            print(f"PAGE {i+1} — extract_tables() with TEXT strategy")
            print(f"{'─'*70}")
            text_tables = page.extract_tables(_TEXT_SETTINGS) or []
            print(f"  Tables found: {len(text_tables)}")
            for ti, tbl in enumerate(text_tables):
                print(f"  Table {ti+1}: {len(tbl)} rows")
                for ri, row in enumerate(tbl[:5]):
                    print(f"    row[{ri}]: {row}")
                if len(tbl) > 5:
                    print(f"    ... ({len(tbl)-5} more rows)")

        # Show what the current parser's all_raw_rows list looks like
        print(f"\n{'='*70}")
        print("RAW ROWS fed to current parser (first 20 rows across all pages)")
        print(f"{'='*70}")
        all_raw_rows: list = []
        for i, page in enumerate(pdf.pages):
            tables: list = []
            for settings in (_LINE_SETTINGS, _TEXT_SETTINGS):
                tables = page.extract_tables(settings) or []
                if tables:
                    break
            for tbl in tables:
                for row in (tbl or []):
                    all_raw_rows.append((i + 1, row))

        print(f"Total raw rows across all {total} pages: {len(all_raw_rows)}")
        print()
        for idx, (pg, row) in enumerate(all_raw_rows[:20]):
            print(f"  [page {pg}, row {idx:3d}] {row}")
        if len(all_raw_rows) > 20:
            print(f"  ... ({len(all_raw_rows)-20} more rows)")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s  %(message)s")

    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python -m src.convert_to_csv <path/to/statement.pdf>")
        print("  python -m src.convert_to_csv --debug <path/to/statement.pdf>")
        sys.exit(1)

    if args[0] == "--debug":
        if len(args) < 2:
            print("Usage: python -m src.convert_to_csv --debug <path/to/statement.pdf>")
            sys.exit(1)
        _debug_pdf(args[1])
        sys.exit(0)

    path = args[0]
    result = convert_to_arthasense_csv(path)
    print(f"\nResult shape : {result.shape}")
    print(f"Date range   : {result['date'].min().date()} -> {result['date'].max().date()}")
    print(f"Debits       : {(result['transaction_type']=='debit').sum()}")
    print(f"Credits      : {(result['transaction_type']=='credit').sum()}")
    print("\nFirst 5 rows:")
    print(result.head().to_string(index=False))
