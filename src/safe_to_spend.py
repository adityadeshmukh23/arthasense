"""
safe_to_spend.py — the one job: "How much can I safely spend this week
without running short before salary day?"

Everything here is deterministic and free of Streamlit so it can be tested.

    weekly = (balance - bills due before payday - buffer)
             / days to payday * min(7, days to payday)

No AI, no network calls.
"""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.preprocessor import BankStatementPreprocessor

DEFAULT_BUFFER = 2000
MIN_WEEKS_FOR_TIP = 4
MAX_WEEKS_AVERAGED = 8
TIP_CUT = 0.25

# ---------------------------------------------------------------------------
# Errors and small helpers
# ---------------------------------------------------------------------------


class StatementError(Exception):
    """A problem with the uploaded file, explained in plain words."""


def inr(amount: float, signed: bool = False) -> str:
    """Format as Indian rupees with lakh grouping, rounded to the rupee."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "—"
    neg = amount < 0
    n = int(round(abs(amount)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    sign = "-" if neg else ("+" if signed and n > 0 else "")
    return f"{sign}₹{s}"


def add_months(d: date, months: int, day: Optional[int] = None) -> date:
    """Same day-of-month `months` later, clipped to the month's length."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    want = day if day is not None else d.day
    return date(y, m, min(want, calendar.monthrange(y, m)[1]))


def _to_date(x: Any) -> date:
    return pd.Timestamp(x).date()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Statement:
    txns: pd.DataFrame                 # date, description, debit, credit, balance, group, is_self
    start: date
    end: date
    balance: Optional[float]           # closing balance from the file, if any
    balance_date: Optional[date]
    balance_reliable: bool
    source: str                        # "CSV" or "Excel"
    notes: list[str] = field(default_factory=list)

    @property
    def days_covered(self) -> int:
        return (self.end - self.start).days + 1


def _ext(source: Any, filename: Optional[str]) -> str:
    name = filename or getattr(source, "name", None) or str(source)
    return Path(name).suffix.lower()


def load_statement(source: Any, filename: Optional[str] = None) -> Statement:
    """Read a CSV or Excel (.xlsx) statement into a Statement.

    Raises StatementError with a plain-language message on any failure.
    """
    ext = _ext(source, filename)
    pp = BankStatementPreprocessor()
    try:
        if ext == ".csv":
            df = pp.clean(pp.load_csv(source), dedupe=False)
            src_label = "CSV"
        elif ext == ".xlsx":
            from src.excel_parser import load_excel_raw
            df = pp.clean(load_excel_raw(source), dedupe=False)
            src_label = "Excel"
        elif ext == ".pdf":
            # PDF parsing was unreliable in testing (a Dr/Cr-column layout lost
            # half the rows and every credit without raising an error), so the
            # safe-to-spend journey refuses PDFs rather than risk a wrong answer.
            raise StatementError(
                "PDF statements aren't supported here yet because we couldn't read them "
                "reliably. Download a CSV or Excel statement from your net banking instead.")
        else:
            raise StatementError("Please upload a CSV or Excel (.xlsx) bank statement.")
    except StatementError:
        raise
    except ImportError:
        raise StatementError("This file type needs an extra library that isn't installed here. Try a CSV export.")
    except Exception as exc:  # noqa: BLE001 — every parser failure becomes a plain message
        name = type(exc).__name__
        if "Scanned" in name:
            raise StatementError("This PDF looks like a scanned image, so we can't read the numbers. Download a CSV from your bank instead.")
        raise StatementError(
            "We couldn't read transactions from this file. Check that it has a date, "
            "a description and debit/credit (or amount) columns."
        ) from exc

    return _build_statement(df, src_label)


def _build_statement(df: pd.DataFrame, src_label: str) -> Statement:
    needed = {"date", "description", "debit", "credit"}
    if df is None or df.empty or not needed.issubset(df.columns):
        raise StatementError("We couldn't find any transactions in this file.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0).abs()
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0).abs()
    df = df[(df["debit"] > 0) | (df["credit"] > 0)]
    if df.empty:
        raise StatementError("We found rows but no amounts in this file.")
    if "balance" not in df.columns:
        df["balance"] = np.nan
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df.reset_index(drop=True)

    df["is_self"] = mark_self_transfers(df)
    df["group"] = [spend_group(d) for d in df["description"]]
    df.loc[df["credit"] > 0, "group"] = "Money in"
    df.loc[df["is_self"], "group"] = "Own-account transfer"

    start, end = _to_date(df["date"].min()), _to_date(df["date"].max())
    notes: list[str] = []

    balance = balance_date = None
    reliable = False
    last = df.iloc[-1]
    if pd.notna(last["balance"]):
        balance = float(last["balance"])
        balance_date = end
        reliable = _balance_consistent(df)
        if not reliable:
            notes.append("The balance column doesn't add up row by row, so please check the balance.")
    else:
        notes.append("This file has no balance column. Enter your current balance.")

    if (end - start).days + 1 < 28:
        notes.append("This statement covers less than 4 weeks, so some bills may be missing.")

    return Statement(df, start, end, balance, balance_date, reliable, src_label, notes)


def _balance_consistent(df: pd.DataFrame, tol: float = 1.0) -> bool:
    b = df["balance"]
    if b.isna().any() or len(df) < 2:
        return bool(b.notna().all())
    expected = b.shift(1) + df["credit"] - df["debit"]
    ok = (expected - b).abs().iloc[1:] <= tol
    return bool(ok.mean() >= 0.9)


# ---------------------------------------------------------------------------
# Classification (plain-word groups)
# ---------------------------------------------------------------------------

_SELF_RE = re.compile(
    r"\bself\b|\bown\s*a/?c\b|\bown\s+account\b|\bto\s+own\b|\bsweep\b|\bmy\s+account\b",
    re.I,
)

# Ordered: first match wins.
_GROUP_RULES: list[tuple[str, re.Pattern]] = [
    ("Rent", re.compile(r"\brent\b|\bhouse\s*owner\b|\blandlord\b|nobroker", re.I)),
    ("EMI / loan", re.compile(r"\bemi\b|\bloan\b|\bnach\b|\bach\s*dr\b|\becs\b|bajaj\s*fin|home\s*credit", re.I)),
    ("Bills and subscriptions", re.compile(
        r"netflix|spotify|prime\s*video|amazon\s*prime|hotstar|youtube|zee5|sony\s*liv|jiocinema|"
        r"apple\.com|icloud|google\s*(one|play)|subscription|membership|\bgym\b|cult\.?fit|"
        r"electricity|bescom|msedcl|tata\s*power|adani|bses|\bpower\b|water\s*bill|\bgas\b|"
        r"broadband|fibernet|fiber|\bwifi\b|postpaid|mobile\s*recharge|dth|airtel|\bjio\b|"
        r"\bbill\b|insurance|premium|\blic\b", re.I)),
    ("Cash", re.compile(r"\batm\b|cash\s*wdl|\bcash\b", re.I)),
    ("Groceries", re.compile(r"blinkit|bigbasket|zepto|dmart|grofers|jiomart|kirana|grocer|"
                             r"reliance\s*fresh|more\s*supermarket|supermarket|instamart", re.I)),
    ("Food delivery", re.compile(r"swiggy|zomato|eatsure|dominos|domino's", re.I)),
    ("Eating out", re.compile(r"cafe|coffee|chai|restaurant|brewpub|\bpub\b|\bbar\b|bistro|"
                              r"dhaba|foods|kitchen|biryani|pizza|bakery|truffles|starbucks|"
                              r"mcdonald|kfc|burger|haldiram", re.I)),
    ("Transport", re.compile(r"uber|\bola\b|rapido|metro|irctc|railway|redbus|petrol|fuel|"
                             r"\bhpcl\b|\bbpcl\b|indian\s*oil|fastag|parking|namma", re.I)),
    ("Shopping", re.compile(r"amazon|flipkart|myntra|ajio|nykaa|meesho|croma|decathlon|ikea|"
                            r"tata\s*cliq|lifestyle|shoppers\s*stop|\bmall\b", re.I)),
    ("Entertainment", re.compile(r"bookmyshow|\bpvr\b|inox|cinema|movie|steam|playstation", re.I)),
    ("Health", re.compile(r"pharma|medplus|apollo|1mg|netmeds|hospital|clinic|diagnostic|"
                          r"pathlab|doctor|dental", re.I)),
]

DISCRETIONARY = ("Food delivery", "Eating out", "Shopping", "Entertainment")
FIXED_GROUPS = ("Rent", "EMI / loan", "Bills and subscriptions")
GROUP_CHOICES = (
    "Rent", "EMI / loan", "Bills and subscriptions", "Groceries", "Food delivery",
    "Eating out", "Transport", "Shopping", "Entertainment", "Health", "Cash",
    "Own-account transfer", "Other",
)


def spend_group(description: str) -> str:
    d = str(description)
    for name, pat in _GROUP_RULES:
        if pat.search(d):
            return name
    return "Other"


def mark_self_transfers(df: pd.DataFrame) -> pd.Series:
    """True where a row is a transfer between the user's own accounts.

    Two signals: the description says so (SELF, OWN A/C, ...), or a debit
    and a credit of the same amount on the same or next day that both look
    like bank transfers (money out and straight back in).
    """
    is_self = df["description"].astype(str).str.contains(_SELF_RE)
    transferish = df["description"].astype(str).str.contains(
        r"\bimps\b|\bneft\b|\brtgs\b|\bupi\b|transfer|\bift\b", case=False, regex=True)
    deb = df[(df["debit"] > 0) & transferish & ~is_self]
    cred = df[(df["credit"] > 0) & transferish & ~is_self]
    used: set = set()
    for i, r in deb.iterrows():
        match = cred[(cred["credit"].sub(r["debit"]).abs() < 0.5)
                     & ((cred["date"] - r["date"]).abs() <= pd.Timedelta(days=1))
                     & ~cred.index.isin(list(used))]
        if not match.empty:
            j = match.index[0]
            used.add(j)
            is_self.loc[i] = True
            is_self.loc[j] = True
    return is_self.astype(bool)


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

_SALARY_RE = re.compile(r"\bsalary\b|\bsal\b|\bpayroll\b|\bsal[\s.-]*cr\b|\bsalaries\b", re.I)


@dataclass
class Salary:
    amount: float
    last_date: date
    count: int
    method: str                    # "keyword" or "pattern"


def _merchant_key(desc: str) -> str:
    s = re.sub(r"[^a-z\s]", " ", str(desc).lower())
    s = re.sub(r"\b(upi|neft|imps|rtgs|ach|dr|cr|ref|no|txn|payment|to|from|by|the|pvt|ltd|"
               r"private|limited|india|order|pos|via|bank)\b", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return " ".join(toks[:3])


def detect_salary(txns: pd.DataFrame) -> Optional[Salary]:
    credits = txns[(txns["credit"] > 0) & ~txns["is_self"]]
    if credits.empty:
        return None
    kw = credits[credits["description"].str.contains(_SALARY_RE) & (credits["credit"] >= 5000)]
    if not kw.empty:
        return Salary(float(kw["credit"].median()), _to_date(kw["date"].max()), len(kw), "keyword")

    # Pattern: the same payer, a large credit in 2+ months, steady amount.
    big = credits[credits["credit"] >= 10000].copy()
    if big.empty:
        return None
    big["key"] = big["description"].map(_merchant_key)
    big["month"] = big["date"].dt.to_period("M")
    best = None
    for key, g in big.groupby("key"):
        if not key or g["month"].nunique() < 2:
            continue
        amt = g["credit"]
        if amt.std(ddof=0) / amt.mean() > 0.15:
            continue
        if best is None or amt.median() > best["credit"].median():
            best = g
    if best is None:
        return None
    return Salary(float(best["credit"].median()), _to_date(best["date"].max()), len(best), "pattern")


def next_payday(last_salary: date, today: date) -> date:
    """Last salary's day of the month, one month later; rolled forward past today."""
    d = add_months(last_salary, 1, last_salary.day)
    k = 1
    while d <= today and k < 36:
        k += 1
        d = add_months(last_salary, k, last_salary.day)
    return d


# ---------------------------------------------------------------------------
# Recurring payments (bills) — replaces the grouping in recurring_detector
# ---------------------------------------------------------------------------


@dataclass
class Recurring:
    name: str
    group: str
    amount: float
    day: int
    last_paid: date
    months: int
    key: str


def detect_recurring(txns: pd.DataFrame, as_of: Optional[date] = None) -> list[Recurring]:
    """Regular monthly debits: rent, EMIs, bills, subscriptions, own-account transfers.

    A merchant counts when it appears in 2+ different months, about once a
    month, on a similar day, with a steady amount (utility bills may vary).
    Day-to-day merchants (food, groceries, cabs, shopping, cash) never count.
    Payments that stopped (nothing in the last ~45 days) are left out.
    """
    deb = txns[txns["debit"] > 0].copy()
    if deb.empty:
        return []
    as_of = as_of or _to_date(deb["date"].max())
    deb["key"] = deb["description"].map(_merchant_key)
    deb["month"] = deb["date"].dt.to_period("M")
    out: list[Recurring] = []
    for key, g in deb.groupby("key"):
        if not key:
            continue
        group = g["group"].mode().iloc[0]
        if group not in FIXED_GROUPS + ("Own-account transfer", "Other"):
            continue
        months = g["month"].nunique()
        if months < 2 or len(g) > months + 1:
            continue
        days = g["date"].dt.day
        if days.std(ddof=0) > 4:
            continue
        amt = g["debit"]
        cv = amt.std(ddof=0) / amt.mean() if amt.mean() else 1
        if cv > (0.35 if group == "Bills and subscriptions" else 0.15):
            continue
        last = _to_date(g["date"].max())
        if (as_of - last).days > 45:
            continue
        recent = amt.iloc[-3:]
        out.append(Recurring(
            name=_display_name(g["description"].iloc[-1]),
            group=group,
            amount=float(round(recent.median())),
            day=int(round(days.median())),
            last_paid=last,
            months=int(months),
            key=key,
        ))
    return sorted(out, key=lambda r: r.day)


def _display_name(desc: str) -> str:
    s = re.sub(r"\ba/c\b", "A/C", str(desc), flags=re.I)
    toks = re.split(r"[\s-]+|(?<!A)/(?!C)", s)
    toks = [t for t in toks if t and "@" not in t and not re.fullmatch(r"\d{6,}", t)
            and t.lower() not in {"upi", "ach", "dr", "cr", "imps", "neft", "nach"}]
    s = " ".join(toks)
    return (s.title().replace("A/C", "a/c") if s else "Regular payment")[:40]


@dataclass
class BillDue:
    name: str
    group: str
    amount: float
    due: date
    status: str        # "due", "overdue", "payday"


def bills_before_payday(items: list[Recurring], balance_date: date, payday: date) -> list[BillDue]:
    """Next expected charge for each recurring item, sorted into:

    - "due":     after the balance date and before payday
    - "overdue": expected on or before the balance date but not seen yet
    - "payday":  expected on payday itself (shown separately, off by default)
    Items next due after payday are covered by the next salary and dropped.
    """
    out: list[BillDue] = []
    for r in items:
        nxt = add_months(r.last_paid, 1, r.day)
        if nxt <= r.last_paid + timedelta(days=15):     # day drifted; skip a month
            nxt = add_months(r.last_paid, 2, r.day)
        if nxt <= balance_date:
            status = "overdue" if (balance_date - nxt).days <= 31 else None
        elif nxt < payday:
            status = "due"
        elif nxt == payday:
            status = "payday"
        else:
            status = None
        if status:
            out.append(BillDue(r.name, r.group, r.amount, nxt, status))
    return sorted(out, key=lambda b: b.due)


def tag_bill_rows(txns: pd.DataFrame, items: list[Recurring]) -> pd.Series:
    keys = {r.key for r in items}
    return (txns["debit"] > 0) & txns["description"].map(_merchant_key).isin(keys)


# ---------------------------------------------------------------------------
# Usual spending and the one change
# ---------------------------------------------------------------------------


def _everyday(txns: pd.DataFrame, bill_rows: pd.Series) -> pd.DataFrame:
    mask = ((txns["debit"] > 0) & ~txns["is_self"] & ~bill_rows
            & ~txns["group"].isin(FIXED_GROUPS))
    return txns[mask]


def weeks_available(txns: pd.DataFrame, as_of: date) -> int:
    start = _to_date(txns["date"].min())
    return int(((as_of - start).days + 1) // 7)


def weekly_totals(txns: pd.DataFrame, bill_rows: pd.Series, as_of: date) -> list[float]:
    """Everyday spend in each of the last (up to 8) full 7-day windows ending at as_of, oldest first."""
    weeks = min(MAX_WEEKS_AVERAGED, weeks_available(txns, as_of))
    e = _everyday(txns, bill_rows)
    out = []
    for k in range(weeks - 1, -1, -1):
        hi = pd.Timestamp(as_of - timedelta(days=7 * k))
        lo = hi - pd.Timedelta(days=6)
        out.append(float(e[(e["date"] >= lo) & (e["date"] <= hi)]["debit"].sum()))
    return out


def usual_weekly_spend(txns: pd.DataFrame, bill_rows: pd.Series, as_of: date) -> Optional[tuple[float, int]]:
    """Median everyday spend per week over up to 8 full weeks ending at as_of.

    The median, not the mean, so a one-off (a medical bill, a flight home)
    in one week doesn't inflate what counts as "usual".
    """
    totals = weekly_totals(txns, bill_rows, as_of)
    if not totals:
        return None
    return float(np.median(totals)), len(totals)


@dataclass
class Suggestion:
    group: str
    weekly_avg: float
    weeks: int
    cut_to: float
    frees: float


def one_change(txns: pd.DataFrame, bill_rows: pd.Series, as_of: date) -> Optional[Suggestion]:
    """Largest discretionary group, averaged over up to 8 weeks; 25% less as an option.

    Hidden (None) with fewer than 4 full weeks of data.
    """
    weeks = min(MAX_WEEKS_AVERAGED, weeks_available(txns, as_of))
    if weeks < MIN_WEEKS_FOR_TIP:
        return None
    lo = pd.Timestamp(as_of - timedelta(days=7 * weeks - 1))
    e = _everyday(txns, bill_rows)
    e = e[(e["date"] >= lo) & (e["date"] <= pd.Timestamp(as_of)) & e["group"].isin(DISCRETIONARY)]
    if e.empty:
        return None
    per_week = e.groupby("group")["debit"].sum() / weeks
    g = per_week.idxmax()
    avg = float(per_week[g])
    if avg < 100:
        return None
    frees = math.floor(avg * TIP_CUT)
    return Suggestion(g, avg, weeks, avg - frees, frees)


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


@dataclass
class Answer:
    balance: float
    bills: float
    buffer: float
    available: float           # balance - bills - buffer
    days: int                  # days to payday
    horizon: int               # min(7, days)
    daily: float               # available / days (0 if short)
    weekly: float              # available * horizon / days, floored to the rupee (0 if short)
    shortfall: Optional[float]

    @property
    def daily_shown(self) -> int:
        """Daily amount as displayed: rounded down, like the check-in screen."""
        return math.floor(self.daily)


def safe_to_spend(balance: float, bills: float, buffer: float, today: date, payday: date) -> Answer:
    """weekly = (balance - bills - buffer) / days * min(7, days).

    Raises ValueError if payday is not after today. Never returns a negative
    weekly figure: a negative remainder is reported as a shortfall.
    """
    for name, v in (("balance", balance), ("bills", bills), ("buffer", buffer)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError(f"{name} is missing")
    if bills < 0 or buffer < 0:
        raise ValueError("bills and buffer cannot be negative")
    days = (payday - today).days
    if days <= 0:
        raise ValueError("Payday must be after today.")
    available = float(balance) - float(bills) - float(buffer)
    horizon = min(7, days)
    if available <= 0:
        return Answer(balance, bills, buffer, available, days, horizon, 0.0, 0.0,
                      shortfall=-available if available < 0 else 0.0)
    daily = available / days
    weekly = math.floor(available * horizon / days)
    return Answer(balance, bills, buffer, available, days, horizon, daily, float(weekly), None)


@dataclass
class Comparison:
    state: str                      # "short", "over" (usual above limit), "fits", "unknown"
    usual_period: Optional[float]   # usual weekly spend scaled to the limit's days
    gap: float = 0.0                # how much less to spend in the period
    covered: float = 0.0            # part of the gap the suggested change covers
    remaining: float = 0.0          # gap left after the suggestion


def compare_to_usual(answer: Answer, usual_weekly: Optional[float],
                     suggestion: Optional["Suggestion"] = None) -> Comparison:
    """Compare the safe limit with usual spending over the same number of days."""
    if answer.shortfall is not None:
        return Comparison("short", None)
    if usual_weekly is None:
        return Comparison("unknown", None)
    usual_period = usual_weekly * answer.horizon / 7
    if usual_period <= answer.weekly:
        return Comparison("fits", usual_period)
    gap = math.ceil(usual_period - answer.weekly)
    covered = 0.0
    if suggestion is not None:
        covered = float(min(gap, math.floor(suggestion.frees * answer.horizon / 7)))
    return Comparison("over", usual_period, float(gap), covered, float(gap - covered))


@dataclass
class CheckIn:
    spent: float
    remaining_week: float          # weekly limit - spent (can be negative)
    over_by: Optional[float]
    days_left: int
    daily_until_payday: float      # recalculated; 0 if short
    shortfall: Optional[float]


def check_in(answer: Answer, spent: float, check_date: date, today: date, payday: date) -> CheckIn:
    """Spend entered by the user excludes the listed bills (those are already reserved)."""
    if spent < 0:
        raise ValueError("Spend cannot be negative.")
    if check_date < today:
        raise ValueError("The check-in date can't be before the start date.")
    remaining = answer.weekly - spent
    over = -remaining if remaining < 0 else None
    days_left = (payday - check_date).days
    left = answer.available - spent
    if days_left <= 0:
        return CheckIn(spent, remaining, over, 0, 0.0, -left if left < 0 else None)
    if left <= 0:
        return CheckIn(spent, remaining, over, days_left, 0.0, -left if left < 0 else 0.0)
    return CheckIn(spent, remaining, over, days_left, math.floor(left / days_left), None)
