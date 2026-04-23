"""
4-layer rule-based transaction categorizer for Indian financial transactions.

Layer 0  — Person-name detection  (→ Personal Transfer / Peer Transfer)
Layer 1  — Merchant map lookup     (src/merchant_categories.py, 200+ entries)
Layer 2  — Generic keyword regex   (word-boundary, not substring)
Layer 3  — Amount heuristics       (tiebreaker only)
Layer 4  — UPI description parsing (extracts merchant and re-enters pipeline)
Global   — Refund / reversal / cashback override (highest priority)

Output columns added to DataFrame:
  category          — top-level category string
  subcategory       — granular subcategory
  confidence        — float 0.0–1.0
  match_confidence  — string enum (layer that matched)
  match_notes       — internal diagnostic notes (not shown in UI)
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import pandas as pd

from src.merchant_categories import BUSINESS_INDICATORS, get_category as _merchant_lookup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = [
    "Food and Dining",
    "Transportation",
    "Utilities and Bills",
    "Entertainment and Subscriptions",
    "Shopping and Retail",
    "Healthcare and Medical",
    "Investment and Savings",
    "Loan and EMI",
    "Transfer and Banking",
    "Income and Salary",
    "Personal Transfer",
    "Education",
    "Uncategorized",
]


# ---------------------------------------------------------------------------
# Global override — refund / reversal / cashback
# ---------------------------------------------------------------------------
_REFUND_PATTERN: re.Pattern = re.compile(
    r"\b(refund|reversal|reversed|cashback|cash[\s\-]?back|reward|revert)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 0 — Person-name detection
# ---------------------------------------------------------------------------
_HONORIFICS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "shri", "smt", "sri", "miss",
})


def _is_person_name(description: str) -> tuple[bool, str]:
    """Return (is_person, condition_code) for the description.

    condition_code values
    ---------------------
    "1"  — 2-4 title-cased words, all letters, no business indicators
    "2"  — UPI ID format (name@provider) with no business token in name part
    "3"  — starts with an honorific (Mr / Dr / Shri …) followed by 1+ words
    ""   — not a person name
    """
    if not description or len(description) > 80:
        return False, ""

    cleaned = description.strip()
    words = cleaned.split()
    if not words:
        return False, ""

    # Condition 3 — honorific prefix
    first = words[0].rstrip(".").lower()
    if first in _HONORIFICS and len(words) >= 2:
        rest_lower = [w.lower().rstrip(".,") for w in words[1:]]
        if not any(w in BUSINESS_INDICATORS for w in rest_lower):
            return True, "3"

    # Condition 2 — UPI ID pattern  (name@provider)
    if "@" in cleaned:
        name_part = cleaned.split("@")[0].strip()
        name_words = re.split(r"[\s_\-]+", name_part)
        if 2 <= len(name_words) <= 4:
            name_lower = [w.lower() for w in name_words]
            if (
                not any(w in BUSINESS_INDICATORS for w in name_lower)
                and all(re.match(r"^[a-zA-Z]+$", w) for w in name_words)
            ):
                return True, "2"

    # Condition 1 — 2-4 all-letter words, no business indicators
    if 2 <= len(words) <= 4:
        words_lower = [w.lower().rstrip(".,") for w in words]
        if (
            all(re.match(r"^[a-zA-Z]+$", w) for w in words)
            and not any(w in BUSINESS_INDICATORS for w in words_lower)
            and all(2 <= len(w) <= 15 for w in words)
        ):
            return True, "1"

    return False, ""


# ---------------------------------------------------------------------------
# Layer 2 — Generic keyword list (word-boundary matching)
# ---------------------------------------------------------------------------
# Each entry: (phrase, category, subcategory)
# Sorted longest-first so multi-word phrases take priority.
_L2_ENTRIES: list[tuple[str, str, str]] = [
    # ── Food and Dining ──────────────────────────────────────────────────
    ("food delivery",       "Food and Dining",                "Food Delivery"),
    ("food order",          "Food and Dining",                "Food Delivery"),
    ("food court",          "Food and Dining",                "Restaurant"),
    ("restaurant",          "Food and Dining",                "Restaurant"),
    ("biryani",             "Food and Dining",                "Restaurant"),
    ("dhaba",               "Food and Dining",                "Restaurant"),
    ("bakery",              "Food and Dining",                "Restaurant"),
    ("canteen",             "Food and Dining",                "Restaurant"),
    ("tiffin",              "Food and Dining",                "Restaurant"),
    ("juice",               "Food and Dining",                "Restaurant"),
    # ── Shopping and Retail ──────────────────────────────────────────────
    ("groceries",           "Shopping and Retail",            "Groceries"),
    ("supermarket",         "Shopping and Retail",            "Groceries"),
    ("vegetables",          "Shopping and Retail",            "Groceries"),
    ("provisions",          "Shopping and Retail",            "Groceries"),
    ("online shopping",     "Shopping and Retail",            "General Online"),
    ("shopping",            "Shopping and Retail",            "General Online"),
    # ── Transportation ───────────────────────────────────────────────────
    ("atm withdrawal",      "Transfer and Banking",           "ATM Withdrawal"),
    ("cash withdrawal",     "Transfer and Banking",           "ATM Withdrawal"),
    ("airlines",            "Transportation",                 "Flights"),
    ("airline",             "Transportation",                 "Flights"),
    ("airways",             "Transportation",                 "Flights"),
    ("flight",              "Transportation",                 "Flights"),
    ("fastag",              "Transportation",                 "Toll"),
    ("toll",                "Transportation",                 "Toll"),
    ("petrol",              "Transportation",                 "Fuel"),
    ("diesel",              "Transportation",                 "Fuel"),
    ("fuel",                "Transportation",                 "Fuel"),
    ("hotel",               "Transportation",                 "Hotels"),
    ("hostel",              "Transportation",                 "Hotels"),
    ("resort",              "Transportation",                 "Hotels"),
    ("taxi",                "Transportation",                 "Cab & Ride Share"),
    ("cab",                 "Transportation",                 "Cab & Ride Share"),
    # ── Utilities and Bills ──────────────────────────────────────────────
    ("electricity bill",    "Utilities and Bills",            "Electricity"),
    ("power bill",          "Utilities and Bills",            "Electricity"),
    ("wifi bill",           "Utilities and Bills",            "Mobile & Internet"),
    ("internet bill",       "Utilities and Bills",            "Mobile & Internet"),
    ("mobile bill",         "Utilities and Bills",            "Mobile & Internet"),
    ("dth recharge",        "Utilities and Bills",            "Other Bills"),
    ("recharge",            "Utilities and Bills",            "Mobile & Internet"),
    ("water bill",          "Utilities and Bills",            "Water"),
    ("gas bill",            "Utilities and Bills",            "Gas"),
    ("lpg",                 "Utilities and Bills",            "Gas"),
    # ── Healthcare ───────────────────────────────────────────────────────
    ("pharmacy",            "Healthcare and Medical",         "Pharmacy"),
    ("medicine",            "Healthcare and Medical",         "Pharmacy"),
    ("hospital",            "Healthcare and Medical",         "Hospital & Clinic"),
    ("clinic",              "Healthcare and Medical",         "Hospital & Clinic"),
    ("medical",             "Healthcare and Medical",         "Hospital & Clinic"),
    ("doctor",              "Healthcare and Medical",         "Hospital & Clinic"),
    ("dentist",             "Healthcare and Medical",         "Hospital & Clinic"),
    ("lab test",            "Healthcare and Medical",         "Diagnostics"),
    ("yoga",                "Healthcare and Medical",         "Fitness"),
    ("gym",                 "Healthcare and Medical",         "Fitness"),
    # ── Investment and Savings ───────────────────────────────────────────
    ("mutual fund sip",     "Investment and Savings",         "Mutual Funds"),
    ("mutual fund",         "Investment and Savings",         "Mutual Funds"),
    ("insurance premium",   "Investment and Savings",         "Insurance"),
    ("insurance",           "Investment and Savings",         "Insurance"),
    ("investment",          "Investment and Savings",         "Stocks & Equity"),
    ("demat",               "Investment and Savings",         "Stocks & Equity"),
    ("fixed deposit",       "Investment and Savings",         "Fixed Income"),
    # ── Loan and EMI ─────────────────────────────────────────────────────
    ("credit card bill",    "Loan and EMI",                   "Credit Card Bill"),
    ("emi payment",         "Loan and EMI",                   "Personal Loan"),
    ("loan repayment",      "Loan and EMI",                   "Personal Loan"),
    ("home loan",           "Loan and EMI",                   "Home Loan"),
    ("car loan",            "Loan and EMI",                   "Car Loan"),
    ("loan",                "Loan and EMI",                   "Personal Loan"),
    # ── Transfer and Banking ─────────────────────────────────────────────
    ("house rent",          "Transfer and Banking",           "Rent"),
    ("maintenance",         "Transfer and Banking",           "Rent"),
    ("rent",                "Transfer and Banking",           "Rent"),
    ("bank transfer",       "Transfer and Banking",           "Bank Transfer"),
    ("fund transfer",       "Transfer and Banking",           "Bank Transfer"),
    ("neft",                "Transfer and Banking",           "Bank Transfer"),
    ("rtgs",                "Transfer and Banking",           "Bank Transfer"),
    ("imps",                "Transfer and Banking",           "Bank Transfer"),
    ("transfer",            "Transfer and Banking",           "Bank Transfer"),
    ("upi",                 "Transfer and Banking",           "UPI Transfer"),
    # ── Income ───────────────────────────────────────────────────────────
    ("salary credit",       "Income and Salary",              "Salary"),
    ("salary",              "Income and Salary",              "Salary"),
    ("payroll",             "Income and Salary",              "Salary"),
    ("dividend",            "Income and Salary",              "Investment Returns"),
    ("interest credit",     "Income and Salary",              "Investment Returns"),
    # ── Education ────────────────────────────────────────────────────────
    ("school fee",          "Education",                      "School & College"),
    ("college fee",         "Education",                      "School & College"),
    ("university fee",      "Education",                      "School & College"),
    ("tuition",             "Education",                      "School & College"),
    ("course fee",          "Education",                      "Online Learning"),
    # ── Entertainment ────────────────────────────────────────────────────
    ("subscription",        "Entertainment and Subscriptions","Subscription"),
    ("gaming",              "Entertainment and Subscriptions","Gaming"),
    ("concert",             "Entertainment and Subscriptions","Movies & Events"),
    ("movie",               "Entertainment and Subscriptions","Movies & Events"),
]

_L2_SORTED = sorted(_L2_ENTRIES, key=lambda x: len(x[0]), reverse=True)
_L2_COMPILED: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), cat, sub)
    for phrase, cat, sub in _L2_SORTED
]


# ---------------------------------------------------------------------------
# Layer 4 — UPI description parser
# ---------------------------------------------------------------------------
_UPI_EXTRACT: re.Pattern = re.compile(
    r"UPI[-/](?:[A-Z0-9]+[-/])?([^@/\-]{3,30})(?:[@/\-]|$)",
    re.IGNORECASE,
)


def _extract_upi_merchant(description: str) -> Optional[str]:
    """Extract probable merchant / sender name from a UPI reference string."""
    m = _UPI_EXTRACT.search(description)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) >= 3 and not candidate.isdigit():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Core single-row classification
# ---------------------------------------------------------------------------
def _classify_one(
    description: str,
    amount: Optional[float] = None,
    transaction_type: Optional[str] = None,
) -> dict:
    """Classify one transaction through all layers.

    Returns
    -------
    dict with keys: category, subcategory, confidence, match_confidence,
    match_notes
    """
    # ── Credit transactions ────────────────────────────────────────────────
    if transaction_type == "credit":
        if _REFUND_PATTERN.search(description or ""):
            return {
                "category": "Income and Salary",
                "subcategory": "Refund",
                "confidence": 1.0,
                "match_confidence": "global_override",
                "match_notes": "refund_reversal_cashback",
            }
        return {
            "category": "Income and Salary",
            "subcategory": "Salary",
            "confidence": 1.0,
            "match_confidence": "credit_type",
            "match_notes": "",
        }

    if not description or not description.strip():
        return {
            "category": "Uncategorized",
            "subcategory": "Unmatched",
            "confidence": 0.0,
            "match_confidence": "unmatched",
            "match_notes": "",
        }

    desc = description.strip()

    # ── Global override: refund / reversal / cashback ──────────────────────
    if _REFUND_PATTERN.search(desc):
        return {
            "category": "Income and Salary",
            "subcategory": "Refund",
            "confidence": 1.0,
            "match_confidence": "global_override",
            "match_notes": "refund_reversal_cashback",
        }

    # ── Layer 4 pre-processing: extract UPI merchant name ──────────────────
    upi_merchant: Optional[str] = _extract_upi_merchant(desc)

    # ── Layer 1a: Merchant map on full description ─────────────────────────
    result = _merchant_lookup(desc)
    if result is not None:
        cat, sub = result
        notes = _layer1_notes(cat, sub)
        return {
            "category": cat,
            "subcategory": sub,
            "confidence": 1.0,
            "match_confidence": "exact_merchant",
            "match_notes": notes,
        }

    # ── Layer 1b: Merchant map on UPI-extracted name ───────────────────────
    if upi_merchant:
        result = _merchant_lookup(upi_merchant)
        if result is not None:
            cat, sub = result
            notes = _layer1_notes(cat, sub)
            return {
                "category": cat,
                "subcategory": sub,
                "confidence": 0.95,
                "match_confidence": "upi_parsed",
                "match_notes": f"upi_extracted;{notes}",
            }

    # ── Layer 0a: Person-name on full description ──────────────────────────
    is_person, condition = _is_person_name(desc)
    if is_person:
        notes = "personal_transfer"
        if amount and amount <= 2000:
            notes += ",possible_split"
        return {
            "category": "Personal Transfer",
            "subcategory": "Peer Transfer",
            "confidence": 0.85,
            "match_confidence": "person_name_detected",
            "match_notes": f"{notes};condition={condition}",
        }

    # ── Layer 0b: Person-name on UPI-extracted name ────────────────────────
    if upi_merchant:
        is_person2, condition2 = _is_person_name(upi_merchant)
        if is_person2:
            return {
                "category": "Personal Transfer",
                "subcategory": "Peer Transfer",
                "confidence": 0.80,
                "match_confidence": "person_name_detected",
                "match_notes": f"personal_transfer;upi_name;condition={condition2}",
            }

    # ── Layer 2a: Keyword regex on full description ────────────────────────
    for pattern, cat, sub in _L2_COMPILED:
        if pattern.search(desc):
            notes = "possible_rent" if sub == "Rent" else ""
            return {
                "category": cat,
                "subcategory": sub,
                "confidence": 0.8,
                "match_confidence": "keyword",
                "match_notes": notes,
            }

    # ── Layer 2b: Keyword regex on UPI-extracted name ─────────────────────
    if upi_merchant:
        for pattern, cat, sub in _L2_COMPILED:
            if pattern.search(upi_merchant):
                return {
                    "category": cat,
                    "subcategory": sub,
                    "confidence": 0.70,
                    "match_confidence": "upi_parsed",
                    "match_notes": "upi_keyword",
                }

    # ── Layer 3: Amount heuristics (last resort) ──────────────────────────
    if amount is not None:
        if amount >= 50_000:
            return {
                "category": "Investment and Savings",
                "subcategory": "Large Transfer",
                "confidence": 0.40,
                "match_confidence": "heuristic",
                "match_notes": "large_amount",
            }
        if amount >= 10_000:
            return {
                "category": "Transfer and Banking",
                "subcategory": "Bank Transfer",
                "confidence": 0.35,
                "match_confidence": "heuristic",
                "match_notes": "medium_amount",
            }

    # ── Fallback ──────────────────────────────────────────────────────────
    return {
        "category": "Uncategorized",
        "subcategory": "Unmatched",
        "confidence": 0.0,
        "match_confidence": "unmatched",
        "match_notes": "",
    }


def _layer1_notes(cat: str, sub: str) -> str:
    if sub == "Rent":
        return "possible_rent"
    if cat == "Loan and EMI":
        return "possible_loan"
    return ""


# ---------------------------------------------------------------------------
# Public classifier class
# ---------------------------------------------------------------------------
class TransactionClassifier:
    """4-layer rule-based transaction classifier for Indian financial data.

    The ``use_zero_shot`` parameter is accepted for API compatibility but
    the classification is now fully rule-based (no ML model required).
    """

    def __init__(self, use_zero_shot: bool = True) -> None:
        # use_zero_shot kept for backward-compatible API; not used internally
        self.use_zero_shot = use_zero_shot

    # ------------------------------------------------------------------
    # Internal classification (zero-shot or rule-based)
    # Returns a dict with keys: category, subcategory, confidence,
    # match_confidence, match_notes — same shape as the module-level
    # _classify_one() so classify_dataframe/classify_batch can use it.
    # ------------------------------------------------------------------
    def _get_zs_pipeline(self):
        if not hasattr(self, "_zs_pipeline"):
            from transformers import pipeline as _pipeline
            self._zs_pipeline = _pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                model_kwargs={"low_cpu_mem_usage": False},
            )
        return self._zs_pipeline

    def _classify_one(
        self,
        description: str,
        amount: Optional[float] = None,
        transaction_type: Optional[str] = None,
    ) -> dict:
        if self.use_zero_shot:
            result = self._get_zs_pipeline()(description, CATEGORIES)
            return {
                "category": result["labels"][0],
                "subcategory": "Zero-shot",
                "confidence": result["scores"][0],
                "match_confidence": "zero_shot",
                "match_notes": "",
            }
        else:
            return _classify_one(description, amount=amount, transaction_type=transaction_type)

    def _classify_batch_zs(self, descriptions: list[str]) -> list[dict]:
        """Run zero-shot classification on all descriptions in one batched call."""
        pipe = self._get_zs_pipeline()
        results = pipe(descriptions, CATEGORIES, batch_size=32)
        return [
            {
                "category": r["labels"][0],
                "subcategory": "Zero-shot",
                "confidence": r["scores"][0],
                "match_confidence": "zero_shot",
                "match_notes": "",
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_single(self, description: str) -> tuple[str, float]:
        """Classify one transaction description.

        Returns
        -------
        tuple[str, float]
            ``(category, confidence)``
        """
        result = self._classify_one(description)
        return result["category"], result["confidence"]

    def classify_batch(
        self,
        descriptions: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Classify a list of descriptions.

        Parameters
        ----------
        descriptions:
            List of transaction description strings.
        progress_callback:
            Optional ``(processed, total) -> None`` progress reporter.

        Returns
        -------
        pd.DataFrame
            Columns: description, category, subcategory, confidence,
            match_confidence, match_notes
        """
        results = []
        total = len(descriptions)
        for i, desc in enumerate(descriptions):
            results.append(self._classify_one(desc))
            if progress_callback is not None and (i % 50 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        df = pd.DataFrame(results)
        df.insert(0, "description", descriptions)
        return df

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add classification columns to a transactions DataFrame.

        Input must have a ``description`` column (case-insensitive).
        If ``transaction_type`` and/or ``amount`` columns are present they
        are used to improve accuracy.

        Columns added / updated
        -----------------------
        category, subcategory, confidence, match_confidence, match_notes

        Returns
        -------
        pd.DataFrame — copy of *df* with new columns appended.
        """
        df = df.copy()

        # Locate description column (case-insensitive)
        desc_col: Optional[str] = next(
            (c for c in df.columns if c.lower() == "description"), None
        )
        if desc_col is None:
            raise ValueError(
                "DataFrame must contain a 'description' column. "
                f"Found: {list(df.columns)}"
            )

        # Locate optional helper columns
        type_col: Optional[str] = next(
            (c for c in df.columns if c.lower() == "transaction_type"), None
        )
        amt_col: Optional[str] = next(
            (c for c in df.columns if c.lower() in ("amount", "debit")), None
        )

        if self.use_zero_shot:
            descriptions = [
                str(row[desc_col]) if row[desc_col] is not None else ""
                for _, row in df.iterrows()
            ]
            rows = self._classify_batch_zs(descriptions)
        else:
            rows = []
            for _, row in df.iterrows():
                desc = str(row[desc_col]) if row[desc_col] is not None else ""
                txn_type = str(row[type_col]).lower() if type_col else None
                amt = float(row[amt_col]) if amt_col and pd.notna(row[amt_col]) else None
                rows.append(self._classify_one(desc, amount=amt, transaction_type=txn_type))

        result_df = pd.DataFrame(rows)
        df["category"] = result_df["category"].values
        df["subcategory"] = result_df["subcategory"].values
        df["confidence"] = result_df["confidence"].values
        df["match_confidence"] = result_df["match_confidence"].values
        df["match_notes"] = result_df["match_notes"].values

        self._print_validation_report(df, desc_col, type_col, amt_col)

        logger.info(
            "Classified %d transactions. Distribution: %s",
            len(df),
            df["category"].value_counts().to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # Validation reports (console only — never shown in UI)
    # ------------------------------------------------------------------
    def _print_validation_report(
        self,
        df: pd.DataFrame,
        desc_col: str,
        type_col: Optional[str],
        amt_col: Optional[str],
    ) -> None:
        sep = "=" * 60

        # ── Report 1: Category distribution by subcategory ────────────
        print(f"\n{sep}")
        print("ARTHASENSE CLASSIFICATION REPORT")
        print(sep)
        print("\n[1] CATEGORY DISTRIBUTION (subcategory)")
        dist = (
            df.groupby(["category", "subcategory"])
            .size()
            .reset_index(name="count")
            .sort_values(["category", "count"], ascending=[True, False])
        )
        for _, row2 in dist.iterrows():
            print(f"    {row2['category']:<35} {row2['subcategory']:<25} {row2['count']:>4}")

        # ── Report 2: Personal Transfer transactions ──────────────────
        print(f"\n[2] PERSONAL TRANSFER TRANSACTIONS")
        personal = df[df["category"] == "Personal Transfer"]
        if personal.empty:
            print("    — none detected —")
        else:
            for _, row2 in personal.iterrows():
                amt_str = ""
                if amt_col and pd.notna(row2.get(amt_col)):
                    amt_str = f"  ₹{row2[amt_col]:,.2f}"
                print(f"    {str(row2[desc_col]):<50}{amt_str}")

        # ── Report 3: Layer 0 condition breakdown ─────────────────────
        print(f"\n[3] LAYER 0 PERSON-NAME DETECTION — CONDITION BREAKDOWN")
        layer0 = df[df["match_confidence"] == "person_name_detected"]
        cond_counts = {"1": 0, "2": 0, "3": 0, "other": 0}
        for notes in layer0["match_notes"]:
            if "condition=1" in str(notes):
                cond_counts["1"] += 1
            elif "condition=2" in str(notes):
                cond_counts["2"] += 1
            elif "condition=3" in str(notes):
                cond_counts["3"] += 1
            else:
                cond_counts["other"] += 1
        print(f"    Condition 1 (2-4 title words, no biz indicator): {cond_counts['1']}")
        print(f"    Condition 2 (UPI name@provider format):          {cond_counts['2']}")
        print(f"    Condition 3 (honorific prefix):                  {cond_counts['3']}")
        total_l0 = layer0.shape[0]
        print(f"    Total Layer-0 matches: {total_l0}")

        # ── Report 4: Uncategorized transactions ──────────────────────
        print(f"\n[4] UNCATEGORIZED TRANSACTIONS (add rules manually)")
        uncat = df[df["category"] == "Uncategorized"]
        if uncat.empty:
            print("    — none —")
        else:
            for _, row2 in uncat.iterrows():
                amt_str = ""
                if amt_col and pd.notna(row2.get(amt_col)):
                    amt_str = f"  ₹{row2[amt_col]:,.2f}"
                print(f"    {str(row2[desc_col]):<50}{amt_str}")

        # ── Report 5: Near-miss business names ────────────────────────
        print(f"\n[5] NEAR-MISS: BUSINESS NAMES THAT NEARLY TRIGGERED LAYER 0")
        near_misses = []
        for _, row2 in df.iterrows():
            desc_val = str(row2[desc_col]).strip()
            words = desc_val.split()
            if 2 <= len(words) <= 4:
                words_lower = [w.lower().rstrip(".,") for w in words]
                # Would pass condition 1 structure but blocked by business indicator
                if (
                    all(re.match(r"^[a-zA-Z]+$", w) for w in words)
                    and any(w in BUSINESS_INDICATORS for w in words_lower)
                    and row2["category"] != "Personal Transfer"
                ):
                    near_misses.append(
                        f"    {desc_val:<50}  → {row2['category']}"
                    )
        if near_misses:
            for line in near_misses[:20]:  # Cap at 20 to avoid flooding console
                print(line)
            if len(near_misses) > 20:
                print(f"    … and {len(near_misses) - 20} more")
        else:
            print("    — none —")

        print(f"\n{sep}\n")
