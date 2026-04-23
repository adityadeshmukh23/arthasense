"""
recurring_detector.py — Detect recurring / subscription payments in a transaction DataFrame.

Public API
----------
detect_recurring(df: pd.DataFrame) -> pd.DataFrame
    Returns a summary DataFrame of recurring merchants with columns:
    [merchant, frequency, avg_amount, total_spent, months_active, likely_subscription]
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Tokens to strip when normalising merchant names for grouping
_NOISE = re.compile(
    r"\b(payment|pay|via|upi|ref|no|order|id|to|from|the|pvt|ltd|"
    r"private|limited|india|services?|solutions?|technologies?|tech)\b",
    re.I,
)
# Non-alphanumeric characters (keep spaces for word boundary splitting)
_NONALPHA = re.compile(r"[^a-z0-9\s]", re.I)


def _normalise(name: str) -> str:
    """
    Reduce a merchant description to a canonical form for grouping.
    Strips common suffixes, punctuation, and trailing digits.
    """
    s = str(name).lower().strip()
    s = _NONALPHA.sub(" ", s)
    s = _NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep only first 3 significant tokens to collapse variations
    tokens = [t for t in s.split() if len(t) > 1][:3]
    return " ".join(tokens)


def _is_similar(a: str, b: str, threshold: float = 0.60) -> bool:
    """
    Simple Jaccard similarity on bigrams to detect near-duplicate merchant names.
    """
    def bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return a == b
    return len(ba & bb) / len(ba | bb) >= threshold


# Known subscription keywords for extra flagging
_SUBSCRIPTION_KEYWORDS = re.compile(
    r"(netflix|spotify|hotstar|disney|prime|youtube|apple|google|zee5|sony|"
    r"jio cinema|jiocinema|coursera|udemy|unacademy|byju|vedantu|"
    r"gym|cult\.fit|curefit|linkedin|microsoft|dropbox|icloud|"
    r"subscription|renewal|annual|monthly plan|recharge)",
    re.I,
)


def _flag_subscription(merchant: str) -> bool:
    return bool(_SUBSCRIPTION_KEYWORDS.search(merchant))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify recurring / subscription payments in a transactions DataFrame.

    A transaction is considered recurring if the same merchant appears in
    2 or more distinct calendar months with a reasonably consistent amount
    (standard deviation < 20 % of the mean).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date (datetime), description (str),
        debit (float), transaction_type (str).

    Returns
    -------
    pd.DataFrame with columns:
        merchant           – canonical merchant name
        frequency          – number of occurrences
        avg_amount         – mean transaction amount (₹)
        total_spent        – total amount across all occurrences (₹)
        months_active      – number of distinct months the charge appeared
        likely_subscription – True if amount is consistent & keyword match
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "merchant", "frequency", "avg_amount",
            "total_spent", "months_active", "likely_subscription",
        ])

    # Work only with debit transactions
    debits = df[df["transaction_type"] == "debit"].copy()
    if debits.empty:
        return pd.DataFrame(columns=[
            "merchant", "frequency", "avg_amount",
            "total_spent", "months_active", "likely_subscription",
        ])

    debits["_norm"] = debits["description"].apply(_normalise)
    debits["_month"] = debits["date"].dt.to_period("M")

    # Group by normalised merchant name
    records = []
    for norm, grp in debits.groupby("_norm"):
        if not norm or len(norm) < 2:
            continue

        freq          = len(grp)
        months_active = grp["_month"].nunique()
        amounts       = grp["debit"].dropna()

        if freq < 2 or months_active < 2:
            continue

        avg_amt   = float(amounts.mean())
        total     = float(amounts.sum())
        std_amt   = float(amounts.std(ddof=0)) if len(amounts) > 1 else 0.0

        # Amount consistency: std/mean < 20 % → consistent charge
        consistent = (std_amt / avg_amt < 0.20) if avg_amt > 0 else True

        # Use the most common raw description as the display merchant name
        display_name = grp["description"].mode().iloc[0] if not grp["description"].empty else norm

        likely_sub = consistent and (
            _flag_subscription(display_name) or avg_amt <= 1500
        )

        records.append({
            "merchant":            display_name,
            "frequency":           freq,
            "avg_amount":          round(avg_amt, 2),
            "total_spent":         round(total, 2),
            "months_active":       months_active,
            "likely_subscription": likely_sub,
            "_consistent":         consistent,
        })

    if not records:
        return pd.DataFrame(columns=[
            "merchant", "frequency", "avg_amount",
            "total_spent", "months_active", "likely_subscription",
        ])

    result = (
        pd.DataFrame(records)
        .sort_values(["months_active", "total_spent"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return result.drop(columns=["_consistent"], errors="ignore")
