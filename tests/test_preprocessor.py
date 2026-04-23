"""Tests for src.preprocessor – BankStatementPreprocessor."""

import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessor import BankStatementPreprocessor, _clean_amount_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(content: str) -> str:
    """Write *content* to a temporary CSV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


SAMPLE_CSV = (
    "Date,Description,Debit,Credit,Balance\n"
    "01-01-2025,Swiggy Order,500,,10000\n"
    "02-01-2025,Salary Credit,,50000,60000\n"
    "03-01-2025,Amazon Purchase,1200,,58800\n"
)


# ---------------------------------------------------------------------------
# test_load_csv
# ---------------------------------------------------------------------------

class TestLoadCSV:
    """load_csv should read a CSV file and return a DataFrame."""

    def test_load_from_file_path(self):
        path = _write_csv(SAMPLE_CSV)
        pp = BankStatementPreprocessor()
        df = pp.load_csv(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_from_stringio(self):
        """load_csv accepts file-like objects (Streamlit UploadedFile-style)."""
        pp = BankStatementPreprocessor()
        buf = StringIO(SAMPLE_CSV)
        df = pp.load_csv(buf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_present(self):
        path = _write_csv(SAMPLE_CSV)
        pp = BankStatementPreprocessor()
        df = pp.load_csv(path)
        cols_lower = [c.lower() for c in df.columns]
        assert "date" in cols_lower
        assert "description" in cols_lower


# ---------------------------------------------------------------------------
# test_clean – date formats
# ---------------------------------------------------------------------------

class TestCleanDateFormats:
    """clean() should parse DD-MM-YYYY, DD/MM/YYYY, and YYYY-MM-DD."""

    def _make_csv(self, date_str: str) -> str:
        return (
            "Date,Description,Debit,Credit,Balance\n"
            f"{date_str},Test Txn,100,,9900\n"
        )

    def test_dd_mm_yyyy_dash(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(self._make_csv("15-06-2025")))
        cleaned = pp.clean(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["date"].day == 15
        assert cleaned.iloc[0]["date"].month == 6

    def test_dd_mm_yyyy_slash(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(self._make_csv("15/06/2025")))
        cleaned = pp.clean(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["date"].day == 15
        assert cleaned.iloc[0]["date"].month == 6

    def test_yyyy_mm_dd(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(self._make_csv("2025-06-15")))
        cleaned = pp.clean(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["date"].day == 15
        assert cleaned.iloc[0]["date"].month == 6


# ---------------------------------------------------------------------------
# test_clean – NaN amounts
# ---------------------------------------------------------------------------

class TestCleanNaNAmounts:
    """Blank, '-', and 'NIL' in debit/credit columns should become 0.0."""

    @pytest.mark.parametrize("nan_val", ["", "-", "NIL"])
    def test_nan_debit_becomes_zero(self, nan_val):
        csv_text = (
            "Date,Description,Debit,Credit,Balance\n"
            f"01-01-2025,Salary,{nan_val},50000,50000\n"
        )
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(csv_text))
        cleaned = pp.clean(df)
        assert cleaned.iloc[0]["debit"] == 0.0


# ---------------------------------------------------------------------------
# test_clean – Indian lakh format
# ---------------------------------------------------------------------------

class TestIndianLakhFormat:
    """Amounts like '1,23,456' should be parsed correctly."""

    def test_lakh_format_parsing(self):
        assert _clean_amount_string("1,23,456") == 123456.0

    def test_lakh_format_with_decimals(self):
        assert _clean_amount_string("1,23,456.78") == 123456.78

    def test_lakh_format_with_rupee_symbol(self):
        result = _clean_amount_string("\u20b91,23,456")
        assert result == 123456.0


# ---------------------------------------------------------------------------
# test_clean – duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Exact duplicate rows (same date, description, debit, credit) should be dropped."""

    def test_duplicates_removed(self):
        csv_text = (
            "Date,Description,Debit,Credit,Balance\n"
            "01-01-2025,Swiggy,500,,10000\n"
            "01-01-2025,Swiggy,500,,10000\n"
            "02-01-2025,Amazon,300,,9700\n"
        )
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(csv_text))
        cleaned = pp.clean(df)
        assert len(cleaned) == 2


# ---------------------------------------------------------------------------
# test_get_summary_stats
# ---------------------------------------------------------------------------

class TestGetSummaryStats:
    """get_summary_stats should return a dict with the documented keys."""

    EXPECTED_KEYS = {
        "total_debits",
        "total_credits",
        "net_cashflow",
        "transaction_count",
        "date_range",
        "avg_monthly_spend",
    }

    def test_summary_keys(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(SAMPLE_CSV))
        cleaned = pp.clean(df)
        stats = pp.get_summary_stats(cleaned)
        assert isinstance(stats, dict)
        assert set(stats.keys()) == self.EXPECTED_KEYS

    def test_summary_values(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(SAMPLE_CSV))
        cleaned = pp.clean(df)
        stats = pp.get_summary_stats(cleaned)
        assert stats["transaction_count"] == 3
        assert stats["total_debits"] > 0
        assert stats["total_credits"] > 0


# ---------------------------------------------------------------------------
# test_clean – output columns
# ---------------------------------------------------------------------------

class TestCleanOutputColumns:
    """Cleaned DataFrame must contain the standard columns."""

    EXPECTED_COLS = {
        "date", "description", "debit", "credit", "balance",
        "amount", "transaction_type", "month_year",
    }

    def test_output_columns(self):
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(SAMPLE_CSV))
        cleaned = pp.clean(df)
        assert self.EXPECTED_COLS.issubset(set(cleaned.columns))

    def test_amount_sign_convention(self):
        """Debits should be positive, credits should be negative."""
        pp = BankStatementPreprocessor()
        df = pp.load_csv(_write_csv(SAMPLE_CSV))
        cleaned = pp.clean(df)
        debits = cleaned[cleaned["transaction_type"] == "debit"]
        credits = cleaned[cleaned["transaction_type"] == "credit"]
        assert (debits["amount"] > 0).all()
        assert (credits["amount"] < 0).all()
