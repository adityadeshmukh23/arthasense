"""Tests for src.classifier – TransactionClassifier (keyword-only mode)."""

import pandas as pd
import pytest

from src.classifier import CATEGORIES, KEYWORD_MAP, TransactionClassifier


# ---------------------------------------------------------------------------
# Fixture: classifier with zero-shot disabled
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    """Return a TransactionClassifier that never loads the ML model."""
    return TransactionClassifier(use_zero_shot=False)


# ---------------------------------------------------------------------------
# test keyword map – 10 common merchants
# ---------------------------------------------------------------------------

class TestKeywordMap:
    """Verify keyword lookup for well-known Indian merchants."""

    @pytest.mark.parametrize(
        "description, expected_category",
        [
            ("Swiggy order #12345", "Food and Dining"),
            ("Uber trip to airport", "Transportation"),
            ("Netflix subscription", "Entertainment and Subscriptions"),
            ("Amazon purchase", "Shopping and Retail"),
            ("Airtel mobile recharge", "Utilities and Bills"),
            ("Apollo pharmacy", "Healthcare and Medical"),
            ("Zerodha fund transfer", "Investment and Savings"),
            ("EMI payment March", "Loan and EMI"),
            ("NEFT to savings", "Transfer and Banking"),
            ("Salary credit Mar 2025", "Income and Salary"),
        ],
    )
    def test_keyword_classification(self, classifier, description, expected_category):
        category, confidence = classifier.classify_single(description)
        assert category == expected_category
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# test all CATEGORIES reachable via keyword map
# ---------------------------------------------------------------------------

class TestCategoryCoverage:
    """Every category in CATEGORIES must be reachable through KEYWORD_MAP."""

    def test_all_categories_in_keyword_map(self):
        categories_in_map = set(KEYWORD_MAP.values())
        for cat in CATEGORIES:
            assert cat in categories_in_map, (
                f"Category '{cat}' has no keyword entry in KEYWORD_MAP"
            )


# ---------------------------------------------------------------------------
# test batch classification
# ---------------------------------------------------------------------------

class TestBatchClassification:
    """classify_batch should return a DataFrame with the correct shape."""

    def test_batch_returns_correct_shape(self, classifier):
        descriptions = [
            "Swiggy dinner",
            "Uber ride",
            "Netflix monthly",
            "Unknown merchant XYZ",
            "Zomato lunch",
        ]
        result = classifier.classify_batch(descriptions)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 3)
        assert list(result.columns) == ["description", "category", "confidence"]

    def test_batch_keyword_matches(self, classifier):
        descriptions = ["Swiggy order", "Uber cab"]
        result = classifier.classify_batch(descriptions)
        assert result.iloc[0]["category"] == "Food and Dining"
        assert result.iloc[1]["category"] == "Transportation"

    def test_batch_unknown_defaults(self, classifier):
        """With zero-shot disabled, unknown merchants fall back to Transfer and Banking."""
        descriptions = ["completely unknown merchant 999"]
        result = classifier.classify_batch(descriptions)
        assert result.iloc[0]["category"] == "Transfer and Banking"
        assert result.iloc[0]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# test empty description handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_description(self, classifier):
        category, confidence = classifier.classify_single("")
        assert category == "Transfer and Banking"
        assert confidence == 0.0

    def test_none_like_description(self, classifier):
        category, confidence = classifier.classify_single("   ")
        assert category == "Transfer and Banking"
