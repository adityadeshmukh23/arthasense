"""Tests for src.anomaly_detector – AnomalyDetector."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import AnomalyDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_normal: int = 50,
    normal_range: tuple = (200, 800),
    outlier_amount: float = 50_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic transaction DataFrame.

    Contains *n_normal* food transactions with amounts drawn uniformly from
    *normal_range*, plus one outlier at *outlier_amount*.  All transactions
    are debits in the "Food and Dining" category.
    """
    rng = np.random.RandomState(seed)

    n_total = n_normal + 1
    dates = pd.date_range("2025-01-01", periods=n_total, freq="D")

    amounts_normal = rng.uniform(normal_range[0], normal_range[1], size=n_normal)
    amounts = np.append(amounts_normal, outlier_amount)

    descriptions = ["Swiggy Order"] * n_normal + ["Swiggy Order Outlier"]

    df = pd.DataFrame(
        {
            "date": dates,
            "description": descriptions,
            "amount": amounts,
            "transaction_type": ["debit"] * n_total,
            "category": ["Food and Dining"] * n_total,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return AnomalyDetector(contamination=0.10)


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Core anomaly detection on synthetic data."""

    def test_fit_returns_self(self, detector, synthetic_data):
        result = detector.fit(synthetic_data)
        assert result is detector
        assert detector.is_fitted is True

    def test_predict_adds_columns(self, detector, synthetic_data):
        detector.fit(synthetic_data)
        predicted = detector.predict(synthetic_data)
        for col in ("anomaly_score", "is_anomaly", "anomaly_reason"):
            assert col in predicted.columns

    def test_outlier_flagged(self, detector, synthetic_data):
        """The single 50,000 outlier should have a higher anomaly score than normal transactions."""
        detector.fit(synthetic_data)
        predicted = detector.predict(synthetic_data)
        outlier_row = predicted[predicted["amount"] == 50_000.0]
        normal_rows = predicted[predicted["amount"] < 1000]
        assert len(outlier_row) == 1
        outlier_score = outlier_row.iloc[0]["anomaly_score"]
        avg_normal_score = normal_rows["anomaly_score"].mean()
        assert outlier_score > avg_normal_score, (
            f"Outlier score ({outlier_score:.3f}) should exceed normal avg ({avg_normal_score:.3f})"
        )

    def test_normal_transactions_mostly_clean(self, detector, synthetic_data):
        """The vast majority of normal transactions should NOT be flagged."""
        detector.fit(synthetic_data)
        predicted = detector.predict(synthetic_data)
        normal = predicted[predicted["amount"] < 1000]
        flagged_ratio = normal["is_anomaly"].sum() / len(normal)
        # With contamination=0.05 we expect at most ~10 % flagged in practice
        assert flagged_ratio < 0.20, (
            f"Too many normal transactions flagged: {flagged_ratio:.0%}"
        )


class TestAnomalySummary:
    """get_anomaly_summary should return a dict with the documented keys."""

    EXPECTED_KEYS = {"total_anomalies", "total_flagged_amount", "top_anomalies"}

    def test_summary_keys(self, detector, synthetic_data):
        detector.fit(synthetic_data)
        predicted = detector.predict(synthetic_data)
        summary = detector.get_anomaly_summary(predicted)
        assert isinstance(summary, dict)
        assert set(summary.keys()) == self.EXPECTED_KEYS

    def test_summary_values(self, detector, synthetic_data):
        detector.fit(synthetic_data)
        predicted = detector.predict(synthetic_data)
        summary = detector.get_anomaly_summary(predicted)
        assert summary["total_anomalies"] >= 0
        assert summary["total_flagged_amount"] >= 0
        assert isinstance(summary["top_anomalies"], list)


class TestUnfittedDetector:
    """Predict on an unfitted detector should return safe defaults."""

    def test_predict_without_fit(self, synthetic_data):
        detector = AnomalyDetector()
        predicted = detector.predict(synthetic_data)
        assert (predicted["anomaly_score"] == 0.0).all()
        assert (predicted["is_anomaly"] == False).all()
