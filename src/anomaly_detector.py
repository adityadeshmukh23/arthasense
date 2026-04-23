"""
Isolation Forest based anomaly detector for financial transactions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalous transactions using Isolation Forest."""

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.merchant_counts: Optional[pd.Series] = None
        self.category_stats: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from transaction DataFrame.

        Features:
            1. amount (log-transformed)
            2. day_of_month
            3. day_of_week
            4. hour_of_day (default 12)
            5. merchant_frequency
            6. amount_zscore per category
            7. is_weekend
            8. days_since_last_transaction
        """
        features = pd.DataFrame(index=df.index)

        # 1. Log-transformed amount
        features["amount_log"] = np.log1p(df["amount"].abs())

        # 2. Day of month
        date_col = pd.to_datetime(df["date"])
        features["day_of_month"] = date_col.dt.day

        # 3. Day of week
        features["day_of_week"] = date_col.dt.dayofweek

        # 4. Hour of day (banks don't provide time)
        features["hour_of_day"] = 12

        # 5. Merchant frequency
        if self.merchant_counts is not None:
            features["merchant_frequency"] = (
                df["description"].map(self.merchant_counts).fillna(0)
            )
        else:
            merchant_counts = df["description"].value_counts()
            features["merchant_frequency"] = (
                df["description"].map(merchant_counts).fillna(0)
            )

        # 6. Amount z-score per category
        if self.category_stats is not None:
            cat_mean = df["category"].map(
                self.category_stats["mean"]
            ).fillna(df["amount"].mean())
            cat_std = df["category"].map(
                self.category_stats["std"]
            ).fillna(df["amount"].std())
        else:
            cat_mean = df.groupby("category")["amount"].transform("mean")
            cat_std = df.groupby("category")["amount"].transform("std")

        cat_std = cat_std.replace(0, 1)
        features["amount_zscore"] = (df["amount"].abs() - cat_mean.abs()) / cat_std

        # 7. Is weekend
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # 8. Days since last transaction
        sorted_dates = date_col.sort_values()
        days_gap = sorted_dates.diff().dt.days.fillna(0)
        features["days_since_last"] = days_gap.reindex(df.index).fillna(0)

        feature_matrix = features.values.astype(float)

        if self.scaler is not None and self.is_fitted:
            feature_matrix = self.scaler.transform(feature_matrix)
        else:
            self.scaler = StandardScaler()
            feature_matrix = self.scaler.fit_transform(feature_matrix)

        return feature_matrix

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the anomaly detector on debit transactions only."""
        debit_df = df[df["transaction_type"] == "debit"].copy()

        if debit_df.empty:
            logger.warning("No debit transactions found for fitting.")
            return self

        logger.info("Fitting anomaly detector on %d debit transactions.", len(debit_df))

        # Store merchant counts and category stats from training data
        self.merchant_counts = debit_df["description"].value_counts()
        self.category_stats = debit_df.groupby("category")["amount"].agg(
            ["mean", "std"]
        )
        self.category_stats["std"] = self.category_stats["std"].fillna(1).replace(0, 1)

        # Build features (this also fits the scaler)
        feature_matrix = self._build_features(debit_df)

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
        )
        self.model.fit(feature_matrix)
        self.is_fitted = True

        logger.info("Anomaly detector fitted successfully.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly scores and flags to the DataFrame.

        Adds columns:
            - anomaly_score: float in [0, 1], higher means more anomalous
            - is_anomaly: bool, True if anomaly_score > 0.65 or duplicate detected
            - anomaly_reason: str, human-readable explanation
        """
        result = df.copy()

        if not self.is_fitted or self.model is None:
            logger.warning("Model not fitted. Returning DataFrame without predictions.")
            result["anomaly_score"] = 0.0
            result["is_anomaly"] = False
            result["anomaly_reason"] = ""
            return result

        debit_mask = result["transaction_type"] == "debit"
        debit_df = result[debit_mask].copy()

        # Initialize columns
        result["anomaly_score"] = 0.0
        result["is_anomaly"] = False
        result["anomaly_reason"] = ""

        if debit_df.empty:
            return result

        feature_matrix = self._build_features(debit_df)

        # Raw scores from Isolation Forest (lower = more anomalous)
        raw_scores = self.model.decision_function(feature_matrix)

        # Convert: score = (1 - raw_score) / 2, so higher = more anomalous
        anomaly_scores = (1 - raw_scores) / 2
        anomaly_scores = np.clip(anomaly_scores, 0, 1)

        result.loc[debit_mask, "anomaly_score"] = anomaly_scores
        result.loc[debit_mask, "is_anomaly"] = anomaly_scores > 0.65

        # Flag duplicate (merchant, amount) within 7 days
        debit_sorted = debit_df.sort_values("date")
        date_col = pd.to_datetime(debit_sorted["date"])

        for i, (idx, row) in enumerate(debit_sorted.iterrows()):
            current_date = date_col.iloc[i]
            mask = (
                (debit_sorted["description"] == row["description"])
                & (debit_sorted["amount"] == row["amount"])
                & (debit_sorted.index != idx)
                & ((current_date - pd.to_datetime(debit_sorted["date"])).abs().dt.days <= 7)
            )
            if mask.any():
                result.loc[idx, "is_anomaly"] = True
                if result.loc[idx, "anomaly_score"] < 0.65:
                    result.loc[idx, "anomaly_score"] = 0.66

        # Generate anomaly reasons
        for idx in debit_df.index:
            if result.loc[idx, "is_anomaly"]:
                row = result.loc[idx]
                feature_row = feature_matrix[debit_df.index.get_loc(idx)]
                result.loc[idx, "anomaly_reason"] = self._get_anomaly_reason(
                    row, feature_row
                )

        logger.info(
            "Predicted %d anomalies out of %d debit transactions.",
            result["is_anomaly"].sum(),
            len(debit_df),
        )
        return result

    def _get_anomaly_reason(self, row: pd.Series, features: np.ndarray) -> str:
        """Generate a human-readable reason for the anomaly."""
        reasons = []
        amount = abs(row["amount"])
        category = row["category"]
        merchant = row["description"]

        # Check for unusually large amount
        if self.category_stats is not None and category in self.category_stats.index:
            cat_mean = abs(self.category_stats.loc[category, "mean"])
            cat_std = self.category_stats.loc[category, "std"]
            if amount > cat_mean + 2 * cat_std:
                reasons.append(
                    f"Unusually large amount (\u20b9{amount:,.0f} vs avg "
                    f"\u20b9{cat_mean:,.0f} for {category})"
                )

        # Check for duplicate merchant charge
        if self.merchant_counts is not None:
            date_col = pd.to_datetime(row["date"]) if isinstance(row["date"], str) else row["date"]
            # The duplicate flag is set in predict(); mention it here
            if row.get("anomaly_score", 0) >= 0.65:
                # Check merchant frequency
                merchant_freq = self.merchant_counts.get(merchant, 0)
                if merchant_freq == 0:
                    reasons.append(
                        f"First-time merchant with high amount (\u20b9{amount:,.0f})"
                    )

        # Check for same merchant charged within 7 days (reason added if score was bumped)
        if row.get("anomaly_score", 0) == 0.66:
            reasons.append(
                f"Same merchant charged twice within 7 days"
            )

        # Fallback if no specific reason identified
        if not reasons:
            if self.category_stats is not None and category in self.category_stats.index:
                cat_mean = abs(self.category_stats.loc[category, "mean"])
                reasons.append(
                    f"Unusual spending pattern (\u20b9{amount:,.0f} vs avg "
                    f"\u20b9{cat_mean:,.0f} for {category})"
                )
            else:
                reasons.append(f"Unusual transaction pattern (\u20b9{amount:,.0f})")

        return "; ".join(reasons)

    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """Return a summary of detected anomalies.

        Returns:
            dict with keys:
                - total_anomalies: int
                - total_flagged_amount: float
                - top_anomalies: list of dicts with transaction details
        """
        if "is_anomaly" not in df.columns:
            df = self.predict(df)

        anomalies = df[df["is_anomaly"] == True].copy()

        total_anomalies = len(anomalies)
        total_flagged_amount = anomalies["amount"].abs().sum() if total_anomalies > 0 else 0.0

        top_anomalies = []
        if total_anomalies > 0:
            top = anomalies.nlargest(
                min(10, total_anomalies), "anomaly_score"
            )
            for _, row in top.iterrows():
                top_anomalies.append(
                    {
                        "date": str(row["date"]),
                        "description": row["description"],
                        "amount": float(row["amount"]),
                        "category": row["category"],
                        "anomaly_score": float(row["anomaly_score"]),
                        "reason": row.get("anomaly_reason", ""),
                    }
                )

        summary = {
            "total_anomalies": total_anomalies,
            "total_flagged_amount": float(total_flagged_amount),
            "top_anomalies": top_anomalies,
        }

        logger.info("Anomaly summary: %d anomalies, \u20b9%.2f flagged.", total_anomalies, total_flagged_amount)
        return summary
