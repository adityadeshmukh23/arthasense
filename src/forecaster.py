"""
Prophet-based time-series spending forecaster with ARIMA and simple-average fallbacks.
"""

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Suppress noisy Prophet / cmdstanpy logging at module level
# ---------------------------------------------------------------------------
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet.plot").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", module="prophet")

logger = logging.getLogger(__name__)


class SpendingForecaster:
    """Forecast monthly spending per category using Prophet (with fallbacks)."""

    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.forecasts: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_prophet_df(
        self, df: pd.DataFrame, category: str
    ) -> pd.DataFrame:
        """Filter debit transactions for *category*, aggregate by month,
        and return a DataFrame with ``ds`` and ``y`` columns suitable for
        Prophet.

        Raises
        ------
        ValueError
            If fewer than 3 months of data are available for the category.
        """
        cat_df = df[
            (df["category"] == category) & (df["transaction_type"] == "debit")
        ].copy()

        if cat_df.empty:
            raise ValueError(
                f"No debit transactions found for category '{category}'"
            )

        cat_df["date"] = pd.to_datetime(cat_df["date"])
        monthly = (
            cat_df.set_index("date")
            .resample("MS")["amount"]
            .sum()
            .reset_index()
        )
        monthly.columns = ["ds", "y"]
        monthly["y"] = monthly["y"].abs()  # ensure positive amounts

        if len(monthly) < 3:
            raise ValueError(
                f"Category '{category}' has only {len(monthly)} month(s) of "
                f"data; at least 3 are required for Prophet."
            )

        return monthly

    # ------------------------------------------------------------------
    # ARIMA fallback
    # ------------------------------------------------------------------
    def _arima_fallback(self, monthly_series: pd.Series) -> dict:
        """Use ARIMA(1,1,1) as a fallback forecaster.

        Parameters
        ----------
        monthly_series : pd.Series
            Monthly aggregated spending values (positive).

        Returns
        -------
        dict
            Keys: predicted_amount, lower_bound, upper_bound.
        """
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(monthly_series.values, order=(1, 1, 1))
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=1)
        predicted = forecast_result.predicted_mean[0]
        conf_int = forecast_result.conf_int(alpha=0.05)
        lower = float(conf_int[0, 0])
        upper = float(conf_int[0, 1])

        return {
            "predicted_amount": max(float(predicted), 0.0),
            "lower_bound": max(lower, 0.0),
            "upper_bound": max(upper, 0.0),
        }

    # ------------------------------------------------------------------
    # Core fit + forecast
    # ------------------------------------------------------------------
    def fit_and_forecast(
        self, df: pd.DataFrame, periods: int = 1
    ) -> Dict[str, dict]:
        """Fit Prophet (or fallback) for every spending category and
        forecast *periods* months ahead.

        Returns
        -------
        dict
            Mapping of category -> forecast dict.
        """
        categories = df.loc[
            df["transaction_type"] == "debit", "category"
        ].unique()

        for category in categories:
            try:
                prophet_df = self._prepare_prophet_df(df, category)
            except ValueError:
                # Fewer than 3 months – use simple average * 1.05
                cat_debits = df[
                    (df["category"] == category)
                    & (df["transaction_type"] == "debit")
                ]["amount"].abs()
                if cat_debits.empty:
                    continue

                avg = float(cat_debits.mean())
                forecast_val = avg * 1.05
                self.forecasts[category] = {
                    "predicted_amount": round(forecast_val, 2),
                    "lower_bound": round(forecast_val * 0.85, 2),
                    "upper_bound": round(forecast_val * 1.15, 2),
                    "trend": "stable",
                    "historical_avg": round(avg, 2),
                    "method": "simple_average",
                }
                logger.info(
                    "Category '%s': insufficient data – used simple average.",
                    category,
                )
                continue

            historical_avg = float(prophet_df["y"].mean())

            # --- Attempt 1: Prophet ---
            try:
                from prophet import Prophet

                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                )
                model.fit(prophet_df)
                self.models[category] = model

                future = model.make_future_dataframe(
                    periods=periods, freq="MS"
                )
                forecast = model.predict(future)

                last_row = forecast.iloc[-1]
                predicted = max(float(last_row["yhat"]), 0.0)
                lower = max(float(last_row["yhat_lower"]), 0.0)
                upper = max(float(last_row["yhat_upper"]), 0.0)

                # Determine trend from Prophet's trend component
                trend_vals = forecast["trend"].values
                if len(trend_vals) >= 2:
                    trend_diff = trend_vals[-1] - trend_vals[-2]
                    if trend_diff > 0.01 * historical_avg:
                        trend = "increasing"
                    elif trend_diff < -0.01 * historical_avg:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"

                self.forecasts[category] = {
                    "predicted_amount": round(predicted, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "trend": trend,
                    "historical_avg": round(historical_avg, 2),
                    "method": "prophet",
                }
                logger.info(
                    "Category '%s': Prophet forecast = %.2f",
                    category,
                    predicted,
                )
                continue

            except Exception as prophet_exc:
                logger.warning(
                    "Prophet failed for '%s': %s. Trying ARIMA fallback.",
                    category,
                    prophet_exc,
                )

            # --- Attempt 2: ARIMA fallback ---
            try:
                arima_result = self._arima_fallback(prophet_df["y"])
                # Determine trend from recent values
                recent = prophet_df["y"].values
                if len(recent) >= 2:
                    diff = recent[-1] - recent[-2]
                    if diff > 0.01 * historical_avg:
                        trend = "increasing"
                    elif diff < -0.01 * historical_avg:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"

                self.forecasts[category] = {
                    "predicted_amount": round(arima_result["predicted_amount"], 2),
                    "lower_bound": round(arima_result["lower_bound"], 2),
                    "upper_bound": round(arima_result["upper_bound"], 2),
                    "trend": trend,
                    "historical_avg": round(historical_avg, 2),
                    "method": "arima",
                }
                logger.info(
                    "Category '%s': ARIMA forecast = %.2f",
                    category,
                    arima_result["predicted_amount"],
                )
                continue

            except Exception as arima_exc:
                logger.warning(
                    "ARIMA also failed for '%s': %s. Using simple average.",
                    category,
                    arima_exc,
                )

            # --- Attempt 3: simple average ---
            forecast_val = historical_avg * 1.05
            self.forecasts[category] = {
                "predicted_amount": round(max(forecast_val, 0.0), 2),
                "lower_bound": round(max(forecast_val * 0.85, 0.0), 2),
                "upper_bound": round(max(forecast_val * 1.15, 0.0), 2),
                "trend": "stable",
                "historical_avg": round(historical_avg, 2),
                "method": "simple_average",
            }
            logger.info(
                "Category '%s': simple-average forecast = %.2f",
                category,
                forecast_val,
            )

        return self.forecasts

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def get_forecast_summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising forecasts across categories.

        Columns
        -------
        category, next_month_forecast, lower_bound, upper_bound,
        vs_last_month_pct_change, trend_direction
        """
        rows = []
        for category, fcast in self.forecasts.items():
            predicted = fcast["predicted_amount"]
            hist_avg = fcast["historical_avg"]
            pct_change = (
                round(((predicted - hist_avg) / hist_avg) * 100, 2)
                if hist_avg != 0
                else 0.0
            )
            rows.append(
                {
                    "category": category,
                    "next_month_forecast": predicted,
                    "lower_bound": fcast["lower_bound"],
                    "upper_bound": fcast["upper_bound"],
                    "vs_last_month_pct_change": pct_change,
                    "trend_direction": fcast["trend"],
                }
            )

        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary.sort_values(
                "next_month_forecast", ascending=False, inplace=True
            )
            summary.reset_index(drop=True, inplace=True)
        return summary

    def get_total_forecast(self) -> dict:
        """Aggregate forecasts across all categories.

        Returns
        -------
        dict
            Keys: total_forecast, total_lower, total_upper.
        """
        total_forecast = sum(
            f["predicted_amount"] for f in self.forecasts.values()
        )
        total_lower = sum(f["lower_bound"] for f in self.forecasts.values())
        total_upper = sum(f["upper_bound"] for f in self.forecasts.values())

        return {
            "total_forecast": round(total_forecast, 2),
            "total_lower": round(total_lower, 2),
            "total_upper": round(total_upper, 2),
        }
