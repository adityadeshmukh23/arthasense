"""
LLM-based financial advice engine powered by Google Gemini.

Provides personalised financial guidance for Indian users by analysing
transaction summaries, category breakdowns, anomalies and forecasts.
Falls back to rule-based advice when the API key is missing or the API
call fails.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Models tried in order; rotates past quota (429) and not-found (404) errors.
# gemini-2.0-flash-lite has a separate, higher free-tier quota from gemini-2.0-flash.
_GEMINI_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are ArthaSense — a sharp, finance-savvy friend who has just reviewed someone's bank statement. You're smart, direct, and genuinely helpful — not a lecture-bot.

Rules you must follow without exception:
- Do NOT open with a greeting, preamble, or filler. Start with the most important insight immediately.
- Always cite the exact ₹ figures from the data. Never say "significant" or "considerable" — say the number.
- Express all amounts in Indian Rupee notation: ₹1,23,456.
- Structure your response as: one short opening line identifying the key money pattern, then 3–4 tight bullet points (one insight each), then one bold "Next step" line.
- Each bullet must name a specific category or metric and say why it matters — in one crisp sentence.
- Tone: smart, warm, direct. Like a friend who happens to be great with money. No corporate jargon, no moralising, no fluff.
- Total response must be under 200 words. Brevity is a feature, not a compromise.
- Use proper markdown: `- ` (hyphen + space) for each bullet point so they render as a list, **bold** for the Next step label.
- The Next step must be a concrete action the user can take today — specific category, specific amount or %, specific product (SIP, RD, etc.). Never say "let's look at" or "explore further" — tell them what to DO."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_inr(amount: float) -> str:
    """Format *amount* in the Indian numbering system (e.g. 1,23,456.78)."""
    if amount < 0:
        return f"-{_fmt_inr(-amount)}"
    amount = round(amount, 2)
    integer_part = int(amount)
    decimal_part = f"{amount:.2f}".split(".")[1]

    s = str(integer_part)
    if len(s) <= 3:
        formatted = s
    else:
        # Last three digits, then groups of two from the right
        last3 = s[-3:]
        remaining = s[:-3]
        groups: list[str] = []
        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        groups.reverse()
        formatted = ",".join(groups) + "," + last3

    return f"\u20b9{formatted}.{decimal_part}"


# ---------------------------------------------------------------------------
# LLMAdvisor
# ---------------------------------------------------------------------------

class LLMAdvisor:
    """Financial advice engine backed by Google Gemini (gemini-1.5-flash).

    When an API key is not supplied or the remote call fails, the advisor
    transparently falls back to deterministic, rule-based advice derived
    from the same financial data.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key: Optional[str] = api_key if api_key else None
        self._client = None
        self.gemini_active: bool = False
        self.init_error: Optional[str] = None
        self._model_name: Optional[str] = None
        self._models_to_try: tuple = _GEMINI_MODELS  # overwritten by discovery

        if self._api_key and GEMINI_AVAILABLE:
            self._client = _genai.Client(api_key=self._api_key)
            self._models_to_try = self._discover_models()
        elif self._api_key and not GEMINI_AVAILABLE:
            self.init_error = "google-genai package not installed"
            logger.warning("google-genai not installed; using fallback advice.")

    def _discover_models(self) -> tuple:
        """List models available to this API key and return them preference-ordered."""
        try:
            found = []
            for m in self._client.models.list():
                name = getattr(m, "name", "") or ""
                name = name.replace("models/", "").strip()
                if not name or "gemini" not in name.lower():
                    continue
                # Only include models that can generate content
                methods = (
                    getattr(m, "supported_generation_methods", None)
                    or getattr(m, "supported_actions", None)
                    or []
                )
                if methods and not any(
                    "generate" in str(x).lower() for x in methods
                ):
                    continue
                found.append(name)

            if not found:
                logger.warning("Model discovery returned no models; using defaults.")
                return _GEMINI_MODELS

            # Prefer flash variants, prefer newer (2.x > 1.x), avoid -exp/-preview
            def _rank(n: str) -> tuple:
                unstable = any(x in n for x in ("-exp", "-preview", "-latest"))
                ver = 2 if "2." in n else 1
                size = 0 if "8b" not in n else 1   # smaller = lower priority
                return (not unstable, ver, -size)

            found.sort(key=_rank, reverse=True)
            logger.info("Discovered models: %s", found)
            return tuple(found)
        except Exception as exc:
            logger.warning("Model discovery failed (%s); using defaults.", exc)
            return _GEMINI_MODELS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_config(self) -> "_genai_types.GenerateContentConfig":
        return _genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5,
            max_output_tokens=600,
        )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        s = str(exc)
        # 429 = quota exhausted; 404 NOT_FOUND = model unavailable in this API version
        return (
            "429" in s or "RESOURCE_EXHAUSTED" in s
            or ("404" in s and "NOT_FOUND" in s)
        )

    def _get_response(self, prompt: str):
        """Non-streaming call with model rotation on 429. Raises on all other errors."""
        config = self._make_config()
        last_exc: Optional[Exception] = None
        for model_name in self._models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                self._model_name = model_name
                self.gemini_active = True
                logger.info("Gemini (blocking) used model: %s", model_name)
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_quota_error(exc):
                    logger.warning("Quota/unavailable for %s, trying next.", model_name)
                    continue
                raise
        raise last_exc

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_financial_context(
        self,
        summary_stats: dict,
        category_totals: pd.DataFrame,
        anomalies: pd.DataFrame,
        forecasts: dict,
    ) -> str:
        """Assemble a structured plain-text prompt from the financial data."""

        sections: list[str] = []

        # --- FINANCIAL SUMMARY -------------------------------------------
        lines: list[str] = ["FINANCIAL SUMMARY"]
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {_fmt_inr(value)}")
            else:
                lines.append(f"  {key}: {value}")
        sections.append("\n".join(lines))

        # --- CATEGORY BREAKDOWN ------------------------------------------
        if category_totals is not None and not category_totals.empty:
            cat_lines: list[str] = ["CATEGORY BREAKDOWN"]
            for _, row in category_totals.iterrows():
                category = row.iloc[0] if not isinstance(row.name, str) else row.name
                amount = row.iloc[-1] if not isinstance(row.name, str) else row.iloc[0]
                if isinstance(amount, (int, float)):
                    cat_lines.append(f"  {category}: {_fmt_inr(amount)}")
                else:
                    cat_lines.append(f"  {category}: {amount}")
            sections.append("\n".join(cat_lines))

        # --- ANOMALOUS TRANSACTIONS (top 5 only to save tokens) ---------
        if anomalies is not None and not anomalies.empty:
            anom_lines: list[str] = [f"ANOMALOUS TRANSACTIONS ({len(anomalies)} total, top 5 shown)"]
            for _, row in anomalies.head(5).iterrows():
                desc = row.get("description", row.iloc[0])
                amt  = row.get("amount", row.get("debit", ""))
                cat  = row.get("category", "")
                score = row.get("anomaly_score", "")
                amt_str = _fmt_inr(float(amt)) if isinstance(amt, (int, float)) else str(amt)
                score_str = f"score={float(score):.2f}" if isinstance(score, (int, float)) else ""
                anom_lines.append(f"  - {desc}: {amt_str} [{cat}] {score_str}".strip())
            sections.append("\n".join(anom_lines))

        # --- FORECAST ----------------------------------------------------
        if forecasts:
            fc_lines: list[str] = ["FORECAST"]
            for key, value in forecasts.items():
                if isinstance(value, (int, float)):
                    fc_lines.append(f"  {key}: {_fmt_inr(value)}")
                elif isinstance(value, list):
                    formatted = [_fmt_inr(v) if isinstance(v, (int, float)) else str(v) for v in value[:6]]
                    fc_lines.append(f"  {key}: {', '.join(formatted)}")
                else:
                    fc_lines.append(f"  {key}: {value}")
            sections.append("\n".join(fc_lines))

        return "\n\n".join(sections)

    def _build_prompt(
        self,
        summary_stats: dict,
        category_totals: pd.DataFrame,
        anomalies: pd.DataFrame,
        forecasts: dict,
    ) -> str:
        context = self._build_financial_context(
            summary_stats, category_totals, anomalies, forecasts
        )
        return (
            "Here's the bank statement data. Give me your sharp read — "
            "one punchy opening line, 3–4 bullet insights with exact ₹ figures, "
            "and one bold Next step. Stay under 180 words.\n\n"
            + context
            + "\n\nYour analysis:"
        )

    # ------------------------------------------------------------------
    # Main advice (blocking)
    # ------------------------------------------------------------------

    def get_advice(
        self,
        summary_stats: dict,
        category_totals: pd.DataFrame,
        anomalies: pd.DataFrame,
        forecasts: dict,
    ) -> str:
        """Return financial advice as a single string."""
        if not self._client:
            return self._generate_fallback_advice(
                summary_stats, category_totals, anomalies, forecasts
            )
        prompt = self._build_prompt(summary_stats, category_totals, anomalies, forecasts)
        try:
            response = self._get_response(prompt, stream=False)
            text = response.text.strip() if response.text else ""
            if not text:
                raise ValueError("Empty response from Gemini")
            return text
        except Exception as exc:  # noqa: BLE001
            self.init_error = str(exc)
            self.gemini_active = False
            logger.warning("Gemini call failed (%s). Using fallback.", exc)
            return self._generate_fallback_advice(
                summary_stats, category_totals, anomalies, forecasts
            )

    # ------------------------------------------------------------------
    # Streaming advice
    # ------------------------------------------------------------------

    def get_advice_streaming(
        self,
        summary_stats: dict,
        category_totals: pd.DataFrame,
        anomalies: pd.DataFrame,
        forecasts: dict,
    ) -> Generator[str, None, None]:
        """Yield advice chunks via Gemini streaming.

        Rotates through _GEMINI_MODELS on 429 quota errors.
        The rotation loop wraps the iteration, not just the call, because
        generate_content_stream() is lazy — it doesn't make the HTTP
        request until the first chunk is consumed.
        """
        if not self._client:
            yield self._generate_fallback_advice(
                summary_stats, category_totals, anomalies, forecasts
            )
            return

        prompt = self._build_prompt(summary_stats, category_totals, anomalies, forecasts)
        config = self._make_config()
        last_exc: Optional[Exception] = None

        for model_name in self._models_to_try:
            try:
                stream = self._client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                any_text = False
                for chunk in stream:          # HTTP call happens here, not above
                    text = getattr(chunk, "text", None)
                    if text:
                        any_text = True
                        yield text
                if not any_text:
                    raise ValueError("Empty streaming response")
                self._model_name = model_name
                self.gemini_active = True
                logger.info("Gemini streaming used model: %s", model_name)
                return                        # success — stop rotating
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_quota_error(exc):
                    logger.warning(
                        "Quota exceeded for %s, trying next model.", model_name
                    )
                    continue
                # Non-quota error — don't try other models
                self.init_error = str(exc)
                self.gemini_active = False
                logger.warning("Gemini streaming failed (%s). Using fallback.", exc)
                yield self._generate_fallback_advice(
                    summary_stats, category_totals, anomalies, forecasts
                )
                return

        # All models quota-exhausted
        self.init_error = str(last_exc)
        self.gemini_active = False
        logger.warning("All Gemini models quota-exhausted. Using fallback.")
        yield self._generate_fallback_advice(
            summary_stats, category_totals, anomalies, forecasts
        )

    # ------------------------------------------------------------------
    # Fallback: rule-based advice
    # ------------------------------------------------------------------

    def _generate_fallback_advice(
        self,
        summary_stats: dict,
        category_totals: pd.DataFrame,
        anomalies: pd.DataFrame,
        forecasts: dict,
    ) -> str:
        """Rule-based fallback advice matching the concise, human-friendly tone."""
        lines: list[str] = []

        total_income = (
            summary_stats.get("total_credits")
            if summary_stats.get("total_credits") is not None
            else summary_stats.get("total_income", 0.0)
        ) or 0.0
        total_expense = abs(
            (
                summary_stats.get("total_debits")
                if summary_stats.get("total_debits") is not None
                else summary_stats.get("total_expense") or summary_stats.get("total_expenses", 0.0)
            ) or 0.0
        )
        txn_count     = summary_stats.get("transaction_count", 0) or summary_stats.get("num_transactions", 0) or 0

        # Opening line
        if total_income > 0:
            savings      = total_income - total_expense
            savings_rate = (savings / total_income) * 100
            if savings_rate >= 20:
                lines.append(
                    f"You're saving {savings_rate:.0f}% of your income — that's genuinely good. Here's what the numbers say:"
                )
            elif savings_rate >= 0:
                lines.append(
                    f"Saving {savings_rate:.0f}% of income. Room to grow — let's see where the leaks are:"
                )
            else:
                lines.append(
                    f"Spending is outpacing income by {_fmt_inr(abs(savings))}. Worth addressing now before it compounds:"
                )
        else:
            lines.append(f"Here's the read on {txn_count} transactions:")

        lines.append("")

        # Bullet insights
        if total_income > 0:
            savings = total_income - total_expense
            lines.append(
                f"- Income {_fmt_inr(total_income)} vs spend {_fmt_inr(total_expense)} — "
                f"net {'surplus' if savings >= 0 else 'deficit'} of {_fmt_inr(abs(savings))}."
            )

        if category_totals is not None and not category_totals.empty:
            try:
                if isinstance(category_totals.index[0], str):
                    amounts  = category_totals.iloc[:, 0].abs()
                    top_cat  = amounts.idxmax()
                    top_amt  = amounts.max()
                    total_cat = amounts.sum()
                else:
                    amounts   = category_totals.iloc[:, -1].abs()
                    idx       = amounts.idxmax()
                    top_cat   = category_totals.iloc[idx, 0] if idx == int(idx) else str(idx)
                    top_amt   = amounts.max()
                    total_cat = amounts.sum()

                pct = (top_amt / total_cat) * 100 if total_cat else 0
                if pct > 40:
                    lines.append(
                        f"- {top_cat} is eating {_fmt_inr(top_amt)} ({pct:.0f}% of spend) — that's a concentrated outflow worth reviewing."
                    )
                else:
                    lines.append(
                        f"- Biggest category is {top_cat} at {_fmt_inr(top_amt)} ({pct:.0f}%) — spending looks reasonably spread out."
                    )
            except Exception:  # noqa: BLE001
                logger.debug("Could not analyse category totals for fallback.")

        if anomalies is not None and not anomalies.empty:
            n = len(anomalies)
            lines.append(
                f"- {n} transaction{'s' if n != 1 else ''} flagged as unusual — worth a quick check to rule out errors or one-offs."
            )

        if forecasts:
            fc = forecasts.get("predicted_expense") or forecasts.get("next_month_expense")
            if isinstance(fc, (int, float)):
                lines.append(
                    f"- Next month's spend is forecast at {_fmt_inr(fc)} — set that aside now so it doesn't catch you off guard."
                )
            elif isinstance(fc, list) and fc and isinstance(fc[0], (int, float)):
                lines.append(
                    f"- Forecast puts next period's expenses around {_fmt_inr(fc[0])} — good to plan for it now."
                )

        lines.append("")

        # Next step
        if total_income > 0:
            savings      = total_income - total_expense
            savings_rate = (savings / total_income) * 100
            if savings_rate < 20 and category_totals is not None and not category_totals.empty:
                try:
                    if isinstance(category_totals.index[0], str):
                        top_cat = category_totals.iloc[:, 0].abs().idxmax()
                    else:
                        top_cat = str(category_totals.iloc[category_totals.iloc[:, -1].abs().idxmax(), 0])
                    lines.append(f"**Next step:** Trim {top_cat} spending by 15–20% to push your savings rate past the 20% mark.")
                except Exception:  # noqa: BLE001
                    lines.append("**Next step:** Set a monthly spending cap for your top category and review it each week.")
            elif savings_rate >= 20:
                lines.append("**Next step:** Park the surplus in a SIP or RD — idle savings in a current account lose value to inflation.")
            else:
                lines.append("**Next step:** Identify one discretionary category you can cut this month to stop the shortfall.")
        else:
            lines.append("**Next step:** Link your income data for a complete picture — it'll make these insights a lot sharper.")

        return "\n".join(lines)
