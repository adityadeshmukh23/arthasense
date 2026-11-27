# ArthaSense

A Streamlit prototype that answers one question for a salaried person in India:

> **How much can I safely spend this week without running short before salary day?**

Upload a bank statement (CSV or Excel), check the details we found, and get a weekly amount with the arithmetic shown. Budgeting help only, not financial or investment advice. Version 3.0.

## The four screens (`app.py`)

1. **Start**: what the app does, what happens to your file, upload (needs consent), or try the sample in one click.
2. **Confirm**: today's date, next salary day, current balance and its "as of" date, bills due before payday, payday bills (off by default), planned transfers to your own accounts, and a buffer (₹2,000 by default). Everything can be edited, added to, removed or retyped.
3. **Answer**: the weekly safe amount next to your usual weekly everyday spend, "How we got this", and one optional change.
4. **Check-in**: enter spending since the last check and see what is left this week and per day until payday. Nothing is saved after the session.

## The formula

```
weekly = (balance − bills before payday − buffer) / days to payday × min(7, days to payday)
```

Rounded down to the rupee. With the synthetic sample:

```
₹31,524 − ₹399 (Jio postpaid, due 22 Nov) − ₹2,000 buffer = ₹29,125
₹29,125 ÷ 11 days (20 Nov → 1 Dec) = ₹2,647.73 a day
₹2,647.73 × 7 = ₹18,534.09 → ₹18,534 this week
```

Rules (all in `src/safe_to_spend.py`, tested in `tests/test_safe_to_spend.py`):

- **Today** defaults to the statement's last date and can be changed. If it differs from the balance's date, you must re-enter or confirm the balance.
- **Payday** is the last salary's day of the month, one month later (editable). Salary is found by keyword ("salary", "sal", "payroll") or by a steady large credit from the same payer in two or more months. If none is found, you must enter the date.
- **Bills** are monthly debits seen in 2+ months on a similar day with a steady amount (rent, EMIs, bills, subscriptions). We include those due after the balance date and before payday, plus any expected but not yet seen (up to 31 days late). Bills due on payday are listed separately and left out unless you tick them.
- **Own-account transfers** ("SELF", "OWN A/C", or the same amount out and back within a day) are never counted as spending. Regular ones due before payday are listed as planned outflows you can remove.
- **Shortfall**: if the balance minus bills and buffer is below zero, the app says how short you are and shows no weekly number.
- **Usual weekly spend**: everyday debits (not bills, rent, EMIs or own-account transfers), averaged over up to 8 full weeks.
- **One change**: the largest discretionary group (food delivery, eating out, shopping, entertainment) averaged over up to 8 weeks, with 25% less offered as an option. It is hidden with fewer than 4 weeks of data.
- **Check-in**: entered spend excludes listed bills. If you go over the weekly limit, the app says by how much and recalculates a daily amount until payday.

There is no AI call anywhere in this journey.

## File support

- **CSV and Excel (.xlsx)**: supported. Tested with the sample and with a fake HDFC-style file (newest-first rows, "Withdrawal Amt." headers, salary without a keyword) in `tests/fixtures/`.
- **PDF**: not accepted in the main journey. In testing, a PDF with a Dr/Cr column parsed silently wrong (half the rows, no credits). The legacy dashboard still accepts PDFs.
- Maximum upload: **10 MB** (`server.maxUploadSize` in `.streamlit/config.toml`).

## Where your data goes

- An uploaded file is read **in memory by the server running the app**. Locally, that is your machine. On the hosted demo it is Streamlit Community Cloud, so the file does leave your device.
- Uploads are not written to disk and are gone when the session ends. Logs contain counts, not transaction text.
- Streamlit's browser usage telemetry is turned off (`browser.gatherUsageStats = false`).
- The main journey sends nothing to any AI service. The legacy dashboard's optional Gemini advice sends Google a summary (category totals, forecasts, and up to 5 flagged transactions with descriptions and amounts). A server-side `GEMINI_API_KEY` is used only on the server and is never sent to the browser.

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py              # new 4-screen journey
streamlit run app_legacy.py       # previous dashboard (categories, anomalies, forecast, optional Gemini)
pytest tests/ -v
```

`transformers`, `torch` and `prophet` are used only by the legacy dashboard's optional features. The new app does not need them.

## Sample data

`data/sample_transactions.csv` is **synthetic**. It describes a made-up salaried person in Bengaluru, 21 May – 20 Nov 2026:

- ₹58,400 salary on the 1st, ₹16,000 rent, a ₹3,450 phone EMI
- four subscriptions and a monthly ₹5,000 transfer to their own account
- everyday spending
- saves 12.5–16.7% in each full month

Regenerate it with `python data/generate_sample_data.py`. The other files in `data/` and `tests/fixtures/` are also synthetic. Names and account numbers in them are invented.

## Legacy dashboard (`app_legacy.py`)

Rule-based categories (optional zero-shot model `cross-encoder/nli-MiniLM2-L6-H768`), Isolation Forest anomaly flags, Prophet/ARIMA/average forecast, and optional Gemini advice. Kept for reference. It is not part of the main journey.

## Known gaps

- `tests/test_classifier.py` targets an older classifier API and fails to import. Not fixed.
- Bill detection needs at least two months of history. A bill seen only once will be missed, so check the list.
- Salary paid on the last working day (not a fixed date) will drift by a day or two. Check the payday.

## Deploy (Streamlit Community Cloud)

Fork, then at [share.streamlit.io](https://share.streamlit.io) pick your fork, branch `main`, main file `app.py`.
