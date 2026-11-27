# ArthaSense

ArthaSense is a Streamlit prototype that reads an Indian bank statement (CSV, Excel or text-based PDF) and shows where the money went: spending by category, recurring payments, unusual transactions and a simple next-month estimate. Optional Gemini advice can be turned on with an API key.

It is a prototype, not a finished product. Version 3.0.

## How it works

| Step | File | What it does |
|------|------|--------------|
| Parse | `src/convert_to_csv.py`, `src/preprocessor.py`, `src/pdf_parser.py`, `src/excel_parser.py` | Reads CSV, Excel (.xlsx/.xls) and text-based PDF statements. Handles DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD and DD MMM YYYY dates, ₹ symbols and lakh-style amounts. Scanned PDFs are rejected. |
| Categorise | `src/classifier.py`, `src/merchant_categories.py` | **Rule-based by default**: merchant keyword rules, UPI handle extraction and person-name detection. An optional zero-shot mode (`cross-encoder/nli-MiniLM2-L6-H768` via `transformers`) can be switched on in Settings; it needs `transformers` and `torch` installed and downloads the model on first use. |
| Recurring payments | `src/recurring_detector.py` | Finds merchants charged in two or more months with a consistent amount. |
| Unusual transactions | `src/anomaly_detector.py` | scikit-learn Isolation Forest fitted on the statement's own debits. |
| Estimate | `src/forecaster.py` | Prophet if installed, otherwise ARIMA (statsmodels), otherwise a simple average. |
| Advice (optional) | `src/llm_advisor.py` | Gemini, only if switched on with a key. Otherwise a rule-based summary. Budgeting tips only; it is instructed not to recommend financial products. |
| UI | `app.py` | Streamlit entry point. |

## Where your data goes

- An uploaded file is read **in memory by the server running the app**. When you run it locally that is your own machine. On the hosted demo it is Streamlit Community Cloud's server, so the file does leave your device.
- The app does not write uploads to disk and does not keep them after the session ends. Log messages contain counts, not transaction text.
- If you turn on Gemini, the app sends Google's Gemini API a text summary: totals per category, summary statistics, forecasts, and up to 5 flagged transactions **with their descriptions and amounts**. Google's terms apply to that data.
- A server-side `GEMINI_API_KEY` is used on the server only. It is never placed in a page widget or sent to the browser. Users can paste their own key instead.

## Limits

- Maximum upload size: **10 MB** (`server.maxUploadSize` in `.streamlit/config.toml`, checked again in `app.py`).
- Scanned (image) PDFs are not supported.
- Category rules are tuned for Indian merchants and will miss some.

## Setup

```bash
git clone https://github.com/adityadeshmukh23/arthasense.git
cd arthasense
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`transformers`, `torch` and `prophet` are only needed for the optional zero-shot mode and the Prophet forecast. The app runs without them.

Optional Gemini key:

```bash
cp .env.example .env              # then set GEMINI_API_KEY in .env
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Sample data

`data/sample_transactions.csv` is **synthetic**. It describes a made-up salaried person (₹58,400 salary on the 1st, ₹16,000 rent, a phone EMI, four subscriptions, everyday spending), 21 May – 20 Nov 2026, saving roughly 12–17% a month. Regenerate it with:

```bash
python data/generate_sample_data.py
```

The other files in `data/` (`sample_hdfc_statement.csv`, `sample_icici_statement.csv`, `sample_sbi_statement.csv`) are also synthetic and exist to test bank-specific parsing. Names and account numbers in them are invented.

## Tests

```bash
pytest tests/ -v
```

## Deploy on Streamlit Community Cloud

1. Fork this repo.
2. At [share.streamlit.io](https://share.streamlit.io), choose your fork, branch `main`, main file `app.py`.
3. Optionally add `GEMINI_API_KEY = "..."` under **Advanced settings → Secrets**.
