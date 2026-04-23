# ArthaSense — AI-Powered Personal Finance Intelligence

ArthaSense is a production-ready AI-powered personal finance intelligence system designed for Indian bank statement analysis. Upload a CSV bank statement from any major Indian bank (SBI, HDFC, ICICI, Kotak) and get instant insights through a 4-stage ML pipeline: zero-shot NLP transaction categorization, Isolation Forest anomaly detection, Prophet time-series forecasting, and Gemini AI-powered financial advice.

The system processes raw bank transaction data and delivers a comprehensive Streamlit dashboard with categorized spending breakdowns, anomaly alerts for suspicious transactions, next-month spending forecasts per category, and a plain-English financial health report — all without requiring any labeled training data.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                    │
│  ┌──────────┬──────────┬──────────┬──────────┬────────┐ │
│  │Categories│Anomalies │ Forecast │AI Advice │ Table  │ │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────┘ │
│       │          │          │          │                  │
│  ┌────▼────┐┌────▼────┐┌───▼────┐┌────▼─────┐          │
│  │Zero-Shot││Isolation││Prophet ││ Gemini   │          │
│  │  NLP    ││ Forest  ││Forecast││  API     │          │
│  │(BART)   ││(sklearn)││        ││          │          │
│  └────┬────┘└────┬────┘└───┬────┘└────┬─────┘          │
│       └──────────┴─────────┴──────────┘                  │
│                        │                                  │
│              ┌─────────▼──────────┐                      │
│              │   Preprocessor     │                      │
│              │  (CSV Ingestion)   │                      │
│              └─────────┬──────────┘                      │
│                        │                                  │
│              ┌─────────▼──────────┐                      │
│              │  Bank Statement    │                      │
│              │   CSV Upload       │                      │
│              └────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## Live Demo

**[https://arthasense.streamlit.app](https://arthasense.streamlit.app)**

Upload any Indian bank statement CSV (SBI, HDFC, ICICI, Kotak) to try it instantly — no login required. Add your own [Gemini API key](https://aistudio.google.com/app/apikey) in the sidebar to enable AI advice.

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/adityadeshmukh23/arthasense.git
cd arthasense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CPU-only torch (saves ~2GB of disk space):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## API Key Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. (Optional) Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
   Get a free key at: https://aistudio.google.com/app/apikey

   The app works fully without a Gemini key — it falls back to rule-based analysis.

## Running the App

```bash
# Generate sample data (optional — already included)
python data/generate_sample_data.py

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Running Tests

```bash
pytest tests/ -v
```

## Module Descriptions

| Module | File | Description |
|--------|------|-------------|
| Preprocessor | `src/preprocessor.py` | CSV ingestion and cleaning for Indian bank formats. Handles multiple date formats, encodings, and amount styles (Indian lakh format). |
| Classifier | `src/classifier.py` | Zero-shot NLP categorization using facebook/bart-large-mnli. Maps transactions to 10 spending categories with keyword fallback. |
| Anomaly Detector | `src/anomaly_detector.py` | Isolation Forest anomaly detection trained on user spending patterns. Flags unusual amounts, duplicate charges, and unknown merchants. |
| Forecaster | `src/forecaster.py` | Prophet time-series forecasting for next-month spending per category. Falls back to ARIMA or simple average if needed. |
| LLM Advisor | `src/llm_advisor.py` | Gemini API integration for plain-English financial health reports. Falls back to rule-based analysis without an API key. |
| Dashboard | `app.py` | Main Streamlit entry point with 5-tab dashboard: Categories, Anomalies, Forecast, AI Advice, and Transactions. |

## Sample CSV Format

The app accepts standard Indian bank CSV exports with these columns:

```csv
Date,Description,Debit,Credit,Balance
01-01-2024,SALARY CREDIT,,50000.00,135000.00
02-01-2024,SWIGGY ORDER #7823,450.00,,134550.00
03-01-2024,UBER AUTO #1234,180.00,,134370.00
05-01-2024,NETFLIX INDIA,649.00,,133721.00
07-01-2024,ATM WITHDRAWAL,5000.00,,128721.00
```

Supported date formats: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, DD MMM YYYY

## Features

- **Zero-Shot Categorization**: No training data needed — uses BART-MNLI for instant classification
- **Anomaly Detection**: Flags unusual transactions based on your own spending patterns
- **Spending Forecast**: Predicts next month's spend per category with confidence intervals
- **AI Financial Advice**: Personalized insights from Google Gemini (optional)
- **Graceful Degradation**: Every component has a fallback — the app always works
- **Indian Format Support**: Handles ₹ symbols, lakh formatting, Indian merchant names
- **Privacy First**: All processing happens locally except optional Gemini API calls

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Select `adityadeshmukh23/arthasense`, branch `main`, main file `arthasense/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy** — the app will be live in ~5 minutes
