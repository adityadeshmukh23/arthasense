"""
ArthaSense — AI-Powered Finance Intelligence Dashboard.

Main Streamlit application that integrates preprocessing, NLP classification,
anomaly detection, forecasting, and LLM-based financial advice.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import html
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.preprocessor import BankStatementPreprocessor, format_indian_currency
from src.pdf_parser import ScannedPDFError, PDFParseError
from src.convert_to_csv import convert_to_arthasense_csv, to_pipeline_df
from src.classifier import TransactionClassifier
from src.anomaly_detector import AnomalyDetector
from src.forecaster import SpendingForecaster
from src.llm_advisor import LLMAdvisor, GEMINI_AVAILABLE
from src.recurring_detector import detect_recurring

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="ArthaSense — Finance Intelligence",
    page_icon="\u20b9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design System CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ================================================================
   ARTHASENSE — PREMIUM FINTECH DESIGN SYSTEM v3
   Deep navy · Electric blue · Emerald green
   Clean, data-forward, professional
================================================================ */

:root {
    --c-bg:        #0a0e1a;
    --c-surface:   #111827;
    --c-surface-2: #1a2235;
    --c-border:    rgba(255,255,255,0.06);
    --c-border-2:  rgba(255,255,255,0.10);
    --c-blue:      #2563eb;
    --c-blue-lt:   #3b82f6;
    --c-blue-dim:  rgba(37,99,235,0.12);
    --c-green:     #10b981;
    --c-green-dim: rgba(16,185,129,0.12);
    --c-red:       #ef4444;
    --c-red-dim:   rgba(239,68,68,0.10);
    --c-amber:     #f59e0b;
    --c-amber-dim: rgba(245,158,11,0.10);
    --c-text:      #f9fafb;
    --c-text-2:    #9ca3af;
    --c-text-3:    #6b7280;
    --c-mono:      'JetBrains Mono', 'Courier New', monospace;
    --c-sans:      'Inter', system-ui, sans-serif;
    --radius:      12px;
    --radius-sm:   8px;
    --shadow:      0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.25);
    /* backward-compat aliases used in Python-generated HTML */
    --c-purple:    #2563eb;
    --c-purple-lt: #3b82f6;
    --c-teal:      #10b981;
    --c-muted:     #6b7280;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: #0a0e1a !important;
    font-family: var(--c-sans) !important;
    color: var(--c-text) !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
    background: transparent !important;
    position: relative;
    z-index: 1;
}
#MainMenu, footer, header { visibility: hidden; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* ── SECTION ENTRANCE ANIMATIONS ─────────────────────────────── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.block-container > div > div:nth-child(1)   { animation: fadeSlideUp .55s .05s ease both; }
.block-container > div > div:nth-child(2)   { animation: fadeSlideUp .55s .12s ease both; }
.block-container > div > div:nth-child(3)   { animation: fadeSlideUp .55s .19s ease both; }
.block-container > div > div:nth-child(4)   { animation: fadeSlideUp .55s .26s ease both; }
.block-container > div > div:nth-child(5)   { animation: fadeSlideUp .55s .33s ease both; }
.block-container > div > div:nth-child(6)   { animation: fadeSlideUp .55s .40s ease both; }
.block-container > div > div:nth-child(n+7) { animation: fadeSlideUp .55s .46s ease both; }

/* ── SIDEBAR ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid var(--c-border-2) !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
    padding-top: 0.5rem;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
}

.as-wordmark {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    font-family: var(--c-sans);
    color: var(--c-text);
}
.as-wordmark-accent { color: var(--c-blue); }
.as-sidebar-divider {
    height: 1px;
    background: var(--c-border);
    margin: 1.25rem 0;
}

/* ── report header ───────────────────────────────────────────── */
.as-report-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding-bottom: 1rem;
    margin-bottom: 0;
}
.as-report-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--c-muted);
    margin-bottom: 0.3rem;
    font-family: var(--c-sans);
}
.as-report-date {
    font-size: 1.35rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: -0.02em;
    font-family: var(--c-sans);
}
.as-header-rule {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0 0 1.5rem 0;
}
.as-dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--c-blue);
    color: #fff;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 9px 18px;
    border-radius: var(--radius-sm);
    border: none;
    cursor: pointer;
    text-decoration: none;
    box-shadow: none;
    transition: background .2s, transform .15s;
}
.as-dl-btn:hover { background: #1d4ed8; transform: translateY(-1px); }

/* ── FINANCIAL NARRATIVE ─────────────────────────────────────── */
.as-story {
    padding: 1rem 1.25rem;
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    border-left: 3px solid var(--c-blue);
    border-radius: 0 var(--radius) var(--radius) 0;
    margin-bottom: 1.5rem;
}
.as-story-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.5rem;
    font-family: var(--c-sans);
}
.as-story-text {
    font-size: 14px;
    line-height: 1.7;
    color: var(--c-text-2);
    font-weight: 400;
}
.as-story-text + .as-story-text { margin-top: 0; }

/* ── KPI GRID ────────────────────────────────────────────────── */
.as-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.as-kpi {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: transform .2s ease, border-color .2s ease;
}
.as-kpi:hover {
    transform: translateY(-2px);
    border-color: var(--c-border-2);
}
/* Icon box inside KPI card */
.as-kpi-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    margin-bottom: 14px;
    flex-shrink: 0;
}
.as-kpi-icon.blue  { background: var(--c-blue-dim);  }
.as-kpi-icon.green { background: var(--c-green-dim); }
.as-kpi-icon.red   { background: var(--c-red-dim);   }
.as-kpi-icon.amber { background: var(--c-amber-dim); }
.as-kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.5rem;
    font-family: var(--c-sans);
}
.as-kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--c-text);
    margin-bottom: 0.4rem;
    font-family: var(--c-mono);
}
/* Colored delta badge */
.as-kpi-delta {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
}
.as-kpi-delta.pos { background: var(--c-green-dim); color: var(--c-green); }
.as-kpi-delta.neg { background: var(--c-red-dim);   color: var(--c-red);   }
.as-kpi-delta.neu { background: var(--c-blue-dim);  color: var(--c-blue-lt); }
/* legacy trend span (keep for fallback) */
.as-kpi-trend {
    font-size: 12px;
    font-weight: 500;
    color: var(--c-text-3);
    display: flex;
    align-items: center;
    gap: 4px;
}
.as-kpi-trend .up   { color: var(--c-green); }
.as-kpi-trend .down { color: var(--c-red);   }

/* ── FORECAST SUMMARY CARDS ──────────────────────────────────── */
.as-fc-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.as-fc-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow);
}
.as-fc-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 8px;
}
.as-fc-value {
    font-size: 1.7rem;
    font-weight: 700;
    font-family: var(--c-mono);
    color: var(--c-text);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    line-height: 1.1;
}
.as-fc-value.blue  { color: var(--c-blue-lt); }
.as-fc-value.green { color: var(--c-green);   }
.as-fc-value.red   { color: var(--c-red);     }
.as-fc-sub { font-size: 12px; color: var(--c-text-3); }

/* ── PANEL CARD ──────────────────────────────────────────────── */
.as-panel {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    padding: 1.25rem;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.as-panel-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.85rem;
    font-family: var(--c-sans);
}

/* ── SECTION HEADERS ─────────────────────────────────────────── */
.as-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.35rem;
}
.as-section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--c-text);
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.as-section-head {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-text-3);
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--c-border);
    margin-bottom: 0.9rem;
    margin-top: 1.5rem;
}

/* ── KEY INSIGHT CARDS ───────────────────────────────────────── */
.as-insights { display: flex; flex-direction: column; gap: 0.65rem; }
.as-insight-card {
    padding: 0.75rem 1rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    border-left: 3px solid transparent;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    transition: transform .2s;
}
.as-insight-card:hover { transform: translateX(2px); }
.as-insight-card.critical { background: var(--c-red-dim);   border-left-color: var(--c-red);   }
.as-insight-card.warning  { background: var(--c-amber-dim); border-left-color: var(--c-amber); }
.as-insight-card.positive { background: var(--c-green-dim); border-left-color: var(--c-green); }
.as-insight-card.neutral  { background: var(--c-blue-dim);  border-left-color: var(--c-blue);  }
.as-insight-title { font-size: 13px; font-weight: 500; color: var(--c-text); }
.as-insight-body  { font-size: 12px; color: var(--c-text-2); line-height: 1.4; }

/* ── CHIPS / BADGES ──────────────────────────────────────────── */
.as-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.4;
}
.as-chip-icon {
    flex-shrink: 0;
    width: 18px; height: 18px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem;
    font-weight: 800;
}
.as-chip-text {
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.as-chip.success { background: var(--c-green-dim); color: var(--c-green);  }
.as-chip.danger  { background: var(--c-red-dim);   color: var(--c-red);    }
.as-chip.primary { background: var(--c-blue-dim);  color: var(--c-blue-lt); }
.as-chip.warning { background: var(--c-amber-dim); color: var(--c-amber);  }
.as-chip.success .as-chip-icon { background: var(--c-green); color: #fff; }
.as-chip.danger  .as-chip-icon { background: var(--c-red);   color: #fff; }
.as-chip.primary .as-chip-icon { background: var(--c-blue);  color: #fff; }
.as-chip.warning .as-chip-icon { background: var(--c-amber); color: #fff; }

/* ── TABS — underline style matching design ──────────────────── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    border: none !important;
    border-bottom: 1px solid var(--c-border) !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 14px 18px !important;
    color: var(--c-text-3) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    background: transparent !important;
    transition: color .2s !important;
    font-family: var(--c-sans) !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--c-text-2) !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--c-blue-lt) !important;
    background: transparent !important;
    border-bottom-color: var(--c-blue) !important;
    box-shadow: none !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── ANOMALY ALERT CARDS ─────────────────────────────────────── */
.as-alert-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.as-alert-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-left: 3px solid var(--c-red);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
    transition: transform .2s ease;
}
.as-alert-card:hover { transform: translateY(-2px); }
.as-alert-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.65rem;
}
.as-anomaly-flag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-red);
}
.as-severity-pill {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.as-severity-pill.critical { background: var(--c-red-dim);   color: var(--c-red);   }
.as-severity-pill.high     { background: var(--c-red-dim);   color: var(--c-red);   }
.as-severity-pill.medium   { background: var(--c-amber-dim); color: var(--c-amber); }
.as-alert-merchant {
    font-size: 16px;
    font-weight: 600;
    color: var(--c-text);
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.as-alert-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.as-alert-meta span { font-size: 12px; color: var(--c-text-3); }
.as-alert-reason {
    font-size: 12px;
    color: var(--c-text-2);
    line-height: 1.5;
    margin-bottom: 0.75rem;
}
.as-alert-actions { display: flex; gap: 0.5rem; }
.as-ghost-btn {
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid var(--c-border-2);
    background: transparent;
    color: var(--c-text-2);
    cursor: pointer;
    font-family: var(--c-sans);
    transition: all .15s;
}
.as-ghost-btn:hover { border-color: rgba(255,255,255,0.25); color: var(--c-text); }

/* ── CLEAR / WARN BANNERS ────────────────────────────────────── */
.as-clear {
    background: var(--c-green-dim);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--c-green);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    font-weight: 500;
    color: var(--c-green);
}
.as-warn {
    background: var(--c-red-dim);
    border: 1px solid rgba(239,68,68,0.18);
    border-left: 3px solid var(--c-red);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.75rem 1rem;
    font-size: 13px;
    font-weight: 500;
    color: var(--c-red);
    margin-bottom: 1rem;
}

/* ── ADVISOR CARDS ───────────────────────────────────────────── */
.as-advisory-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.as-advisory-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow);
    transition: transform .2s ease;
}
.as-advisory-card:hover { transform: translateY(-2px); }
.as-advisory-card-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.4rem;
}
.as-advisory-card-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--c-text);
    font-family: var(--c-mono);
}
.as-advisory-card-sub { font-size: 12px; color: var(--c-text-3); margin-top: 0.3rem; }
.as-advisory-card.ac-savings  .as-advisory-card-value { color: var(--c-green);   }
.as-advisory-card.ac-risk     .as-advisory-card-value { color: var(--c-red);     }
.as-advisory-card.ac-forecast .as-advisory-card-value { color: var(--c-blue-lt); }

/* ── TIP CARDS ───────────────────────────────────────────────── */
.as-tip-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-left: 3px solid rgba(37,99,235,0.4);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow);
    transition: border-left-color .2s;
}
.as-tip-card:hover { border-left-color: var(--c-blue); }
.as-tip-headline { font-size: 14px; font-weight: 500; color: var(--c-text); margin-bottom: 0.2rem; }
.as-tip-body { font-size: 12px; color: var(--c-text-2); margin-bottom: 0.5rem; line-height: 1.5; }
.as-tip-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    float: right;
}
.as-tip-badge.save   { background: var(--c-green-dim); color: var(--c-green);  }
.as-tip-badge.risk   { background: var(--c-red-dim);   color: var(--c-red);    }
.as-tip-badge.invest { background: var(--c-blue-dim);  color: var(--c-blue-lt); }
.as-tip-badge.plan   { background: var(--c-amber-dim); color: var(--c-amber);  }

/* ── CHAT WRAPPER ────────────────────────────────────────────── */
.as-chat-wrap {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.as-chat-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--c-border);
    background: rgba(37,99,235,0.05);
}
.as-chat-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
    letter-spacing: 0.03em;
}
.as-chat-name { font-size: 14px; font-weight: 600; color: var(--c-text); line-height: 1.2; }
.as-chat-status {
    font-size: 11px;
    color: var(--c-green);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 4px;
}
.as-chat-body {
    padding: 1rem 1.25rem;
    font-size: 14px;
    line-height: 1.8;
    color: var(--c-text-2);
}
.as-chat-body p { margin-bottom: 0.75rem; }
.as-chat-body p:last-child { margin-bottom: 0; }
.as-chat-footer {
    padding: 0.5rem 1.25rem 0.65rem;
    font-size: 11px;
    color: var(--c-text-3);
    text-align: right;
    border-top: 1px solid var(--c-border);
}
.as-chat-info {
    padding: 0.6rem 1rem;
    background: rgba(37,99,235,0.05);
    border-bottom: 1px solid rgba(37,99,235,0.1);
    font-size: 13px;
    color: var(--c-text-2);
}
.as-status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--c-green);
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}

/* ── DOWNLOAD / PRIMARY / SECONDARY BUTTONS ─────────────────── */
[data-testid="stDownloadButton"] button {
    background: var(--c-blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    box-shadow: none !important;
    transition: background .2s, transform .15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stBaseButton-primary"] button,
.stButton button[kind="primary"] {
    background: var(--c-blue) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: #fff !important;
    box-shadow: none !important;
    transition: background .2s, transform .15s !important;
}
[data-testid="stBaseButton-primary"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stBaseButton-secondary"] button,
.stButton button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--c-border-2) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    color: var(--c-text-2) !important;
    transition: all .2s ease !important;
}
[data-testid="stBaseButton-secondary"] button:hover {
    background: var(--c-surface-2) !important;
    color: var(--c-text) !important;
}

/* ── METRIC CARDS ────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
    box-shadow: var(--shadow) !important;
}
div[data-testid="stMetric"] label {
    color: var(--c-text-3) !important;
    font-size: 11px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--c-text) !important;
    font-family: var(--c-mono) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--c-text-3) !important; }

/* ── STREAMLIT WIDGET OVERRIDES ──────────────────────────────── */
[data-testid="stTextInput"] input {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border-2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--c-text) !important;
    font-size: 14px !important;
    transition: border-color .2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--c-blue) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div > div {
    background: var(--c-surface) !important;
    border-color: var(--c-border-2) !important;
    color: var(--c-text) !important;
}
[data-testid="stSlider"] > div > div > div { background: var(--c-blue) !important; }
/* Expanders in main content keep card style */
[data-testid="stExpander"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
}
/* Sidebar expanders — accordion style (no card box) */
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] > div,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--c-border) !important;
    border-radius: 0 !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary,
[data-testid="stSidebar"] [data-testid="stExpander"] button[kind="header"] {
    background: transparent !important;
    padding: 9px 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--c-text-2) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] button[kind="header"]:hover {
    background: rgba(255,255,255,0.04) !important;
    color: var(--c-text) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
[data-testid="stSidebar"] [data-testid="stExpander"] button[kind="header"] svg {
    color: var(--c-text-3) !important;
}
/* Streamlit 1.35+ uses a div-based expander header */
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    color: var(--c-text-3) !important;
}
[data-testid="stSidebar"] details { background: transparent !important; }
[data-testid="stSidebar"] details summary { background: transparent !important; color: var(--c-text-2) !important; }

/* File uploader — dashed upload zone style */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(37,99,235,0.35) !important;
    border-radius: var(--radius-sm) !important;
    background: transparent !important;
    transition: border-color .2s, background .2s !important;
    padding: 4px 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--c-blue) !important;
    background: var(--c-blue-dim) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--c-text-2) !important;
    font-size: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--c-text-3) !important;
    font-size: 11px !important;
}
/* File uploader Browse button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: var(--c-surface-2) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploaderDropzone"] button span {
    color: var(--c-text) !important;
}

/* Sidebar footer */
.as-sidebar-footer {
    padding: 1rem 0.5rem 0.75rem;
    border-top: 1px solid var(--c-border);
    margin-top: 1rem;
}
.as-sidebar-footer p {
    font-size: 11px !important;
    color: var(--c-text-3) !important;
    line-height: 1.6;
}
.as-sidebar-footer span { color: var(--c-blue-lt) !important; font-weight: 500; }

/* ── WIDGET LABELS (radio, number, text, checkbox, select) ───── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: var(--c-text) !important;
}
[data-testid="stRadio"] label,
[data-testid="stRadio"] span[data-testid="stMarkdownContainer"] p {
    color: var(--c-text) !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] p {
    color: var(--c-text) !important;
}
[data-testid="stNumberInput"] input {
    background: var(--c-surface-2) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stNumberInput"] > div {
    background: var(--c-surface-2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stNumberInput"] button {
    background: var(--c-surface-2) !important;
    color: var(--c-text-2) !important;
    border: none !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--c-blue-dim) !important;
    color: var(--c-blue-lt) !important;
}
[data-testid="stTextInput"] input {
    background: var(--c-surface-2) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTextInput"] > div {
    background: var(--c-surface-2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span {
    color: var(--c-text) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--c-text) !important;
}

/* ── STATUS / SPINNER ────────────────────────────────────────── */
[data-testid="stStatus"],
[data-testid="stStatusWidget"],
[data-testid="stStatus"] > div,
[data-testid="stStatusWidget"] > div {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    color: var(--c-text) !important;
}
[data-testid="stStatus"] p,
[data-testid="stStatus"] span,
[data-testid="stStatus"] div,
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {
    color: var(--c-text) !important;
}
[data-testid="stStatus"] [data-testid="stMarkdownContainer"] p {
    color: var(--c-text-2) !important;
    font-size: 13px !important;
}

/* ── DATAFRAMES ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
}
[data-testid="stDataFrame"] table { font-size: 13px; }
[data-testid="stDataFrame"] th {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--c-text-3) !important;
    background: var(--c-surface-2) !important;
}

/* ── SPENDING PERSONALITY ────────────────────────────────────── */
.as-personality-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-left: 3px solid var(--c-green);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0;
    box-shadow: var(--shadow);
}
.as-personality-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-text-3);
    margin-bottom: 0.4rem;
}
.as-personality-type {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--c-text);
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.as-personality-fact {
    font-size: 13px;
    color: #9CA3AF;
}

/* ── HEALTH SCORE ────────────────────────────────────────────── */
@keyframes glowRed   { 0%,100%{text-shadow:0 0 16px rgba(239,68,68,.5)}  50%{text-shadow:0 0 32px rgba(239,68,68,.8)}  }
@keyframes glowAmber { 0%,100%{text-shadow:0 0 16px rgba(245,158,11,.5)} 50%{text-shadow:0 0 32px rgba(245,158,11,.8)} }
@keyframes glowGreen { 0%,100%{text-shadow:0 0 16px rgba(16,185,129,.5)} 50%{text-shadow:0 0 32px rgba(16,185,129,.8)} }

.as-health-score-num {
    font-size: 80px;
    font-weight: 800;
    color: var(--c-text);
    text-align: center;
    line-height: 1;
    letter-spacing: -0.04em;
    font-family: var(--c-mono);
}
.as-health-score-num.score-red   { color: var(--c-red);   animation: glowRed   2.5s ease-in-out infinite; }
.as-health-score-num.score-amber { color: var(--c-amber); animation: glowAmber 2.5s ease-in-out infinite; }
.as-health-score-num.score-green { color: var(--c-green); animation: glowGreen 2.5s ease-in-out infinite; }

.as-health-status {
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    letter-spacing: 0.04em;
    margin-top: 0.25rem;
    margin-bottom: 0.35rem;
}
.as-health-context {
    font-size: 13px;
    color: var(--c-text-2);
    text-align: center;
    line-height: 1.6;
    padding: 0 0.5rem;
    margin-bottom: 1rem;
}

/* ── STREAMLIT ALERT BANNER ──────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: var(--radius) !important;
}

/* ── EMPTY STATE ─────────────────────────────────────────────── */
.as-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 6rem 1rem 4rem;
    text-align: center;
}
.as-empty-line  { width: 32px; height: 1px; background: var(--c-border-2); margin-bottom: 2rem; }
.as-empty-title { font-size: 16px; font-weight: 500; color: var(--c-text); margin-bottom: 0.5rem; }
.as-empty-sub   { font-size: 13px; color: var(--c-text-3); margin-bottom: 2rem; }

/* ── PLANNER CARD ────────────────────────────────────────────── */
.as-planner-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    padding: 1.25rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}

/* ── TRANSACTIONS TAB ────────────────────────────────────────── */
.tx-toolbar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.tx-search-wrap {
    flex: 1;
    min-width: 180px;
    position: relative;
}
.tx-search-wrap input {
    width: 100%;
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius-sm);
    padding: 0.45rem 0.75rem 0.45rem 2rem;
    font-size: 13px;
    color: var(--c-text);
    outline: none;
    box-sizing: border-box;
}
.tx-search-wrap::before {
    content: "🔍";
    position: absolute;
    left: 0.55rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 11px;
    pointer-events: none;
}
.tx-filters { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.tx-filter-btn {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 12px;
    font-weight: 500;
    color: var(--c-text-2);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
}
.tx-filter-btn:hover { border-color: var(--c-blue); color: var(--c-blue-lt); }
.tx-filter-btn.active {
    background: var(--c-blue);
    border-color: var(--c-blue);
    color: #fff;
}
.tx-filter-btn.anom-active {
    background: var(--c-red-dim);
    border-color: var(--c-red);
    color: var(--c-red);
}
/* Streamlit buttons used as tab-style filter pills */
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease, transform 0.1s ease !important;
    padding: 0.3rem 1rem !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:active {
    transform: scale(0.96) !important;
    transition: transform 0.08s ease !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button[kind="primary"] {
    background: var(--c-blue) !important;
    border-color: var(--c-blue) !important;
    color: #fff !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button[kind="secondary"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    color: var(--c-text-2) !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: var(--c-blue) !important;
    color: var(--c-blue-lt) !important;
}
.tx-table-wrap {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    overflow-x: auto;
}
.tx-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    min-width: 640px;
}
.tx-table thead tr {
    background: var(--c-surface-2);
    border-bottom: 1px solid var(--c-border);
}
.tx-table thead th {
    padding: 0.65rem 1rem;
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--c-text-3);
    white-space: nowrap;
}
.tx-table thead th:last-child,
.tx-table tbody td:last-child { text-align: right; }
.tx-table tbody tr {
    border-bottom: 1px solid var(--c-border);
    transition: background 0.1s;
}
.tx-table tbody tr:last-child { border-bottom: none; }
.tx-table tbody tr:hover { background: rgba(255,255,255,0.03); }
.tx-table tbody td {
    padding: 0.7rem 1rem;
    color: var(--c-text);
    vertical-align: middle;
}
.tx-date { color: var(--c-text-3); font-size: 12px; white-space: nowrap; }
.tx-merchant { font-weight: 500; color: var(--c-text); }
.tx-merchant-sub { font-size: 11px; color: var(--c-text-3); margin-top: 1px; }
.tx-cat-pill {
    display: inline-block;
    background: var(--c-blue-dim);
    color: var(--c-blue-lt);
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
}
.tx-amount { font-weight: 600; white-space: nowrap; }
.tx-amount.debit  { color: var(--c-red); }
.tx-amount.credit { color: var(--c-green); }
.tx-anomaly-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--c-red);
    margin-right: 5px;
    vertical-align: middle;
    flex-shrink: 0;
}
.tx-row-anom { background: rgba(239,68,68,0.04) !important; }
.tx-empty {
    padding: 2.5rem;
    text-align: center;
    color: var(--c-text-3);
    font-size: 13px;
}
</style>

<script>
/* ===================================================================
   ArthaSense Live FX — KPI counter animation + health score glow
   Runs once; MutationObserver re-applies after every Streamlit render
=================================================================== */
(function () {
    'use strict';

    /* ── KPI COUNTER ANIMATION ────────────────────────────────── */
    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    function countUp(el, raw, ms) {
        if (el._asDone) return;
        el._asDone = true;
        var neg    = /[-−]/.test(raw);
        var prefix = raw.includes('₹') ? '₹' : '';
        var suffix = raw.includes('%') ? '%' : '';
        var num    = parseFloat(raw.replace(/[^0-9.]/g, '')) || 0;
        var t0     = performance.now();
        function tick(now) {
            var prog = Math.min((now - t0) / ms, 1);
            var cur  = Math.round(num * easeOut(prog));
            el.textContent = prefix + (neg ? '-' : '') + cur.toLocaleString('en-IN') + suffix;
            if (prog < 1) requestAnimationFrame(tick);
        }
        el.textContent = prefix + '0' + suffix;
        requestAnimationFrame(tick);
    }

    function applyCounters() {
        /* Custom .as-kpi-value elements */
        document.querySelectorAll('.as-kpi-value').forEach(function(el) {
            var raw = (el.dataset.raw || el.textContent || '').trim();
            if (!el.dataset.raw) el.dataset.raw = raw;
            if (raw && (raw.includes('₹') || /[0-9]{3}/.test(raw))) {
                el._asDone = false;
                countUp(el, raw, 1400);
            }
        });
        /* Native st.metric values */
        document.querySelectorAll('[data-testid="stMetricValue"]').forEach(function(el) {
            var raw = (el.dataset.raw || el.textContent || '').trim();
            if (!el.dataset.raw) el.dataset.raw = raw;
            if (raw && (raw.includes('₹') || /[0-9]{3}/.test(raw) || raw.includes('%'))) {
                el._asDone = false;
                countUp(el, raw, 1400);
                el.style.fontFamily = "'JetBrains Mono', monospace";
            }
        });
    }

    /* ── 3. HEALTH SCORE GLOW CLASS ───────────────────────────── */
    function applyScoreGlow() {
        document.querySelectorAll('.as-health-score-num').forEach(function(el) {
            var v = parseInt(el.textContent, 10);
            if (isNaN(v)) return;
            el.classList.remove('score-red', 'score-amber', 'score-green');
            if      (v < 40)  el.classList.add('score-red');
            else if (v < 70)  el.classList.add('score-amber');
            else              el.classList.add('score-green');
        });
    }

    /* ── MASTER INIT ──────────────────────────────────────────── */
    function init() {
        applyCounters();
        applyScoreGlow();
    }

    /* Re-fire on every Streamlit DOM update (debounced) */
    var _t;
    new MutationObserver(function() {
        clearTimeout(_t);
        _t = setTimeout(init, 280);
    }).observe(document.body, { childList: true, subtree: true });

    setTimeout(init, 800);
})();
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Colour palette and chart helpers
# ---------------------------------------------------------------------------
PALETTE = [
    "#2563eb", "#10b981", "#f59e0b", "#ef4444",
    "#0ea5e9", "#f97316", "#ec4899", "#84cc16",
    "#6366f1", "#3b82f6",
]


def _base_layout(title: str = "", height: int | None = None) -> dict:
    """Transparent, font-consistent Plotly layout base."""
    d: dict = dict(
        title=dict(text=title, font=dict(size=11, color="#6B7280",
                                          family="Inter, system-ui, sans-serif"),
                   x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=11,
                  color="#9CA3AF"),
        margin=dict(t=48, b=36, l=12, r=12, autoexpand=True),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
            borderwidth=1,
            font=dict(size=11, color="#9CA3AF"),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#6B7280"),
            automargin=True,
            linecolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#6B7280"),
            automargin=True,
            linecolor="rgba(255,255,255,0.06)",
        ),
    )
    if height:
        d["height"] = height
    return d


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.25rem;">
        <div class="as-wordmark"><span class="as-wordmark-accent">Artha</span>Sense</div>
        <div style="font-size:11.5px;color:var(--c-text-3);letter-spacing:0.08em;text-transform:uppercase;margin-top:6px;">Finance Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Statement",
        type=["csv", "pdf", "xlsx", "xls"],
        help="Any bank statement — PDF, Excel (.xlsx/.xls), or CSV. Any bank or format.",
        label_visibility="collapsed",
    )
    _MAX_UPLOAD_MB = 50
    if uploaded_file and uploaded_file.size > _MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large — maximum is {_MAX_UPLOAD_MB} MB.")
        st.stop()
    use_sample = st.button(
        "Load Sample Data",
        use_container_width=True,
        help="805-row synthetic dataset — Jan–Jun 2024.",
    )

    st.markdown('<div class="as-sidebar-divider" style="margin-top:1.25rem;"></div>',
                unsafe_allow_html=True)

    with st.expander("Settings", expanded=False):
        st.markdown('<div style="padding-top:0.25rem;"></div>', unsafe_allow_html=True)
        use_zero_shot = st.toggle(
            "Zero-shot NLP",
            value=False,
            help="Use DistilBART-MNLI for classification. More accurate than rule-based.",
        )
        if use_zero_shot:
            st.caption("First-run downloads ~600 MB model — takes 1–2 min.")
        enable_gemini = st.toggle(
            "Gemini AI Advisor",
            value=False,
            help="Requires a valid API key below.",
        )
        try:
            _gemini_default = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        except Exception:
            _gemini_default = os.getenv("GEMINI_API_KEY", "")
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=_gemini_default,
        )
    st.markdown('<div class="as-sidebar-divider"></div>', unsafe_allow_html=True)
    # ── Monthly Budget Limits ─────────────────────────────────────────────
    with st.expander("Budget Limits", expanded=False):

        # ── Step 1: compute actual per-category monthly averages ──────────
        # Read from session_state["df"] which is populated after the ML pipeline
        # runs. On the first load (no data yet) _budget_actuals stays empty.
        _budget_actuals: dict[str, int] = {}
        _budget_data_key = st.session_state.get("pipeline_key")  # unique per dataset+settings

        _bdf_raw = st.session_state.get("df")
        if _bdf_raw is not None and not _bdf_raw.empty:
            _bdf_deb = _bdf_raw[_bdf_raw["transaction_type"] == "debit"]
            if not _bdf_deb.empty and "category" in _bdf_deb.columns:
                _b_months = max(1, int(_bdf_deb["date"].dt.to_period("M").nunique()))
                for _bc, _bg in _bdf_deb.groupby("category"):
                    _budget_actuals[str(_bc)] = max(1, int(round(
                        float(_bg["debit"].sum()) / _b_months
                    )))

        # ── Step 2: seed widget states when a new dataset is first seen ───
        # We track which pipeline_key we last seeded for.  If it changes
        # (new file uploaded & analysed), we overwrite all budget_* widget
        # keys with the freshly computed averages and clear the old budgets
        # dict so stale categories from the previous file don't persist.
        # Once seeded, user edits are preserved because Streamlit manages
        # session_state[widget_key] as widget state across reruns.
        if _budget_actuals and st.session_state.get("_budgets_for_key") != _budget_data_key:
            _existing_budgets = st.session_state.get("budgets", {})
            _user_has_budgets = bool(_existing_budgets) and any(
                v > 0 for v in _existing_budgets.values()
            )
            if _user_has_budgets:
                st.warning("Re-uploading will clear your budget settings. Continue?")
                if not st.button("Yes, clear budgets and re-upload", key="confirm_budget_reset"):
                    st.stop()
            for _bc, _bavg in _budget_actuals.items():
                st.session_state[f"budget_{_bc}"] = _bavg
            st.session_state["_budgets_for_key"] = _budget_data_key
            st.session_state["budgets"] = {}   # clear stale categories from old file

        # ── Step 3: ensure the budgets dict exists ────────────────────────
        if "budgets" not in st.session_state:
            st.session_state["budgets"] = {}

        # ── Step 4: render inputs ─────────────────────────────────────────
        if not _budget_actuals:
            # No data loaded yet — show a short message and no inputs
            st.caption(
                "Upload and analyse a bank statement to auto-fill budget "
                "defaults from your actual monthly spending."
            )
            # Show inputs for the standard category list so the sidebar
            # is not completely empty before any file is loaded
            _fallback_cats = [
                "Food and Dining", "Shopping and Retail",
                "Utilities and Bills", "Transportation",
                "Entertainment and Subscriptions",
            ]
            for _bc in _fallback_cats:
                _bkey = f"budget_{_bc}"
                if _bkey not in st.session_state:
                    st.session_state[_bkey] = 0
                st.number_input(
                    _bc,
                    min_value=0,
                    step=100,
                    key=_bkey,
                    help="Enter monthly limit",
                )
                st.session_state["budgets"][_bc] = st.session_state[_bkey]
        else:
            st.caption(
                "Defaults = your actual monthly averages. "
                "Edit any value to set a custom limit."
            )
            # Render one input per category that appears in the real data,
            # sorted alphabetically for a consistent order.
            for _bc in sorted(_budget_actuals.keys()):
                _bavg = _budget_actuals[_bc]
                _bkey = f"budget_{_bc}"

                # If this widget key has never been set (e.g. first run after
                # seeding), set it now so Streamlit picks up the default.
                if _bkey not in st.session_state:
                    st.session_state[_bkey] = _bavg

                # Show the computed monthly average as a hint above the input
                st.caption(f"Avg monthly spend: {format_indian_currency(_bavg)}")
                st.number_input(
                    _bc,
                    min_value=0,
                    step=100,
                    key=_bkey,
                )
                # Sync the (possibly user-edited) widget value back to budgets
                st.session_state["budgets"][_bc] = st.session_state[_bkey]

    st.markdown('<div class="as-sidebar-divider"></div>', unsafe_allow_html=True)
    with st.expander("About", expanded=False):
        st.markdown(
            '<p style="font-size:11px;color:var(--c-text-3);line-height:1.7;padding:4px 8px;">'
            'ArthaSense — ML pipeline for Indian bank statement analysis. '
            '<span style="color:var(--c-blue-lt);">NLP</span> classification, '
            '<span style="color:var(--c-blue-lt);">Isolation Forest</span> anomaly detection, '
            '<span style="color:var(--c-blue-lt);">Prophet/ARIMA</span> forecasting, '
            '<span style="color:var(--c-blue-lt);">Gemini</span> advice.'
            "</p>",
            unsafe_allow_html=True,
        )

    # Sidebar footer
    _sf_df = st.session_state.get("df")
    _sf_source = st.session_state.get("file_source", "")
    if _sf_df is not None and not _sf_df.empty and "date" in _sf_df.columns:
        _sf_n = len(_sf_df)
        _sf_min = _sf_df["date"].min().strftime("%b %Y")
        _sf_max = _sf_df["date"].max().strftime("%b %Y")
        _is_sample = st.session_state.get("use_sample_flag", False)
        _sf_label = "Sample Dataset" if _is_sample else "Loaded Dataset"
        st.markdown(
            f'<div class="as-sidebar-footer"><p>v2.0 · {_sf_label}<br>'
            f'{_sf_min} – {_sf_max} · {_sf_n:,} transactions</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="as-sidebar-footer"><p>v2.0 · ArthaSense<br>'
            'Upload a statement to begin</p></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_CSV_PATH = Path(__file__).parent / "data" / "sample_transactions.csv"


def _html_table(df: "pd.DataFrame", right_cols: list = None) -> str:
    """Render a dataframe as a dark-themed .tx-table HTML string."""
    right_cols = right_cols or []
    cols = list(df.columns)
    rows = [
        '<div class="tx-table-wrap"><table class="tx-table"><thead><tr>'
    ]
    for c in cols:
        align = ' style="text-align:right"' if c in right_cols else ''
        rows.append(f'<th{align}>{html.escape(str(c))}</th>')
    rows.append('</tr></thead><tbody>')
    for _, row in df.iterrows():
        rows.append('<tr>')
        for c in cols:
            val = '' if (row[c] is None or (hasattr(row[c], '__float__') and pd.isna(row[c]))) else str(row[c])
            align = ' style="text-align:right"' if c in right_cols else ''
            rows.append(f'<td{align}>{html.escape(val)}</td>')
        rows.append('</tr>')
    rows.append('</tbody></table></div>')
    return ''.join(rows)


def _file_hash(f) -> str:
    return hashlib.md5(f.getvalue()).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def _get_classifier(use_zs: bool) -> TransactionClassifier:
    return TransactionClassifier(use_zero_shot=use_zs)


@st.cache_resource(show_spinner=False)
def _get_forecaster() -> SpendingForecaster:
    return SpendingForecaster()


@st.cache_resource(show_spinner=False)
def _get_anomaly_detector() -> AnomalyDetector:
    return AnomalyDetector(contamination=0.05)


# ---------------------------------------------------------------------------
# Intelligence helpers
# ---------------------------------------------------------------------------

# Named constants — replaces magic numbers in _compute_health_score
_HS = dict(
    SAVINGS_RATE_TARGET_PCT=20.0,
    SAVINGS_MAX_PTS=25.0,
    ANOMALY_PENALTY_FACTOR=8.0,
    DIVERSITY_TOP_CAT_CLIFF=40.0,
    DIVERSITY_MAX_PTS=25.0,
    RECURRING_MAX_PTS=10.0,
    ANOMALY_MAX_DEDUCT=20.0,
)


def _savings_rate(credits: float, debits: float) -> float:
    """Return savings rate as a percentage. Returns 0.0 if credits is zero."""
    return (credits - debits) / credits * 100 if credits else 0.0


def _compute_health_score(stats: dict, df: pd.DataFrame,
                           anomaly_summary: dict,
                           forecast_summary: pd.DataFrame) -> tuple[int, dict]:
    """
    0–100 composite financial health score.

    Returns
    -------
    (total_score, component_dict)
    component_dict keys: savings_pts, anomaly_pts, diversity_pts, trend_pts
    and improvement tips as a list of str under 'tips'.
    """
    components: dict = {}
    tips: list[str] = []

    # ── Savings Rate (0–25 pts) ───────────────────────────────────────────
    cr = stats.get("total_credits", 0)
    db = stats.get("total_debits", 0)
    sr_pct = _savings_rate(cr, db)
    savings_pts = min(_HS["SAVINGS_MAX_PTS"], max(0.0, sr_pct * 1.25)) if cr > 0 else 10.0
    components["savings_pts"] = int(round(savings_pts))
    if sr_pct < _HS["SAVINGS_RATE_TARGET_PCT"]:
        target_pts = min(_HS["SAVINGS_MAX_PTS"], int(round(_HS["SAVINGS_RATE_TARGET_PCT"] * 1.25)))
        gain = target_pts - components["savings_pts"]
        if gain > 0:
            tips.append(
                f"Increase savings rate from {sr_pct:.0f}% to "
                f"{_HS['SAVINGS_RATE_TARGET_PCT']:.0f}% → +{gain} points"
            )

    # ── Expense Diversity (0–25 pts) ─────────────────────────────────────
    n_cats = 0
    top_pct = 100.0
    if "category" in df.columns:
        deb = df[df["transaction_type"] == "debit"]
        if not deb.empty:
            n_cats = int(deb["category"].nunique())
            cat_totals = deb.groupby("category")["debit"].sum()
            top_pct = float(cat_totals.max() / cat_totals.sum() * 100) if cat_totals.sum() > 0 else 100.0
    top_pct = min(top_pct, 100.0)  # guard against floating-point overshoot
    diversity_pts = (
        min(_HS["DIVERSITY_MAX_PTS"], n_cats * 2.5)
        * (1 - max(0, (top_pct - _HS["DIVERSITY_TOP_CAT_CLIFF"]) / 100))
    )
    diversity_pts = max(0.0, diversity_pts)
    components["diversity_pts"] = int(round(diversity_pts))
    if top_pct > 50:
        tips.append(
            f"Reduce top category concentration from {top_pct:.0f}% "
            f"→ +{max(0, int(_HS['DIVERSITY_MAX_PTS']) - components['diversity_pts'])} points potential"
        )

    # ── Anomaly Rate (0–25 pts) ───────────────────────────────────────────
    n_a   = anomaly_summary.get("total_anomalies", 0)
    n_txn = max(1, stats.get("transaction_count", 1))
    anom_rate = n_a / n_txn
    anomaly_pts = max(0.0, _HS["SAVINGS_MAX_PTS"] - anom_rate * 100 * _HS["ANOMALY_PENALTY_FACTOR"])
    components["anomaly_pts"] = int(round(anomaly_pts))
    if n_a > 0:
        tips.append(
            f"Review {n_a} flagged anomal{'y' if n_a==1 else 'ies'} "
            f"→ up to +{int(_HS['SAVINGS_MAX_PTS']) - components['anomaly_pts']} points if resolved"
        )

    # ── Forecast Trend (0–25 pts) ─────────────────────────────────────────
    if not forecast_summary.empty and "vs_last_month_pct_change" in forecast_summary.columns:
        chg = float(forecast_summary["vs_last_month_pct_change"].mean())
        trend_pts = 25.0 if chg <= 0 else (18.0 if chg <= 10 else (10.0 if chg <= 20 else 4.0))
        if chg > 10:
            tips.append(
                f"Forecast shows spending trending up {chg:.0f}% "
                f"→ reduce to stabilise score"
            )
    else:
        trend_pts = 12.0
    components["trend_pts"] = int(round(trend_pts))

    total = int(min(100, max(0, round(
        savings_pts + diversity_pts + anomaly_pts + trend_pts
    ))))
    components["tips"] = tips
    return total, components


def _gauge_fig(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
                     "tickfont": {"size": 9, "color": "#6B7280"},
                     "tickcolor": "rgba(255,255,255,0.1)"},
            "bar": {"color": "#2563eb", "thickness": 0.28},
            "bgcolor": "#1a2235",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 100], "color": "#1a2235"},
            ],
            "threshold": {"line": {"color": "#2563eb", "width": 2},
                          "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=4, l=16, r=16), height=140,
        font={"family": "Inter, system-ui, sans-serif"},
    )
    return fig


def _key_insights(stats: dict, df: pd.DataFrame,
                  anomaly_summary: dict,
                  forecast_summary: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Returns list of (variant, icon-text, message) tuples — max 4."""
    out: list[tuple[str, str, str]] = []
    cr = stats.get("total_credits", 0)
    db = stats.get("total_debits", 0)

    # 1. Cashflow
    if cr > 0:
        sr = (cr - db) / cr * 100
        net = cr - db
        if sr >= 20:
            out.append(("success", "+",
                f"Savings rate {sr:.1f}% — surplus {format_indian_currency(net)}"))
        elif sr >= 0:
            out.append(("warning", "~",
                f"Savings rate {sr:.1f}% — financial planners recommend 20% or more"))
        else:
            out.append(("danger", "!",
                f"Deficit of {format_indian_currency(abs(net))} — spending exceeds income"))
    else:
        out.append(("primary", "i",
            f"Total outflows {format_indian_currency(db)} across "
            f"{stats.get('transaction_count', 0):,} transactions"))

    # 2. Top category
    if "category" in df.columns:
        deb = df[df["transaction_type"] == "debit"]
        if not deb.empty:
            cs = deb.groupby("category")["debit"].sum()
            tc, ta = cs.idxmax(), cs.max()
            pct = ta / cs.sum() * 100
            out.append(("primary" if pct <= 40 else "warning", "i",
                f"{tc} is top spend — {format_indian_currency(ta)} ({pct:.1f}%)"))

    # 3. Anomalies
    n = anomaly_summary.get("total_anomalies", 0)
    if n > 0:
        fa = anomaly_summary.get("total_flagged_amount", 0)
        out.append(("danger", "!",
            f"{n} unusual transaction{'s' if n != 1 else ''} — "
            f"{format_indian_currency(fa)} flagged"))
    else:
        out.append(("success", "+", "No anomalous activity detected"))

    # 4. Forecast
    if not forecast_summary.empty and "vs_last_month_pct_change" in forecast_summary.columns:
        chg = float(forecast_summary["vs_last_month_pct_change"].mean())
        if chg > 5:
            out.append(("warning", "~",
                f"Spend forecast rising {chg:.1f}% next month — tighten budget"))
        elif chg < -5:
            out.append(("success", "+",
                f"Spend forecast falling {abs(chg):.1f}% — positive trend"))

    return out[:4]


def _financial_story(stats: dict, df: pd.DataFrame,
                     anomaly_summary: dict,
                     forecast_summary: pd.DataFrame) -> list[str]:
    """2–3 plain-English sentences narrating the period."""
    cr = stats.get("total_credits", 0)
    db = stats.get("total_debits", 0)
    net = cr - db
    n_txns = stats.get("transaction_count", 0)
    dr = stats.get("date_range")

    period = ""
    if dr:
        days = max(1, (dr[1] - dr[0]).days)
        months = round(days / 30.4)
        period = f"Over {months} month{'s' if months != 1 else ''}"

    # S1 — cashflow picture
    if cr > 0:
        sr = _savings_rate(cr, db)
        direction = "saving" if net >= 0 else "overspending by"
        s1 = (f"{period}, you earned {format_indian_currency(cr)} and spent "
              f"{format_indian_currency(db)}, {direction} "
              f"{format_indian_currency(abs(net))} "
              f"({abs(sr):.0f}% {'savings rate' if net >= 0 else 'deficit'}) "
              f"across {n_txns:,} transactions.")
    else:
        s1 = (f"{period}, {n_txns:,} transactions totalling "
              f"{format_indian_currency(db)} in outflows were recorded.")

    # S2 — top two categories
    s2 = ""
    if "category" in df.columns:
        deb = df[df["transaction_type"] == "debit"]
        if not deb.empty:
            cs = deb.groupby("category")["debit"].sum().nlargest(2)
            cats = list(cs.index)
            amts = list(cs.values)
            if len(cats) >= 2:
                combined_pct = (amts[0] + amts[1]) / deb["debit"].sum() * 100
                s2 = (f"{cats[0]} ({format_indian_currency(amts[0])}) and "
                      f"{cats[1]} ({format_indian_currency(amts[1])}) "
                      f"together represent {combined_pct:.0f}% of total outflows.")
            elif cats:
                s2 = (f"{cats[0]} was the single largest category "
                      f"at {format_indian_currency(amts[0])}.")

    # S3 — anomaly or forecast
    n_anom = anomaly_summary.get("total_anomalies", 0)
    if n_anom > 0:
        s3 = (f"{n_anom} transaction{'s' if n_anom != 1 else ''} "
              f"{'were' if n_anom != 1 else 'was'} flagged as unusual — "
              "review the Anomalies tab for details.")
    elif not forecast_summary.empty and "next_month_forecast" in forecast_summary.columns:
        fc_total = float(forecast_summary["next_month_forecast"].sum())
        s3 = (f"Next month's projected spend is "
              f"{format_indian_currency(fc_total)} — plan accordingly.")
    else:
        s3 = "Spending patterns are consistent with no major irregularities detected."

    return [s for s in [s1, s2, s3] if s]


def _advisory_cards_html(stats: dict, anomaly_summary: dict,
                          forecast_summary: pd.DataFrame) -> str:
    """Three structured metric cards for the AI Advisor tab."""
    cr = stats.get("total_credits", 0)
    db = stats.get("total_debits", 0)
    n_anom = anomaly_summary.get("total_anomalies", 0)
    total_txns = max(1, stats.get("transaction_count", 1))

    # Savings rate
    if cr > 0:
        sr = _savings_rate(cr, db)
        sr_val = f"{sr:.1f}%"
        sr_sub = "Above target (20%)" if sr >= 20 else "Below target (20%)"
    else:
        sr_val, sr_sub = "N/A", "No income data"

    # Risk level
    anom_pct = n_anom / total_txns * 100
    if anom_pct == 0 and (cr == 0 or _savings_rate(cr, db) >= 20):
        risk_val, risk_sub = "Low", "No anomalies · savings healthy"
    elif anom_pct <= 2 and (cr == 0 or _savings_rate(cr, db) >= 0):
        risk_val, risk_sub = "Medium", f"{n_anom} anomal{'ies' if n_anom != 1 else 'y'} flagged"
    else:
        risk_val, risk_sub = "High", f"{n_anom} anomalies · review urgently"

    # Forecast
    if not forecast_summary.empty and "next_month_forecast" in forecast_summary.columns:
        fc = float(forecast_summary["next_month_forecast"].sum())
        fc_val = format_indian_currency(fc)
        fc_sub = "Projected next-month spend"
    else:
        fc_val, fc_sub = "N/A", "Insufficient data"

    return f"""
    <div class="as-advisory-grid">
        <div class="as-advisory-card ac-savings">
            <div class="as-advisory-card-label">Savings Rate</div>
            <div class="as-advisory-card-value">{sr_val}</div>
            <div class="as-advisory-card-sub">{sr_sub}</div>
        </div>
        <div class="as-advisory-card ac-risk">
            <div class="as-advisory-card-label">Risk Level</div>
            <div class="as-advisory-card-value">{risk_val}</div>
            <div class="as-advisory-card-sub">{risk_sub}</div>
        </div>
        <div class="as-advisory-card ac-forecast">
            <div class="as-advisory-card-label">Next Month</div>
            <div class="as-advisory-card-value">{fc_val}</div>
            <div class="as-advisory-card-sub">{fc_sub}</div>
        </div>
    </div>
    """


def _alert_severity(score: float) -> tuple[str, str]:
    if score >= 0.80:
        return "critical", "Critical"
    if score >= 0.72:
        return "high", "High"
    return "medium", "Medium"


# ---------------------------------------------------------------------------
# File-format label helper
# ---------------------------------------------------------------------------

def _fmt_label(source) -> str:
    ext = BankStatementPreprocessor._get_ext(source)
    return {"pdf": "PDF", ".xlsx": "Excel (.xlsx)", ".xls": "Excel (.xls)"}.get(
        ext, "CSV"
    )


# ---------------------------------------------------------------------------
# ML pipeline (accepts pre-parsed cleaned DataFrame)
# ---------------------------------------------------------------------------

def run_pipeline(df_clean: pd.DataFrame, cache_key: str, use_zs: bool) -> None:
    """Run NLP + anomaly + forecast on an already-parsed DataFrame."""
    settings_key = f"{cache_key}_{use_zs}"
    if st.session_state.get("pipeline_key") == settings_key:
        return

    preprocessor = BankStatementPreprocessor()
    stats = preprocessor.get_summary_stats(df_clean)

    with st.status("Analysing your statement…", expanded=True) as status:
        st.write("Classifying transactions…")
        classifier = _get_classifier(use_zs)
        df = classifier.classify_dataframe(df_clean.copy())

        st.write("Detecting anomalies…")
        detector = _get_anomaly_detector()
        detector.fit(df)
        df = detector.predict(df)
        anomaly_summary = detector.get_anomaly_summary(df)

        st.write("Building forecast…")
        forecaster = _get_forecaster()
        try:
            forecaster.fit_and_forecast(df)
        except Exception:
            logger.exception("Forecaster failed")
            st.warning("Forecast unavailable — see logs for details.")
        forecast_summary = forecaster.get_forecast_summary()
        total_forecast   = forecaster.get_total_forecast()

        status.update(label="Analysis complete", state="complete", expanded=False)

    st.session_state.update({
        "df": df, "stats": stats,
        "anomaly_summary":  anomaly_summary,
        "forecast_summary": forecast_summary,
        "total_forecast":   total_forecast,
        "forecaster":       forecaster,
        "pipeline_key":     settings_key,
    })
    if "category" in df.columns:
        st.session_state["_txn_cats"] = ["All"] + sorted(
            df["category"].dropna().unique().tolist()
        )


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
file_source: object = None
cache_key:   str    = None

if uploaded_file is not None:
    file_source = uploaded_file
    cache_key   = _file_hash(uploaded_file)
elif use_sample:
    if not SAMPLE_CSV_PATH.exists():
        st.error(f"Sample file not found: `{SAMPLE_CSV_PATH}`")
        st.stop()
    file_source = SAMPLE_CSV_PATH
    cache_key   = "sample"
    st.session_state["use_sample_flag"] = True
    st.session_state["analysis_confirmed"] = True   # no preview gate for sample
elif st.session_state.get("use_sample_flag"):
    file_source = SAMPLE_CSV_PATH
    cache_key   = "sample"

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
if file_source is None:
    st.markdown("""
    <div class="as-empty-state">
        <div class="as-empty-line"></div>
        <div class="as-empty-title">No financial data loaded</div>
        <div class="as-empty-sub">Upload a CSV or PDF bank statement, or load sample data to begin</div>
    </div>
    """, unsafe_allow_html=True)
    _empty_col = st.columns([1, 2, 1])[1]
    with _empty_col:
        if st.button("Load Sample Data", use_container_width=True, type="primary"):
            st.session_state["use_sample_flag"] = True
            st.session_state["analysis_confirmed"] = True
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Stage 1 — Parse file (fast, no ML)
# ---------------------------------------------------------------------------
preview_gate_key = f"preview_{cache_key}"

# Detect when a new file replaces the previous one → reset all state
if st.session_state.get("_current_file_key") != preview_gate_key:
    for _k in ("preview_df", "analysis_confirmed", "pipeline_key",
                "df", "stats", "anomaly_summary", "forecast_summary",
                "total_forecast", "forecaster"):
        st.session_state.pop(_k, None)
    st.session_state["_current_file_key"] = preview_gate_key

if "preview_df" not in st.session_state:
    _pp = BankStatementPreprocessor()
    _label = _fmt_label(file_source)

    # Determine file extension to pick the right ingestion path
    _ext = ""
    if hasattr(file_source, "name"):
        _ext = Path(file_source.name).suffix.lower()
    elif isinstance(file_source, (str, Path)):
        _ext = Path(file_source).suffix.lower()

    _use_converter = _ext in (".pdf", ".xlsx", ".xls", ".csv")

    _spinner_label = (
        "Detecting statement format and extracting transactions..."
        if _ext == ".pdf"
        else f"Parsing {_label} file..."
    )
    with st.spinner(_spinner_label):
        try:
            if _use_converter:
                # PDF / Excel → convert_to_arthasense_csv → to_pipeline_df
                _converted = convert_to_arthasense_csv(file_source)
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                _parsed = to_pipeline_df(_converted)

                # Compute summary for success message
                _n_txn    = len(_converted)
                _date_col = _converted["date"]
                _dr_start = _date_col.min().strftime("%d %b %Y")
                _dr_end   = _date_col.max().strftime("%d %b %Y")
                # Pick label: bank_source for PDF, file type for others
                if _ext == ".pdf" and "bank_source" in _converted.columns:
                    _src_label = str(_converted["bank_source"].iloc[0])
                elif _ext in (".xlsx", ".xls"):
                    _src_label = "Excel"
                elif _ext == ".csv":
                    _src_label = "CSV"
                else:
                    _src_label = _ext.lstrip(".").upper() if _ext else "File"
                st.session_state["conversion_message"] = (
                    f"**{_n_txn:,} transactions** extracted from "
                    f"**{_src_label}** statement "
                    f"({_dr_start} to {_dr_end})"
                )
            else:
                # CSV → existing preprocessor path (unchanged)
                _parsed = _pp.load_file(file_source)
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                st.session_state.pop("conversion_message", None)

            st.session_state["preview_df"] = _parsed

        except ScannedPDFError as _e:
            st.error(
                f"**Scanned PDF detected.** {_e}\n\n"
                "Please download a text-based statement from your bank's "
                "internet-banking portal and try again."
            )
            st.stop()
        except (PDFParseError, ValueError) as _e:
            st.markdown(f"""
            <div class="as-panel" style="border-left:3px solid #ef4444;padding:1.2rem 1.5rem;margin:1rem 0;">
                <div style="color:#ef4444;font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Could not extract transactions</div>
                <div style="color:#D1D5DB;font-size:0.85rem;line-height:1.6;">{_e}</div>
                <div style="color:#6B7280;font-size:0.8rem;margin-top:0.75rem;">
                    <strong style="color:#9CA3AF;">Supported file types:</strong>
                    PDF (any text-based bank statement) &middot; Excel (.xlsx / .xls) &middot; CSV
                </div>
                <div style="color:#6B7280;font-size:0.8rem;margin-top:0.4rem;">
                    If your PDF is a scanned image, download the statement as <strong>CSV or Excel</strong> from your bank&apos;s internet-banking portal and upload that instead.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        except ImportError as _e:
            st.error(
                f"**Missing dependency:** {_e}\n\n"
                "Run `pip install pdfplumber openpyxl` and restart the app."
            )
            st.stop()
        except Exception as _e:
            st.error(f"**Unexpected error during file parsing:** {_e}")
            st.stop()

# ---------------------------------------------------------------------------
# Stage 2 — Preview / confirm gate (skipped for sample data)
# ---------------------------------------------------------------------------

# Show conversion success banner for PDF / Excel uploads
if _conv_msg := st.session_state.get("conversion_message"):
    st.success(_conv_msg)

if not st.session_state.get("analysis_confirmed"):
    _pv   = st.session_state["preview_df"]
    _n    = len(_pv)
    _lbl  = _fmt_label(file_source)
    _dr   = (
        f"{_pv['date'].min().strftime('%d %b %Y')} — {_pv['date'].max().strftime('%d %b %Y')}"
        if "date" in _pv.columns and not _pv["date"].isna().all()
        else "date range unknown"
    )

    st.markdown(f"""
    <div class="as-story">
        <div class="as-story-label">File Preview &mdash; confirm before running analysis</div>
        <div class="as-story-text">
            <strong>{_lbl}</strong> parsed successfully &mdash;
            <strong>{_n:,} transactions</strong> detected &middot; {_dr}
        </div>
        <div class="as-story-text" style="opacity:0.6;font-size:0.8rem;margin-top:0.3rem;">
            Review the first rows below to confirm the parser read your file correctly,
            then click <strong>Confirm &amp; Run Analysis</strong> to start the ML pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show first 12 rows of the parsed, standardised DataFrame
    _preview_cols = ["date", "description", "debit", "credit",
                     "balance", "transaction_type"]
    _show_cols = [c for c in _preview_cols if c in _pv.columns]
    _disp = _pv[_show_cols].head(12).copy()
    if "date" in _disp.columns:
        _disp["date"] = _disp["date"].dt.strftime("%d %b %Y")
    for _nc in ["debit", "credit", "balance"]:
        if _nc in _disp.columns:
            _disp[_nc] = _disp[_nc].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) and x != 0 else ""
            )
    st.markdown(_html_table(_disp), unsafe_allow_html=True)

    _col_confirm, _col_cancel = st.columns([1, 4])
    with _col_confirm:
        if st.button("Confirm & Run Analysis", type="primary", use_container_width=True):
            st.session_state["analysis_confirmed"] = True
            st.rerun()
    with _col_cancel:
        if st.button("Upload a Different File", use_container_width=True):
            for _k in ("preview_df", "analysis_confirmed", "_current_file_key",
                       "use_sample_flag", "pipeline_key"):
                st.session_state.pop(_k, None)
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Stage 3 — Run ML pipeline on confirmed parsed DataFrame
# ---------------------------------------------------------------------------
_preview_df: pd.DataFrame = st.session_state["preview_df"]

try:
    run_pipeline(_preview_df, cache_key, use_zero_shot)
except Exception as exc:
    st.error(f"Processing failed: {exc}")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
stats: dict       = st.session_state["stats"]
anomaly_summary   = st.session_state["anomaly_summary"]
forecast_summary  = st.session_state["forecast_summary"]
total_forecast    = st.session_state["total_forecast"]

health_score, health_components = _compute_health_score(
    stats, df, anomaly_summary, forecast_summary
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
dr = stats.get("date_range")
_dr_str = ""
if dr:
    _dr_str = f"{dr[0].strftime('%d %b %Y')} &ndash; {dr[1].strftime('%d %b %Y')}"

st.markdown(f"""
<div class="as-report-header">
    <div>
        <div class="as-report-label">Financial Report</div>
        <div class="as-report-date">{_dr_str}</div>
    </div>
</div>
<div class="as-header-rule"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Financial Narrative
# ---------------------------------------------------------------------------
story_lines = _financial_story(stats, df, anomaly_summary, forecast_summary)
story_html = " ".join(story_lines)
st.markdown(
    '<div class="as-section-label">Narrative</div>'
    f'<div class="as-story"><div class="as-story-text">{story_html}</div></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
total_debits  = stats.get("total_debits", 0)
total_credits = stats.get("total_credits", 0)
net_cashflow  = stats.get("net_cashflow", 0)
txn_count     = stats.get("transaction_count", 0)
avg_monthly   = stats.get("avg_monthly_spend", 0)

# Compute MoM trend for context
def _mom_trend(df: pd.DataFrame, col: str, txn_type: str) -> str:
    """Return 'vs last period' string with direction arrow."""
    _months = sorted(df[df["transaction_type"] == txn_type]["date"].dt.to_period("M").unique())
    if len(_months) < 2:
        return f"Avg {format_indian_currency(avg_monthly)}/mo"
    _prev = float(df[(df["transaction_type"] == txn_type) &
                     (df["date"].dt.to_period("M") == _months[-2])][col].sum())
    _curr = float(df[(df["transaction_type"] == txn_type) &
                     (df["date"].dt.to_period("M") == _months[-1])][col].sum())
    if _prev == 0:
        return "No prior month"
    _chg = (_curr - _prev) / _prev * 100
    _arrow = "&#9650;" if _chg >= 0 else "&#9660;"
    _cls = "up" if _chg >= 0 else "down"
    _sign = "+" if _chg >= 0 else ""
    return f'<span class="{_cls}">{_arrow} {_sign}{_chg:.1f}%</span> vs last month'

_spend_trend  = _mom_trend(df, "debit", "debit")
_income_trend = _mom_trend(df, "credit", "credit")
_net_sign = "+" if net_cashflow >= 0 else ""
_net_color = "#10b981" if net_cashflow >= 0 else "#ef4444"
_n_months = max(1, df["date"].dt.to_period("M").nunique())

_sr_delta_cls = "pos" if _savings_rate(total_credits, total_debits) >= 20 else "neu"
_sr_pct_str   = f"{_savings_rate(total_credits, total_debits):.1f}% rate"
st.markdown(f"""
<div class="as-kpi-grid">
    <div class="as-kpi">
        <div class="as-kpi-icon blue">&#8593;</div>
        <div class="as-kpi-label">Total Credits</div>
        <div class="as-kpi-value">{format_indian_currency(total_credits)}</div>
        <div class="as-kpi-trend">{_income_trend}</div>
    </div>
    <div class="as-kpi">
        <div class="as-kpi-icon red">&#8595;</div>
        <div class="as-kpi-label">Total Debits</div>
        <div class="as-kpi-value">{format_indian_currency(total_debits)}</div>
        <div class="as-kpi-trend">{_spend_trend}</div>
    </div>
    <div class="as-kpi">
        <div class="as-kpi-icon green">&#8801;</div>
        <div class="as-kpi-label">Net Savings</div>
        <div class="as-kpi-value" style="color:{_net_color};">{_net_sign}{format_indian_currency(net_cashflow)}</div>
        <span class="as-kpi-delta {_sr_delta_cls}">{_sr_pct_str}</span>
    </div>
    <div class="as-kpi">
        <div class="as-kpi-icon amber">&#9889;</div>
        <div class="as-kpi-label">Transactions</div>
        <div class="as-kpi-value">{txn_count:,}</div>
        <span class="as-kpi-delta neu">~{format_indian_currency(avg_monthly)}/month</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Health Score + Key Insights
# ---------------------------------------------------------------------------
insights = _key_insights(stats, df, anomaly_summary, forecast_summary)

# Compute savings rate for dynamic label
_hs_cr = stats.get("total_credits", 0)
_hs_db = stats.get("total_debits", 0)
_hs_sr = ((_hs_cr - _hs_db) / _hs_cr * 100) if _hs_cr > 0 else 0.0

# Dynamic status label logic
if _hs_sr < 0 and health_score < 80:
    _hs_status_label = "Structurally At Risk"
    _hs_status_color = "#ef4444"
    _hs_context = "Spending outpaces income consistently. Structural intervention needed."
elif _hs_sr < 0 and health_score >= 80:
    _hs_status_label = "Strong Portfolio, Critical Cash Gap"
    _hs_status_color = "#F59E0B"
    _hs_context = "Diversification and investments are healthy but monthly cash flow is negative."
elif health_score >= 85:
    _hs_status_label = "Financially Resilient"
    _hs_status_color = "#10b981"
    _hs_context = "Savings rate, spending diversity, and transaction patterns are all sound."
elif health_score >= 65:
    _hs_status_label = "On Track"
    _hs_status_color = "#10b981"
    _hs_context = "Good fundamentals with room to improve savings rate or reduce anomalies."
else:
    _hs_status_label = "Needs Attention"
    _hs_status_color = "#F59E0B"
    _hs_context = "Some spending or anomaly patterns require review to improve your score."

col_g, col_i = st.columns([1, 2], gap="medium")

with col_g:
    st.markdown('<div class="as-panel-label">Financial Health Score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="as-panel">'
        f'<div class="as-health-score-num">{health_score}</div>'
        f'<div class="as-health-status" style="color:{_hs_status_color};">{_hs_status_label}</div>'
        f'<div class="as-health-context">{_hs_context}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _gauge_fig(health_score),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Score breakdown bars
    _breakdown_items = [
        ("Savings Rate",      health_components["savings_pts"],  25),
        ("Expense Diversity", health_components["diversity_pts"], 25),
        ("Anomaly Rate",      health_components["anomaly_pts"],   25),
        ("Forecast Trend",    health_components["trend_pts"],     25),
    ]
    _bd_html = ['<div style="padding:0 0 0.5rem;">']
    for _lbl, _pts, _max in _breakdown_items:
        _fill = _pts / _max * 100
        _bar_c = "#10b981" if _fill >= 70 else ("#f59e0b" if _fill >= 40 else "#ef4444")
        _bd_html.append(
            f'<div style="display:flex;align-items:center;gap:0.5rem;'
            f'margin-bottom:0.5rem;font-size:12px;">'
            f'<span style="width:110px;color:#6B7280;">{_lbl}</span>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);'
            f'border-radius:2px;height:4px;">'
            f'<div style="width:{_fill:.0f}%;background:{_bar_c};'
            f'height:100%;border-radius:2px;"></div></div>'
            f'<span style="width:38px;text-align:right;font-weight:600;color:{_bar_c};">'
            f'{_pts}/{_max}</span>'
            f'</div>'
        )
    _bd_html.append('</div>')

    # Improvement tips
    _tips = health_components.get("tips", [])
    if _tips:
        _bd_html.append(
            '<div style="padding-top:0.75rem;border-top:1px solid rgba(255,255,255,0.06);">'
            '<div class="as-section-label" style="margin-bottom:0.5rem;">Improvement Tips</div>'
        )
        for _tip in _tips:
            _bd_html.append(
                f'<div style="font-size:13px;padding:0.3rem 0;color:#9CA3AF;">'
                f'&rsaquo; {_tip}</div>'
            )
        _bd_html.append('</div>')

    st.markdown("".join(_bd_html), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_i:
    # Map _key_insights variants to new card classes
    _variant_map = {"success": "positive", "danger": "critical",
                    "warning": "warning", "primary": "neutral"}
    insights_html = ['<div class="as-panel-label">Key Insights</div>',
                     '<div class="as-insights">']
    for variant, _icon, text in insights:
        _card_cls = _variant_map.get(variant, "neutral")
        # Split text at first " — " into title/body if present
        if " — " in text:
            _title, _body = text.split(" — ", 1)
        else:
            _title, _body = text, ""
        insights_html.append(
            f'<div class="as-insight-card {_card_cls}">'
            f'<div class="as-insight-title">{_title}</div>'
            + (f'<div class="as-insight-body">{_body}</div>' if _body else '')
            + '</div>'
        )
    insights_html.append('</div>')
    st.markdown("".join(insights_html), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Spending Personality card
# ---------------------------------------------------------------------------
def _spending_personality(df: pd.DataFrame) -> tuple[str, str]:
    """Return (label, fact_sentence) using rule-based logic."""
    deb = df[df["transaction_type"] == "debit"].copy()
    if deb.empty or "category" not in deb.columns:
        return "Diversified Spender", "No spending pattern data available."

    # Top category
    cat_totals = deb.groupby("category")["debit"].sum()
    top_cat = cat_totals.idxmax()
    top_cat_pct = cat_totals.max() / cat_totals.sum() * 100

    # Weekend spending
    deb["_dow"] = deb["date"].dt.dayofweek  # 0=Mon, 5=Sat, 6=Sun
    total_spend = deb["debit"].sum()
    weekend_spend = deb[deb["_dow"].isin([5, 6])]["debit"].sum()
    weekend_pct = (weekend_spend / total_spend * 100) if total_spend > 0 else 0

    # Peak day of week
    dow_spend = deb.groupby("_dow")["debit"].sum()
    peak_dow = int(dow_spend.idxmax())
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    peak_day_name = dow_names[peak_dow]

    # Single merchant dominance
    if "description" in deb.columns:
        merch_totals = deb.groupby("description")["debit"].sum()
        if not merch_totals.empty and len(cat_totals) > 0:
            top_cat_deb = deb[deb["category"] == top_cat]
            top_merch_in_cat = top_cat_deb.groupby("description")["debit"].sum()
            cat_amount = cat_totals.max()
            single_dominates = (
                not top_merch_in_cat.empty and
                top_merch_in_cat.max() / cat_amount > 0.55
            )
        else:
            single_dominates = False
    else:
        single_dominates = False

    # Rules
    _top = top_cat.lower() if top_cat else ""
    if "food" in _top and weekend_pct > 60:
        label = "Weekend Diner"
        fact = f"{weekend_pct:.0f}% of your Food spending occurs on weekends."
    elif "shopping" in _top and single_dominates:
        label = "Brand Loyalist"
        fact = f"{top_cat} accounts for {top_cat_pct:.0f}% of total spend, concentrated on one merchant."
    elif "entertainment" in _top or "subscription" in _top:
        label = "Experience Spender"
        fact = f"{top_cat} is your largest category at {top_cat_pct:.0f}% of total outflows."
    elif "travel" in _top or "transport" in _top:
        label = "Frequent Traveler"
        fact = f"You spend most on {top_cat} — {top_cat_pct:.0f}% of all outflows."
    elif top_cat_pct < 30:
        label = "Diversified Spender"
        fact = f"No single category exceeds 30% of spend. Highest is {top_cat} at {top_cat_pct:.0f}%."
    else:
        label = "Focused Spender"
        fact = f"{top_cat} dominates at {top_cat_pct:.0f}% of spend, peaking on {peak_day_name}s."

    return label, fact

_personality_label, _personality_fact = _spending_personality(df)
st.markdown(
    '<div class="as-section-label">Spending Personality</div>'
    f'<div class="as-personality-card">'
    f'<div class="as-personality-type">{_personality_label}</div>'
    f'<div class="as-personality-fact">{_personality_fact}</div>'
    f'</div>'
    '<div style="margin-bottom:1.5rem;"></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs  (no emojis — clean uppercase labels)
# ---------------------------------------------------------------------------
tab_cat, tab_anom, tab_fc, tab_ai, tab_txn, tab_plan = st.tabs(
    ["Categories", "Anomalies", "Forecast", "Advisor", "Transactions", "Planner"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Categories
# ═══════════════════════════════════════════════════════════════════════════
with tab_cat:
    debit_df = df[df["transaction_type"] == "debit"].copy()

    if debit_df.empty:
        st.info("No debit transactions to categorise.")
    else:
        cat_spend = (
            debit_df.groupby("category")["debit"].sum()
            .sort_values(ascending=False)
            .reset_index(name="total_spent")
        )

        # Top 8 + Others bucket
        if len(cat_spend) > 8:
            pie_data = pd.concat([
                cat_spend.head(8),
                pd.DataFrame([{"category": "Others",
                               "total_spent": cat_spend.iloc[8:]["total_spent"].sum()}])
            ], ignore_index=True)
        else:
            pie_data = cat_spend.copy()

        col_pie, col_bar = st.columns(2, gap="medium")

        with col_pie:
            fig_pie = px.pie(
                pie_data, names="category", values="total_spent",
                color_discrete_sequence=PALETTE, hole=0.44,
            )
            fig_pie.update_traces(
                textinfo="percent",
                textposition="inside",
                textfont_size=10,
                marker=dict(line=dict(color="rgba(0,0,0,0.06)", width=1)),
                hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
            )
            fig_pie.update_layout(**_base_layout("Spending by Category"))
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="v", x=1.01, y=0.5,
                            font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={"displayModeBar": False})

        with col_bar:
            top3 = cat_spend.head(3)["category"].tolist()
            bar_src = debit_df[debit_df["category"].isin(top3)].copy()
            if not bar_src.empty:
                monthly_cat = (
                    bar_src.groupby(["month_year", "category"])["debit"]
                    .sum().reset_index()
                )
                mo = bar_src.sort_values("date")["month_year"].unique().tolist()
                monthly_cat["month_year"] = pd.Categorical(
                    monthly_cat["month_year"], categories=mo, ordered=True
                )
                monthly_cat.sort_values("month_year", inplace=True)

                fig_bar = px.bar(
                    monthly_cat, x="month_year", y="debit", color="category",
                    barmode="stack", color_discrete_sequence=PALETTE,
                    labels={"debit": "Amount (\u20b9)", "month_year": "Month"},
                )
                fig_bar.update_traces(
                    marker_line_width=0,
                    hovertemplate="%{fullData.name}<br>\u20b9%{y:,.0f}<extra></extra>",
                )
                fig_bar.update_layout(**_base_layout("Monthly Trend — Top 3 Categories"))
                fig_bar.update_layout(
                    legend=dict(
                        orientation="h", y=-0.22, x=0, xanchor="left",
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown('<div class="as-section-head">Category Breakdown</div>',
                    unsafe_allow_html=True)
        total_s = cat_spend["total_spent"].sum()
        disp = cat_spend.copy()
        disp["pct"] = (disp["total_spent"] / total_s * 100).round(1)
        disp["txn_count"] = disp["category"].map(
            debit_df.groupby("category").size())
        disp["avg_txn"] = (disp["total_spent"] / disp["txn_count"]).round(2)
        disp["total_spent"] = disp["total_spent"].apply(format_indian_currency)
        disp["avg_txn"]     = disp["avg_txn"].apply(format_indian_currency)
        disp["pct"]         = disp["pct"].astype(str) + "%"
        disp.columns        = ["Category", "Total Spent", "% of Total",
                                "Transactions", "Avg Transaction"]
        st.markdown(_html_table(disp, right_cols=["Total Spent", "% of Total", "Transactions", "Avg Transaction"]),
                    unsafe_allow_html=True)

        # ── Budget vs Actual ──────────────────────────────────────────────
        st.markdown('<div class="as-section-head">Budget vs Actual</div>',
                    unsafe_allow_html=True)
        budgets = st.session_state.get("budgets", {})
        if not budgets:
            st.caption("Set monthly budget limits in the sidebar to see progress bars.")
        else:
            # Determine "current month" as the most recent month in the data
            _cur_month = df["date"].max().to_period("M") if not df.empty else None
            _month_debits = (
                df[(df["transaction_type"] == "debit") &
                   (df["date"].dt.to_period("M") == _cur_month)]
                .groupby("category")["debit"].sum()
                if _cur_month is not None else pd.Series(dtype=float)
            )

            for cat, budget in budgets.items():
                if budget <= 0:
                    continue
                spent = float(_month_debits.get(cat, 0.0))
                pct   = spent / budget if budget > 0 else 0
                if spent == 0 and cat not in cat_spend["category"].values:
                    continue  # skip categories with no spend at all

                # Color thresholds
                if pct < 0.70:
                    bar_color = "#10b981"
                    label_cls = "success"
                elif pct <= 1.0:
                    bar_color = "#F59E0B"
                    label_cls = "warning"
                else:
                    bar_color = "#ef4444"
                    label_cls = "danger"

                fill_pct  = min(pct * 100, 100)
                pct_label = f"{pct * 100:.0f}%"
                st.markdown(
                    f'<div style="margin-bottom:0.85rem;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.82rem;margin-bottom:0.25rem;">'
                    f'<span>{html.escape(cat)}</span>'
                    f'<span class="as-chip {label_cls}" style="padding:0.1rem 0.5rem;">'
                    f'{format_indian_currency(spent)} of {format_indian_currency(budget)} '
                    f'({pct_label})</span></div>'
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;'
                    f'height:8px;overflow:hidden;">'
                    f'<div style="width:{fill_pct:.1f}%;background:{bar_color};'
                    f'height:100%;border-radius:4px;transition:width 0.4s;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Month-over-Month Change ───────────────────────────────────────
        st.markdown('<div class="as-section-head">Month-over-Month Change</div>',
                    unsafe_allow_html=True)
        _all_months = sorted(
            df[df["transaction_type"] == "debit"]["date"]
            .dt.to_period("M").unique()
        )
        if len(_all_months) < 2:
            st.caption("Requires at least 2 months of data.")
        else:
            _prev_m = _all_months[-2]
            _curr_m = _all_months[-1]

            def _month_cat_spend(period):
                return (
                    df[(df["transaction_type"] == "debit") &
                       (df["date"].dt.to_period("M") == period)]
                    .groupby("category")["debit"].sum()
                )

            _prev_spend = _month_cat_spend(_prev_m)
            _curr_spend = _month_cat_spend(_curr_m)
            _all_cats   = sorted(set(_prev_spend.index) | set(_curr_spend.index))

            mom_rows = []
            for _cat in _all_cats:
                _p = float(_prev_spend.get(_cat, 0.0))
                _c = float(_curr_spend.get(_cat, 0.0))
                if _p == 0 and _c == 0:
                    continue
                _chg = ((_c - _p) / _p * 100) if _p > 0 else (100.0 if _c > 0 else 0.0)
                mom_rows.append({"category": _cat, "prev": _p, "curr": _c, "pct": _chg})

            mom_rows.sort(key=lambda r: -abs(r["pct"]))

            if mom_rows:
                # Highlight biggest increase
                _biggest = max(mom_rows, key=lambda r: r["pct"])
                if _biggest["pct"] > 10:
                    st.markdown(
                        f'<div class="as-chip danger" style="margin-bottom:0.75rem;">'
                        f'Biggest increase: <strong>{html.escape(_biggest["category"])}</strong> '
                        f'up {_biggest["pct"]:.0f}% vs last month '
                        f'({format_indian_currency(_biggest["prev"])} '
                        f'→ {format_indian_currency(_biggest["curr"])})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                _mom_html = [
                    f'<div style="font-size:0.75rem;color:var(--text-color);'
                    f'opacity:0.5;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;'
                    f'gap:0.5rem;padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<span>Category</span>'
                    f'<span style="text-align:right">{str(_prev_m)}</span>'
                    f'<span style="text-align:right">{str(_curr_m)}</span>'
                    f'<span style="text-align:right">Change</span>'
                    f'</div>'
                ]
                for _r in mom_rows:
                    _arrow = "&#9650;" if _r["pct"] >= 0 else "&#9660;"   # up/down triangle
                    _col   = "#ef4444" if _r["pct"] > 0 else "#10b981"
                    _sign  = "+" if _r["pct"] >= 0 else ""
                    _mom_html.append(
                        f'<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;'
                        f'gap:0.5rem;padding:0.45rem 0;font-size:0.82rem;'
                        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span>{html.escape(_r["category"])}</span>'
                        f'<span style="text-align:right">{format_indian_currency(_r["prev"])}</span>'
                        f'<span style="text-align:right">{format_indian_currency(_r["curr"])}</span>'
                        f'<span style="text-align:right;color:{_col};font-weight:600;">'
                        f'{_arrow} {_sign}{_r["pct"]:.1f}%</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="as-panel" style="padding:0.75rem 1rem;">{"".join(_mom_html)}</div>',
                    unsafe_allow_html=True,
                )

        # ── Subscriptions & Recurring Payments ────────────────────────────
        st.markdown('<div class="as-section-head">Subscriptions &amp; Recurring Payments</div>',
                    unsafe_allow_html=True)
        recur_df = detect_recurring(df)
        if recur_df.empty:
            st.markdown(
                '<div class="as-clear">No recurring payment patterns detected '
                '(requires 2+ months of data).</div>',
                unsafe_allow_html=True,
            )
        else:
            total_monthly_recurring = recur_df[
                recur_df["likely_subscription"]
            ]["avg_amount"].sum()

            # Summary chip row
            sub_count = recur_df["likely_subscription"].sum()
            high_subs = recur_df[
                recur_df["likely_subscription"] & (recur_df["avg_amount"] > 200)
            ]
            st.markdown(
                f'<div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:1rem;">'
                f'<span class="as-chip primary">'
                f'{len(recur_df)} recurring merchants detected</span>'
                f'<span class="as-chip warning">'
                f'Est. monthly cost: {format_indian_currency(total_monthly_recurring)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # "Cancel to save" highlight for subscriptions over ₹200/month
            if not high_subs.empty:
                cancel_html = ['<div style="margin-bottom:1rem;">']
                for _, row in high_subs.iterrows():
                    cancel_html.append(
                        f'<div class="as-chip danger" style="margin-bottom:0.4rem;display:block;">'
                        f'Cancel <strong>{html.escape(str(row["merchant"]))}</strong> to save '
                        f'{format_indian_currency(row["avg_amount"])}/month'
                        f' ({format_indian_currency(row["avg_amount"] * 12)}/year)</div>'
                    )
                cancel_html.append("</div>")
                st.markdown("".join(cancel_html), unsafe_allow_html=True)

            # Table of all recurring payments
            recur_disp = recur_df.copy()
            recur_disp["avg_amount"]   = recur_disp["avg_amount"].apply(format_indian_currency)
            recur_disp["total_spent"]  = recur_disp["total_spent"].apply(format_indian_currency)
            recur_disp["likely_subscription"] = recur_disp["likely_subscription"].map(
                {True: "Yes", False: "No"}
            )
            recur_disp.columns = [
                "Merchant", "Occurrences", "Avg Amount",
                "Total Spent", "Months Active", "Subscription",
            ]
            st.markdown(_html_table(recur_disp, right_cols=["Occurrences", "Avg Amount", "Total Spent", "Months Active"]),
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Anomalies (alert card grid)
# ═══════════════════════════════════════════════════════════════════════════
with tab_anom:
    anom_df = df[df["is_anomaly"] == True].copy()
    n_anom  = len(anom_df)

    if n_anom == 0:
        st.markdown(
            '<div class="as-clear">No anomalous transactions detected — '
            'spending patterns are consistent.</div>',
            unsafe_allow_html=True,
        )
    else:
        flagged_total = format_indian_currency(
            anomaly_summary.get("total_flagged_amount", 0))
        st.markdown(
            f'<div class="as-warn">'
            f'<strong>{n_anom} anomalous transaction'
            f'{"s" if n_anom != 1 else ""} detected</strong>'
            f' &mdash; {flagged_total} flagged</div>',
            unsafe_allow_html=True,
        )

        # Security-alert cards (top 6)
        anom_sorted = anom_df.sort_values("anomaly_score", ascending=False)
        cards_to_show = anom_sorted.head(6)

        card_html = ['<div class="as-alert-grid">']
        for _, row in cards_to_show.iterrows():
            sev_cls, sev_label = _alert_severity(float(row["anomaly_score"]))
            date_str = (row["date"].strftime("%d %b %Y")
                        if pd.notna(row["date"]) else "Unknown date")
            desc   = html.escape(str(row.get("description", "")))
            amt    = format_indian_currency(float(row.get("debit", 0)))
            reason = html.escape(str(row.get("anomaly_reason", "")))
            cat    = html.escape(str(row.get("category", "")))
            score  = float(row["anomaly_score"])
            # display desc truncated for merchant name
            merch_disp = desc[:38] if len(desc) > 38 else desc

            card_html.append(f"""
            <div class="as-alert-card">
                <div class="as-alert-header-row">
                    <span class="as-anomaly-flag">ANOMALY DETECTED</span>
                    <span class="as-severity-pill {sev_cls}">{sev_label}</span>
                </div>
                <div class="as-alert-merchant" title="{desc}">{merch_disp}</div>
                <div class="as-alert-meta">
                    <span>{amt}</span>
                    <span>{date_str}</span>
                </div>
                <div class="as-alert-reason">{reason}</div>
                <div class="as-alert-actions">
                    <button class="as-ghost-btn">Mark Resolved</button>
                    <button class="as-ghost-btn">View Details</button>
                </div>
            </div>""")
        card_html.append('</div>')
        st.markdown("".join(card_html), unsafe_allow_html=True)

        # Histogram (all debit scores)
        st.markdown('<div class="as-section-head">Anomaly Score Distribution</div>',
                    unsafe_allow_html=True)
        scores_col, _ = st.columns([2, 1])
        with scores_col:
            debit_scores = df[df["transaction_type"] == "debit"]["anomaly_score"]
            fig_h = px.histogram(
                debit_scores, nbins=40,
                labels={"value": "Anomaly Score", "count": "Transactions"},
                color_discrete_sequence=["#2563eb"],
            )
            fig_h.update_traces(marker_line_width=0, opacity=0.82)
            fig_h.add_vline(x=0.65, line_dash="dash", line_color="#ef4444",
                            line_width=1.5,
                            annotation_text="Threshold 0.65",
                            annotation_font_size=10,
                            annotation_font_color="#ef4444",
                            annotation_position="top right")
            fig_h.update_layout(**_base_layout("Score Distribution — Debit Transactions"))
            st.plotly_chart(fig_h, use_container_width=True,
                            config={"displayModeBar": False})

        # Remaining anomalies as compact table
        if n_anom > 6:
            st.markdown(
                f'<div class="as-section-head">All Flagged Transactions ({n_anom})</div>',
                unsafe_allow_html=True,
            )
            tbl = anom_sorted[
                ["date", "description", "debit", "category",
                 "anomaly_score", "anomaly_reason"]
            ].copy()
            tbl["date"]  = tbl["date"].dt.strftime("%d %b %Y")
            tbl["debit"] = tbl["debit"].apply(format_indian_currency)
            tbl["anomaly_score"] = tbl["anomaly_score"].round(3)
            tbl.columns = ["Date", "Description", "Amount", "Category",
                           "Score", "Reason"]
            st.markdown(_html_table(tbl, right_cols=["Amount", "Score"]),
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Forecast
# ═══════════════════════════════════════════════════════════════════════════
with tab_fc:
    if forecast_summary.empty:
        st.info("Not enough data for forecasts (requires 3+ months).")
    else:
        hist = (
            df[df["transaction_type"] == "debit"]
            .groupby(pd.Grouper(key="date", freq="MS"))["debit"]
            .sum().reset_index()
        )
        hist.columns = ["date", "amount"]

        # ── 3 summary cards at top ────────────────────────────────
        _tp = total_forecast["total_forecast"]
        _tl = total_forecast["total_lower"]
        _tu = total_forecast["total_upper"]
        _last_actual = float(hist["amount"].iloc[-1]) if not hist.empty else 0
        _vs_pct = ((_tp - _last_actual) / _last_actual * 100) if _last_actual else 0
        _vs_sign = "+" if _vs_pct >= 0 else ""
        _vs_cls  = "red" if _vs_pct >= 0 else "green"
        _n_months_fc = max(1, df["date"].dt.to_period("M").nunique())
        _conf_pct = max(60, min(98, int(100 - abs(_tp - _last_actual) / max(_last_actual, 1) * 50)))
        _next_mo = (hist["date"].max() + pd.DateOffset(months=1)).strftime("%b %Y") if not hist.empty else "Next Month"
        st.markdown(f"""
        <div class="as-fc-grid">
            <div class="as-fc-card">
                <div class="as-fc-label">Next Month Forecast</div>
                <div class="as-fc-value blue">{format_indian_currency(_tp)}</div>
                <div class="as-fc-sub">Projected total spend ({_next_mo})</div>
            </div>
            <div class="as-fc-card">
                <div class="as-fc-label">vs Last Month</div>
                <div class="as-fc-value {_vs_cls}">{_vs_sign}{_vs_pct:.1f}%</div>
                <div class="as-fc-sub">&#8593; {format_indian_currency(abs(_tp - _last_actual))} {'above' if _vs_pct >= 0 else 'below'} last month</div>
            </div>
            <div class="as-fc-card">
                <div class="as-fc-label">Model Confidence</div>
                <div class="as-fc-value green">{_conf_pct}%</div>
                <div class="as-fc-sub">Prophet/ARIMA · {_n_months_fc}-month training</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Forecast chart ────────────────────────────────────────
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist["date"], y=hist["amount"],
            mode="lines+markers", name="Historical",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=6, color="#2563eb",
                        line=dict(color="rgba(255,255,255,0.9)", width=1.5)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
            hovertemplate="%{x|%b %Y}<br>\u20b9%{y:,.0f}<extra>Historical</extra>",
        ))

        if not hist.empty and hist["date"].max() is not pd.NaT:
            nm = hist["date"].max() + pd.DateOffset(months=1)

            fig_fc.add_trace(go.Scatter(
                x=[nm], y=[_tp], mode="markers", name="Forecast",
                marker=dict(color="#10b981", size=13, symbol="diamond",
                            line=dict(color="white", width=2)),
                hovertemplate="%{x|%b %Y}<br>\u20b9%{y:,.0f}<extra>Forecast</extra>",
            ))
            fig_fc.add_trace(go.Scatter(
                x=[nm, nm], y=[_tl, _tu], mode="lines",
                name="Confidence Range",
                line=dict(color="rgba(16,185,129,0.5)", width=6, dash="dot"),
                hoverinfo="skip",
            ))

        fig_fc.update_layout(**_base_layout("Spending Forecast — Next 3 Months"))
        fig_fc.update_layout(
            xaxis_title="Month", yaxis_title="Amount (\u20b9)",
            legend=dict(orientation="h", y=-0.18, x=0, xanchor="left",
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_fc, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Category forecast table (styled HTML) ────────────────
        st.markdown(
            '<div class="as-section-head">Category Forecasts — ' + _next_mo + '</div>',
            unsafe_allow_html=True,
        )
        _fc_rows = ""
        for _, _fcr in forecast_summary.iterrows():
            _chg = float(_fcr["vs_last_month_pct_change"])
            _chg_cls  = "color:var(--c-red)" if _chg > 0 else "color:var(--c-green)"
            _chg_sign = "+" if _chg > 0 else ""
            _hist_avg = float(_fcr.get("lower_bound", 0))
            _fc_rows += f"""
            <tr>
                <td style="font-size:13px;font-weight:500;color:var(--c-text);">{html.escape(str(_fcr["category"]))}</td>
                <td style="text-align:right;font-family:var(--c-mono);font-size:12px;color:var(--c-text-2);">{format_indian_currency(float(_fcr["lower_bound"]))}</td>
                <td style="text-align:right;font-family:var(--c-mono);font-size:13px;font-weight:600;color:var(--c-text);">{format_indian_currency(float(_fcr["next_month_forecast"]))}</td>
                <td style="text-align:right;font-size:12px;font-weight:600;{_chg_cls};">{_chg_sign}{_chg:.1f}%</td>
                <td style="text-align:right;font-size:12px;color:var(--c-text-3);">{_conf_pct}%</td>
            </tr>"""
        st.markdown(f"""
        <div style="background:var(--c-surface);border:1px solid var(--c-border);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr>
                    <th style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--c-text-3);text-align:left;padding:8px 12px;border-bottom:1px solid var(--c-border);background:var(--c-surface-2);">Category</th>
                    <th style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--c-text-3);text-align:right;padding:8px 12px;border-bottom:1px solid var(--c-border);background:var(--c-surface-2);">Jun Actual</th>
                    <th style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--c-text-3);text-align:right;padding:8px 12px;border-bottom:1px solid var(--c-border);background:var(--c-surface-2);">Forecast</th>
                    <th style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--c-text-3);text-align:right;padding:8px 12px;border-bottom:1px solid var(--c-border);background:var(--c-surface-2);">Change</th>
                    <th style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--c-text-3);text-align:right;padding:8px 12px;border-bottom:1px solid var(--c-border);background:var(--c-surface-2);">Confidence</th>
                </tr></thead>
                <tbody>{_fc_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Advisor (structured + narrative)
# ═══════════════════════════════════════════════════════════════════════════
with tab_ai:
    # Compute savings rate metrics for display
    _ai_cr = stats.get("total_credits", 0)
    _ai_db = stats.get("total_debits", 0)
    _ai_sr = _savings_rate(_ai_cr, _ai_db)
    _ai_sr_str = f"{_ai_sr:.1f}%"
    # "You spend ₹X for every ₹1 earned"
    _spend_per_rupee = (_ai_db / _ai_cr) if _ai_cr > 0 else 0.0
    _spend_ctx = f"You spend ₹{_spend_per_rupee:.2f} for every ₹1 earned."

    # Advisory summary cards
    st.markdown(
        _advisory_cards_html(stats, anomaly_summary, forecast_summary),
        unsafe_allow_html=True,
    )

    # Savings rate hero
    _sr_color = "#10b981" if _ai_sr >= 20 else ("#f59e0b" if _ai_sr >= 0 else "#ef4444")
    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<div class="as-panel-label">Savings Rate</div>'
        f'<div style="font-size:2.25rem;font-weight:700;color:{_sr_color};'
        f'letter-spacing:-0.03em;line-height:1.1;">{_ai_sr_str}</div>'
        f'<div style="font-size:13px;color:#6B7280;margin-top:0.25rem;">{_spend_ctx}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Improvement tips as cards
    _tips_for_advisor = health_components.get("tips", [])
    if _tips_for_advisor:
        st.markdown('<div class="as-section-label" style="margin-bottom:0.65rem;">Improvement Tips</div>',
                    unsafe_allow_html=True)
        _tip_pts_map = [
            ("Increase savings rate to 20%+", "Redirect discretionary spend — even 5% reduction compounds significantly.", "+ 12 pts"),
            ("Review flagged anomalies", "Unusual transactions may represent fraud or unplanned overspend.", "+ 8 pts"),
            ("Reduce top-category concentration", "Spreading spend across more categories improves your diversity score.", "+ 6 pts"),
            ("Monitor forecast trend", "If next-month spend is trending higher, identify and cut one variable expense.", "+ 5 pts"),
        ]
        for i, _tip in enumerate(_tips_for_advisor):
            _pts_label = _tip_pts_map[i][2] if i < len(_tip_pts_map) else ""
            # Extract headline and detail from the generated tip
            _tip_parts = _tip.split(" → ")
            _tip_headline = _tip_parts[0] if _tip_parts else _tip
            _tip_detail = _tip_pts_map[i][1] if i < len(_tip_pts_map) else ""
            st.markdown(
                f'<div class="as-tip-card">'
                f'<div class="as-tip-headline">{_tip_headline}</div>'
                f'<div class="as-tip-body">{_tip_detail}</div>'
                f'<span class="as-tip-badge">{_pts_label}</span>'
                f'<div style="clear:both;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    cat_totals = (
        df[df["transaction_type"] == "debit"]
        .groupby("category")["debit"].sum()
        .sort_values(ascending=False).reset_index()
    )
    anom_for_advice = df[df["is_anomaly"] == True][
        ["date", "description", "amount", "category",
         "anomaly_score", "anomaly_reason"]
    ].copy()
    use_gemini_now = enable_gemini and gemini_key.strip()
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    model_label = "Gemini AI" if use_gemini_now else "Rule-based Analysis"

    # Chat wrapper
    st.markdown(f"""
    <div class="as-chat-wrap">
        <div class="as-chat-head">
            <div class="as-chat-avatar">AS</div>
            <div>
                <div class="as-chat-name">ArthaSense Advisor</div>
                <div class="as-chat-status">
                    <span class="as-status-dot"></span>{model_label}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not use_gemini_now:
        st.markdown(
            '<div class="as-chat-info">Rule-based analysis active. '
            'Enable Gemini AI in Settings for personalised LLM advice.</div>',
            unsafe_allow_html=True,
        )

    if use_gemini_now:
        if not GEMINI_AVAILABLE:
            st.warning("Install google-generativeai to enable AI advisor. Using rule-based advice.")
        advisor = LLMAdvisor(api_key=gemini_key.strip())
        _init_failed = not advisor.gemini_active and bool(advisor.init_error)
        if _init_failed:
            st.warning(f"Gemini init failed: {advisor.init_error}")
        stream  = advisor.get_advice_streaming(
            summary_stats=stats, category_totals=cat_totals,
            anomalies=anom_for_advice, forecasts=total_forecast,
        )
        st.markdown('<div class="as-chat-body">', unsafe_allow_html=True)
        st.write_stream(stream)
        if not _init_failed and advisor.init_error:
            if "quota" in advisor.init_error.lower() or "429" in advisor.init_error:
                st.warning(
                    "All Gemini models quota-exhausted for today. "
                    "Enable billing at aistudio.google.com or wait until quota resets. "
                    "Showing rule-based advice."
                )
            else:
                st.warning(f"Gemini call failed: {advisor.init_error}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        advisor      = LLMAdvisor(api_key=None)
        advice_text  = advisor.get_advice(
            summary_stats=stats, category_totals=cat_totals,
            anomalies=anom_for_advice, forecasts=total_forecast,
        )
        st.markdown('<div class="as-chat-body">', unsafe_allow_html=True)
        st.markdown(advice_text)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="as-chat-footer">{model_label} &middot; {now_str}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Transactions
# ═══════════════════════════════════════════════════════════════════════════
with tab_txn:
    # ── Search + type filter ─────────────────────────────────────────────
    _txn_search = ""

    _type_options = ["All", "Debits", "Credits", "⚠ Anomalies"]
    if "txn_type_filter" not in st.session_state:
        st.session_state["txn_type_filter"] = "All"

    _type_cols = st.columns(len(_type_options) + 4)
    for _i, _opt in enumerate(_type_options):
        _is_active = st.session_state["txn_type_filter"] == _opt
        if _type_cols[_i].button(
            _opt,
            key=f"txn_type_{_i}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state["txn_type_filter"] = _opt

    # ── Category filter via selectbox ────────────────────────────────────
    _txn_cats = st.session_state.get("_txn_cats", ["All"])
    _selected_cat = st.selectbox(
        "Filter by category",
        options=_txn_cats,
        key="txn_cat_filter",
        label_visibility="collapsed",
    )

    # ── Build filtered dataframe ─────────────────────────────────────────
    _txn_work = df.copy()
    _type_sel = st.session_state.get("txn_type_filter", "All")

    if _txn_search:
        _q = _txn_search.strip()
        _desc_m = _txn_work["description"].astype(str).str.contains(_q, case=False, na=False, regex=False)
        _cat_m  = (_txn_work["category"].astype(str).str.contains(_q, case=False, na=False, regex=False)
                   if "category" in _txn_work.columns
                   else pd.Series(False, index=_txn_work.index))
        _txn_work = _txn_work[_desc_m | _cat_m]

    if _type_sel == "Debits":
        _txn_work = _txn_work[_txn_work["transaction_type"] == "debit"]
    elif _type_sel == "Credits":
        _txn_work = _txn_work[_txn_work["transaction_type"] == "credit"]
    elif _type_sel == "⚠ Anomalies":
        if "is_anomaly" in _txn_work.columns:
            _txn_work = _txn_work[_txn_work["is_anomaly"] == True]

    if _selected_cat and _selected_cat != "All" and "category" in _txn_work.columns:
        _txn_work = _txn_work[_txn_work["category"] == _selected_cat]

    _ROW_LIMIT = 500
    _total_rows = len(_txn_work)
    _txn_work = _txn_work.iloc[:_ROW_LIMIT]

    # ── Render custom HTML table ─────────────────────────────────────────
    def _build_tx_table(rows_df):
        if rows_df.empty:
            return '<div class="tx-empty">No transactions match your filters.</div>'

        html = ['<div class="tx-table-wrap"><table class="tx-table">']
        html.append(
            '<thead><tr>'
            '<th>Date</th><th>Description</th><th>Category</th>'
            '<th>Type</th><th style="text-align:right">Amount</th>'
            '</tr></thead><tbody>'
        )

        for _, row in rows_df.iterrows():
            is_anom = bool(row.get("is_anomaly", False))
            tr_cls  = ' class="tx-row-anom"' if is_anom else ''

            # Date
            try:
                date_str = row["date"].strftime("%-d %b %Y") if pd.notna(row["date"]) else ""
            except Exception:
                date_str = str(row.get("date", ""))

            # Description — truncate long names
            desc = str(row.get("description", ""))
            desc_display = (desc[:42] + "…") if len(desc) > 45 else desc
            anom_dot = '<span class="tx-anomaly-dot"></span>' if is_anom else ''

            # Category pill
            cat = str(row.get("category", "Uncategorized"))
            cat_html = f'<span class="tx-cat-pill">{cat}</span>'

            # Transaction type badge
            txn_type = str(row.get("transaction_type", "")).capitalize()

            # Amount
            debit_val  = row.get("debit",  0) or 0
            credit_val = row.get("credit", 0) or 0
            if debit_val and float(debit_val) > 0:
                amt_html = f'<span class="tx-amount debit">- ₹{float(debit_val):,.2f}</span>'
            elif credit_val and float(credit_val) > 0:
                amt_html = f'<span class="tx-amount credit">+ ₹{float(credit_val):,.2f}</span>'
            else:
                amt_html = '<span class="tx-amount" style="color:var(--c-text-3)">—</span>'

            html.append(
                f'<tr{tr_cls}>'
                f'<td><span class="tx-date">{date_str}</span></td>'
                f'<td><span class="tx-merchant">{anom_dot}{desc_display}</span></td>'
                f'<td>{cat_html}</td>'
                f'<td><span style="font-size:12px;color:var(--c-text-3)">{txn_type}</span></td>'
                f'<td style="text-align:right">{amt_html}</td>'
                f'</tr>'
            )

        html.append('</tbody></table></div>')
        return "".join(html)

    if _total_rows > _ROW_LIMIT:
        st.caption(f"Showing first {_ROW_LIMIT} of {_total_rows} transactions — download CSV for all.")

    st.markdown(_build_tx_table(_txn_work), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.download_button(
        label="Download Analysed CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="arthasense_analysed.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Planner (What-If Scenario)
# ═══════════════════════════════════════════════════════════════════════════
with tab_plan:
    _plan_debits = df[df["transaction_type"] == "debit"].copy()
    _plan_credits = df[df["transaction_type"] == "credit"].copy()

    _total_debit  = float(_plan_debits["debit"].sum()) if not _plan_debits.empty else 0
    _total_credit = float(_plan_credits["credit"].sum()) if not _plan_credits.empty else 0
    _n_months     = max(1, df["date"].dt.to_period("M").nunique())
    _monthly_spend  = _total_debit  / _n_months
    _monthly_income = _total_credit / _n_months
    _monthly_savings = max(0.0, _monthly_income - _monthly_spend)

    # Category totals for dropdowns
    _cat_monthly = (
        _plan_debits.groupby("category")["debit"].sum() / _n_months
        if not _plan_debits.empty else pd.Series(dtype=float)
    )
    _available_cats = sorted(_cat_monthly[_cat_monthly > 0].index.tolist())

    # ── Tool 1: Spending Cut Simulator ───────────────────────────────────────
    st.markdown(
        '<div class="as-section-label">Tool</div>'
        '<div class="as-section-title">Spending Cut Simulator</div>',
        unsafe_allow_html=True,
    )
    _pl_col1, _pl_col2 = st.columns(2, gap="large")
    with _pl_col1:
        _sel_cat = st.selectbox(
            "Category to cut",
            options=_available_cats if _available_cats else ["No categories"],
            key="planner_cat",
        )
        _cut_pct = st.slider(
            "Reduce spending by (%)",
            min_value=5, max_value=100, value=25, step=5,
            key="planner_cut_pct",
        )

    if _sel_cat and _sel_cat in _cat_monthly.index:
        _cat_spend_mo  = float(_cat_monthly[_sel_cat])
        _saved_monthly = _cat_spend_mo * _cut_pct / 100
        _saved_yearly  = _saved_monthly * 12
        _months_to_lakh = (100000 / _saved_monthly) if _saved_monthly > 0 else float("inf")

        with _pl_col2:
            st.markdown(
                f'<div class="as-panel">'
                f'<div class="as-panel-label">Projected Savings</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">'
                f'<div><div style="font-size:12px;color:#6B7280;">Monthly</div>'
                f'<div style="font-size:1.5rem;font-weight:600;color:#10b981;">'
                f'{format_indian_currency(_saved_monthly)}</div></div>'
                f'<div><div style="font-size:12px;color:#6b7280;">Annual</div>'
                f'<div style="font-size:1.5rem;font-weight:600;color:#3b82f6;">'
                f'{format_indian_currency(_saved_yearly)}</div></div>'
                f'</div>'
                f'<div style="font-size:13px;color:#9ca3af;'
                f'padding:0.6rem;background:rgba(16,185,129,0.06);'
                f'border-radius:6px;border-left:2px solid #10b981;">'
                f'Months to ₹1 lakh: '
                f'<strong style="color:#ffffff;">'
                f'{"Never" if _months_to_lakh == float("inf") else f"{_months_to_lakh:.1f}"}'
                f'</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Tool 2: Goal Tracker ──────────────────────────────────────────────────
    st.markdown(
        '<div class="as-section-label" style="margin-top:2rem;">Tool</div>'
        '<div class="as-section-title">Goal Tracker</div>',
        unsafe_allow_html=True,
    )
    _g_col1, _g_col2 = st.columns(2, gap="large")
    with _g_col1:
        _goal_name   = st.text_input("Goal name", value="New Laptop", key="planner_goal_name")
        _goal_amount = st.number_input(
            "Target amount (₹)", min_value=1000, value=50000, step=1000,
            key="planner_goal_amount",
        )

    with _g_col2:
        if _monthly_savings > 0:
            _months_needed = _goal_amount / _monthly_savings
            _months_half   = _goal_amount / (_monthly_savings * 2)
            _biggest_cat_mo = float(_cat_monthly.max()) if not _cat_monthly.empty else 0
            _cut_needed_pct = min(100, (_monthly_savings / _biggest_cat_mo * 100)
                                  if _biggest_cat_mo > 0 else 100)
            _biggest_cat_name = _cat_monthly.idxmax() if not _cat_monthly.empty else "—"

            st.markdown(
                f'<div class="as-panel">'
                f'<div class="as-panel-label">Goal: {_goal_name}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">'
                f'<div><div style="font-size:12px;color:#6B7280;">At current rate</div>'
                f'<div style="font-size:1.5rem;font-weight:600;color:#f59e0b;">'
                f'{_months_needed:.1f} mo</div></div>'
                f'<div><div style="font-size:12px;color:#6b7280;">At 2x savings</div>'
                f'<div style="font-size:1.5rem;font-weight:600;color:#10b981;">'
                f'{_months_half:.1f} mo</div></div>'
                f'</div>'
                f'<div style="font-size:13px;color:#9ca3af;'
                f'padding:0.6rem;background:rgba(37,99,235,0.06);'
                f'border-radius:6px;border-left:2px solid #2563eb;">'
                f'Cut <strong style="color:#ffffff;">{_biggest_cat_name}</strong> by '
                f'<strong style="color:#ffffff;">{_cut_needed_pct:.0f}%</strong> '
                f'to halve your timeline.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="as-warn">No net monthly savings detected. '
                'Increase income or reduce spending to track a goal.</div>',
                unsafe_allow_html=True,
            )


