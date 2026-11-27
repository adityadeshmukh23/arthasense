"""
ArthaSense — how much can I safely spend this week before salary day?

Four screens: Start -> Confirm -> Answer -> Check-in.
No AI calls. Nothing is saved after the session.
The previous dashboard lives in app_legacy.py.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.safe_to_spend import (
    DEFAULT_BUFFER, GROUP_CHOICES, StatementError, bills_before_payday, check_in,
    detect_recurring, detect_salary, inr, load_statement, next_payday, one_change,
    compare_to_usual, safe_to_spend, tag_bill_rows, usual_weekly_spend,
)

APP_VERSION = "3.0"
MAX_UPLOAD_MB = 10        # keep in sync with server.maxUploadSize in .streamlit/config.toml
SAMPLE_PATH = Path(__file__).parent / "data" / "sample_transactions.csv"
UPLOAD_TYPES = ["csv", "xlsx"]

st.set_page_config(page_title="ArthaSense — safe to spend", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --accent:#0F766E; --ink:#1C2421; --muted:#5B6661; --line:#E3E7E4; --soft:#F3F6F4; --warn:#9A3412; }
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
.block-container { max-width: 680px; padding-top: 3.5rem; padding-bottom: 4rem; }
h1 { font-size: 1.9rem !important; line-height: 1.25 !important; color: var(--ink); font-weight: 700 !important; }
h2, h3 { color: var(--ink); }
.as-brand { font-size: .85rem; font-weight: 600; color: var(--accent); letter-spacing:.02em; }
.as-step { font-size: .8rem; color: var(--muted); margin: .2rem 0 .4rem; }
.as-lead { font-size: 1.05rem; color: var(--muted); line-height: 1.55; }
.as-card { border:1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; background:#fff; margin: .6rem 0; }
.as-soft { background: var(--soft); border-color: var(--soft); }
.as-label { font-size: .8rem; color: var(--muted); text-transform: none; margin-bottom:.15rem; }
.as-big { font-size: 2.4rem; font-weight: 700; color: var(--accent); line-height: 1.15; font-variant-numeric: tabular-nums; }
.as-mid { font-size: 1.5rem; font-weight: 650; color: var(--ink); line-height: 1.2; font-variant-numeric: tabular-nums; }
.as-short { font-size: 1.8rem; font-weight: 700; color: var(--warn); }
.as-small { font-size: .88rem; color: var(--muted); line-height: 1.5; }
.as-tag { display:inline-block; font-size:.75rem; padding:.1rem .5rem; border-radius: 999px; background: var(--soft); color: var(--muted); margin-left:.3rem; }
.as-sum { width:100%; border-collapse: collapse; font-variant-numeric: tabular-nums; border: none !important; }
.as-sum tr, .as-sum td { border-left: none !important; border-right: none !important; border-top: none !important; background: transparent !important; }
.as-sum td { padding: .4rem 0 !important; border-bottom: 1px solid var(--line) !important; font-size: .95rem; }
.as-sum td:last-child { text-align: right; white-space: nowrap; padding-left: .8rem; }
.as-sum tr.total td { font-weight: 700; border-bottom: none; }
@media (max-width: 640px) { .as-big { font-size: 2rem; } h1 { font-size: 1.55rem !important; } }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("screen", "start")


def go(screen: str) -> None:
    ss.screen = screen
    st.rerun()


def inr2(x: float) -> str:
    """Rupees with paise, for the arithmetic line, so the sums visibly add up."""
    paise = int(round(x * 100))
    return f"{inr(paise // 100)}.{paise % 100:02d}"


def fmt_d(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b}"


def fmt_dy(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def header(step: int | None, title: str) -> None:
    st.markdown('<div class="as-brand">ArthaSense</div>', unsafe_allow_html=True)
    if step:
        st.markdown(f'<div class="as-step">Step {step} of 4</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")


def footer() -> None:
    st.markdown(
        '<p class="as-small" style="margin-top:2rem;">Budgeting help only. Not financial or '
        f'investment advice. Nothing is saved after your session. v{APP_VERSION}</p>',
        unsafe_allow_html=True)


def reset_all() -> None:
    for k in list(ss.keys()):
        del ss[k]
    ss.screen = "start"


def open_statement(source, name: str, is_sample: bool) -> None:
    with st.spinner("Reading your statement…"):
        stmt = load_statement(source, name)
    for k in [k for k in ss.keys() if k.startswith(("cf_", "ed_", "ci_"))]:
        del ss[k]
    ss.stmt, ss.is_sample, ss.file_name = stmt, is_sample, name
    ss.checkins = []
    go("confirm")


# =========================================================================
# 1. Start
# =========================================================================
def screen_start() -> None:
    header(None, "How much can you safely spend this week?")
    st.markdown(
        '<p class="as-lead">Upload a bank statement. We work out how much you can spend this week '
        'and still cover your bills until salary day.</p>', unsafe_allow_html=True)

    if st.button("Try it with sample data", type="primary", width="stretch"):
        try:
            open_statement(SAMPLE_PATH, SAMPLE_PATH.name, True)
        except StatementError as e:
            st.error(str(e))
    st.markdown('<p class="as-small">The sample is made-up data for a fictional salaried person. '
                'No consent needed.</p>', unsafe_allow_html=True)

    st.markdown("### Or use your own statement")
    st.markdown(
        '<div class="as-card as-soft as-small"><b>What happens to your file.</b> It is read in memory '
        'by the server running this app (on the hosted demo, Streamlit Community Cloud). It is not '
        'saved to disk, not sent to any AI service, and is gone when your session ends.</div>',
        unsafe_allow_html=True)
    up = st.file_uploader(f"Bank statement (CSV or Excel, up to {MAX_UPLOAD_MB} MB)",
                          type=UPLOAD_TYPES)
    agree = st.checkbox("I agree to let this app read my statement for this session only.")
    if st.button("Read my statement", disabled=not (up and agree), width="stretch"):
        if up.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is larger than {MAX_UPLOAD_MB} MB.")
            return
        try:
            open_statement(up, up.name, False)
        except StatementError as e:
            st.error(str(e))
    if up and not agree:
        st.markdown('<p class="as-small">Tick the box above to continue.</p>', unsafe_allow_html=True)
    footer()


# =========================================================================
# 2. Confirm
# =========================================================================
BILL_COLS = {
    "Include": st.column_config.CheckboxColumn("Include", width="small"),
    "Name": st.column_config.TextColumn("Name", required=True),
    "Amount": st.column_config.NumberColumn("Amount (₹)", min_value=0, step=1, format="%d", required=True),
    "Due": st.column_config.DateColumn("Due", format="D MMM"),
    "Type": st.column_config.SelectboxColumn("Type", options=list(GROUP_CHOICES), required=True),
    "Note": st.column_config.TextColumn("Note", disabled=True),
}


def _bill_frame(bills, include: bool) -> pd.DataFrame:
    rows = [{"Include": include, "Name": b.name, "Amount": int(round(b.amount)), "Due": b.due,
             "Type": b.group, "Note": "Expected, not seen yet" if b.status == "overdue" else ""}
            for b in bills]
    return pd.DataFrame(rows, columns=list(BILL_COLS))


def _editor(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return st.data_editor(df, key=key, num_rows="dynamic", hide_index=True,
                          column_config=BILL_COLS, width="stretch")


def _included_total(df: pd.DataFrame) -> tuple[float, int]:
    if df is None or df.empty:
        return 0.0, 0
    inc = df[df["Include"].fillna(False).astype(bool)]
    amt = pd.to_numeric(inc["Amount"], errors="coerce").fillna(0).clip(lower=0)
    return float(amt.sum()), int(len(inc))


def screen_confirm() -> None:
    stmt = ss.stmt
    header(2, "Check the details")
    tag = '<span class="as-tag">Sample · synthetic data</span>' if ss.is_sample else ""
    st.markdown(
        f'<p class="as-small">Read {len(stmt.txns):,} transactions from {fmt_dy(stmt.start)} to '
        f'{fmt_dy(stmt.end)} ({stmt.source}).{tag}</p>', unsafe_allow_html=True)
    for n in stmt.notes:
        st.warning(n)

    txns = stmt.txns
    salary = detect_salary(txns)

    # ---- Dates ---------------------------------------------------------
    st.markdown("### Dates")
    today = st.date_input("Today", value=stmt.end, key="cf_today", format="DD/MM/YYYY",
                          help="Starts as the last date in your statement. Change it if needed.")
    if salary:
        default_pay = next_payday(salary.last_date, today)
        pay_help = (f"Last salary: {inr(salary.amount)} on {fmt_dy(salary.last_date)}. "
                    "We assume the same day next month. Change it if your payday differs.")
    else:
        default_pay = None
        pay_help = "We couldn't find a salary credit. Enter your next salary date."
    sig_pay = f"cf_payday_{today}_{default_pay}"
    payday = st.date_input("Next salary day", value=default_pay, key=sig_pay,
                           min_value=today + timedelta(days=1), format="DD/MM/YYYY")
    st.markdown(f'<p class="as-small">{pay_help}</p>', unsafe_allow_html=True)
    if salary is None:
        st.warning("No salary found in this statement. Enter your next salary date to continue.")
    elif salary.count < 2:
        st.info("We only saw one salary credit, so please check the date above.")

    # ---- Balance -------------------------------------------------------
    st.markdown("### Balance")
    bal_default = stmt.balance
    balance = st.number_input("Current balance (₹)", value=bal_default, step=100.0, format="%.0f",
                              key="cf_balance", placeholder="Enter your balance")
    if stmt.balance is not None:
        st.markdown(f'<p class="as-small">From your statement: {inr(stmt.balance)} as of '
                    f'{fmt_dy(stmt.balance_date)}.</p>', unsafe_allow_html=True)
    from_statement = (stmt.balance is not None and balance is not None and balance == stmt.balance
                      and stmt.balance_reliable and stmt.balance_date == today)
    balance_ok = balance is not None
    if balance is not None and not from_statement:
        if stmt.balance_date and stmt.balance_date != today:
            st.warning(f"Your balance is from {fmt_dy(stmt.balance_date)}, but today is set to "
                       f"{fmt_dy(today)}. Enter today's balance or confirm it is unchanged.")
        balance_ok = st.checkbox(f"This is my balance on {fmt_dy(today)}", key=f"cf_balok_{today}_{balance}")
    # The date the balance is true on: the statement's date, or "today" once the user confirms.
    balance_date = stmt.balance_date if from_statement else today

    # ---- Bills ---------------------------------------------------------
    st.markdown("### Bills before payday")
    recurring = detect_recurring(txns, balance_date)
    bills_tot = payday_tot = planned_tot = 0.0
    n_bills = 0
    if payday and payday > today:
        due = bills_before_payday(recurring, balance_date, payday)
        regular = [b for b in due if b.status in ("due", "overdue") and b.group != "Own-account transfer"]
        on_payday = [b for b in due if b.status == "payday" and b.group != "Own-account transfer"]
        planned = [b for b in due if b.status in ("due", "overdue") and b.group == "Own-account transfer"]
        sig = f"{balance_date}_{payday}"
        st.markdown(
            f'<p class="as-small">Regular payments we expect between {fmt_d(balance_date)} and '
            f'{fmt_d(payday)}, plus any we expected earlier but haven\'t seen. Edit amounts, untick, '
            'add rows or change the type.</p>', unsafe_allow_html=True)
        if not regular:
            st.markdown('<p class="as-small"><i>None found. Add any we missed.</i></p>', unsafe_allow_html=True)
        ed = _editor(_bill_frame(regular, True), f"ed_bills_{sig}")
        bills_tot, n_bills = _included_total(ed)

        if on_payday:
            st.markdown("#### Due on payday")
            st.markdown('<p class="as-small">These usually go out on salary day. Tick them if they '
                        'might leave before your salary arrives.</p>', unsafe_allow_html=True)
            ed2 = _editor(_bill_frame(on_payday, False), f"ed_payday_{sig}")
            payday_tot, k = _included_total(ed2)
            n_bills += k

        if planned:
            st.markdown("#### Planned transfers to your own accounts")
            st.markdown('<p class="as-small">Not counted as spending, but this money leaves this '
                        'account before payday. Untick to leave it out.</p>', unsafe_allow_html=True)
            ed3 = _editor(_bill_frame(planned, True), f"ed_planned_{sig}")
            planned_tot, k = _included_total(ed3)
            n_bills += k

        with st.expander("All regular payments we found"):
            if recurring:
                st.dataframe(pd.DataFrame([{
                    "Name": r.name, "Type": r.group, "Usual amount": inr(r.amount),
                    "Usual day": r.day, "Last paid": fmt_dy(r.last_paid), "Months seen": r.months,
                } for r in recurring]), hide_index=True, width="stretch")
                st.caption("Payments due after payday are covered by your next salary, so they are not listed above.")
            else:
                st.caption("We didn't find payments that repeat each month.")
    else:
        st.markdown('<p class="as-small">Set your next salary day to see bills due before it.</p>',
                    unsafe_allow_html=True)

    # ---- Buffer --------------------------------------------------------
    st.markdown("### Buffer")
    buffer = st.number_input("Money to keep aside, untouched (₹)", min_value=0, value=DEFAULT_BUFFER,
                             step=500, key="cf_buffer",
                             help="A cushion for surprises. It is never part of your weekly amount.")

    # ---- Go ------------------------------------------------------------
    missing = []
    if not payday:
        missing.append("your next salary day")
    if balance is None:
        missing.append("your current balance")
    elif not balance_ok:
        missing.append(f"confirmation of your balance on {fmt_dy(today)}")
    if missing:
        st.markdown(f'<p class="as-small">To continue, add {" and ".join(missing)}.</p>',
                    unsafe_allow_html=True)
    if st.button("Show my weekly amount", type="primary", disabled=bool(missing), width="stretch"):
        bill_rows = tag_bill_rows(txns, recurring)
        ss.plan = {
            "today": today, "payday": payday, "balance": float(balance), "balance_date": balance_date,
            "bills": bills_tot + payday_tot + planned_tot, "n_bills": n_bills,
            "bills_split": (bills_tot, payday_tot, planned_tot), "buffer": float(buffer),
            "usual": usual_weekly_spend(txns, bill_rows, min(balance_date, stmt.end)),
            "tip": one_change(txns, bill_rows, min(balance_date, stmt.end)),
        }
        ss.checkins = []
        go("answer")
    if st.button("Start again with a different file", width="stretch"):
        reset_all()
        st.rerun()
    footer()


# =========================================================================
# 3. Answer
# =========================================================================
def _sum_table(p: dict, a) -> str:
    bills_tot, payday_tot, planned_tot = p["bills_split"]
    rows = [(f"Balance on {fmt_dy(p['today'])}", inr(a.balance))]
    rows.append((f"− Bills before payday", inr(bills_tot)))
    if payday_tot:
        rows.append(("− Bills on payday you included", inr(payday_tot)))
    if planned_tot:
        rows.append(("− Planned transfers to your own accounts", inr(planned_tot)))
    rows.append(("− Buffer", inr(a.buffer)))
    html = "".join(f"<tr><td>{l}</td><td>{v}</td></tr>" for l, v in rows)
    html += f'<tr class="total"><td>= Left until payday</td><td>{inr(a.available)}</td></tr>'
    return f'<table class="as-sum">{html}</table>'


def screen_answer() -> None:
    p = ss.plan
    try:
        a = safe_to_spend(p["balance"], p["bills"], p["buffer"], p["today"], p["payday"])
    except ValueError as e:
        header(3, "Something needs fixing")
        st.error(str(e))
        if st.button("Back to details", width="stretch"):
            go("confirm")
        return
    ss.answer = a
    end = p["today"] + timedelta(days=a.horizon - 1)

    if a.shortfall is not None:
        header(3, "You may run short before payday")
        st.markdown(f'<div class="as-card"><div class="as-label">Shortfall</div>'
                    f'<div class="as-short">{inr(a.shortfall)}</div>'
                    f'<div class="as-small">Your bills and buffer come to more than your balance. '
                    f'There is no safe amount to spend until {fmt_d(p["payday"])}.</div></div>',
                    unsafe_allow_html=True)
        st.markdown("**What you can do:** lower the buffer, check the bill list for anything you can "
                    "move after payday, or plan how to cover the gap.")
    else:
        period = "this week" if a.horizon == 7 else f"in the next {a.horizon} day{'s' if a.horizon > 1 else ''}"
        header(3, f"You can spend {inr(a.weekly)} {period}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="as-card"><div class="as-label">Safe to spend, {fmt_d(p["today"])} – {fmt_d(end)}</div>'
                        f'<div class="as-big">{inr(a.weekly)}</div>'
                        f'<div class="as-small">About {inr(a.daily_shown)} a day</div></div>', unsafe_allow_html=True)
        with c2:
            if p["usual"]:
                usual, wk = p["usual"]
                st.markdown(f'<div class="as-card as-soft"><div class="as-label">You usually spend</div>'
                            f'<div class="as-mid">{inr(usual)} a week</div>'
                            f'<div class="as-small">Everyday spending in a typical week (median of the last {wk} week{"s" if wk > 1 else ""}). '
                            f'Excludes bills, rent, EMIs and own-account transfers.</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="as-card as-soft as-small">Not enough history to show your usual weekly spend.</div>',
                            unsafe_allow_html=True)
        if a.horizon < 7:
            st.markdown(f'<p class="as-small">Payday is in {a.days} day{"s" if a.days > 1 else ""}, so this covers the days until then.</p>',
                        unsafe_allow_html=True)

    st.markdown("### How we got this")
    st.markdown(_sum_table(p, a), unsafe_allow_html=True)
    if a.shortfall is None:
        st.markdown(
            f'<p class="as-small" style="margin-top:.6rem;">{inr(a.available)} × {a.horizon} ÷ {a.days} days to payday '
            f'({fmt_d(p["payday"])}) = {inr2(a.available * a.horizon / a.days)}, rounded down to '
            f'<b>{inr(a.weekly)}</b>.</p>', unsafe_allow_html=True)

    tip = p["tip"]
    usual = p["usual"][0] if p["usual"] else None
    cmp_ = compare_to_usual(a, usual, tip)
    span = "a week" if a.horizon == 7 else f"over these {a.horizon} days"
    if cmp_.state == "over":
        st.markdown("### Your usual spending is above the limit")
        st.markdown(
            f'<div class="as-card">To reach payday with your {inr(a.buffer)} buffer intact, spend '
            f'<b>{inr(cmp_.gap)} less</b> than usual {span}: {inr(a.weekly)} instead of about '
            f'{inr(cmp_.usual_period)}.</div>', unsafe_allow_html=True)
        if tip:
            st.markdown("#### One change that helps")
            rest = (f"That leaves <b>{inr(cmp_.remaining)}</b> still to find {span}."
                    if cmp_.remaining > 0 else "That covers the whole gap.")
            st.markdown(
                f'<div class="as-card">{tip.group} has averaged {inr(tip.weekly_avg)} a week over the last '
                f'{tip.weeks} weeks. Spending 25% less, about {inr(tip.cut_to)}, saves <b>{inr(cmp_.covered)}</b> '
                f'of the {inr(cmp_.gap)} gap. {rest} Optional.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="as-small">We need at least 4 weeks of history to suggest a specific change.</p>',
                        unsafe_allow_html=True)
    elif cmp_.state == "fits":
        st.markdown("### Your usual spending fits")
        st.markdown(
            f'<div class="as-card">You typically spend about {inr(cmp_.usual_period)} {span}, which is within the limit. '
            f'{inr(a.weekly)} is the <b>most</b> you can spend and still reach payday with your buffer. '
            f'It is a ceiling, not a target. Whatever you don\'t spend stays in your account.</div>',
            unsafe_allow_html=True)
        if tip:
            st.markdown(
                f'<p class="as-small">Optional: {tip.group} has averaged {inr(tip.weekly_avg)} a week. '
                f'25% less, about {inr(tip.cut_to)}, keeps {inr(tip.frees)} more.</p>', unsafe_allow_html=True)

    st.write("")
    if a.shortfall is None and st.button("Track my spending this week", type="primary", width="stretch"):
        go("checkin")
    if st.button("Change details", width="stretch"):
        go("confirm")
    footer()


# =========================================================================
# 4. Check-in
# =========================================================================
def screen_checkin() -> None:
    p, a = ss.plan, ss.answer
    header(4, "Check in")
    week_end = p["today"] + timedelta(days=a.horizon - 1)
    st.markdown(f'<p class="as-small">Limit {inr(a.weekly)} for {fmt_d(p["today"])} – {fmt_d(week_end)}. '
                f'Payday {fmt_d(p["payday"])}.</p>', unsafe_allow_html=True)
    st.markdown('<div class="as-card as-soft as-small">Nothing is saved. Your statement and check-ins '
                'are cleared when you close this tab or the session ends.</div>', unsafe_allow_html=True)

    last_date = ss.checkins[-1][0] if ss.checkins else p["today"]
    with st.form("ci_form", clear_on_submit=True):
        d = st.date_input("Check-in date", value=max(last_date, p["today"]), min_value=p["today"],
                          max_value=p["payday"] - timedelta(days=1), format="DD/MM/YYYY")
        amt = st.number_input("Spent since your last check (₹)", min_value=0, step=50, value=None,
                              placeholder="0",
                              help="Everyday spending only. Don't include the bills already listed.")
        st.caption("Don't include the bills on your list. They are already set aside.")
        if st.form_submit_button("Add", type="primary", width="stretch"):
            if amt is None:
                st.error("Enter an amount, even if it's 0.")
            else:
                ss.checkins.append((d, float(amt)))
                st.rerun()

    spent = sum(x for _, x in ss.checkins)
    check_date = ss.checkins[-1][0] if ss.checkins else p["today"]
    c = check_in(a, spent, check_date, p["today"], p["payday"])

    week_over = check_date > week_end
    if c.shortfall is not None and c.shortfall > 0:
        st.markdown(f'<div class="as-card"><div class="as-label">You have spent more than is left until payday</div>'
                    f'<div class="as-short">Short by {inr(c.shortfall)}</div></div>', unsafe_allow_html=True)
    elif week_over:
        st.markdown(f'<div class="as-card"><div class="as-label">Until payday, from {fmt_d(check_date)}</div>'
                    f'<div class="as-big">{inr(c.daily_until_payday)} a day</div>'
                    f'<div class="as-small">For {c.days_left} day{"s" if c.days_left != 1 else ""}. Spent since '
                    f'{fmt_d(p["today"])}: {inr(spent)}. The {fmt_d(p["today"])} – {fmt_d(week_end)} week has ended.</div></div>',
                    unsafe_allow_html=True)
    elif c.over_by:
        st.markdown(f'<div class="as-card"><div class="as-label">Over this week\'s limit by</div>'
                    f'<div class="as-short">{inr(c.over_by)}</div>'
                    f'<div class="as-small">To stay safe until payday, keep to about <b>{inr(c.daily_until_payday)} a day</b> '
                    f'for the next {c.days_left} day{"s" if c.days_left != 1 else ""}.</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="as-card"><div class="as-label">Left this week</div>'
                    f'<div class="as-big">{inr(c.remaining_week)}</div>'
                    f'<div class="as-small">Spent so far {inr(spent)}. Until payday you have about '
                    f'{inr(c.daily_until_payday)} a day for {c.days_left} day{"s" if c.days_left != 1 else ""}.</div></div>',
                    unsafe_allow_html=True)

    if c.days_left > 0 and (c.shortfall is None or c.shortfall == 0):
        st.markdown(
            f'<p class="as-small">Recalculated from {fmt_d(check_date)}: ({inr(a.available)} − {inr(spent)} spent) '
            f'÷ {c.days_left} day{"s" if c.days_left != 1 else ""} to payday = {inr(c.daily_until_payday)} a day, '
            f'rounded down.</p>', unsafe_allow_html=True)

    if week_over:
        st.info("For a fresh weekly amount, go back and enter your current balance.")

    if ss.checkins:
        st.markdown("#### Your check-ins")
        st.dataframe(pd.DataFrame([{"Date": fmt_dy(dd), "Spent": inr(x)} for dd, x in ss.checkins]),
                     hide_index=True, width="stretch")
        if st.button("Remove last check-in", width="stretch"):
            ss.checkins.pop()
            st.rerun()
    if st.button("Back to my answer", width="stretch"):
        go("answer")
    footer()


# =========================================================================
screens = {"start": screen_start, "confirm": screen_confirm, "answer": screen_answer, "checkin": screen_checkin}
if ss.screen != "start" and "stmt" not in ss:
    ss.screen = "start"
if ss.screen in ("answer", "checkin") and "plan" not in ss:
    ss.screen = "confirm"
if ss.screen == "checkin" and "answer" not in ss:
    ss.screen = "answer"
screens[ss.screen]()
