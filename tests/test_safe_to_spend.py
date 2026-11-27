"""Tests for src.safe_to_spend — the formula, detection rules and edge cases."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.safe_to_spend import (
    StatementError, _build_statement, add_months, bills_before_payday, check_in,
    detect_recurring, detect_salary, inr, load_statement, next_payday, one_change,
    safe_to_spend, tag_bill_rows, usual_weekly_spend, Recurring,
)

SAMPLE = Path(__file__).parent.parent / "data" / "sample_transactions.csv"


def _stmt(rows):
    df = pd.DataFrame(rows, columns=["date", "description", "debit", "credit", "balance"])
    df["date"] = pd.to_datetime(df["date"])
    return _build_statement(df, "CSV")


# --------------------------------------------------------------------- formula

class TestFormula:
    def test_basic(self):
        a = safe_to_spend(31524, 399, 2000, date(2026, 11, 20), date(2026, 12, 1))
        assert a.days == 11 and a.horizon == 7
        assert a.available == 29125
        assert a.weekly == 18534          # floor(29125 / 11 * 7) = floor(18534.09)
        assert a.shortfall is None

    def test_fewer_than_seven_days_uses_all_days(self):
        a = safe_to_spend(10000, 0, 1000, date(2026, 11, 27), date(2026, 11, 30))
        assert a.days == 3 and a.horizon == 3
        assert a.weekly == 9000           # whole remainder is for these 3 days

    def test_one_day(self):
        a = safe_to_spend(5000, 0, 0, date(2026, 11, 30), date(2026, 12, 1))
        assert a.weekly == 5000

    def test_shortfall_is_reported_not_negative(self):
        a = safe_to_spend(3000, 2500, 2000, date(2026, 11, 20), date(2026, 12, 1))
        assert a.weekly == 0 and a.daily == 0
        assert a.shortfall == 1500

    def test_exactly_zero_left(self):
        a = safe_to_spend(2000, 0, 2000, date(2026, 11, 20), date(2026, 12, 1))
        assert a.weekly == 0 and a.shortfall == 0

    @pytest.mark.parametrize("payday", [date(2026, 11, 20), date(2026, 11, 19)])
    def test_zero_or_negative_days_raise(self, payday):
        with pytest.raises(ValueError):
            safe_to_spend(10000, 0, 0, date(2026, 11, 20), payday)

    def test_missing_balance_raises(self):
        with pytest.raises(ValueError):
            safe_to_spend(float("nan"), 0, 0, date(2026, 11, 20), date(2026, 12, 1))

    def test_negative_buffer_raises(self):
        with pytest.raises(ValueError):
            safe_to_spend(1000, 0, -5, date(2026, 11, 20), date(2026, 12, 1))

    def test_weekly_never_exceeds_available(self):
        for days in range(1, 40):
            a = safe_to_spend(12345, 100, 2000, date(2026, 1, 1), date(2026, 1, 1) + timedelta(days=days))
            assert 0 <= a.weekly <= a.available


# --------------------------------------------------------------------- check-in

class TestCheckIn:
    def setup_method(self):
        self.today, self.payday = date(2026, 11, 20), date(2026, 12, 1)
        self.a = safe_to_spend(31524, 399, 2000, self.today, self.payday)

    def test_under_limit(self):
        c = check_in(self.a, 5000, date(2026, 11, 23), self.today, self.payday)
        assert c.remaining_week == 13534 and c.over_by is None
        assert c.days_left == 8
        assert c.daily_until_payday == (29125 - 5000) // 8

    def test_over_limit_recalculates_daily(self):
        c = check_in(self.a, 20000, date(2026, 11, 24), self.today, self.payday)
        assert c.over_by == 20000 - 18534
        assert c.daily_until_payday == (29125 - 20000) // 7

    def test_spent_everything(self):
        c = check_in(self.a, 30000, date(2026, 11, 24), self.today, self.payday)
        assert c.daily_until_payday == 0 and c.shortfall == 875

    def test_date_before_start_rejected(self):
        with pytest.raises(ValueError):
            check_in(self.a, 10, date(2026, 11, 19), self.today, self.payday)


# --------------------------------------------------------------------- dates

def test_add_months_clips_month_end():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)


def test_next_payday_rule_and_roll_forward():
    assert next_payday(date(2026, 11, 1), date(2026, 11, 20)) == date(2026, 12, 1)
    # stale statement: roll forward past "today"
    assert next_payday(date(2026, 9, 30), date(2026, 11, 20)) == date(2026, 11, 30)


def test_inr_format():
    assert inr(123456.4) == "₹1,23,456"
    assert inr(999) == "₹999"
    assert inr(-15000) == "-₹15,000"
    assert inr(10000000) == "₹1,00,00,000"


# --------------------------------------------------------------------- the sample

@pytest.fixture(scope="module")
def st():
    return load_statement(SAMPLE)


class TestSample:
    def test_balance_and_dates(self, st):
        assert st.end == date(2026, 11, 20) and st.balance_date == st.end
        assert st.balance_reliable

    def test_salary(self, st):
        s = detect_salary(st.txns)
        assert s.amount == 58400 and s.last_date == date(2026, 11, 1)

    def test_friend_payback_is_not_salary(self, st):
        s = detect_salary(st.txns)
        assert s.count == 6

    def test_self_transfers_flagged_and_excluded(self, st):
        t = st.txns
        assert t[t["description"].str.contains("Self", case=False)]["is_self"].all()
        rows = tag_bill_rows(t, detect_recurring(t, st.end))
        usual, _ = usual_weekly_spend(t, rows, st.end)
        assert usual < 10000   # would be far higher if transfers/rent counted

    def test_recurring_found(self, st):
        groups = {r.group for r in detect_recurring(st.txns, st.end)}
        assert {"Rent", "EMI / loan", "Bills and subscriptions", "Own-account transfer"} <= groups
        names = " ".join(r.name.lower() for r in detect_recurring(st.txns, st.end))
        for word in ("rent", "emi", "netflix", "spotify", "jio", "broadband"):
            assert word in names
        assert "swiggy" not in names and "atm" not in names

    def test_bills_before_payday(self, st):
        rec = detect_recurring(st.txns, st.end)
        bills = bills_before_payday(rec, st.end, date(2026, 12, 1))
        assert [(b.name, b.amount, b.status) for b in bills] == [("Jio Postpaid Bill", 399, "due")]

    def test_tip_present_with_8_weeks(self, st):
        rows = tag_bill_rows(st.txns, detect_recurring(st.txns, st.end))
        tip = one_change(st.txns, rows, st.end)
        assert tip is not None and tip.weeks == 8
        assert tip.group in ("Food delivery", "Eating out", "Shopping", "Entertainment")
        assert tip.frees == int(tip.weekly_avg * 0.25)


# --------------------------------------------------------------------- bill windows

def _rec(day, last, group="Bills and subscriptions"):
    return Recurring("X", group, 500, day, last, 3, "x")


def test_bill_due_after_balance_date_before_payday():
    b = bills_before_payday([_rec(25, date(2026, 10, 25))], date(2026, 11, 20), date(2026, 12, 1))
    assert b[0].status == "due" and b[0].due == date(2026, 11, 25)


def test_unpaid_overdue_bill_included():
    b = bills_before_payday([_rec(10, date(2026, 10, 10))], date(2026, 11, 20), date(2026, 12, 1))
    assert b[0].status == "overdue"


def test_payday_bill_separate():
    b = bills_before_payday([_rec(1, date(2026, 11, 1))], date(2026, 11, 20), date(2026, 12, 1))
    assert b[0].status == "payday"


def test_bill_after_payday_dropped():
    assert bills_before_payday([_rec(5, date(2026, 11, 5))], date(2026, 11, 20), date(2026, 12, 1)) == []


# --------------------------------------------------------------------- bad / thin data

def test_short_history_hides_tip_and_notes():
    st = _stmt([
        ("2026-11-01", "SALARY CREDIT", 0, 40000, 40000),
        ("2026-11-03", "SWIGGY ORDER", 500, 0, 39500),
        ("2026-11-10", "ZOMATO ORDER", 400, 0, 39100),
    ])
    assert any("less than 4 weeks" in n for n in st.notes)
    rows = tag_bill_rows(st.txns, [])
    assert one_change(st.txns, rows, st.end) is None


def test_no_salary_found():
    st = _stmt([
        ("2026-11-01", "UPI FROM FRIEND", 0, 2000, 2000),
        ("2026-11-03", "SWIGGY ORDER", 500, 0, 1500),
    ])
    assert detect_salary(st.txns) is None


def test_salary_by_pattern_without_keyword():
    st = _stmt([
        ("2026-09-30", "NEFT CR ACME CORP", 0, 45000, 45000),
        ("2026-10-31", "NEFT CR ACME CORP", 0, 45000, 90000),
        ("2026-11-05", "SWIGGY ORDER", 500, 0, 89500),
    ])
    s = detect_salary(st.txns)
    assert s is not None and s.method == "pattern" and s.last_date == date(2026, 10, 31)


def test_missing_balance_column():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-11-01", "2026-11-02"]),
                       "description": ["A", "B"], "debit": [0, 100], "credit": [1000, 0]})
    st = _build_statement(df, "PDF")
    assert st.balance is None and any("balance" in n for n in st.notes)


def test_inconsistent_balance_flagged():
    st = _stmt([
        ("2026-11-01", "A", 0, 1000, 1000),
        ("2026-11-02", "B", 100, 0, 5000),
        ("2026-11-03", "C", 100, 0, 200),
    ])
    assert st.balance_reliable is False


def test_paired_transfer_out_and_back_is_self():
    st = _stmt([
        ("2026-11-01", "IMPS TRANSFER 99881", 7000, 0, 3000),
        ("2026-11-01", "IMPS TRANSFER 99882", 0, 7000, 10000),
        ("2026-11-02", "UPI SWIGGY", 300, 0, 9700),
    ])
    assert st.txns["is_self"].tolist() == [True, True, False]


def test_bad_files(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(StatementError):
        load_statement(p)
    q = tmp_path / "junk.csv"
    q.write_text("foo,bar\n1,2\n")
    with pytest.raises(StatementError):
        load_statement(q)
    pdf = tmp_path / "statement.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(StatementError, match="PDF"):
        load_statement(pdf)
    e = tmp_path / "empty.csv"
    e.write_text("")
    with pytest.raises(StatementError):
        load_statement(e)


def test_same_day_identical_purchases_kept(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("Date,Description,Debit,Credit,Balance\n"
                 "01-11-2026,SALARY CREDIT,,30000,30000\n"
                 "02-11-2026,CHAI POINT,60,,29940\n"
                 "02-11-2026,CHAI POINT,60,,29880\n")
    st = load_statement(p)
    assert len(st.txns) == 3 and st.balance == 29880


def test_newest_first_statement_balance(tmp_path):
    p = tmp_path / "desc.csv"
    p.write_text("Date,Description,Debit,Credit,Balance\n"
                 "03-11-2026,CAFE,50,,29850\n"
                 "02-11-2026,CHAI,100,,29900\n"
                 "01-11-2026,SALARY CREDIT,,30000,30000\n")
    st = load_statement(p)
    assert st.balance == 29850 and st.end == date(2026, 11, 3)


FAKE = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["fake_newest_first.csv", "fake_statement.xlsx"])
def test_fake_bank_layouts(name):
    """HDFC-style headers ("Withdrawal Amt."), newest-first rows, salary without a keyword."""
    s = load_statement(FAKE / name)
    t = s.txns
    assert len(t) == 105 and s.balance == 25485 and s.end == date(2026, 11, 24)
    assert t["debit"].sum() == 109515 and t["credit"].sum() == 126000
    sal = detect_salary(t)
    assert sal.method == "pattern" and sal.last_date == date(2026, 10, 31)
    pay = next_payday(sal.last_date, s.end)
    bills = bills_before_payday(detect_recurring(t, s.end), s.end, pay)
    assert [(b.group, b.amount) for b in bills] == [("Own-account transfer", 4000), ("Bills and subscriptions", 649)]
    a = safe_to_spend(s.balance, 4649, 2000, s.end, pay)
    assert a.days == 6 and a.horizon == 6 and a.weekly == 18836
