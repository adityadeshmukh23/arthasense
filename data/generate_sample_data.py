"""
Generate the SYNTHETIC sample statement used by the "Try sample" button.

Everything here is made up. It describes one fictional salaried person in
their second year of work in Bengaluru. It is not based on any real person
or real account.

Profile (per full month):
  - Salary  ₹58,400 credited on the 1st (same amount every month)
  - Rent    ₹16,000 on the 3rd
  - Own-account transfer ₹5,000 on the 2nd (moving savings, not spending)
  - Phone EMI ₹3,450 on the 5th
  - Broadband ₹589 (10th), Netflix ₹199 (12th), Spotify ₹119 (18th),
    Jio postpaid ₹399 (22nd); electricity varies (around the 15th)
  - Day-to-day spending: groceries, food delivery, transport, shopping, cash
  - Saves roughly 12–17 % of salary in each full month (June–October)
  - One-off in November: ₹23,000 dental treatment on 9 Nov, which leaves
    the account tight until the 1 Dec salary

Statement period: 21 May 2026 – 20 Nov 2026 (six months).
Output: data/sample_transactions.csv  (Date,Description,Debit,Credit,Balance)
Deterministic: random.seed(3).
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(3)

START = date(2026, 5, 21)
END = date(2026, 11, 20)
OPENING_BALANCE = 12_000.00

SALARY = 58_400.00
FIXED = [
    # (day, description, amount)
    (1, "NEFT CR-NORTHWIND SOFTWARE PVT LTD-SALARY", None),  # salary, credit
    (2, "IMPS-SELF TRANSFER-OWN A/C XX4410", 5_000.00),
    (3, "UPI-HOUSE RENT-ownerrent@okaxis", 16_000.00),
    (5, "ACH DR-BAJAJ FINANCE LTD-PHONE EMI", 3_450.00),
    (10, "ACT FIBERNET BROADBAND BILL", 589.00),
    (12, "NETFLIX.COM SUBSCRIPTION", 199.00),
    (18, "SPOTIFY INDIA SUBSCRIPTION", 119.00),
    (22, "JIO POSTPAID BILL", 399.00),
]

# (description options, (min, max) amount, (min, max) count per month)
VARIABLE = [
    (["BLINKIT ORDER", "BIGBASKET ORDER", "DMART POS", "ZEPTO ORDER"], (350, 1400), (8, 10)),
    (["SWIGGY ORDER", "ZOMATO ORDER"], (220, 620), (8, 11)),
    (["UBER TRIP", "OLA RIDE", "RAPIDO BIKE", "NAMMA METRO RECHARGE"], (90, 420), (10, 14)),
    (["AMAZON PAY ORDER", "MYNTRA ORDER", "FLIPKART ORDER"], (450, 2400), (2, 4)),
    (["TOIT BREWPUB", "MEGHANA FOODS", "THIRD WAVE COFFEE", "TRUFFLES"], (450, 1600), (3, 5)),
    (["UPI-CHAI POINT", "UPI-LOCAL KIRANA STORE", "UPI-MEDPLUS PHARMACY",
      "UPI-SAGAR CAFE", "UPI-LAUNDRY SERVICE"], (40, 450), (6, 10)),
    (["PVR CINEMAS", "BOOKMYSHOW"], (250, 700), (0, 1)),
]
ATM = ("ATM WDL-HDFC ATM KORAMANGALA", 3_000.00)
ONE_OFF = (date(2026, 11, 9), "MANIPAL HOSPITALS DENTAL TREATMENT", 23_000.00)


def _month_starts(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = date(d.year + (d.month // 12), d.month % 12 + 1, 1)


def _days_in_month(d: date) -> int:
    nxt = date(d.year + (d.month // 12), d.month % 12 + 1, 1)
    return (nxt - d).days


def generate_transactions() -> list[dict]:
    txns: list[dict] = []
    for m in _month_starts(START, END):
        ndays = _days_in_month(m)
        # Fixed items
        for day, desc, amt in FIXED:
            d = m.replace(day=day)
            if not (START <= d <= END):
                continue
            if amt is None:
                txns.append({"date": d, "description": desc, "debit": None, "credit": SALARY})
            else:
                txns.append({"date": d, "description": desc, "debit": amt, "credit": None})
        # Electricity (varies with season)
        d = m.replace(day=15)
        if START <= d <= END:
            txns.append({"date": d, "description": "BESCOM ELECTRICITY BILL",
                         "debit": float(random.randint(780, 1350)), "credit": None})
        # ATM cash on the 7th
        d = m.replace(day=7)
        if START <= d <= END:
            txns.append({"date": d, "description": ATM[0], "debit": ATM[1], "credit": None})
        # Variable spending, scaled to the part of the month inside the period
        lo_day = START.day if (m.year, m.month) == (START.year, START.month) else 1
        hi_day = END.day if (m.year, m.month) == (END.year, END.month) else ndays
        share = (hi_day - lo_day + 1) / ndays
        for options, (amin, amax), (cmin, cmax) in VARIABLE:
            n = round(random.randint(cmin, cmax) * share)
            for _ in range(n):
                day = random.randint(lo_day, hi_day)
                txns.append({
                    "date": m.replace(day=day),
                    "description": random.choice(options),
                    "debit": float(random.randint(amin, amax)),
                    "credit": None,
                })
        # Occasional small credit: a friend settling a shared bill
        if random.random() < 0.5:
            day = random.randint(lo_day, hi_day)
            txns.append({"date": m.replace(day=day),
                         "description": "UPI CR-SPLIT BILL SETTLEMENT FROM FRIEND",
                         "debit": None, "credit": float(random.randint(300, 900))})

    # One-off November event: an unplanned dental treatment paid in full.
    # This is what leaves the account tight before the next salary.
    txns.append({"date": ONE_OFF[0], "description": ONE_OFF[1], "debit": ONE_OFF[2], "credit": None})

    # Make sure the statement ends on END (a small last-day purchase)
    txns.append({"date": END, "description": "UPI-CHAI POINT", "debit": 60.0, "credit": None})

    # Stable order: by date, credits first on the same day
    txns.sort(key=lambda t: (t["date"], t["credit"] is None))
    bal = OPENING_BALANCE
    for t in txns:
        bal += (t["credit"] or 0) - (t["debit"] or 0)
        t["balance"] = round(bal, 2)
    return txns


def write_csv(txns: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        for t in txns:
            w.writerow([
                t["date"].strftime("%d-%m-%Y"),
                t["description"],
                f"{t['debit']:.2f}" if t["debit"] else "",
                f"{t['credit']:.2f}" if t["credit"] else "",
                f"{t['balance']:.2f}",
            ])


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_transactions.csv"
    rows = generate_transactions()
    write_csv(rows, out)
    print(f"Wrote {len(rows)} synthetic rows to {out}")
