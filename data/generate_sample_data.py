"""
Generates a realistic 800-row synthetic Indian bank statement CSV.
Deterministic output using random.seed(42).
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

MERCHANTS = {
    "Food and Dining": [
        "SWIGGY ORDER #{num}", "ZOMATO FOOD DEL #{num}", "DOMINOS PIZZA #{num}",
        "MCDONALD'S POS", "STARBUCKS INDIA", "CHAAYOS TEA", "HALDIRAMS POS",
        "BARBEQUE NATION", "KFC ORDER #{num}", "CAFE COFFEE DAY",
        "BIRYANI BY KILO #{num}", "SUBWAY POS", "PIZZA HUT ORDER #{num}",
    ],
    "Transportation": [
        "UBER AUTO #{num}", "OLA CAB #{num}", "IRCTC TICKET #{num}",
        "RAPIDO BIKE #{num}", "METRO CARD RECHARGE", "INDIAN RAILWAYS #{num}",
        "MAKEMYTRIP FLIGHT", "REDBUS BOOKING #{num}", "DELHI METRO TOPUP",
        "PARKING CHARGES POS",
    ],
    "Utilities and Bills": [
        "AIRTEL RECHARGE", "JIO RECHARGE", "VI RECHARGE",
        "TATA POWER BILL", "MAHANAGAR GAS BILL", "BSES ELECTRICITY",
        "BROADBAND BILL ACT", "MUNICIPAL TAX PAYMENT", "WATER BILL PAYMENT",
    ],
    "Entertainment and Subscriptions": [
        "NETFLIX INDIA", "SPOTIFY INDIA", "AMAZON PRIME", "HOTSTAR VIP",
        "YOUTUBE PREMIUM", "SONY LIV", "BOOKMYSHOW #{num}", "PVR CINEMAS POS",
        "ZEE5 SUBSCRIPTION", "APPLE ITUNES",
    ],
    "Shopping and Retail": [
        "AMAZON PAY #{num}", "FLIPKART #{num}", "MYNTRA #{num}",
        "BIG BAZAAR POS", "RELIANCE FRESH POS", "DMART POS",
        "CROMA ELECTRONICS", "NYKAA ORDER #{num}", "AJIO ORDER #{num}",
        "DECATHLON POS", "IKEA POS", "SHOPPERS STOP POS",
    ],
    "Healthcare and Medical": [
        "APOLLO PHARMACY POS", "MEDPLUS POS", "PHARMEASY ORDER #{num}",
        "NETMEDS ORDER #{num}", "DR LAL PATHLAB", "MAX HOSPITAL",
        "FORTIS HEALTHCARE", "1MG ORDER #{num}",
    ],
    "Investment and Savings": [
        "ZERODHA BROKERAGE", "GROWW INVESTMENT", "ICICI MUTUAL FUND SIP",
        "SBI MUTUAL FUND SIP", "NSE EQUITY PURCHASE", "KUVERA SIP",
        "PAYTM MONEY SIP", "HDFC MUTUAL FUND SIP",
    ],
    "Loan and EMI": [
        "SBI HOME LOAN EMI", "HDFC CAR LOAN EMI", "LIC PREMIUM",
        "BAJAJ FINSERV EMI", "ICICI CREDIT CARD BILL",
    ],
    "Transfer and Banking": [
        "ATM WITHDRAWAL", "NEFT TO #{num}", "IMPS TO #{num}",
        "UPI TO #{num}", "POS PURCHASE #{num}", "CASH DEPOSIT",
        "RTGS TRANSFER #{num}",
    ],
}

INCOME_DESCRIPTIONS = [
    "SALARY CREDIT", "SALARY CREDIT", "SALARY CREDIT",
    "FREELANCE PAYMENT NEFT", "SBI NEFT FROM #{num}",
    "INTEREST CREDIT", "CASHBACK CREDIT", "REFUND CREDIT",
    "DIVIDEND CREDIT",
]

AMOUNT_RANGES = {
    "Food and Dining": (80, 2500),
    "Transportation": (30, 3500),
    "Utilities and Bills": (200, 3000),
    "Entertainment and Subscriptions": (149, 1500),
    "Shopping and Retail": (200, 8000),
    "Healthcare and Medical": (100, 5000),
    "Investment and Savings": (500, 10000),
    "Loan and EMI": (2000, 15000),
    "Transfer and Banking": (500, 10000),
}

CATEGORY_WEIGHTS = {
    "Food and Dining": 25,
    "Transportation": 15,
    "Shopping and Retail": 15,
    "Utilities and Bills": 8,
    "Entertainment and Subscriptions": 8,
    "Transfer and Banking": 10,
    "Healthcare and Medical": 5,
    "Investment and Savings": 6,
    "Loan and EMI": 8,
}


def generate_transactions():
    transactions = []
    balance = 85000.0
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)

    # Generate salary credits on 1st of each month
    for month in range(1, 7):
        salary_date = datetime(2024, month, 1)
        salary_amount = round(random.uniform(45000, 55000), 2)
        balance += salary_amount
        desc = random.choice(["SALARY CREDIT", "SALARY CREDIT FROM EMPLOYER", "MONTHLY SALARY NEFT"])
        transactions.append({
            "date": salary_date,
            "description": desc,
            "debit": "",
            "credit": salary_amount,
            "balance": round(balance, 2),
        })

    # Generate additional income (freelance, interest, etc.) - ~2-3 per month
    for month in range(1, 7):
        num_extra_income = random.randint(1, 3)
        for _ in range(num_extra_income):
            day = random.randint(5, 28)
            dt = datetime(2024, month, day)
            amount = round(random.uniform(2000, 15000), 2)
            desc_template = random.choice(INCOME_DESCRIPTIONS)
            desc = desc_template.replace("#{num}", str(random.randint(100000, 999999)))
            balance += amount
            transactions.append({
                "date": dt,
                "description": desc,
                "debit": "",
                "credit": amount,
                "balance": round(balance, 2),
            })

    # Generate debit transactions to reach ~800 total
    target_debits = 800 - len(transactions)
    categories = list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in categories]

    for _ in range(target_debits):
        category = random.choices(categories, weights=weights, k=1)[0]
        month = random.randint(1, 6)
        day = random.randint(1, 28)
        dt = datetime(2024, month, day)

        merchant_template = random.choice(MERCHANTS[category])
        merchant = merchant_template.replace("#{num}", str(random.randint(1000, 999999)))

        lo, hi = AMOUNT_RANGES[category]
        amount = round(random.uniform(lo, hi), 2)

        balance -= amount
        transactions.append({
            "date": dt,
            "description": merchant,
            "debit": amount,
            "credit": "",
            "balance": round(balance, 2),
        })

    # Insert anomalies
    anomalies = []

    # Anomaly 1: Unusually large food transaction
    anom_date = datetime(2024, 3, 15)
    balance -= 18500
    anomalies.append({
        "date": anom_date,
        "description": "UNKNOWN MERCHANT POS 99281",
        "debit": 18500.00,
        "credit": "",
        "balance": round(balance, 2),
    })

    # Anomaly 2: Unusually large transaction to obscure merchant
    anom_date = datetime(2024, 5, 22)
    balance -= 24999
    anomalies.append({
        "date": anom_date,
        "description": "GLOBALTECH SERVICES PVT LTD",
        "debit": 24999.00,
        "credit": "",
        "balance": round(balance, 2),
    })

    # Anomaly 3: Duplicate transaction (same merchant + amount within same week)
    anom_date1 = datetime(2024, 4, 10)
    anom_date2 = datetime(2024, 4, 12)
    balance -= 4500 * 2
    anomalies.append({
        "date": anom_date1,
        "description": "CROMA ELECTRONICS POS",
        "debit": 4500.00,
        "credit": "",
        "balance": round(balance, 2),
    })
    anomalies.append({
        "date": anom_date2,
        "description": "CROMA ELECTRONICS POS",
        "debit": 4500.00,
        "credit": "",
        "balance": round(balance, 2),
    })

    # Anomaly 4: Large unknown merchant
    anom_date = datetime(2024, 2, 8)
    balance -= 21000
    anomalies.append({
        "date": anom_date,
        "description": "OBSCURE TRADING CO NEFT",
        "debit": 21000.00,
        "credit": "",
        "balance": round(balance, 2),
    })

    transactions.extend(anomalies)

    # Sort by date
    transactions.sort(key=lambda x: x["date"])

    # Recalculate running balance
    balance = 85000.0
    for t in transactions:
        if t["credit"] != "":
            balance += float(t["credit"])
        if t["debit"] != "":
            balance -= float(t["debit"])
        t["balance"] = round(balance, 2)

    return transactions


def write_csv(transactions, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Description", "Debit", "Credit", "Balance"])
        writer.writeheader()
        for t in transactions:
            writer.writerow({
                "Date": t["date"].strftime("%d-%m-%Y"),
                "Description": t["description"],
                "Debit": t["debit"] if t["debit"] != "" else "",
                "Credit": t["credit"] if t["credit"] != "" else "",
                "Balance": t["balance"],
            })


def print_stats(transactions):
    total_debit = sum(float(t["debit"]) for t in transactions if t["debit"] != "")
    total_credit = sum(float(t["credit"]) for t in transactions if t["credit"] != "")
    anomaly_count = 5  # We inserted 5 anomalous transactions
    print(f"Transaction count: {len(transactions)}")
    print(f"Total debit: ₹{total_debit:,.2f}")
    print(f"Total credit: ₹{total_credit:,.2f}")
    print(f"Net cashflow: ₹{total_credit - total_debit:,.2f}")
    print(f"Anomaly count: {anomaly_count}")
    print(f"Date range: {transactions[0]['date'].strftime('%d-%m-%Y')} to {transactions[-1]['date'].strftime('%d-%m-%Y')}")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "sample_transactions.csv"
    transactions = generate_transactions()
    write_csv(transactions, output_path)
    print_stats(transactions)
    print(f"\nCSV saved to: {output_path}")
