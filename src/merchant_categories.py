"""
merchant_categories.py — Comprehensive Indian merchant/keyword lookup.

Returns (category, subcategory) tuples for use by the 4-layer classifier.
200+ merchants covering all sample dataset entries and common Indian UPI merchants.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Business / organisation indicator tokens
# Exported for use in Layer 0 person-name detection in classifier.py
# ---------------------------------------------------------------------------
BUSINESS_INDICATORS: frozenset[str] = frozenset({
    "store", "shop", "shops", "mart", "market", "markets",
    "pvt", "pvt.", "ltd", "ltd.", "llp", "co.", "co,",
    "technologies", "technology", "tech",
    "services", "service",
    "enterprises", "enterprise",
    "solutions", "solution",
    "kitchen", "kitchens",
    "foods", "foodworks",
    "motors", "automobiles",
    "clinic", "clinics",
    "hospital", "hospitals",
    "academy", "academics",
    "institute", "institution",
    "school", "college", "university",
    "trading", "traders",
    "industries", "industry",
    "corporation", "corp",
    "incorporated", "inc",
    "limited",
    "foundation", "trust", "ngo",
    "agency", "agencies",
    "group",
    "media", "studio", "studios",
    "finance", "financial",
    "capital",
    "consultancy", "consulting",
    "management",
    "international", "global",
    "systems", "system",
    "networks", "network",
    "digital",
    "retail",
    # Food & dining business indicators
    "canteen", "supplier", "restaurant",
    "company", "mishthan", "blenders",
    "sweets", "bakes", "vypar",
    "naturals", "hospitality",
    # Local/retail business indicators
    "kirana", "medicals", "readymade",
    "parlour", "stationers", "photocopy",
    # Education / institutional
    "hall",          # "IIT Kanpur Hall Account" — prevents person-name false positive
    "cycle",         # "Kishore Cycle Store"
})

# ---------------------------------------------------------------------------
# MERCHANT_MAP — lowercase keyword → (category, subcategory)
# Keys sorted longest-first at compile time so multi-word phrases win.
# ---------------------------------------------------------------------------
MERCHANT_MAP: dict[str, tuple[str, str]] = {

    # ── Food and Dining — Food Delivery ─────────────────────────────────
    "swiggy": ("Food and Dining", "Food Delivery"),
    "zomato": ("Food and Dining", "Food Delivery"),
    "dunzo": ("Food and Dining", "Food Delivery"),
    "eatsure": ("Food and Dining", "Food Delivery"),
    "faasos": ("Food and Dining", "Food Delivery"),
    "box8": ("Food and Dining", "Food Delivery"),
    "freshmenu": ("Food and Dining", "Food Delivery"),
    "behrouz": ("Food and Dining", "Food Delivery"),
    "rebel foods": ("Food and Dining", "Food Delivery"),
    "biryani by kilo": ("Food and Dining", "Food Delivery"),
    "biryani blues": ("Food and Dining", "Food Delivery"),
    "paradise biryani": ("Food and Dining", "Food Delivery"),

    # ── Food and Dining — Restaurants (local/campus merchants) ──────────
    "rawat enterprises":             ("Food and Dining", "Restaurants"),
    "rawat enterprise":              ("Food and Dining", "Restaurants"),
    "hall canteen 02":               ("Food and Dining", "Restaurants"),
    "pawan and company":             ("Food and Dining", "Restaurants"),
    "kavita supplier":               ("Food and Dining", "Restaurants"),
    "family restaurant":             ("Food and Dining", "Restaurants"),
    "naughty blenders":              ("Food and Dining", "Restaurants"),
    "mama mio ice cream":            ("Food and Dining", "Restaurants"),
    "godavari sweets and bakes":     ("Food and Dining", "Restaurants"),
    "paras vypar":                   ("Food and Dining", "Restaurants"),
    "ooru canteen":                  ("Food and Dining", "Restaurants"),
    "burger singh iit kiosk kanpur": ("Food and Dining", "Restaurants"),
    "saga foods":                    ("Food and Dining", "Restaurants"),
    "kunwar mishthan greh":          ("Food and Dining", "Restaurants"),
    "bluedove hospitality":          ("Food and Dining", "Restaurants"),
    "roop sagar foodworks":          ("Food and Dining", "Restaurants"),

    # ── Food and Dining — Beverages (local) ──────────────────────────────
    "coffeebean tealeaf":  ("Food and Dining", "Beverages"),
    "coffee delight":      ("Food and Dining", "Beverages"),
    "marina coffee":       ("Food and Dining", "Beverages"),

    # ── Food and Dining — Restaurants ────────────────────────────────────
    "dominos pizza": ("Food and Dining", "Restaurants"),
    "domino's pizza": ("Food and Dining", "Restaurants"),
    "domino's": ("Food and Dining", "Restaurants"),
    "dominos": ("Food and Dining", "Restaurants"),
    "pizza hut": ("Food and Dining", "Restaurants"),
    "kfc": ("Food and Dining", "Restaurants"),
    "mcdonald's": ("Food and Dining", "Restaurants"),
    "mcdonald": ("Food and Dining", "Restaurants"),
    "burger singh": ("Food and Dining", "Restaurants"),
    "burger king": ("Food and Dining", "Restaurants"),
    "subway": ("Food and Dining", "Restaurants"),
    "haldirams": ("Food and Dining", "Restaurants"),
    "barbeque nation": ("Food and Dining", "Restaurants"),
    "wow momo": ("Food and Dining", "Restaurants"),
    "la pino": ("Food and Dining", "Restaurants"),
    "naturals ice cream": ("Food and Dining", "Restaurants"),
    "theobroma": ("Food and Dining", "Restaurants"),
    "punjab grill": ("Food and Dining", "Restaurants"),
    "mainland china": ("Food and Dining", "Restaurants"),

    # ── Food and Dining — Beverages ──────────────────────────────────────
    "starbucks": ("Food and Dining", "Beverages"),
    "chaayos": ("Food and Dining", "Beverages"),
    "cafe coffee day": ("Food and Dining", "Beverages"),
    "barista": ("Food and Dining", "Beverages"),
    "third wave coffee": ("Food and Dining", "Beverages"),
    "blue tokai": ("Food and Dining", "Beverages"),
    "ccd": ("Food and Dining", "Beverages"),

    # ── Shopping and Retail — Groceries ──────────────────────────────────
    "bigbasket": ("Shopping and Retail", "Groceries"),
    "big basket": ("Shopping and Retail", "Groceries"),
    "grofers": ("Shopping and Retail", "Groceries"),
    "zepto": ("Shopping and Retail", "Groceries"),
    "blinkit": ("Shopping and Retail", "Groceries"),
    "jiomart": ("Shopping and Retail", "Groceries"),
    "d-mart": ("Shopping and Retail", "Groceries"),
    "dmart": ("Shopping and Retail", "Groceries"),
    "reliance fresh": ("Shopping and Retail", "Groceries"),
    "more supermarket": ("Shopping and Retail", "Groceries"),
    "nature's basket": ("Shopping and Retail", "Groceries"),
    "lulu hypermarket": ("Shopping and Retail", "Groceries"),
    "star bazaar": ("Shopping and Retail", "Groceries"),
    "spencers": ("Shopping and Retail", "Groceries"),
    "big bazaar": ("Shopping and Retail", "Groceries"),

    # ── Shopping and Retail — Fashion ─────────────────────────────────────
    "myntra": ("Shopping and Retail", "Fashion"),
    "ajio": ("Shopping and Retail", "Fashion"),
    "nykaa fashion": ("Shopping and Retail", "Fashion"),
    "shoppers stop": ("Shopping and Retail", "Fashion"),
    "pantaloons": ("Shopping and Retail", "Fashion"),
    "zara": ("Shopping and Retail", "Fashion"),
    "h&m": ("Shopping and Retail", "Fashion"),
    "westside": ("Shopping and Retail", "Fashion"),
    "max fashion": ("Shopping and Retail", "Fashion"),
    "fabindia": ("Shopping and Retail", "Fashion"),
    "global desi": ("Shopping and Retail", "Fashion"),
    "forever 21": ("Shopping and Retail", "Fashion"),
    "adidas": ("Shopping and Retail", "Fashion"),
    "nike": ("Shopping and Retail", "Fashion"),
    "puma": ("Shopping and Retail", "Fashion"),
    "reebok": ("Shopping and Retail", "Fashion"),
    "woodland": ("Shopping and Retail", "Fashion"),

    # ── Shopping and Retail — Electronics ─────────────────────────────────
    "croma electronics": ("Shopping and Retail", "Electronics"),
    "croma": ("Shopping and Retail", "Electronics"),
    "reliance digital": ("Shopping and Retail", "Electronics"),
    "vijay sales": ("Shopping and Retail", "Electronics"),
    "apple itunes": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "samsung store": ("Shopping and Retail", "Electronics"),
    "mi store": ("Shopping and Retail", "Electronics"),

    # ── Shopping and Retail — Home & Furniture ────────────────────────────
    "ikea": ("Shopping and Retail", "Home & Furniture"),
    "pepperfry": ("Shopping and Retail", "Home & Furniture"),
    "urban ladder": ("Shopping and Retail", "Home & Furniture"),
    "hometown": ("Shopping and Retail", "Home & Furniture"),
    "nilkamal": ("Shopping and Retail", "Home & Furniture"),
    "wooden street": ("Shopping and Retail", "Home & Furniture"),

    # ── Shopping and Retail — Beauty & Personal Care ──────────────────────
    "nykaa": ("Shopping and Retail", "Beauty & Personal Care"),
    "purplle": ("Shopping and Retail", "Beauty & Personal Care"),
    "mamaearth": ("Shopping and Retail", "Beauty & Personal Care"),
    "minimalist": ("Shopping and Retail", "Beauty & Personal Care"),
    "sugar cosmetics": ("Shopping and Retail", "Beauty & Personal Care"),

    # ── Shopping and Retail — Sports & Fitness ────────────────────────────
    "decathlon": ("Shopping and Retail", "Sports & Fitness"),

    # ── Shopping and Retail — General Online ─────────────────────────────
    "amazon": ("Shopping and Retail", "General Online"),
    "flipkart": ("Shopping and Retail", "General Online"),
    "meesho": ("Shopping and Retail", "General Online"),
    "snapdeal": ("Shopping and Retail", "General Online"),
    "tatacliq": ("Shopping and Retail", "General Online"),
    "paytm mall": ("Shopping and Retail", "General Online"),

    # ── Transportation — Cab & Ride Share ────────────────────────────────
    "uber": ("Transportation", "Cab & Ride Share"),
    "ola cab": ("Transportation", "Cab & Ride Share"),
    "ola": ("Transportation", "Cab & Ride Share"),
    "rapido": ("Transportation", "Cab & Ride Share"),
    "bluesmart": ("Transportation", "Cab & Ride Share"),
    "blu smart": ("Transportation", "Cab & Ride Share"),
    "meru": ("Transportation", "Cab & Ride Share"),
    "namma yatri": ("Transportation", "Cab & Ride Share"),

    # ── Transportation — Metro & Transit ─────────────────────────────────
    "delhi metro": ("Transportation", "Metro & Transit"),
    "mumbai metro": ("Transportation", "Metro & Transit"),
    "metro card recharge": ("Transportation", "Metro & Transit"),
    "metro card": ("Transportation", "Metro & Transit"),
    "metro rail": ("Transportation", "Metro & Transit"),
    "dmrc": ("Transportation", "Metro & Transit"),
    "nmmc": ("Transportation", "Metro & Transit"),
    "bmtc": ("Transportation", "Metro & Transit"),
    "best bus": ("Transportation", "Metro & Transit"),

    # ── Transportation — Train ────────────────────────────────────────────
    "irctc": ("Transportation", "Train"),
    "indian railways": ("Transportation", "Train"),
    "railway": ("Transportation", "Train"),

    # ── Transportation — Bus ──────────────────────────────────────────────
    "redbus": ("Transportation", "Bus"),
    "abhibus": ("Transportation", "Bus"),

    # ── Transportation — Flights ──────────────────────────────────────────
    "indigo": ("Transportation", "Flights"),
    "air india": ("Transportation", "Flights"),
    "spicejet": ("Transportation", "Flights"),
    "vistara": ("Transportation", "Flights"),
    "akasa": ("Transportation", "Flights"),
    "goair": ("Transportation", "Flights"),
    "air asia": ("Transportation", "Flights"),

    # ── Transportation — Travel Booking ──────────────────────────────────
    "makemytrip": ("Transportation", "Travel Booking"),
    "goibibo": ("Transportation", "Travel Booking"),
    "yatra": ("Transportation", "Travel Booking"),
    "cleartrip": ("Transportation", "Travel Booking"),
    "easemytrip": ("Transportation", "Travel Booking"),
    "ixigo": ("Transportation", "Travel Booking"),

    # ── Transportation — Hotels ───────────────────────────────────────────
    "oyo": ("Transportation", "Hotels"),
    "treebo": ("Transportation", "Hotels"),
    "fabhotels": ("Transportation", "Hotels"),
    "marriott": ("Transportation", "Hotels"),
    "taj hotels": ("Transportation", "Hotels"),
    "hyatt": ("Transportation", "Hotels"),

    # ── Transportation — Fuel ────────────────────────────────────────────
    "iocl": ("Transportation", "Fuel"),
    "bpcl": ("Transportation", "Fuel"),
    "indian oil": ("Transportation", "Fuel"),

    # ── Transportation — Parking ─────────────────────────────────────────
    "parking charges":   ("Transportation", "Parking"),
    "parking":           ("Transportation", "Parking"),
    "kishore cycle store": ("Transportation", "Parking"),

    # ── Transportation — Toll ────────────────────────────────────────────
    "paytm fastag": ("Transportation", "Toll"),
    "fasttag": ("Transportation", "Toll"),

    # ── Utilities and Bills — Mobile & Internet ──────────────────────────
    "airtel": ("Utilities and Bills", "Mobile & Internet"),
    "reliance jio": ("Utilities and Bills", "Mobile & Internet"),
    "jio recharge": ("Utilities and Bills", "Mobile & Internet"),
    "jio": ("Utilities and Bills", "Mobile & Internet"),
    "myjio": ("Utilities and Bills", "Mobile & Internet"),
    "my jio": ("Utilities and Bills", "Mobile & Internet"),
    "vodafone": ("Utilities and Bills", "Mobile & Internet"),
    "bsnl": ("Utilities and Bills", "Mobile & Internet"),
    "act fibernet": ("Utilities and Bills", "Mobile & Internet"),
    "hathway": ("Utilities and Bills", "Mobile & Internet"),
    "tikona": ("Utilities and Bills", "Mobile & Internet"),
    "you broadband": ("Utilities and Bills", "Mobile & Internet"),
    "broadband": ("Utilities and Bills", "Mobile & Internet"),
    "vi recharge": ("Utilities and Bills", "Mobile & Internet"),

    # ── Utilities and Bills — Electricity ────────────────────────────────
    "tata power": ("Utilities and Bills", "Electricity"),
    "adani electricity": ("Utilities and Bills", "Electricity"),
    "bescom": ("Utilities and Bills", "Electricity"),
    "msedcl": ("Utilities and Bills", "Electricity"),
    "bses electricity": ("Utilities and Bills", "Electricity"),
    "bses": ("Utilities and Bills", "Electricity"),
    "cesc": ("Utilities and Bills", "Electricity"),
    "tneb": ("Utilities and Bills", "Electricity"),
    "electricity": ("Utilities and Bills", "Electricity"),

    # ── Utilities and Bills — Gas ─────────────────────────────────────────
    "mahanagar gas": ("Utilities and Bills", "Gas"),
    "indraprastha gas": ("Utilities and Bills", "Gas"),
    "igl": ("Utilities and Bills", "Gas"),
    "mgl": ("Utilities and Bills", "Gas"),
    "gas bill": ("Utilities and Bills", "Gas"),

    # ── Utilities and Bills — Water ───────────────────────────────────────
    "water bill payment": ("Utilities and Bills", "Water"),
    "water bill": ("Utilities and Bills", "Water"),
    "bwssb": ("Utilities and Bills", "Water"),
    "jal board": ("Utilities and Bills", "Water"),

    # ── Utilities and Bills — Other Bills ────────────────────────────────
    "municipal tax payment": ("Utilities and Bills", "Other Bills"),
    "municipal": ("Utilities and Bills", "Other Bills"),
    "bbmp": ("Utilities and Bills", "Other Bills"),
    "property tax": ("Utilities and Bills", "Other Bills"),
    "tata sky": ("Utilities and Bills", "Other Bills"),
    "dish tv": ("Utilities and Bills", "Other Bills"),

    # ── Entertainment and Subscriptions — OTT & Streaming ────────────────
    "netflix": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "amazon prime": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "hotstar vip": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "hotstar": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "disney+": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "disney": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "youtube premium": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "zee5 subscription": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "zee5": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "sonyliv": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "sony liv": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "jiocinema": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "hungama": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "voot": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "altbalaji": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "mxplayer": ("Entertainment and Subscriptions", "OTT & Streaming"),
    "apple tv": ("Entertainment and Subscriptions", "OTT & Streaming"),

    # ── Entertainment and Subscriptions — Music ──────────────────────────
    "spotify": ("Entertainment and Subscriptions", "Music"),
    "apple music": ("Entertainment and Subscriptions", "Music"),
    "gaana": ("Entertainment and Subscriptions", "Music"),
    "wynk": ("Entertainment and Subscriptions", "Music"),
    "jiosaavn": ("Entertainment and Subscriptions", "Music"),

    # ── Entertainment and Subscriptions — Movies & Events ────────────────
    "bookmyshow": ("Entertainment and Subscriptions", "Movies & Events"),
    "pvr cinemas": ("Entertainment and Subscriptions", "Movies & Events"),
    "pvr": ("Entertainment and Subscriptions", "Movies & Events"),
    "inox": ("Entertainment and Subscriptions", "Movies & Events"),
    "cinepolis": ("Entertainment and Subscriptions", "Movies & Events"),
    "zomato live": ("Entertainment and Subscriptions", "Movies & Events"),
    "paytm insider": ("Entertainment and Subscriptions", "Movies & Events"),

    # ── Entertainment and Subscriptions — Gaming ─────────────────────────
    "steam": ("Entertainment and Subscriptions", "Gaming"),
    "playstation": ("Entertainment and Subscriptions", "Gaming"),
    "xbox": ("Entertainment and Subscriptions", "Gaming"),
    "google play": ("Entertainment and Subscriptions", "Gaming"),
    "nintendo": ("Entertainment and Subscriptions", "Gaming"),

    # ── Healthcare and Medical — Pharmacy ────────────────────────────────
    "apollo pharmacy": ("Healthcare and Medical", "Pharmacy"),
    "tata 1mg": ("Healthcare and Medical", "Pharmacy"),
    "1mg": ("Healthcare and Medical", "Pharmacy"),
    "medplus": ("Healthcare and Medical", "Pharmacy"),
    "pharmeasy": ("Healthcare and Medical", "Pharmacy"),
    "netmeds": ("Healthcare and Medical", "Pharmacy"),
    "healthkart": ("Healthcare and Medical", "Pharmacy"),

    # ── Healthcare and Medical — Hospital & Clinic ───────────────────────
    "apollo": ("Healthcare and Medical", "Hospital & Clinic"),
    "fortis": ("Healthcare and Medical", "Hospital & Clinic"),
    "max hospital": ("Healthcare and Medical", "Hospital & Clinic"),
    "manipal": ("Healthcare and Medical", "Hospital & Clinic"),
    "columbia asia": ("Healthcare and Medical", "Hospital & Clinic"),
    "aiims": ("Healthcare and Medical", "Hospital & Clinic"),
    "practo": ("Healthcare and Medical", "Hospital & Clinic"),

    # ── Healthcare and Medical — Diagnostics ─────────────────────────────
    "dr lal pathlab": ("Healthcare and Medical", "Diagnostics"),
    "lal pathlab": ("Healthcare and Medical", "Diagnostics"),
    "dr lal": ("Healthcare and Medical", "Diagnostics"),
    "thyrocare": ("Healthcare and Medical", "Diagnostics"),
    "healthians": ("Healthcare and Medical", "Diagnostics"),
    "srl diagnostics": ("Healthcare and Medical", "Diagnostics"),

    # ── Healthcare and Medical — Fitness ─────────────────────────────────
    "cult.fit": ("Healthcare and Medical", "Fitness"),
    "curefit": ("Healthcare and Medical", "Fitness"),
    "gold's gym": ("Healthcare and Medical", "Fitness"),
    "anytime fitness": ("Healthcare and Medical", "Fitness"),

    # ── Investment and Savings — Stocks & Equity ─────────────────────────
    "nse equity purchase": ("Investment and Savings", "Stocks & Equity"),
    "nse equity": ("Investment and Savings", "Stocks & Equity"),
    "zerodha brokerage": ("Investment and Savings", "Stocks & Equity"),
    "zerodha": ("Investment and Savings", "Stocks & Equity"),
    "groww investment": ("Investment and Savings", "Stocks & Equity"),
    "groww": ("Investment and Savings", "Stocks & Equity"),
    "upstox": ("Investment and Savings", "Stocks & Equity"),
    "angelone": ("Investment and Savings", "Stocks & Equity"),
    "angel one": ("Investment and Savings", "Stocks & Equity"),
    "angel broking": ("Investment and Savings", "Stocks & Equity"),
    "5paisa": ("Investment and Savings", "Stocks & Equity"),
    "nse": ("Investment and Savings", "Stocks & Equity"),
    "bse": ("Investment and Savings", "Stocks & Equity"),

    # ── Investment and Savings — Mutual Funds ────────────────────────────
    "hdfc mutual fund sip": ("Investment and Savings", "Mutual Funds"),
    "icici mutual fund sip": ("Investment and Savings", "Mutual Funds"),
    "sbi mutual fund sip": ("Investment and Savings", "Mutual Funds"),
    "hdfc mutual fund": ("Investment and Savings", "Mutual Funds"),
    "icici mutual fund": ("Investment and Savings", "Mutual Funds"),
    "sbi mutual fund": ("Investment and Savings", "Mutual Funds"),
    "axis mutual fund": ("Investment and Savings", "Mutual Funds"),
    "kotak mutual fund": ("Investment and Savings", "Mutual Funds"),
    "paytm money sip": ("Investment and Savings", "Mutual Funds"),
    "kuvera sip": ("Investment and Savings", "Mutual Funds"),
    "kuvera": ("Investment and Savings", "Mutual Funds"),
    "coin by zerodha": ("Investment and Savings", "Mutual Funds"),
    "paytm money": ("Investment and Savings", "Mutual Funds"),
    "mutual fund": ("Investment and Savings", "Mutual Funds"),
    "smallcase": ("Investment and Savings", "Mutual Funds"),

    # ── Investment and Savings — Insurance ───────────────────────────────
    "lic premium": ("Investment and Savings", "Insurance"),
    "lic": ("Investment and Savings", "Insurance"),
    "hdfc life": ("Investment and Savings", "Insurance"),
    "icici lombard": ("Investment and Savings", "Insurance"),
    "icici prudential": ("Investment and Savings", "Insurance"),
    "sbi life": ("Investment and Savings", "Insurance"),
    "max life": ("Investment and Savings", "Insurance"),
    "star health": ("Investment and Savings", "Insurance"),
    "policybazaar": ("Investment and Savings", "Insurance"),
    "bajaj allianz": ("Investment and Savings", "Insurance"),

    # ── Investment and Savings — Fixed Income ────────────────────────────
    "fixed deposit": ("Investment and Savings", "Fixed Income"),
    "recurring deposit": ("Investment and Savings", "Fixed Income"),
    "sovereign gold": ("Investment and Savings", "Fixed Income"),
    "gold bond": ("Investment and Savings", "Fixed Income"),

    # ── Loan and EMI — Home Loan ──────────────────────────────────────────
    "sbi home loan emi": ("Loan and EMI", "Home Loan"),
    "hdfc home loan emi": ("Loan and EMI", "Home Loan"),
    "sbi home loan": ("Loan and EMI", "Home Loan"),
    "hdfc home loan": ("Loan and EMI", "Home Loan"),
    "lic housing": ("Loan and EMI", "Home Loan"),
    "home loan": ("Loan and EMI", "Home Loan"),

    # ── Loan and EMI — Car Loan ───────────────────────────────────────────
    "hdfc car loan emi": ("Loan and EMI", "Car Loan"),
    "hdfc car loan": ("Loan and EMI", "Car Loan"),
    "car loan": ("Loan and EMI", "Car Loan"),

    # ── Loan and EMI — Personal Loan ─────────────────────────────────────
    "bajaj finserv emi": ("Loan and EMI", "Personal Loan"),
    "bajaj finserv": ("Loan and EMI", "Personal Loan"),
    "personal loan": ("Loan and EMI", "Personal Loan"),
    "loan repayment": ("Loan and EMI", "Personal Loan"),
    "equitas": ("Loan and EMI", "Personal Loan"),
    "muthoot": ("Loan and EMI", "Personal Loan"),
    "manappuram": ("Loan and EMI", "Personal Loan"),

    # ── Loan and EMI — Credit Card Bill ──────────────────────────────────
    "icici credit card bill": ("Loan and EMI", "Credit Card Bill"),
    "hdfc credit card bill": ("Loan and EMI", "Credit Card Bill"),
    "sbi credit card bill": ("Loan and EMI", "Credit Card Bill"),
    "credit card bill": ("Loan and EMI", "Credit Card Bill"),
    "credit card payment": ("Loan and EMI", "Credit Card Bill"),
    "card statement": ("Loan and EMI", "Credit Card Bill"),

    # ── Education ────────────────────────────────────────────────────────
    "coursera": ("Education", "Online Learning"),
    "udemy": ("Education", "Online Learning"),
    "unacademy": ("Education", "Online Learning"),
    "byju": ("Education", "Online Learning"),
    "vedantu": ("Education", "Online Learning"),
    "upgrad": ("Education", "Online Learning"),
    "toppr": ("Education", "Online Learning"),
    "physicswallah": ("Education", "Online Learning"),
    "chegg": ("Education", "Online Learning"),
    "duolingo": ("Education", "Online Learning"),
    "school fee": ("Education", "School & College"),
    "college fee": ("Education", "School & College"),
    "tuition": ("Education", "School & College"),
    "university fee": ("Education", "School & College"),
    # IIT Kanpur — hall-fee payments are common in GPay statements
    "indian institute of technology kanpur hall account": ("Education", "School & College"),
    "indian institute of technology kanpur":              ("Education", "School & College"),
    "iit kanpur hall account":                            ("Education", "School & College"),
    "iit kanpur":                                         ("Education", "School & College"),

    # ── Transfer and Banking — ATM ────────────────────────────────────────
    "atm withdrawal": ("Transfer and Banking", "ATM Withdrawal"),
    "atm": ("Transfer and Banking", "ATM Withdrawal"),

    # ── Transfer and Banking — Cash ───────────────────────────────────────
    "cash deposit": ("Transfer and Banking", "Cash Deposit"),

    # ── Transfer and Banking — Wallet Topup ──────────────────────────────
    "amazon pay": ("Transfer and Banking", "Wallet Topup"),
    "paytm wallet": ("Transfer and Banking", "Wallet Topup"),
    "phonepe wallet": ("Transfer and Banking", "Wallet Topup"),
    "mobikwik": ("Transfer and Banking", "Wallet Topup"),
    "freecharge": ("Transfer and Banking", "Wallet Topup"),

    # ── Transfer and Banking — Bank Transfer ─────────────────────────────
    "fund transfer": ("Transfer and Banking", "Bank Transfer"),
    "bank transfer": ("Transfer and Banking", "Bank Transfer"),

    # ── Income and Salary ─────────────────────────────────────────────────
    "monthly salary neft": ("Income and Salary", "Salary"),
    "monthly salary": ("Income and Salary", "Salary"),
    "salary credit": ("Income and Salary", "Salary"),
    "salary": ("Income and Salary", "Salary"),
    "payroll": ("Income and Salary", "Salary"),
    "stipend": ("Income and Salary", "Salary"),
    "bonus": ("Income and Salary", "Salary"),
    "reimbursement": ("Income and Salary", "Salary"),
    "freelance": ("Income and Salary", "Freelance"),
    "project payment": ("Income and Salary", "Freelance"),
    "interest credit": ("Income and Salary", "Investment Returns"),
    "dividend credit": ("Income and Salary", "Investment Returns"),
    "dividend": ("Income and Salary", "Investment Returns"),
    # Note: cashback/refund/reversal are handled by global override in classifier.py
    # but listed here as fallback if the override is somehow bypassed
    "cashback credit": ("Income and Salary", "Refund"),
    "cashback": ("Income and Salary", "Refund"),
    "refund": ("Income and Salary", "Refund"),
    "reversal": ("Income and Salary", "Refund"),
}

# ---------------------------------------------------------------------------
# Compile a single regex — longest keys first so multi-word phrases win
# ---------------------------------------------------------------------------
_SORTED_KEYS: list[str] = sorted(MERCHANT_MAP.keys(), key=len, reverse=True)

# Keys that need explicit word-boundary protection (short / potentially ambiguous)
_NEEDS_BOUNDARY: frozenset[str] = frozenset({
    "ola", "nse", "bse", "igl", "mgl", "pvr", "kfc", "ccd", "oyo",
    "emi", "sip", "ppf", "nps", "bse", "fd",
})

_PATTERN_PARTS: list[str] = []
for _k in _SORTED_KEYS:
    escaped = re.escape(_k)
    if _k in _NEEDS_BOUNDARY or (len(_k) <= 3 and _k.isalpha()):
        _PATTERN_PARTS.append(r"\b" + escaped + r"\b")
    else:
        _PATTERN_PARTS.append(escaped)

_MERCHANT_PATTERN: re.Pattern[str] = re.compile(
    "|".join(_PATTERN_PARTS),
    re.IGNORECASE,
)


def get_category(description: str) -> Optional[tuple[str, str]]:
    """
    Look up the (category, subcategory) tuple for a transaction description.

    Uses longest-match regex over MERCHANT_MAP. Returns None if no merchant
    keyword is recognised — the caller should then continue to the next layer.

    Parameters
    ----------
    description : str
        Transaction description / merchant name (any case).

    Returns
    -------
    tuple[str, str]  — (category, subcategory)
    None             — no match found
    """
    if not description:
        return None
    m = _MERCHANT_PATTERN.search(description)
    if m:
        keyword = m.group(0).lower()
        return MERCHANT_MAP.get(keyword)
    return None
