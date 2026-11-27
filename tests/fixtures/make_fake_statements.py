"""Fake statement for testing only. Made-up person and numbers."""
import csv, random
from pathlib import Path
OUT = Path(__file__).parent
from datetime import date, timedelta
random.seed(11)
rows=[]; bal=9000.0
def add(d,desc,dr=0,cr=0):
    global bal; bal+=cr-dr; rows.append([d,desc,dr,cr,round(bal,2)])
d=date(2026,8,29)
while d<=date(2026,11,24):
    if d in (date(2026,8,31),date(2026,9,30),date(2026,10,31)): add(d,"NEFT CR-GLOBEX INDIA PVT LTD",cr=42000)
    if d.day==5: add(d,"UPI/DR/600112233441/RENT PAYMENT/landlord@ybl",dr=12000)
    if d.day==10: add(d,"NACH DR HDFC PERSONAL LOAN EMI",dr=6200)
    if d.day==25: add(d,"TRF TO OWN A/C 0099",dr=4000)
    if d.day==28: add(d,"NETFLIX.COM",dr=649)
    if d.day==12: add(d,"AIRTEL POSTPAID",dr=599)
    for _ in range(random.choice([0,1,1,2])):
        desc,lo,hi=random.choice([("UPI/DR/SWIGGY",150,600),("UPI/DR/ZOMATO",150,550),("UPI/DR/ZEPTO",200,900),("UPI/DR/UBER",90,400),("UPI/DR/AMAZON",300,1800),("UPI/DR/STARBUCKS",250,600)])
        add(d,f"{desc}/{random.randint(10**11,10**12-1)}",dr=float(random.randint(lo,hi)))
    d+=timedelta(days=1)
print("rows",len(rows),"end bal",bal)
hdr=["Txn Date","Narration","Withdrawal Amt.","Deposit Amt.","Closing Balance"]
fmt=lambda r:[r[0].strftime("%d/%m/%Y"),r[1],f"{r[2]:.2f}" if r[2] else "",f"{r[3]:.2f}" if r[3] else "",f"{r[4]:,.2f}"]
with open(str(OUT / "fake_newest_first.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr); [w.writerow(fmt(r)) for r in reversed(rows)]
import openpyxl
wb=openpyxl.Workbook(); ws=wb.active; ws.append(["FAKE BANK LTD - STATEMENT (TEST DATA)"]); ws.append([]); ws.append(hdr)
for r in rows: ws.append([r[0].strftime("%d/%m/%Y"),r[1],r[2] or None,r[3] or None,r[4]])
wb.save(str(OUT / "fake_statement.xlsx"))
from fpdf import FPDF
pdf=FPDF(); pdf.add_page(); pdf.set_font("Helvetica",size=8)
pdf.cell(0,6,"FAKE BANK LTD  Account Statement  (test data, not a real account)",new_x="LMARGIN",new_y="NEXT")
pdf.cell(0,6,"Statement Period: 29/08/2026 To 24/11/2026",new_x="LMARGIN",new_y="NEXT")
cols=[22,78,28,28,30]
for h,wd in zip(["Date","Narration","Withdrawal","Deposit","Balance"],cols): pdf.cell(wd,5,h,border=1)
pdf.ln()
for r in rows:
    vals=fmt(r)
    for v,wd in zip(vals,cols): pdf.cell(wd,5,v[:48],border=1)
    pdf.ln()
pdf.output(str(OUT / "fake_statement.pdf"))
