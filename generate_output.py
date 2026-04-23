"""Generate classified output from sample_transactions.csv."""

import pandas as pd
from src.classifier import TransactionClassifier

df = pd.read_csv("sample_transactions.csv")

# classify_dataframe() checks for 'transaction_type' column for credit override
df = df.rename(columns={"type": "transaction_type"})

classifier = TransactionClassifier(use_zero_shot=True)
result = classifier.classify_dataframe(df)

result.to_csv("sample_output.csv", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
print(result)
print(f"\nSaved to sample_output.csv ({len(result)} rows)")
