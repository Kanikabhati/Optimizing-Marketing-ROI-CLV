import pandas as pd
# Load the Online Retail dataset (update filename if needed)
df = pd.read_excel('Data/raw/Online Retail.xlsx')
# Basic exploration
print("Shape:", df.shape)
print("Columns:", df.columns)
print("First 5 rows:\n", df.head())
# Count missing CustomerID
missing_customer_ids = df['CustomerID'].isna().sum()
print("Missing CustomerIDs:", missing_customer_ids)
# Count transactions with negative Quantity (returns)
negative_qty = (df['Quantity'] < 0).sum()
print("Negative Quantity (returns):", negative_qty)
# Count duplicate rows
duplicate_rows = df.duplicated().sum()
print("Duplicate rows:", duplicate_rows)
