import pandas as pd

# Load the Online Retail dataset (make sure filename matches)
df = pd.read_excel('Data/raw/Online Retail.xlsx')

# Explore dataset
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

# Check missing Customer IDs
missing_customer_ids = df['CustomerID'].isna().sum()
print("Missing CustomerIDs:", missing_customer_ids)

# Count returns (negative Quantity)
negative_qty = (df['Quantity'] < 0).sum()
print("Negative Quantity entries (returns):", negative_qty)

# Check duplicate rows
duplicate_rows = df.duplicated().sum()
print("Duplicate rows:", duplicate_rows)

# Data cleaning
df_clean = df.dropna(subset=['CustomerID'])
df_clean = df_clean.drop_duplicates()
df_clean['IsReturn'] = df_clean['Quantity'] < 0
df_clean = df_clean[df_clean['Country'] == 'United Kingdom']  # Focus on UK data

# Save cleaned data to processed folder
df_clean.to_csv('Data/processed/online_retail_cleaned.csv', index=False)
