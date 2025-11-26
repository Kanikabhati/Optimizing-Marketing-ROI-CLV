import pandas as pd
import numpy as np

# Load cleaned transaction data
df_clean = pd.read_csv('Data/processed/online_retail_cleaned.csv')

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Define snapshot date for recency calculations
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate RFM (Recency, Frequency, Monetary) per customer
rfm = df_clean.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Quantity', lambda x: np.sum(x * df_clean.loc[x.index, 'UnitPrice']))
)

# Advanced features
basket_stats = df_clean.groupby('CustomerID').agg(
    AverageOrderValue=('Quantity', lambda x: np.sum(x * df_clean.loc[x.index, 'UnitPrice']) / x.count()),
    ProductVariety=('StockCode', pd.Series.nunique),
    Country=('Country', 'first')
)

# Customer lifespan
lifespan = df_clean.groupby('CustomerID').agg(
    FirstPurchase=('InvoiceDate', 'min'),
    LastPurchase=('InvoiceDate', 'max')
)
lifespan['CustomerLifespanYears'] = (lifespan['LastPurchase'] - lifespan['FirstPurchase']).dt.days / 365.0

# Assemble all features
features = pd.concat([rfm, basket_stats, lifespan['CustomerLifespanYears']], axis=1)

# Calculate Customer Lifetime Value (LTV)
features['LTV'] = features['AverageOrderValue'] * features['Frequency'] * features['CustomerLifespanYears']
features = features.fillna(0)

# Save features to processed folder
features.to_csv('Data/processed/customer_features.csv')
print("Feature engineering complete. Features saved as customer_features.csv")
print("Shape:", features.shape)
print("Columns:", features.columns.tolist())