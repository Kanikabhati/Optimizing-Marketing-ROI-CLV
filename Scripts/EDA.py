import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature-engineered data
features = pd.read_csv('Data/processed/customer_features.csv')

print("\n--- COLUMN NAMES AND FIRST FEW ROWS ---")
for i, col in enumerate(features.columns):
    print(f"Column {i}: '{col}'")
print(features.head())

# Manually set numeric columns based on actual output above!
# UPDATE this list to match exactly, including spaces/capitalization
numeric_cols = [
    'Recency',
    'Frequency',
    'Monetary',
    'AverageOrderValue',
    'ProductVariety',
    'CustomerLifespanYears',
    'LTV'
]

print("\n--- USING THESE NUMERIC COLUMNS FOR ANALYSIS ---")
print(numeric_cols)

# Print statistical summary
print("\n--- STATISTICAL SUMMARY ---")
summary_stats = features[numeric_cols].describe(percentiles=[.25, .5, .75])
print(summary_stats)

# Plot and save distributions for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(features[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'Data/processed/{col}_distribution.png')
    plt.close()

# Correlation matrix heatmap
print("\n--- CORRELATION MATRIX ---")
corr_matrix = features[numeric_cols].corr()
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('Data/processed/correlation_matrix.png')
plt.close()

print("\nEDA complete! Summary stats printed, distribution plots and correlation matrix saved as PNGs in Data/processed.")