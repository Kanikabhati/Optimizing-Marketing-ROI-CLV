#segment summary
import pandas as pd

# Load segmented customers
segments = pd.read_csv('Data/processed/customer_segments.csv')

# Summary statistics by segment
summary = segments.groupby('Segment').agg({
    'LTV': ['count', 'mean', 'sum'],
    'Recency': 'mean'
})
print("--- Segment Summary Statistics ---")
print(summary)

#visualize segments
import matplotlib.pyplot as plt

# Segment sizes (customer count per segment)
size = segments['Segment'].value_counts().sort_index()
plt.figure(figsize=(6,4))
size.plot(kind='bar')
plt.title('Number of Customers per Segment')
plt.xlabel('Segment')
plt.ylabel('Customer Count')
plt.tight_layout()
plt.savefig('Data/processed/segment_size_bar.png')
plt.close()

# Total LTV per segment
revenue = segments.groupby('Segment')['LTV'].sum()
plt.figure(figsize=(6,4))
revenue.plot(kind='bar')
plt.title('Total LTV by Segment')
plt.xlabel('Segment')
plt.ylabel('Total Lifetime Value')
plt.tight_layout()
plt.savefig('Data/processed/segment_ltv_bar.png')
plt.close()
print("Visualization saved as segment_size_bar.png and segment_ltv_bar.png")