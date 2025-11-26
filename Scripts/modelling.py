import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load features
features = pd.read_csv('Data/processed/customer_features.csv')

# Define predictors and target
X = features[['Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'ProductVariety', 'CustomerLifespanYears']]
y = features['LTV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost Model R^2 Score: {r2:.2f}")

#Segment Customers with K-Means
from sklearn.cluster import KMeans
# Segment on predicted LTV and Recency
segmentation_data = features[['LTV', 'Recency']].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42).fit(segmentation_data)
features['Segment'] = kmeans.labels_

# Save segmented data
features.to_csv('Data/processed/customer_segments.csv', index=False)
print("Customer segmentation complete. Segments saved as customer_segments.csv.")