import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

# Load features
features = pd.read_csv('Data/processed/customer_features.csv')

# Define predictors and target
X = features[['Recency', 'Frequency', 'Monetary',
              'AverageOrderValue', 'ProductVariety',
              'CustomerLifespanYears']]
y = features['LTV']

# Train-test split (for final hold‑out evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define stronger, regularized XGBoost model
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# Proper cross‑validation on unfitted model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print("CV scores:", scores)
print("Mean R2:", scores.mean())
print("Std R2:", scores.std())

# Fit on train split for final test evaluation
model.fit(X_train, y_train)

# Evaluate on hold‑out test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost Model R^2 Score (test): {r2:.2f}")

# Segment customers with K‑Means using predicted LTV & Recency
segmentation_data = features[['LTV', 'Recency']].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(segmentation_data)
features['Segment'] = kmeans.labels_

# Save segmented data
features.to_csv('Data/processed/customer_segments.csv', index=False)
print("Customer segmentation complete. Segments saved as customer_segments.csv.")
