import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

# ============================
# Probabilistic CLV: Data Loader (BG/NBD)
# ============================

def load_transactions(path="Data/processed/online_retail_cleaned.csv"):
    """
    Load transactional data for probabilistic CLV (BG/NBD).
    """
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])

    # Remove returns if IsReturn exists (0 = normal sale, 1 = return)
    if "IsReturn" in df.columns:
        df = df[df["IsReturn"] == 0]

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Compute transaction revenue = Quantity * UnitPrice
    df["revenue"] = df["Quantity"] * df["UnitPrice"]

    # Keep only positive revenue
    df = df[df["revenue"] > 0]

    # Rename to standard names used later in the CLV pipeline
    df = df.rename(columns={
        "CustomerID": "customer_id",
        "InvoiceDate": "date"
    })

    # Ensure consistent types
    df["customer_id"] = df["customer_id"].astype(str)

    # Keep only required columns
    df = df[["customer_id", "date", "revenue"]]

    return df

def make_rfm_summary(transactions: pd.DataFrame):
    """
    Aggregate transaction-level data to customer-level summary
    for BG/NBD and Gamma-Gamma models.
    """
    max_date = transactions["date"].max()

    summary = summary_data_from_transaction_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=max_date
    )
    return summary

def make_calibration_holdout(transactions: pd.DataFrame, calibration_end: str):
    """
    Create calibration (train) and holdout (test) sets for BG/NBD + Gamma-Gamma.
    """
    data_cal_holdout = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="date",
        calibration_period_end=calibration_end,
        observation_period_end=transactions["date"].max(),
        monetary_value_col="revenue"
    )
    return data_cal_holdout

def fit_bgnbd_ggf(cal_holdout: pd.DataFrame, penalizer: float = 1.0):
    """
    Fit BG/NBD (repeat transactions) and Gamma-Gamma (monetary value)
    on the calibration period.
    """
    # Filter to customers with positive monetary_value_cal for Gamma-Gamma
    cal_pos = cal_holdout[cal_holdout["monetary_value_cal"] > 0].copy()

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        frequency=cal_holdout["frequency_cal"],
        recency=cal_holdout["recency_cal"],
        T=cal_holdout["T_cal"]
    )

    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(
        frequency=cal_pos["frequency_cal"],
        monetary_value=cal_pos["monetary_value_cal"]
    )

    return bgf, ggf

def predict_clv(cal_holdout: pd.DataFrame,
                bgf: BetaGeoFitter,
                ggf: GammaGammaFitter,
                horizon_days: int = 180,
                discount_rate_monthly: float = 0.01):
    """
    Predict expected CLV over a future horizon (in days)
    using BG/NBD (for transaction counts) and Gamma-Gamma (for spend).
    """
    cal_holdout = cal_holdout.copy()

    # Expected number of purchases in next horizon_days
    cal_holdout["pred_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days,
        cal_holdout["frequency_cal"],
        cal_holdout["recency_cal"],
        cal_holdout["T_cal"]
    )

    # Expected average order value; set NaN for customers with non-positive monetary_value_cal
    mask_pos = cal_holdout["monetary_value_cal"] > 0
    cal_holdout["pred_avg_value"] = None
    cal_holdout.loc[mask_pos, "pred_avg_value"] = ggf.conditional_expected_average_profit(
        frequency=cal_holdout.loc[mask_pos, "frequency_cal"],
        monetary_value=cal_holdout.loc[mask_pos, "monetary_value_cal"]
    )

    # Convert horizon to months for the CLV helper
    horizon_months = horizon_days / 30.0

    cal_holdout["pred_clv"] = None
    cal_holdout.loc[mask_pos, "pred_clv"] = ggf.customer_lifetime_value(
        bgf,
        frequency=cal_holdout.loc[mask_pos, "frequency_cal"],
        recency=cal_holdout.loc[mask_pos, "recency_cal"],
        T=cal_holdout.loc[mask_pos, "T_cal"],
        monetary_value=cal_holdout.loc[mask_pos, "monetary_value_cal"],
        time=horizon_months,
        discount_rate=discount_rate_monthly
    )

    return cal_holdout

def evaluate_bgnbd(cal_holdout: pd.DataFrame, bgf: BetaGeoFitter):
    """
    Evaluate BG/NBD on the holdout by predicting the number of
    purchases in the holdout window and comparing with actuals.
    """
    # Use only customers who actually have a non-zero holdout duration
    eval_data = cal_holdout[cal_holdout["duration_holdout"] > 0].copy()

    # Predict expected purchases during holdout duration for each customer
    pred_holdout = bgf.predict(
        t=eval_data["duration_holdout"],
        frequency=eval_data["frequency_cal"],
        recency=eval_data["recency_cal"],
        T=eval_data["T_cal"]
    )

    # Drop any NaNs in predictions just in case
    mask = ~pd.isna(pred_holdout)
    y_true = eval_data.loc[mask, "frequency_holdout"]
    y_pred = pred_holdout[mask]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    print(f"BG/NBD holdout MAE (transactions): {mae:.3f}")
    print(f"BG/NBD holdout RMSE (transactions): {rmse:.3f}")

    return mae, rmse

# ============================
# Existing Machine Learning CLV Model (XGBoost) + Segmentation
# ============================

features = pd.read_csv("Data/processed/customer_features.csv")

X = features[[
    "Recency",
    "Frequency",
    "Monetary",
    "AverageOrderValue",
    "ProductVariety",
    "CustomerLifespanYears"
]]
y = features["LTV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print("CV scores:", scores)
print("Mean R2:", scores.mean())
print("Std R2:", scores.std())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost Model R^2 Score (test): {r2:.2f}")

segmentation_data = features[["LTV", "Recency"]].fillna(0)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(segmentation_data)
features["Segment"] = kmeans.labels_

features.to_csv("Data/processed/customer_segments.csv", index=False)
print("Customer segmentation complete. Segments saved as customer_segments.csv.")

# ============================
# Temporary test block for BG/NBD pipeline
# ============================

if __name__ == "__main__":
    # Load transactions
    tx = load_transactions()
    print("Sample transactions for BG/NBD CLV:")
    print(tx.head())
    print(tx.dtypes)

    # Calibration/holdout split
    calibration_end = "2011-06-30"  # adjust if needed
    cal_holdout = make_calibration_holdout(tx, calibration_end=calibration_end)
    print("\nCalibration/holdout data (columns):")
    print(cal_holdout.columns)

    # Fit models
    bgf, ggf = fit_bgnbd_ggf(cal_holdout, penalizer=1.0)

    # Evaluate BG/NBD on holdout
    mae_tx, rmse_tx = evaluate_bgnbd(cal_holdout, bgf)

    # Predict CLV
    clv_results = predict_clv(cal_holdout, bgf, ggf, horizon_days=180)
    print("\nCLV prediction sample:")
    print(clv_results[["frequency_cal", "recency_cal", "T_cal",
                       "monetary_value_cal", "pred_purchases",
                       "pred_avg_value", "pred_clv"]].head())

    # Save CLV for downstream optimization
    clv_out = clv_results.reset_index().rename(columns={"index": "customer_id"})
    clv_out = clv_out[["customer_id", "pred_clv", "pred_purchases", "pred_avg_value"]]
    clv_out.to_csv("Data/processed/customer_clv_bgnbd.csv", index=False)
    print("\nSaved BG/NBD CLV to Data/processed/customer_clv_bgnbd.csv")
