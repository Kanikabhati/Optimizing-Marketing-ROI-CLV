import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def load_clv_time_series(path="Data/processed/customer_clv_bgnbd.csv"):
    """
    Build a monthly total CLV time series from per-customer CLV.
    Assumes CLV horizon ~ 6 months and spreads it evenly.
    """
    clv_df = pd.read_csv(path)
    clv_df = clv_df.dropna(subset=["pred_clv"])

    horizon_months = 6
    start_month = pd.to_datetime("2011-01-01")

    records = []
    for _, row in clv_df.iterrows():
        monthly_clv = row["pred_clv"] / horizon_months
        for m in range(horizon_months):
            month = start_month + pd.DateOffset(months=m)
            records.append({"month": month, "clv": monthly_clv})

    ts_df = pd.DataFrame(records)
    monthly_series = (
        ts_df.groupby("month")["clv"]
        .sum()
        .sort_index()
    )

    return monthly_series

def fit_arima_and_forecast(series, forecast_steps=6):
    """
    Fit ARIMA on full series (no explicit test split, since we only
    have a few points) and forecast forward.
    """
    series = series.asfreq("MS").sort_index()

    # Simple ARIMA(1,1,0) to reduce parameters for tiny sample
    model = ARIMA(series, order=(1, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_steps)
    return model_fit, forecast

if __name__ == "__main__":
    clv_series = load_clv_time_series()
    print("Monthly CLV time series:")
    print(clv_series)

    model_fit, forecast = fit_arima_and_forecast(clv_series, forecast_steps=6)

    print("\nForecasted total CLV for next months:")
    print(forecast)

    # Save forecast
    future_df = pd.DataFrame({
        "month": forecast.index,
        "forecast_clv": forecast.values
    })
    future_df.to_csv("Data/processed/monthly_clv_forecast.csv", index=False)
    print("\nSaved monthly CLV forecast to Data/processed/monthly_clv_forecast.csv")
