"""Feature engineering for demand forecasting."""
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLS = [7, 28]


def add_features(df):
    df = df.sort_values(["store", "sku", "date"]).copy()
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.0)

    g = df.groupby(["store", "sku"])["units"]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    for w in ROLLS:
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).std())

    df["store_id"] = df["store"].astype("category").cat.codes
    df["sku_id"] = df["sku"].astype("category").cat.codes
    return df


FEATURES = (["price", "promo", "dow", "month", "weekofyear", "is_weekend",
             "doy_sin", "doy_cos", "store_id", "sku_id"]
            + [f"lag_{l}" for l in LAGS]
            + [f"roll_mean_{w}" for w in ROLLS]
            + [f"roll_std_{w}" for w in ROLLS])
