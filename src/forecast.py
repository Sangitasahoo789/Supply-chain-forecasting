"""Forecasting models: baselines + gradient boosting."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from features import FEATURES


def train_test_split_time(df, test_days=56):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def baseline_forecasts(test):
    """Seasonal-naive (last week) and 28-day moving average, from existing features."""
    out = test.copy()
    out["pred_snaive"] = out["lag_7"]
    out["pred_ma28"] = out["roll_mean_28"]
    return out


def fit_gbm(train):
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=42)
    model.fit(train[FEATURES], train["units"])
    return model
