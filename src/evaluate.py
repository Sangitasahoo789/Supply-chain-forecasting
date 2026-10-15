"""Forecast accuracy metrics."""
import numpy as np


def wmape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    denom = np.abs(a).sum()
    return np.abs(a - f).sum() / denom if denom else np.nan


def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(f, float))))


def rmse(a, f):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(f, float)) ** 2)))


def bias_pct(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = a.sum()
    return (f.sum() - s) / s * 100 if s else np.nan


def scorecard(a, f):
    return {"WMAPE": wmape(a, f), "MAE": mae(a, f), "RMSE": rmse(a, f), "Bias%": bias_pct(a, f)}
