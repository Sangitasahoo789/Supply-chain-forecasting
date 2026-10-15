"""Generate a realistic synthetic CPG demand dataset.

Simulates daily unit sales across stores (or wholesale partners) and SKUs with
trend, weekly + annual seasonality, promotions, price, holidays and Poisson noise.
Swap this for a real dataset (e.g. Walmart M5, Corporacion Favorita) by emitting
the same long schema: [date, store, sku, units, price, promo].
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
STORES = ["North", "South", "Central"]
SKUS = ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"]
START, DAYS = "2023-01-01", 730
HOLIDAYS = {(12, 24): 1.8, (12, 23): 1.6, (11, 24): 1.7, (7, 4): 1.4, (2, 14): 1.3}


def _base_level(store, sku):
    return 20 + 8 * STORES.index(store) + 6 * SKUS.index(sku)


def generate():
    dates = pd.date_range(START, periods=DAYS, freq="D")
    rows = []
    for store in STORES:
        for sku in SKUS:
            base = _base_level(store, sku)
            trend = np.linspace(0, RNG.uniform(2, 8), DAYS)
            price0 = RNG.uniform(3, 12)
            for i, d in enumerate(dates):
                weekend = 1.25 if d.dayofweek >= 5 else 1.0
                annual = 1 + 0.20 * np.sin(2 * np.pi * (d.dayofyear / 365.0))
                promo = 1 if RNG.random() < 0.08 else 0
                price = round(price0 * (0.8 if promo else 1.0), 2)
                hol = HOLIDAYS.get((d.month, d.day), 1.0)
                mean = (base + trend[i]) * weekend * annual * (1.6 if promo else 1.0) * hol
                rows.append((d, store, sku, RNG.poisson(max(mean, 0.1)), price, promo))
    return pd.DataFrame(rows, columns=["date", "store", "sku", "units", "price", "promo"])


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate()
    df.to_csv("data/raw/sales.csv", index=False)
    print(f"Wrote data/raw/sales.csv  rows={len(df):,}  stores={df.store.nunique()}  "
          f"skus={df.sku.nunique()}  dates={df.date.nunique()}")
