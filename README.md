# Supply Chain Demand Forecasting & Inventory Optimization

Forecast daily product demand, turn the forecast into a complete inventory plan, and
decide how to allocate limited stock across locations. An end-to-end, reproducible
demand-planning project in Python, built around the standard supply chain framework
(Chopra & Meindl).

---

## Overview

Demand planners answer three questions every cycle, and this project answers all three:

1. **How much will we sell?** A machine learning model forecasts daily demand per store
   and SKU, accounting for seasonality, promotions and price.
2. **How much to order, and how much buffer?** The forecast feeds a fixed-order-quantity
   policy: EOQ, reorder point, safety stock and total inventory cost.
3. **Given limited supply, who gets what?** A fair-share rule allocates inbound stock
   across locations, and a risk-pooling analysis tests whether to centralize or distribute.

The dataset is generated synthetically (realistic trend, weekly and annual seasonality,
promotions, holidays and noise) so the whole project runs anywhere with no downloads.

---

## Results

**Forecast accuracy** on a 56-day out-of-sample hold-out (time-based split, no leakage):

| Model | WMAPE | MAE | RMSE | Bias % |
|---|---|---|---|---|
| Seasonal-naive (last week) | 0.269 | 12.77 | 18.26 | -3.57 |
| Moving average (28d) | 0.208 | 9.86 | 14.72 | -6.13 |
| **Gradient Boosting** | **0.167** | **7.94** | **11.62** | **-0.13** |

Gradient Boosting cuts WMAPE about **38% versus seasonal-naive** and **20% versus a
28-day moving average**, with near-zero bias.



**Risk pooling.** Consolidating three stores into one central warehouse cuts total
safety stock by about **42%**, the classic aggregation result from the course.


---

## How it works

1. **Data** (`src/generate_data.py`) builds a realistic CPG sales dataset:
   `date, store, sku, units, price, promo`.
2. **Features** (`src/features.py`) adds calendar fields, lagged demand (1/7/14/28 days)
   and rolling mean/std. All lags are shifted so no future information leaks into a row.
3. **Forecasting** (`src/forecast.py`) compares two naive baselines against gradient
   boosting, scored with WMAPE, MAE, RMSE and bias (`src/evaluate.py`).
4. **Inventory economics** (`src/allocation.py`) applies the Chopra & Meindl framework
   at a 95% service level and a 7-day lead time:
   - Safety stock = z * sigma_DLT, where sigma_DLT = sqrt(LT * sigma_d^2 + avg^2 * sL^2)
     covers both demand and lead-time variability
   - Reorder point, EOQ = sqrt(2 D S / H), order-up-to level
   - Total annual inventory cost = (D / Q) * S + (Q / 2 + SS) * H
5. **Risk pooling** quantifies the safety-stock savings from centralizing inventory.
6. **Allocation** splits a constrained inbound supply across stores in proportion to
   forecasted demand (fair-share).

---

## Repository structure

```
supply-chain-demand-forecasting/
|- main.py                 # runs the whole pipeline end-to-end
|- 01_analysis.ipynb       # narrated notebook (open and Run All)
|- requirements.txt
|- src/
|  |- generate_data.py     # synthetic CPG demand (swap in M5 / Favorita for real data)
|  |- features.py          # calendar, lags, rolling stats
|  |- forecast.py          # baselines + HistGradientBoosting
|  |- evaluate.py          # WMAPE, MAE, RMSE, bias
|  |- allocation.py        # EOQ, safety stock, risk pooling, allocation
|- reports/                # metrics, plans, figures
```

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Or open `01_analysis.ipynb` in Jupyter and run all cells. Everything is reproducible
from a fixed random seed. To use real data, replace `src/generate_data.py` with a loader
that emits the same columns.

---

## Tech stack

Python, pandas, NumPy, scikit-learn (HistGradientBoostingRegressor), Matplotlib.

---

## What this project demonstrates

- Time-series feature engineering and leakage-safe backtesting
- Beating naive baselines with gradient boosting, measured honestly
- A full, textbook-correct inventory policy (EOQ, reorder point, safety stock, cost)
- Risk-pooling / centralization analysis quantified from the data
- Constrained allocation logic a planning team can act on
- Clean, modular, reproducible code

---

