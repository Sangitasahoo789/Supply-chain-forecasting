"""Inventory economics + constrained allocation (course Session 5: Chopra & Meindl).

Fixed-order-quantity framework:
  - EOQ = sqrt(2 D S / H)
  - Safety stock with demand AND lead-time variability:
        sigma_DLT = sqrt(LT * sigma_d^2 + avg_d^2 * sL^2)
        SS = z * sigma_DLT
  - Reorder point  = avg_daily * LT + SS
  - Order-up-to    = avg_daily * (LT + RP) + SS         (periodic review)
  - Total annual inventory cost = (D/Q) * S + (Q/2 + SS) * H
Plus a risk-pooling (centralization) analysis and a constrained fair-share
allocation when inbound supply is short.
"""
import numpy as np
import pandas as pd

Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}

# Parameters grounded in the course example (Chopra & Meindl, Session 5)
SERVICE = 0.95          # z = 1.645 at 95% service level
LEAD_TIME = 7           # days  (deck example LT = 7)
LEAD_TIME_STD = 2.0     # days  (supply / lead-time variability, sL)
REVIEW = 7              # days  (periodic review period, RP)
ORDER_COST = 50.0       # S: ordering cost per order (transport + admin), $
HOLDING_RATE = 0.20     # h: annual holding cost as a fraction of unit value
SUPPLY_RATIO = 0.8      # inbound supply vs total need (forces allocation)


def sigma_dlt(sigma_d, avg_d, lead_time=LEAD_TIME, lead_std=LEAD_TIME_STD):
    """Std dev of demand over the lead time (demand + supply variability)."""
    return np.sqrt(lead_time * sigma_d ** 2 + avg_d ** 2 * lead_std ** 2)


def inventory_policy(feat):
    z = Z[SERVICE]
    s = (feat.groupby(["store", "sku"])
         .agg(avg_daily=("units", "mean"), sigma_d=("units", "std"),
              price=("price", "mean")).reset_index())
    s["sigma_DLT"] = sigma_dlt(s["sigma_d"], s["avg_daily"]).round(1)
    s["safety_stock"] = (z * s["sigma_DLT"]).round(1)
    s["reorder_point"] = (s["avg_daily"] * LEAD_TIME + s["safety_stock"]).round(1)
    D = s["avg_daily"] * 365.0
    H = HOLDING_RATE * s["price"]
    s["EOQ"] = np.sqrt(2 * D * ORDER_COST / H).round(0)
    s["order_up_to"] = (s["avg_daily"] * (LEAD_TIME + REVIEW) + s["safety_stock"]).round(1)
    s["annual_inv_cost"] = ((D / s["EOQ"]) * ORDER_COST
                            + (s["EOQ"] / 2 + s["safety_stock"]) * H).round(0)
    return s


def risk_pooling(feat):
    """Decentralized (per-store) vs centralized (pooled) safety stock per SKU.

    Isolates demand variability (SS = z * sigma_d * sqrt(LT)) to show the pure
    aggregation benefit, as in the McCain / TOOPS DC-consolidation case.
    """
    z = Z[SERVICE]
    base = (feat.groupby(["store", "sku"])
            .agg(avg_daily=("units", "mean"), sigma_d=("units", "std")).reset_index())
    base["ss_store"] = z * base["sigma_d"] * np.sqrt(LEAD_TIME)
    rows = []
    for sku, g in base.groupby("sku"):
        ss_dec = g["ss_store"].sum()
        sigma_pool = np.sqrt((g["sigma_d"] ** 2).sum())          # independent demand
        ss_cen = z * sigma_pool * np.sqrt(LEAD_TIME)
        rows.append((sku, round(ss_dec, 1), round(ss_cen, 1),
                     round((1 - ss_cen / ss_dec) * 100, 1)))
    return pd.DataFrame(rows, columns=["sku", "ss_decentralized", "ss_centralized", "reduction_%"])


def build_allocation(feat, test):
    policy = inventory_policy(feat)
    fc = (test.groupby(["store", "sku"])["pred_gbm"].mean().reset_index()
          .rename(columns={"pred_gbm": "fcst_daily"}))
    fc["need_LT"] = (fc["fcst_daily"] * LEAD_TIME).round(1)
    plan = policy.merge(fc, on=["store", "sku"])
    plan["allocated"] = 0.0
    for _, grp in plan.groupby("sku"):
        need = grp["need_LT"].sum()
        supply = need * SUPPLY_RATIO
        plan.loc[grp.index, "allocated"] = (grp["need_LT"] / need * supply).round(1)
    plan["fill_rate_%"] = (plan["allocated"] / plan["need_LT"] * 100).round(1)
    cols = ["store", "sku", "avg_daily", "sigma_DLT", "safety_stock", "reorder_point",
            "EOQ", "order_up_to", "annual_inv_cost", "need_LT", "allocated", "fill_rate_%"]
    return plan[cols].sort_values(["sku", "store"]).reset_index(drop=True)
