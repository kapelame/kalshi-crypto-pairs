#!/usr/bin/env python3
"""
Kalshi Crypto 15-Min ML v3
===========================
基于 collector v3 数据训练预测模型

策略核心:
  对每个15-min合约预测 P(YES) = P(价格会涨)
  如果 model_P(YES) > market_price → 买 YES (市场低估)
  如果 model_P(YES) < market_price → 买 NO  (市场高估)
  edge = |model_P - market_P|

特征:
  1. time_to_expiry (到期剩余, 最关键变量)
  2. price_vs_strike_pct (真实价格vs参考价偏离%)
  3. ob_imbalance (orderbook不平衡, 方向信号)
  4. mid momentum (概率变化速度)
  5. spread (共识度)
  6. cross-asset mids (BTC领先信号)

ML target:
  settlement outcome (YES=1, NO=0)
  从 Kalshi API 查询每个 ticker 的结算结果

评估:
  Brier score, calibration, accuracy, edge分析

用法: python3 kalshi_ml_v3.py [csv_path]
"""

import pandas as pd
import numpy as np
import requests
import sys
import os
import json
import warnings
import sqlite3
import csv as csvmod
from datetime import datetime

warnings.filterwarnings("ignore")

CSV = "kalshi_v3_features.csv"
DB = "kalshi_v3.db"
ASSETS = ["btc", "eth", "sol", "xrp"]
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

if len(sys.argv) > 1:
    CSV = sys.argv[1]

if not os.path.exists(CSV) and os.path.exists(f"/content/{CSV}"):
    CSV = f"/content/{CSV}"

# Export from DB if needed
if not os.path.exists(CSV) and os.path.exists(DB):
    print(f"  Exporting from {DB}...")
    conn = sqlite3.connect(DB)
    cursor = conn.execute("SELECT * FROM features ORDER BY ts_unix")
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    conn.close()

W = 80


# ==============================================================================
# Step 1: Load data + get settlement outcomes
# ==============================================================================

def load_data():
    print(f"\n{'=' * W}")
    print(f"  STEP 1: Load Data & Settlement Outcomes")
    print(f"{'=' * W}")

    df = pd.read_csv(CSV)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {CSV}")

    # Get all unique tickers
    all_tickers = {}
    for a in ASSETS:
        col = f"{a}_ticker"
        if col in df.columns:
            for t in df[col].dropna().unique():
                all_tickers[t] = a

    print(f"  Found {len(all_tickers)} unique tickers across {len(ASSETS)} assets")

    # Look up settlements
    settlements = {}
    settled = 0
    for ticker, asset in all_tickers.items():
        try:
            r = requests.get(f"{KALSHI_API}/markets/{ticker}", timeout=5)
            if r.status_code == 200:
                m = r.json().get("market", {})
                result = m.get("result", "")
                if result in ("yes", "no"):
                    settlements[ticker] = 1 if result == "yes" else 0
                    settled += 1
        except:
            pass

    yes_n = sum(1 for v in settlements.values() if v == 1)
    no_n = sum(1 for v in settlements.values() if v == 0)
    print(f"  Settlements: {settled}/{len(all_tickers)} "
          f"(YES={yes_n}, NO={no_n})")

    if settled == 0:
        print(f"\n  !! No settled markets found.")
        print(f"  !! Markets may still be active or pending settlement.")
        print(f"  !! Run this script again after markets have settled (5min after close).")
        return None, None

    # Add target column for each asset
    for a in ASSETS:
        col = f"{a}_ticker"
        target_col = f"{a}_target"
        df[target_col] = df[col].map(settlements)

    return df, settlements


# ==============================================================================
# Step 2: Feature engineering
# ==============================================================================

def engineer_features(df):
    print(f"\n{'=' * W}")
    print(f"  STEP 2: Feature Engineering")
    print(f"{'=' * W}")

    # We train a per-asset model: each asset's data is a training sample
    # Features are relative to that asset + cross-asset context
    rows = []

    for a in ASSETS:
        p = a
        target_col = f"{a}_target"
        ticker_col = f"{a}_ticker"

        asset_df = df[df[target_col].notna()].copy()
        if len(asset_df) == 0:
            continue

        for _, row in asset_df.iterrows():
            feat = {
                "asset": a,
                "ticker": row.get(ticker_col, ""),
                "ts_unix": row["ts_unix"],
                "target": int(row[target_col]),

                # Core features
                "mid": row.get(f"{p}_mid"),
                "spread": row.get(f"{p}_spread"),
                "time_to_expiry": row.get(f"{p}_time_to_expiry"),
                "time_to_expiry_pct": (row.get(f"{p}_time_to_expiry") or 0) / 900.0,
                "ob_imbalance": row.get(f"{p}_ob_imbalance"),
                "price_vs_strike_pct": row.get(f"{p}_price_vs_strike_pct"),
                "volume": row.get(f"{p}_volume", 0),
                "oi": row.get(f"{p}_oi", 0),

                # Momentum
                "mom_5s": row.get(f"{p}_mom_5s"),
                "mom_15s": row.get(f"{p}_mom_15s"),
                "mom_30s": row.get(f"{p}_mom_30s"),

                # Cross-asset context (other assets' mids)
                "btc_mid_ctx": row.get("btc_mid") if a != "btc" else None,
            }

            # BTC lead signal: if BTC is moving, others may follow
            if a != "btc":
                btc_mom = row.get("btc_mom_5s")
                own_mom = row.get(f"{p}_mom_5s")
                if btc_mom is not None and own_mom is not None:
                    feat["btc_lead"] = btc_mom - own_mom
                else:
                    feat["btc_lead"] = None
            else:
                feat["btc_lead"] = None

            # Mid distance from 0.5 (how decisive is the market)
            mid = row.get(f"{p}_mid")
            if mid is not None:
                feat["mid_decisiveness"] = abs(mid - 0.5)
            else:
                feat["mid_decisiveness"] = None

            rows.append(feat)

    feat_df = pd.DataFrame(rows)
    print(f"  Generated {len(feat_df)} training samples from {len(ASSETS)} assets")
    print(f"  Target distribution: YES={feat_df['target'].sum()} "
          f"NO={(feat_df['target'] == 0).sum()}")

    return feat_df


# ==============================================================================
# Step 3: Train model
# ==============================================================================

def train_model(feat_df):
    print(f"\n{'=' * W}")
    print(f"  STEP 3: Train Model")
    print(f"{'=' * W}")

    # Feature columns
    feature_cols = [
        "mid", "spread", "time_to_expiry", "time_to_expiry_pct",
        "ob_imbalance", "price_vs_strike_pct", "volume", "oi",
        "mom_5s", "mom_15s", "mom_30s",
        "btc_mid_ctx", "btc_lead", "mid_decisiveness",
    ]

    # Drop rows with all NaN features
    feat_df = feat_df.dropna(subset=["mid", "time_to_expiry"])
    if len(feat_df) < 50:
        print(f"  !! Only {len(feat_df)} valid samples, need at least 50")
        return None, None, None

    X = feat_df[feature_cols].copy()
    y = feat_df["target"].values

    # Fill NaN with 0 for optional features
    X = X.fillna(0)

    # Time-based train/test split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Features: {feature_cols}")

    # Try XGBoost first, fall back to sklearn
    model = None
    model_name = ""

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        model_name = "XGBoost"
    except Exception:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
            )
            model.fit(X_train, y_train)
            model_name = "GradientBoosting (sklearn)"
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns,
                index=X_test.index)
            X_train = pd.DataFrame(
                X_train_s, columns=X_train.columns, index=X_train.index)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            model_name = "LogisticRegression"

    print(f"  Model: {model_name}")

    return model, (X_train, X_test, y_train, y_test), feature_cols


# ==============================================================================
# Step 4: Evaluate
# ==============================================================================

def evaluate(model, data_split, feature_cols, feat_df):
    print(f"\n{'=' * W}")
    print(f"  STEP 4: Evaluation")
    print(f"{'=' * W}")

    X_train, X_test, y_train, y_test = data_split

    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Accuracy
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    print(f"  Accuracy:  train={train_acc:.3f}  test={test_acc:.3f}")

    # Brier score (lower = better, 0.25 = random)
    train_brier = np.mean((train_proba - y_train) ** 2)
    test_brier = np.mean((test_proba - y_test) ** 2)
    print(f"  Brier:     train={train_brier:.4f}  test={test_brier:.4f}  "
          f"(random=0.2500)")

    # Calibration (bin predictions and compare to actual rates)
    print(f"\n  CALIBRATION (test set):")
    print(f"  {'Bin':>12} {'Count':>6} {'Pred P':>7} {'Actual':>7} {'Gap':>7}")
    print(f"  {'-' * 45}")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        mask = (test_proba >= lo) & (test_proba < hi)
        n = mask.sum()
        if n > 0:
            pred_mean = test_proba[mask].mean()
            actual_mean = y_test[mask].mean()
            gap = pred_mean - actual_mean
            print(f"  {f'[{lo:.1f},{hi:.1f})':>12} {n:>6} {pred_mean:>7.3f} "
                  f"{actual_mean:>7.3f} {gap:>+7.3f}")

    # Feature importance
    print(f"\n  FEATURE IMPORTANCE:")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_cols))

    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        bar = "#" * int(importances[i] / max(importances) * 30)
        print(f"    {feature_cols[i]:<25} {importances[i]:.4f} {bar}")

    # Edge analysis: compare model P(YES) to market mid
    print(f"\n  EDGE ANALYSIS (test set):")
    split_idx = int(len(feat_df.dropna(subset=["mid", "time_to_expiry"])) * 0.8)
    test_feat = feat_df.dropna(subset=["mid", "time_to_expiry"]).iloc[split_idx:]

    if len(test_feat) == len(test_proba):
        market_price = test_feat["mid"].values
        edge = test_proba - market_price

        # When model says YES more than market
        buy_yes = edge > 0.05
        buy_no = edge < -0.05
        neutral = ~buy_yes & ~buy_no

        print(f"    BUY YES  (model > market+5%): {buy_yes.sum():>5} samples")
        print(f"    BUY NO   (model < market-5%): {buy_no.sum():>5} samples")
        print(f"    NEUTRAL  (within 5%):         {neutral.sum():>5} samples")

        if buy_yes.sum() > 0:
            yes_actual = y_test[buy_yes].mean()
            yes_edge = edge[buy_yes].mean()
            print(f"    BUY YES actual win rate: {yes_actual:.3f} "
                  f"(avg edge: {yes_edge:+.3f})")

        if buy_no.sum() > 0:
            no_actual = 1 - y_test[buy_no].mean()
            no_edge = (-edge[buy_no]).mean()
            print(f"    BUY NO  actual win rate: {no_actual:.3f} "
                  f"(avg edge: {no_edge:+.3f})")

        # Simulated PnL
        print(f"\n  SIMULATED PnL (test set, $1 per contract):")
        pnl = 0
        trades = 0
        for i in range(len(test_proba)):
            e = edge[i]
            actual = y_test[i]
            if abs(e) < 0.05:
                continue
            trades += 1
            if e > 0:  # buy YES
                cost = market_price[i]
                pnl += (1.0 - cost) if actual == 1 else (-cost)
            else:  # buy NO
                cost = 1.0 - market_price[i]
                pnl += (1.0 - cost) if actual == 0 else (-cost)

        print(f"    trades: {trades}")
        print(f"    PnL:    ${pnl:.2f}")
        if trades > 0:
            print(f"    per trade: ${pnl/trades:.4f}")

    return test_proba, test_pred


# ==============================================================================
# Step 5: Summary
# ==============================================================================

def summary(model, test_proba, y_test, feature_cols):
    print(f"\n{'=' * W}")
    print(f"  SUMMARY")
    print(f"{'=' * W}")

    test_brier = np.mean((test_proba - y_test) ** 2)
    test_acc = ((test_proba >= 0.5).astype(int) == y_test).mean()

    if test_brier < 0.20:
        grade = "GOOD"
    elif test_brier < 0.24:
        grade = "MODERATE"
    elif test_brier < 0.25:
        grade = "MARGINAL"
    else:
        grade = "POOR (worse than random)"

    print(f"  Model grade: {grade}")
    print(f"  Brier score: {test_brier:.4f} (random=0.2500)")
    print(f"  Accuracy:    {test_acc:.3f}")

    if test_brier < 0.24:
        print(f"\n  RECOMMENDATION: Model shows calibration potential.")
        print(f"  Consider live paper-trading with small size.")
    else:
        print(f"\n  RECOMMENDATION: Need more data or feature improvement.")
        print(f"  Suggestions:")
        print(f"    - Collect more hours of data (more 15-min cycles)")
        print(f"    - Add trade flow features (taker side imbalance)")
        print(f"    - Try per-asset models instead of pooled")

    # Save model predictions for analysis
    pred_file = "kalshi_v3_predictions.csv"
    pred_df = pd.DataFrame({
        "model_p": test_proba,
        "actual": y_test,
    })
    pred_df.to_csv(pred_file, index=False)
    print(f"\n  Predictions saved to {pred_file}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"\n{'#' * W}")
    print(f"  Kalshi ML v3 - Settlement Prediction")
    print(f"{'#' * W}")

    # Step 1
    df, settlements = load_data()
    if df is None:
        return

    # Step 2
    feat_df = engineer_features(df)
    if len(feat_df) < 50:
        print(f"\n  !! Not enough training data ({len(feat_df)} < 50)")
        print(f"  !! Collect more data (need at least ~30min = 2 transitions)")
        return

    # Step 3
    result = train_model(feat_df)
    if result[0] is None:
        return
    model, data_split, feature_cols = result

    # Step 4
    test_proba, test_pred = evaluate(
        model, data_split, feature_cols, feat_df)

    # Step 5
    summary(model, test_proba, data_split[3], feature_cols)


if __name__ == "__main__":
    main()
