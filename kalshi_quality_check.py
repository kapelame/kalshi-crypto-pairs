"""
================================================================================
Kalshi Data Quality Check
================================================================================
在采集完成后运行此脚本, 检查数据质量并决定是否可以训练ML

用法: 在Colab中新建一个cell, 粘贴此代码运行
================================================================================
"""

import pandas as pd
import numpy as np

CSV = "kalshi_features.csv"  # collector导出的CSV路径

# 如果在Colab中CSV在/content/下
import os
if not os.path.exists(CSV) and os.path.exists(f"/content/{CSV}"):
    CSV = f"/content/{CSV}"

df = pd.read_csv(CSV)

print(f"{'=' * 65}")
print(f"  DATA QUALITY REPORT")
print(f"{'=' * 65}")
print(f"  rows:     {len(df)}")

# 时间跨度
span = df["ts_unix"].iloc[-1] - df["ts_unix"].iloc[0]
print(f"  timespan: {span:.0f}s = {span/60:.1f}min = {span/3600:.2f}h")
print(f"  15m cycles: {span/900:.1f}")

# Gap检测
diffs = df["ts_unix"].diff().dropna()
gaps = diffs[diffs > 5]
total_gap = gaps.sum() if len(gaps) > 0 else 0
gap_pct = total_gap / span * 100 if span > 0 else 0

print(f"\n  GAPS:")
print(f"    gaps > 5s:  {len(gaps)}")
print(f"    total gap:  {total_gap:.0f}s ({gap_pct:.1f}%)")
if len(gaps) > 0:
    for idx in gaps.index[:10]:  # 最多显示10个
        print(f"    row {idx}: {diffs[idx]:.0f}s")
    if len(gaps) > 10:
        print(f"    ... and {len(gaps)-10} more")

# 连续段分析
gap_mask = diffs > 5
seg_starts = [0] + list(gap_mask[gap_mask].index)
seg_lengths = []
for i, start in enumerate(seg_starts):
    end = seg_starts[i+1] if i+1 < len(seg_starts) else len(df)
    seg_lengths.append(end - start)
print(f"\n  SEGMENTS:")
print(f"    count:    {len(seg_lengths)}")
print(f"    lengths:  {seg_lengths}")
print(f"    longest:  {max(seg_lengths)} rows")

# 体制检测 (简版)
vw = df["vw_mid"].dropna().values
if len(vw) >= 60:
    slopes = np.zeros(len(vw))
    for i in range(30, len(vw)):
        y = vw[i-30:i]
        x = np.arange(30)
        cov = np.sum((x - x.mean()) * (y - y.mean()))
        var = np.sum((x - x.mean()) ** 2)
        slopes[i] = cov / var if var > 0 else 0

    n_trend = (np.abs(slopes[30:]) > 0.003).sum()
    n_total = len(slopes) - 30
    n_calm = n_total - n_trend

    # 简单波动率分类
    stds = np.zeros(len(vw))
    for i in range(30, len(vw)):
        stds[i] = np.std(vw[i-30:i])
    median_std = np.median(stds[30:])
    n_osc = ((np.abs(slopes[30:]) <= 0.003) & (stds[30:] > median_std)).sum()

    print(f"\n  REGIME:")
    print(f"    trend:     {n_trend:>5} ({n_trend/n_total:.0%})")
    print(f"    non-trend: {n_calm:>5} ({n_calm/n_total:.0%})")
    print(f"    oscillate: {n_osc:>5} (est. of non-trend)")

    # VW mid范围
    print(f"\n  VW_MID RANGE:")
    print(f"    min={vw.min():.4f}  max={vw.max():.4f}  "
          f"mean={vw.mean():.4f}  std={vw.std():.4f}")

# 偏差分析
if "max_abs_dev" in df.columns:
    dev = df["max_abs_dev"].dropna()
    print(f"\n  DEVIATION:")
    print(f"    mean={dev.mean():.4f}  median={dev.median():.4f}  "
          f"max={dev.max():.4f}")
    above_3pct = (dev > 0.03).sum()
    print(f"    above 3%: {above_3pct} ({above_3pct/len(dev):.0%})")

# 信号分布
if "signal" in df.columns:
    sig_counts = df["signal"].value_counts()
    print(f"\n  SIGNALS:")
    for s, c in sig_counts.items():
        print(f"    {s:<20} {c:>5} ({c/len(df):.0%})")

# 最终判定
print(f"\n{'=' * 65}")
print(f"  VERDICT")
print(f"{'=' * 65}")

issues = []
warnings = []

if len(df) < 2000:
    issues.append(f"rows too few: {len(df)} < 2000 minimum")
if span < 3600:
    issues.append(f"timespan too short: {span/60:.0f}min < 60min minimum")
if gap_pct > 10:
    issues.append(f"gap too high: {gap_pct:.1f}% > 10% limit")
elif gap_pct > 5:
    warnings.append(f"gap elevated: {gap_pct:.1f}%")

if len(vw) >= 60:
    if n_trend / n_total < 0.05:
        warnings.append(f"very few trend periods ({n_trend/n_total:.0%}), "
                        f"regime detector may lack training data")
    if n_trend / n_total > 0.50:
        warnings.append(f"mostly trending ({n_trend/n_total:.0%}), "
                        f"convergence strategy may have limited opportunities")

if max(seg_lengths) < 1000:
    warnings.append(f"longest continuous segment only {max(seg_lengths)} rows")

if not issues:
    print(f"  [OK] Data ready for kalshi_ml_v3.py!")
    if warnings:
        print(f"  Notes:")
        for w in warnings:
            print(f"    - {w}")
    print(f"\n  Next: upload kalshi_features.csv and run kalshi_ml_v3.py")
else:
    print(f"  [FAIL] Data not ready:")
    for i in issues:
        print(f"    X {i}")
    if warnings:
        for w in warnings:
            print(f"    ! {w}")
    print(f"\n  Action: continue collecting data")
