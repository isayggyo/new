"""
GARCH 변동성 예측 검증 v1
==========================
질문: GARCH(1,1)이 BTC 변동성을 유의미하게 예측하는가?
      그리고 예측 변동성이 크래시 전에 상승 신호를 주는가?

설계:
  - 데이터: BTC 일봉 (yfinance)
  - 모델: GARCH(1,1) + GJR-GARCH(1,1) (비대칭 충격 반영)
  - 검증:
    1. 예측 정확도 (QLIKE, MSE) vs 벤치마크 (역사적 변동성)
    2. 크래시 전 30/60일 내 예측 변동성 상승 추세 (Kendall tau)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy.stats import kendalltau

START = "2017-01-01"
END   = "2023-12-31"

CRASH_EVENTS = {
    "2018 Bull Crash":   pd.Timestamp("2018-01-08"),
    "2020 COVID Crash":  pd.Timestamp("2020-03-12"),
    "2022 FTX Collapse": pd.Timestamp("2022-11-08"),
}

# ═══════════════════════════════
# 1. 데이터
# ═══════════════════════════════
print("데이터 로딩...")
raw   = yf.download("BTC-USD", start=START, end=END,
                    auto_adjust=True, progress=False)
close = raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

ret = np.log(close / close.shift(1)).dropna() * 100  # % 수익률
print(f"  {len(ret)}일치 로드 완료")

# ═══════════════════════════════
# 2. 롤링 예측 (walk-forward)
# ═══════════════════════════════
TRAIN_SIZE = 365   # 초기 학습 1년
results_garch = []
results_gjr   = []
realized_vol  = []

print(f"롤링 예측 중 ({len(ret) - TRAIN_SIZE}일)...")
for i in range(TRAIN_SIZE, len(ret)):
    train = ret.iloc[:i]
    rv    = train.iloc[-20:].std()  # 20일 실현변동성 (벤치마크)
    realized_vol.append(rv)

    try:
        # GARCH(1,1)
        m1 = arch_model(train, vol='Garch', p=1, q=1, dist='normal')
        r1 = m1.fit(disp='off', show_warning=False)
        fc1 = r1.forecast(horizon=1)
        results_garch.append(float(np.sqrt(fc1.variance.iloc[-1, 0])))

        # GJR-GARCH(1,1)
        m2 = arch_model(train, vol='Garch', p=1, o=1, q=1, dist='normal')
        r2 = m2.fit(disp='off', show_warning=False)
        fc2 = r2.forecast(horizon=1)
        results_gjr.append(float(np.sqrt(fc2.variance.iloc[-1, 0])))
    except Exception:
        results_garch.append(np.nan)
        results_gjr.append(np.nan)

    if (i - TRAIN_SIZE) % 100 == 0:
        print(f"  진행: {i - TRAIN_SIZE}/{len(ret) - TRAIN_SIZE}")

pred_dates = ret.index[TRAIN_SIZE:]
garch_vol  = pd.Series(results_garch, index=pred_dates)
gjr_vol    = pd.Series(results_gjr,   index=pred_dates)
real_vol   = pd.Series(realized_vol,  index=pred_dates)

# ═══════════════════════════════
# 3. 예측 정확도
# ═══════════════════════════════
actual_vol = ret.rolling(20).std().iloc[TRAIN_SIZE:]

valid = actual_vol.dropna()
g_mse  = ((garch_vol[valid.index] - valid) ** 2).mean()
gjr_mse = ((gjr_vol[valid.index] - valid) ** 2).mean()
hv_mse  = ((real_vol[valid.index] - valid) ** 2).mean()

print(f"\n예측 정확도 (MSE vs 실현변동성):")
print(f"  Historical Vol: {hv_mse:.4f}")
print(f"  GARCH(1,1):     {g_mse:.4f}")
print(f"  GJR-GARCH:      {gjr_mse:.4f}")

# ═══════════════════════════════
# 4. 크래시 전 추세 분석
# ═══════════════════════════════
print("\n크래시 전 변동성 추세:")
crash_results = []

for crash_name, crash_date in CRASH_EVENTS.items():
    tau, p, pre_g = 0.0, 1.0, pd.Series(dtype=float)
    for window in [30, 60]:
        pre_start = crash_date - pd.Timedelta(days=window)
        _pre = garch_vol[(garch_vol.index >= pre_start) &
                          (garch_vol.index < crash_date)].dropna()
        if len(_pre) < 10:
            continue
        pre_g = _pre
        tau, p = kendalltau(np.arange(len(pre_g)), pre_g.values)

    crash_val = garch_vol.get(crash_date,
                    garch_vol.iloc[garch_vol.index.get_indexer([crash_date], method='nearest')[0]])

    result = dict(crash=crash_name, crash_date=crash_date,
                  tau=tau, p=p, crash_vol=crash_val,
                  pre_g=pre_g)
    crash_results.append(result)

    signal = tau > 0 and p < 0.05
    print(f"\n  [{crash_name}]")
    print(f"    크래시 당일 예측 vol: {crash_val:.4f}%")
    print(f"    사전 60일 Kendall tau={tau:.3f} p={p:.4f} "
          f"{'[TREND]' if signal else '[NO TREND]'}")

# ═══════════════════════════════
# 5. 시각화
# ═══════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax = axes[0]
ax.plot(garch_vol.index, garch_vol.values,
        color='steelblue', lw=0.8, label='GARCH(1,1)', alpha=0.8)
ax.plot(gjr_vol.index, gjr_vol.values,
        color='darkorange', lw=0.8, label='GJR-GARCH', alpha=0.8)
ax.plot(actual_vol.index, actual_vol.values,
        color='gray', lw=0.6, label='Realized Vol', alpha=0.6)
for name, date in CRASH_EVENTS.items():
    ax.axvline(date, color='red', lw=1.5, linestyle='--')
ax.set_title("GARCH Predicted Volatility vs Realized")
ax.set_ylabel("Volatility (%)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

ax = axes[1]
for r in crash_results:
    if r["pre_g"] is None or len(r["pre_g"]) == 0:
        continue
    ax.plot(r["pre_g"].index, r["pre_g"].values,
            lw=1.2, label=f"{r['crash']} tau={r['tau']:.2f}")
    ax.axvline(r["crash_date"], color='red', lw=1.5, linestyle='--')
ax.set_title("Pre-crash GARCH Vol (60-day window)")
ax.set_ylabel("GARCH Vol (%)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig("garch_validation_v1.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: garch_validation_v1.png")

# ═══════════════════════════════
# 6. 요약
# ═══════════════════════════════
print("\n" + "=" * 60)
print("GARCH 검증 v1 -- 최종 판정")
print("=" * 60)
print(f"예측 정확도: GARCH MSE={g_mse:.4f} vs HV={hv_mse:.4f} "
      f"({'GARCH 우위' if g_mse < hv_mse else 'HV 우위'})")
signal_count = sum(1 for r in crash_results
                   if r["tau"] > 0 and r["p"] < 0.05)
print(f"크래시 전 상승 추세: {signal_count}/{len(crash_results)}")
for r in crash_results:
    sig = "[SIGNAL]" if (r["tau"] > 0 and r["p"] < 0.05) else "[NULL]"
    print(f"  {r['crash']:22s} tau={r['tau']:+.3f} p={r['p']:.4f} {sig}")

if g_mse < hv_mse and signal_count >= 2:
    print("\n판정: SIGNAL -- GARCH 유효. 버블 지표와 조합 진행")
elif g_mse < hv_mse or signal_count >= 1:
    print("\n판정: WEAK -- 부분 유효")
else:
    print("\n판정: NULL -- GARCH 예측력 없음")
print("=" * 60)
