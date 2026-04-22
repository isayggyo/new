"""
변동성 레짐 감지 v1 -- HMM 2-state
=====================================
질문: BTC 일봉 수익률에서 2-state HMM으로 Low/High vol 레짐을
      구분하고, 레짐 전환이 Flash Crash와 일치하는가?

설계:
  - 데이터: BTC 일봉 (yfinance)
  - 피처: 20일 실현변동성 + 로그수익률
  - 모델: GaussianHMM (2 states)
  - 검증: 크래시 전 N일 내 High vol 레짐 전환 여부
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

ASSETS  = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"]
START   = "2017-01-01"
END     = "2023-12-31"
VOL_WIN = 20   # 실현변동성 롤링 윈도우

CRASH_EVENTS = {
    "2018 Bull Crash":   pd.Timestamp("2018-01-08"),
    "2020 COVID Crash":  pd.Timestamp("2020-03-12"),
    "2022 FTX Collapse": pd.Timestamp("2022-11-08"),
}

# ═══════════════════════════════
# 1. 데이터
# ═══════════════════════════════
print("데이터 로딩...")
frames = {}
for asset in ASSETS:
    raw = yf.download(asset, start=START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    frames[asset] = close

prices  = pd.DataFrame(frames).dropna()
returns = np.log(prices / prices.shift(1)).dropna()
print(f"  {len(returns)}일치 로드 완료")

# ═══════════════════════════════
# 2. 피처 생성
# ═══════════════════════════════
btc_ret = returns["BTC-USD"]
btc_vol = btc_ret.rolling(VOL_WIN).std()

# 다변량: BTC 수익률 + 실현변동성
features = pd.DataFrame({
    "ret": btc_ret,
    "vol": btc_vol,
}).dropna()

scaler = StandardScaler()
X = scaler.fit_transform(features.values)

# ═══════════════════════════════
# 3. HMM 학습
# ═══════════════════════════════
print("HMM 학습 중...")
model = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=200, random_state=42)
model.fit(X)

states     = model.predict(X)
state_prob = model.predict_proba(X)

# High vol 레짐 = 변동성 평균이 높은 state
vol_by_state = {s: features["vol"].values[states == s].mean()
                for s in [0, 1]}
high_state = max(vol_by_state, key=vol_by_state.get)
low_state  = 1 - high_state

print(f"  State {low_state}: Low vol  (mean={vol_by_state[low_state]:.4f})")
print(f"  State {high_state}: High vol (mean={vol_by_state[high_state]:.4f})")

regime_series = pd.Series(states, index=features.index)
high_prob     = pd.Series(state_prob[:, high_state], index=features.index)

# ═══════════════════════════════
# 4. 크래시 분석
# ═══════════════════════════════
print("\n크래시 분석:")
results = []
for crash_name, crash_date in CRASH_EVENTS.items():
    if crash_date not in regime_series.index:
        crash_date = regime_series.index[regime_series.index.get_loc(crash_date, method='nearest')]

    # 크래시 전 30/60일 내 High vol 전환 여부
    for window in [30, 60]:
        pre_start = crash_date - pd.Timedelta(days=window)
        pre_reg   = regime_series[(regime_series.index >= pre_start) &
                                   (regime_series.index < crash_date)]
        if len(pre_reg) == 0:
            continue
        high_days = (pre_reg == high_state).sum()
        high_pct  = high_days / len(pre_reg) * 100

        # 전환 시점: 처음으로 High vol로 바뀐 날
        transitions = pre_reg[(pre_reg == high_state) &
                               (pre_reg.shift(1) == low_state)]
        first_trans = transitions.index[0] if len(transitions) > 0 else None
        days_before = (crash_date - first_trans).days if first_trans else None

    # 크래시 당일 state
    crash_state = regime_series.get(crash_date, None)
    crash_prob  = high_prob.get(crash_date, None)

    result = dict(crash=crash_name, crash_date=crash_date,
                  crash_state="HIGH" if crash_state == high_state else "LOW",
                  crash_high_prob=crash_prob,
                  high_pct_60d=high_pct,
                  first_transition=first_trans,
                  days_before=days_before)
    results.append(result)

    print(f"\n  [{crash_name}]")
    print(f"    크래시 당일 레짐: {result['crash_state']} (P={crash_prob:.3f})")
    print(f"    사전 60일 High vol 비율: {high_pct:.1f}%")
    if days_before:
        print(f"    첫 High vol 전환: 크래시 {days_before}일 전")
    else:
        print(f"    첫 High vol 전환: 없음 (이미 High vol 상태)")

# ═══════════════════════════════
# 5. 시각화
# ═══════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 상단: BTC 가격 + 레짐 색상
ax = axes[0]
ax.plot(prices["BTC-USD"].index, prices["BTC-USD"].values,
        color='black', lw=0.8, label='BTC Price')
# 레짐 배경색
for i in range(len(regime_series) - 1):
    color = 'salmon' if regime_series.iloc[i] == high_state else 'lightgreen'
    ax.axvspan(regime_series.index[i], regime_series.index[i+1],
               alpha=0.3, color=color, lw=0)
for crash_name, crash_date in CRASH_EVENTS.items():
    ax.axvline(crash_date, color='red', lw=1.5, linestyle='--')
ax.set_yscale('log')
ax.set_title("BTC Price + HMM Regime (Green=Low, Red=High)")
ax.set_ylabel("Price (log)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# 중단: High vol 확률
ax = axes[1]
ax.fill_between(high_prob.index, high_prob.values,
                alpha=0.6, color='salmon')
ax.axhline(0.5, color='gray', linestyle='--', lw=1)
for crash_name, crash_date in CRASH_EVENTS.items():
    ax.axvline(crash_date, color='red', lw=1.5, linestyle='--')
ax.set_title("P(High Vol Regime)")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# 하단: 실현변동성
ax = axes[2]
ax.plot(features.index, features["vol"].values, color='steelblue', lw=0.8)
for crash_name, crash_date in CRASH_EVENTS.items():
    ax.axvline(crash_date, color='red', lw=1.5, linestyle='--',
               label=crash_name)
ax.set_title("BTC 20-day Realized Volatility")
ax.set_ylabel("Vol")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("regime_detection_v1.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: regime_detection_v1.png")

# ═══════════════════════════════
# 6. 요약
# ═══════════════════════════════
print("\n" + "=" * 60)
print("변동성 레짐 감지 v1 -- 최종 판정")
print("=" * 60)
on_crash = sum(1 for r in results if r["crash_state"] == "HIGH")
print(f"크래시 당일 High vol 레짐: {on_crash}/{len(results)}")
for r in results:
    print(f"  {r['crash']:22s} {r['crash_state']:4s} P={r['crash_high_prob']:.3f} "
          f"| 사전60일 High {r['high_pct_60d']:.0f}%"
          + (f" | 전환 {r['days_before']}일전" if r['days_before'] else ""))
print("=" * 60)
