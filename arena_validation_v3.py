"""
Arena 검증 v3 — Alpha 스윕
============================
질문: 레짐 가중치 alpha 수준별로 경매 정렬도가 어떻게 달라지는가?

alpha = 0.0: 순수 confidence (v1 기준)
alpha = 0.3: 약한 레짐 nudge
alpha = 0.5: 균형 (추천)
alpha = 1.0: 강한 레짐 반영
alpha = 2.0: 사실상 하드 게이트

지표: 레짐-경매 정렬도 (%p), V^pi 선명도, 분기 안정성
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp

# ── 캐시 로드 ─────────────────────────────────────────────
assert os.path.exists("arena_val_features.pkl"), "arena_validation_v1.py 먼저 실행"
assert os.path.exists("arena_val_hmm.pkl"),      "arena_validation_v1.py 먼저 실행"
assert os.path.exists("arena_val_highvol.pt"),   "arena_validation_v1.py 먼저 실행"
assert os.path.exists("arena_val_lowvol.pt"),    "arena_validation_v1.py 먼저 실행"

with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)

from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN        = 16
STATE_DIM  = 128
ACTION_DIM = 3
SEED       = 42
TRAIN_END  = "2022-06-30"

torch.manual_seed(SEED)
np.random.seed(SEED)

states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
highp_aligned   = high_p.reindex(dates_arr).fillna(0).values

split_idx = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_highp   = highp_aligned[split_idx:]


# ── 모델 ──────────────────────────────────────────────────
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim=128, action_dim=3, h1=64, h2=32):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.LayerNorm(h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(),
        )
        self.policy_head = nn.Linear(h2, action_dim)
        self.value_head  = nn.Linear(h2, 1)

    def forward(self, s):
        f      = self.trunk(s)
        logits = self.policy_head(f)
        probs  = F.softmax(logits, dim=-1)
        H      = -torch.sum(probs * torch.log(probs + 1e-9))
        conf   = (1.0 - H / torch.log(torch.tensor(float(self.action_dim)))).item()
        vpi    = self.value_head(f).squeeze(-1).item()
        return logits, conf, vpi


net_hv = PolicyValueNet(); net_hv.load_state_dict(torch.load("arena_val_highvol.pt", map_location='cpu'))
net_lv = PolicyValueNet(); net_lv.load_state_dict(torch.load("arena_val_lowvol.pt",  map_location='cpu'))
net_hv.eval(); net_lv.eval()


# ══════════════════════════════════════════════════════════
# Alpha 스윕
# ══════════════════════════════════════════════════════════
ALPHAS = [0.0, 0.3, 0.5, 1.0, 2.0]

def run_auction(alpha: float):
    """alpha로 경매 시뮬레이션 → 지표 반환"""
    vpi_hv, vpi_lv, winners = [], [], []

    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)

            p_high = float(val_highp[i])

            # machine.py tick_auction 로직 그대로
            regime_hv = 1.0 + alpha * p_high
            regime_lv = 1.0 + alpha * (1.0 - p_high)

            bid_hv = conf_hv * regime_hv
            bid_lv = conf_lv * regime_lv

            vpi_hv.append(vpi_h)
            vpi_lv.append(vpi_l)
            winners.append(0 if bid_hv >= bid_lv else 1)

    vpi_hv  = np.array(vpi_hv)
    vpi_lv  = np.array(vpi_lv)
    winners = np.array(winners)

    hm = val_regimes == high_state
    lm = ~hm

    align = 0.0
    if hm.sum() > 0 and lm.sum() > 0:
        align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100

    hv_win_rate_high = (winners[hm] == 0).mean() * 100 if hm.sum() > 0 else 0
    hv_win_rate_low  = (winners[lm] == 0).mean() * 100 if lm.sum() > 0 else 0

    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std

    _, ks_p = ks_2samp(vpi_hv, vpi_lv)

    return {
        "alpha":            alpha,
        "alignment":        align,
        "hv_win_high_vol":  hv_win_rate_high,
        "hv_win_low_vol":   hv_win_rate_low,
        "sharpness":        sharpness,
        "ks_p":             ks_p,
        "winners":          winners,
        "vpi_hv":           vpi_hv,
        "vpi_lv":           vpi_lv,
    }


print(f"{'alpha':>6} {'align':>9} {'HV@high':>9} {'HV@low':>9} {'sharp':>8} {'ks_p':>8}")
print("-" * 55)

results = []
for alpha in ALPHAS:
    r = run_auction(alpha)
    results.append(r)
    print(f"  {alpha:4.1f}  {r['alignment']:+8.1f}%p  "
          f"{r['hv_win_high_vol']:7.1f}%  "
          f"{r['hv_win_low_vol']:7.1f}%  "
          f"{r['sharpness']:7.3f}  "
          f"{r['ks_p']:.4f}")


# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 정렬도 vs alpha
ax = axes[0, 0]
aligns = [r["alignment"] for r in results]
bars = ax.bar([str(a) for a in ALPHAS], aligns,
              color=['steelblue' if a >= 0 else 'salmon' for a in aligns],
              edgecolor='black', linewidth=0.5)
ax.axhline(0, color='gray', lw=0.8, linestyle='--')
for bar, val in zip(bars, aligns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:+.1f}', ha='center', va='bottom', fontsize=9)
ax.set_title("레짐-경매 정렬도 vs Alpha")
ax.set_xlabel("alpha"); ax.set_ylabel("Alignment (%p)")

# 2. HV 낙찰률 (고변동성 vs 저변동성 국면)
ax = axes[0, 1]
x   = np.arange(len(ALPHAS))
w   = 0.35
ax.bar(x - w/2, [r["hv_win_high_vol"] for r in results],
       width=w, label='High vol regime', color='salmon', edgecolor='black', lw=0.5)
ax.bar(x + w/2, [r["hv_win_low_vol"] for r in results],
       width=w, label='Low vol regime', color='steelblue', edgecolor='black', lw=0.5)
ax.axhline(50, color='gray', lw=0.8, linestyle='--', label='random (50%)')
ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ALPHAS])
ax.set_title("HighVol 낙찰률 by 레짐")
ax.set_xlabel("alpha"); ax.set_ylabel("HighVol win rate (%)")
ax.legend(fontsize=8); ax.set_ylim(0, 105)

# 3. 승자 시계열 (alpha=0 vs best alpha)
best_alpha_idx = int(np.argmax([r["alignment"] for r in results]))
ax = axes[1, 0]
r0   = results[0]   # alpha=0
rbest = results[best_alpha_idx]

# HMM 레짐 배경
ax.fill_between(val_dates, val_highp, alpha=0.2, color='salmon', label='P(High Regime)')
w0 = pd.Series(r0["winners"].astype(float), index=val_dates).rolling(10).mean()
wb = pd.Series(rbest["winners"].astype(float), index=val_dates).rolling(10).mean()
ax.plot(val_dates, 1 - w0.values, lw=0.8, color='gray',
        alpha=0.8, label=f'alpha=0.0 HV rate')
ax.plot(val_dates, 1 - wb.values, lw=0.8, color='red',
        alpha=0.8, label=f'alpha={ALPHAS[best_alpha_idx]} HV rate')
ax.set_title(f"HighVol 낙찰률 추이 (alpha=0 vs best={ALPHAS[best_alpha_idx]})")
ax.set_ylabel("HighVol win rate (rolling 10)"); ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 4. V^pi 선명도 vs alpha
ax = axes[1, 1]
sharps = [r["sharpness"] for r in results]
ax.plot(ALPHAS, sharps, marker='o', ms=8, lw=1.5, color='purple')
for a, s in zip(ALPHAS, sharps):
    ax.annotate(f'{s:.2f}', (a, s), textcoords="offset points",
                xytext=(0, 8), ha='center', fontsize=9)
ax.set_title("V^pi 분기 선명도 vs Alpha")
ax.set_xlabel("alpha"); ax.set_ylabel("Sharpness (Cohen d)")

plt.tight_layout()
plt.savefig("arena_validation_v3.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v3.png")


# ══════════════════════════════════════════════════════════
# 최종 판정
# ══════════════════════════════════════════════════════════
best_r = results[best_alpha_idx]

print("\n" + "=" * 60)
print("Arena 검증 v3 -- Alpha 스윕 결과")
print("=" * 60)
print(f"최적 alpha: {ALPHAS[best_alpha_idx]}")
print(f"  정렬도:  {best_r['alignment']:+.1f}%p  "
      f"(alpha=0 대비 {best_r['alignment']-results[0]['alignment']:+.1f}%p)")
print(f"  HV@high: {best_r['hv_win_high_vol']:.1f}%  "
      f"HV@low: {best_r['hv_win_low_vol']:.1f}%")
print(f"  선명도:  {best_r['sharpness']:.3f}")

print(f"\nalpha별 요약:")
for r in results:
    marker = " <-- best" if r["alpha"] == ALPHAS[best_alpha_idx] else ""
    print(f"  alpha={r['alpha']:.1f}: align={r['alignment']:+.1f}%p  "
          f"sharp={r['sharpness']:.2f}{marker}")

if best_r["alignment"] > 20:
    verdict = f"alpha={ALPHAS[best_alpha_idx]} 채택 권장 -- machine.py 통합"
elif best_r["alignment"] > 5:
    verdict = f"alpha={ALPHAS[best_alpha_idx]} 약한 개선 -- 추가 튜닝 필요"
else:
    verdict = "레짐 가중치 효과 없음"

print(f"\n판정: {verdict}")
print("=" * 60)
