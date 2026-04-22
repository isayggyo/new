"""
Arena 검증 v5 - P vs Q 측도 비교
==================================
질문: HMM(물리 측도 P) vs 옵션 내재(위험 중립 측도 Q) 레짐 확률을
     경매에 사용했을 때 alignment가 달라지는가?

이론:
  dQ/dP = 확률적 할인 인자 (Stochastic Discount Factor)
  고변동성 상태에서 SDF가 높음 (scarcity premium)

근사:
  lambda_h = DVOL / RV  (고변동성 상태의 상대 가격)
  lambda_l = 1.0        (정규화 기준)

  Q_high = P_high * lambda_h / (P_high * lambda_h + (1-P_high))

  DVOL >> RV (공포 국면): Q_high > P_high
  DVOL == RV (중립):      Q_high == P_high
  DVOL <  RV (실현 > 내재): Q_high < P_high
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
from scipy.stats import ks_2samp, pearsonr

assert os.path.exists("arena_val_features.pkl")
assert os.path.exists("arena_val_hmm.pkl")
assert os.path.exists("arena_val_highvol.pt")
assert os.path.exists("arena_val_lowvol.pt")
assert os.path.exists("dvol_btc.csv")

with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)

dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)

# ── VRP, Q 측도 계산 ───────────────────────────────────────
iv_ann = dvol['dvol'] / 100.0
rv_ann = feat['btc_vol'] * np.sqrt(252)

# 라돈-니코딤 근사: lambda_h = DVOL/RV
lambda_h = (iv_ann / rv_ann.reindex(iv_ann.index)).clip(0.5, 5.0)  # 극단값 방어

p_high_full = high_p.reindex(feat.index).fillna(0.5)

# Q_high 계산
lh = lambda_h.reindex(feat.index).fillna(1.0)
q_high_num = p_high_full * lh
q_high = q_high_num / (q_high_num + (1 - p_high_full))

# P-Q 갭
pq_gap = q_high - p_high_full

print("=== P vs Q 측도 통계 ===")
overlap_mask = lh != 1.0  # DVOL 유효 구간
print(f"P_high 평균: {p_high_full[overlap_mask].mean():.3f}")
print(f"Q_high 평균: {q_high[overlap_mask].mean():.3f}")
print(f"P-Q 갭 평균: {pq_gap[overlap_mask].mean():+.3f}  (양수=Q가 P보다 고변동성 더 무겁게)")
print(f"lambda_h 평균: {lh[overlap_mask].mean():.3f}  (DVOL/RV)")

# ── state(128) 구성 ────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN       = 16
TRAIN_END = "2022-06-30"
SEED      = 42
ALPHA     = 2.0

torch.manual_seed(SEED)
np.random.seed(SEED)

states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
p_high_aligned  = high_p.reindex(dates_arr).fillna(0).values
q_high_aligned  = q_high.reindex(dates_arr).fillna(high_p.reindex(dates_arr).fillna(0.5)).values
lambda_aligned  = lh.reindex(dates_arr).fillna(1.0).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_p_high  = p_high_aligned[split_idx:]
val_q_high  = q_high_aligned[split_idx:]
val_lambda  = lambda_aligned[split_idx:]


# ── 모델 로드 (v1과 동일 학습된 에이전트) ─────────────────
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


net_hv = PolicyValueNet()
net_lv = PolicyValueNet()
net_hv.load_state_dict(torch.load("arena_val_highvol.pt", map_location='cpu'))
net_lv.load_state_dict(torch.load("arena_val_lowvol.pt",  map_location='cpu'))
net_hv.eval(); net_lv.eval()


# ── 경매 시뮬레이션 ────────────────────────────────────────
def run_auction(prob_arr, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)
            p = float(prob_arr[i])
            bid_hv = conf_hv * (1.0 + ALPHA * p)
            bid_lv = conf_lv * (1.0 + ALPHA * (1.0 - p))
            vpi_hv.append(vpi_h)
            vpi_lv.append(vpi_l)
            winners.append(0 if bid_hv >= bid_lv else 1)

    vpi_hv  = np.array(vpi_hv)
    vpi_lv  = np.array(vpi_lv)
    winners = np.array(winners)

    hm = val_regimes == high_state
    lm = ~hm
    align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100 \
            if hm.sum() > 0 and lm.sum() > 0 else 0.0

    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)

    print(f"[{label}]  align={align:+.1f}%p  sharp={sharpness:.3f}  KS p={ks_p:.4f}")
    return {"label": label, "align": align, "sharpness": sharpness,
            "ks_p": ks_p, "winners": winners, "vpi_hv": vpi_hv, "vpi_lv": vpi_lv}


print("\n=== 경매 비교 ===")
res_p = run_auction(val_p_high, "P-measure (HMM)")
res_q = run_auction(val_q_high, "Q-measure (SDF 보정)")


# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. P vs Q 시계열 + lambda
ax = axes[0]
ax2 = ax.twinx()
ax.plot(val_dates, val_p_high, color='steelblue', lw=1.0, alpha=0.8, label='P_high (HMM)')
ax.plot(val_dates, val_q_high, color='darkorange', lw=1.0, alpha=0.8, label='Q_high (SDF)')
ax.fill_between(val_dates, val_p_high, val_q_high,
                where=val_q_high > val_p_high, alpha=0.2, color='red', label='Q > P (fear premium)')
ax.fill_between(val_dates, val_p_high, val_q_high,
                where=val_q_high < val_p_high, alpha=0.2, color='green', label='Q < P (IV < RV)')
ax2.plot(val_dates, val_lambda, color='purple', lw=0.7, alpha=0.5, linestyle='--', label='lambda (DVOL/RV)')
ax2.axhline(1.0, color='purple', lw=0.5, linestyle=':')
ax2.set_ylabel("lambda (DVOL/RV)", color='purple')
ax.set_title("P vs Q 레짐 확률 (물리 측도 vs 위험 중립 측도)")
ax.set_ylabel("Regime Probability")
ax.legend(fontsize=7, loc='upper left')
ax2.legend(fontsize=7, loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 2. 낙찰 추이 비교
ax = axes[1]
hm_mask = val_regimes == high_state
ax.fill_between(val_dates, hm_mask.astype(float), alpha=0.15, color='salmon', label='High vol regime (HMM)')
wp = pd.Series(res_p["winners"].astype(float), index=val_dates).rolling(10).mean()
wq = pd.Series(res_q["winners"].astype(float), index=val_dates).rolling(10).mean()
ax.plot(val_dates, 1 - wp.values, lw=0.9, color='steelblue', label=f'P-measure HV rate (align={res_p["align"]:+.1f}%p)')
ax.plot(val_dates, 1 - wq.values, lw=0.9, color='darkorange', label=f'Q-measure HV rate (align={res_q["align"]:+.1f}%p)')
ax.axhline(0.5, color='gray', lw=0.6, linestyle='--')
ax.set_title("HighVol 낙찰률 추이 - P vs Q 측도")
ax.set_ylabel("HV win rate (rolling 10)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 3. P-Q 갭 vs 낙찰률 변화
ax = axes[2]
pq_gap_val = val_q_high - val_p_high
winner_diff = (1 - wq.values) - (1 - wp.values)
ax2 = ax.twinx()
ax.bar(val_dates, pq_gap_val, alpha=0.4,
       color=['red' if g > 0 else 'green' for g in pq_gap_val],
       width=1, label='Q - P gap')
ax2.plot(val_dates, pd.Series(winner_diff, index=val_dates).rolling(10).mean(),
         color='black', lw=0.9, label='HV win rate diff (Q-P)')
ax2.axhline(0, color='black', lw=0.5, linestyle='--')
ax.set_title("P-Q 갭 vs 낙찰률 변화 (Q가 P보다 HV를 더 낙찰시키는가?)")
ax.set_ylabel("Q_high - P_high")
ax2.set_ylabel("HV rate diff (Q - P)")
ax.legend(fontsize=7, loc='upper left')
ax2.legend(fontsize=7, loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig("arena_validation_v5.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v5.png")


# ══════════════════════════════════════════════════════════
# 최종 판정
# ══════════════════════════════════════════════════════════
pq_corr, pq_corr_p = pearsonr(val_p_high, val_q_high)
gap_winner_corr, gap_winner_p = pearsonr(
    pq_gap_val,
    pd.Series(winner_diff, index=val_dates).fillna(0).values
)

print("\n" + "=" * 60)
print("Arena v5 - P vs Q 측도 비교 결과")
print("=" * 60)
print(f"P_high vs Q_high 상관:     r={pq_corr:.3f}  (p={pq_corr_p:.4f})")
print(f"P-Q 갭 vs 낙찰 변화 상관: r={gap_winner_corr:.3f}  (p={gap_winner_p:.4f})")
print()
print(f"{'':20s} {'P-measure':>12} {'Q-measure':>12} {'변화':>10}")
print("-" * 56)
print(f"  {'Alignment(%p)':18s} {res_p['align']:>+12.1f} {res_q['align']:>+12.1f} "
      f"{res_q['align']-res_p['align']:>+10.1f}")
print(f"  {'Sharpness':18s} {res_p['sharpness']:>12.3f} {res_q['sharpness']:>12.3f} "
      f"{res_q['sharpness']-res_p['sharpness']:>+10.3f}")

delta_align = res_q['align'] - res_p['align']
if abs(delta_align) < 2.0:
    verdict = "Q-measure 조정 효과 없음 - P/Q 갭이 경매 결과에 반영 안됨"
elif delta_align > 0:
    verdict = f"Q-measure가 더 나음 (+{delta_align:.1f}%p) - 공포 프리미엄이 경매를 개선"
else:
    verdict = f"P-measure가 더 나음 ({delta_align:.1f}%p) - HMM이 옵션 내재 확률보다 정확"

print(f"\n판정: {verdict}")
print("=" * 60)
