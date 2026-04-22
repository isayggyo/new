"""
Arena 검증 v6 - 인트라데이 HMM P_high vs 일별 HMM P_high
=========================================================
질문: 1시간 단위로 재학습한 HMM의 regime_prob을 Arena에 넣으면
     alignment가 개선되는가?

변경: P_high 소스만 교체 (에이전트 동일)
  기존: 일별 HMM (arena_val_hmm.pkl)
  신규: 1시간 HMM → 일별 다운샘플 (intraday_hmm.pkl)
"""
import warnings; warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp

# ── 데이터 로드 ────────────────────────────────────────────
with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime_daily, high_p_daily, high_state_daily = pickle.load(f)
with open("intraday_hmm.pkl", 'rb') as f:
    intra = pickle.load(f)

# 1h P_high -> 일별 (일 종가 시점 기준)
high_p_1h    = intra["high_p_1h"]
high_p_intra = high_p_1h.resample('D').last()   # 일 마지막 시간봉
high_p_intra.index = high_p_intra.index.normalize()

# ── state(128) ─────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN       = 16
TRAIN_END = "2022-06-30"
ALPHA     = 2.0
SEED      = 42
torch.manual_seed(SEED); np.random.seed(SEED)

states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime_daily.reindex(dates_arr).fillna(0).values.astype(int)

# 두 P_high 시계열 정렬
p_daily_aligned = high_p_daily.reindex(dates_arr).fillna(0).values
p_intra_aligned = high_p_intra.reindex(dates_arr).fillna(
    pd.Series(p_daily_aligned, index=dates_arr)
).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_p_daily = p_daily_aligned[split_idx:]
val_p_intra = p_intra_aligned[split_idx:]

print(f"검증 구간: {val_dates[0].date()} ~ {val_dates[-1].date()}  ({len(val_data)}일)")
print(f"P_high(일별) 평균: {val_p_daily.mean():.3f}  >0.5 비율: {(val_p_daily>0.5).mean()*100:.1f}%")
print(f"P_high(인트라) 평균: {val_p_intra.mean():.3f}  >0.5 비율: {(val_p_intra>0.5).mean()*100:.1f}%")


# ── 모델 로드 ──────────────────────────────────────────────
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


# ── 경매 ──────────────────────────────────────────────────
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
            vpi_hv.append(vpi_h); vpi_lv.append(vpi_l)
            winners.append(0 if bid_hv >= bid_lv else 1)

    vpi_hv  = np.array(vpi_hv)
    vpi_lv  = np.array(vpi_lv)
    winners = np.array(winners)

    hm = val_regimes == high_state_daily
    lm = ~hm
    align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100 \
            if hm.sum() > 0 and lm.sum() > 0 else 0.0

    hv_high = (winners[hm] == 0).mean() * 100 if hm.sum() > 0 else 0
    hv_low  = (winners[lm] == 0).mean() * 100 if lm.sum() > 0 else 0

    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)

    print(f"[{label}]")
    print(f"  align={align:+.1f}%p  HV@high={hv_high:.1f}%  HV@low={hv_low:.1f}%"
          f"  sharp={sharpness:.3f}  KS p={ks_p:.4f}")
    return dict(label=label, align=align, hv_high=hv_high, hv_low=hv_low,
                sharpness=sharpness, ks_p=ks_p, winners=winners,
                vpi_hv=vpi_hv, vpi_lv=vpi_lv)


print("\n=== 경매 비교 ===")
res_daily = run_auction(val_p_daily, "일별 HMM")
res_intra = run_auction(val_p_intra, "인트라데이 HMM (1h)")


# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# 1. P_high 비교 시계열
ax = axes[0]
ax.plot(val_dates, val_p_daily, color='steelblue', lw=0.9,
        alpha=0.8, label='P_high 일별 HMM')
ax.plot(val_dates, val_p_intra, color='darkorange', lw=0.9,
        alpha=0.8, label='P_high 인트라데이 HMM (1h)')
ax.fill_between(val_dates,
                (val_regimes == high_state_daily).astype(float) * 0.05,
                alpha=0.2, color='red', label='High regime (일별 HMM)')
ax.axhline(0.5, color='gray', lw=0.6, linestyle='--')
ax.set_title("P_high 비교: 일별 HMM vs 인트라데이 HMM")
ax.set_ylabel("P(High Vol Regime)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 2. 낙찰률 추이
ax = axes[1]
wd = pd.Series(res_daily["winners"].astype(float), index=val_dates).rolling(10).mean()
wi = pd.Series(res_intra["winners"].astype(float), index=val_dates).rolling(10).mean()
ax.fill_between(val_dates,
                (val_regimes == high_state_daily).astype(float),
                alpha=0.1, color='salmon', label='High regime')
ax.plot(val_dates, 1 - wd.values, lw=0.9, color='steelblue',
        label=f'일별 HMM (align={res_daily["align"]:+.1f}%p)')
ax.plot(val_dates, 1 - wi.values, lw=0.9, color='darkorange',
        label=f'인트라데이 HMM (align={res_intra["align"]:+.1f}%p)')
ax.axhline(0.5, color='gray', lw=0.6, linestyle='--')
ax.set_title("HighVol 낙찰률 추이 (rolling 10)")
ax.set_ylabel("HV win rate")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 3. 비교 막대
ax = axes[2]
metrics = ['align', 'hv_high', 'hv_low', 'sharpness']
labels  = ['Alignment(%p)', 'HV@High(%)', 'HV@Low(%)', 'Sharpness']
x = np.arange(len(metrics))
w = 0.35
v_d = [res_daily[m] for m in metrics]
v_i = [res_intra[m] for m in metrics]
ax.bar(x - w/2, v_d, width=w, label='일별 HMM',       color='steelblue', edgecolor='black', lw=0.5)
ax.bar(x + w/2, v_i, width=w, label='인트라데이 HMM', color='darkorange', edgecolor='black', lw=0.5)
for xi, (vd, vi) in zip(x, zip(v_d, v_i)):
    ax.text(xi - w/2, vd + 0.3, f'{vd:.1f}', ha='center', fontsize=8)
    ax.text(xi + w/2, vi + 0.3, f'{vi:.1f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("지표 비교 (alpha=2.0)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("arena_validation_v6.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v6.png")


# ── 최종 판정 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Arena v6 - 인트라데이 HMM 비교 결과")
print("=" * 60)
delta = res_intra['align'] - res_daily['align']
if delta > 5:
    verdict = f"인트라데이 HMM 채택 권장 (+{delta:.1f}%p)"
elif delta > 0:
    verdict = f"소폭 개선 (+{delta:.1f}%p) - 효과 미미"
elif abs(delta) < 2:
    verdict = "차이 없음 - 일별 HMM으로 충분"
else:
    verdict = f"일별 HMM이 더 나음 ({delta:.1f}%p)"

print(f"판정: {verdict}")
print(f"  일별:     align={res_daily['align']:+.1f}%p  sharp={res_daily['sharpness']:.3f}")
print(f"  인트라:   align={res_intra['align']:+.1f}%p  sharp={res_intra['sharpness']:.3f}")
print("=" * 60)
