"""
Arena 검증 v7 - Q_tail을 regime_prob으로
==========================================
P_high (HMM) 대신 Q_tail (옵션 내재 꼬리 확률) 사용

Q_tail 계산:
  IV_lo = ATM_IV - 0.5 * RR_skew  (K_lo = S*0.85 방향)
  IV_hi = ATM_IV + 0.5 * RR_skew  (K_hi = S*1.15 방향)
  Q_down = N(-d2(K_lo, IV_lo))
  Q_up   = N( d2(K_hi, IV_hi))  ... d2 = (ln(S/K) - IV²T/2) / (IV√T)
  Q_tail = Q_down + Q_up

비교: P_high vs Q_tail vs blend(0.5*P + 0.5*Q)
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
from scipy.stats import norm, ks_2samp

THRESHOLD = 0.15

# ── 데이터 로드 ─────────────────────────────────────────
with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)
with open("option_iv_surface.pkl", 'rb') as f:
    iv_surf = pickle.load(f)

# ── Q_tail 시계열 ────────────────────────────────────────
def q_tail_from_smile(atm_iv, rr_skew, T_years):
    """ATM IV + RR skew → Q(±15% 꼬리) 디지털 공식"""
    if np.isnan(atm_iv) or atm_iv <= 0 or T_years <= 0:
        return np.nan, np.nan, np.nan
    # skew 보정: 없으면 대칭 가정
    rr = rr_skew if not np.isnan(rr_skew) else 0.0
    iv_lo = np.clip(atm_iv - 0.5 * abs(rr), 0.05, 5.0)  # 하방 (put skew 높음)
    iv_hi = np.clip(atm_iv + 0.5 * abs(rr), 0.05, 5.0)  # 상방 (call)
    # Q(S_T < K_lo): N(-d2), d2 = (ln(S/K_lo) - IV²T/2) / (IV√T)
    m_lo = np.log(1.0 / (1 - THRESHOLD))   # ln(S / K_lo) = ln(1/0.85) > 0
    d2_lo = (m_lo - 0.5 * iv_lo**2 * T_years) / (iv_lo * np.sqrt(T_years))
    q_down = norm.cdf(-d2_lo)
    # Q(S_T > K_hi): N(d2), d2 = (ln(S/K_hi) - IV²T/2) / (IV√T)
    m_hi = np.log(1.0 / (1 + THRESHOLD))   # ln(S / K_hi) = ln(1/1.15) < 0
    d2_hi = (m_hi - 0.5 * iv_hi**2 * T_years) / (iv_hi * np.sqrt(T_years))
    q_up = norm.cdf(d2_hi)
    return float(q_down + q_up), float(q_down), float(q_up)

# ATM IV ffill(최대 20일), RR skew는 있으면 사용
atm_iv  = iv_surf['atm_iv_front'].reindex(feat.index).ffill(limit=20)
rr_skew = iv_surf['rr_skew'].reindex(feat.index).ffill(limit=20)

# T = 30일 (1개월 ATM IV 기준)
T_FIXED = 30 / 365.0

q_tail_arr = []
for i in range(len(feat)):
    iv  = float(atm_iv.iloc[i]) if not pd.isna(atm_iv.iloc[i]) else np.nan
    rr  = float(rr_skew.iloc[i]) if not pd.isna(rr_skew.iloc[i]) else np.nan
    qt, _, _ = q_tail_from_smile(iv, rr, T_FIXED)
    q_tail_arr.append(qt)

q_tail_series = pd.Series(q_tail_arr, index=feat.index, name='q_tail')
q_valid = q_tail_series.notna().mean()
print(f"Q_tail 유효 비율: {q_valid*100:.1f}%")
print(f"Q_tail 평균: {q_tail_series.mean():.3f}  (범위 {q_tail_series.min():.3f}~{q_tail_series.max():.3f})")

# DVOL로 ATM IV 보완 (Q_tail NaN 구간)
dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)
dvol_iv = dvol['dvol'].reindex(feat.index).ffill() / 100.0
for i in range(len(feat)):
    if np.isnan(q_tail_arr[i]):
        iv = float(dvol_iv.iloc[i]) if not pd.isna(dvol_iv.iloc[i]) else np.nan
        qt, _, _ = q_tail_from_smile(iv, np.nan, T_FIXED)
        q_tail_arr[i] = qt

q_tail_series = pd.Series(q_tail_arr, index=feat.index, name='q_tail')
q_valid2 = q_tail_series.notna().mean()
print(f"DVOL 보완 후 유효: {q_valid2*100:.1f}%")
q_tail_series = q_tail_series.fillna(q_tail_series.median())

# ── state(128) ──────────────────────────────────────────
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
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
p_aligned       = high_p.reindex(dates_arr).fillna(0).values
q_aligned       = q_tail_series.reindex(dates_arr).fillna(q_tail_series.median()).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_p       = p_aligned[split_idx:]
val_q       = q_aligned[split_idx:]

print(f"\nVal Q_tail  평균={val_q.mean():.3f}  std={val_q.std():.3f}")
print(f"Val P_high  평균={val_p.mean():.3f}  std={val_p.std():.3f}")
print(f"P-Q 상관: {np.corrcoef(val_p, val_q)[0,1]:.3f}")

# ── 모델 로드 ──────────────────────────────────────────
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

# ── 경매 ──────────────────────────────────────────────
def run_auction(prob_arr, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)
            p = float(np.clip(prob_arr[i], 0.0, 1.0))
            bid_hv = conf_hv * (1.0 + ALPHA * p)
            bid_lv = conf_lv * (1.0 + ALPHA * (1.0 - p))
            vpi_hv.append(vpi_h); vpi_lv.append(vpi_l)
            winners.append(0 if bid_hv >= bid_lv else 1)

    winners = np.array(winners)
    hm = val_regimes == high_state
    lm = ~hm
    align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100 \
            if hm.sum() > 0 and lm.sum() > 0 else 0.0
    hv_high = (winners[hm] == 0).mean() * 100 if hm.sum() > 0 else 0
    hv_low  = (winners[lm] == 0).mean() * 100 if lm.sum() > 0 else 0
    vpi_hv  = np.array(vpi_hv); vpi_lv = np.array(vpi_lv)
    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)
    print(f"[{label}]  align={align:+.1f}%p  HV@high={hv_high:.1f}%  HV@low={hv_low:.1f}%"
          f"  sharp={sharpness:.3f}  KS p={ks_p:.4f}")
    return dict(label=label, align=align, hv_high=hv_high, hv_low=hv_low,
                sharpness=sharpness, ks_p=ks_p, winners=winners)

print("\n=== 경매 비교 (alpha=2.0) ===")
res_p     = run_auction(val_p,                         "P_high (HMM)")
res_q     = run_auction(val_q,                         "Q_tail (옵션)")
res_blend = run_auction(0.5 * val_p + 0.5 * val_q,    "Blend (P+Q)/2")
res_qw    = run_auction(0.3 * val_p + 0.7 * val_q,    "Q-heavy (0.3P+0.7Q)")

# ── 시각화 ────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax2 = ax.twinx()
ax.plot(val_dates, val_p, color='steelblue', lw=0.9, label='P_high (HMM)', alpha=0.8)
ax.plot(val_dates, val_q, color='darkorange', lw=0.9, label='Q_tail (옵션)', alpha=0.8)
ax2.fill_between(val_dates, (val_regimes == high_state).astype(float),
                 alpha=0.1, color='red', label='High regime')
ax.axhline(0.5, color='gray', lw=0.5, linestyle='--')
ax.set_title("P_high vs Q_tail 시계열")
ax.set_ylabel("Regime Probability"); ax.legend(fontsize=8, loc='upper left')
ax2.set_ylabel("High regime", color='red')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[1]
results_all = [res_p, res_q, res_blend, res_qw]
x = np.arange(len(results_all))
w = 0.25
bars_align = ax.bar(x, [r['align'] for r in results_all], w*3,
                    color=['steelblue','darkorange','purple','green'],
                    edgecolor='black', lw=0.5)
for bar, r in zip(bars_align, results_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{r['align']:+.1f}", ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([r['label'] for r in results_all], fontsize=9)
ax.axhline(0, color='gray', lw=0.7, linestyle='--')
ax.set_title("Alignment 비교 (alpha=2.0)")
ax.set_ylabel("Alignment (%p)")

ax = axes[2]
ax.bar(x - w, [r['hv_high'] for r in results_all], w,
       label='HV@High regime', color='salmon', edgecolor='black', lw=0.5)
ax.bar(x,     [r['hv_low']  for r in results_all], w,
       label='HV@Low regime',  color='steelblue', edgecolor='black', lw=0.5)
ax.axhline(50, color='gray', lw=0.6, linestyle='--', label='random')
ax.set_xticks(x); ax.set_xticklabels([r['label'] for r in results_all], fontsize=9)
ax.set_title("HV 낙찰률 by 레짐")
ax.set_ylabel("HV win rate (%)"); ax.legend(fontsize=8); ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig("arena_validation_v7.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v7.png")

print("\n" + "=" * 60)
print("Arena v7 - P vs Q regime_prob 비교")
print("=" * 60)
best = max(results_all, key=lambda r: r['align'])
print(f"최고 alignment: [{best['label']}]  {best['align']:+.1f}%p")
print(f"P_high 대비 변화: {best['align'] - res_p['align']:+.1f}%p")
for r in results_all:
    marker = " <--" if r['label'] == best['label'] else ""
    print(f"  {r['label']:22s}  align={r['align']:+.1f}%p  "
          f"HV@high={r['hv_high']:.1f}%  HV@low={r['hv_low']:.1f}%{marker}")
print("=" * 60)
