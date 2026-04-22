"""
Arena 검증 v4 — 분산 스왑 Reward
==================================
질문: VRP(변동성 위험 프리미엄) 기반 reward로 재학습하면
     에이전트 분기가 이론적으로 더 날카로워지는가?

Reward 교체:
  기존(v1): HighVol = max(0,d)*2 - max(0,-d),  LowVol = -abs(d)+0.1
  신규(v4): HighVol = RV_next - IV_curr         (long variance)
            LowVol  = IV_curr - RV_next         (short variance)

이론 근거: Carr-Madan (1998) 분산 스왑 복제
  IV = DVOL / 100  (Deribit 연환산 내재변동성)
  RV = btc_vol * sqrt(252)  (연환산 실현변동성)
  VRP = IV - RV  (위험 프리미엄, 평균 +22.5%)

비교 지표: alignment(%p), sharpness, KS p-value, V^pi 분포 + VRP 상관
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

assert os.path.exists("arena_val_features.pkl"), "arena_validation_v1.py 먼저 실행"
assert os.path.exists("arena_val_hmm.pkl"),      "arena_validation_v1.py 먼저 실행"
assert os.path.exists("dvol_btc.csv"),           "check_dvol.py 먼저 실행"

# ── 데이터 로드 ────────────────────────────────────────────
with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)

dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)

# ── VRP 계산 ───────────────────────────────────────────────
iv_ann  = dvol['dvol'] / 100.0                           # 연환산 내재변동성
rv_ann  = feat['btc_vol'] * np.sqrt(252)                 # 연환산 실현변동성
vrp_raw = iv_ann.reindex(feat.index) - rv_ann            # VRP = IV - RV

# DVOL 없는 구간(2018~2021-03-23)은 VRP=0 처리 (reward 중립)
vrp_series = vrp_raw.fillna(0.0)

from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN        = 16
STATE_DIM  = 128
ACTION_DIM = 3
LR         = 1e-3
N_STEPS    = 3000
GAMMA      = 0.99
SEED       = 42
TRAIN_END  = "2022-06-30"

torch.manual_seed(SEED)
np.random.seed(SEED)

# state(128) 재구성
states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
highp_aligned   = high_p.reindex(dates_arr).fillna(0).values
vrp_aligned     = vrp_series.reindex(dates_arr).fillna(0).values   # VRP 정렬

# iv, rv 별도 저장 (시각화용)
iv_aligned  = iv_ann.reindex(dates_arr).fillna(method='ffill').fillna(0.7).values
rv_aligned  = rv_ann.reindex(dates_arr).fillna(0).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
train_data  = states_arr[:split_idx]
train_vrp   = vrp_aligned[:split_idx]

val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_highp   = highp_aligned[split_idx:]
val_vrp     = vrp_aligned[split_idx:]
val_iv      = iv_aligned[split_idx:]
val_rv      = rv_aligned[split_idx:]

print(f"Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()}  ({split_idx})")
print(f"Val:   {val_dates[0].date()} ~ {val_dates[-1].date()}  ({len(val_data)})")
print(f"VRP 학습 구간 (DVOL 유효): "
      f"{(train_vrp != 0).sum()}/{split_idx} 일 ({(train_vrp != 0).mean()*100:.0f}%)")


# ══════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════
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


# ── VRP 기반 reward ────────────────────────────────────────
def vrp_reward(idx, bias_type, data_iv, data_rv):
    """분산 스왑 1기간 P&L:
       HighVol(long var):  RV_next - IV_curr
       LowVol (short var): IV_curr - RV_next
    """
    if idx + 1 >= len(data_iv):
        return 0.0
    iv_curr = float(data_iv[idx])
    rv_next = float(data_rv[idx + 1])
    pnl = rv_next - iv_curr
    if bias_type == "HighVol":
        return float(np.clip(pnl, -1.0, 1.0))
    else:
        return float(np.clip(-pnl, -1.0, 1.0))


def train_agent_vrp(data, bias_type, iv_data, rv_data):
    net = PolicyValueNet(STATE_DIM, ACTION_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)

    for step in range(N_STEPS):
        idx    = np.random.randint(0, n - 2)
        s      = torch.FloatTensor(data[idx])
        s1     = torch.FloatTensor(data[idx + 1])
        logits, _, vpi = net(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        lp     = dist.log_prob(action)
        r      = vrp_reward(idx, bias_type, iv_data, rv_data)
        with torch.no_grad():
            _, _, v1 = net(s1)
        td   = r + GAMMA * v1
        adv  = td - vpi
        fv   = net.trunk(s)
        vt   = net.value_head(fv).squeeze(-1)
        vl   = F.mse_loss(vt, torch.tensor(td))
        ent  = -torch.sum(probs * torch.log(probs + 1e-9))
        loss = -lp * adv + 0.5 * vl - 0.01 * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


# ── v1 기존 reward ─────────────────────────────────────────
def v1_reward(s, ns, bias_type):
    curr_vol = float(s[-8 + 2])
    next_vol = float(ns[-8 + 2])
    d = next_vol - curr_vol
    if bias_type == "HighVol":
        return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)


def train_agent_v1(data, bias_type):
    net = PolicyValueNet(STATE_DIM, ACTION_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for step in range(N_STEPS):
        idx    = np.random.randint(0, n - 1)
        s      = torch.FloatTensor(data[idx])
        s1     = torch.FloatTensor(data[idx + 1])
        logits, _, vpi = net(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        lp     = dist.log_prob(action)
        r      = v1_reward(data[idx], data[idx + 1], bias_type)
        with torch.no_grad():
            _, _, v1 = net(s1)
        td   = r + GAMMA * v1
        adv  = td - vpi
        fv   = net.trunk(s)
        vt   = net.value_head(fv).squeeze(-1)
        vl   = F.mse_loss(vt, torch.tensor(td))
        ent  = -torch.sum(probs * torch.log(probs + 1e-9))
        loss = -lp * adv + 0.5 * vl - 0.01 * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


# ── 평가 함수 (alpha=2.0 기본) ─────────────────────────────
ALPHA = 2.0

def evaluate(net_hv, net_lv, label=""):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)

            p_high = float(val_highp[i])
            bid_hv = conf_hv * (1.0 + ALPHA * p_high)
            bid_lv = conf_lv * (1.0 + ALPHA * (1.0 - p_high))

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

    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std

    _, ks_p = ks_2samp(vpi_hv, vpi_lv)

    vpi_diff = vpi_hv - vpi_lv
    corr_vrp, corr_p = pearsonr(vpi_diff, val_vrp) if val_vrp.std() > 1e-6 else (0.0, 1.0)

    print(f"\n[{label}]")
    print(f"  alignment:  {align:+.1f}%p")
    print(f"  sharpness:  {sharpness:.3f}")
    print(f"  KS p-value: {ks_p:.4f}")
    print(f"  VRP corr:   {corr_vrp:+.3f}  (p={corr_p:.4f})")

    return {
        "label": label, "align": align, "sharpness": sharpness,
        "ks_p": ks_p, "vrp_corr": corr_vrp, "vrp_corr_p": corr_p,
        "vpi_hv": vpi_hv, "vpi_lv": vpi_lv, "winners": winners,
    }


# ══════════════════════════════════════════════════════════
# 학습 및 비교
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("v1 학습 중 (기존 OHLCV reward)...")
print("=" * 55)
net_hv_v1 = train_agent_v1(train_data, "HighVol")
net_lv_v1 = train_agent_v1(train_data, "LowVol")
res_v1    = evaluate(net_hv_v1, net_lv_v1, "v1 OHLCV reward")

# train iv/rv 배열
train_iv = iv_aligned[:split_idx]
train_rv = rv_aligned[:split_idx]

print("\n" + "=" * 55)
print("v4 학습 중 (VRP / 분산 스왑 reward)...")
print("=" * 55)
net_hv_v4 = train_agent_vrp(train_data, "HighVol", train_iv, train_rv)
net_lv_v4 = train_agent_vrp(train_data, "LowVol",  train_iv, train_rv)
res_v4    = evaluate(net_hv_v4, net_lv_v4, "v4 VRP reward")


# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# ── 비교 막대 ─────────────────────────────────────────────
metrics = ["align", "sharpness", "vrp_corr"]
titles  = ["Alignment (%p)", "Sharpness (Cohen d)", "V^π diff vs VRP corr"]
for col, (m, t) in enumerate(zip(metrics, titles)):
    ax = axes[0, col] if col < 2 else axes[0, 1]
    break

ax = axes[0, 0]
vals = [res_v1["align"], res_v4["align"]]
bars = ax.bar(["v1 (OHLCV)", "v4 (VRP)"], vals,
              color=['steelblue', 'darkorange'], edgecolor='black', lw=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:+.1f}%p', ha='center', fontsize=10, fontweight='bold')
ax.set_title("Alignment 비교 (alpha=2.0)")
ax.set_ylabel("Alignment (%p)")
ax.axhline(0, color='gray', lw=0.7, linestyle='--')

ax = axes[0, 1]
vals = [res_v1["sharpness"], res_v4["sharpness"]]
bars = ax.bar(["v1 (OHLCV)", "v4 (VRP)"], vals,
              color=['steelblue', 'darkorange'], edgecolor='black', lw=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title("V^π 분기 선명도 비교")
ax.set_ylabel("Sharpness (Cohen d)")

# ── V^pi 분포 (v1 vs v4) ──────────────────────────────────
ax = axes[1, 0]
bins = np.linspace(-3, 3, 60)
ax.hist(res_v1["vpi_hv"], bins=bins, alpha=0.5, color='red',
        label='v1 HighVol V^π', density=True)
ax.hist(res_v1["vpi_lv"], bins=bins, alpha=0.5, color='blue',
        label='v1 LowVol V^π', density=True)
ax.set_title(f"v1 V^π 분포  (KS p={res_v1['ks_p']:.4f})")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.hist(res_v4["vpi_hv"], bins=bins, alpha=0.5, color='red',
        label='v4 HighVol V^π', density=True)
ax.hist(res_v4["vpi_lv"], bins=bins, alpha=0.5, color='blue',
        label='v4 LowVol V^π', density=True)
ax.set_title(f"v4 V^π 분포  (KS p={res_v4['ks_p']:.4f})")
ax.legend(fontsize=8)

# ── V^pi 차이 vs VRP 시계열 ───────────────────────────────
ax = axes[2, 0]
vpi_diff_v1 = res_v1["vpi_hv"] - res_v1["vpi_lv"]
vpi_diff_v4 = res_v4["vpi_hv"] - res_v4["vpi_lv"]

ax2 = ax.twinx()
ax2.fill_between(val_dates, val_vrp, alpha=0.2, color='green', label='VRP (IV-RV)')
ax2.set_ylabel("VRP (annualized)", color='green')

ax.plot(val_dates, pd.Series(vpi_diff_v1, index=val_dates).rolling(10).mean(),
        color='steelblue', lw=0.9, label=f'v1 V^π diff (r={res_v1["vrp_corr"]:+.3f})')
ax.plot(val_dates, pd.Series(vpi_diff_v4, index=val_dates).rolling(10).mean(),
        color='darkorange', lw=0.9, label=f'v4 V^π diff (r={res_v4["vrp_corr"]:+.3f})')
ax.set_title("V^π 차이(HV-LV) vs VRP")
ax.set_ylabel("V^π HV - LV (rolling 10)")
ax.legend(fontsize=7, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# ── IV vs RV 시계열 ───────────────────────────────────────
ax = axes[2, 1]
ax.plot(val_dates, val_iv * 100, color='red',  lw=1.0, label='IV (DVOL, %)')
ax.plot(val_dates, val_rv * 100, color='blue', lw=1.0, label='RV (realized, %)')
ax.fill_between(val_dates, val_iv * 100, val_rv * 100,
                where=val_iv > val_rv, alpha=0.2, color='green', label='VRP > 0')
ax.fill_between(val_dates, val_iv * 100, val_rv * 100,
                where=val_iv < val_rv, alpha=0.3, color='red', label='VRP < 0 (crash)')
ax.set_title("IV vs RV (검증 구간)")
ax.set_ylabel("Annualized Vol (%)")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig("arena_validation_v4.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v4.png")


# ══════════════════════════════════════════════════════════
# 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Arena 검증 v4 -- VRP Reward 비교 결과")
print("=" * 60)
print(f"{'지표':20s} {'v1 (OHLCV)':>14} {'v4 (VRP)':>14} {'변화':>10}")
print("-" * 60)
for key, name in [("align","Alignment(%p)"), ("sharpness","Sharpness"),
                  ("ks_p","KS p-value"), ("vrp_corr","VRP 상관")]:
    v1v = res_v1[key]; v4v = res_v4[key]
    delta = v4v - v1v
    print(f"  {name:18s} {v1v:>14.3f} {v4v:>14.3f} {delta:>+10.3f}")

print()
better = sum([
    res_v4["align"]    > res_v1["align"],
    res_v4["sharpness"]> res_v1["sharpness"],
    res_v4["ks_p"]     < res_v1["ks_p"],
    abs(res_v4["vrp_corr"]) > abs(res_v1["vrp_corr"]),
])

if better >= 3:
    verdict = "VRP reward 채택 권장 - 이론적 기반 + 실증 모두 개선"
elif better == 2:
    verdict = "VRP reward 부분 개선 - 지표별 트레이드오프 확인 필요"
else:
    verdict = "VRP reward 효과 없음 - OHLCV reward가 더 실용적"

print(f"판정: {verdict}")
print(f"      ({better}/4 지표 개선)")
print("=" * 60)
