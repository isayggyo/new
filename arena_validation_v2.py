"""
Arena 검증 v2 — 재귀적 학습 (Walk-Forward)
============================================
질문: 에이전트가 더 많은 데이터를 볼수록 실제로 나아지는가?

설계:
  - Walk-forward: 6개월 단위로 학습 데이터 확장
  - 각 윈도우에서 평가: 레짐-경매 정렬도, V^pi 분기 선명도
  - 개선 여부: 초기 vs 후기 성능 비교

지표:
  - alignment(%p):  레짐에 맞는 에이전트 낙찰 비율 차이
  - sharpness:      |mean(V^pi_HV) - mean(V^pi_LV)| / pooled_std  (분포 분리도)
  - stability:      rolling V^pi 차이의 std (낮을수록 일관됨)
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

# ── v1 결과 재사용 ────────────────────────────────────────
FEAT_CACHE = "arena_val_features.pkl"
HMM_CACHE  = "arena_val_hmm.pkl"

assert os.path.exists(FEAT_CACHE), "arena_validation_v1.py 먼저 실행"
assert os.path.exists(HMM_CACHE),  "arena_validation_v1.py 먼저 실행"

with open(FEAT_CACHE, 'rb') as f:
    feat, btc_price = pickle.load(f)
with open(HMM_CACHE, 'rb') as f:
    regime, high_p, high_state = pickle.load(f)

from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN       = 16
STATE_DIM = 128
ACTION_DIM = 3
HIDDEN1   = 64
HIDDEN2   = 32
LR        = 1e-3
N_STEPS   = 3000
GAMMA     = 0.99
SEED      = 42

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


# ══════════════════════════════════════════════════════════
# 모델 + 학습 함수 (v1과 동일)
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


def real_reward(s, action, ns, bias_type):
    curr_vol = float(s[-8 + 2])
    next_vol = float(ns[-8 + 2])
    d = next_vol - curr_vol
    if bias_type == "HighVol":
        return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)


def train_agent_on(data, bias_type, net=None):
    """data: np array (N, 128). 기존 net에서 이어 학습 가능."""
    if net is None:
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
        r      = real_reward(data[idx], action.item(), data[idx + 1], bias_type)
        with torch.no_grad():
            _, _, v1 = net(s1)
        td     = r + GAMMA * v1
        adv    = td - vpi
        fv     = net.trunk(s)
        vt     = net.value_head(fv).squeeze(-1)
        vl     = F.mse_loss(vt, torch.tensor(td))
        ent    = -torch.sum(probs * torch.log(probs + 1e-9))
        loss   = -lp * adv + 0.5 * vl - 0.01 * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


def evaluate(net_hv, net_lv, eval_data, eval_regimes):
    """레짐-경매 정렬도 + V^pi 분기 선명도 반환"""
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for s_np in eval_data:
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)
            vpi_hv.append(vpi_h); vpi_lv.append(vpi_l)
            winners.append(0 if conf_hv >= conf_lv else 1)

    vpi_hv  = np.array(vpi_hv)
    vpi_lv  = np.array(vpi_lv)
    winners = np.array(winners)

    # 정렬도
    hm = eval_regimes == high_state
    lm = ~hm
    align = 0.0
    if hm.sum() > 0 and lm.sum() > 0:
        align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100

    # 분기 선명도: Cohen's d 스타일
    diff_mean = abs(vpi_hv.mean() - vpi_lv.mean())
    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = diff_mean / pool_std

    # KS p-value
    _, ks_p = ks_2samp(vpi_hv, vpi_lv)

    return {"align": align, "sharpness": sharpness, "ks_p": ks_p,
            "vpi_hv_mean": vpi_hv.mean(), "vpi_lv_mean": vpi_lv.mean()}


# ══════════════════════════════════════════════════════════
# Walk-Forward 실험
# ══════════════════════════════════════════════════════════
# 6개월 단위 윈도우 구성
WINDOW_MONTHS = 6
STEP_MONTHS   = 3

# 시작점: 최소 1년 학습 데이터 필요
start_train = dates_arr[0]
first_eval  = start_train + pd.DateOffset(months=12)

windows = []
cur = first_eval
while cur + pd.DateOffset(months=STEP_MONTHS) <= dates_arr[-1]:
    eval_end = cur + pd.DateOffset(months=STEP_MONTHS)
    windows.append((start_train, cur, cur, eval_end))
    cur = eval_end

print(f"Walk-Forward 윈도우: {len(windows)}개")
print(f"  첫 윈도우: train ~ {windows[0][1].date()}, eval {windows[0][2].date()} ~ {windows[0][3].date()}")
print(f"  마지막:    train ~ {windows[-1][1].date()}, eval {windows[-1][2].date()} ~ {windows[-1][3].date()}")

WF_CACHE = "arena_wf_results.pkl"

if os.path.exists(WF_CACHE):
    print(f"\nWalk-Forward 캐시 로드: {WF_CACHE}")
    with open(WF_CACHE, 'rb') as f:
        results = pickle.load(f)
else:
    results   = []
    net_hv    = None
    net_lv    = None

    for i, (tr_start, tr_end, ev_start, ev_end) in enumerate(windows):
        tr_mask   = (dates_arr >= tr_start) & (dates_arr < tr_end)
        ev_mask   = (dates_arr >= ev_start) & (dates_arr < ev_end)

        tr_data   = states_arr[tr_mask]
        ev_data   = states_arr[ev_mask]
        ev_reg    = regimes_aligned[ev_mask]

        if len(tr_data) < 100 or len(ev_data) < 10:
            continue

        print(f"\n[{i+1}/{len(windows)}] "
              f"Train ~ {tr_end.date()}  ({len(tr_data)}),  "
              f"Eval {ev_start.date()} ~ {ev_end.date()}  ({len(ev_data)})")

        # 누적 학습 (기존 net에서 이어서)
        net_hv = train_agent_on(tr_data, "HighVol", net_hv)
        net_lv = train_agent_on(tr_data, "LowVol",  net_lv)

        metrics = evaluate(net_hv, net_lv, ev_data, ev_reg)
        metrics["eval_start"] = ev_start
        metrics["eval_end"]   = ev_end
        metrics["n_train"]    = len(tr_data)
        results.append(metrics)

        print(f"  align={metrics['align']:+.1f}%p  "
              f"sharpness={metrics['sharpness']:.3f}  "
              f"ks_p={metrics['ks_p']:.4f}")

    with open(WF_CACHE, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nWalk-Forward 결과 저장: {WF_CACHE}")

# ══════════════════════════════════════════════════════════
# 결과 분석
# ══════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
df_res["mid_date"] = df_res["eval_start"] + (df_res["eval_end"] - df_res["eval_start"]) / 2

print(f"\n{'기간':20s} {'n_train':>8} {'align':>8} {'sharp':>8} {'ks_p':>8}")
print("-" * 60)
for _, row in df_res.iterrows():
    print(f"{str(row['eval_start'].date()):20s} "
          f"{row['n_train']:8.0f} "
          f"{row['align']:+8.1f}%p "
          f"{row['sharpness']:8.3f} "
          f"{row['ks_p']:8.4f}")

# 선형 추세
from numpy.polynomial import polynomial as P
x = np.arange(len(df_res))

align_trend     = np.polyfit(x, df_res["align"].values, 1)[0]
sharpness_trend = np.polyfit(x, df_res["sharpness"].values, 1)[0]

print(f"\n추세 (윈도우당):")
print(f"  정렬도:    {align_trend:+.3f}%p/window "
      f"({'개선' if align_trend > 0 else '악화'})")
print(f"  분기선명도: {sharpness_trend:+.4f}/window "
      f"({'개선' if sharpness_trend > 0 else '악화'})")

# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 10))

x_dates = df_res["mid_date"].values

ax = axes[0]
ax.plot(x_dates, df_res["align"].values,
        marker='o', ms=5, color='steelblue', lw=1.2, label='Alignment (%p)')
trend_line = np.polyval(np.polyfit(x, df_res["align"].values, 1), x)
ax.plot(x_dates, trend_line, '--', color='steelblue', alpha=0.5, label='trend')
ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.set_title(f"레짐-경매 정렬도 추이 (trend={align_trend:+.2f}%p/step)")
ax.set_ylabel("Alignment (%p)"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[1]
ax.plot(x_dates, df_res["sharpness"].values,
        marker='o', ms=5, color='darkorange', lw=1.2, label='Sharpness (Cohen d)')
trend_line2 = np.polyval(np.polyfit(x, df_res["sharpness"].values, 1), x)
ax.plot(x_dates, trend_line2, '--', color='darkorange', alpha=0.5, label='trend')
ax.set_title(f"V^pi 분기 선명도 추이 (trend={sharpness_trend:+.4f}/step)")
ax.set_ylabel("Sharpness"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[2]
ax.plot(x_dates, df_res["vpi_hv_mean"].values,
        marker='o', ms=4, color='red', lw=1.0, label='HighVol V^pi mean')
ax.plot(x_dates, df_res["vpi_lv_mean"].values,
        marker='o', ms=4, color='steelblue', lw=1.0, label='LowVol V^pi mean')
ax.fill_between(x_dates,
                df_res["vpi_hv_mean"].values,
                df_res["vpi_lv_mean"].values,
                alpha=0.15, color='purple', label='divergence gap')
ax.set_title("V^pi 절대 수준 추이")
ax.set_ylabel("V^pi mean"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig("arena_validation_v2.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v2.png")

# ══════════════════════════════════════════════════════════
# 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Arena 검증 v2 -- 재귀 학습 판정")
print("=" * 60)

improving_align  = align_trend > 0
improving_sharp  = sharpness_trend > 0
final_align_ok   = df_res["align"].iloc[-3:].mean() > 20
final_sharp_ok   = df_res["sharpness"].iloc[-3:].mean() > 5

print(f"정렬도 개선 추세:    {'YES' if improving_align  else 'NO '} ({align_trend:+.3f}%p/step)")
print(f"선명도 개선 추세:    {'YES' if improving_sharp  else 'NO '} ({sharpness_trend:+.4f}/step)")
print(f"최근 3구간 정렬도:   {'OK ' if final_align_ok   else 'LOW'} ({df_res['align'].iloc[-3:].mean():.1f}%p)")
print(f"최근 3구간 선명도:   {'OK ' if final_sharp_ok   else 'LOW'} ({df_res['sharpness'].iloc[-3:].mean():.3f})")

n = sum([improving_align, improving_sharp, final_align_ok, final_sharp_ok])
if n >= 3:
    verdict = "VALID -- 재귀 학습으로 성능 개선 확인"
elif n == 2:
    verdict = "PARTIAL -- 일부 지표 개선, 추가 학습 필요"
else:
    verdict = "NULL -- 재귀 학습 효과 없음"

print(f"\n판정: {verdict}")
print("=" * 60)
