"""
Arena 검증 v1 — 에이전트 학습 동역학
======================================
검증 대상: machine.py의 핵심 학습 메커니즘

질문 1: HighVol vs LowVol V^pi 분포가 실제로 갈리는가?
질문 2: 고변동성 국면에서 HighVol이 경매를 더 자주 이기는가?
질문 3: 학습이 수렴하는가? (loss 감소, V^pi 안정화)

설계:
  - state(128): 16일 × 8 실제 피처 flatten (torch.randn 대체)
  - reward_fn:  BTC 실현변동성 기반 (L2 norm 대체)
  - 레짐 Ground Truth: HMM 2-state (검증 완료)
  - 검증 지표: V^pi KL divergence, 경매 승자-레짐 정렬도, 학습 곡선
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# ── 설정 ─────────────────────────────────────────────────
START      = "2017-01-01"
END        = "2023-12-31"
WIN        = 16          # 16일 × 8 피처 = 128
STATE_DIM  = 128
ACTION_DIM = 3           # HIGH / MEDIUM / LOW vol 예측
HIDDEN1    = 64
HIDDEN2    = 32
TRAIN_END  = "2022-06-30"  # train/val split
LR         = 1e-3
N_STEPS    = 5000
GAMMA      = 0.99
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

FEAT_CACHE = "arena_val_features.pkl"
HMM_CACHE  = "arena_val_hmm.pkl"


# ══════════════════════════════════════════════════════════
# 1. 피처 엔지니어링
# ══════════════════════════════════════════════════════════
def build_features():
    print("데이터 로딩...")
    frames = {}
    for asset in ["BTC-USD", "ETH-USD"]:
        raw = yf.download(asset, start=START, end=END,
                          auto_adjust=True, progress=False)
        c = raw["Close"]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        frames[asset] = c

    prices  = pd.DataFrame(frames).dropna()
    ret_btc = np.log(prices["BTC-USD"] / prices["BTC-USD"].shift(1))
    ret_eth = np.log(prices["ETH-USD"] / prices["ETH-USD"].shift(1))
    btc_vol      = ret_btc.rolling(20).std()
    btc_vol_slow = ret_btc.rolling(60).std()

    feat = pd.DataFrame({
        "btc_ret":      ret_btc,
        "eth_ret":      ret_eth,
        "btc_vol":      btc_vol,
        "btc_mom5":     prices["BTC-USD"].pct_change(5),
        "btc_mom20":    prices["BTC-USD"].pct_change(20),
        "vol_ratio":    btc_vol / (btc_vol_slow + 1e-9),
        "btc_eth_corr": ret_btc.rolling(20).corr(ret_eth),
        "vol_accel":    btc_vol / (btc_vol.shift(5) + 1e-9) - 1,
    }).dropna()

    print(f"  {len(feat)}일, {feat.index.min().date()} ~ {feat.index.max().date()}")
    return feat, prices["BTC-USD"].reindex(feat.index)


if os.path.exists(FEAT_CACHE):
    with open(FEAT_CACHE, 'rb') as f:
        feat, btc_price = pickle.load(f)
    print(f"피처 캐시 로드: {len(feat)}일")
else:
    feat, btc_price = build_features()
    with open(FEAT_CACHE, 'wb') as f:
        pickle.dump((feat, btc_price), f)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)   # (N, 8)


# ══════════════════════════════════════════════════════════
# 2. HMM 레짐 (Ground Truth)
# ══════════════════════════════════════════════════════════
def fit_hmm(X_sc, feat_df):
    hmm_feat = pd.DataFrame({
        "ret": feat_df["btc_ret"].values,
        "vol": feat_df["btc_vol"].values,
    })
    hmm_sc = StandardScaler().fit_transform(hmm_feat.values)

    model = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=200, random_state=42)
    model.fit(hmm_sc)
    states = model.predict(hmm_sc)
    probs  = model.predict_proba(hmm_sc)

    vol_by_state = {s: feat_df["btc_vol"].values[states == s].mean()
                    for s in [0, 1]}
    high_state = max(vol_by_state, key=vol_by_state.get)

    regime = pd.Series(states, index=feat_df.index)
    high_p = pd.Series(probs[:, high_state], index=feat_df.index)
    print(f"HMM: state{high_state}=HighVol(mean={vol_by_state[high_state]:.4f}), "
          f"state{1-high_state}=LowVol(mean={vol_by_state[1-high_state]:.4f})")
    return regime, high_p, high_state


if os.path.exists(HMM_CACHE):
    with open(HMM_CACHE, 'rb') as f:
        regime, high_p, high_state = pickle.load(f)
    print(f"HMM 캐시 로드")
else:
    print("HMM 학습...")
    regime, high_p, high_state = fit_hmm(X_scaled, feat)
    with open(HMM_CACHE, 'wb') as f:
        pickle.dump((regime, high_p, high_state), f)


# ══════════════════════════════════════════════════════════
# 3. 슬라이딩 윈도우 state(128) 생성
# ══════════════════════════════════════════════════════════
states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())   # (128,)
    dates_arr.append(feat.index[i])

states_arr = np.array(states_arr, dtype=np.float32)    # (N, 128)
dates_arr  = pd.DatetimeIndex(dates_arr)

# vol label: HIGH(0) / MEDIUM(1) / LOW(2) 기반 action reward
vol_vals  = feat["btc_vol"].values[WIN:]
v_high    = np.percentile(vol_vals, 80)
v_low     = np.percentile(vol_vals, 40)
vol_class = np.where(vol_vals >= v_high, 0,
            np.where(vol_vals <= v_low,  2, 1))         # 0=HIGH,1=MID,2=LOW

regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)

# Train / Validation split
split_idx = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
print(f"\nstate(128) windows: {len(states_arr)}")
print(f"  Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()} ({split_idx})")
print(f"  Val:   {dates_arr[split_idx].date()} ~ {dates_arr[-1].date()} ({len(states_arr)-split_idx})")


# ══════════════════════════════════════════════════════════
# 4. 모델 (machine.py와 동일 아키텍처)
# ══════════════════════════════════════════════════════════
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim=128, action_dim=3,
                 hidden1=64, hidden2=32):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden2, action_dim)
        self.value_head  = nn.Linear(hidden2, 1)

    def forward(self, state):
        feat    = self.trunk(state)
        logits  = self.policy_head(feat)
        probs   = F.softmax(logits, dim=-1)
        H       = -torch.sum(probs * torch.log(probs + 1e-9))
        H_max   = torch.log(torch.tensor(float(self.action_dim)))
        conf    = (1.0 - H / H_max).item()
        v_pi    = self.value_head(feat).squeeze(-1).item()
        return logits, conf, v_pi


# ══════════════════════════════════════════════════════════
# 5. 실제 보상 함수 (BTC 변동성 기반)
# ══════════════════════════════════════════════════════════
def real_reward(state_vec: np.ndarray, action: int,
                next_state_vec: np.ndarray, bias_type: str) -> float:
    """
    state_vec: (128,) = flatten(16일 × 8피처)
    마지막 8개 값 = 현재 일의 피처 (스케일됨)
    피처 순서: btc_ret(0), eth_ret(1), btc_vol(2), ...

    HighVol: 변동성 상승 국면에 보상
    LowVol:  변동성 안정 국면에 보상
    """
    curr_vol  = float(state_vec[-8 + 2])   # btc_vol (scaled)
    next_vol  = float(next_state_vec[-8 + 2])
    delta_vol = next_vol - curr_vol

    if bias_type == "HighVol":
        # 큰 변동성 증가에 강한 보상, 감소엔 패널티
        return float(max(0, delta_vol) * 2.0 - max(0, -delta_vol))
    else:  # LowVol
        # 안정적 저변동성에 보상
        return float(-abs(delta_vol) + 0.1)


# ══════════════════════════════════════════════════════════
# 6. A2C 학습
# ══════════════════════════════════════════════════════════
def train_agent(bias_type: str, tag: str, save_path: str):
    net = PolicyValueNet(STATE_DIM, ACTION_DIM).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    train_data = states_arr[:split_idx]
    n = len(train_data)

    vpi_history    = []
    reward_history = []
    loss_history   = []

    print(f"\n{tag} 학습 ({N_STEPS} steps)...")
    for step in range(N_STEPS):
        idx    = np.random.randint(0, n - 1)
        s_np   = train_data[idx]
        s_np1  = train_data[idx + 1]

        s  = torch.FloatTensor(s_np)
        s1 = torch.FloatTensor(s_np1)

        logits, conf, v_pi = net(s)
        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        r = real_reward(s_np, action.item(), s_np1, bias_type)

        with torch.no_grad():
            _, _, v_next = net(s1)
        td_target = r + GAMMA * v_next
        advantage = td_target - v_pi

        feat_vec   = net.trunk(s)
        v_tensor   = net.value_head(feat_vec).squeeze(-1)
        value_loss = F.mse_loss(v_tensor, torch.tensor(td_target))

        entropy     = -torch.sum(probs * torch.log(probs + 1e-9))
        policy_loss = -log_prob * advantage
        loss        = policy_loss + 0.5 * value_loss - 0.01 * entropy

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        vpi_history.append(v_pi)
        reward_history.append(r)
        loss_history.append(loss.item())

        if (step + 1) % 1000 == 0:
            avg_vpi = np.mean(vpi_history[-500:])
            avg_r   = np.mean(reward_history[-500:])
            print(f"  step={step+1:5d} | V^pi={avg_vpi:.3f} | "
                  f"reward={avg_r:.3f} | loss={np.mean(loss_history[-500:]):.4f}")

    torch.save(net.state_dict(), save_path)
    return net, vpi_history, reward_history, loss_history


HV_PATH = "arena_val_highvol.pt"
LV_PATH = "arena_val_lowvol.pt"

net_hv = PolicyValueNet(STATE_DIM, ACTION_DIM)
net_lv = PolicyValueNet(STATE_DIM, ACTION_DIM)

if os.path.exists(HV_PATH) and os.path.exists(LV_PATH):
    net_hv.load_state_dict(torch.load(HV_PATH, map_location='cpu'))
    net_lv.load_state_dict(torch.load(LV_PATH, map_location='cpu'))
    print("에이전트 가중치 로드")
    vpi_hv = vpi_lv = reward_hv = reward_lv = loss_hv = loss_lv = None
else:
    net_hv, vpi_hv, reward_hv, loss_hv = train_agent("HighVol", "[HighVol]", HV_PATH)
    net_lv, vpi_lv, reward_lv, loss_lv = train_agent("LowVol",  "[LowVol]",  LV_PATH)


# ══════════════════════════════════════════════════════════
# 7. 검증 — Validation Set 시뮬레이션
# ══════════════════════════════════════════════════════════
print("\n검증 시뮬레이션...")

val_data     = states_arr[split_idx:]
val_dates    = dates_arr[split_idx:]
val_regimes  = regimes_aligned[split_idx:]
val_highprob = high_p.reindex(val_dates).fillna(0).values

vpi_hv_val, vpi_lv_val = [], []
conf_hv_val, conf_lv_val = [], []
winners = []

net_hv.eval(); net_lv.eval()
with torch.no_grad():
    for i, s_np in enumerate(val_data):
        s = torch.FloatTensor(s_np)

        logits_hv, conf_hv, vpi_hv_i = net_hv(s)
        logits_lv, conf_lv, vpi_lv_i = net_lv(s)

        vpi_hv_val.append(vpi_hv_i)
        vpi_lv_val.append(vpi_lv_i)
        conf_hv_val.append(conf_hv)
        conf_lv_val.append(conf_lv)

        # VCG 경매: confidence 높은 쪽이 낙찰
        winners.append(0 if conf_hv >= conf_lv else 1)  # 0=HighVol, 1=LowVol

vpi_hv_val = np.array(vpi_hv_val)
vpi_lv_val = np.array(vpi_lv_val)
winners    = np.array(winners)


# ══════════════════════════════════════════════════════════
# 8. 검증 지표 계산
# ══════════════════════════════════════════════════════════

# ── 질문 1: V^pi 분포 갈리는가? (KS test) ────────────────
ks_stat, ks_p = ks_2samp(vpi_hv_val, vpi_lv_val)
print(f"\n[질문 1] V^pi 분포 갈림 (KS test)")
print(f"  HighVol V^pi: mean={vpi_hv_val.mean():.3f} std={vpi_hv_val.std():.3f}")
print(f"  LowVol  V^pi: mean={vpi_lv_val.mean():.3f} std={vpi_lv_val.std():.3f}")
print(f"  KS stat={ks_stat:.4f}  p={ks_p:.4f}  "
      f"{'[DIVERGED]' if ks_p < 0.05 else '[SAME]'}")

# ── 질문 2: 레짐-경매 정렬도 ─────────────────────────────
# high_regime=1 일 때 HighVol(0)이 이기면 aligned
high_reg_mask = val_regimes == high_state
low_reg_mask  = ~high_reg_mask

hv_wins_highvol_regime = (winners[high_reg_mask] == 0).mean() * 100
hv_wins_lowvol_regime  = (winners[low_reg_mask]  == 0).mean() * 100

print(f"\n[질문 2] 레짐-경매 정렬도")
print(f"  High vol 레짐에서 HighVol 낙찰: {hv_wins_highvol_regime:.1f}%")
print(f"  Low  vol 레짐에서 HighVol 낙찰: {hv_wins_lowvol_regime:.1f}%")
alignment = hv_wins_highvol_regime - hv_wins_lowvol_regime
print(f"  정렬 차이: {alignment:+.1f}%p  "
      f"{'[ALIGNED]' if alignment > 5 else '[RANDOM]'}")

# ── 질문 3: 학습 수렴 (val set V^pi 안정성) ──────────────
vpi_diff = vpi_hv_val - vpi_lv_val
rolling_diff = pd.Series(vpi_diff).rolling(50).mean()
drift = abs(rolling_diff.iloc[-50:].mean() - rolling_diff.iloc[:50].mean())
print(f"\n[질문 3] V^pi 차이 안정성")
print(f"  초반 50스텝 V^pi 차이 평균: {rolling_diff.iloc[:50].mean():.3f}")
print(f"  후반 50스텝 V^pi 차이 평균: {rolling_diff.iloc[-50:].mean():.3f}")
print(f"  드리프트: {drift:.4f}  {'[STABLE]' if drift < 0.5 else '[DRIFTING]'}")


# ══════════════════════════════════════════════════════════
# 9. 시각화
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# V^pi 시계열
ax = axes[0]
ax.plot(val_dates, vpi_hv_val, lw=0.7, color='red',
        alpha=0.8, label='HighVol V^pi')
ax.plot(val_dates, vpi_lv_val, lw=0.7, color='steelblue',
        alpha=0.8, label='LowVol V^pi')
ax.axhline(0, color='gray', lw=0.5, linestyle=':')
for name, date in [("2022 FTX", pd.Timestamp("2022-11-08"))]:
    if val_dates[0] <= date <= val_dates[-1]:
        ax.axvline(date, color='black', lw=1.5, linestyle='--', label=name)
ax.set_title(f"V^pi 비교 (KS p={ks_p:.4f})")
ax.set_ylabel("V^pi"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 경매 승자 + HMM 레짐
ax = axes[1]
# HMM high vol 확률 배경
ax.fill_between(val_dates, val_highprob, alpha=0.3,
                color='salmon', label='P(High Vol regime)')
# 경매 승자: 0=HighVol, 1=LowVol
winner_series = pd.Series(winners.astype(float), index=val_dates)
ax.scatter(val_dates[winners == 0], np.ones((winners==0).sum()) * 0.95,
           s=4, color='red', alpha=0.5, label='HighVol wins')
ax.scatter(val_dates[winners == 1], np.ones((winners==1).sum()) * 0.05,
           s=4, color='steelblue', alpha=0.5, label='LowVol wins')
ax.set_title(f"경매 승자 vs HMM 레짐 (정렬도={alignment:+.1f}%p)")
ax.set_ylabel("P(High Regime) / Winner")
ax.set_ylim(-0.1, 1.1); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# V^pi 차이 (HighVol - LowVol) 롤링
ax = axes[2]
diff_series = pd.Series(vpi_diff, index=val_dates)
roll_mean   = diff_series.rolling(30).mean()
ax.plot(val_dates, diff_series.values, lw=0.4, color='gray', alpha=0.5)
ax.plot(val_dates, roll_mean.values, lw=1.2, color='purple',
        label='30d rolling mean (HighVol - LowVol)')
ax.axhline(0, color='gray', lw=0.8, linestyle='--')
ax.fill_between(val_dates, 0, val_highprob * diff_series.abs().max(),
                alpha=0.1, color='salmon', label='High vol regime')
ax.set_title("V^pi 차이 추이 (양수=HighVol 우위)")
ax.set_ylabel("V^pi diff"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# V^pi 분포 히스토그램
ax = axes[3]
ax.hist(vpi_hv_val, bins=60, alpha=0.6, color='red',
        density=True, label=f'HighVol (mean={vpi_hv_val.mean():.2f})')
ax.hist(vpi_lv_val, bins=60, alpha=0.6, color='steelblue',
        density=True, label=f'LowVol  (mean={vpi_lv_val.mean():.2f})')
ax.set_title("V^pi 분포 비교")
ax.set_xlabel("V^pi"); ax.set_ylabel("Density"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("arena_validation_v1.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v1.png")


# ══════════════════════════════════════════════════════════
# 10. 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Arena 검증 v1 -- 최종 판정")
print("=" * 60)

q1 = ks_p < 0.05
q2 = alignment > 5.0
q3 = drift < 0.5

print(f"Q1 V^pi 분포 갈림:       {'YES' if q1 else 'NO '} (KS p={ks_p:.4f})")
print(f"Q2 레짐-경매 정렬:       {'YES' if q2 else 'NO '} ({alignment:+.1f}%p)")
print(f"Q3 학습 수렴/안정:       {'YES' if q3 else 'NO '} (drift={drift:.4f})")

n_pass = sum([q1, q2, q3])
if n_pass == 3:
    verdict = "VALID -- 다중 에이전트 학습 동역학 확인"
elif n_pass == 2:
    verdict = "PARTIAL -- 일부 동역학 확인, 하이퍼파라미터 조정 필요"
elif n_pass == 1:
    verdict = "WEAK -- 학습은 되나 에이전트 분기 불충분"
else:
    verdict = "NULL -- 실제 피처로도 에이전트 분기 없음"

print(f"\n판정: {verdict}")
print("=" * 60)
