"""
Arena 검증 v9 - P/Q 역할 분리
================================
P (HMM)    -> 경매 regime_prob  (감지)
Q (옵션)   -> 에이전트 reward   (변동성 추출)

reward:
  HighVol: +Q_tail  (long vol: 시장이 공포 프리미엄 많이 내재할수록 이득)
  LowVol:  -Q_tail  (short vol: 프리미엄 압축될수록 이득)

비교: v1 (OHLCV reward) vs v9 (Q reward, P 경매)
핵심 질문: V^pi가 Q_tail을 추적하는가?
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
from scipy.stats import norm, ks_2samp, pearsonr

# ── 데이터 ────────────────────────────────────────────────
with open("arena_val_features.pkl", 'rb') as f:
    feat, _ = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)
with open("option_iv_surface.pkl", 'rb') as f:
    iv_surf = pickle.load(f)

dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)

# ── Q_tail ─────────────────────────────────────────────────
THRESHOLD = 0.15

def q_tail(atm_iv, rr, T=30/365):
    if np.isnan(atm_iv) or atm_iv <= 0: return np.nan
    rr = rr if not np.isnan(rr) else 0.0
    il = np.clip(atm_iv - 0.5*abs(rr), 0.05, 5)
    ih = np.clip(atm_iv + 0.5*abs(rr), 0.05, 5)
    d_lo = (np.log(1/(1-THRESHOLD)) - 0.5*il**2*T) / (il*np.sqrt(T))
    d_hi = (np.log(1/(1+THRESHOLD)) - 0.5*ih**2*T) / (ih*np.sqrt(T))
    return float(norm.cdf(-d_lo) + norm.cdf(d_hi))

atm_iv = iv_surf['atm_iv_front'].reindex(feat.index).ffill(limit=20)
rr_sk  = iv_surf['rr_skew'].reindex(feat.index).ffill(limit=20)
dv_iv  = dvol['dvol'].reindex(feat.index).ffill() / 100.0

qs = []
for i in range(len(feat)):
    iv = float(atm_iv.iloc[i]) if not pd.isna(atm_iv.iloc[i]) \
         else (float(dv_iv.iloc[i]) if not pd.isna(dv_iv.iloc[i]) else np.nan)
    rr = float(rr_sk.iloc[i]) if not pd.isna(rr_sk.iloc[i]) else np.nan
    qs.append(q_tail(iv, rr))

q_series = pd.Series(qs, index=feat.index)
q_med    = float(q_series.median())
q_series = q_series.fillna(q_med)
q_arr    = q_series.values

# ── state(128) ─────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN       = 16
STATE_DIM = 128
ACTION_DIM = 3
LR        = 1e-3
N_STEPS   = 3000
GAMMA     = 0.99
ALPHA     = 2.0
SEED      = 42
TRAIN_END = "2022-06-30"
torch.manual_seed(SEED); np.random.seed(SEED)

states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    states_arr.append(X_scaled[i - WIN:i].flatten())
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
highp_aligned   = high_p.reindex(dates_arr).fillna(0).values
q_aligned       = q_series.reindex(dates_arr).fillna(q_med).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
train_data  = states_arr[:split_idx]
train_q     = q_aligned[:split_idx]
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_highp   = highp_aligned[split_idx:]
val_q       = q_aligned[split_idx:]

print(f"Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()}")
print(f"Val  : {val_dates[0].date()} ~ {val_dates[-1].date()}")
print(f"Train Q_tail mean={train_q.mean():.3f}  std={train_q.std():.3f}")


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


# ── Reward 정의 ────────────────────────────────────────────
def reward_v1(s, ns, bias_type):
    """기존: OHLCV 기반 변동성 방향"""
    d = float(ns[-8 + 2]) - float(s[-8 + 2])
    if bias_type == "HighVol":
        return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)


def reward_q(idx, bias_type, q_data):
    """신규: Q_tail 기반 (long/short vol premium)"""
    qt = float(q_data[idx])
    if bias_type == "HighVol":
        return float(np.clip(qt - q_med, -1.0, 1.0))   # 프리미엄 높을수록 이득
    else:
        return float(np.clip(q_med - qt, -1.0, 1.0))   # 프리미엄 낮을수록 이득


def train_v1(data, bias_type):
    net = PolicyValueNet(STATE_DIM); opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(data)
    for _ in range(N_STEPS):
        i = np.random.randint(0, n-1)
        s=torch.FloatTensor(data[i]); s1=torch.FloatTensor(data[i+1])
        logits,_,vpi=net(s); probs=F.softmax(logits,-1)
        dist=torch.distributions.Categorical(probs); a=dist.sample(); lp=dist.log_prob(a)
        r=reward_v1(data[i],data[i+1],bias_type)
        with torch.no_grad(): _,_,v1=net(s1)
        td=r+GAMMA*v1; adv=td-vpi
        fv=net.trunk(s); vt=net.value_head(fv).squeeze(-1)
        loss=-lp*adv+0.5*F.mse_loss(vt,torch.tensor(td))-0.01*(-torch.sum(probs*torch.log(probs+1e-9)))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    return net


def train_q_reward(data, q_data, bias_type):
    net = PolicyValueNet(STATE_DIM); opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(data)
    for _ in range(N_STEPS):
        i = np.random.randint(0, n-2)
        s=torch.FloatTensor(data[i]); s1=torch.FloatTensor(data[i+1])
        logits,_,vpi=net(s); probs=F.softmax(logits,-1)
        dist=torch.distributions.Categorical(probs); a=dist.sample(); lp=dist.log_prob(a)
        r=reward_q(i, bias_type, q_data)
        with torch.no_grad(): _,_,v1=net(s1)
        td=r+GAMMA*v1; adv=td-vpi
        fv=net.trunk(s); vt=net.value_head(fv).squeeze(-1)
        loss=-lp*adv+0.5*F.mse_loss(vt,torch.tensor(td))-0.01*(-torch.sum(probs*torch.log(probs+1e-9)))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    return net


def evaluate(net_hv, net_lv, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s=torch.FloatTensor(s_np)
            _,ch,vh=net_hv(s); _,cl,vl=net_lv(s)
            p=float(val_highp[i])
            bid_hv=ch*(1+ALPHA*p); bid_lv=cl*(1+ALPHA*(1-p))
            vpi_hv.append(vh); vpi_lv.append(vl)
            winners.append(0 if bid_hv>=bid_lv else 1)
    vpi_hv=np.array(vpi_hv); vpi_lv=np.array(vpi_lv); winners=np.array(winners)
    hm=val_regimes==high_state; lm=~hm
    align=((winners[hm]==0).mean()-(winners[lm]==0).mean())*100
    pool_std=np.sqrt((vpi_hv.std()**2+vpi_lv.std()**2)/2+1e-9)
    sharpness=abs(vpi_hv.mean()-vpi_lv.mean())/pool_std
    _,ks_p=ks_2samp(vpi_hv,vpi_lv)
    vpi_diff=vpi_hv-vpi_lv
    corr_q,corr_p=pearsonr(vpi_diff,val_q) if val_q.std()>1e-6 else (0,1)
    print(f"[{label}]  align={align:+.1f}%p  sharp={sharpness:.3f}  "
          f"KS p={ks_p:.4f}  V^pi-Q corr={corr_q:+.3f}(p={corr_p:.3f})")
    return dict(label=label, align=align, sharpness=sharpness, ks_p=ks_p,
                vpi_hv=vpi_hv, vpi_lv=vpi_lv, winners=winners,
                corr_q=corr_q, corr_p=corr_p)


# ── 학습 및 평가 ──────────────────────────────────────────
print("\n=== v1: OHLCV reward (기준) ===")
nhv_v1 = train_v1(train_data, "HighVol")
nlv_v1 = train_v1(train_data, "LowVol")
res_v1 = evaluate(nhv_v1, nlv_v1, "v1 OHLCV reward")

print("\n=== v9: Q_tail reward + P 경매 ===")
nhv_v9 = train_q_reward(train_data, train_q, "HighVol")
nlv_v9 = train_q_reward(train_data, train_q, "LowVol")
res_v9 = evaluate(nhv_v9, nlv_v9, "v9 Q_tail reward")


# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# V^pi 분포 비교
for ax, res, title in zip([axes[0,0],axes[0,1]],
                          [res_v1, res_v9],
                          ['v1 (OHLCV reward)', 'v9 (Q_tail reward)']):
    bins = np.linspace(-3,3,50)
    ax.hist(res['vpi_hv'], bins=bins, alpha=0.5, color='red',  label='HighVol', density=True)
    ax.hist(res['vpi_lv'], bins=bins, alpha=0.5, color='blue', label='LowVol',  density=True)
    ax.set_title(f"{title}\nalign={res['align']:+.1f}%p  sharp={res['sharpness']:.3f}"
                 f"  V^pi-Q corr={res['corr_q']:+.3f}")
    ax.legend(fontsize=8)

# V^pi diff vs Q_tail 시계열
ax = axes[1,0]
ax2 = ax.twinx()
for res, color, lbl in [(res_v1,'steelblue','v1'),(res_v9,'darkorange','v9')]:
    diff = pd.Series(res['vpi_hv']-res['vpi_lv'], index=val_dates).rolling(10).mean()
    ax.plot(val_dates, diff, lw=0.9, color=color, label=f'{lbl} V^pi diff (r={res["corr_q"]:+.3f})')
ax2.fill_between(val_dates, val_q, alpha=0.2, color='green', label='Q_tail')
ax2.set_ylabel("Q_tail", color='green')
ax.set_title("V^pi(HV-LV) vs Q_tail")
ax.set_ylabel("V^pi diff (rolling 10)")
ax.legend(fontsize=7, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 지표 비교
ax = axes[1,1]
metrics = ['align','sharpness','corr_q']
labels  = ['Alignment(%p)', 'Sharpness', 'V^pi-Q corr']
x = np.arange(len(metrics))
ax.bar(x-0.2, [abs(res_v1[m]) for m in metrics], 0.4,
       label='v1 OHLCV', color='steelblue', edgecolor='black', lw=0.5)
ax.bar(x+0.2, [abs(res_v9[m]) for m in metrics], 0.4,
       label='v9 Q_tail', color='darkorange', edgecolor='black', lw=0.5)
for xi, m in zip(x, metrics):
    ax.text(xi-0.2, abs(res_v1[m])+0.1, f'{res_v1[m]:.2f}', ha='center', fontsize=8)
    ax.text(xi+0.2, abs(res_v9[m])+0.1, f'{res_v9[m]:.2f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("v1 vs v9 지표 비교")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("arena_validation_v9.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v9.png")

# ── 판정 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Arena v9 - P 감지 / Q 변동성 추출 분리")
print("=" * 60)
print(f"{'':20s} {'v1(OHLCV)':>12} {'v9(Q rwd)':>12} {'delta':>8}")
print("-" * 56)
for k, name in [('align','Alignment(%p)'),('sharpness','Sharpness'),('corr_q','V^pi-Q corr')]:
    print(f"  {name:18s} {res_v1[k]:>12.3f} {res_v9[k]:>12.3f} {res_v9[k]-res_v1[k]:>+8.3f}")

corr_improved  = abs(res_v9['corr_q']) > abs(res_v1['corr_q'])
align_held     = res_v9['align'] > res_v1['align'] - 10
sharp_improved = res_v9['sharpness'] > res_v1['sharpness']

print(f"\n  V^pi-Q corr 개선: {'YES' if corr_improved  else 'NO'}")
print(f"  Alignment 유지:   {'YES' if align_held     else 'NO'}")
print(f"  Sharpness 개선:   {'YES' if sharp_improved else 'NO'}")

if corr_improved and align_held:
    verdict = "P/Q 역할 분리 성공 - V^pi가 Q 기반 변동성을 추출하면서 경매 정렬도 유지"
elif corr_improved:
    verdict = "V^pi-Q 연결 확인 - alignment 희생이 있지만 이론적 연결 성립"
else:
    verdict = "Q reward가 V^pi-Q 연결을 만들지 못함"
print(f"\n  판정: {verdict}")
print("=" * 60)
