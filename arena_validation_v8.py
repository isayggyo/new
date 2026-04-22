"""
Arena 검증 v8 - Q_tail을 state feature로 추가
===============================================
state(128) + Q_tail(1) = state(129)
에이전트가 옵션 내재 꼬리 확률을 직접 학습

비교: v1 state(128) vs v8 state(129)
지표: alignment, sharpness, KS p-value
"""
import warnings; warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm, johnsonsb, ks_2samp

# ── 데이터 ────────────────────────────────────────────────
with open("arena_val_features.pkl", 'rb') as f:
    feat, btc_price = pickle.load(f)
with open("arena_val_hmm.pkl", 'rb') as f:
    regime, high_p, high_state = pickle.load(f)
with open("option_iv_surface.pkl", 'rb') as f:
    iv_surf = pickle.load(f)

dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)

# ── Q_tail 시계열 ─────────────────────────────────────────
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

# StandardScaler로 Q_tail 정규화 (state 편입)
from sklearn.preprocessing import StandardScaler
scaler_ohlcv = StandardScaler()
X_scaled = scaler_ohlcv.fit_transform(feat.values)

q_arr    = q_series.values
q_mean   = q_arr.mean(); q_std = q_arr.std() + 1e-9
q_scaled = (q_arr - q_mean) / q_std          # 표준화된 Q_tail

# ── state 구성: 128 + 1 = 129 ─────────────────────────────
WIN       = 16
STATE_DIM = 129    # 128 + Q_tail
TRAIN_END = "2022-06-30"
ALPHA     = 2.0
LR        = 1e-3
N_STEPS   = 3000
GAMMA     = 0.99
SEED      = 42
torch.manual_seed(SEED); np.random.seed(SEED)

states_arr, dates_arr = [], []
for i in range(WIN, len(X_scaled)):
    base = X_scaled[i - WIN:i].flatten()           # (128,)
    q    = np.array([q_scaled[i]], dtype=np.float32)  # (1,)
    states_arr.append(np.concatenate([base, q]))
    dates_arr.append(feat.index[i])

states_arr      = np.array(states_arr, dtype=np.float32)
dates_arr       = pd.DatetimeIndex(dates_arr)
regimes_aligned = regime.reindex(dates_arr).fillna(0).values.astype(int)
highp_aligned   = high_p.reindex(dates_arr).fillna(0).values

split_idx   = int(np.searchsorted(dates_arr, pd.Timestamp(TRAIN_END)))
train_data  = states_arr[:split_idx]
val_data    = states_arr[split_idx:]
val_dates   = dates_arr[split_idx:]
val_regimes = regimes_aligned[split_idx:]
val_highp   = highp_aligned[split_idx:]

q_train = q_scaled[:split_idx]; q_val = q_scaled[split_idx:]
valid_frac = (np.abs(q_train - (q_mean - q_mean)/q_std) > 0).mean()  # 항상 1
dvol_valid = (~pd.isna(dv_iv) | ~pd.isna(atm_iv)).reindex(dates_arr[:split_idx]).mean()
print(f"Train Q_tail 유효(DVOL/IV): {dvol_valid*100:.0f}%")
print(f"Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()}  ({split_idx})")
print(f"Val:   {val_dates[0].date()} ~ {val_dates[-1].date()}  ({len(val_data)})")


# ── 모델 ──────────────────────────────────────────────────
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim=129, action_dim=3, h1=64, h2=32):
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


def real_reward(s, ns, bias_type):
    # 마지막 요소가 Q_tail, 나머지는 원래 state
    curr_vol = float(s[WIN*8 - 8 + 2])   # btc_vol in last window step
    next_vol = float(ns[WIN*8 - 8 + 2])
    d = next_vol - curr_vol
    if bias_type == "HighVol":
        return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)


def train_agent(data, bias_type):
    net = PolicyValueNet(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        idx    = np.random.randint(0, n - 1)
        s      = torch.FloatTensor(data[idx])
        s1     = torch.FloatTensor(data[idx + 1])
        logits, _, vpi = net(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        lp     = dist.log_prob(action)
        r      = real_reward(data[idx], data[idx + 1], bias_type)
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


def evaluate(net_hv, net_lv, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, conf_hv, vpi_h = net_hv(s)
            _, conf_lv, vpi_l = net_lv(s)
            p = float(val_highp[i])
            bid_hv = conf_hv * (1.0 + ALPHA * p)
            bid_lv = conf_lv * (1.0 + ALPHA * (1.0 - p))
            vpi_hv.append(vpi_h); vpi_lv.append(vpi_l)
            winners.append(0 if bid_hv >= bid_lv else 1)

    vpi_hv  = np.array(vpi_hv); vpi_lv = np.array(vpi_lv)
    winners = np.array(winners)
    hm = val_regimes == high_state; lm = ~hm
    align = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100
    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)
    print(f"[{label}]  align={align:+.1f}%p  sharp={sharpness:.3f}  KS p={ks_p:.4f}")
    return dict(label=label, align=align, sharpness=sharpness, ks_p=ks_p,
                vpi_hv=vpi_hv, vpi_lv=vpi_lv, winners=winners)


# ── v1 베이스라인 (128차원, 기존 저장 모델) ──────────────────
print("\n=== v1 베이스라인 로드 (state=128) ===")
class PolicyValueNet128(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = 3
        self.trunk = nn.Sequential(
            nn.Linear(128,64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64,32),  nn.LayerNorm(32), nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, 3)
        self.value_head  = nn.Linear(32, 1)
    def forward(self, s):
        f=self.trunk(s); p=F.softmax(self.policy_head(f),-1)
        H=-torch.sum(p*torch.log(p+1e-9))
        return self.policy_head(f),(1-H/torch.log(torch.tensor(3.))).item(),self.value_head(f).squeeze(-1).item()

nhv128=PolicyValueNet128(); nhv128.load_state_dict(torch.load("arena_val_highvol.pt",map_location='cpu'))
nlv128=PolicyValueNet128(); nlv128.load_state_dict(torch.load("arena_val_lowvol.pt", map_location='cpu'))
nhv128.eval(); nlv128.eval()

# v1 평가 (val_data의 앞 128차원만 사용)
val_data_128 = val_data[:, :128]
vpi_hv1, vpi_lv1, winners1 = [], [], []
with torch.no_grad():
    for i, s_np in enumerate(val_data_128):
        s=torch.FloatTensor(s_np)
        _,ch,vh=nhv128(s); _,cl,vl=nlv128(s)
        p=float(val_highp[i])
        bid_hv=ch*(1+ALPHA*p); bid_lv=cl*(1+ALPHA*(1-p))
        vpi_hv1.append(vh); vpi_lv1.append(vl)
        winners1.append(0 if bid_hv>=bid_lv else 1)

vpi_hv1=np.array(vpi_hv1); vpi_lv1=np.array(vpi_lv1); winners1=np.array(winners1)
hm=val_regimes==high_state; lm=~hm
align1=((winners1[hm]==0).mean()-(winners1[lm]==0).mean())*100
sharp1=abs(vpi_hv1.mean()-vpi_lv1.mean())/(np.sqrt((vpi_hv1.std()**2+vpi_lv1.std()**2)/2)+1e-9)
_,ks1=ks_2samp(vpi_hv1,vpi_lv1)
res_v1=dict(label="v1 state(128)", align=align1, sharpness=sharp1, ks_p=ks1,
            vpi_hv=vpi_hv1, vpi_lv=vpi_lv1, winners=winners1)
print(f"[v1 state(128)]  align={align1:+.1f}%p  sharp={sharp1:.3f}  KS p={ks1:.4f}")

# ── v8 재학습 (129차원) ────────────────────────────────────
print("\n=== v8 재학습 (state=129, +Q_tail) ===")
net_hv_v8 = train_agent(train_data, "HighVol")
net_lv_v8 = train_agent(train_data, "LowVol")
res_v8    = evaluate(net_hv_v8, net_lv_v8, "v8 state(129, +Q_tail)")


# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (res, color) in zip(axes[:2], [(res_v1,'steelblue'),(res_v8,'darkorange')]):
    bins = np.linspace(-3, 3, 50)
    ax.hist(res['vpi_hv'], bins=bins, alpha=0.5, color='red',   label='HighVol V^pi', density=True)
    ax.hist(res['vpi_lv'], bins=bins, alpha=0.5, color='blue',  label='LowVol V^pi',  density=True)
    ax.set_title(f"{res['label']}\nalign={res['align']:+.1f}%p  sharp={res['sharpness']:.3f}")
    ax.legend(fontsize=8)

ax = axes[2]
metrics = ['align','sharpness']
x = np.arange(len(metrics))
ax.bar(x-0.2, [res_v1[m] for m in metrics], 0.4,
       label='v1 (128)', color='steelblue', edgecolor='black', lw=0.5)
ax.bar(x+0.2, [res_v8[m] for m in metrics], 0.4,
       label='v8 (+Q_tail)', color='darkorange', edgecolor='black', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(['Alignment(%p)','Sharpness'])
ax.legend(fontsize=9); ax.set_title("v1 vs v8 비교")
for xi, (v1, v8) in zip(x, zip([res_v1[m] for m in metrics],[res_v8[m] for m in metrics])):
    ax.text(xi-0.2, v1+0.1, f'{v1:.1f}', ha='center', fontsize=9)
    ax.text(xi+0.2, v8+0.1, f'{v8:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("arena_validation_v8.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v8.png")

print("\n" + "=" * 55)
print("Arena v8 - Q_tail in state(129) vs v1 state(128)")
print("=" * 55)
print(f"{'':20s} {'v1(128)':>10} {'v8(+Q)':>10} {'delta':>8}")
print("-" * 50)
for k, name in [('align','Alignment(%p)'),('sharpness','Sharpness'),('ks_p','KS p-val')]:
    v1v=res_v1[k]; v8v=res_v8[k]
    print(f"  {name:18s} {v1v:>10.3f} {v8v:>10.3f} {v8v-v1v:>+8.3f}")
delta_a = res_v8['align'] - res_v1['align']
delta_s = res_v8['sharpness'] - res_v1['sharpness']
if delta_a > 5 or delta_s > 1:
    verdict = "Q_tail in state 유효 - 에이전트 학습 개선"
elif delta_a > 0 and delta_s > 0:
    verdict = "소폭 개선 - Q_tail이 보조 신호로 작용"
elif abs(delta_a) < 3 and abs(delta_s) < 0.5:
    verdict = "차이 없음 - Q_tail의 state 편입 효과 없음"
else:
    verdict = "혼재 - alignment/sharpness 트레이드오프"
print(f"\n판정: {verdict}")
print("=" * 55)
