"""
Arena 검증 v10 - Auxiliary Q-head
===================================
B안: OHLCV reward + Q_tail auxiliary head

아키텍처:
  trunk(128→64→32)
    ├─ policy_head  → logits (action)
    ├─ value_head   → V^pi
    └─ q_head       → Q_tail 예측 (auxiliary)

loss = A2C_loss(OHLCV) + lambda_q * MSE(q_head, Q_tail)

핵심 가설:
  - OHLCV reward 유지 → natural V^pi-Q corr(r=0.415) 보존
  - q_head가 trunk representation을 Q_tail 쪽으로 명시적으로 앵커링
  - 결과적으로 alignment + sharpness + V^pi-Q corr 동시 향상

비교: v1 (OHLCV) vs v9 (Q reward) vs v10 (OHLCV + Q aux, lambda sweep)
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

# ── state(128) ─────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

WIN        = 16
STATE_DIM  = 128
ACTION_DIM = 3
LR         = 1e-3
N_STEPS    = 3000
GAMMA      = 0.99
ALPHA      = 2.0
SEED       = 42
TRAIN_END  = "2022-06-30"
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

print(f"Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()}  n={split_idx}")
print(f"Val  : {val_dates[0].date()} ~ {val_dates[-1].date()}  n={len(val_dates)}")
print(f"Train Q_tail  mean={train_q.mean():.3f}  std={train_q.std():.3f}")
print(f"Val   Q_tail  mean={val_q.mean():.3f}   std={val_q.std():.3f}")


# ── 모델 ──────────────────────────────────────────────────
class PolicyValueNet(nn.Module):
    """기존 v1/v9 와 동일 (비교 기준)"""
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


class PolicyValueNetQ(nn.Module):
    """v10: OHLCV reward + Auxiliary Q_tail head"""
    def __init__(self, state_dim=128, action_dim=3, h1=64, h2=32):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.LayerNorm(h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(),
        )
        self.policy_head = nn.Linear(h2, action_dim)
        self.value_head  = nn.Linear(h2, 1)
        self.q_head      = nn.Linear(h2, 1)   # auxiliary Q_tail predictor

    def forward(self, s):
        """경매 인터페이스 — v1/v9 와 동일한 반환 시그니처"""
        f      = self.trunk(s)
        logits = self.policy_head(f)
        probs  = F.softmax(logits, dim=-1)
        H      = -torch.sum(probs * torch.log(probs + 1e-9))
        conf   = (1.0 - H / torch.log(torch.tensor(float(self.action_dim)))).item()
        vpi    = self.value_head(f).squeeze(-1).item()
        return logits, conf, vpi

    def forward_train(self, s):
        """학습용 — q_pred 포함"""
        f      = self.trunk(s)
        logits = self.policy_head(f)
        probs  = F.softmax(logits, dim=-1)
        vpi    = self.value_head(f).squeeze(-1)
        q_pred = torch.sigmoid(self.q_head(f)).squeeze(-1)   # Q_tail in [0,1]
        return logits, probs, vpi, q_pred


# ── Reward ─────────────────────────────────────────────────
def reward_v1(s, ns, bias_type):
    d = float(ns[-8 + 2]) - float(s[-8 + 2])
    if bias_type == "HighVol":
        return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)

def reward_q(idx, bias_type, q_data):
    qt = float(q_data[idx])
    if bias_type == "HighVol":
        return float(np.clip(qt - q_med, -1.0, 1.0))
    return float(np.clip(q_med - qt, -1.0, 1.0))


# ── 학습 함수 ──────────────────────────────────────────────
def train_v1(data, bias_type):
    net = PolicyValueNet(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        i  = np.random.randint(0, n-1)
        s  = torch.FloatTensor(data[i])
        s1 = torch.FloatTensor(data[i+1])
        logits, _, vpi = net(s)
        probs = F.softmax(logits, -1)
        dist  = torch.distributions.Categorical(probs)
        a = dist.sample(); lp = dist.log_prob(a)
        r = reward_v1(data[i], data[i+1], bias_type)
        with torch.no_grad(): _, _, v1 = net(s1)
        td  = r + GAMMA * v1; adv = td - vpi
        fv  = net.trunk(s); vt = net.value_head(fv).squeeze(-1)
        loss = -lp*adv + 0.5*F.mse_loss(vt, torch.tensor(td)) \
               - 0.01*(-torch.sum(probs*torch.log(probs+1e-9)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net


def train_q_reward(data, q_data, bias_type):
    net = PolicyValueNet(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        i  = np.random.randint(0, n-2)
        s  = torch.FloatTensor(data[i])
        s1 = torch.FloatTensor(data[i+1])
        logits, _, vpi = net(s)
        probs = F.softmax(logits, -1)
        dist  = torch.distributions.Categorical(probs)
        a = dist.sample(); lp = dist.log_prob(a)
        r = reward_q(i, bias_type, q_data)
        with torch.no_grad(): _, _, v1 = net(s1)
        td  = r + GAMMA * v1; adv = td - vpi
        fv  = net.trunk(s); vt = net.value_head(fv).squeeze(-1)
        loss = -lp*adv + 0.5*F.mse_loss(vt, torch.tensor(td)) \
               - 0.01*(-torch.sum(probs*torch.log(probs+1e-9)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net


def train_v10(data, q_data, bias_type, lambda_q=0.5):
    """OHLCV reward + Q_tail auxiliary loss"""
    net = PolicyValueNetQ(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        i  = np.random.randint(0, n-1)
        s  = torch.FloatTensor(data[i])
        s1 = torch.FloatTensor(data[i+1])

        logits, probs, vpi, q_pred = net.forward_train(s)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample(); lp = dist.log_prob(a)

        r = reward_v1(data[i], data[i+1], bias_type)   # OHLCV reward 유지
        with torch.no_grad(): _, _, v1 = net(s1)

        td    = r + GAMMA * v1
        adv   = td - vpi.item()
        entr  = -torch.sum(probs * torch.log(probs + 1e-9))

        a2c_loss = -lp * adv + 0.5 * F.mse_loss(vpi, torch.tensor(td)) - 0.01 * entr
        q_target = torch.tensor(float(q_data[i]))
        q_aux    = F.mse_loss(q_pred, q_target)

        loss = a2c_loss + lambda_q * q_aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net


# ── 평가 함수 ──────────────────────────────────────────────
def evaluate(net_hv, net_lv, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, ch, vh = net_hv(s)
            _, cl, vl = net_lv(s)
            p = float(val_highp[i])
            bid_hv = ch * (1 + ALPHA * p)
            bid_lv = cl * (1 + ALPHA * (1 - p))
            vpi_hv.append(vh); vpi_lv.append(vl)
            winners.append(0 if bid_hv >= bid_lv else 1)

    vpi_hv  = np.array(vpi_hv); vpi_lv = np.array(vpi_lv)
    winners = np.array(winners)
    hm = val_regimes == high_state; lm = ~hm

    align     = ((winners[hm] == 0).mean() - (winners[lm] == 0).mean()) * 100
    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2) / 2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)
    vpi_diff  = vpi_hv - vpi_lv
    corr_q, p_q = pearsonr(vpi_diff, val_q) if val_q.std() > 1e-6 else (0, 1)

    print(f"[{label:30s}]  align={align:+.1f}%p  sharp={sharpness:.3f}  "
          f"KS_p={ks_p:.4f}  V^pi-Q r={corr_q:+.3f}(p={p_q:.3f})")
    return dict(label=label, align=align, sharpness=sharpness, ks_p=ks_p,
                vpi_hv=vpi_hv, vpi_lv=vpi_lv, winners=winners,
                corr_q=corr_q, p_q=p_q)


# ── 기준선 재현 ─────────────────────────────────────────────
print("\n=== 기준선 재현 ===")
nhv_v1 = train_v1(train_data, "HighVol")
nlv_v1 = train_v1(train_data, "LowVol")
res_v1 = evaluate(nhv_v1, nlv_v1, "v1 OHLCV reward")

nhv_v9 = train_q_reward(train_data, train_q, "HighVol")
nlv_v9 = train_q_reward(train_data, train_q, "LowVol")
res_v9 = evaluate(nhv_v9, nlv_v9, "v9 Q_tail reward")


# ── v10 lambda sweep ────────────────────────────────────────
LAMBDAS = [0.1, 0.3, 0.5, 1.0, 2.0]
results_v10 = []

print(f"\n=== v10 lambda sweep (OHLCV + Q_aux) ===")
for lam in LAMBDAS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    nhv = train_v10(train_data, train_q, "HighVol", lambda_q=lam)
    nlv = train_v10(train_data, train_q, "LowVol",  lambda_q=lam)
    res = evaluate(nhv, nlv, f"v10 lambda={lam}")
    results_v10.append(res)

# 최적 lambda 선택: alignment 우선, 동점 시 sharpness
best_v10 = max(results_v10, key=lambda r: (r['align'], r['sharpness']))
print(f"\n최적 lambda: {best_v10['label']}  align={best_v10['align']:+.1f}%p")


# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ① V^pi 분포 — v1 / v9 / best v10
for ax, res, title in zip(
    [axes[0,0], axes[0,1], axes[0,2]],
    [res_v1, res_v9, best_v10],
    ['v1 OHLCV reward', 'v9 Q_tail reward', f'v10 {best_v10["label"]}']
):
    bins = np.linspace(-3, 3, 50)
    ax.hist(res['vpi_hv'], bins=bins, alpha=0.5, color='red',  label='HighVol', density=True)
    ax.hist(res['vpi_lv'], bins=bins, alpha=0.5, color='blue', label='LowVol',  density=True)
    ax.set_title(f"{title}\nalign={res['align']:+.1f}%p  sharp={res['sharpness']:.3f}"
                 f"  V-Q r={res['corr_q']:+.3f}")
    ax.legend(fontsize=8)

# ② lambda sweep: align vs V^pi-Q corr
ax = axes[1, 0]
lam_vals = LAMBDAS
aligns   = [r['align']  for r in results_v10]
corrs    = [r['corr_q'] for r in results_v10]
ax2      = ax.twinx()
ax.plot(lam_vals, aligns, 'o-', color='steelblue', lw=1.5, label='Alignment (%p)')
ax2.plot(lam_vals, corrs, 's--', color='darkorange', lw=1.5, label='V-Q corr')
ax.axhline(res_v1['align'],  color='steelblue',  lw=0.7, linestyle=':', label='v1 align')
ax2.axhline(res_v1['corr_q'],color='darkorange', lw=0.7, linestyle=':', label='v1 corr')
ax.set_xlabel("lambda_q")
ax.set_ylabel("Alignment (%p)", color='steelblue')
ax2.set_ylabel("V^pi-Q corr", color='darkorange')
ax.set_title("lambda sweep: Alignment vs V-Q corr")
ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')

# ③ V^pi diff vs Q_tail 시계열
ax = axes[1, 1]
ax2 = ax.twinx()
for res, color, lbl in [(res_v1, 'steelblue', 'v1'),
                         (res_v9, 'green',     'v9'),
                         (best_v10, 'red',     f'v10(best)')]:
    diff = pd.Series(res['vpi_hv'] - res['vpi_lv'], index=val_dates).rolling(10).mean()
    ax.plot(val_dates, diff, lw=0.9, color=color,
            label=f"{lbl} (r={res['corr_q']:+.3f})")
ax2.fill_between(val_dates, val_q, alpha=0.15, color='gray', label='Q_tail')
ax2.set_ylabel("Q_tail", color='gray')
ax.set_title("V^pi(HV-LV) vs Q_tail (rolling 10)")
ax.set_ylabel("V^pi diff")
ax.legend(fontsize=7, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# ④ 3-way 지표 비교 bar
ax = axes[1, 2]
metrics = ['align', 'sharpness', 'corr_q']
labels  = ['Align(%p)', 'Sharpness', 'V-Q corr']
x = np.arange(len(metrics))
w = 0.22
for xi, (color, res, lbl) in enumerate([
    ('steelblue',  res_v1,    'v1'),
    ('green',      res_v9,    'v9'),
    ('red',        best_v10,  'v10'),
]):
    vals = [abs(res[m]) for m in metrics]
    bars = ax.bar(x + (xi-1)*w, vals, w,
                  label=lbl, color=color, edgecolor='black', lw=0.5)
    for b, v, m in zip(bars, vals, metrics):
        sign = '+' if res[m] >= 0 else '-'
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                f"{sign}{v:.2f}", ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("v1 vs v9 vs v10 지표 비교")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("arena_validation_v10.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v10.png")


# ── 판정 ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Arena v10 - Auxiliary Q-head (OHLCV + Q_aux)")
print("=" * 65)
print(f"{'':30s} {'v1':>10} {'v9':>10} {'v10(best)':>10}")
print("-" * 62)
for k, name in [('align','Alignment(%p)'), ('sharpness','Sharpness'), ('corr_q','V^pi-Q corr')]:
    print(f"  {name:28s} {res_v1[k]:>10.3f} {res_v9[k]:>10.3f} {best_v10[k]:>10.3f}")

# 판정 기준: v1 대비 모든 지표 개선 여부
align_up = best_v10['align'] > res_v1['align']
sharp_up = best_v10['sharpness'] > res_v1['sharpness']
corr_up  = abs(best_v10['corr_q']) > abs(res_v1['corr_q'])

align_delta = f"{best_v10['align']-res_v1['align']:+.1f}%p"
print(f"\n  v1 대비 Alignment 향상: {'YES (' + align_delta + ')' if align_up else 'NO'}")
print(f"  v1 대비 Sharpness 향상: {'YES' if sharp_up else 'NO'}")
print(f"  v1 대비 V^pi-Q corr 향상: {'YES' if corr_up else 'NO'}")

n_improved = sum([align_up, sharp_up, corr_up])
if n_improved == 3:
    verdict = "Auxiliary Q-head 성공 - 모든 지표 동시 향상 (B안 채택)"
elif n_improved == 2:
    verdict = "부분 성공 - 2/3 지표 향상, 트레이드오프 존재"
elif align_up:
    verdict = "Alignment 향상만 확인 - Q 표현 개선 효과 제한적"
else:
    verdict = "Auxiliary Q-head 효과 없음 - C안(Meta-Arena)으로 이동"
print(f"\n  판정: {verdict}")
print(f"  다음 단계: {'C안 Meta-Arena 구현' if n_improved < 2 else 'B안 결과로 C안 피더 구성'}")
print("=" * 65)
