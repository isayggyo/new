"""
Arena 검증 v11 - Meta-Arena (C안)
====================================
2단계 경매:
  Level 1 (내부): HighVol vs LowVol  -- 각 agent pool 내부
  Level 2 (메타): v1_winner vs v10_winner 경쟁

메타 입찰 공식:
  meta_bid = conf_winner * (1 + beta * Q_tail)

v1  = OHLCV reward  (자연 V^pi-Q corr=+0.415)
v10 = OHLCV + Q_aux (경매 정렬 우선, align=+73.3%p)

가설:
  - v1은 Q_tail이 낮을 때(평온) 더 신뢰 가능
  - v10은 Q_tail이 높을 때(위기) 더 신뢰 가능
  - 메타 경매가 이 두 특성을 최적 혼합

평가:
  1. 메타 경매 alignment vs v1 / v10 단독
  2. Q_tail 수준별 정렬 (저/중/고 Q_tail 분위)
  3. 메타 레짐 전환 패턴
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

# ── state ─────────────────────────────────────────────────
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

print(f"Train: {dates_arr[0].date()} ~ {dates_arr[split_idx-1].date()}")
print(f"Val  : {val_dates[0].date()} ~ {val_dates[-1].date()}")


# ── 모델 정의 ──────────────────────────────────────────────
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


class PolicyValueNetQ(nn.Module):
    def __init__(self, state_dim=128, action_dim=3, h1=64, h2=32):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.LayerNorm(h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(),
        )
        self.policy_head = nn.Linear(h2, action_dim)
        self.value_head  = nn.Linear(h2, 1)
        self.q_head      = nn.Linear(h2, 1)

    def forward(self, s):
        f      = self.trunk(s)
        logits = self.policy_head(f)
        probs  = F.softmax(logits, dim=-1)
        H      = -torch.sum(probs * torch.log(probs + 1e-9))
        conf   = (1.0 - H / torch.log(torch.tensor(float(self.action_dim)))).item()
        vpi    = self.value_head(f).squeeze(-1).item()
        return logits, conf, vpi

    def forward_train(self, s):
        f      = self.trunk(s)
        logits = self.policy_head(f)
        probs  = F.softmax(logits, dim=-1)
        vpi    = self.value_head(f).squeeze(-1)
        q_pred = torch.sigmoid(self.q_head(f)).squeeze(-1)
        return logits, probs, vpi, q_pred


# ── 학습 함수 ──────────────────────────────────────────────
def reward_v1(s, ns, bias_type):
    d = float(ns[-8 + 2]) - float(s[-8 + 2])
    if bias_type == "HighVol": return float(max(0, d) * 2.0 - max(0, -d))
    return float(-abs(d) + 0.1)

def train_v1(data, bias_type):
    net = PolicyValueNet(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        i  = np.random.randint(0, n-1)
        s  = torch.FloatTensor(data[i]); s1 = torch.FloatTensor(data[i+1])
        logits, _, vpi = net(s); probs = F.softmax(logits, -1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample(); lp = dist.log_prob(a)
        r = reward_v1(data[i], data[i+1], bias_type)
        with torch.no_grad(): _, _, v1 = net(s1)
        td = r + GAMMA * v1; adv = td - vpi
        fv = net.trunk(s); vt = net.value_head(fv).squeeze(-1)
        loss = -lp*adv + 0.5*F.mse_loss(vt, torch.tensor(td)) \
               - 0.01*(-torch.sum(probs*torch.log(probs+1e-9)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net

def train_v10(data, q_data, bias_type, lambda_q=1.0):
    net = PolicyValueNetQ(STATE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n   = len(data)
    for _ in range(N_STEPS):
        i  = np.random.randint(0, n-1)
        s  = torch.FloatTensor(data[i]); s1 = torch.FloatTensor(data[i+1])
        logits, probs, vpi, q_pred = net.forward_train(s)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample(); lp = dist.log_prob(a)
        r = reward_v1(data[i], data[i+1], bias_type)
        with torch.no_grad(): _, _, v1 = net(s1)
        td = r + GAMMA * v1; adv = td - vpi.item()
        entr = -torch.sum(probs * torch.log(probs + 1e-9))
        a2c_loss = -lp*adv + 0.5*F.mse_loss(vpi, torch.tensor(td)) - 0.01*entr
        q_aux = F.mse_loss(q_pred, torch.tensor(float(q_data[i])))
        loss  = a2c_loss + lambda_q * q_aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net


# ── 단독 평가 ──────────────────────────────────────────────
def evaluate_single(net_hv, net_lv, label):
    vpi_hv, vpi_lv, winners = [], [], []
    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s = torch.FloatTensor(s_np)
            _, ch, vh = net_hv(s); _, cl, vl = net_lv(s)
            p = float(val_highp[i])
            bid_hv = ch*(1+ALPHA*p); bid_lv = cl*(1+ALPHA*(1-p))
            vpi_hv.append(vh); vpi_lv.append(vl)
            winners.append(0 if bid_hv >= bid_lv else 1)
    vpi_hv = np.array(vpi_hv); vpi_lv = np.array(vpi_lv); winners = np.array(winners)
    hm = val_regimes == high_state; lm = ~hm
    align     = ((winners[hm]==0).mean() - (winners[lm]==0).mean()) * 100
    pool_std  = np.sqrt((vpi_hv.std()**2 + vpi_lv.std()**2)/2 + 1e-9)
    sharpness = abs(vpi_hv.mean() - vpi_lv.mean()) / pool_std
    _, ks_p   = ks_2samp(vpi_hv, vpi_lv)
    corr_q, p_q = pearsonr(vpi_hv - vpi_lv, val_q) if val_q.std() > 1e-6 else (0, 1)
    print(f"[{label:25s}]  align={align:+.1f}%p  sharp={sharpness:.3f}  V-Q r={corr_q:+.3f}")
    return dict(label=label, align=align, sharpness=sharpness, ks_p=ks_p,
                vpi_hv=vpi_hv, vpi_lv=vpi_lv, winners=winners,
                corr_q=corr_q, p_q=p_q)


# ── 메타 경매 ─────────────────────────────────────────────
def run_meta_auction(pool_v1, pool_v10, beta=1.0):
    """
    2단계 경매:
      Lv1: 각 pool 내부 HV vs LV → 내부 winner 결정
      Lv2: v1_winner vs v10_winner, meta_bid = conf * (1 + beta * Q_tail)
    """
    net_hv1, net_lv1 = pool_v1
    net_hv10, net_lv10 = pool_v10

    meta_winners     = []   # 0=v1 선택, 1=v10 선택
    final_hv_winners = []   # 최종 HV가 이겼는지
    conf_v1_arr, conf_v10_arr = [], []

    with torch.no_grad():
        for i, s_np in enumerate(val_data):
            s  = torch.FloatTensor(s_np)
            p  = float(val_highp[i])
            qt = float(val_q[i])

            # Lv1 — v1 pool
            _, ch1, vh1 = net_hv1(s); _, cl1, vl1 = net_lv1(s)
            bid_hv1 = ch1*(1+ALPHA*p); bid_lv1 = cl1*(1+ALPHA*(1-p))
            v1_hv_wins = (bid_hv1 >= bid_lv1)
            conf_v1 = ch1 if v1_hv_wins else cl1

            # Lv1 — v10 pool
            _, ch10, vh10 = net_hv10(s); _, cl10, vl10 = net_lv10(s)
            bid_hv10 = ch10*(1+ALPHA*p); bid_lv10 = cl10*(1+ALPHA*(1-p))
            v10_hv_wins = (bid_hv10 >= bid_lv10)
            conf_v10 = ch10 if v10_hv_wins else cl10

            # Lv2 — 메타 경매: Q_tail 높으면 v10 우대, 낮으면 v1 우대
            # v10은 Q-anchor (고Q_tail 구간 특화)
            # v1은 자연 V^pi-Q corr (저Q_tail 구간 특화)
            meta_bid_v1  = conf_v1  * (1 + beta * (1 - qt))
            meta_bid_v10 = conf_v10 * (1 + beta * qt)
            use_v10 = (meta_bid_v10 > meta_bid_v1)

            meta_winners.append(1 if use_v10 else 0)
            # 최종 HV winner: 선택된 pool의 Lv1 결과
            final_hv = v10_hv_wins if use_v10 else v1_hv_wins
            final_hv_winners.append(int(final_hv))

            conf_v1_arr.append(conf_v1); conf_v10_arr.append(conf_v10)

    meta_winners     = np.array(meta_winners)
    final_hv_winners = np.array(final_hv_winners)
    hm = val_regimes == high_state; lm = ~hm

    align     = ((final_hv_winners[hm]==1).mean() - (final_hv_winners[lm]==1).mean()) * 100
    v1_usage  = (meta_winners == 0).mean() * 100
    v10_usage = (meta_winners == 1).mean() * 100

    # Q_tail 분위별 v10 사용률 (v10이 고Q 구간에서 더 많이 쓰이는지)
    q_low  = val_q < np.percentile(val_q, 33)
    q_mid  = (val_q >= np.percentile(val_q, 33)) & (val_q < np.percentile(val_q, 67))
    q_high = val_q >= np.percentile(val_q, 67)
    v10_at_lo  = (meta_winners[q_low]  == 1).mean() * 100
    v10_at_mid = (meta_winners[q_mid]  == 1).mean() * 100
    v10_at_hi  = (meta_winners[q_high] == 1).mean() * 100

    print(f"  beta={beta:.1f}  align={align:+.1f}%p  "
          f"v1={v1_usage:.0f}%  v10={v10_usage:.0f}%  "
          f"v10@Q_lo={v10_at_lo:.0f}%  v10@Q_hi={v10_at_hi:.0f}%")
    return dict(beta=beta, align=align, meta_winners=meta_winners,
                final_hv_winners=final_hv_winners,
                v1_usage=v1_usage, v10_usage=v10_usage,
                v10_at_lo=v10_at_lo, v10_at_mid=v10_at_mid, v10_at_hi=v10_at_hi,
                conf_v1=np.array(conf_v1_arr), conf_v10=np.array(conf_v10_arr))


# ── 학습 ─────────────────────────────────────────────────
print("\n=== 에이전트 학습 ===")
torch.manual_seed(SEED); np.random.seed(SEED)
nhv_v1 = train_v1(train_data, "HighVol")
nlv_v1 = train_v1(train_data, "LowVol")

torch.manual_seed(SEED); np.random.seed(SEED)
nhv_v10 = train_v10(train_data, train_q, "HighVol", lambda_q=1.0)
nlv_v10 = train_v10(train_data, train_q, "LowVol",  lambda_q=1.0)

print("\n=== 단독 평가 (기준선) ===")
res_v1  = evaluate_single(nhv_v1,  nlv_v1,  "v1  OHLCV")
res_v10 = evaluate_single(nhv_v10, nlv_v10, "v10 OHLCV+Q_aux")


# ── 메타 beta sweep ─────────────────────────────────────────
print("\n=== Meta-Arena beta sweep ===")
BETAS = [0.0, 0.5, 1.0, 2.0, 3.0]
meta_results = []
for b in BETAS:
    mr = run_meta_auction((nhv_v1, nlv_v1), (nhv_v10, nlv_v10), beta=b)
    meta_results.append(mr)

best_meta = max(meta_results, key=lambda r: r['align'])
print(f"\n최적 meta beta={best_meta['beta']}  align={best_meta['align']:+.1f}%p")


# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ① Alignment 비교
ax = axes[0, 0]
labels_bar = ['v1\n(OHLCV)', 'v10\n(+Q_aux)'] + [f'Meta\nb={b}' for b in BETAS]
aligns_bar = [res_v1['align'], res_v10['align']] + [r['align'] for r in meta_results]
colors_bar = ['steelblue','darkorange'] + ['gray']*2 + ['red'] + ['gray']*2
colors_bar[2 + BETAS.index(best_meta['beta'])] = 'red'
bars = ax.bar(range(len(labels_bar)), aligns_bar,
              color=['steelblue','darkorange']+['lightcoral']*len(BETAS),
              edgecolor='black', lw=0.5)
bars[2 + BETAS.index(best_meta['beta'])].set_color('red')
for b, v in zip(bars, aligns_bar):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:+.1f}", ha='center', fontsize=8)
ax.set_xticks(range(len(labels_bar))); ax.set_xticklabels(labels_bar, fontsize=8)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title("Alignment 비교 (단독 vs Meta-Arena)")
ax.set_ylabel("Alignment (%p)")

# ② v10 사용률 vs Q_tail 분위
ax = axes[0, 1]
x_q = np.arange(3)
for mr in meta_results[:3]:
    ax.plot(x_q, [mr['v10_at_lo'], mr['v10_at_mid'], mr['v10_at_hi']],
            'o-', lw=1.2, label=f"beta={mr['beta']}")
ax.axhline(50, color='gray', lw=0.6, linestyle='--', label='random')
ax.set_xticks(x_q); ax.set_xticklabels(['Q_low\n(<33%ile)', 'Q_mid', 'Q_high\n(>67%ile)'])
ax.set_title("v10 사용률 by Q_tail 분위")
ax.set_ylabel("v10 선택 비율 (%)")
ax.legend(fontsize=8)

# ③ 메타 선택 시계열 (최적 beta)
ax = axes[0, 2]
ax2 = ax.twinx()
meta_choice = pd.Series(best_meta['meta_winners'], index=val_dates)
q_roll      = pd.Series(val_q, index=val_dates).rolling(5).mean()
ax.fill_between(val_dates, meta_choice, alpha=0.5, color='darkorange', label='v10 선택(1)')
ax2.plot(val_dates, q_roll, lw=1.0, color='green', alpha=0.8, label='Q_tail (5d MA)')
ax.set_title(f"Meta 선택 패턴 (beta={best_meta['beta']})\nv10 선택 vs Q_tail")
ax.set_ylabel("v10 선택 (1=v10, 0=v1)"); ax2.set_ylabel("Q_tail", color='green')
ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# ④ V^pi 분포 — v1 vs v10
for ax, res, title in zip([axes[1,0], axes[1,1]], [res_v1, res_v10],
                           ['v1 (OHLCV)', 'v10 (OHLCV+Q_aux)']):
    bins = np.linspace(-3, 3, 50)
    ax.hist(res['vpi_hv'], bins=bins, alpha=0.5, color='red',  label='HighVol', density=True)
    ax.hist(res['vpi_lv'], bins=bins, alpha=0.5, color='blue', label='LowVol',  density=True)
    ax.set_title(f"{title}\nalign={res['align']:+.1f}%p  sharp={res['sharpness']:.3f}"
                 f"  V-Q r={res['corr_q']:+.3f}")
    ax.legend(fontsize=8)

# ⑤ beta sweep 종합
ax = axes[1, 2]
betas_x = [r['beta'] for r in meta_results]
aligns_m = [r['align'] for r in meta_results]
v10_hi   = [r['v10_at_hi'] for r in meta_results]
ax2 = ax.twinx()
ax.plot(betas_x, aligns_m, 'o-', color='red', lw=1.5, label='Meta Alignment')
ax.axhline(res_v1['align'],  color='steelblue',  lw=0.8, linestyle='--', label='v1 align')
ax.axhline(res_v10['align'], color='darkorange', lw=0.8, linestyle='--', label='v10 align')
ax2.plot(betas_x, v10_hi, 's--', color='green', lw=1.2, label='v10@Q_hi %')
ax.set_xlabel("meta beta")
ax.set_ylabel("Alignment (%p)", color='red')
ax2.set_ylabel("v10 사용률 @Q_high (%)", color='green')
ax.set_title("Meta beta sweep")
ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig("arena_validation_v11.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: arena_validation_v11.png")


# ── 판정 ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Arena v11 - Meta-Arena (C안)")
print("=" * 65)
print(f"  v1  OHLCV:      align={res_v1['align']:+.1f}%p  V-Q r={res_v1['corr_q']:+.3f}")
print(f"  v10 OHLCV+Qaux: align={res_v10['align']:+.1f}%p  V-Q r={res_v10['corr_q']:+.3f}")
print(f"  Meta best:      align={best_meta['align']:+.1f}%p  (beta={best_meta['beta']})")

meta_beat_both = best_meta['align'] > max(res_v1['align'], res_v10['align'])
q_selective    = best_meta['v10_at_hi'] > best_meta['v10_at_lo'] + 10

print(f"\n  Meta > max(v1,v10): {'YES' if meta_beat_both else 'NO'}")
print(f"  Q_tail 선택적 v10 사용: {'YES' if q_selective else 'NO'} "
      f"(hi={best_meta['v10_at_hi']:.0f}% vs lo={best_meta['v10_at_lo']:.0f}%)")

if meta_beat_both and q_selective:
    verdict = "Meta-Arena 성공: 고Q_tail에서 v10, 저Q_tail에서 v1 선택적 활용"
elif meta_beat_both:
    verdict = "Meta-Arena Alignment 향상: 단 Q_tail 선택성 약함"
elif q_selective:
    verdict = "Q_tail 선택성 확인: Alignment는 v10 단독보다 낮음"
else:
    verdict = "단순 합성 이득 없음 - v10 단독 사용이 최선"

print(f"\n  판정: {verdict}")
print("=" * 65)
