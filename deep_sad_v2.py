"""
Deep SAD v2 — DVOL (Implied Volatility) 피처 추가
==================================================
v1 결론: Test AUROC 0.46, FTX MISS (OHLCV만으로 한계)
v2 목표: Deribit DVOL 피처 5개 추가 → FTX 탐지 개선 여부 확인

변경점:
  - 데이터 범위: 2021-03-24 ~ 2023-12-31 (DVOL 가용 기간)
  - 피처: 8 OHLCV + 5 DVOL = 13개
  - 동일 아키텍처, 동일 하이퍼파라미터
  - 비교: 같은 기간에서 DVOL 없는 버전 vs DVOL 있는 버전
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle, requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ── 설정 ─────────────────────────────────────────────────
START_DVOL = "2021-03-24"
END        = "2023-12-31"
WIN        = 20
HIDDEN_DIM = 64
LATENT_DIM = 32
AE_EPOCHS  = 80
SAD_EPOCHS = 150
LR         = 1e-3
ETA        = 5.0
ANOM_PCT   = 0.05
BATCH_SIZE = 32
SEED       = 42

CRASH_EVENTS = {
    "2022 FTX Collapse": pd.Timestamp("2022-11-08"),
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════
# 1. DVOL 데이터 (Deribit 공개 API)
# ══════════════════════════════════════════════════════════
DVOL_CACHE = "dvol_btc.csv"

def fetch_dvol() -> pd.Series:
    if os.path.exists(DVOL_CACHE):
        df = pd.read_csv(DVOL_CACHE, index_col=0, parse_dates=True)
        print(f"DVOL 캐시 로드: {len(df)}행")
        return df["dvol"]

    print("DVOL 다운로드...")
    frames = []
    BASE   = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    for year in range(2021, 2024):
        s = int(pd.Timestamp(f"{year}-01-01").timestamp() * 1000)
        e = int(pd.Timestamp(f"{year}-12-31").timestamp() * 1000)
        r = requests.get(BASE, params={"currency":"BTC","resolution":"1D",
                                        "start_timestamp":s,"end_timestamp":e,
                                        "count":400}, timeout=15)
        rows = r.json()["result"]["data"]
        df   = pd.DataFrame(rows, columns=["ts","open","high","low","close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        frames.append(df.set_index("date")[["close"]].rename(columns={"close":"dvol"}))

    dvol = pd.concat(frames).drop_duplicates().sort_index()["dvol"]
    dvol.to_csv(DVOL_CACHE, header=True)
    print(f"DVOL 저장: {len(dvol)}행, {dvol.index.min().date()} ~ {dvol.index.max().date()}")
    return dvol


dvol_raw = fetch_dvol()


# ══════════════════════════════════════════════════════════
# 2. OHLCV 피처 + DVOL 피처 결합
# ══════════════════════════════════════════════════════════
def build_features_v2(dvol: pd.Series) -> pd.DataFrame:
    print("OHLCV 로딩...")
    frames = {}
    for asset in ["BTC-USD", "ETH-USD"]:
        raw = yf.download(asset, start=START_DVOL, end=END,
                          auto_adjust=True, progress=False)
        c = raw["Close"]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        frames[asset] = c

    prices  = pd.DataFrame(frames).dropna()
    ret_btc = np.log(prices["BTC-USD"] / prices["BTC-USD"].shift(1))
    ret_eth = np.log(prices["ETH-USD"] / prices["ETH-USD"].shift(1))
    btc_vol      = ret_btc.rolling(20).std()
    btc_vol_slow = ret_btc.rolling(60).std()

    # OHLCV 8개
    ohlcv = pd.DataFrame({
        "btc_ret":      ret_btc,
        "eth_ret":      ret_eth,
        "btc_vol":      btc_vol,
        "btc_mom5":     prices["BTC-USD"].pct_change(5),
        "btc_mom20":    prices["BTC-USD"].pct_change(20),
        "vol_ratio":    btc_vol / (btc_vol_slow + 1e-9),
        "btc_eth_corr": ret_btc.rolling(20).corr(ret_eth),
        "vol_accel":    btc_vol / (btc_vol.shift(5) + 1e-9) - 1,
    })

    # DVOL 5개 (annualized IV → daily 환산)
    dv = dvol.reindex(ohlcv.index, method='ffill')
    dvol_ma   = dv.rolling(20).mean()
    # IV 프리미엄: DVOL(연율) vs realized vol(연율 환산)
    rv_annual = btc_vol * np.sqrt(252) * 100   # % annualized

    iv_feats = pd.DataFrame({
        "dvol":             dv,
        "dvol_change":      dv.pct_change(),
        "dvol_ma_ratio":    dv / (dvol_ma + 1e-9),
        "dvol_mom5":        dv.pct_change(5),
        "iv_rv_spread":     dv - rv_annual,
    }, index=ohlcv.index)

    feat = pd.concat([ohlcv, iv_feats], axis=1).dropna()
    print(f"  {len(feat)}일 피처: {feat.index.min().date()} ~ {feat.index.max().date()}")
    print(f"  컬럼 수: {feat.shape[1]}개 (OHLCV 8 + DVOL 5)")
    return feat


FEAT_CACHE2 = "sad_features_v2.pkl"
if os.path.exists(FEAT_CACHE2):
    print(f"피처 캐시 로드: {FEAT_CACHE2}")
    with open(FEAT_CACHE2, 'rb') as f:
        feat = pickle.load(f)
else:
    feat = build_features_v2(dvol_raw)
    with open(FEAT_CACHE2, 'wb') as f:
        pickle.dump(feat, f)
    print(f"캐시 저장: {FEAT_CACHE2}")

INPUT_DIM = feat.shape[1]   # 13


# ══════════════════════════════════════════════════════════
# 3. 슬라이딩 윈도우 + 레이블
# ══════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

windows, dates_w = [], []
for i in range(WIN, len(X_scaled)):
    windows.append(X_scaled[i - WIN:i])
    dates_w.append(feat.index[i])

windows = np.array(windows, dtype=np.float32)
dates_w = pd.DatetimeIndex(dates_w)

last_vol   = feat["btc_vol"].values[WIN:]
vol_thresh = np.percentile(last_vol, (1 - ANOM_PCT) * 100)
labels     = np.where(last_vol >= vol_thresh, -1, 1).astype(np.float32)

# FTX 전후 3일 강제 anomaly
for crash_date in CRASH_EVENTS.values():
    center = dates_w.get_indexer([crash_date], method='nearest')[0]
    for offset in range(-3, 4):
        j = center + offset
        if 0 <= j < len(labels):
            labels[j] = -1.0

n_anom = (labels == -1).sum()
print(f"\n레이블 - normal: {(labels==1).sum()}  anomaly: {n_anom} ({n_anom/len(labels)*100:.1f}%)")

# Train: ~ 2022-06-30, Test: 2022-07-01 ~ (FTX Nov 2022가 test에 포함)
split_date = pd.Timestamp("2022-07-01")
split = int(np.searchsorted(dates_w, split_date))
X_tr, X_te = windows[:split], windows[split:]
y_tr, y_te = labels[:split],  labels[split:]
d_tr, d_te = dates_w[:split], dates_w[split:]

print(f"Train: {d_tr[0].date()} ~ {d_tr[-1].date()} ({len(X_tr)}일)")
print(f"Test:  {d_te[0].date()} ~ {d_te[-1].date()} ({len(X_te)}일)")

X_tr_t = torch.FloatTensor(X_tr).to(device)
X_te_t = torch.FloatTensor(X_te).to(device)
y_tr_t = torch.FloatTensor(y_tr).to(device)


# ══════════════════════════════════════════════════════════
# 4. 모델 (v1과 동일 아키텍처, input_dim만 13으로)
# ══════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super().__init__()
        self.seq_len  = seq_len
        self.encoder  = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.dec_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dec_fc   = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z      = self.encoder(x)
        z_rep  = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.dec_lstm(z_rep)
        return self.dec_fc(out), z


# ══════════════════════════════════════════════════════════
# 5. 학습 헬퍼
# ══════════════════════════════════════════════════════════
def train_ae(ae, X_t, epochs, lr, tag=""):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for epoch in range(epochs):
        ae.train()
        perm = torch.randperm(len(X_t))
        loss_sum = 0.0
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[perm[i:i+BATCH_SIZE]]
            recon, _ = ae(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        if epoch % 20 == 0:
            print(f"  {tag}AE Epoch {epoch:3d} | loss: {loss_sum:.4f}")


def compute_center(ae, X_t):
    ae.eval()
    with torch.no_grad():
        zs = [ae(X_t[i:i+BATCH_SIZE])[1] for i in range(0, len(X_t), BATCH_SIZE)]
    c = torch.cat(zs).mean(0)
    return torch.where(c.abs() < 1e-4, torch.full_like(c, 1e-4), c)


def train_sad(encoder, c, X_t, y_t, epochs, lr, tag=""):
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(epochs):
        encoder.train()
        perm = torch.randperm(len(X_t))
        loss_sum = 0.0
        for i in range(0, len(X_t), BATCH_SIZE):
            idx  = perm[i:i+BATCH_SIZE]
            xb, yb = X_t[idx], y_t[idx]
            z    = encoder(xb)
            dist = torch.sum((z - c.detach()) ** 2, dim=1)
            mask_a = yb < 0
            parts  = []
            if (~mask_a).any(): parts.append(dist[~mask_a])
            if  mask_a.any():   parts.append(ETA / (dist[mask_a] + 1e-6))
            loss = torch.cat(parts).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        if epoch % 30 == 0:
            print(f"  {tag}SAD Epoch {epoch:3d} | loss: {loss_sum:.4f}")


def score_all(encoder, c, X_t):
    encoder.eval()
    c_cpu = c.detach().cpu()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_SIZE):
            z = encoder(X_t[i:i+BATCH_SIZE].to(device)).cpu()
            scores.append(torch.sum((z - c_cpu) ** 2, dim=1).numpy())
    return np.concatenate(scores)


# ══════════════════════════════════════════════════════════
# 6. 실험 A — OHLCV only (8 features, 같은 기간)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("실험 A: OHLCV only (8 features)")
print("=" * 50)

ohlcv_cols = ["btc_ret","eth_ret","btc_vol","btc_mom5","btc_mom20",
              "vol_ratio","btc_eth_corr","vol_accel"]
feat_A = feat[ohlcv_cols]
scaler_A = StandardScaler()
XA = scaler_A.fit_transform(feat_A.values)

wins_A = np.array([XA[i-WIN:i] for i in range(WIN, len(XA))], dtype=np.float32)

AE_A = "sad_v2_ae_A.pt"; ENC_A = "sad_v2_enc_A.pt"
ae_A = LSTMAutoEncoder(8, HIDDEN_DIM, LATENT_DIM, WIN).to(device)

X_trA = torch.FloatTensor(wins_A[:split]).to(device)
X_teA = torch.FloatTensor(wins_A[split:]).to(device)
y_trA = torch.FloatTensor(y_tr).to(device)

if os.path.exists(AE_A):
    ae_A.load_state_dict(torch.load(AE_A, map_location=device))
    print("AE-A 로드")
else:
    train_ae(ae_A, X_trA, AE_EPOCHS, LR, tag="A-")
    torch.save(ae_A.state_dict(), AE_A)

cA = compute_center(ae_A, X_trA)

if os.path.exists(ENC_A):
    ae_A.encoder.load_state_dict(torch.load(ENC_A, map_location=device))
    print("Enc-A 로드")
else:
    train_sad(ae_A.encoder, cA, X_trA, y_trA, SAD_EPOCHS, LR*0.1, tag="A-")
    torch.save(ae_A.encoder.state_dict(), ENC_A)

sc_A_tr = score_all(ae_A.encoder, cA, X_trA)
sc_A_te = score_all(ae_A.encoder, cA, X_teA)
sc_A    = np.concatenate([sc_A_tr, sc_A_te])


# ══════════════════════════════════════════════════════════
# 7. 실험 B — OHLCV + DVOL (13 features)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("실험 B: OHLCV + DVOL (13 features)")
print("=" * 50)

AE_B = "sad_v2_ae_B.pt"; ENC_B = "sad_v2_enc_B.pt"
ae_B = LSTMAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, WIN).to(device)

X_trB = X_tr_t
X_teB = X_te_t
y_trB = y_tr_t

if os.path.exists(AE_B):
    ae_B.load_state_dict(torch.load(AE_B, map_location=device))
    print("AE-B 로드")
else:
    train_ae(ae_B, X_trB, AE_EPOCHS, LR, tag="B-")
    torch.save(ae_B.state_dict(), AE_B)

cB = compute_center(ae_B, X_trB)

if os.path.exists(ENC_B):
    ae_B.encoder.load_state_dict(torch.load(ENC_B, map_location=device))
    print("Enc-B 로드")
else:
    train_sad(ae_B.encoder, cB, X_trB, y_trB, SAD_EPOCHS, LR*0.1, tag="B-")
    torch.save(ae_B.encoder.state_dict(), ENC_B)

sc_B_tr = score_all(ae_B.encoder, cB, X_trB)
sc_B_te = score_all(ae_B.encoder, cB, X_teB)
sc_B    = np.concatenate([sc_B_tr, sc_B_te])


# ══════════════════════════════════════════════════════════
# 8. 비교 평가
# ══════════════════════════════════════════════════════════
all_labels = np.concatenate([y_tr, y_te])
y_bin_all  = (all_labels == -1).astype(int)
y_bin_test = (y_te == -1).astype(int)

auroc_A = roc_auc_score(y_bin_test, sc_A_te)
auroc_B = roc_auc_score(y_bin_test, sc_B_te)

print(f"\n성능 비교 (Test set):")
print(f"  OHLCV only  AUROC: {auroc_A:.4f}")
print(f"  +DVOL       AUROC: {auroc_B:.4f}  (delta={auroc_B-auroc_A:+.4f})")

print(f"\nFTX Collapse 이상 점수:")
ftx = pd.Timestamp("2022-11-08")
idx = dates_w.get_indexer([ftx], method='nearest')[0]
pct_A = float((sc_A < sc_A[idx]).mean() * 100)
pct_B = float((sc_B < sc_B[idx]).mean() * 100)
print(f"  OHLCV only: score={sc_A[idx]:.4f}  상위 {100-pct_A:.1f}%ile")
print(f"  +DVOL:      score={sc_B[idx]:.4f}  상위 {100-pct_B:.1f}%ile")


# ══════════════════════════════════════════════════════════
# 9. DVOL vs 이상 점수 시계열
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

ax = axes[0]
dvol_aligned = dvol_raw.reindex(dates_w, method='ffill')
ax.plot(dates_w, dvol_aligned.values, color='purple', lw=0.9, label='DVOL')
ax.axvline(ftx, color='red', lw=1.5, linestyle='--', label='FTX')
ax.set_title("Deribit DVOL Index (BTC 30d Implied Vol)")
ax.set_ylabel("DVOL"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[1]
sA = pd.Series(sc_A, index=dates_w)
ax.plot(dates_w, sA.values, color='steelblue', lw=0.8, alpha=0.9, label='OHLCV only')
ax.axvline(ftx, color='red', lw=1.5, linestyle='--')
ax.axhline(np.percentile(sc_A, 95), color='steelblue', linestyle=':', lw=1)
ax.set_title(f"Deep SAD Score A: OHLCV only (Test AUROC={auroc_A:.3f})")
ax.set_ylabel("Score"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[2]
sB = pd.Series(sc_B, index=dates_w)
ax.plot(dates_w, sB.values, color='darkorange', lw=0.8, alpha=0.9, label='OHLCV+DVOL')
ax.axvline(ftx, color='red', lw=1.5, linestyle='--')
ax.axhline(np.percentile(sc_B, 95), color='darkorange', linestyle=':', lw=1)
ax.set_title(f"Deep SAD Score B: OHLCV+DVOL (Test AUROC={auroc_B:.3f})")
ax.set_ylabel("Score"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[3]
ax.hist(sc_A_te[(y_te==1)], bins=50, alpha=0.5, color='steelblue',
        density=True, label='A normal')
ax.hist(sc_A_te[(y_te==-1)], bins=15, alpha=0.5, color='steelblue',
        density=True, linestyle='--', histtype='step', lw=2, label='A anomaly')
ax.hist(sc_B_te[(y_te==1)], bins=50, alpha=0.5, color='darkorange',
        density=True, label='B normal')
ax.hist(sc_B_te[(y_te==-1)], bins=15, alpha=0.5, color='darkorange',
        density=True, linestyle='--', histtype='step', lw=2, label='B anomaly')
ax.set_title("Test Score Distribution: A vs B")
ax.set_xlabel("Score"); ax.set_ylabel("Density"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("deep_sad_v2.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: deep_sad_v2.png")


# ══════════════════════════════════════════════════════════
# 10. 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Deep SAD v2 -- DVOL 추가 효과")
print("=" * 60)
print(f"Test AUROC A (OHLCV):       {auroc_A:.4f}")
print(f"Test AUROC B (OHLCV+DVOL):  {auroc_B:.4f}  ({auroc_B-auroc_A:+.4f})")
print(f"\nFTX percentile A: {100-pct_A:.1f}%ile")
print(f"FTX percentile B: {100-pct_B:.1f}%ile")

if auroc_B > auroc_A + 0.05 and pct_B >= 90:
    verdict = "DVOL SIGNAL -- IV 피처 유효. machine.py state(128) 편입 진행"
elif auroc_B > auroc_A:
    verdict = "DVOL WEAK -- 소폭 개선. 피처 추가 정제 필요"
else:
    verdict = "DVOL NULL -- IV 추가 효과 없음"

print(f"\n판정: {verdict}")
print("=" * 60)
