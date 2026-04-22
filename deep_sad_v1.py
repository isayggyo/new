"""
Deep SAD 검증 v1
================
질문: OHLCV 기반 피처로 Deep SAD가 Flash Crash를 이상 징후로 탐지하는가?
      GARCH/HMM NULL 결론 후, semi-supervised hypersphere 접근이 다른가?

설계:
  - 데이터: BTC/ETH 일봉 (yfinance), 2017-2023
  - 피처 (8개): log return, 20d vol, 5d/20d momentum,
                vol ratio, BTC-ETH corr, vol acceleration
  - 입력: 20일 슬라이딩 윈도우 -> LSTM encoder -> 32차원 latent
  - Anomaly 레이블: top 5% realized vol (자동) + 3 crash 이벤트
  - Phase 1: AE unsupervised 사전학습
  - Phase 2: Deep SAD semi-supervised 파인튜닝
  - 검증: AUROC, crash 당일 percentile
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
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
START      = "2017-01-01"
END        = "2023-12-31"
WIN        = 20
INPUT_DIM  = 8
HIDDEN_DIM = 64
LATENT_DIM = 32
AE_EPOCHS  = 50
SAD_EPOCHS = 100
LR         = 1e-3
ETA        = 5.0       # anomaly 가중치
ANOM_PCT   = 0.05      # 상위 5% -> anomaly 자동 레이블
BATCH_SIZE = 64
SEED       = 42

CRASH_EVENTS = {
    "2018 Bull Crash":   pd.Timestamp("2018-01-08"),
    "2020 COVID Crash":  pd.Timestamp("2020-03-12"),
    "2022 FTX Collapse": pd.Timestamp("2022-11-08"),
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

FEAT_CACHE = "sad_features.pkl"
AE_PATH    = "sad_ae.pt"
ENC_PATH   = "sad_encoder.pt"


# ══════════════════════════════════════════════════════════
# 1. 피처 엔지니어링
# ══════════════════════════════════════════════════════════
def build_features() -> pd.DataFrame:
    print("데이터 로딩...")
    frames = {}
    for asset in ["BTC-USD", "ETH-USD"]:
        raw = yf.download(asset, start=START, end=END,
                          auto_adjust=True, progress=False)
        c = raw["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
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

    print(f"  {len(feat)}일 피처 생성: {feat.index.min().date()} ~ {feat.index.max().date()}")
    return feat


if os.path.exists(FEAT_CACHE):
    print(f"피처 캐시 로드: {FEAT_CACHE}")
    with open(FEAT_CACHE, 'rb') as f:
        feat = pickle.load(f)
else:
    feat = build_features()
    with open(FEAT_CACHE, 'wb') as f:
        pickle.dump(feat, f)
    print(f"피처 캐시 저장: {FEAT_CACHE}")


# ══════════════════════════════════════════════════════════
# 2. 슬라이딩 윈도우 + 레이블
# ══════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat.values)

windows, dates_w = [], []
for i in range(WIN, len(X_scaled)):
    windows.append(X_scaled[i - WIN:i])
    dates_w.append(feat.index[i])

windows = np.array(windows, dtype=np.float32)   # (N, 20, 8)
dates_w = pd.DatetimeIndex(dates_w)

# 자동 레이블: top 5% realized vol -> anomaly
last_vol  = feat["btc_vol"].values[WIN:]
vol_thresh = np.percentile(last_vol, (1 - ANOM_PCT) * 100)
labels = np.where(last_vol >= vol_thresh, -1, 1).astype(np.float32)

# 크래시 이벤트 + 전후 3일 강제 anomaly
for crash_date in CRASH_EVENTS.values():
    center = dates_w.get_indexer([crash_date], method='nearest')[0]
    for offset in range(-3, 4):
        j = center + offset
        if 0 <= j < len(labels):
            labels[j] = -1.0

n_anom   = (labels == -1).sum()
n_normal = (labels ==  1).sum()
print(f"\n레이블 - normal: {n_normal}  anomaly: {n_anom} ({n_anom/len(labels)*100:.1f}%)")

# Train/Test 시간순 분할 70/30
split = int(len(windows) * 0.70)
X_tr, X_te = windows[:split], windows[split:]
y_tr, y_te = labels[:split],  labels[split:]
d_tr, d_te = dates_w[:split], dates_w[split:]

X_tr_t = torch.FloatTensor(X_tr).to(device)
X_te_t = torch.FloatTensor(X_te).to(device)
y_tr_t = torch.FloatTensor(y_tr).to(device)


# ══════════════════════════════════════════════════════════
# 3. 모델 정의
# ══════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)     # h: (1, B, hidden)
        return self.fc(h[-1])        # (B, latent)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 latent_dim: int, seq_len: int):
        super().__init__()
        self.seq_len     = seq_len
        self.encoder     = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.dec_lstm    = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dec_fc      = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        z     = self.encoder(x)                              # (B, latent)
        z_rep = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, latent)
        out, _ = self.dec_lstm(z_rep)                        # (B, T, hidden)
        recon  = self.dec_fc(out)                            # (B, T, input)
        return recon, z


# ══════════════════════════════════════════════════════════
# 4. Phase 1 — AE 사전학습
# ══════════════════════════════════════════════════════════
ae    = LSTMAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, WIN).to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=LR)

if os.path.exists(AE_PATH):
    print(f"\nAE 로드: {AE_PATH}")
    ae.load_state_dict(torch.load(AE_PATH, map_location=device))
else:
    print(f"\nPhase 1: AE 사전학습 ({AE_EPOCHS} epochs)...")
    for epoch in range(AE_EPOCHS):
        ae.train()
        perm       = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        for i in range(0, len(X_tr_t), BATCH_SIZE):
            xb       = X_tr_t[perm[i:i + BATCH_SIZE]]
            recon, _ = ae(xb)
            loss     = F.mse_loss(recon, xb)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | recon loss: {epoch_loss:.4f}")
    torch.save(ae.state_dict(), AE_PATH)
    print(f"AE 저장: {AE_PATH}")


# ══════════════════════════════════════════════════════════
# 5. 하이퍼스피어 센터 c 초기화
# ══════════════════════════════════════════════════════════
ae.eval()
with torch.no_grad():
    zs = []
    for i in range(0, len(X_tr_t), BATCH_SIZE):
        _, z = ae(X_tr_t[i:i + BATCH_SIZE])
        zs.append(z)
    c = torch.cat(zs).mean(dim=0)                                  # (latent,)
    c = torch.where(c.abs() < 1e-4, torch.full_like(c, 1e-4), c)  # collapse 방지

print(f"\n센터 c norm: {c.norm().item():.4f}")


# ══════════════════════════════════════════════════════════
# 6. Phase 2 — Deep SAD 파인튜닝
# ══════════════════════════════════════════════════════════
encoder = ae.encoder

if os.path.exists(ENC_PATH):
    print(f"SAD 인코더 로드: {ENC_PATH}")
    encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
else:
    print(f"\nPhase 2: Deep SAD 파인튜닝 ({SAD_EPOCHS} epochs)...")
    opt_sad = torch.optim.Adam(encoder.parameters(), lr=LR * 0.1)

    for epoch in range(SAD_EPOCHS):
        encoder.train()
        perm       = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0

        for i in range(0, len(X_tr_t), BATCH_SIZE):
            idx  = perm[i:i + BATCH_SIZE]
            xb   = X_tr_t[idx]
            yb   = y_tr_t[idx]

            z    = encoder(xb)
            dist = torch.sum((z - c.detach()) ** 2, dim=1)   # (B,)

            # SAD 손실: normal -> dist^2, anomaly -> eta/dist^2
            mask_anom   = yb < 0
            mask_normal = ~mask_anom

            parts = []
            if mask_normal.any():
                parts.append(dist[mask_normal])
            if mask_anom.any():
                parts.append(ETA / (dist[mask_anom] + 1e-6))

            loss = torch.cat(parts).mean()
            opt_sad.zero_grad(); loss.backward(); opt_sad.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | SAD loss: {epoch_loss:.4f}")

    torch.save(encoder.state_dict(), ENC_PATH)
    print(f"SAD 인코더 저장: {ENC_PATH}")


# ══════════════════════════════════════════════════════════
# 7. 이상 점수 계산
# ══════════════════════════════════════════════════════════
def score_windows(X_tensor: torch.Tensor) -> np.ndarray:
    encoder.eval()
    c_cpu = c.detach().cpu()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            z  = encoder(X_tensor[i:i + BATCH_SIZE].to(device)).cpu()
            d  = torch.sum((z - c_cpu) ** 2, dim=1)
            scores.append(d.numpy())
    return np.concatenate(scores)


tr_scores  = score_windows(X_tr_t)
te_scores  = score_windows(X_te_t)
all_scores = np.concatenate([tr_scores, te_scores])
all_labels = np.concatenate([y_tr, y_te])


# ══════════════════════════════════════════════════════════
# 8. 평가
# ══════════════════════════════════════════════════════════
y_bin_all  = (all_labels == -1).astype(int)
y_bin_test = (y_te == -1).astype(int)

auroc_all  = roc_auc_score(y_bin_all,  all_scores)
auroc_test = roc_auc_score(y_bin_test, te_scores)

print(f"\n성능:")
print(f"  전체 AUROC:  {auroc_all:.4f}")
print(f"  Test AUROC:  {auroc_test:.4f}")

print("\n크래시 당일 이상 점수:")
crash_results = []
for crash_name, crash_date in CRASH_EVENTS.items():
    idx   = dates_w.get_indexer([crash_date], method='nearest')[0]
    score = all_scores[idx]
    pct   = float((all_scores < score).mean() * 100)
    crash_results.append((crash_name, crash_date, score, pct))
    print(f"  {crash_name}: score={score:.4f}  상위 {100-pct:.1f}%ile")

top20 = np.argsort(all_scores)[-20:][::-1]
print(f"\n이상 점수 상위 20일:")
for i in top20:
    lbl = "ANOM" if all_labels[i] < 0 else "norm"
    print(f"  {dates_w[i].date()}  score={all_scores[i]:.4f}  [{lbl}]")


# ══════════════════════════════════════════════════════════
# 9. 시각화
# ══════════════════════════════════════════════════════════
score_series = pd.Series(all_scores, index=dates_w)
thresh95     = np.percentile(all_scores, 95)

fig, axes = plt.subplots(3, 1, figsize=(14, 11))

ax = axes[0]
ax.plot(score_series.index, score_series.values,
        lw=0.7, color='steelblue', alpha=0.8)
ax.axhline(thresh95, color='orange', linestyle='--', lw=1, label='95th pct')
for name, date in CRASH_EVENTS.items():
    ax.axvline(date, color='red', lw=1.5, linestyle='--', label=name)
ax.set_title("Deep SAD Anomaly Score over Time")
ax.set_ylabel("||z - c||²")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

ax = axes[1]
anom_mask   = all_labels == -1
normal_mask = all_labels ==  1
ax.scatter(dates_w[normal_mask], all_scores[normal_mask],
           s=2, color='steelblue', alpha=0.3, label='Normal')
ax.scatter(dates_w[anom_mask], all_scores[anom_mask],
           s=8, color='red', alpha=0.8, label='Anomaly (top 5%)')
for name, date in CRASH_EVENTS.items():
    ax.axvline(date, color='darkred', lw=1.5, linestyle='--')
ax.set_title(f"Anomaly Distribution (Test AUROC={auroc_test:.3f})")
ax.set_ylabel("Score")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())

ax = axes[2]
ax.hist(all_scores[normal_mask], bins=80, alpha=0.6,
        color='steelblue', label='Normal', density=True)
ax.hist(all_scores[anom_mask],  bins=30, alpha=0.6,
        color='red',       label='Anomaly', density=True)
ax.set_title("Score Distribution: Normal vs Anomaly")
ax.set_xlabel("Score"); ax.set_ylabel("Density")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("deep_sad_v1.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: deep_sad_v1.png")


# ══════════════════════════════════════════════════════════
# 10. 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Deep SAD v1 -- 최종 판정")
print("=" * 60)
print(f"Test AUROC: {auroc_test:.4f}")

crash_hit = sum(1 for *_, pct in crash_results if pct >= 90)
print(f"크래시 상위 10%ile 탐지: {crash_hit}/{len(crash_results)}")
for name, date, score, pct in crash_results:
    sig = "[HIT]" if pct >= 90 else "[MISS]"
    print(f"  {name:22s}  score={score:.4f}  pct={pct:.1f}%  {sig}")

if auroc_test >= 0.70 and crash_hit >= 2:
    verdict = "SIGNAL -- Deep SAD 유효. IV 피처 추가 진행"
elif auroc_test >= 0.60 or crash_hit >= 1:
    verdict = "WEAK -- 부분 유효"
else:
    verdict = "NULL -- IV 피처 없이는 한계"

print(f"\n판정: {verdict}")
print("=" * 60)
