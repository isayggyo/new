"""
VAE 사전 검증 실험 v1 — Flash Crash 전조 신호 존재 여부
=========================================================
질문: BTC/ETH/LTC/XRP 일봉 데이터에서 VAE 재구성 오차가
      Flash Crash 전에 통계적으로 유의미하게 상승하는가?

설계:
  - TDA 검증과 동일한 데이터/크래시/통계 방법론 (직접 비교 가능)
  - 모델: LSTM-VAE (인코더: LSTM→μ,σ, 디코더: LSTM→재구성)
  - 입력: 30일 슬라이딩 윈도우 × 4자산 로그수익률
  - 이상 점수: 재구성 오차 (MSE) + KL divergence
  - 학습: 크래시 제외한 '정상' 구간만 사용
  - 통계: Kendall's τ (상승 추세) + Mann-Whitney U (사전/사후)

판정 기준 (TDA와 동일):
  SIGNAL: Kendall τ > 0 & p < 0.05 (2/3 이상)
  NULL:   조건 미충족 → VAE 단독 접근 폐기, DA-VAE 재고
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, mannwhitneyu

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
ASSETS      = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"]
START       = "2017-01-01"
END         = "2023-12-31"
WINDOW_SIZE = 30
PRE_CRASH   = 480
POST_WINDOW = 30

LATENT_DIM  = 16
HIDDEN_DIM  = 64
EPOCHS        = 80
BATCH_SIZE    = 64
LR            = 1e-3
KL_WEIGHT_MAX = 0.5    # annealing 최종 목표값
KL_ANNEAL     = 30     # 이 epoch까지 0 -> KL_WEIGHT_MAX 선형 증가
FREE_BITS     = 0.1    # 차원당 최소 KL (nats) — collapse 방지

CRASH_EVENTS = {
    "2018 Bull Crash":   pd.Timestamp("2018-01-08"),
    "2020 COVID Crash":  pd.Timestamp("2020-03-12"),
    "2022 FTX Collapse": pd.Timestamp("2022-11-08"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
# 2. 데이터
# ═══════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("데이터 로딩 중...")
    frames = {}
    for asset in ASSETS:
        raw = yf.download(asset, start=START, end=END, auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[asset] = close
    df = pd.DataFrame(frames).dropna()
    print(f"  완료: {len(df)}일치, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df

def make_windows(returns: np.ndarray) -> np.ndarray:
    """(N, WINDOW_SIZE, n_assets) 슬라이딩 윈도우"""
    n = len(returns)
    windows = []
    for i in range(WINDOW_SIZE, n):
        windows.append(returns[i - WINDOW_SIZE:i])
    return np.array(windows, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
# 3. LSTM-VAE 모델
# ═══════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)   # h: (1, B, hidden)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_var(h)

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc(z))                          # (B, hidden)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)   # (B, T, hidden)
        out, _ = self.lstm(h)
        return self.out(out)                             # (B, T, input_dim)

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var, kl_weight, free_bits=FREE_BITS):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    # Free bits: 차원별 KL을 free_bits 이하로 내려가지 못하게 클리핑
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, latent)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.mean()
    return recon_loss + kl_weight * kl, recon_loss.item(), kl.item()

# ═══════════════════════════════════════════════════════════════
# 4. 학습 (정상 구간만)
# ═══════════════════════════════════════════════════════════════
def get_normal_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """크래시 전후 60일을 제외한 정상 구간 마스크"""
    mask = np.ones(len(dates), dtype=bool)
    for crash_date in CRASH_EVENTS.values():
        buffer = pd.Timedelta(days=60)
        mask &= ~((dates >= crash_date - buffer) & (dates <= crash_date + buffer))
    return mask

def train_vae(windows: np.ndarray, dates: pd.DatetimeIndex) -> LSTMVAE:
    mask   = get_normal_mask(dates)
    X_norm = windows[mask]
    print(f"\n학습 데이터: {len(X_norm)}/{len(windows)} 윈도우 (정상 구간)")

    # 스케일러: 정상 데이터 기준 정규화
    scaler = StandardScaler()
    shape  = X_norm.shape
    X_flat = scaler.fit_transform(X_norm.reshape(-1, shape[-1]))
    X_norm_scaled = X_flat.reshape(shape)

    tensor = torch.FloatTensor(X_norm_scaled)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMVAE(
        input_dim=len(ASSETS),
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=WINDOW_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("학습 중...")
    for epoch in range(EPOCHS):
        # KL Annealing: 0 -> KL_WEIGHT_MAX 선형 증가
        kl_w = min(KL_WEIGHT_MAX, KL_WEIGHT_MAX * (epoch + 1) / KL_ANNEAL)
        total_loss = total_recon = total_kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, recon_val, kl_val = vae_loss(recon, batch, mu, log_var, kl_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss  += loss.item()
            total_recon += recon_val
            total_kl    += kl_val
        if (epoch + 1) % 10 == 0:
            n = len(loader)
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/n:.4f}  "
                  f"recon={total_recon/n:.4f}  kl={total_kl/n:.4f}  kl_w={kl_w:.3f}")

    return model, scaler

# ═══════════════════════════════════════════════════════════════
# 5. 이상 점수 시계열
# ═══════════════════════════════════════════════════════════════
def compute_anomaly_scores(model: LSTMVAE, windows: np.ndarray,
                            scaler: StandardScaler) -> np.ndarray:
    model.eval()
    shape  = windows.shape
    X_flat = scaler.transform(windows.reshape(-1, shape[-1]))
    X_scaled = X_flat.reshape(shape)

    tensor = torch.FloatTensor(X_scaled).to(device)
    scores = []

    with torch.no_grad():
        for i in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[i:i + BATCH_SIZE]
            recon, mu, log_var = model(batch)
            # 재구성 오차 (window별 MSE)
            recon_err = F.mse_loss(recon, batch, reduction='none')
            recon_err = recon_err.mean(dim=(1, 2))   # (B,)
            # KL divergence (window별)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=1)
            score = recon_err + KL_WEIGHT_MAX * kl
            scores.append(score.cpu().numpy())

    return np.concatenate(scores)

# ═══════════════════════════════════════════════════════════════
# 6. 통계 검증 (TDA v1과 동일)
# ═══════════════════════════════════════════════════════════════
def analyze_crash(crash_name, crash_date, score_series: pd.Series) -> dict:
    pre_start   = crash_date - pd.Timedelta(days=PRE_CRASH)
    pre_window  = score_series[(score_series.index >= pre_start) &
                                (score_series.index < crash_date)]
    post_end    = crash_date + pd.Timedelta(days=POST_WINDOW)
    post_window = score_series[(score_series.index >= crash_date) &
                                (score_series.index < post_end)]

    if len(pre_window) < 20 or len(post_window) < 5:
        print(f"  [{crash_name}] 데이터 부족: pre={len(pre_window)}, post={len(post_window)}")
        return None

    tau, tau_p = kendalltau(np.arange(len(pre_window)), pre_window.values)
    pre_last30  = pre_window.iloc[-POST_WINDOW:] if len(pre_window) >= POST_WINDOW else pre_window
    u_stat, u_p = mannwhitneyu(pre_last30.values, post_window.values, alternative='greater')

    result = dict(crash=crash_name, crash_date=crash_date,
                  pre_n=len(pre_window), post_n=len(post_window),
                  pre_mean=pre_window.mean(), post_mean=post_window.mean(),
                  kendall_tau=tau, kendall_p=tau_p, mann_whitney_p=u_p,
                  pre_window=pre_window, post_window=post_window)

    signal = tau > 0 and tau_p < 0.05
    print(f"\n  [{crash_name}]")
    print(f"    L1 평균: 사전={pre_window.mean():.4f}, 사후={post_window.mean():.4f}")
    print(f"    Kendall's tau = {tau:.4f}, p = {tau_p:.4f} {'[TREND]' if signal else '[NO TREND]'}")
    print(f"    Mann-Whitney p = {u_p:.4f} {'[PRE>POST]' if u_p < 0.05 else '[NO DIFF]'}")
    return result

# ═══════════════════════════════════════════════════════════════
# 7. 시각화
# ═══════════════════════════════════════════════════════════════
def plot_results(score_series: pd.Series, results: list):
    valid = [r for r in results if r is not None]
    fig, axes = plt.subplots(len(valid) + 1, 1, figsize=(14, 4 * (len(valid) + 1)))

    ax = axes[0]
    ax.plot(score_series.index, score_series.values, color='steelblue', lw=0.8, alpha=0.8)
    for r in valid:
        ax.axvline(r["crash_date"], color='red', lw=2, linestyle='--', label=r["crash"])
    ax.set_title("LSTM-VAE Anomaly Score — Full Period")
    ax.set_ylabel("Score (Recon + KL)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for i, r in enumerate(valid):
        ax = axes[i + 1]
        ax.fill_between(r["pre_window"].index, r["pre_window"].values,
                        alpha=0.3, color='orange', label='Pre-crash')
        ax.fill_between(r["post_window"].index, r["post_window"].values,
                        alpha=0.3, color='blue', label='Post-crash')
        ax.plot(r["pre_window"].index, r["pre_window"].values, color='darkorange', lw=1.2)
        ax.plot(r["post_window"].index, r["post_window"].values, color='navy', lw=1.2)
        ax.axvline(r["crash_date"], color='red', lw=2.5, linestyle='--')
        tau, tau_p, u_p = r["kendall_tau"], r["kendall_p"], r["mann_whitney_p"]
        sig = "SIGNAL [O]" if (tau > 0 and tau_p < 0.05) else "NULL [X]"
        ax.set_title(f"{r['crash']} | tau={tau:.3f} p={tau_p:.3f} | MW_p={u_p:.3f} | {sig}")
        ax.set_ylabel("Anomaly Score")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig("vae_validation_v1.png", dpi=150, bbox_inches='tight')
    print("\n그래프 저장: vae_validation_v1.png")

# ═══════════════════════════════════════════════════════════════
# 8. 요약
# ═══════════════════════════════════════════════════════════════
def print_summary(results: list):
    valid = [r for r in results if r is not None]
    print("\n" + "=" * 65)
    print("VAE 검증 실험 v1 — 최종 판정")
    print("=" * 65)

    signal_count = sum(1 for r in valid if r["kendall_tau"] > 0 and r["kendall_p"] < 0.05)
    mw_count     = sum(1 for r in valid if r["mann_whitney_p"] < 0.05)

    print(f"Kendall tau > 0 & p < 0.05: {signal_count}/{len(valid)}")
    print(f"Mann-Whitney 사전>사후 유의:  {mw_count}/{len(valid)}")
    print()
    for r in valid:
        sig = "[SIGNAL]" if (r["kendall_tau"] > 0 and r["kendall_p"] < 0.05) else "[NULL]"
        print(f"  {r['crash']:22s} tau={r['kendall_tau']:+.3f} p={r['kendall_p']:.3f} "
              f"MW_p={r['mann_whitney_p']:.3f} {sig}")

    print("\n[ 최종 판정 ]")
    if signal_count >= 2:
        print("  SIGNAL — VAE 신호 확인. DA-VAE 업그레이드 진행")
    elif signal_count == 1:
        print("  WEAK — 추가 검증 필요 (윈도우 크기, KL 가중치 조정)")
    else:
        print("  NULL — VAE 단독 신호 없음. DA-VAE도 무의미 → 접근 재고")
    print("=" * 65)

    with open("summary_vae_v1.txt", "w", encoding="utf-8") as f:
        f.write("VAE 검증 실험 v1\n")
        f.write(f"Kendall signal: {signal_count}/{len(valid)}\n")
        f.write(f"Mann-Whitney:   {mw_count}/{len(valid)}\n\n")
        for r in valid:
            sig = "[SIGNAL]" if (r["kendall_tau"] > 0 and r["kendall_p"] < 0.05) else "[NULL]"
            f.write(f"{r['crash']:22s} tau={r['kendall_tau']:+.3f} p={r['kendall_p']:.3f} "
                    f"MW_p={r['mann_whitney_p']:.3f} {sig}\n")

# ═══════════════════════════════════════════════════════════════
# 9. 메인
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("VAE 검증 실험 v1 시작\n" + "=" * 65)
    print(f"Device: {device}")

    prices  = load_data()
    returns = np.log(prices / prices.shift(1)).dropna()
    dates   = returns.index

    windows = make_windows(returns.values)
    win_dates = dates[WINDOW_SIZE:]
    print(f"윈도우 생성: {len(windows)}개")

    model, scaler = train_vae(windows, win_dates)

    print("\n이상 점수 계산 중...")
    scores = compute_anomaly_scores(model, windows, scaler)
    score_series = pd.Series(scores, index=win_dates, name="anomaly_score")
    print(f"점수 범위: {scores.min():.4f} ~ {scores.max():.4f}")

    print("\n통계 검증 중...")
    results = []
    for crash_name, crash_date in CRASH_EVENTS.items():
        if crash_date < score_series.index[0] or crash_date > score_series.index[-1]:
            print(f"  [{crash_name}] 범위 밖 — 스킵")
            results.append(None)
            continue
        results.append(analyze_crash(crash_name, crash_date, score_series))

    plot_results(score_series, results)
    print_summary(results)
