"""
VAE 검증 실험 v2 -- 분봉 실시간 감지
======================================
질문: BTC/ETH/LTC/XRP 1분봉 데이터에서 LSTM-VAE 이상 점수가
      Flash Crash 발생 후 T분 내에 통계적으로 유의미하게 급등하는가?

변경점 (v1 대비):
  - 데이터: 일봉 -> 1분봉 (Binance 공개 API)
  - 윈도우: 30일 -> 60분
  - 목표: 전조 예측 -> 실시간 감지
  - 검증: Kendall tau(추세) + 감지 지연 T분 분석

판정 기준:
  SIGNAL: 크래시 후 60분 내 이상 점수가 사전 기준치의 3-sigma 초과
           + Mann-Whitney (크래시 전 2시간 vs 크래시 후 2시간) p < 0.05
  NULL:   조건 미충족
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT"]
INTERVAL    = "1m"
WINDOW_SIZE = 60       # 60분 윈도우
CONTEXT     = 2 * 60   # 크래시 전후 2시간 비교
DETECT_WIN  = 60       # 감지 윈도우 (크래시 후 60분)
SIGMA_THRESH = 3.0     # 이상 판정 기준 (z-score)

LATENT_DIM   = 16
HIDDEN_DIM   = 64
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 1e-3
KL_WEIGHT_MAX = 0.5
KL_ANNEAL    = 20
FREE_BITS    = 0.1

# 크래시 이벤트 (UTC 기준 분봉 시작점)
CRASH_EVENTS = {
    "2018 Bull Crash":   pd.Timestamp("2018-01-08 00:00:00", tz="UTC"),
    "2020 COVID Crash":  pd.Timestamp("2020-03-12 00:00:00", tz="UTC"),
    "2022 FTX Collapse": pd.Timestamp("2022-11-08 00:00:00", tz="UTC"),
}

# 학습용 정상 구간 (크래시와 무관한 안정적 구간)
NORMAL_PERIOD = {
    "start": pd.Timestamp("2021-01-01", tz="UTC"),
    "end":   pd.Timestamp("2021-06-30", tz="UTC"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
# 2. Binance API 데이터 로딩
# ═══════════════════════════════════════════════════════════════
BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Binance 1분봉 종가 시계열 반환"""
    all_rows = []
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)
    cur_ms   = start_ms

    while cur_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": cur_ms,
            "endTime":   end_ms,
            "limit":     1000,
        }
        for attempt in range(5):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                data = resp.json()
                break
            except Exception:
                if attempt == 4:
                    data = []
                time.sleep(2 ** attempt)
        if not data or isinstance(data, dict):
            break
        all_rows.extend(data)
        cur_ms = data[-1][0] + 60000   # 다음 시작점
        time.sleep(0.1)                # rate limit 방지

    if not all_rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")["close"].astype(float)
    df = df[~df.index.duplicated()]
    return df

def load_period(start: pd.Timestamp, end: pd.Timestamp,
                label: str = "") -> pd.DataFrame:
    print(f"  [{label}] {start.date()} ~ {end.date()} 로딩 중...")
    frames = {}
    for sym in SYMBOLS:
        s = fetch_klines(sym, start, end)
        if len(s) == 0:
            print(f"    {sym}: 데이터 없음")
            continue
        frames[sym] = s
        print(f"    {sym}: {len(s)}개")

    if not frames:
        return pd.DataFrame()
    # ffill로 단기 갭 메우고, 그래도 남은 NaN만 제거
    df = pd.DataFrame(frames).ffill().dropna()
    # 학습 때와 동일하게 SYMBOLS 순서로 컬럼 강제 정렬 (없는 심볼은 0으로 채움)
    for sym in SYMBOLS:
        if sym not in df.columns:
            print(f"    {sym}: 누락 -- 0으로 대체")
            df[sym] = df.iloc[:, 0]  # 첫 번째 심볼 복사 후 0 수익률 유도
    df = df[SYMBOLS]
    return df

# ═══════════════════════════════════════════════════════════════
# 3. 전처리
# ═══════════════════════════════════════════════════════════════
def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def make_windows(returns: np.ndarray) -> np.ndarray:
    n = len(returns)
    return np.array([returns[i - WINDOW_SIZE:i]
                     for i in range(WINDOW_SIZE, n)], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
# 4. LSTM-VAE (v1과 동일)
# ═══════════════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_var(h)

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        return self.out(out)

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
        return self.decoder(z), mu, log_var

def vae_loss(recon, x, mu, log_var, kl_weight):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl = torch.clamp(kl_per_dim, min=FREE_BITS).mean()
    return recon_loss + kl_weight * kl, recon_loss.item(), kl.item()

# ═══════════════════════════════════════════════════════════════
# 5. 학습
# ═══════════════════════════════════════════════════════════════
def train_vae(windows: np.ndarray) -> tuple:
    scaler = StandardScaler()
    shape  = windows.shape
    X_scaled = scaler.fit_transform(
        windows.reshape(-1, shape[-1])
    ).reshape(shape)

    tensor = torch.FloatTensor(X_scaled)
    loader = DataLoader(TensorDataset(tensor),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMVAE(len(SYMBOLS), HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"  학습 중 ({len(windows)}개 윈도우, {EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        kl_w = min(KL_WEIGHT_MAX, KL_WEIGHT_MAX * (epoch + 1) / KL_ANNEAL)
        total_loss = total_recon = total_kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, r, k = vae_loss(recon, batch, mu, log_var, kl_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon += r
            total_kl    += k
        if (epoch + 1) % 10 == 0:
            n = len(loader)
            print(f"    Epoch {epoch+1}/{EPOCHS}  "
                  f"recon={total_recon/n:.4f}  kl={total_kl/n:.4f}  kl_w={kl_w:.3f}")
    return model, scaler

# ═══════════════════════════════════════════════════════════════
# 6. 이상 점수 계산
# ═══════════════════════════════════════════════════════════════
def compute_scores(model, windows, scaler) -> np.ndarray:
    model.eval()
    shape    = windows.shape
    X_scaled = scaler.transform(
        windows.reshape(-1, shape[-1])
    ).reshape(shape)
    tensor = torch.FloatTensor(X_scaled).to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[i:i + BATCH_SIZE]
            recon, mu, log_var = model(batch)
            err = F.mse_loss(recon, batch, reduction='none').mean(dim=(1, 2))
            kl  = torch.clamp(
                -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()),
                min=FREE_BITS
            ).mean(dim=1)
            scores.append((err + KL_WEIGHT_MAX * kl).cpu().numpy())
    return np.concatenate(scores)

# ═══════════════════════════════════════════════════════════════
# 7. 감지 분석
# ═══════════════════════════════════════════════════════════════
def analyze_crash(crash_name, crash_time, score_series: pd.Series,
                  baseline_mu, baseline_sigma) -> dict:
    pre_start  = crash_time - pd.Timedelta(minutes=CONTEXT)
    post_end   = crash_time + pd.Timedelta(minutes=CONTEXT)
    detect_end = crash_time + pd.Timedelta(minutes=DETECT_WIN)

    pre  = score_series[(score_series.index >= pre_start) &
                         (score_series.index < crash_time)]
    post = score_series[(score_series.index >= crash_time) &
                         (score_series.index < post_end)]
    det  = score_series[(score_series.index >= crash_time) &
                         (score_series.index < detect_end)]

    if len(pre) < 10 or len(post) < 10:
        print(f"  [{crash_name}] 데이터 부족: pre={len(pre)}, post={len(post)}")
        return None

    # 3-sigma 초과 감지
    threshold    = baseline_mu + SIGMA_THRESH * baseline_sigma
    first_detect = None
    for t, v in det.items():
        if v > threshold:
            first_detect = t
            break
    detect_lag = int((first_detect - crash_time).total_seconds() / 60) \
                 if first_detect else None

    # Mann-Whitney: 사전 vs 사후
    _, u_p = mannwhitneyu(post.values, pre.values, alternative='greater')

    result = dict(
        crash=crash_name, crash_time=crash_time,
        pre_mean=pre.mean(), post_mean=post.mean(),
        threshold=threshold,
        detect_lag=detect_lag,
        mann_whitney_p=u_p,
        pre=pre, post=post, det=det,
    )

    detected = detect_lag is not None and detect_lag <= DETECT_WIN
    print(f"\n  [{crash_name}]")
    print(f"    기준치: {threshold:.4f} (mu+{SIGMA_THRESH}sigma)")
    print(f"    사전 평균: {pre.mean():.4f}  사후 평균: {post.mean():.4f}")
    print(f"    감지 지연: {detect_lag}분" if detect_lag else "    감지: 없음")
    print(f"    Mann-Whitney p = {u_p:.4f} {'[POST>PRE]' if u_p < 0.05 else '[NO DIFF]'}")
    print(f"    판정: {'[DETECTED]' if detected else '[MISSED]'}")
    return result

# ═══════════════════════════════════════════════════════════════
# 8. 시각화
# ═══════════════════════════════════════════════════════════════
def plot_results(results: list):
    valid = [r for r in results if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(len(valid), 1, figsize=(14, 4 * len(valid)))
    if len(valid) == 1:
        axes = [axes]

    for ax, r in zip(axes, valid):
        combined = pd.concat([r["pre"], r["post"]]).sort_index()
        ax.plot(combined.index, combined.values, color='steelblue', lw=0.8)
        ax.fill_between(r["pre"].index, r["pre"].values,
                        alpha=0.2, color='green', label='Pre-crash')
        ax.fill_between(r["post"].index, r["post"].values,
                        alpha=0.2, color='red', label='Post-crash')
        ax.axvline(r["crash_time"], color='red', lw=2, linestyle='--', label='Crash')
        ax.axhline(r["threshold"], color='orange', lw=1.5,
                   linestyle=':', label=f'{SIGMA_THRESH}-sigma threshold')

        lag  = r["detect_lag"]
        u_p  = r["mann_whitney_p"]
        det  = f"lag={lag}min" if lag is not None else "MISSED"
        sig  = "[O]" if u_p < 0.05 else "[X]"
        ax.set_title(f"{r['crash']} | {det} | MW_p={u_p:.4f} {sig}")
        ax.set_ylabel("Anomaly Score")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    plt.tight_layout()
    plt.savefig("vae_validation_v2.png", dpi=150, bbox_inches='tight')
    print("\n그래프 저장: vae_validation_v2.png")

# ═══════════════════════════════════════════════════════════════
# 9. 요약
# ═══════════════════════════════════════════════════════════════
def print_summary(results: list):
    valid = [r for r in results if r is not None]
    print("\n" + "=" * 65)
    print("VAE 검증 실험 v2 (분봉) -- 최종 판정")
    print("=" * 65)

    detected  = [r for r in valid if r["detect_lag"] is not None
                 and r["detect_lag"] <= DETECT_WIN]
    mw_sig    = [r for r in valid if r["mann_whitney_p"] < 0.05]

    print(f"60분 내 감지:            {len(detected)}/{len(valid)}")
    print(f"Mann-Whitney 유의:       {len(mw_sig)}/{len(valid)}")
    print()

    for r in valid:
        lag = r["detect_lag"]
        det = f"lag={lag}min" if lag is not None else "MISSED"
        sig = "[O]" if r["mann_whitney_p"] < 0.05 else "[X]"
        print(f"  {r['crash']:22s} {det:12s} MW_p={r['mann_whitney_p']:.4f} {sig}")

    print("\n[ 최종 판정 ]")
    if len(detected) >= 2 and len(mw_sig) >= 2:
        print("  SIGNAL -- 실시간 감지 확인. DA-VAE 업그레이드 진행")
    elif len(detected) >= 1 or len(mw_sig) >= 1:
        print("  WEAK -- 부분 감지. 파라미터 튜닝 또는 DA-VAE 검토")
    else:
        print("  NULL -- 분봉에서도 감지 없음. 접근 재고")
    print("=" * 65)

    with open("summary_vae_v2.txt", "w", encoding="utf-8") as f:
        f.write("VAE 검증 실험 v2 (분봉)\n")
        f.write(f"60분 내 감지: {len(detected)}/{len(valid)}\n")
        f.write(f"Mann-Whitney: {len(mw_sig)}/{len(valid)}\n\n")
        for r in valid:
            lag = r["detect_lag"]
            det = f"lag={lag}min" if lag is not None else "MISSED"
            f.write(f"{r['crash']:22s} {det:12s} "
                    f"MW_p={r['mann_whitney_p']:.4f}\n")

# ═══════════════════════════════════════════════════════════════
# 10. 메인
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("VAE 검증 실험 v2 (분봉) 시작")
    print("=" * 65)
    print(f"Device: {device}")

    import pickle, os
    MODEL_PATH   = "vae_v2_model.pt"
    SCALER_PATH  = "vae_v2_scaler.pkl"
    DATA_PATH    = "vae_v2_normal_windows.npy"
    BASELINE_PATH = "vae_v2_baseline.pkl"

    # [1] 정상 구간 데이터 (캐시 우선)
    print("\n[1] 정상 구간 데이터 로딩...")
    if os.path.exists(DATA_PATH):
        print("  캐시 로드 중...")
        normal_win = np.load(DATA_PATH)
        print(f"  학습 윈도우: {len(normal_win)}개 (캐시)")
    else:
        normal_prices = load_period(
            NORMAL_PERIOD["start"], NORMAL_PERIOD["end"], "Normal 2021-H1"
        )
        if normal_prices.empty:
            print("정상 데이터 로딩 실패. 종료.")
            exit(1)
        print(f"  정상 구간: {len(normal_prices)}분")
        normal_ret = to_log_returns(normal_prices)
        normal_win = make_windows(normal_ret.values)
        np.save(DATA_PATH, normal_win)
        print(f"  학습 윈도우: {len(normal_win)}개 (저장 완료)")

    # [2] VAE 학습 (캐시 우선)
    print("\n[2] VAE 학습...")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("  저장된 모델 로드 중...")
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        model  = LSTMVAE(len(SYMBOLS), HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("  로드 완료.")
    else:
        model, scaler = train_vae(normal_win)
        torch.save(model.state_dict(), MODEL_PATH)
        pickle.dump(scaler, open(SCALER_PATH, "wb"))
        print("  모델 저장 완료.")

    print("\n[3] 정상 기준치 계산...")
    baseline_scores = compute_scores(model, normal_win, scaler)
    baseline_mu     = baseline_scores.mean()
    baseline_sigma  = baseline_scores.std()
    print(f"  기준치: mu={baseline_mu:.4f}, sigma={baseline_sigma:.4f}")
    print(f"  3-sigma 임계값: {baseline_mu + 3*baseline_sigma:.4f}")

    # 크래시별 분석
    results = []
    for crash_name, crash_time in CRASH_EVENTS.items():
        print(f"\n[4] {crash_name} 분석 중...")
        fetch_start = crash_time - pd.Timedelta(minutes=CONTEXT + WINDOW_SIZE + 10)
        fetch_end   = crash_time + pd.Timedelta(minutes=CONTEXT + 10)

        crash_cache = f"vae_v2_crash_{crash_name.replace(' ', '_')}.pkl"
        if os.path.exists(crash_cache):
            print(f"  캐시 로드 중...")
            score_series = pickle.load(open(crash_cache, "rb"))
        else:
            prices = load_period(fetch_start, fetch_end, crash_name)
            if prices.empty:
                print(f"  데이터 없음 -- 스킵")
                results.append(None)
                continue
            ret     = to_log_returns(prices)
            windows = make_windows(ret.values)
            if len(windows) == 0:
                results.append(None)
                continue
            win_dates    = ret.index[WINDOW_SIZE:]
            scores       = compute_scores(model, windows, scaler)
            score_series = pd.Series(scores, index=win_dates)
            pickle.dump(score_series, open(crash_cache, "wb"))

        res = analyze_crash(crash_name, crash_time, score_series,
                            baseline_mu, baseline_sigma)
        results.append(res)

    plot_results(results)
    print_summary(results)
