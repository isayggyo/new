"""
Deep SAD v3 — IV Skew + Term Structure (Dzhafarov 방식)
========================================================
v2 결론: DVOL level NULL
v3 목표: IV 구조적 피처 (skew, term structure, gradient) 추가

추가 피처 (7개):
  - atm_iv_front: 전월 ATM IV (1개월 만기)
  - atm_iv_back:  후월 ATM IV (3개월 만기)
  - term_slope:   atm_front / atm_back - 1  (역전 = 공포 신호)
  - rr_skew:      IV_OTM_put - IV_OTM_call  (풋 수요 과잉 시 양수)
  - butterfly:    (IV_OTM_put + IV_OTM_call) / 2 - ATM_IV
  - delta_rr:     5일 RR 변화율 (skew 가속도, Dzhafarov 핵심)
  - delta_term:   5일 term structure 변화율

데이터: Deribit 월말 금요일 옵션 만기, BTC 가격 기준 ±15% moneyness
BS 역산: price_btc × spot_usd → IV (r=0, European)
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle, time, json, calendar
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import brentq
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

SPLIT_DATE = pd.Timestamp("2022-07-01")

CRASH_EVENTS = {"2022 FTX Collapse": pd.Timestamp("2022-11-08")}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DERIBIT = "https://www.deribit.com/api/v2/public"
OPTION_CACHE = "option_iv_surface.pkl"


# ══════════════════════════════════════════════════════════
# 1. BTC 가격 로드 (IV 계산에 필요)
# ══════════════════════════════════════════════════════════
print("BTC 가격 로드...")
raw_btc = yf.download("BTC-USD", start="2021-01-01", end=END,
                      auto_adjust=True, progress=False)
btc_close = raw_btc["Close"]
if isinstance(btc_close, pd.DataFrame):
    btc_close = btc_close.iloc[:, 0]
btc_close = btc_close.dropna()
print(f"  {len(btc_close)}일, {btc_close.index.min().date()} ~ {btc_close.index.max().date()}")


# ══════════════════════════════════════════════════════════
# 2. IV Surface 피처 빌더
# ══════════════════════════════════════════════════════════
def last_friday(year: int, month: int) -> pd.Timestamp:
    last_day = calendar.monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_day)
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d


def deribit_date(dt: pd.Timestamp) -> str:
    months = ["JAN","FEB","MAR","APR","MAY","JUN",
              "JUL","AUG","SEP","OCT","NOV","DEC"]
    return f"{dt.day}{months[dt.month-1]}{str(dt.year)[2:]}"


def fetch_ohlcv(instrument: str, start: pd.Timestamp, end: pd.Timestamp):
    """1D OHLCV for a Deribit instrument. Price in BTC."""
    params = {
        "instrument_name": instrument,
        "resolution":      "1D",
        "start_timestamp": int(start.timestamp() * 1000),
        "end_timestamp":   int(end.timestamp() * 1000),
    }
    try:
        r = requests.get(f"{DERIBIT}/get_tradingview_chart_data",
                         params=params, timeout=10)
        res = r.json().get("result", {})
        if not res.get("ticks"):
            return None
        df = pd.DataFrame({"ts": res["ticks"], "close_btc": res["close"]})
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        return df.set_index("date")[["close_btc"]]
    except Exception:
        return None


def bs_price(S, K, T, sigma, opt_type="C"):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, S - K) if opt_type == "C" else max(0.0, K - S)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "C":
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price_btc: float, spot: float, K: float,
                T: float, opt_type: str = "C") -> float:
    price_usd = price_btc * spot
    intrinsic = max(0.0, spot - K) if opt_type == "C" else max(0.0, K - spot)
    if price_usd <= intrinsic or T <= 1e-6:
        return np.nan
    try:
        f = lambda s: bs_price(spot, K, T, s, opt_type) - price_usd
        if f(0.001) * f(10.0) > 0:
            return np.nan
        return brentq(f, 0.001, 10.0, maxiter=100)
    except Exception:
        return np.nan


def round_strike(price: float) -> int:
    """Deribit 호환 strike 반올림 ($1000 단위)"""
    return int(round(price / 1000.0) * 1000)


def build_iv_surface() -> pd.DataFrame:
    """
    월말 금요일 만기 옵션에서 ATM, OTM±15% IV 추출
    -> term structure + skew 피처 생성
    """
    # 월말 금요일 만기 목록 생성 (2021-03 ~ 2023-12)
    expiries = []
    for y in range(2021, 2024):
        for m in range(1, 13):
            if y == 2021 and m < 3:
                continue
            expiries.append(last_friday(y, m))
    print(f"타겟 만기 {len(expiries)}개: {expiries[0].date()} ~ {expiries[-1].date()}")

    # 만기별 BTC 평균가격 (만기 전 30일)
    expiry_spot = {}
    for exp in expiries:
        window = btc_close[(btc_close.index >= exp - pd.Timedelta(days=35)) &
                            (btc_close.index <= exp)]
        if len(window) > 0:
            expiry_spot[exp] = float(window.mean())

    # 각 만기에 대해 4개 instrument 데이터 수집
    # (ATM call/put, OTM85% put, OTM115% call)
    inst_cache = {}   # {instrument_name: DataFrame}

    print("Deribit 옵션 데이터 수집 중...")
    for i, exp in enumerate(expiries):
        spot = expiry_spot.get(exp, 30000)
        # target strikes
        strikes = {
            "atm":  round_strike(spot),
            "put85": round_strike(spot * 0.85),
            "call115": round_strike(spot * 1.15),
        }
        exp_str = deribit_date(exp)
        fetch_start = exp - pd.Timedelta(days=95)
        fetch_end   = exp + pd.Timedelta(days=1)

        for label, K in strikes.items():
            for opt_type in ["C", "P"]:
                if label == "atm":
                    name = f"BTC-{exp_str}-{K}-{opt_type}"
                elif label == "put85":
                    name = f"BTC-{exp_str}-{K}-P"
                elif label == "call115":
                    name = f"BTC-{exp_str}-{K}-C"
                else:
                    continue

                if label in ["put85"] and opt_type == "C":
                    continue
                if label in ["call115"] and opt_type == "P":
                    continue

                if name not in inst_cache:
                    df = fetch_ohlcv(name, fetch_start, fetch_end)
                    inst_cache[name] = df
                    time.sleep(0.08)   # rate limit

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(expiries)} 만기 처리 완료")

    print(f"수집된 instrument: {sum(1 for v in inst_cache.values() if v is not None)}/"
          f"{len(inst_cache)}개")

    # IV 계산 — 날짜별 surface 피처
    all_dates = pd.date_range(start=START_DVOL, end=END, freq='B')
    records   = {}

    for date in all_dates:
        spot_val = btc_close.get(date, None)
        if spot_val is None:
            idx = btc_close.index.get_indexer([date], method='nearest')[0]
            if idx < 0:
                continue
            spot_val = float(btc_close.iloc[idx])

        # 해당 날짜로부터 가장 가까운 front/back expiry 찾기
        future_expiries = [e for e in expiries if e > date]
        if len(future_expiries) < 2:
            continue

        front_exp = min(future_expiries, key=lambda e: abs((e - date).days - 30))
        back_exp  = min(future_expiries, key=lambda e: abs((e - date).days - 90))

        if front_exp == back_exp:
            remaining = [e for e in future_expiries if e != front_exp]
            if not remaining:
                continue
            back_exp = min(remaining, key=lambda e: abs((e - date).days - 90))

        def get_iv(exp, target_mon, opt_type):
            K = round_strike(spot_val * target_mon)
            exp_str = deribit_date(exp)
            name = f"BTC-{exp_str}-{K}-{opt_type}"
            df = inst_cache.get(name)
            if df is None or date not in df.index:
                return np.nan
            price_btc = float(df.loc[date, "close_btc"])
            if price_btc <= 0:
                return np.nan
            T = max((exp - date).days / 365.0, 1e-6)
            return implied_vol(price_btc, spot_val, K, T, opt_type)

        atm_f  = get_iv(front_exp, 1.00, "C")
        put_f  = get_iv(front_exp, 0.85, "P")
        call_f = get_iv(front_exp, 1.15, "C")
        atm_b  = get_iv(back_exp,  1.00, "C")

        if np.isnan(atm_f) and not np.isnan(get_iv(front_exp, 1.00, "P")):
            atm_f = get_iv(front_exp, 1.00, "P")

        rr_skew   = (put_f  - call_f) if not (np.isnan(put_f) or np.isnan(call_f)) else np.nan
        butterfly = ((put_f + call_f) / 2 - atm_f) if not np.isnan(atm_f) else np.nan
        term_slope = (atm_f / atm_b - 1) if not (np.isnan(atm_f) or np.isnan(atm_b) or atm_b < 1e-6) else np.nan

        records[date] = {
            "atm_iv_front": atm_f,
            "atm_iv_back":  atm_b,
            "rr_skew":      rr_skew,
            "butterfly":    butterfly,
            "term_slope":   term_slope,
        }

    iv_df = pd.DataFrame.from_dict(records, orient='index')
    iv_df.index.name = 'date'

    # 최대 5일 ffill (옵션이 거래 안 된 날 보간)
    iv_df = iv_df.ffill(limit=5)

    # gradient 피처 (ffill 후 계산)
    iv_df["delta_rr"]   = iv_df["rr_skew"].pct_change(5)
    iv_df["delta_term"] = iv_df["term_slope"].pct_change(5)

    valid = iv_df.dropna(how='all')
    print(f"\nIV surface: {len(valid)}행 유효 데이터 (ffill 적용)")
    print(f"  비율: atm_iv_front {iv_df['atm_iv_front'].notna().mean()*100:.0f}%, "
          f"rr_skew {iv_df['rr_skew'].notna().mean()*100:.0f}%")
    return iv_df


if os.path.exists(OPTION_CACHE):
    print(f"IV surface 캐시 로드: {OPTION_CACHE}")
    with open(OPTION_CACHE, 'rb') as f:
        iv_surface = pickle.load(f)
else:
    iv_surface = build_iv_surface()
    with open(OPTION_CACHE, 'wb') as f:
        pickle.dump(iv_surface, f)
    print(f"IV surface 캐시 저장: {OPTION_CACHE}")


# ══════════════════════════════════════════════════════════
# 3. OHLCV + DVOL + IV Surface 결합
# ══════════════════════════════════════════════════════════
feat_v2_cache = "sad_features_v2.pkl"
if os.path.exists(feat_v2_cache):
    with open(feat_v2_cache, 'rb') as f:
        feat_base = pickle.load(f)
else:
    raise FileNotFoundError("sad_features_v2.pkl 없음 — deep_sad_v2.py 먼저 실행")

# IV surface를 feat_base 날짜에 align + ffill
iv_aligned = iv_surface.reindex(feat_base.index, method='ffill')

feat_full = pd.concat([feat_base, iv_aligned], axis=1)

# IV 피처가 없는 날은 제거 (atm_iv_front 기준)
iv_cols  = ["atm_iv_front","atm_iv_back","rr_skew","butterfly",
            "term_slope","delta_rr","delta_term"]
feat_iv  = feat_full.dropna(subset=["atm_iv_front"])
print(f"\nIV 결합 후 행수: {len(feat_iv)}일")
print(f"  기간: {feat_iv.index.min().date()} ~ {feat_iv.index.max().date()}")
print(f"  총 피처: {feat_iv.shape[1]}개")

# IV 데이터 있는 날의 FTX 주변 IV 구조 확인
ftx = pd.Timestamp("2022-11-08")
ftx_window = feat_iv.loc[ftx - pd.Timedelta(days=30): ftx + pd.Timedelta(days=10), iv_cols]
print(f"\nFTX 전후 IV 구조 (rr_skew, term_slope 행 수: {len(ftx_window)})")


# ══════════════════════════════════════════════════════════
# 4. 슬라이딩 윈도우 + 레이블 (전체 피처)
# ══════════════════════════════════════════════════════════
def make_dataset(feat_df, win=WIN):
    scaler  = StandardScaler()
    # NaN: ffill → bfill → 0 (평균 대체) 순서로 처리
    feat_clean = feat_df.ffill().bfill().fillna(0)
    X_sc    = scaler.fit_transform(feat_clean.values)
    wins, dates = [], []
    for i in range(win, len(X_sc)):
        wins.append(X_sc[i - win:i])
        dates.append(feat_df.index[i])
    wins   = np.array(wins, dtype=np.float32)
    dates  = pd.DatetimeIndex(dates)
    # anomaly label: top 5% btc_vol + crash ±3d
    last_vol  = feat_df["btc_vol"].values[win:]
    thresh    = np.percentile(last_vol, (1 - ANOM_PCT) * 100)
    labels    = np.where(last_vol >= thresh, -1, 1).astype(np.float32)
    crash_idx = dates.get_indexer([ftx], method='nearest')[0]
    for offset in range(-3, 4):
        j = crash_idx + offset
        if 0 <= j < len(labels):
            labels[j] = -1.0
    return wins, dates, labels, scaler


wins_C, dates_C, labels_C, _ = make_dataset(feat_iv)

split_idx = int(np.searchsorted(dates_C, SPLIT_DATE))
X_tr_C = torch.FloatTensor(wins_C[:split_idx]).to(device)
X_te_C = torch.FloatTensor(wins_C[split_idx:]).to(device)
y_tr_C = torch.FloatTensor(labels_C[:split_idx]).to(device)
y_te_C = labels_C[split_idx:]

print(f"\nDataset C (OHLCV+DVOL+IVS): {len(wins_C)}개 windows")
print(f"  Train: {dates_C[:split_idx][0].date()} ~ {dates_C[:split_idx][-1].date()}")
print(f"  Test:  {dates_C[split_idx:][0].date()} ~ {dates_C[split_idx:][-1].date()}")
print(f"  Anomaly: {(labels_C==-1).sum()} / {len(labels_C)}")

n_anom_test = (y_te_C == -1).sum()
print(f"  Test anomaly: {n_anom_test}")
if n_anom_test == 0:
    print("  [경고] Test set에 anomaly 없음 - AUROC 계산 불가")


# ══════════════════════════════════════════════════════════
# 5. 모델 + 학습 헬퍼
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


def run_experiment(X_tr, X_te, y_tr_t, y_te,
                   input_dim, tag, ae_path, enc_path):
    ae  = LSTMAutoEncoder(input_dim, HIDDEN_DIM, LATENT_DIM, WIN).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)

    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        print(f"  {tag} AE 로드")
    else:
        print(f"  {tag} AE 학습 ({AE_EPOCHS} epochs)...")
        for epoch in range(AE_EPOCHS):
            ae.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), BATCH_SIZE):
                xb = X_tr[perm[i:i+BATCH_SIZE]]
                recon, _ = ae(xb)
                loss = F.mse_loss(recon, xb)
                opt.zero_grad(); loss.backward(); opt.step()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}")
        torch.save(ae.state_dict(), ae_path)

    # center
    ae.eval()
    with torch.no_grad():
        zs = [ae(X_tr[i:i+BATCH_SIZE])[1] for i in range(0, len(X_tr), BATCH_SIZE)]
    c = torch.cat(zs).mean(0)
    c = torch.where(c.abs() < 1e-4, torch.full_like(c, 1e-4), c)

    enc = ae.encoder
    opt_sad = torch.optim.Adam(enc.parameters(), lr=LR * 0.1)

    if os.path.exists(enc_path):
        enc.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"  {tag} Enc 로드")
    else:
        print(f"  {tag} SAD 학습 ({SAD_EPOCHS} epochs)...")
        for epoch in range(SAD_EPOCHS):
            enc.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), BATCH_SIZE):
                idx  = perm[i:i+BATCH_SIZE]
                xb, yb = X_tr[idx], y_tr_t[idx]
                z    = enc(xb)
                dist = torch.sum((z - c.detach()) ** 2, dim=1)
                mask = yb < 0
                parts = []
                if (~mask).any(): parts.append(dist[~mask])
                if   mask.any(): parts.append(ETA / (dist[mask] + 1e-6))
                loss = torch.cat(parts).mean()
                opt_sad.zero_grad(); loss.backward(); opt_sad.step()
            if epoch % 30 == 0:
                print(f"    SAD Epoch {epoch:3d}")
        torch.save(enc.state_dict(), enc_path)

    # score
    enc.eval()
    c_cpu = c.detach().cpu()
    def score(X_t):
        sc = []
        with torch.no_grad():
            for i in range(0, len(X_t), BATCH_SIZE):
                z = enc(X_t[i:i+BATCH_SIZE].to(device)).cpu()
                sc.append(torch.sum((z - c_cpu)**2, dim=1).numpy())
        return np.concatenate(sc)

    sc_tr = score(X_tr)
    sc_te = score(X_te)
    return sc_tr, sc_te, c


# ══════════════════════════════════════════════════════════
# 6. 실험 실행
# ══════════════════════════════════════════════════════════
# 공정 비교: 동일 날짜 범위에서 OHLCV only vs OHLCV+DVOL+IVS
ohlcv_cols = ["btc_ret","eth_ret","btc_vol","btc_mom5","btc_mom20",
              "vol_ratio","btc_eth_corr","vol_accel"]
feat_ohlcv = feat_iv[ohlcv_cols]
wins_A, dates_A, labels_A, _ = make_dataset(feat_ohlcv)

split_A = int(np.searchsorted(dates_A, SPLIT_DATE))
X_trA = torch.FloatTensor(wins_A[:split_A]).to(device)
X_teA = torch.FloatTensor(wins_A[split_A:]).to(device)
y_trA = torch.FloatTensor(labels_A[:split_A]).to(device)
y_teA = labels_A[split_A:]

print("\n" + "="*50)
print("실험 A: OHLCV only (같은 기간)")
print("="*50)
scA_tr, scA_te, _ = run_experiment(
    X_trA, X_teA, y_trA, y_teA, 8,
    "A", "sad_v3_ae_A.pt", "sad_v3_enc_A.pt")

print("\n" + "="*50)
print("실험 C: OHLCV + DVOL + IVS (13+7=20 features)")
print("="*50)
scC_tr, scC_te, _ = run_experiment(
    X_tr_C, X_te_C, y_tr_C, y_te_C, feat_iv.shape[1],
    "C", "sad_v3_ae_C.pt", "sad_v3_enc_C.pt")


# ══════════════════════════════════════════════════════════
# 7. 평가
# ══════════════════════════════════════════════════════════
# FTX 탐지
scA_all = np.concatenate([scA_tr, scA_te])
scC_all = np.concatenate([scC_tr, scC_te])

idx_ftx = dates_C.get_indexer([ftx], method='nearest')[0]

pct_A = float((scA_all < scA_all[idx_ftx]).mean() * 100)
pct_C = float((scC_all < scC_all[idx_ftx]).mean() * 100)

print(f"\nFTX Collapse 이상 점수:")
print(f"  OHLCV only:        score={scA_all[idx_ftx]:.4f}  상위 {100-pct_A:.1f}%ile")
print(f"  OHLCV+DVOL+IVS:    score={scC_all[idx_ftx]:.4f}  상위 {100-pct_C:.1f}%ile")

# AUROC (test set에 anomaly 있는 경우만)
y_bin_A = (y_teA == -1).astype(int)
y_bin_C = (y_te_C == -1).astype(int)

if y_bin_A.sum() > 0:
    auroc_A = roc_auc_score(y_bin_A, scA_te)
    print(f"\nTest AUROC A: {auroc_A:.4f}")
else:
    print("\nTest AUROC A: N/A (anomaly 없음)")
    auroc_A = None

if y_bin_C.sum() > 0:
    valid_mask = ~np.isnan(scC_te)
    if valid_mask.sum() > 0 and y_bin_C[valid_mask].sum() > 0:
        auroc_C = roc_auc_score(y_bin_C[valid_mask], scC_te[valid_mask])
        print(f"Test AUROC C: {auroc_C:.4f}")
    else:
        print("Test AUROC C: N/A (NaN 또는 anomaly 없음)")
        auroc_C = None
else:
    print("Test AUROC C: N/A (anomaly 없음)")
    auroc_C = None


# ══════════════════════════════════════════════════════════
# 8. FTX 전후 IV Structure 시각화 (Dzhafarov 핵심)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(14, 16))

ftx_win_start = ftx - pd.Timedelta(days=60)
ftx_win_end   = ftx + pd.Timedelta(days=15)

def plot_iv_feature(ax, col, label, color, title):
    s = feat_iv[col].dropna()
    s = s[(s.index >= ftx_win_start) & (s.index <= ftx_win_end)]
    if len(s) == 0:
        ax.text(0.5, 0.5, f'{col}: 데이터 없음', transform=ax.transAxes,
                ha='center', va='center')
        return
    ax.plot(s.index, s.values, color=color, lw=1.0, label=label)
    ax.axvline(ftx, color='red', lw=1.5, linestyle='--', label='FTX')
    ax.axhline(0, color='gray', lw=0.5, linestyle=':')
    ax.set_title(title); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plot_iv_feature(axes[0], "atm_iv_front", "ATM IV (front month)", "steelblue",
                "ATM IV Front Month (공포 수준)")
plot_iv_feature(axes[1], "term_slope", "Term Slope (front/back-1)", "purple",
                "Term Structure Slope (역전=양수=단기 공포)")
plot_iv_feature(axes[2], "rr_skew", "RR Skew (put-call IV)", "darkorange",
                "Risk Reversal Skew (양수=풋 수요 > 콜)")
plot_iv_feature(axes[3], "delta_rr", "Delta RR (5d chg)", "red",
                "RR Skew 5d 변화율 (Dzhafarov gradient)")

ax = axes[4]
sc_series = pd.Series(scC_all, index=dates_C)
sc_win = sc_series[(sc_series.index >= ftx_win_start) & (sc_series.index <= ftx_win_end)]
if len(sc_win) > 0:
    ax.plot(sc_win.index, sc_win.values, color='darkgreen', lw=1.0, label='SAD Score (C)')
    ax.axvline(ftx, color='red', lw=1.5, linestyle='--')
ax.set_title("Deep SAD Anomaly Score (OHLCV+DVOL+IVS)")
ax.set_ylabel("||z-c||²"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.savefig("deep_sad_v3.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: deep_sad_v3.png")


# ══════════════════════════════════════════════════════════
# 9. 최종 판정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Deep SAD v3 -- IV Structure 효과")
print("=" * 60)
print(f"FTX percentile A (OHLCV):          {100-pct_A:.1f}%ile")
print(f"FTX percentile C (OHLCV+DVOL+IVS): {100-pct_C:.1f}%ile")

if auroc_A and auroc_C:
    print(f"\nTest AUROC A: {auroc_A:.4f}")
    print(f"Test AUROC C: {auroc_C:.4f}  (delta={auroc_C-auroc_A:+.4f})")

# IV structure 구조적 관찰
rr_before  = feat_iv.loc[(feat_iv.index >= ftx - pd.Timedelta(days=30)) &
                          (feat_iv.index < ftx), "rr_skew"].dropna()
rr_normal  = feat_iv.loc[feat_iv.index < ftx - pd.Timedelta(days=30), "rr_skew"].dropna()
if len(rr_before) > 0 and len(rr_normal) > 0:
    rr_z = (rr_before.mean() - rr_normal.mean()) / (rr_normal.std() + 1e-9)
    print(f"\nFTX 전 30일 RR skew z-score: {rr_z:.2f}")
    print(f"  (|z| > 2.0 = 구조적 이상)")

term_before = feat_iv.loc[(feat_iv.index >= ftx - pd.Timedelta(days=30)) &
                           (feat_iv.index < ftx), "term_slope"].dropna()
term_normal = feat_iv.loc[feat_iv.index < ftx - pd.Timedelta(days=30), "term_slope"].dropna()
if len(term_before) > 0 and len(term_normal) > 0:
    term_z = (term_before.mean() - term_normal.mean()) / (term_normal.std() + 1e-9)
    print(f"FTX 전 30일 term slope z-score: {term_z:.2f}")

if pct_C >= 90:
    verdict = "IVS SIGNAL -- FTX 탐지. machine.py SafetyMonitor 통합 진행"
elif pct_C >= 75:
    verdict = "IVS WEAK -- 부분 개선. 피처 정제 필요"
else:
    verdict = "IVS NULL -- 구조적 IV 피처도 한계"

print(f"\n판정: {verdict}")
print("=" * 60)
