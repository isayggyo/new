"""
Binance 1h OHLCV 수집 + 인트라데이 HMM
"""
import warnings; warnings.filterwarnings('ignore')
import time, pickle
import numpy as np
import pandas as pd
import requests

# ── Binance 1h 수집 ────────────────────────────────────────
CACHE = "btc_1h.pkl"

def fetch_binance_1h(symbol="BTCUSDT", start="2021-01-01", end="2024-12-31"):
    url    = "https://api.binance.com/api/v3/klines"
    ts     = int(pd.Timestamp(start).timestamp() * 1000)
    ts_end = int(pd.Timestamp(end).timestamp() * 1000)
    rows   = []
    while ts < ts_end:
        r = requests.get(url, params={
            "symbol": symbol, "interval": "1h",
            "startTime": ts, "limit": 1000
        }, timeout=10)
        data = r.json()
        if not data or isinstance(data, dict):
            break
        rows.extend(data)
        ts = data[-1][0] + 3_600_000   # 다음 캔들
        time.sleep(0.1)
        print(f"  {pd.Timestamp(ts, unit='ms').date()}  ({len(rows)} rows)", end='\r')
    df = pd.DataFrame(rows, columns=[
        'ts','open','high','low','close','volume',
        'close_ts','qvol','n_trades','taker_base','taker_quote','ignore'
    ])
    df['ts']    = pd.to_datetime(df['ts'], unit='ms')
    df          = df.set_index('ts')[['open','high','low','close','volume']].astype(float)
    df.index.name = 'datetime'
    return df

if __name__ == "__main__":
    import os
    if os.path.exists(CACHE):
        print(f"캐시 로드: {CACHE}")
        with open(CACHE,'rb') as f:
            df = pickle.load(f)
    else:
        print("Binance 1h 수집 중...")
        df = fetch_binance_1h()
        with open(CACHE,'wb') as f:
            pickle.dump(df, f)
        print(f"\n저장 완료: {len(df)} rows")

    print(f"\n기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"행 수: {len(df)}")

    # ── 피처 계산 ──────────────────────────────────────────
    df['ret']     = np.log(df['close'] / df['close'].shift(1))
    df['rv24']    = df['ret'].rolling(24).std() * np.sqrt(24*365)   # 연환산
    df['rv168']   = df['ret'].rolling(168).std() * np.sqrt(24*365)  # 1주
    df['rv_ratio']= df['rv24'] / (df['rv168'] + 1e-9)
    df['mom24']   = df['ret'].rolling(24).sum()
    df.dropna(inplace=True)

    # ── HMM ───────────────────────────────────────────────
    from hmmlearn.hmm import GaussianHMM

    feat_cols = ['ret', 'rv24', 'rv_ratio', 'mom24']
    X = df[feat_cols].values.astype(np.float64)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    hmm = GaussianHMM(n_components=2, covariance_type='full',
                      n_iter=200, random_state=42)
    hmm.fit(X_sc)
    states = hmm.predict(X_sc)

    # 고변동성 상태 식별
    means_rv = [X_sc[states == s, 1].mean() for s in range(2)]
    high_state = int(np.argmax(means_rv))

    post = hmm.predict_proba(X_sc)
    high_p_1h = pd.Series(post[:, high_state], index=df.index, name='p_high')

    print(f"\nHMM high state: {high_state}")
    print(f"P_high 평균: {high_p_1h.mean():.3f}")
    print(f"P_high > 0.8 비율: {(high_p_1h > 0.8).mean()*100:.1f}%")
    print(f"P_high > 0.5 비율: {(high_p_1h > 0.5).mean()*100:.1f}%")

    # ── DVOL 매핑 (일별 → 시간별) ─────────────────────────
    dvol = pd.read_csv("dvol_btc.csv", index_col=0, parse_dates=True)
    dvol_hourly = dvol['dvol'].reindex(df.index, method='ffill') / 100.0  # 연환산

    iv_1h = dvol_hourly
    rv_1h = df['rv24']

    # Q_high 계산
    lambda_h = (iv_1h / (rv_1h + 1e-9)).clip(0.5, 5.0)
    q_high_num = high_p_1h * lambda_h
    q_high_1h  = q_high_num / (q_high_num + (1 - high_p_1h))

    pq_gap = q_high_1h - high_p_1h

    print(f"\nQ_high 평균: {q_high_1h.mean():.3f}")
    print(f"P-Q 갭 평균: {pq_gap.mean():+.4f}")
    print(f"lambda_h 평균: {lambda_h.mean():.3f}")

    # 위기 국면 확인
    crises = {
        "2021-05 BTC -50%": ("2021-05-01","2021-05-31"),
        "2022-05 Luna":     ("2022-05-05","2022-05-20"),
        "2022-11 FTX":      ("2022-11-06","2022-11-16"),
        "2022-06 3AC":      ("2022-06-10","2022-06-20"),
    }
    print("\n=== 위기 국면 P_high ===")
    for name, (s, e) in crises.items():
        mask = (df.index >= s) & (df.index <= e)
        if mask.sum() > 0:
            ph = high_p_1h[mask].mean()
            qh = q_high_1h[mask].mean()
            lh = lambda_h[mask].mean()
            print(f"  {name:20s}  P={ph:.3f}  Q={qh:.3f}  lambda={lh:.2f}")

    # ── 저장 ──────────────────────────────────────────────
    result = {
        "df": df,
        "high_p_1h": high_p_1h,
        "q_high_1h": q_high_1h,
        "high_state": high_state,
        "lambda_h": lambda_h,
        "pq_gap": pq_gap,
    }
    with open("intraday_hmm.pkl",'wb') as f:
        pickle.dump(result, f)
    print("\n저장: intraday_hmm.pkl")
