"""
Breeden-Litzenberger Recheck — 디지털 옵션 방식
=================================================
2차 미분 대신 1차 미분 (디지털 옵션 가격):
  Q(S_T > K) = -dC/dK   (r=0)
  Q(S_T < K) =  dP/dK

꼬리 확률:
  Q_up   = Q(S_T > K_hi)  where K_hi = S * 1.15
  Q_down = Q(S_T < K_lo)  where K_lo = S * 0.85
  Q_tail = Q_up + Q_down
"""
import warnings; warnings.filterwarnings('ignore')
import time, pickle, os
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# ── Deribit 유틸 ──────────────────────────────────────────
BASE = "https://www.deribit.com/api/v2/public"

def deribit(method, **params):
    r = requests.get(f"{BASE}/{method}", params=params, timeout=15)
    return r.json().get("result", {})

def get_mark_price_on_date(instrument, date_str):
    ts = int(pd.Timestamp(date_str).timestamp() * 1000)
    res = deribit("get_tradingview_chart_data",
                  instrument_name=instrument,
                  start_timestamp=ts,
                  end_timestamp=ts + 86_400_000,
                  resolution="1D")
    if not res or "close" not in res or not res["close"]:
        return None
    return float(res["close"][0])

def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*norm.cdf(d2)

def bs_put(S, K, T, sigma):
    return bs_call(S, K, T, sigma) - S + K

def implied_vol(price_usd, S, K, T, opt_type='C'):
    intrinsic = max(S-K, 0.0) if opt_type=='C' else max(K-S, 0.0)
    if price_usd <= intrinsic + 0.5: return None
    fn = (lambda s: bs_call(S, K, T, s) - price_usd) if opt_type=='C' \
         else (lambda s: bs_put(S, K, T, s) - price_usd)
    try: return brentq(fn, 0.01, 30.0, xtol=1e-5)
    except: return None

# ── 캐시 로드 + 평온 날짜 보완 ─────────────────────────────
CACHE = "bl_chains.pkl"
with open(CACHE,'rb') as f:
    chains = pickle.load(f)

CALM_KEY = "평온 2022-01"
if CALM_KEY not in chains:
    print(f"평온 구간 fetch 중 ({CALM_KEY})...")
    date_str, expiry = "2022-01-15", "28JAN22"
    spot_ts = pd.Timestamp(date_str)
    with open("arena_val_features.pkl",'rb') as f:
        _, btc_price = pickle.load(f)
    idx  = btc_price.index.get_indexer([spot_ts], method='nearest')[0]
    spot = float(btc_price.iloc[idx])
    print(f"  spot=${spot:,.0f}")

    # 스트라이크: spot ± 40%, 2500 단위
    step = 2500
    lo   = int(round(spot * 0.60 / step) * step)
    hi   = int(round(spot * 1.40 / step) * step)
    strikes = list(range(lo, hi + step, step))

    chain = []
    for K in strikes:
        for opt in ['C','P']:
            instr = f"BTC-{expiry}-{K}-{opt}"
            p = get_mark_price_on_date(instr, date_str)
            if p and p > 0:
                chain.append({"strike": K, "type": opt,
                               "price_btc": p, "price_usd": p * spot})
        time.sleep(0.05)
    df_calm = pd.DataFrame(chain)
    print(f"  {CALM_KEY}: {len(df_calm)} options  "
          f"strikes {df_calm.strike.min() if len(df_calm)>0 else '-'} ~ "
          f"{df_calm.strike.max() if len(df_calm)>0 else '-'}")
    chains[CALM_KEY] = {"date": date_str, "expiry": expiry,
                        "spot": spot, "chain": df_calm}
    with open(CACHE,'wb') as f:
        pickle.dump(chains, f)
    print("  캐시 업데이트 완료")
else:
    print(f"평온 구간 캐시 존재: {CALM_KEY}")


# ── 만기일 맵 ─────────────────────────────────────────────
EXPIRY_DATE = {
    "29OCT21": "2021-10-29", "28MAY21": "2021-05-28",
    "27MAY22": "2022-05-27", "24JUN22": "2022-06-24",
    "25NOV22": "2022-11-25", "28JAN22": "2022-01-28",
}

# ── 디지털 옵션으로 Q_tail 계산 ────────────────────────────
THRESHOLD = 0.15

def compute_q_tail(chain_df, spot, date_str, expiry_code):
    """
    1단계: 각 옵션 → implied vol
    2단계: IV(K) 스플라인 피팅
    3단계: K_lo, K_hi에서 디지털 옵션 가격으로 Q 계산
           Q(S_T > K) = -dC/dK ≈ N(d2) using local IV
    """
    T = (pd.Timestamp(EXPIRY_DATE[expiry_code]) -
         pd.Timestamp(date_str)).days / 365.0
    if T <= 1/365: return None, None, None, None

    K_lo = spot * (1 - THRESHOLD)
    K_hi = spot * (1 + THRESHOLD)

    # OTM 옵션만 IV 계산 (더 정확)
    iv_points = {}
    for _, row in chain_df.iterrows():
        K    = float(row['strike'])
        ptype = row['type']
        pusd = float(row['price_usd'])
        # OTM: call for K>spot, put for K<spot, ATM 모두 허용
        use_call = (K >= spot * 0.95) and ptype == 'C'
        use_put  = (K <= spot * 1.05) and ptype == 'P'
        if not (use_call or use_put): continue
        iv = implied_vol(pusd, spot, K, T, ptype)
        if iv and 0.05 < iv < 20:
            iv_points[K] = iv

    if len(iv_points) < 3:
        return None, None, None, None

    K_arr  = np.array(sorted(iv_points.keys()), dtype=float)
    iv_arr = np.array([iv_points[k] for k in K_arr])

    # 로그-모네니스 공간에서 스플라인
    lm_arr = np.log(K_arr / spot)
    try:
        if len(K_arr) >= 5:
            spline = UnivariateSpline(lm_arr, iv_arr, k=min(3, len(K_arr)-1), s=len(K_arr)*0.02)
        else:
            spline = UnivariateSpline(lm_arr, iv_arr, k=1, s=0)
    except Exception:
        return None, None, None, None

    def get_iv(K):
        lm = np.log(K / spot)
        # 범위 밖은 가장 가까운 값으로 클램프
        lm_c = np.clip(lm, lm_arr[0], lm_arr[-1])
        return float(np.clip(spline(lm_c), 0.05, 20.0))

    # Q(S_T > K_hi) using digital call = N(d2) with local IV
    def q_above(K):
        iv = get_iv(K)
        d2 = (np.log(spot/K) - 0.5*iv**2*T) / (iv*np.sqrt(T))
        return float(norm.cdf(d2))   # Q(S_T > K) = N(d2) [r=0]

    # Q(S_T < K_lo) = 1 - Q(S_T > K_lo) = N(-d2)
    def q_below(K):
        iv = get_iv(K)
        d2 = (np.log(spot/K) - 0.5*iv**2*T) / (iv*np.sqrt(T))
        return float(norm.cdf(-d2))  # Q(S_T < K) = N(-d2)

    q_up   = q_above(K_hi)
    q_down = q_below(K_lo)
    q_tail = q_up + q_down

    return q_tail, q_down, q_up, iv_points


# ── HMM 데이터 ─────────────────────────────────────────────
with open("arena_val_hmm.pkl",'rb') as f:
    _, high_p_daily, _ = pickle.load(f)
with open("intraday_hmm.pkl",'rb') as f:
    intra = pickle.load(f)
high_p_1h = intra["high_p_1h"]


# ── 전체 분석 ─────────────────────────────────────────────
ORDER = ["평온 2022-01", "2021-05 급락", "Luna", "3AC", "FTX"]
results = []

print(f"\n{'날짜':20s} {'T(d)':>6} {'옵션수':>6} {'Q_tail':>8} "
      f"{'Q_down':>8} {'Q_up':>8} {'P_daily':>8}")
print("-" * 72)

for label in ORDER:
    if label not in chains: continue
    data     = chains[label]
    date_str = data['date']
    expiry   = data['expiry']
    spot     = data['spot']
    chain_df = data['chain']

    T_days = (pd.Timestamp(EXPIRY_DATE[expiry]) - pd.Timestamp(date_str)).days

    q_tail, q_down, q_up, iv_pts = compute_q_tail(chain_df, spot, date_str, expiry)

    # P_high
    ts = pd.Timestamp(date_str)
    idx_p = high_p_daily.index.get_indexer([ts], method='nearest')[0]
    p_daily = float(high_p_daily.iloc[idx_p])
    day_mask = high_p_1h.index.normalize() == ts.normalize()
    p_intra = float(high_p_1h[day_mask].mean()) if day_mask.sum() > 0 else float('nan')

    n_iv = len(iv_pts) if iv_pts else 0

    if q_tail is not None:
        print(f"  {label:18s} {T_days:6d} {n_iv:6d} {q_tail:8.3f} "
              f"{q_down:8.3f} {q_up:8.3f} {p_daily:8.3f}")
        results.append({
            "label": label, "date": date_str, "spot": spot,
            "T_days": T_days, "q_tail": q_tail,
            "q_down": q_down, "q_up": q_up,
            "p_daily": p_daily, "p_intra": p_intra,
            "iv_pts": iv_pts,
        })
    else:
        print(f"  {label:18s} {T_days:6d} {n_iv:6d} {'FAIL':>8}")


# ── 시각화 ────────────────────────────────────────────────
if results:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # IV 스마일
    ax = axes[0]
    colors = ['green','orange','red','purple','brown']
    for i, r in enumerate(results):
        if not r['iv_pts']: continue
        K_pts = sorted(r['iv_pts'].keys())
        iv_pts_sorted = [r['iv_pts'][k] for k in K_pts]
        lm = np.log(np.array(K_pts) / r['spot'])
        ax.plot(lm, iv_pts_sorted, 'o-', color=colors[i % len(colors)],
                ms=5, lw=1.2, label=r['label'])
    ax.axvline(-np.log(1/(1-THRESHOLD)), color='gray', lw=0.7, linestyle='--', label='±15%')
    ax.axvline(np.log(1/(1+THRESHOLD)),  color='gray', lw=0.7, linestyle='--')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title("IV 스마일 (로그-모네니스)")
    ax.set_xlabel("log(K/S)"); ax.set_ylabel("Implied Vol")
    ax.legend(fontsize=7)

    # P vs Q_tail 비교
    ax = axes[1]
    x = np.arange(len(results))
    w = 0.3
    ax.bar(x - w, [r['p_daily']  for r in results], w,
           label='P_high (HMM daily)', color='steelblue', edgecolor='black', lw=0.5)
    ax.bar(x,     [r['q_tail']   for r in results], w,
           label='Q_tail (BL 15%)',    color='darkorange', edgecolor='black', lw=0.5)
    ax.bar(x + w, [r['q_down']   for r in results], w,
           label='Q_down only',        color='salmon', edgecolor='black', lw=0.5)
    for xi, r in zip(x, results):
        ax.text(xi,     r['q_tail']  + 0.01, f"{r['q_tail']:.2f}",  ha='center', fontsize=8)
        ax.text(xi - w, r['p_daily'] + 0.01, f"{r['p_daily']:.2f}", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in results], rotation=15, fontsize=8)
    ax.set_title("P (HMM) vs Q_tail (Breeden-Litzenberger)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("bl_density_v2.png", dpi=150, bbox_inches='tight')
    print("\n그래프 저장: bl_density_v2.png")

    # ── 최종 판정 ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("P vs Q 비교 — 디지털 옵션 방식 (recheck)")
    print("=" * 65)

    calm  = next((r for r in results if "평온" in r['label']), None)
    crisis = [r for r in results if "평온" not in r['label']]

    if calm and crisis:
        print(f"\n평온:  Q_tail={calm['q_tail']:.3f}  P={calm['p_daily']:.3f}")
        for c in crisis:
            print(f"{c['label']:14s}: Q_tail={c['q_tail']:.3f}  P={c['p_daily']:.3f}  "
                  f"Q/P ratio={c['q_tail']/(calm['q_tail']+1e-6):.1f}x vs "
                  f"P ratio={c['p_daily']/(calm['p_daily']+1e-6):.1f}x")

        q_crisis_mean = np.mean([r['q_tail'] for r in crisis])
        p_crisis_mean = np.mean([r['p_daily'] for r in crisis])
        q_ratio = q_crisis_mean / (calm['q_tail'] + 1e-6)
        p_ratio = p_crisis_mean / (calm['p_daily'] + 1e-6)

        print(f"\n위기/평온 배율:  Q = {q_ratio:.1f}x  P = {p_ratio:.1f}x")
        if q_ratio > p_ratio * 1.5:
            verdict = "Q가 P보다 위기 신호를 더 강하게 포착 (P != Q 확인)"
        elif abs(q_ratio - p_ratio) / max(q_ratio, p_ratio) < 0.3:
            verdict = "Q와 P 유사한 위기 반응 (P ~ Q)"
        else:
            verdict = "P가 Q보다 위기 반응 더 강함"
        print(f"판정: {verdict}")
    print("=" * 65)
