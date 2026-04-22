"""
Breeden-Litzenberger Q-density
================================
d²C/dK² = e^{-rT} × q(K)  (r=0 for crypto)

절차:
  1. 위기/평온 대표 날짜 5개 선택
  2. 각 날짜의 ~30일 만기 옵션 체인 (±40% moneyness) Deribit 수집
  3. B-L 수치 미분 → Q-density
  4. Q(|ret|>15%) vs P_high (HMM) 비교

날짜: 평온(2021-10-01), 2021-05 급락(2021-05-19),
      Luna(2022-05-10), 3AC(2022-06-14), FTX(2022-11-09)
"""
import warnings; warnings.filterwarnings('ignore')
import time, pickle, os
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

CACHE = "bl_chains.pkl"

# ── 날짜 / 만기 설정 ──────────────────────────────────────
# (날짜, 만기, 레이블)  만기=Deribit DDMONYY 형식
TARGETS = [
    ("2021-10-01", "29OCT21", "평온 2021-10"),
    ("2021-05-19", "28MAY21", "2021-05 급락"),
    ("2022-05-10", "27MAY22", "Luna"),
    ("2022-06-14", "24JUN22", "3AC"),
    ("2022-11-09", "25NOV22", "FTX"),
]

# ── Deribit 유틸 ──────────────────────────────────────────
BASE = "https://www.deribit.com/api/v2/public"

def deribit(method, **params):
    r = requests.get(f"{BASE}/{method}", params=params, timeout=15)
    return r.json().get("result", {})

def get_mark_price_on_date(instrument, date_str):
    """일별 OHLCV에서 해당 날짜 종가(mark price) 반환 (BTC 단위)"""
    ts = int(pd.Timestamp(date_str).timestamp() * 1000)
    res = deribit("get_tradingview_chart_data",
                  instrument_name=instrument,
                  start_timestamp=ts,
                  end_timestamp=ts + 86_400_000,
                  resolution="1D")
    if not res or "close" not in res or not res["close"]:
        return None
    return float(res["close"][0])   # BTC 단위

def get_spot_on_date(date_str):
    """btc_price 시리즈에서 spot 가져오기"""
    with open("arena_val_features.pkl",'rb') as f:
        _, btc_price = pickle.load(f)
    idx = btc_price.index.get_indexer([pd.Timestamp(date_str)], method='nearest')[0]
    return float(btc_price.iloc[idx])

def deribit_strikes(spot, pct_range=0.40, step_pct=0.05):
    """spot ± pct_range를 step_pct 간격으로 정규 스트라이크 생성"""
    lo = spot * (1 - pct_range)
    hi = spot * (1 + pct_range)
    # Deribit 스트라이크: 1000 단위 (BTC)
    step = max(1000, round(spot * step_pct / 1000) * 1000)
    lo_r = int(round(lo / step) * step)
    hi_r = int(round(hi / step) * step)
    return list(range(lo_r, hi_r + step, step))

# ── 옵션 체인 수집 ────────────────────────────────────────
def fetch_chain(date_str, expiry_code, label, spot):
    strikes = deribit_strikes(spot)
    chain   = []
    for K in strikes:
        for opt in ['C', 'P']:
            instr = f"BTC-{expiry_code}-{K}-{opt}"
            p = get_mark_price_on_date(instr, date_str)
            if p is not None and p > 0:
                chain.append({"strike": K, "type": opt,
                               "price_btc": p, "price_usd": p * spot})
        time.sleep(0.05)
    df = pd.DataFrame(chain)
    print(f"  {label}: {len(df)} 옵션  "
          f"strikes {df['strike'].min() if len(df)>0 else '-'} ~ "
          f"{df['strike'].max() if len(df)>0 else '-'}")
    return df

if os.path.exists(CACHE):
    print(f"캐시 로드: {CACHE}")
    with open(CACHE,'rb') as f:
        chains = pickle.load(f)
else:
    chains = {}
    for date_str, expiry, label in TARGETS:
        spot = get_spot_on_date(date_str)
        print(f"\n{label} ({date_str})  spot=${spot:,.0f}")
        df = fetch_chain(date_str, expiry, label, spot)
        chains[label] = {"date": date_str, "expiry": expiry,
                         "spot": spot, "chain": df}
        time.sleep(0.5)
    with open(CACHE,'wb') as f:
        pickle.dump(chains, f)
    print(f"\n저장: {CACHE}")


# ── Breeden-Litzenberger ──────────────────────────────────
def bs_call(S, K, T, sigma, r=0.0):
    if sigma <= 0 or T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, sigma, r=0.0):
    return bs_call(S, K, T, sigma, r) - S + K*np.exp(-r*T)

def bl_density(chain_df, spot, date_str, expiry_code, r=0.0):
    """콜 가격 → B-L 밀도"""
    # 만기까지 남은 일수
    expiry_map = {
        "29OCT21": "2021-10-29", "28MAY21": "2021-05-28",
        "27MAY22": "2022-05-27", "24JUN22": "2022-06-24",
        "25NOV22": "2022-11-25",
    }
    T = (pd.Timestamp(expiry_map[expiry_code]) -
         pd.Timestamp(date_str)).days / 365.0
    if T <= 0: T = 1/365

    calls = chain_df[chain_df['type'] == 'C'].copy()
    puts  = chain_df[chain_df['type'] == 'P'].copy()

    # put-call parity로 콜 가격 보완 (OTM put → ITM call 변환)
    # C = P + S - K*e^{-rT}
    for _, row in puts.iterrows():
        K = row['strike']
        if K < spot:  # OTM put → ITM call (B-L은 OTM 중심이므로 활용)
            c_parity = row['price_usd'] + spot - K * np.exp(-r * T)
            if K not in calls['strike'].values:
                calls = pd.concat([calls, pd.DataFrame([{
                    'strike': K, 'type': 'C',
                    'price_btc': c_parity/spot, 'price_usd': c_parity
                }])], ignore_index=True)

    calls = calls.sort_values('strike').drop_duplicates('strike')
    if len(calls) < 5:
        return None, None, None

    K_arr = calls['strike'].values.astype(float)
    C_arr = calls['price_usd'].values.astype(float)

    # 단조감소 강제 (arbitrage-free)
    for i in range(1, len(C_arr)):
        if C_arr[i] > C_arr[i-1]:
            C_arr[i] = C_arr[i-1] * 0.999

    # 균일 그리드로 보간 후 수치 이중 미분
    K_min, K_max = K_arr[0], K_arr[-1]
    K_grid = np.linspace(K_min, K_max, 300)
    cs     = CubicSpline(K_arr, C_arr, bc_type='natural')
    C_grid = np.maximum(cs(K_grid), 0)

    dK  = K_grid[1] - K_grid[0]
    d2C = np.gradient(np.gradient(C_grid, dK), dK)
    q   = np.maximum(d2C * np.exp(r * T), 0)

    # 정규화
    area = np.trapz(q, K_grid)
    if area < 1e-10:
        return None, None, None
    q /= area

    return K_grid, q, T


# ── HMM P_high 로드 ───────────────────────────────────────
with open("arena_val_hmm.pkl",'rb') as f:
    _, high_p_daily, _ = pickle.load(f)
with open("intraday_hmm.pkl",'rb') as f:
    intra = pickle.load(f)
high_p_1h = intra["high_p_1h"]


# ── 분석 ─────────────────────────────────────────────────
THRESHOLD = 0.15   # 15% 이상 이동 = 고변동성 레짐 정의

results = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for idx, (label, data) in enumerate(chains.items()):
    date_str  = data['date']
    expiry    = data['expiry']
    spot      = data['spot']
    chain_df  = data['chain']

    if len(chain_df) < 5:
        print(f"{label}: 데이터 부족 ({len(chain_df)}개)")
        continue

    K_grid, q, T = bl_density(chain_df, spot, date_str, expiry)
    if K_grid is None:
        print(f"{label}: B-L 밀도 계산 실패")
        continue

    # Q(|ret| > 15%) = tail probability
    lo = spot * (1 - THRESHOLD)
    hi = spot * (1 + THRESHOLD)
    mask_tail = (K_grid < lo) | (K_grid > hi)
    q_tail    = np.trapz(q[mask_tail], K_grid[mask_tail]) if mask_tail.sum() > 1 else 0.0
    q_down    = np.trapz(q[K_grid < lo], K_grid[K_grid < lo]) if (K_grid < lo).sum() > 1 else 0.0
    q_up      = np.trapz(q[K_grid > hi], K_grid[K_grid > hi]) if (K_grid > hi).sum() > 1 else 0.0

    # P_high (일별 HMM)
    ts = pd.Timestamp(date_str)
    p_daily = float(high_p_daily.reindex([ts], method='nearest').iloc[0]) \
              if ts in high_p_daily.index or True else 0.0
    p_intra_day = high_p_1h[high_p_1h.index.date == pd.Timestamp(date_str).date()]
    p_intra = float(p_intra_day.mean()) if len(p_intra_day) > 0 else 0.0

    print(f"\n[{label}]  spot=${spot:,.0f}  T={T*365:.0f}d")
    print(f"  Q(tail>15%): {q_tail:.3f}  Q(down): {q_down:.3f}  Q(up): {q_up:.3f}")
    print(f"  P_high(일별): {p_daily:.3f}  P_high(1h): {p_intra:.3f}")
    print(f"  Q_tail / P_daily ratio: {q_tail/(p_daily+1e-6):.2f}x")

    results.append({
        "label": label, "date": date_str, "spot": spot,
        "q_tail": q_tail, "q_down": q_down, "q_up": q_up,
        "p_daily": p_daily, "p_intra": p_intra,
        "K_grid": K_grid, "q": q, "T": T,
    })

    # 밀도 플롯
    ax = axes_flat[idx]
    ax.plot(K_grid / spot, q * spot, color='darkorange', lw=1.5, label='Q-density (BL)')
    ax.axvline(1 - THRESHOLD, color='red', lw=0.8, linestyle='--', alpha=0.7)
    ax.axvline(1 + THRESHOLD, color='red', lw=0.8, linestyle='--', alpha=0.7, label='±15%')
    ax.axvline(1.0, color='gray', lw=0.5)
    ax.fill_between(K_grid/spot, q*spot,
                    where=(K_grid < lo) | (K_grid > hi),
                    alpha=0.3, color='red', label=f'Q(tail)={q_tail:.2f}')
    ax.set_title(f"{label}\nP={p_daily:.2f}  Q_tail={q_tail:.2f}")
    ax.set_xlabel("K/S"); ax.set_ylabel("q(K)×S")
    ax.legend(fontsize=7)
    ax.set_xlim(0.5, 1.5)

# 마지막 패널: P vs Q 비교
if results:
    ax = axes_flat[len(results)]
    labels_r = [r["label"] for r in results]
    q_tails  = [r["q_tail"]  for r in results]
    p_dailys = [r["p_daily"] for r in results]
    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w/2, p_dailys, w, label='P_high (HMM)', color='steelblue', edgecolor='black', lw=0.5)
    ax.bar(x + w/2, q_tails,  w, label='Q_tail (BL)',  color='darkorange', edgecolor='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace(" ","\\n") for r in results], fontsize=7)
    ax.set_title("P (HMM) vs Q (Breeden-Litzenberger)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)

for i in range(len(results)+1, 6):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.savefig("bl_density.png", dpi=150, bbox_inches='tight')
print("\n그래프 저장: bl_density.png")


# ── 최종 판정 ─────────────────────────────────────────────
if results:
    print("\n" + "=" * 65)
    print("Breeden-Litzenberger P vs Q 비교")
    print("=" * 65)
    print(f"{'날짜':20s} {'Q_tail':>8} {'Q_down':>8} {'P_daily':>8} {'ratio':>8}")
    print("-" * 55)
    for r in results:
        print(f"  {r['label']:18s} {r['q_tail']:8.3f} {r['q_down']:8.3f} "
              f"{r['p_daily']:8.3f} {r['q_tail']/(r['p_daily']+1e-6):8.1f}x")

    crisis = [r for r in results if r['label'] != "평온 2021-10"]
    calm   = [r for r in results if r['label'] == "평온 2021-10"]

    if crisis and calm:
        q_crisis = np.mean([r['q_tail'] for r in crisis])
        q_calm   = np.mean([r['q_tail'] for r in calm])
        p_crisis = np.mean([r['p_daily'] for r in crisis])
        p_calm   = np.mean([r['p_daily'] for r in calm])
        print(f"\n위기 국면:  Q_tail={q_crisis:.3f}  P_high={p_crisis:.3f}")
        print(f"평온 국면:  Q_tail={q_calm:.3f}   P_high={p_calm:.3f}")

        if q_crisis > q_calm * 1.5 and p_crisis > p_calm * 1.5:
            verdict = "P와 Q 모두 위기 국면 포착 - 방향 일치"
        elif q_crisis > q_calm * 1.5 and p_crisis <= p_calm * 1.5:
            verdict = "Q만 위기 포착 - P가 과소추정 (P != Q)"
        elif q_crisis <= q_calm * 1.5 and p_crisis > p_calm * 1.5:
            verdict = "P만 위기 포착 - Q가 과소추정"
        else:
            verdict = "P, Q 모두 위기 미포착 - 데이터 문제 가능성"
        print(f"\n판정: {verdict}")
    print("=" * 65)
