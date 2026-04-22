"""
TDA 사전 검증 실험 v1 — Flash Crash 전조 신호 존재 여부
========================================================
질문: BTC/ETH/LTC/XRP 일봉 데이터에서 Vietoris-Rips TDA를 통해 계산한
      Persistence Landscape L1-norm이 Flash Crash 60-480일 전에
      통계적으로 유의미한 상승 추세를 보이는가?

방법론: Ismail et al. (2020) — "Topological Data Analysis for Detecting
        Hidden Patterns in Cryptocurrency Flash Crashes" 적용

설계:
  - 대상 크래시: BTC 3대 폭락 (2018-01, 2020-03, 2022-11)
  - 다변량 TDA: BTC+ETH+LTC+XRP 4개 자산 동시 입력
  - 슬라이딩 윈도우: 30일 수익률 → point cloud (n=30, d=4)
  - Persistent Homology: H0+H1 (VietorisRipsPersistence)
  - 신호 지표: Persistence Landscape L1-norm
  - 통계 검증: Kendall's τ (상승 추세) + Mann-Whitney U (사전/사후 비교)

판정 기준:
  SIGNAL:   Kendall's τ > 0 and p < 0.05 (크래시 전 60-480일 윈도우)
            + 크래시 전 30일 L1-norm이 크래시 후 30일보다 유의미하게 높음
  NULL:     위 조건 모두 불충족 → TDA 레이어 폐기
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import kendalltau, mannwhitneyu
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
ASSETS = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"]
START  = "2017-01-01"
END    = "2023-12-31"

# BTC Flash Crash 기준점 (일봉 기준 고점 → 대폭락 시작일)
CRASH_EVENTS = {
    "2018 Bull Crash":    pd.Timestamp("2018-01-08"),   # BTC 고점 직후 급락
    "2020 COVID Crash":   pd.Timestamp("2020-03-12"),   # 50%+ 단일일 낙폭
    "2022 FTX Collapse":  pd.Timestamp("2022-11-08"),   # FTX 파산 공시일
}

WINDOW_SIZE  = 30    # point cloud 구성 윈도우 (30일 수익률)
PRE_CRASH    = 480   # 크래시 전 분석 구간 (일)
POST_WINDOW  = 30    # 사전 vs 사후 비교용 후방 윈도우
MAX_EDGE_LEN = 1.0   # Vietoris-Rips max edge length (정규화 후)

# ═══════════════════════════════════════════════════════════════
# 2. 데이터 로딩
# ═══════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("데이터 로딩 중...")
    frames = {}
    for asset in ASSETS:
        raw = yf.download(asset, start=START, end=END, auto_adjust=True, progress=False)
        close = raw["Close"]
        # yfinance >= 0.2: Close returns DataFrame with MultiIndex column
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[asset] = close

    df = pd.DataFrame(frames).dropna()
    print(f"  로드 완료: {len(df)}일치, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

# ═══════════════════════════════════════════════════════════════
# 3. TDA 파이프라인
# ═══════════════════════════════════════════════════════════════
def compute_persistence_l1(window_returns: np.ndarray, max_edge_length: float) -> float:
    """
    window_returns: shape (T, 4) — T일 × 4자산 로그 수익률
    Returns: Persistence Landscape L1-norm (H0 + H1 합산)
    """
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceLandscape

    scaler = StandardScaler()
    X = scaler.fit_transform(window_returns)   # (T, 4) 표준화
    X_3d = X[np.newaxis, :, :]                 # (1, T, 4) giotto-tda 입력 형식

    vrp = VietorisRipsPersistence(
        homology_dimensions=[0, 1],
        max_edge_length=max_edge_length,
        n_jobs=1
    )
    diagram = vrp.fit_transform(X_3d)          # (1, n_pairs, 3)

    pl = PersistenceLandscape(n_layers=1, n_bins=100)
    landscape = pl.fit_transform(diagram)      # (1, n_layers, n_bins, n_dims)

    # L1-norm: 전체 landscape 절댓값 합산
    l1_norm = float(np.sum(np.abs(landscape)))
    return l1_norm

def build_tda_timeseries(returns: pd.DataFrame) -> pd.Series:
    """슬라이딩 윈도우로 전체 기간 L1-norm 시계열 생성"""
    dates = []
    l1_values = []

    n = len(returns)
    print(f"TDA 시계열 계산 중 (총 {n - WINDOW_SIZE}개 윈도우)...")

    for i in range(WINDOW_SIZE, n):
        window = returns.iloc[i - WINDOW_SIZE:i].values   # (30, 4)
        l1 = compute_persistence_l1(window, MAX_EDGE_LEN)
        dates.append(returns.index[i])
        l1_values.append(l1)

        if (i - WINDOW_SIZE) % 100 == 0:
            print(f"  진행: {i - WINDOW_SIZE}/{n - WINDOW_SIZE} ({100*(i-WINDOW_SIZE)/(n-WINDOW_SIZE):.1f}%)")

    return pd.Series(l1_values, index=dates, name="L1_norm")

# ═══════════════════════════════════════════════════════════════
# 4. 통계 검증
# ═══════════════════════════════════════════════════════════════
def analyze_crash(crash_name: str, crash_date: pd.Timestamp,
                  tda_series: pd.Series) -> dict:
    """단일 크래시 이벤트에 대한 통계 분석"""

    # 크래시 전 PRE_CRASH일 구간
    pre_start = crash_date - pd.Timedelta(days=PRE_CRASH)
    pre_window = tda_series[(tda_series.index >= pre_start) &
                             (tda_series.index < crash_date)]

    # 크래시 후 POST_WINDOW일 구간 (비교용)
    post_end = crash_date + pd.Timedelta(days=POST_WINDOW)
    post_window = tda_series[(tda_series.index >= crash_date) &
                              (tda_series.index < post_end)]

    if len(pre_window) < 20 or len(post_window) < 5:
        print(f"  [{crash_name}] 데이터 부족: pre={len(pre_window)}, post={len(post_window)}")
        return None

    # Kendall's τ — 상승 추세 검정
    t_ranks = np.arange(len(pre_window))
    tau, tau_p = kendalltau(t_ranks, pre_window.values)

    # Mann-Whitney U — 사전 30일 vs 사후 30일 비교
    pre_last30 = pre_window.iloc[-POST_WINDOW:] if len(pre_window) >= POST_WINDOW else pre_window
    u_stat, u_p = mannwhitneyu(pre_last30.values, post_window.values,
                                alternative='greater')   # 사전 > 사후 가설

    result = {
        "crash":         crash_name,
        "crash_date":    crash_date,
        "pre_n":         len(pre_window),
        "post_n":        len(post_window),
        "pre_mean":      pre_window.mean(),
        "post_mean":     post_window.mean(),
        "kendall_tau":   tau,
        "kendall_p":     tau_p,
        "mann_whitney_p": u_p,
        "pre_window":    pre_window,
        "post_window":   post_window,
    }

    signal = tau > 0 and tau_p < 0.05
    print(f"\n  [{crash_name}]")
    print(f"    분석 구간: {pre_start.date()} ~ {crash_date.date()} (n={len(pre_window)})")
    print(f"    L1 평균: 사전={pre_window.mean():.4f}, 사후={post_window.mean():.4f}")
    print(f"    Kendall's tau = {tau:.4f}, p = {tau_p:.4f} {'[TREND]' if signal else '[NO TREND]'}")
    print(f"    Mann-Whitney p = {u_p:.4f} {'[PRE>POST]' if u_p < 0.05 else '[NO DIFF]'}")

    return result

# ═══════════════════════════════════════════════════════════════
# 5. 시각화
# ═══════════════════════════════════════════════════════════════
def plot_results(tda_series: pd.Series, crash_results: list):
    fig, axes = plt.subplots(len(crash_results) + 1, 1,
                              figsize=(14, 4 * (len(crash_results) + 1)))

    # 전체 시계열
    ax = axes[0]
    ax.plot(tda_series.index, tda_series.values, color='steelblue', lw=0.8, alpha=0.8)
    for res in crash_results:
        if res is None: continue
        ax.axvline(res["crash_date"], color='red', lw=2, linestyle='--',
                   label=res["crash"])
    ax.set_title("TDA Persistence Landscape L1-norm — Full Period")
    ax.set_ylabel("L1-norm")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 크래시별 확대
    for i, res in enumerate(crash_results):
        if res is None: continue
        ax = axes[i + 1]

        # 사전/사후 색 구분
        ax.fill_between(res["pre_window"].index, res["pre_window"].values,
                        alpha=0.3, color='orange', label='Pre-crash window')
        ax.fill_between(res["post_window"].index, res["post_window"].values,
                        alpha=0.3, color='blue', label='Post-crash window')
        ax.plot(res["pre_window"].index, res["pre_window"].values,
                color='darkorange', lw=1.2)
        ax.plot(res["post_window"].index, res["post_window"].values,
                color='navy', lw=1.2)
        ax.axvline(res["crash_date"], color='red', lw=2.5, linestyle='--')

        tau = res["kendall_tau"]
        tau_p = res["kendall_p"]
        u_p = res["mann_whitney_p"]
        signal_str = "SIGNAL [O]" if (tau > 0 and tau_p < 0.05) else "NULL [X]"
        ax.set_title(f"{res['crash']} | τ={tau:.3f} p={tau_p:.3f} | MW_p={u_p:.3f} | {signal_str}")
        ax.set_ylabel("L1-norm")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig("tda_validation_v1.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n그래프 저장: tda_validation_v1.png")

# ═══════════════════════════════════════════════════════════════
# 6. 요약 출력
# ═══════════════════════════════════════════════════════════════
def print_summary(results: list):
    print("\n" + "=" * 65)
    print("TDA 검증 실험 v1 — 최종 판정")
    print("=" * 65)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("유효 결과 없음")
        return

    signal_count = sum(1 for r in valid_results
                       if r["kendall_tau"] > 0 and r["kendall_p"] < 0.05)
    mw_count = sum(1 for r in valid_results if r["mann_whitney_p"] < 0.05)

    print(f"\n분석 크래시: {len(valid_results)}개")
    print(f"Kendall τ > 0 & p < 0.05: {signal_count}/{len(valid_results)}")
    print(f"Mann-Whitney 사전>사후 유의: {mw_count}/{len(valid_results)}")

    print("\n[ 크래시별 상세 ]")
    for r in valid_results:
        tau_sig = "[O]" if (r["kendall_tau"] > 0 and r["kendall_p"] < 0.05) else "[X]"
        mw_sig  = "[O]" if r["mann_whitney_p"] < 0.05 else "[X]"
        print(f"  {r['crash']:22s} τ={r['kendall_tau']:+.3f} p={r['kendall_p']:.3f} {tau_sig} | "
              f"MW_p={r['mann_whitney_p']:.3f} {mw_sig}")

    print("\n[ 최종 판정 ]")
    if signal_count >= 2:
        print("  SIGNAL — TDA 레이어 진행 근거 확보")
        print("  → Step 2: RMD 교체 (MCD+IsoForest) 및 실시간 파이프라인 구축")
    elif signal_count == 1:
        print("  WEAK SIGNAL — 단일 크래시에서만 신호 감지")
        print("  → 추가 크래시 데이터 또는 파라미터 튜닝 후 재검증 필요")
    else:
        print("  NULL — TDA 레이어 신호 없음")
        print("  → TDA 폐기. 대안: RMD 단독 또는 다른 위상 지표 탐색")

    print("=" * 65)

    # txt 요약 저장
    with open("summary_tda_v1.txt", "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("TDA 검증 실험 v1 — Flash Crash 전조 신호\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"자산: {', '.join(ASSETS)}\n")
        f.write(f"기간: {START} ~ {END}\n")
        f.write(f"윈도우: {WINDOW_SIZE}일, 사전분석: {PRE_CRASH}일\n\n")
        f.write(f"Kendall τ 유의: {signal_count}/{len(valid_results)}\n")
        f.write(f"Mann-Whitney 유의: {mw_count}/{len(valid_results)}\n\n")
        for r in valid_results:
            tau_sig = "[SIGNAL]" if (r["kendall_tau"] > 0 and r["kendall_p"] < 0.05) else "[NULL]"
            f.write(f"{r['crash']:22s} τ={r['kendall_tau']:+.3f} p={r['kendall_p']:.3f} "
                    f"MW_p={r['mann_whitney_p']:.3f} → {tau_sig}\n")

# ═══════════════════════════════════════════════════════════════
# 7. 메인 실행
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("TDA 검증 실험 v1 시작\n" + "=" * 65)

    # 데이터 준비
    prices  = load_data()
    returns = compute_log_returns(prices)

    # TDA 시계열 (시간 소요 — 약 2-5분)
    tda_series = build_tda_timeseries(returns)
    print(f"\nTDA 시계열 완성: {len(tda_series)}개 데이터포인트")
    print(f"L1-norm 범위: {tda_series.min():.4f} ~ {tda_series.max():.4f}")

    # 크래시별 통계 분석
    print("\n통계 검증 중...")
    results = []
    for crash_name, crash_date in CRASH_EVENTS.items():
        if crash_date < tda_series.index[0] or crash_date > tda_series.index[-1]:
            print(f"  [{crash_name}] 데이터 범위 밖 — 스킵")
            results.append(None)
            continue
        res = analyze_crash(crash_name, crash_date, tda_series)
        results.append(res)

    # 시각화 및 요약
    plot_results(tda_series, results)
    print_summary(results)
