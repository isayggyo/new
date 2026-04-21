"""
맥락 개입 실험 v5 — 마지막 검증
================================
가설: "행동 편향 변수(D)의 신호는 VIX 레벨이 아니라 VIX 상승 모멘텀에 반응한다."

v4에서 발견: VIX 대역 모두에서 D > C 방향 일관, VIX 22-28에서 ΔIC +0.163 최대.
v5가 묻는 것: 그 신호는 VIX가 '올라가는 중'일 때 강해지는가, 아니면 레벨 자체인가?

설계:
  - VIX 대역 3개 × 방향 2개 = 6-cell 매트릭스
  - 각 셀에서 D vs C IC 비교
  - 예측: VIX 상승 × 22-28 구간에서 ΔIC 최대

VIX 방향: vix_pct_change_5d > 0 → rising, ≤ 0 → falling
VIX 대역: 12-17 / 17-22 / 22-28
통계: Bonferroni p < 0.05/6 = 0.0083

판정 기준:
  MECHANISM: rising 셀이 같은 대역 falling 셀보다 ΔIC 일관되게 높음
             + 최대 셀(22-28 rising)이 통계적으로 유의
  LEVEL:     rising/falling 무관하게 high VIX에서 D > C
  NULL:      패턴 없음 — v4 VIX 22-28 결과는 샘플 노이즈
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, ttest_rel

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
TICKERS  = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "JNJ", "XOM", "CVX", "MCD", "NKE"]
SECTORS  = {
    'AAPL': 'Tech',    'MSFT': 'Tech',    'NVDA': 'Tech',
    'JPM':  'Finance', 'BAC':  'Finance',
    'JNJ':  'Health',
    'XOM':  'Energy',  'CVX':  'Energy',
    'MCD':  'Consumer','NKE':  'Consumer',
}
START    = "2015-01-01"
END      = "2024-12-31"
VARIANTS = ['A', 'C', 'D']

N_FOLDS  = 3
N_SEEDS  = 3
HIDDEN   = (64, 32)
MAX_ITER = 80
LOOKBACK = 10

VIX_BANDS   = {'12-17': (12, 17), '17-22': (17, 22), '22-28': (22, 28)}
BONFERRONI  = 0.05 / 6   # 0.0083

# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ═══════════════════════════════════════════════════════════════
print("[1/6] 데이터 수집 중...")

def dl(tks, field="Close"):
    d = yf.download(tks, start=START, end=END, auto_adjust=True, progress=False)[field]
    return d.to_frame(name=tks) if isinstance(d, pd.Series) else d

prices  = dl(TICKERS).ffill().dropna()
returns = prices.pct_change().dropna()
volumes = dl(TICKERS, field="Volume").ffill().dropna()

macro_src = dl(["^VIX", "^TNX", "^IRX", "HYG", "LQD", "UUP", "USO"]).ffill()
spy_ret   = dl(["SPY"]).ffill().pct_change()['SPY']

macro = pd.DataFrame(index=macro_src.index)
macro['vix_level']    = macro_src['^VIX']
macro['vix_mom_5d']   = macro_src['^VIX'].pct_change(5)   # VIX 방향 핵심 변수
macro['yield_slope']  = macro_src['^TNX'] - macro_src['^IRX']
macro['yield_chg']    = macro['yield_slope'].diff(5)
macro['credit_proxy'] = np.log(macro_src['LQD'] / macro_src['HYG'])
macro['credit_chg']   = macro['credit_proxy'].diff(5)
macro['dxy_ret_20d']  = macro_src['UUP'].pct_change(20)
macro['oil_ret_20d']  = macro_src['USO'].pct_change(20)
macro = macro.dropna()

# ═══════════════════════════════════════════════════════════════
# 3. 행동 특징
# ═══════════════════════════════════════════════════════════════
print("[2/6] 행동 특징 계산 중...")

sector_cs_std = {}
for sec in set(SECTORS.values()):
    members = [t for t, s in SECTORS.items() if s == sec]
    if len(members) > 1:
        sector_cs_std[sec] = returns[members].std(axis=1)

behavioral = {}
for ticker in TICKERS:
    r = returns[ticker]; p = prices[ticker]; v = volumes[ticker]
    bf = pd.DataFrame(index=r.index)
    bf['anchor']       = p / p.rolling(252, min_periods=60).max()
    ret5 = r.rolling(5).sum().abs()
    bf['overreaction'] = ret5 / (ret5.rolling(60).mean() + 1e-8)
    r_neg = r.where(r < 0, np.nan); r_pos = r.where(r > 0, np.nan)
    dv = r_neg.rolling(20, min_periods=2).std()
    uv = r_pos.rolling(20, min_periods=2).std()
    bf['loss_aversion'] = dv / (uv + 1e-8)
    sec = SECTORS[ticker]
    cs  = sector_cs_std.get(sec, pd.Series(0, index=r.index))
    bf['herding']     = 1 - cs / (cs.rolling(60).mean() + 1e-8)
    vol_ratio = v.rolling(20).mean() / (v.rolling(60).mean() + 1e-8)
    bf['disposition'] = vol_ratio * bf['anchor']
    behavioral[ticker] = bf.dropna()

# ═══════════════════════════════════════════════════════════════
# 4. 공통 유틸리티
# ═══════════════════════════════════════════════════════════════
def price_features(r):
    f = pd.DataFrame(index=r.index)
    f['ret_1d']  = r
    f['ret_5d']  = r.rolling(5).sum()
    f['ret_20d'] = r.rolling(20).sum()
    f['vol_20d'] = r.rolling(20).std()
    f['mom']     = r.rolling(20).mean() / (r.rolling(60).std() + 1e-8)
    return f.dropna()


def build_dataset(ticker, variant):
    pf = price_features(returns[ticker]); r = returns[ticker]
    extra = None if variant == 'A' else (macro if variant == 'C' else behavioral[ticker])
    idx = pf.index
    if extra is not None:
        idx = idx.intersection(extra.index)
    pf, r = pf.loc[idx], r.loc[idx]
    ex = extra.loc[idx].values if extra is not None else None
    pv = pf.values
    X, y, dates = [], [], []
    for i in range(LOOKBACK, len(pv) - 1):
        w = pv[i - LOOKBACK:i].flatten()
        X.append(np.concatenate([w, ex[i]]) if ex is not None else w)
        y.append(r.values[i + 1])
        dates.append(idx[i + 1])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def walk_forward_splits(n, n_folds=N_FOLDS, min_train=0.45):
    start = int(n * min_train)
    test_size = (n - start) // n_folds
    splits = []
    for k in range(n_folds):
        tr_end = start + k * test_size
        te_end = tr_end + test_size if k < n_folds - 1 else n
        splits.append((0, tr_end, tr_end, te_end))
    return splits


def ic_score(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho


# ═══════════════════════════════════════════════════════════════
# 5. 실험 실행
# ═══════════════════════════════════════════════════════════════
print("[3/6] 모델 학습 및 예측 수집 중...")

all_preds = []

for ticker in TICKERS:
    datasets = {v: build_dataset(ticker, v) for v in VARIANTS}
    common = datasets['A'][2]
    for v in ['C', 'D']:
        common = common.intersection(datasets[v][2])

    aligned = {}
    for v in VARIANTS:
        X, y, d = datasets[v]
        mask = d.isin(common)
        aligned[v] = (X[mask], y[mask], d[mask])

    n = len(common)
    splits = walk_forward_splits(n)

    for fold_i, (a, b, c, e) in enumerate(splits):
        fold_preds = {}
        for seed in range(N_SEEDS):
            for v in VARIANTS:
                X, y, d = aligned[v]
                Xtr, ytr = X[a:b], y[a:b]
                Xte, yte = X[c:e], y[c:e]
                if len(Xtr) < 10 or len(Xte) < 5:
                    continue
                sc = StandardScaler()
                model = MLPRegressor(
                    hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER,
                    random_state=seed, early_stopping=False,
                    learning_rate_init=1e-3, alpha=1e-4, tol=1e-3,
                )
                model.fit(sc.fit_transform(Xtr), ytr)
                ypred = model.predict(sc.transform(Xte))

                if seed == 0:
                    fold_preds[v] = (yte, ypred, d[c:e])

        if 'A' in fold_preds and 'C' in fold_preds and 'D' in fold_preds:
            yte, _, dates = fold_preds['A']
            for i, (dt, yt) in enumerate(zip(dates, yte)):
                all_preds.append({
                    'ticker': ticker,
                    'date':   dt,
                    'y_true':   yt,
                    'y_pred_A': fold_preds['A'][1][i],
                    'y_pred_C': fold_preds['C'][1][i],
                    'y_pred_D': fold_preds['D'][1][i],
                })

preds = pd.DataFrame(all_preds)
preds['date'] = pd.to_datetime(preds['date'])

# VIX 레벨 & 방향 붙이기
vix_level = macro['vix_level'].reindex(preds['date']).values
vix_mom   = macro['vix_mom_5d'].reindex(preds['date']).values
preds['vix']       = vix_level
preds['vix_dir']   = np.where(vix_mom > 0, 'rising', 'falling')

# ═══════════════════════════════════════════════════════════════
# 6. 6-cell 매트릭스 분석
# ═══════════════════════════════════════════════════════════════
print("\n[4/6] VIX 대역 × 방향 매트릭스 분석")
print("=" * 65)
print(f"\n  Bonferroni 임계값: p < {BONFERRONI:.4f}")

BANDS = list(VIX_BANDS.keys())
DIRS  = ['rising', 'falling']

matrix_delta = np.full((3, 2), np.nan)
matrix_p     = np.full((3, 2), np.nan)
matrix_n     = np.zeros((3, 2), dtype=int)
matrix_ic_c  = np.full((3, 2), np.nan)
matrix_ic_d  = np.full((3, 2), np.nan)

print(f"\n  {'':12} {'rising':>22} {'falling':>22}")
print(f"  {'대역':<12} {'n':>5} {'C IC':>7} {'D IC':>7} {'ΔIC':>7} | "
      f"{'n':>5} {'C IC':>7} {'D IC':>7} {'ΔIC':>7}")
print(f"  {'-'*65}")

for bi, band in enumerate(BANDS):
    lo, hi = VIX_BANDS[band]
    row_parts = [f"  VIX {band:<7}"]
    for di, dirn in enumerate(DIRS):
        sub = preds[(preds['vix'] >= lo) & (preds['vix'] < hi) &
                    (preds['vix_dir'] == dirn)]
        n = len(sub)
        matrix_n[bi, di] = n
        if n < 15:
            row_parts.append(f" {'부족':>5} {'':>7} {'':>7} {'':>7}")
            continue
        ic_c = ic_score(sub['y_true'], sub['y_pred_C'])
        ic_d = ic_score(sub['y_true'], sub['y_pred_D'])
        delta = ic_d - ic_c
        matrix_ic_c[bi, di] = ic_c
        matrix_ic_d[bi, di] = ic_d
        matrix_delta[bi, di] = delta

        # 종목별 paired t-test
        ic_pairs = []
        for tk in TICKERS:
            stk = sub[sub['ticker'] == tk]
            if len(stk) < 5:
                continue
            ic_pairs.append((
                ic_score(stk['y_true'], stk['y_pred_C']),
                ic_score(stk['y_true'], stk['y_pred_D']),
            ))
        if len(ic_pairs) >= 3:
            arr = np.array(ic_pairs)
            _, p = ttest_rel(arr[:, 1], arr[:, 0])
        else:
            p = np.nan
        matrix_p[bi, di] = p

        sig = "***" if (not np.isnan(p)) and p < 0.001 else \
              "**"  if (not np.isnan(p)) and p < 0.01  else \
              "*"   if (not np.isnan(p)) and p < BONFERRONI else \
              "†"   if (not np.isnan(p)) and p < 0.05  else ""
        row_parts.append(f" {n:>5} {ic_c:>7.4f} {ic_d:>7.4f} {delta:>+6.4f}{sig:<1}")

    print("".join(row_parts))

# ═══════════════════════════════════════════════════════════════
# 7. 방향 효과 검증: rising vs falling 내 ΔIC 차이
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'대역':<12} {'rising ΔIC':>12} {'falling ΔIC':>13} {'방향 효과':>12}")
print(f"  {'-'*50}")
direction_effect = {}
for bi, band in enumerate(BANDS):
    dr = matrix_delta[bi, 0]
    df_ = matrix_delta[bi, 1]
    if np.isnan(dr) or np.isnan(df_):
        print(f"  VIX {band:<7} {'데이터 부족':>40}")
        continue
    effect = dr - df_
    direction_effect[band] = effect
    marker = " ←" if effect > 0.01 else ""
    print(f"  VIX {band:<7} {dr:>+12.4f} {df_:>+13.4f} {effect:>+12.4f}{marker}")

# ═══════════════════════════════════════════════════════════════
# 8. 시각화
# ═══════════════════════════════════════════════════════════════
print("\n[5/6] 시각화 생성 중...")

# 그림 1: ΔIC 히트맵 (3×2)
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

for ax, title, data, fmt in zip(
        axes1,
        ["ΔIC (D − C)", "p-value (paired t-test)"],
        [matrix_delta, matrix_p],
        ["{:+.3f}", "{:.3f}"],
):
    masked = np.ma.masked_invalid(data)
    if title.startswith("ΔIC"):
        vmin, vmax = -0.25, 0.25
        cmap = "RdYlGn"
    else:
        vmin, vmax = 0, 0.15
        cmap = "RdYlGn_r"

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Rising ↑', 'Falling ↓'], fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels([f'VIX {b}' for b in BANDS], fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    for bi in range(3):
        for di in range(2):
            val = data[bi, di]
            n   = matrix_n[bi, di]
            if np.isnan(val):
                continue
            sig_str = ""
            if title.startswith("p"):
                sig_str = "***" if val < 0.001 else "**" if val < 0.01 \
                          else "*" if val < BONFERRONI else "†" if val < 0.05 else ""
            txt = fmt.format(val) + sig_str
            ax.text(di, bi, f"{txt}\n(n={n})", ha='center', va='center',
                    fontsize=10, color='black',
                    fontweight='bold' if sig_str else 'normal')

plt.suptitle("VIX 대역 × 방향: D vs C 비교\n(가설: rising 구간에서 ΔIC 최대)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("vix_momentum_matrix.png", dpi=140, bbox_inches='tight')
print("  → vix_momentum_matrix.png 저장")
plt.close()

# 그림 2: 대역별 rising vs falling ΔIC 비교 (bar)
fig2, ax2 = plt.subplots(figsize=(10, 5))
x = np.arange(len(BANDS))
w = 0.35
valid_bands = [b for b in BANDS if not np.isnan(matrix_delta[BANDS.index(b), 0])
                                 and not np.isnan(matrix_delta[BANDS.index(b), 1])]

for i, band in enumerate(BANDS):
    dr  = matrix_delta[i, 0]
    df_ = matrix_delta[i, 1]
    pr  = matrix_p[i, 0]
    pf_ = matrix_p[i, 1]
    if np.isnan(dr) or np.isnan(df_):
        continue
    b1 = ax2.bar(i - w/2, dr,  w, color='#e84', alpha=0.85, label='Rising' if i == 0 else '')
    b2 = ax2.bar(i + w/2, df_, w, color='#48c', alpha=0.85, label='Falling' if i == 0 else '')

    def sig_label(p):
        if np.isnan(p): return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else \
               f"*\np<{BONFERRONI:.3f}" if p < BONFERRONI else f"†\np={p:.2f}" if p < 0.05 else f"ns\np={p:.2f}"

    ymax = max(abs(dr), abs(df_)) + 0.02
    ax2.text(i - w/2, dr + (0.005 if dr >= 0 else -0.02), sig_label(pr),
             ha='center', fontsize=8, color='#a30')
    ax2.text(i + w/2, df_ + (0.005 if df_ >= 0 else -0.02), sig_label(pf_),
             ha='center', fontsize=8, color='#246')

ax2.axhline(0, color='black', lw=0.8)
ax2.set_xticks(range(len(BANDS)))
ax2.set_xticklabels([f'VIX {b}' for b in BANDS], fontsize=11)
ax2.set_ylabel("ΔIC (D − C)", fontsize=11)
ax2.set_title("VIX 대역 × 방향별 행동 편향 신호 강도\n(bar 높이 = D가 C보다 얼마나 나은가)", fontsize=12)
ax2.legend(fontsize=11); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("vix_direction_ic.png", dpi=140, bbox_inches='tight')
print("  → vix_direction_ic.png 저장")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 9. 최종 판정
# ═══════════════════════════════════════════════════════════════
print("\n[6/6] 최종 판정")

# 방향 효과 일관성 (3개 대역 모두 rising ΔIC > falling ΔIC)
direction_consistent = all(v > 0 for v in direction_effect.values()) if direction_effect else False
n_bands_positive     = sum(1 for v in direction_effect.values() if v > 0)

# 최대 셀(22-28, rising) 유의성
bi_22 = BANDS.index('22-28')
p_peak = matrix_p[bi_22, 0]
peak_sig = (not np.isnan(p_peak)) and (p_peak < BONFERRONI)
peak_delta = matrix_delta[bi_22, 0]

# 판정
if direction_consistent and peak_sig:
    verdict = "MECHANISM CONFIRMED — VIX 상승 모멘텀이 행동 편향 신호의 원인."
elif direction_consistent and not peak_sig:
    verdict = "SUGGESTIVE MECHANISM — 방향 효과 일관 but 유의성 부족. 샘플 제약."
elif not direction_consistent and peak_sig:
    verdict = "LEVEL EFFECT — 레벨(22-28)이 원인, 방향 무관."
else:
    verdict = "NULL — 방향 효과 없음. VIX 22-28의 v4 결과는 샘플 노이즈."

summary = [
    "=" * 65,
    "v5 최종 요약 — VIX 모멘텀 × 행동 편향",
    "=" * 65,
    "",
    "가설: D 신호는 VIX '레벨'이 아니라 '상승 중'이라는 상태에서 강해진다.",
    "",
    "[ 6-cell 매트릭스 결과 ]",
    f"  {'대역':<10} {'rising ΔIC':>12} {'falling ΔIC':>14} {'방향효과':>10}",
]
for bi, band in enumerate(BANDS):
    dr  = matrix_delta[bi, 0]
    df_ = matrix_delta[bi, 1]
    if np.isnan(dr) or np.isnan(df_):
        summary.append(f"  VIX {band:<6} {'데이터 부족':>36}")
    else:
        eff = dr - df_
        summary.append(f"  VIX {band:<6} {dr:>+12.4f} {df_:>+14.4f} {eff:>+10.4f}")
summary += [
    "",
    f"[ 방향 효과 일관성 ] {n_bands_positive}/{len(direction_effect)} 대역에서 rising > falling",
    f"[ 최대 셀 유의성  ] VIX 22-28 rising: ΔIC={peak_delta:+.4f}, p={p_peak:.4f} "
    f"({'PASS' if peak_sig else 'FAIL'}, 기준 p<{BONFERRONI:.4f})",
    "",
    f"최종 판정: {verdict}",
    "",
    "연구 시리즈 결론:",
    "  v2: 거시 정보 (C) → 예측력 없음          (semi-strong EMH 성립)",
    "  v3: 행동 편향 (D) → 예측력 없음          (null, normal 구간만 미약)",
    "  v4: permutation, 주간 재현 모두 실패      (v3 p=0.037은 노이즈)",
    f"  v5: {verdict[:50]}",
    "=" * 65,
]

summary_text = "\n".join(summary)
print("\n" + summary_text)
with open("summary_v5.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

preds.to_csv("results_v5.csv", index=False)
print("\n  → summary_v5.txt, results_v5.csv 저장")
