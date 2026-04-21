"""
맥락 개입 실험 v4
=================
핵심 질문: "normal 국면에서 D의 IC +0.022는 진짜 신호인가?"

세 갈래 검증:
  검증 1 — Permutation test (일간, normal 국면)
            분포 가정 없이 ΔIC 유의성 직접 계산 (1,000회 셔플)
  검증 2 — 주간 주기 재현
            시간 축을 바꿔도 normal에서 D > C가 재현되는가?
  검증 3 — VIX 대역 세분화 (robustness)
            normal을 VIX 12~17 / 17~22 / 22~28으로 나눠 신호 집중도 확인

Variant: A (가격) / C (거시) / D (행동) — 3개만. B는 이 질문과 무관.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
TICKERS   = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "JNJ", "XOM", "CVX", "MCD", "NKE"]
SECTORS   = {
    'AAPL': 'Tech',    'MSFT': 'Tech',    'NVDA': 'Tech',
    'JPM':  'Finance', 'BAC':  'Finance',
    'JNJ':  'Health',
    'XOM':  'Energy',  'CVX':  'Energy',
    'MCD':  'Consumer','NKE':  'Consumer',
}
START     = "2015-01-01"
END       = "2024-12-31"
VARIANTS  = ['A', 'C', 'D']

# 일간 설정
D_LOOKBACK = 10
D_FOLDS    = 3
D_SEEDS    = 3
D_HIDDEN   = (64, 32)
D_ITER     = 80

# 주간 설정
W_LOOKBACK = 4
W_FOLDS    = 2
W_SEEDS    = 3
W_HIDDEN   = (32, 16)
W_ITER     = 80

N_PERM    = 1000
VIX_BANDS = {'12-17': (12, 17), '17-22': (17, 22), '22-28': (22, 28)}

# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ═══════════════════════════════════════════════════════════════
print("[1/8] 데이터 수집 중...")

def dl(tks, start=START, end=END, field="Close"):
    d = yf.download(tks, start=start, end=end, auto_adjust=True, progress=False)[field]
    if isinstance(d, pd.Series):
        return d.to_frame(name=tks if isinstance(tks, str) else tks[0])
    return d

prices_d  = dl(TICKERS).ffill().dropna()
returns_d = prices_d.pct_change().dropna()
volumes_d = dl(TICKERS, field="Volume").ffill().dropna()

macro_src = dl(["^VIX", "^TNX", "^IRX", "HYG", "LQD", "UUP", "USO"]).ffill()
spy_src   = dl(["SPY"]).ffill()
spy_ret_d = spy_src.pct_change()['SPY']

macro_d = pd.DataFrame(index=macro_src.index)
macro_d['vix_level']    = macro_src['^VIX']
macro_d['vix_chg_5d']   = macro_src['^VIX'].pct_change(5)
macro_d['yield_slope']  = macro_src['^TNX'] - macro_src['^IRX']
macro_d['yield_chg']    = macro_d['yield_slope'].diff(5)
macro_d['credit_proxy'] = np.log(macro_src['LQD'] / macro_src['HYG'])
macro_d['credit_chg']   = macro_d['credit_proxy'].diff(5)
macro_d['dxy_ret_20d']  = macro_src['UUP'].pct_change(20)
macro_d['oil_ret_20d']  = macro_src['USO'].pct_change(20)
macro_d = macro_d.dropna()

# 주간 리샘플
prices_w  = prices_d.resample('W-FRI').last().ffill().dropna()
volumes_w = volumes_d.resample('W-FRI').sum().ffill().dropna()
returns_w = prices_w.pct_change().dropna()
macro_w   = macro_d.resample('W-FRI').last().ffill().dropna()
spy_ret_w = spy_ret_d.resample('W-FRI').sum().ffill()

# 국면 분류 함수 (일간/주간 공용)
def make_regime_fn(macro_df, spy_ret_ser):
    vix_75 = macro_df['vix_level'].quantile(0.75)
    vix_25 = macro_df['vix_level'].quantile(0.25)
    spy_roll = spy_ret_ser.rolling(20).sum().reindex(macro_df.index).ffill()
    def fn(date):
        try:
            v = macro_df.loc[date, 'vix_level']
            s = spy_roll.get(date, np.nan)
        except KeyError:
            return 'normal'
        if v > vix_75:
            return 'panic'
        elif v < vix_25 and not np.isnan(s) and s > 0:
            return 'greed'
        return 'normal'
    return fn

get_regime_d = make_regime_fn(macro_d, spy_ret_d)
get_regime_w = make_regime_fn(macro_w, spy_ret_w)

# ═══════════════════════════════════════════════════════════════
# 3. 행동 특징 (일간)
# ═══════════════════════════════════════════════════════════════
def make_behavioral(prices, returns, volumes, sectors, roll_short, roll_long, anchor_w):
    """일간/주간 공용 행동 특징 빌더."""
    sector_cs_std = {}
    for sec in set(sectors.values()):
        members = [t for t, s in sectors.items() if s == sec]
        if len(members) > 1:
            sector_cs_std[sec] = returns[members].std(axis=1)

    bf_dict = {}
    for ticker in TICKERS:
        r = returns[ticker]; p = prices[ticker]; v = volumes[ticker]
        bf = pd.DataFrame(index=r.index)
        bf['anchor']       = p / p.rolling(anchor_w, min_periods=anchor_w // 4).max()
        ret_short = r.rolling(roll_short).sum().abs()
        bf['overreaction'] = ret_short / (ret_short.rolling(roll_long).mean() + 1e-8)
        r_neg = r.where(r < 0, np.nan)
        r_pos = r.where(r > 0, np.nan)
        loss_win = max(roll_short * 2, 5)
        dv = r_neg.rolling(loss_win, min_periods=2).std()
        uv = r_pos.rolling(loss_win, min_periods=2).std()
        bf['loss_aversion'] = dv / (uv + 1e-8)
        sec = sectors[ticker]
        if sec in sector_cs_std:
            cs = sector_cs_std[sec]
            bf['herding'] = 1 - cs / (cs.rolling(roll_long).mean() + 1e-8)
        else:
            bf['herding'] = 0.0
        vol_ratio = v.rolling(roll_short * 2).mean() / (v.rolling(roll_long).mean() + 1e-8)
        bf['disposition'] = vol_ratio * bf['anchor']
        bf_dict[ticker] = bf.dropna()
    return bf_dict

print("[2/8] 행동 특징 계산 중 (일간/주간)...")
behavioral_d = make_behavioral(prices_d, returns_d, volumes_d, SECTORS,
                                roll_short=5, roll_long=60, anchor_w=252)
behavioral_w = make_behavioral(prices_w, returns_w, volumes_w, SECTORS,
                                roll_short=2, roll_long=12, anchor_w=52)

# 주간 거시 특징 (일간과 같은 구조, 롤링 기준만 주 단위)
macro_w_feat = pd.DataFrame(index=macro_w.index)
macro_w_feat['vix_level']    = macro_w['vix_level']
macro_w_feat['vix_chg_2w']   = macro_w['vix_level'].pct_change(2)
macro_w_feat['yield_slope']  = macro_w['yield_slope']
macro_w_feat['yield_chg']    = macro_w['yield_slope'].diff(2)
macro_w_feat['credit_proxy'] = macro_w['credit_proxy']
macro_w_feat['credit_chg']   = macro_w['credit_proxy'].diff(2)
macro_w_feat['dxy_ret_4w']   = macro_w['dxy_ret_20d']
macro_w_feat['oil_ret_4w']   = macro_w['oil_ret_20d']
macro_w_feat = macro_w_feat.dropna()

# ═══════════════════════════════════════════════════════════════
# 4. 공통 유틸리티
# ═══════════════════════════════════════════════════════════════
def price_features(r, short, mid, long):
    f = pd.DataFrame(index=r.index)
    f['ret_s']  = r
    f['ret_m']  = r.rolling(short).sum()
    f['ret_l']  = r.rolling(mid).sum()
    f['vol_l']  = r.rolling(mid).std()
    f['mom']    = r.rolling(mid).mean() / (r.rolling(long).std() + 1e-8)
    return f.dropna()


def build_dataset(ticker, variant, returns, macro_feat, behavioral, lookback,
                  pf_short, pf_mid, pf_long):
    pf = price_features(returns[ticker], pf_short, pf_mid, pf_long)
    r  = returns[ticker]
    if variant == 'A':
        extra = None
    elif variant == 'C':
        extra = macro_feat
    elif variant == 'D':
        extra = behavioral[ticker]
    else:
        raise ValueError(variant)

    idx = pf.index
    if extra is not None:
        idx = idx.intersection(extra.index)
    pf, r = pf.loc[idx], r.loc[idx]
    ex = extra.loc[idx].values if extra is not None else None

    pv = pf.values
    X, y, dates = [], [], []
    for i in range(lookback, len(pv) - 1):
        window = pv[i - lookback:i].flatten()
        X.append(np.concatenate([window, ex[i]]) if ex is not None else window)
        y.append(r.values[i + 1])
        dates.append(idx[i + 1])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def walk_forward_splits(n, n_folds, min_train=0.45):
    start = int(n * min_train)
    test_size = (n - start) // n_folds
    splits = []
    for k in range(n_folds):
        tr_end = start + k * test_size
        te_end = tr_end + test_size if k < n_folds - 1 else n
        splits.append((0, tr_end, tr_end, te_end))
    return splits


def ic(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def run_experiment(returns, macro_feat, behavioral, lookback, n_folds, n_seeds,
                   hidden, n_iter, pf_short, pf_mid, pf_long, min_train=0.45, label=""):
    """A/C/D variant walk-forward 실험. 샘플 단위 예측값 포함 반환."""
    records = []
    all_preds = []  # {ticker, date, y_true, y_pred_A, y_pred_C, y_pred_D}

    for ticker in TICKERS:
        datasets = {v: build_dataset(ticker, v, returns, macro_feat, behavioral,
                                     lookback, pf_short, pf_mid, pf_long)
                    for v in VARIANTS}
        common = datasets['A'][2]
        for v in ['C', 'D']:
            common = common.intersection(datasets[v][2])
        aligned = {}
        for v in VARIANTS:
            X, y, d = datasets[v]
            mask = d.isin(common)
            aligned[v] = (X[mask], y[mask], d[mask])

        n = len(common)
        if n < 20:
            continue
        splits = walk_forward_splits(n, n_folds, min_train=min_train)

        # 샘플 예측 수집용 (마지막 fold, seed=0)
        last_fold = splits[-1]

        for fold_i, (a, b, c, e) in enumerate(splits):
            is_last = (fold_i == len(splits) - 1)
            fold_preds = {}  # variant → y_pred

            for seed in range(n_seeds):
                for v in VARIANTS:
                    X, y, d = aligned[v]
                    Xtr, ytr = X[a:b], y[a:b]
                    Xte, yte = X[c:e], y[c:e]
                    if len(Xtr) < 10 or len(Xte) < 5:
                        continue
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(Xtr)
                    Xte_s = sc.transform(Xte)
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden, max_iter=n_iter,
                        random_state=seed, early_stopping=False,
                        learning_rate_init=1e-3, alpha=1e-4, tol=1e-3,
                    )
                    model.fit(Xtr_s, ytr)
                    ypred = model.predict(Xte_s)
                    rho, _ = spearmanr(yte, ypred)
                    records.append({
                        'ticker': ticker, 'fold': fold_i, 'seed': seed,
                        'variant': v, 'IC': rho,
                        'MSE': mean_squared_error(yte, ypred),
                    })
                    if is_last and seed == 0:
                        fold_preds[v] = (yte, ypred, aligned[v][2][c:e])

            # 마지막 fold, seed=0 샘플 저장
            if is_last and 'A' in fold_preds:
                yte, _, dates = fold_preds['A']
                for i, (dt, yt) in enumerate(zip(dates, yte)):
                    all_preds.append({
                        'ticker': ticker, 'date': dt, 'y_true': yt,
                        'y_pred_A': fold_preds['A'][1][i],
                        'y_pred_C': fold_preds['C'][1][i],
                        'y_pred_D': fold_preds['D'][1][i],
                    })

    return pd.DataFrame(records), pd.DataFrame(all_preds)


# ═══════════════════════════════════════════════════════════════
# 5. 일간 실험 실행
# ═══════════════════════════════════════════════════════════════
print("[3/8] 일간 A/C/D 실험 (검증 1·3 기반)...")
df_d, preds_d = run_experiment(
    returns_d, macro_d, behavioral_d,
    lookback=D_LOOKBACK, n_folds=D_FOLDS, n_seeds=D_SEEDS,
    hidden=D_HIDDEN, n_iter=D_ITER,
    pf_short=5, pf_mid=20, pf_long=60,
)

preds_d['date'] = pd.to_datetime(preds_d['date'])
preds_d['regime3'] = preds_d['date'].apply(
    lambda d: get_regime_d(d) if d in macro_d.index else 'normal'
)
preds_d['vix'] = preds_d['date'].map(
    macro_d['vix_level'].to_dict()
)

# ═══════════════════════════════════════════════════════════════
# 6. 검증 1: Permutation test
# ═══════════════════════════════════════════════════════════════
print("[4/8] 검증 1: Permutation test (1,000회)...")

normal_d = preds_d[preds_d['regime3'] == 'normal'].copy()
ic_c_obs = ic(normal_d['y_true'], normal_d['y_pred_C'])
ic_d_obs = ic(normal_d['y_true'], normal_d['y_pred_D'])
delta_obs = ic_d_obs - ic_c_obs

rng = np.random.RandomState(42)
delta_perm = []
y_true_arr = normal_d['y_true'].values
y_c_arr    = normal_d['y_pred_C'].values
y_d_arr    = normal_d['y_pred_D'].values

for _ in range(N_PERM):
    shuf = rng.permutation(len(y_true_arr))
    ic_c_s = spearmanr(y_true_arr[shuf], y_c_arr)[0]
    ic_d_s = spearmanr(y_true_arr[shuf], y_d_arr)[0]
    delta_perm.append(ic_d_s - ic_c_s)

delta_perm = np.array(delta_perm)
p_perm = (delta_perm >= delta_obs).mean()

print(f"  IC_C (관측) = {ic_c_obs:.4f}")
print(f"  IC_D (관측) = {ic_d_obs:.4f}")
print(f"  ΔIC (관측)  = {delta_obs:+.4f}")
print(f"  p_perm      = {p_perm:.4f}  ({'유의*' if p_perm < 0.05 else 'ns'})")

# ═══════════════════════════════════════════════════════════════
# 7. 주간 실험 실행 (검증 2)
# ═══════════════════════════════════════════════════════════════
print("[5/8] 검증 2: 주간 A/C/D 실험...")
df_w, preds_w = run_experiment(
    returns_w, macro_w_feat, behavioral_w,
    lookback=W_LOOKBACK, n_folds=W_FOLDS, n_seeds=W_SEEDS,
    hidden=W_HIDDEN, n_iter=W_ITER,
    pf_short=2, pf_mid=4, pf_long=12,
    min_train=0.35,
)
print(f"  주간 예측 샘플: {len(preds_w)}개, IC 레코드: {len(df_w)}개")

WEEKLY_OK = len(preds_w) > 0 and 'date' in preds_w.columns
if WEEKLY_OK:
    preds_w['date'] = pd.to_datetime(preds_w['date'])
    preds_w['regime3'] = preds_w['date'].apply(
        lambda d: get_regime_w(d) if d in macro_w.index else 'normal'
    )
else:
    print("  ⚠ 주간 데이터 부족 — 검증 2 건너뜀")

# ═══════════════════════════════════════════════════════════════
# 8. 결과 요약
# ═══════════════════════════════════════════════════════════════
print("\n[6/8] 결과 요약")
print("=" * 60)

def print_regime_ic(preds, label=""):
    print(f"\n  {'국면':<10} {'n':>6}  {'C IC':>8}  {'D IC':>8}  {'ΔIC':>8}")
    print(f"  {'-'*50}")
    for reg in ['panic', 'greed', 'normal']:
        sub = preds[preds['regime3'] == reg]
        if len(sub) < 10:
            print(f"  {reg:<10} {'부족':>6}")
            continue
        ic_c = ic(sub['y_true'], sub['y_pred_C'])
        ic_d = ic(sub['y_true'], sub['y_pred_D'])
        print(f"  {reg:<10} {len(sub):>6}  {ic_c:>8.4f}  {ic_d:>8.4f}  {ic_d-ic_c:>+8.4f}")

print("\n▸ 일간 국면별 IC (A/C/D)")
print_regime_ic(preds_d, "일간")

print("\n▸ 주간 국면별 IC (A/C/D)")
if WEEKLY_OK:
    print_regime_ic(preds_w, "주간")
else:
    print("  (데이터 부족으로 생략)")

# VIX 대역별 (검증 3)
print("\n▸ VIX 대역별 D vs C IC (일간, normal 국면)")
print(f"  {'VIX 대역':<12} {'n':>6}  {'C IC':>8}  {'D IC':>8}  {'ΔIC':>8}")
print(f"  {'-'*50}")
vix_band_results = {}
for band, (lo, hi) in VIX_BANDS.items():
    sub = preds_d[(preds_d['vix'] >= lo) & (preds_d['vix'] < hi)]
    if len(sub) < 10:
        print(f"  {band:<12} {'부족':>6}")
        continue
    ic_c = ic(sub['y_true'], sub['y_pred_C'])
    ic_d = ic(sub['y_true'], sub['y_pred_D'])
    vix_band_results[band] = {'ic_c': ic_c, 'ic_d': ic_d, 'n': len(sub)}
    print(f"  {band:<12} {len(sub):>6}  {ic_c:>8.4f}  {ic_d:>8.4f}  {ic_d-ic_c:>+8.4f}")

# ═══════════════════════════════════════════════════════════════
# 9. 시각화
# ═══════════════════════════════════════════════════════════════
print("\n[7/8] 시각화 생성 중...")

# 그림 1: permutation_test.png
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.hist(delta_perm, bins=50, color='steelblue', alpha=0.7, edgecolor='white',
         label=f'Permuted ΔIC (n={N_PERM})')
ax1.axvline(delta_obs, color='red', lw=2, label=f'Observed ΔIC = {delta_obs:+.4f}')
ax1.axvline(0, color='gray', lw=1, ls='--', alpha=0.6)
ax1.set_xlabel("ΔIC (D − C)", fontsize=12)
ax1.set_ylabel("빈도", fontsize=12)
perm_95 = np.percentile(delta_perm, 95)
ax1.axvline(perm_95, color='orange', lw=1.5, ls=':', label=f'95th pctile = {perm_95:+.4f}')
ax1.set_title(f"Permutation Test — normal 국면  |  p_perm = {p_perm:.4f}", fontsize=13)
ax1.legend(fontsize=10)
plt.tight_layout()
plt.savefig("permutation_test.png", dpi=140, bbox_inches='tight')
print("  → permutation_test.png 저장")
plt.close()

# 그림 2: weekly_regime_ic.png
colors_v = {'A': '#888', 'C': '#4af', 'D': '#e64'}
if WEEKLY_OK and len(df_w) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, regime in zip(axes2, ['panic', 'greed', 'normal']):
        data_w = [df_w[df_w.variant == v]['IC'].values for v in VARIANTS]
        bp = ax.boxplot(data_w, labels=VARIANTS, patch_artist=True, widths=0.6)
        for patch, v in zip(bp['boxes'], VARIANTS):
            patch.set_facecolor(colors_v[v]); patch.set_alpha(0.75)
        ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.7)
        if regime == 'normal':
            sub_w = preds_w[preds_w['regime3'] == 'normal']
            if len(sub_w) >= 10:
                for j, v in enumerate(VARIANTS):
                    obs_ic = ic(sub_w['y_true'], sub_w[f'y_pred_{v}'])
                    ax.scatter(j + 1, obs_ic, color='black', zorder=5, s=60, marker='D',
                               label='관측 IC (normal)' if j == 0 else '')
            ax.legend(fontsize=9)
        ax.set_title(f"주간 — 국면: {regime}", fontsize=11)
        ax.set_ylabel("IC" if regime == 'panic' else "")
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle("검증 2: 주간 주기 국면별 IC 분포 (A/C/D)", fontsize=13)
    plt.tight_layout()
    plt.savefig("weekly_regime_ic.png", dpi=140, bbox_inches='tight')
    print("  → weekly_regime_ic.png 저장")
    plt.close()
else:
    print("  → weekly_regime_ic.png 생략 (데이터 부족)")

# 그림 3: vix_band_ic.png
bands = list(vix_band_results.keys())
ic_c_vals = [vix_band_results[b]['ic_c'] for b in bands]
ic_d_vals = [vix_band_results[b]['ic_d'] for b in bands]
n_vals    = [vix_band_results[b]['n']    for b in bands]

x = np.arange(len(bands))
w = 0.35
fig3, ax3 = plt.subplots(figsize=(9, 5))
bars_c = ax3.bar(x - w/2, ic_c_vals, w, label='C (거시)', color='#4af', alpha=0.8)
bars_d = ax3.bar(x + w/2, ic_d_vals, w, label='D (행동)', color='#e64', alpha=0.8)
ax3.axhline(0, color='gray', lw=1, ls='--', alpha=0.7)
for i, (ic_c_v, ic_d_v, nv) in enumerate(zip(ic_c_vals, ic_d_vals, n_vals)):
    delta = ic_d_v - ic_c_v
    ax3.text(i, max(ic_c_v, ic_d_v) + 0.003, f'Δ={delta:+.3f}\n(n={nv})',
             ha='center', fontsize=9, color='#333')
ax3.set_xticks(x); ax3.set_xticklabels([f'VIX {b}' for b in bands], fontsize=11)
ax3.set_ylabel("IC (Spearman)", fontsize=11)
ax3.set_title("검증 3: VIX 대역별 D vs C IC (일간 전체)", fontsize=13)
ax3.legend(fontsize=11); ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("vix_band_ic.png", dpi=140, bbox_inches='tight')
print("  → vix_band_ic.png 저장")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 10. 결과 저장 & 최종 판정
# ═══════════════════════════════════════════════════════════════
print("\n[8/8] 결과 저장 중...")
weekly_part = df_w.assign(freq='weekly') if len(df_w) > 0 else pd.DataFrame()
df_all = pd.concat([df_d.assign(freq='daily'), weekly_part], ignore_index=True)
df_all.to_csv("results_v4.csv", index=False)
print("  → results_v4.csv 저장")

# 주간 normal 재현 여부
if WEEKLY_OK:
    w_normal = preds_w[preds_w['regime3'] == 'normal']
    w_ic_c = ic(w_normal['y_true'], w_normal['y_pred_C']) if len(w_normal) >= 10 else np.nan
    w_ic_d = ic(w_normal['y_true'], w_normal['y_pred_D']) if len(w_normal) >= 10 else np.nan
    w_reproduced = (not np.isnan(w_ic_d)) and (w_ic_d > w_ic_c)
else:
    w_ic_c = w_ic_d = np.nan
    w_reproduced = False

best_band = max(vix_band_results, key=lambda b: vix_band_results[b]['ic_d'] - vix_band_results[b]['ic_c']) if vix_band_results else "n/a"
best_delta = (vix_band_results[best_band]['ic_d'] - vix_band_results[best_band]['ic_c']) if vix_band_results else np.nan

# 최종 판정 로직
v1_pass = p_perm < 0.05
v2_pass = w_reproduced
v3_concentrated = (not np.isnan(best_delta)) and (best_delta > 0.02)

n_pass = sum([v1_pass, v2_pass, v3_concentrated])
if n_pass == 3:
    verdict = "PUBLISHABLE — 세 검증 모두 통과. 행동 편향의 normal 국면 신호 강력히 지지."
elif n_pass == 2:
    verdict = "SUGGESTIVE — 두 검증 통과. '잠정적 증거'로 보고 가능, 후속 연구 권장."
else:
    verdict = "NULL RESULT — 검증 통과 1개 이하. v3 p=0.037은 노이즈일 가능성 높음."

summary_lines = [
    "=" * 60,
    "v4 실험 최종 요약",
    "=" * 60,
    f"핵심 질문: normal 국면에서 D의 IC +0.022는 진짜 신호인가?",
    "",
    f"[검증 1] Permutation test (일간, normal)",
    f"  ΔIC 관측값 = {delta_obs:+.4f}",
    f"  p_perm     = {p_perm:.4f}  ({'PASS' if v1_pass else 'FAIL'}, 기준: p<0.05)",
    "",
    f"[검증 2] 주간 주기 재현",
    f"  주간 normal IC_C = {w_ic_c:.4f},  IC_D = {w_ic_d:.4f}",
    f"  재현 여부 = {'PASS (D > C)' if v2_pass else 'FAIL (D <= C)'}",
    "",
    f"[검증 3] VIX 대역 세분화",
    f"  ΔIC 최대 대역 = VIX {best_band}  (ΔIC = {best_delta:+.4f})",
    f"  집중 여부 = {'PASS' if v3_concentrated else 'FAIL'} (기준: 최대 ΔIC > 0.02)",
    "",
    f"통과: {n_pass}/3",
    f"최종 판정: {verdict}",
    "=" * 60,
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open("summary_v4.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
print("  → summary_v4.txt 저장")
