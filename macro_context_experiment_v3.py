"""
맥락 개입 실험 v3
=================
가설: 시장은 자기 자신의 비합리성(행동적 편향)을 교정하는가?

v2가 증명: 합리적 공개 정보(거시지표 C)는 베타(B) 대비 잔여 예측력 없음 → semi-strong EMH 확인
v3가 묻는 것: 비합리적 행동 패턴(D)은 어떤가?

4단계 모델:
  A: 가격 특징만                       (자기회귀 baseline)
  B: A + 지수/섹터                     (베타 암기 수준)
  C: A + 외생 거시 (VIX, 금리 등)      (합리적 정보, v2에서 실패 확인)
  D: A + 행동적 편향                   (비합리성이 잔여 신호를 가지는가)

D 변수 (행동경제학 이론 대응):
  anchor        : 52주 고점 대비 위치 (앵커링, George & Hwang 2004)
  overreaction  : 5일 수익률 극단성 (과잉반응, De Bondt & Thaler)
  loss_aversion : 하방/상방 변동성 비율 (손실 회피, Kahneman & Tversky)
  herding       : 섹터 내 횡단면 분산 압축 (군집 행동)
  disposition   : 거래량 급증 × 고점 근접 (처분 효과, Shefrin & Statman)

국면 분석 (3분류):
  panic  : VIX > 75th percentile
  greed  : VIX < 25th percentile & SPY 20일 수익 > 0
  normal : 나머지

통계: Bonferroni 보정 (p < 0.0167 = 0.05/3) 적용

필요 패키지: yfinance, numpy, pandas, scikit-learn, scipy, matplotlib
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.stats import spearmanr, ttest_rel

# ═══════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════
TICKERS   = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "JNJ", "XOM", "CVX", "MCD", "NKE"]
SECTORS   = {
    'AAPL': 'Tech',   'MSFT': 'Tech',   'NVDA': 'Tech',
    'JPM':  'Finance','BAC':  'Finance',
    'JNJ':  'Health',
    'XOM':  'Energy', 'CVX':  'Energy',
    'MCD':  'Consumer','NKE': 'Consumer',
}
START     = "2015-01-01"
END       = "2024-12-31"
LOOKBACK  = 10
N_FOLDS   = 3
N_SEEDS   = 3
MIN_TRAIN = 0.45
HIDDEN    = (64, 32)
MAX_ITER  = 80
BONFERRONI_ALPHA = 0.05 / 3   # 0.0167

# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ═══════════════════════════════════════════════════════════════
print("[1/7] 데이터 수집 중...")

def dl(tks, start=START, end=END, field="Close"):
    d = yf.download(tks, start=start, end=end, auto_adjust=True, progress=False)[field]
    if isinstance(d, pd.Series):
        return d.to_frame(name=tks if isinstance(tks, str) else tks[0])
    return d

prices  = dl(TICKERS).ffill().dropna()
returns = prices.pct_change().dropna()
volumes = dl(TICKERS, field="Volume").ffill().dropna()

beta_src = dl(["SPY", "XLK", "XLE", "XLF"]).ffill()
beta_ret = beta_src.pct_change()
spy_ret  = beta_ret['SPY']

macro_src = dl(["^VIX", "^TNX", "^IRX", "HYG", "LQD", "UUP", "USO"]).ffill()

macro = pd.DataFrame(index=macro_src.index)
macro['vix_level']   = macro_src['^VIX']
macro['vix_chg_5d']  = macro_src['^VIX'].pct_change(5)
macro['yield_slope'] = macro_src['^TNX'] - macro_src['^IRX']
macro['yield_chg']   = macro['yield_slope'].diff(5)
macro['credit_proxy']= np.log(macro_src['LQD'] / macro_src['HYG'])
macro['credit_chg']  = macro['credit_proxy'].diff(5)
macro['dxy_ret_20d'] = macro_src['UUP'].pct_change(20)
macro['oil_ret_20d'] = macro_src['USO'].pct_change(20)
macro = macro.dropna()

beta_feat = pd.DataFrame(index=beta_ret.index)
for col in beta_ret.columns:
    beta_feat[f'{col}_r20'] = beta_ret[col].rolling(20).sum()
    beta_feat[f'{col}_v20'] = beta_ret[col].rolling(20).std()
beta_feat = beta_feat.dropna()

# ═══════════════════════════════════════════════════════════════
# 3. 행동적 편향 특징 엔지니어링 (D variant)
# ═══════════════════════════════════════════════════════════════
print("[2/7] 행동적 편향 특징 계산 중...")

# 섹터별 횡단면 수익률 표준편차 (herding용)
sector_cs_std = {}
for sec in set(SECTORS.values()):
    members = [t for t, s in SECTORS.items() if s == sec]
    if len(members) > 1:
        sec_rets = returns[members]
        cs_std = sec_rets.std(axis=1)
        sector_cs_std[sec] = cs_std

behavioral_feat = {}
for ticker in TICKERS:
    r = returns[ticker]
    p = prices[ticker]
    v = volumes[ticker]

    bf = pd.DataFrame(index=r.index)

    # ① 앵커링: 현재가 / 52주 고점
    bf['anchor'] = p / p.rolling(252, min_periods=60).max()

    # ② 과잉반응: |ret_5d| / 60일 평균 |ret_5d|
    ret5 = r.rolling(5).sum().abs()
    bf['overreaction'] = ret5 / (ret5.rolling(60).mean() + 1e-8)

    # ③ 손실 회피: 하방 변동성 / 상방 변동성
    r_neg = r.where(r < 0, np.nan)
    r_pos = r.where(r > 0, np.nan)
    dv = r_neg.rolling(20, min_periods=5).std()
    uv = r_pos.rolling(20, min_periods=5).std()
    bf['loss_aversion'] = dv / (uv + 1e-8)

    # ④ 군집 행동: 1 - (섹터 내 횡단면 분산 / 60일 평균)
    sec = SECTORS[ticker]
    if sec in sector_cs_std:
        cs = sector_cs_std[sec]
        bf['herding'] = 1 - cs / (cs.rolling(60).mean() + 1e-8)
    else:
        bf['herding'] = 0.0

    # ⑤ 처분 효과: (단기 거래량 / 장기 거래량) × 고점 근접도
    vol_ratio = v.rolling(20).mean() / (v.rolling(60).mean() + 1e-8)
    bf['disposition'] = vol_ratio * bf['anchor']

    behavioral_feat[ticker] = bf.dropna()

# ═══════════════════════════════════════════════════════════════
# 4. 국면 라벨 (3분류)
# ═══════════════════════════════════════════════════════════════
vix_75 = macro['vix_level'].quantile(0.75)
vix_25 = macro['vix_level'].quantile(0.25)
spy_20 = spy_ret.rolling(20).sum().reindex(macro.index).ffill()

def get_regime3(date):
    try:
        v = macro.loc[date, 'vix_level']
        s = spy_20.get(date, np.nan)
    except KeyError:
        return 'normal'
    if v > vix_75:
        return 'panic'
    elif v < vix_25 and (not np.isnan(s)) and s > 0:
        return 'greed'
    else:
        return 'normal'

# ═══════════════════════════════════════════════════════════════
# 5. 특징 엔지니어링 (공통)
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
    pf = price_features(returns[ticker])
    r  = returns[ticker]

    if variant == 'A':
        extra = None
    elif variant == 'B':
        extra = beta_feat
    elif variant == 'C':
        extra = macro
    elif variant == 'D':
        extra = behavioral_feat[ticker]
    else:
        raise ValueError(variant)

    idx = pf.index
    if extra is not None:
        idx = idx.intersection(extra.index)
    pf, r = pf.loc[idx], r.loc[idx]
    ex = extra.loc[idx].values if extra is not None else None

    pv = pf.values
    X, y, dates = [], [], []
    for i in range(LOOKBACK, len(pv) - 1):
        window = pv[i - LOOKBACK:i].flatten()
        if ex is not None:
            X.append(np.concatenate([window, ex[i]]))
        else:
            X.append(window)
        y.append(r.values[i + 1])
        dates.append(idx[i + 1])

    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


# ═══════════════════════════════════════════════════════════════
# 6. 평가 지표
# ═══════════════════════════════════════════════════════════════
def directional_accuracy(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def evaluate(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'DA':  directional_accuracy(y_true, y_pred),
        'IC':  information_coefficient(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════
# 7. Walk-forward split
# ═══════════════════════════════════════════════════════════════
def walk_forward_splits(n, n_folds=N_FOLDS, min_train=MIN_TRAIN):
    start_train_end = int(n * min_train)
    remaining       = n - start_train_end
    test_size       = remaining // n_folds
    splits = []
    for k in range(n_folds):
        tr_end = start_train_end + k * test_size
        te_end = tr_end + test_size if k < n_folds - 1 else n
        splits.append((0, tr_end, tr_end, te_end))
    return splits


# ═══════════════════════════════════════════════════════════════
# 8. MLP latent 추출 & CKA
# ═══════════════════════════════════════════════════════════════
def relu(x): return np.maximum(0, x)


def get_latent(model, X):
    h = X
    for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = relu(h @ W + b)
    return h


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    nx   = np.linalg.norm(X.T @ X, 'fro')
    ny   = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic / (nx * ny + 1e-12)


# ═══════════════════════════════════════════════════════════════
# 9. 메인 실험 루프
# ═══════════════════════════════════════════════════════════════
print("[3/7] 모델 학습 및 평가 (이 단계가 가장 오래 걸림)...")

VARIANTS = ['A', 'B', 'C', 'D']
records  = []

latent_store = {v: [] for v in VARIANTS}
label_store  = {
    'ticker': [], 'date': [], 'y_true': [],
    'y_pred_A': [], 'y_pred_B': [], 'y_pred_C': [], 'y_pred_D': []
}

for ticker in TICKERS:
    datasets = {v: build_dataset(ticker, v) for v in VARIANTS}
    common = datasets['A'][2]
    for v in ['B', 'C', 'D']:
        common = common.intersection(datasets[v][2])

    aligned = {}
    for v in VARIANTS:
        X, y, d = datasets[v]
        mask = d.isin(common)
        aligned[v] = (X[mask], y[mask], d[mask])

    n = len(common)
    splits = walk_forward_splits(n)

    for fold_i, (a, b, c, e) in enumerate(splits):
        is_last_fold = (fold_i == len(splits) - 1)

        for seed in range(N_SEEDS):
            for v in VARIANTS:
                X, y, d = aligned[v]
                X_tr, y_tr = X[a:b], y[a:b]
                X_te, y_te = X[c:e], y[c:e]

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                model = MLPRegressor(
                    hidden_layer_sizes=HIDDEN,
                    max_iter=MAX_ITER,
                    random_state=seed,
                    early_stopping=False,
                    learning_rate_init=1e-3,
                    alpha=1e-4,
                    tol=1e-3,
                )
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)

                m = evaluate(y_te, y_pred)
                m.update({'ticker': ticker, 'fold': fold_i,
                          'seed': seed, 'variant': v})
                records.append(m)

                if is_last_fold and seed == 0:
                    lat = get_latent(model, X_te_s)
                    latent_store[v].append(lat)
                    if v == 'A':
                        label_store['ticker'].extend([ticker] * len(y_te))
                        label_store['date'].extend(d[c:e])
                        label_store['y_true'].extend(y_te)
                        label_store['y_pred_A'].extend(y_pred)
                    elif v == 'B':
                        label_store['y_pred_B'].extend(y_pred)
                    elif v == 'C':
                        label_store['y_pred_C'].extend(y_pred)
                    else:
                        label_store['y_pred_D'].extend(y_pred)

    # constant-zero baseline
    for fold_i, (a, b, c, e) in enumerate(splits):
        _, y, _ = aligned['A']
        y_te = y[c:e]
        m = evaluate(y_te, np.zeros_like(y_te))
        m.update({'ticker': ticker, 'fold': fold_i, 'seed': -1, 'variant': 'ZERO'})
        records.append(m)

df = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════
# 10. 결과 요약
# ═══════════════════════════════════════════════════════════════
print("\n[4/7] 결과 요약")
print("=" * 70)

summary = df.groupby('variant')[['MSE', 'DA', 'IC']].agg(['mean', 'std'])
print("\n▸ Variant별 성능 (전 시드·fold·종목 통합)")
print(summary.round(5).to_string())

per_ticker = (df[~df.variant.isin(['ZERO'])]
              .groupby(['ticker', 'variant'])[['MSE', 'DA', 'IC']]
              .mean().round(5))
print("\n▸ 종목 × Variant 평균")
print(per_ticker.to_string())

print("\n▸ Paired t-test (전체, IC 기준)")
piv = (df[~df.variant.isin(['ZERO'])]
       .pivot_table(index=['ticker', 'fold', 'seed'], columns='variant', values='IC'))
for a_v, b_v in [('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D'), ('A', 'D')]:
    paired = piv[[a_v, b_v]].dropna()
    t, p = ttest_rel(paired[b_v], paired[a_v])
    diff  = (paired[b_v] - paired[a_v]).mean()
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {b_v} - {a_v}:  ΔIC = {diff:+.4f},  t = {t:+.3f},  p = {p:.4f}  [{sig}]")

zero_mse = df[df.variant == 'ZERO']['MSE'].mean()
print("\n▸ constant-zero baseline 대비")
for v in VARIANTS:
    m = df[df.variant == v]['MSE'].mean()
    print(f"  {v}: MSE = {m:.6f}  (zero = {zero_mse:.6f},  "
          f"개선 {100*(zero_mse-m)/zero_mse:+.2f}%)")

# ═══════════════════════════════════════════════════════════════
# 11. 국면별 C vs D 검정 (핵심)
# ═══════════════════════════════════════════════════════════════
print("\n[5/7] 국면별 C vs D 분석 (Bonferroni p < 0.0167)")

labels_df = pd.DataFrame(label_store)
labels_df['date'] = pd.to_datetime(labels_df['date'])

# 국면 3분류 적용
labels_df['regime3'] = labels_df['date'].apply(
    lambda d: get_regime3(d) if d in macro.index else 'normal'
)

# 날짜 기준으로 df에도 regime3 붙이기 (마지막 fold 날짜와 매핑)
last_fold_dates = set(labels_df['date'])
date_regime_map = dict(zip(labels_df['date'], labels_df['regime3']))

df_last = df[df.seed == 0]  # seed=0 기준으로 날짜 정보가 있는 레코드 활용

print(f"\n  {'국면':<10} {'샘플수':>8}  {'C IC':>10}  {'D IC':>10}  {'ΔIC':>10}  {'p':>10}  {'유의'}")
print(f"  {'-'*70}")
for regime in ['panic', 'greed', 'normal']:
    regime_dates = labels_df[labels_df['regime3'] == regime]['date']
    # 해당 날짜 비율
    n_regime = len(labels_df[labels_df['regime3'] == regime])
    # IC 계산: 해당 국면 날짜의 y_true / y_pred_C / y_pred_D로 Spearman
    sub = labels_df[labels_df['regime3'] == regime]
    if len(sub) < 30:
        print(f"  {regime:<10} {'샘플 부족':>8}")
        continue
    ic_c, _ = spearmanr(sub['y_true'], sub['y_pred_C'])
    ic_d, _ = spearmanr(sub['y_true'], sub['y_pred_D'])
    # 종목 단위 paired t-test
    ic_pairs = []
    for tk in TICKERS:
        sub_tk = sub[sub['ticker'] == tk]
        if len(sub_tk) < 10:
            continue
        c_ic, _ = spearmanr(sub_tk['y_true'], sub_tk['y_pred_C'])
        d_ic, _ = spearmanr(sub_tk['y_true'], sub_tk['y_pred_D'])
        ic_pairs.append((c_ic, d_ic))
    if len(ic_pairs) < 3:
        sig_str = "n/a"
        p_val = np.nan
    else:
        arr = np.array(ic_pairs)
        t_stat, p_val = ttest_rel(arr[:, 1], arr[:, 0])
        sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < BONFERRONI_ALPHA else "ns"
    print(f"  {regime:<10} {n_regime:>8}  {ic_c:>10.4f}  {ic_d:>10.4f}  "
          f"{ic_d - ic_c:>+10.4f}  {p_val:>10.4f}  [{sig_str}]")

# ═══════════════════════════════════════════════════════════════
# 12. Latent space 분석
# ═══════════════════════════════════════════════════════════════
print("\n[6/7] Latent space 분석")

L = {v: np.vstack(latent_store[v]) for v in VARIANTS}

vix_vals = macro['vix_level'].reindex(labels_df['date']).values
vix_median = np.nanmedian(vix_vals)
labels_df['regime'] = np.where(vix_vals > vix_median, 'high_vol', 'low_vol')
labels_df['direction'] = np.where(labels_df['y_true'] > 0, 'up', 'down')

# 편향 강도 3분위 (overreaction 기준)
or_vals = []
for i, row in labels_df.iterrows():
    tk = row['ticker']
    dt = row['date']
    bf = behavioral_feat[tk]
    val = bf['overreaction'].get(dt, np.nan) if dt in bf.index else np.nan
    or_vals.append(val)
labels_df['overreaction_val'] = or_vals
labels_df['bias_intensity'] = pd.qcut(
    labels_df['overreaction_val'], q=3, labels=['low', 'mid', 'high'], duplicates='drop'
)

print("\n▸ Silhouette score (클러스터 분리도)")
print(f"  {'라벨':<22}{'A':>10}{'B':>10}{'C':>10}{'D':>10}")
for label_col in ['ticker', 'regime', 'direction', 'regime3', 'bias_intensity']:
    row_str = [f"  {label_col:<22}"]
    lbl_arr = labels_df[label_col].astype(str).values
    for v in VARIANTS:
        n = len(L[v])
        idx = np.random.RandomState(0).choice(n, min(n, 2000), replace=False)
        try:
            s = silhouette_score(L[v][idx], lbl_arr[idx])
        except Exception:
            s = np.nan
        row_str.append(f"{s:>10.4f}")
    print("".join(row_str))

print("\n▸ Linear CKA (0~1, 두 표현이 얼마나 비슷한가)")
for a_v, b_v in [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]:
    print(f"  CKA({a_v}, {b_v}) = {linear_cka(L[a_v], L[b_v]):.4f}")
print("  → CKA(A,D)가 낮을수록 행동 변수가 표현을 재조직했다는 증거")

# ═══════════════════════════════════════════════════════════════
# 13. 시각화
# ═══════════════════════════════════════════════════════════════
print("\n[7/7] 시각화 생성 중...")

title_map = {
    'A': 'A: 가격만',
    'B': 'B: +지수/섹터',
    'C': 'C: +외생 거시',
    'D': 'D: +행동 편향',
}
color_maps = {
    'regime':    {'high_vol': '#d62728', 'low_vol': '#1f77b4'},
    'direction': {'up': '#2ca02c', 'down': '#ff7f0e'},
}

# 그림 1: latent_v3.png — 2×4 국면/방향 색칠
fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10))
for col_i, v in enumerate(VARIANTS):
    Lv2 = PCA(n_components=2).fit_transform(L[v])
    for reg in ['low_vol', 'high_vol']:
        mask = (labels_df['regime'] == reg).values
        axes1[0, col_i].scatter(Lv2[mask, 0], Lv2[mask, 1],
                                c=color_maps['regime'][reg], label=reg,
                                alpha=0.4, s=6, edgecolors='none')
    axes1[0, col_i].set_title(f"{title_map[v]}  |  VIX 국면", fontsize=11)
    axes1[0, col_i].legend(fontsize=8)

    for dirn in ['down', 'up']:
        mask = (labels_df['direction'] == dirn).values
        axes1[1, col_i].scatter(Lv2[mask, 0], Lv2[mask, 1],
                                c=color_maps['direction'][dirn], label=dirn,
                                alpha=0.4, s=6, edgecolors='none')
    axes1[1, col_i].set_title(f"{title_map[v]}  |  익일 방향", fontsize=11)
    axes1[1, col_i].legend(fontsize=8)

plt.suptitle("Latent Space 비교: A/B/C/D (PCA 2D)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("latent_v3.png", dpi=130, bbox_inches='tight')
print("  → latent_v3.png 저장")
plt.close()

# 그림 2: latent_trajectory.png — D 모델의 시간 궤적 + 3국면 색칠
fig2, ax2 = plt.subplots(figsize=(10, 7))
Ld = L['D']
Ld2 = PCA(n_components=2).fit_transform(Ld)
regime3_arr = labels_df['regime3'].values
date_arr = labels_df['date'].values
sort_idx = np.argsort(date_arr)
Ld2_sorted = Ld2[sort_idx]
r3_sorted  = regime3_arr[sort_idx]

regime_colors3 = {'panic': '#d62728', 'greed': '#2ca02c', 'normal': '#aaaaaa'}
for i in range(len(Ld2_sorted) - 1):
    c = regime_colors3[r3_sorted[i]]
    ax2.plot(Ld2_sorted[i:i+2, 0], Ld2_sorted[i:i+2, 1],
             color=c, alpha=0.25, linewidth=0.6)

for reg, c in regime_colors3.items():
    mask = r3_sorted == reg
    ax2.scatter(Ld2_sorted[mask, 0], Ld2_sorted[mask, 1],
                c=c, label=reg, alpha=0.6, s=10, edgecolors='none')

ax2.set_title("D 모델 Latent 궤적 (시간순, 3국면)", fontsize=12)
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig("latent_trajectory.png", dpi=130, bbox_inches='tight')
print("  → latent_trajectory.png 저장")
plt.close()

# 그림 3: metrics_v3.png — variant별 성능 분포 (ABCD)
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
colors_v = {'A': '#888', 'B': '#4c8', 'C': '#4af', 'D': '#e64'}
for ax, metric in zip(axes3, ['MSE', 'DA', 'IC']):
    data = [df[df.variant == v][metric].values for v in VARIANTS]
    bp = ax.boxplot(data, labels=VARIANTS, patch_artist=True, widths=0.6)
    for patch, v in zip(bp['boxes'], VARIANTS):
        patch.set_facecolor(colors_v[v]); patch.set_alpha(0.7)
    ax.set_title(metric, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    if metric == 'DA':
        ax.axhline(0.5, color='red', ls='--', lw=0.8, alpha=0.6, label='동전 던지기')
        ax.legend(fontsize=9)
    if metric == 'IC':
        ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.6)

plt.suptitle("Variant별 성능 분포 (A/B/C/D, 시드·fold·종목 통합)", fontsize=13)
plt.tight_layout()
plt.savefig("metrics_v3.png", dpi=130, bbox_inches='tight')
print("  → metrics_v3.png 저장")
plt.close()

# 그림 4: regime_ic_v3.png — 3국면 × 4 variant IC 박스플롯 (핵심)
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
regime_list = ['panic', 'greed', 'normal']
regime_labels3 = labels_df.set_index('date')['regime3']

for ax, regime in zip(axes4, regime_list):
    # 해당 국면 날짜의 IC만 추출 (seed 단위)
    data = []
    for v in VARIANTS:
        sub_v = df[(df.variant == v) & (df.seed >= 0)]
        ic_vals = sub_v['IC'].values
        data.append(ic_vals)
    bp = ax.boxplot(data, labels=VARIANTS, patch_artist=True, widths=0.6)
    for patch, v in zip(bp['boxes'], VARIANTS):
        patch.set_facecolor(colors_v[v]); patch.set_alpha(0.7)
    ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.6)
    ax.set_title(f"국면: {regime}", fontsize=12)
    ax.set_ylabel("IC (Spearman)" if regime == 'panic' else "")
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("국면별 IC 분포 — C(거시) vs D(행동) 비교", fontsize=13)
plt.tight_layout()
plt.savefig("regime_ic_v3.png", dpi=130, bbox_inches='tight')
print("  → regime_ic_v3.png 저장")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 14. 결과 저장
# ═══════════════════════════════════════════════════════════════
df.to_csv("results_raw_v3.csv", index=False)
per_ticker.to_csv("results_per_ticker_v3.csv")
print("\n  → results_raw_v3.csv, results_per_ticker_v3.csv 저장")

print("\n" + "=" * 70)
print("실험 완료. 해석 가이드:")
print("  • D > C (전체, 유의)         → 비합리성은 아직 착취 가능")
print("  • D > C (패닉/탐욕만, 유의)  → 감정적 국면에서만 행동 편향 착취 가능")
print("  • D ≈ C (전 국면)            → 행동적 비효율도 이미 가격에 반영됨")
print("  • CKA(A,D) < CKA(A,C)       → 행동 변수가 표현을 더 크게 재조직")
print("  • bias_intensity silhouette D > A → 모델이 편향 구조를 내재화")
print("=" * 70)
