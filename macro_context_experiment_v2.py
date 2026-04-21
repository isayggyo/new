"""
맥락 개입 실험 v2
=================
가설: 외생 거시 변수는 개별 종목 수익률 예측 모델의 성능뿐 아니라
      내부 표현 구조(latent space)를 재조직하는가?

v1 대비 개선:
  1. early_stopping=False → 시계열 누수 차단
  2. Walk-forward validation (N_FOLDS 구간) × 다중 시드 (N_SEEDS)
  3. 3단계 모델 비교
     - A: 가격·수익률 특징만
     - B: A + 지수/섹터 (베타 baseline — "동조화 암기" 수준)
     - C: A + 진짜 외생 맥락 (VIX, 금리 스프레드, 크레딧, DXY, 유가)
     C가 B를 이겨야 "거시 맥락" 주장이 성립
  4. 지표: MSE + Directional Accuracy + Information Coefficient
     + constant-zero baseline (모델이 단순히 0 예측으로 수렴했는지 체크)
  5. Latent 분석: 테스트셋 전체 사용, 시장 국면(VIX 분위)/수익 방향별 색칠
     정량화는 silhouette score + linear CKA
  6. 시드 간 paired t-test로 모델 우열의 통계적 유의성

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
START     = "2015-01-01"      # 충분한 학습 이력 확보
END       = "2024-12-31"
LOOKBACK  = 10
N_FOLDS   = 4
N_SEEDS   = 5
MIN_TRAIN = 0.45              # 첫 fold의 최소 train 비율
HIDDEN    = (64, 32)
MAX_ITER  = 200

# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ═══════════════════════════════════════════════════════════════
print("[1/6] 데이터 수집 중...")

def dl(tks, start=START, end=END):
    """다운로드 유틸. 단일 티커면 Series, 여러 개면 DataFrame."""
    d = yf.download(tks, start=start, end=end, auto_adjust=True,
                    progress=False)["Close"]
    if isinstance(d, pd.Series):
        return d.to_frame(name=tks if isinstance(tks, str) else tks[0])
    return d

prices  = dl(TICKERS).ffill().dropna()
returns = prices.pct_change().dropna()

# 베타 baseline용 변수 (B): 시장/섹터 지수 자체의 수익률
beta_src = dl(["SPY", "XLK", "XLE", "XLF"]).ffill()
beta_ret = beta_src.pct_change()

# 진짜 외생 맥락 변수 (C): 서로 다른 자산군의 신호
#   VIX: 변동성 레짐
#   ^TNX (10Y) - ^IRX (3M): 금리 기울기
#   HYG / LQD: 크레딧 스프레드 프록시 (HY가 IG 대비 약하면 위험회피)
#   UUP: 달러 강세
#   USO: 유가
macro_src = dl(["^VIX", "^TNX", "^IRX", "HYG", "LQD", "UUP", "USO"]).ffill()

macro = pd.DataFrame(index=macro_src.index)
macro['vix_level']    = macro_src['^VIX']
macro['vix_chg_5d']   = macro_src['^VIX'].pct_change(5)
macro['yield_slope']  = macro_src['^TNX'] - macro_src['^IRX']
macro['yield_chg']    = macro['yield_slope'].diff(5)
# 크레딧 스프레드 프록시: LQD/HYG 상대 강도 (값 ↑ → 위험회피)
macro['credit_proxy'] = np.log(macro_src['LQD'] / macro_src['HYG'])
macro['credit_chg']   = macro['credit_proxy'].diff(5)
macro['dxy_ret_20d']  = macro_src['UUP'].pct_change(20)
macro['oil_ret_20d']  = macro_src['USO'].pct_change(20)
macro = macro.dropna()

# 베타 baseline 특징 구성
beta_feat = pd.DataFrame(index=beta_ret.index)
for col in beta_ret.columns:
    beta_feat[f'{col}_r20'] = beta_ret[col].rolling(20).sum()
    beta_feat[f'{col}_v20'] = beta_ret[col].rolling(20).std()
beta_feat = beta_feat.dropna()

# ═══════════════════════════════════════════════════════════════
# 3. 특징 엔지니어링
# ═══════════════════════════════════════════════════════════════
def price_features(r: pd.Series) -> pd.DataFrame:
    """개별 종목의 과거 수익률 기반 특징 (모든 모델 공통)."""
    f = pd.DataFrame(index=r.index)
    f['ret_1d']  = r
    f['ret_5d']  = r.rolling(5).sum()
    f['ret_20d'] = r.rolling(20).sum()
    f['vol_20d'] = r.rolling(20).std()
    f['mom']     = r.rolling(20).mean() / (r.rolling(60).std() + 1e-8)
    return f.dropna()


def build_dataset(ticker: str, variant: str):
    """
    variant ∈ {'A', 'B', 'C'}
      A: 가격 특징만
      B: A + 지수/섹터 (베타 baseline)
      C: A + 외생 거시 (진짜 맥락)
    반환: X (N × D), y (N,), 날짜 인덱스 (N,)
    """
    pf = price_features(returns[ticker])
    r  = returns[ticker]

    if variant == 'A':
        extra = None
    elif variant == 'B':
        extra = beta_feat
    elif variant == 'C':
        extra = macro
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
# 4. 평가 지표
# ═══════════════════════════════════════════════════════════════
def directional_accuracy(y_true, y_pred):
    """방향성 일치 비율. y=0인 날은 제외."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true, y_pred):
    """Spearman rank correlation. 순위 기반이라 스케일에 강건함."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def evaluate(y_true, y_pred):
    return {
        'MSE':  mean_squared_error(y_true, y_pred),
        'DA':   directional_accuracy(y_true, y_pred),
        'IC':   information_coefficient(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════
# 5. Walk-forward split
# ═══════════════════════════════════════════════════════════════
def walk_forward_splits(n, n_folds=N_FOLDS, min_train=MIN_TRAIN):
    """
    expanding-window 방식.
    fold k에서 train = [0, t_k), test = [t_k, t_k + test_size).
    """
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
# 6. MLP latent 추출 & CKA
# ═══════════════════════════════════════════════════════════════
def relu(x): return np.maximum(0, x)


def get_latent(model, X):
    """마지막 히든레이어 출력까지 forward pass."""
    h = X
    for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = relu(h @ W + b)
    return h


def linear_cka(X, Y):
    """Centered Kernel Alignment (linear). 0~1, 높을수록 두 표현이 유사."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    nx   = np.linalg.norm(X.T @ X, 'fro')
    ny   = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic / (nx * ny + 1e-12)


# ═══════════════════════════════════════════════════════════════
# 7. 메인 실험 루프
# ═══════════════════════════════════════════════════════════════
print("[2/6] 모델 학습 및 평가 (이 단계가 가장 오래 걸림)...")

VARIANTS = ['A', 'B', 'C']
records  = []

# latent 분석용: variant C 모델의 마지막 fold에서만 저장
latent_store = {'A': [], 'B': [], 'C': []}
label_store  = {'ticker': [], 'date': [], 'y_true': [], 'y_pred_A': [],
                'y_pred_B': [], 'y_pred_C': []}

for ticker in TICKERS:
    datasets = {v: build_dataset(ticker, v) for v in VARIANTS}
    # 세 variant의 공통 인덱스로 정렬 (길이 다를 수 있음)
    common = datasets['A'][2]
    for v in ['B', 'C']:
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

                # 핵심 수정: early_stopping=False (시계열 누수 차단)
                model = MLPRegressor(
                    hidden_layer_sizes=HIDDEN,
                    max_iter=MAX_ITER,
                    random_state=seed,
                    early_stopping=False,
                    learning_rate_init=1e-3,
                    alpha=1e-4,           # 약한 L2로 과적합 완화
                )
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)

                m = evaluate(y_te, y_pred)
                m.update({'ticker': ticker, 'fold': fold_i,
                          'seed': seed, 'variant': v})
                records.append(m)

                # 마지막 fold & seed=0에서만 latent 저장 (시각화용)
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
                    else:
                        label_store['y_pred_C'].extend(y_pred)

    # constant-zero baseline (ticker 단위로 한 번만)
    for fold_i, (a, b, c, e) in enumerate(splits):
        _, y, _ = aligned['A']
        y_te = y[c:e]
        zero_pred = np.zeros_like(y_te)
        m = evaluate(y_te, zero_pred)
        m.update({'ticker': ticker, 'fold': fold_i,
                  'seed': -1, 'variant': 'ZERO'})
        records.append(m)

df = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════
# 8. 결과 요약
# ═══════════════════════════════════════════════════════════════
print("\n[3/6] 결과 요약")
print("=" * 70)

# 8-1. variant별 전체 평균 (± seed·fold·ticker 분산)
summary = df.groupby('variant')[['MSE', 'DA', 'IC']].agg(['mean', 'std'])
print("\n▸ Variant별 성능 (전 시드·fold·종목 통합)")
print(summary.round(5).to_string())

# 8-2. ticker × variant 테이블 (seed·fold 평균)
per_ticker = (df[df.variant != 'ZERO']
              .groupby(['ticker', 'variant'])[['MSE', 'DA', 'IC']]
              .mean()
              .round(5))
print("\n▸ 종목 × Variant 평균")
print(per_ticker.to_string())

# 8-3. 통계 검정: 시드 쌍 기준 paired t-test (A vs B, B vs C, A vs C)
print("\n▸ Paired t-test (시드·fold·종목 단위, IC 기준)")
piv = (df[df.variant != 'ZERO']
       .pivot_table(index=['ticker', 'fold', 'seed'],
                    columns='variant', values='IC'))
for a, b in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
    paired = piv[[a, b]].dropna()
    t, p = ttest_rel(paired[b], paired[a])
    diff = (paired[b] - paired[a]).mean()
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {b} - {a}:  ΔIC = {diff:+.4f},  t = {t:+.3f},  p = {p:.4f}  [{sig}]")

# 8-4. constant-zero baseline 대비 얼마나 나은가
print("\n▸ constant-zero baseline 대비")
zero_mse = df[df.variant == 'ZERO']['MSE'].mean()
for v in VARIANTS:
    m = df[df.variant == v]['MSE'].mean()
    print(f"  {v}: MSE = {m:.6f}  (zero = {zero_mse:.6f},  "
          f"개선 {100*(zero_mse-m)/zero_mse:+.2f}%)")

# ═══════════════════════════════════════════════════════════════
# 9. Latent space 분석
# ═══════════════════════════════════════════════════════════════
print("\n[4/6] Latent space 분석")

L = {v: np.vstack(latent_store[v]) for v in VARIANTS}
labels_df = pd.DataFrame(label_store)

# 9-1. 시장 국면 라벨 생성 (VIX 분위 기준)
labels_df['date'] = pd.to_datetime(labels_df['date'])
vix = macro['vix_level'].reindex(labels_df['date']).values
vix_median = np.nanmedian(vix)
labels_df['regime'] = np.where(vix > vix_median, 'high_vol', 'low_vol')
labels_df['direction'] = np.where(labels_df['y_true'] > 0, 'up', 'down')

# 9-2. silhouette score (국면 분리도)
print("\n▸ Silhouette score (클러스터 분리도, 클수록 좋음)")
print(f"  {'라벨':<20}{'A (가격)':>12}{'B (베타)':>12}{'C (맥락)':>12}")
for label_col in ['ticker', 'regime', 'direction']:
    row = [f"  {label_col:<20}"]
    for v in VARIANTS:
        # 샘플이 많으면 silhouette 느려짐 → 최대 2000개로 subsample
        n = len(L[v])
        idx = np.random.RandomState(0).choice(n, min(n, 2000), replace=False)
        try:
            s = silhouette_score(L[v][idx], labels_df[label_col].values[idx])
        except Exception:
            s = np.nan
        row.append(f"{s:>12.4f}")
    print("".join(row))

# 9-3. Linear CKA (표현 유사도)
print("\n▸ Linear CKA (0~1, 두 표현이 얼마나 비슷한가)")
print(f"  CKA(A, B) = {linear_cka(L['A'], L['B']):.4f}")
print(f"  CKA(A, C) = {linear_cka(L['A'], L['C']):.4f}")
print(f"  CKA(B, C) = {linear_cka(L['B'], L['C']):.4f}")
print("  → A vs C가 낮을수록 '맥락이 표현을 재조직했다'는 증거")

# ═══════════════════════════════════════════════════════════════
# 10. 시각화
# ═══════════════════════════════════════════════════════════════
print("\n[5/6] 시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

color_maps = {
    'ticker':    {t: plt.cm.tab10(i) for i, t in enumerate(TICKERS)},
    'regime':    {'high_vol': '#d62728', 'low_vol': '#1f77b4'},
    'direction': {'up': '#2ca02c', 'down': '#ff7f0e'},
}

for col_i, v in enumerate(VARIANTS):
    Lv = L[v]
    pca = PCA(n_components=2)
    Lv2 = pca.fit_transform(Lv)

    # 위: 국면별 색칠 (regime)
    ax = axes[0, col_i]
    for reg in ['low_vol', 'high_vol']:
        mask = (labels_df['regime'] == reg).values
        ax.scatter(Lv2[mask, 0], Lv2[mask, 1],
                   c=color_maps['regime'][reg], label=reg,
                   alpha=0.5, s=8, edgecolors='none')
    title_map = {'A': 'A: 가격만', 'B': 'B: +지수/섹터 (베타)',
                 'C': 'C: +외생 거시 (맥락)'}
    ax.set_title(f"{title_map[v]}  |  국면별 분리", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc='best', fontsize=9)

    # 아래: 수익 방향별 색칠 (direction)
    ax = axes[1, col_i]
    for dirn in ['down', 'up']:
        mask = (labels_df['direction'] == dirn).values
        ax.scatter(Lv2[mask, 0], Lv2[mask, 1],
                   c=color_maps['direction'][dirn], label=dirn,
                   alpha=0.5, s=8, edgecolors='none')
    ax.set_title(f"{title_map[v]}  |  익일 방향별 분리", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc='best', fontsize=9)

plt.suptitle("맥락 개입에 따른 Latent Space 재조직 (PCA 2D projection)",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig("latent_v2.png", dpi=140, bbox_inches='tight')
print("  → latent_v2.png 저장")

# 지표 요약 바 차트
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = ['MSE', 'DA', 'IC']
colors_v = {'A': '#888', 'B': '#4c8', 'C': '#e64'}
for ax, metric in zip(axes2, metrics):
    data = []
    for v in VARIANTS:
        vals = df[df.variant == v][metric].values
        data.append(vals)
    bp = ax.boxplot(data, labels=VARIANTS, patch_artist=True, widths=0.6)
    for patch, v in zip(bp['boxes'], VARIANTS):
        patch.set_facecolor(colors_v[v]); patch.set_alpha(0.7)
    ax.set_title(metric, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    if metric == 'DA':
        ax.axhline(0.5, color='red', ls='--', lw=0.8, alpha=0.6,
                   label='동전 던지기')
        ax.legend(fontsize=9)
    if metric == 'IC':
        ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.6)

plt.suptitle("Variant별 성능 분포 (시드·fold·종목 통합)", fontsize=13)
plt.tight_layout()
plt.savefig("metrics_v2.png", dpi=140, bbox_inches='tight')
print("  → metrics_v2.png 저장")

# ═══════════════════════════════════════════════════════════════
# 11. 결과 저장
# ═══════════════════════════════════════════════════════════════
print("\n[6/6] 결과 파일 저장")
df.to_csv("results_raw.csv", index=False)
per_ticker.to_csv("results_per_ticker.csv")
print("  → results_raw.csv, results_per_ticker.csv 저장")

print("\n" + "=" * 70)
print("실험 완료. 해석 가이드:")
print("  • C가 B보다 유의하게 나음     → 거시 맥락이 베타 이상의 신호")
print("  • B가 A보다 나음 / C≈B         → '맥락 효과'는 대부분 베타 암기")
print("  • 모든 모델 ≈ zero baseline    → 일간 수익률은 예측 불가 (정직한 결론)")
print("  • CKA(A,C)가 낮고 regime       → 맥락이 표현을 구조적으로 재조직")
print("    silhouette(C) > (A)")
print("=" * 70)
