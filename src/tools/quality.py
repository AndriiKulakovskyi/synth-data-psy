import json
from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


NUM_BINS = 20


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as integers."""
    df_enc = df.copy()
    for col in df_enc.columns:
        if not np.issubdtype(df_enc[col].dtype, np.number):
            df_enc[col] = pd.factorize(df_enc[col])[0]
    return df_enc


def _histogram(series: pd.Series, bins: int = NUM_BINS) -> np.ndarray:
    """Return a normalized histogram for the given series."""
    if np.issubdtype(series.dtype, np.number):
        values = series.dropna().to_numpy()
        if values.size == 0:
            return np.zeros(bins, dtype=float)
        mn, mx = values.min(), values.max()
        if mn == mx:
            hist = np.zeros(bins, dtype=float)
            hist[0] = 1.0
        else:
            hist, _ = np.histogram(values, bins=bins, range=(mn, mx), density=True)
    else:
        values = series.dropna().astype(str)
        if values.empty:
            return np.zeros(bins, dtype=float)
        categories = sorted(values.unique())
        counts = values.value_counts().reindex(categories, fill_value=0)
        hist = counts.values.astype(float)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    size = max(p.size, q.size)
    if p.size != size:
        p = np.pad(p, (0, size - p.size))
    if q.size != size:
        q = np.pad(q, (0, size - q.size))
    return 0.5 * np.abs(p - q).sum()


def univariate_tvd(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, float]:
    tvds: Dict[str, float] = {}
    for col in df1.columns:
        if col not in df2.columns:
            continue
        h1 = _histogram(df1[col])
        h2 = _histogram(df2[col])
        tvds[col] = _tvd(h1, h2)
    return tvds


def multivariate_tvd(df1: pd.DataFrame, df2: pd.DataFrame, columns: Sequence[str], bins: int = NUM_BINS) -> float:
    a = df1[list(columns)].copy()
    b = df2[list(columns)].copy()
    for col in columns:
        if not np.issubdtype(a[col].dtype, np.number):
            cats = pd.concat([a[col], b[col]]).astype(str).unique()
            mapping = {c: i for i, c in enumerate(sorted(cats))}
            a[col] = a[col].astype(str).map(mapping)
            b[col] = b[col].astype(str).map(mapping)
    values = pd.concat([a, b])
    ranges = [(values[c].min(), values[c].max()) for c in columns]
    bins_list = [bins for _ in columns]
    h1, _ = np.histogramdd(a.values, bins=bins_list, range=ranges)
    h2, _ = np.histogramdd(b.values, bins=bins_list, range=ranges)
    h1 = h1 / h1.sum() if h1.sum() > 0 else h1
    h2 = h2 / h2.sum() if h2.sum() > 0 else h2
    return _tvd(h1.ravel(), h2.ravel())


def accuracy_from_tvds(tvds: Sequence[float]) -> float:
    if not tvds:
        return float('nan')
    return 1.0 - float(np.mean(tvds))


def compute_multivariate_accuracies(df1: pd.DataFrame, df2: pd.DataFrame, degree: int, max_combinations: int = 100) -> List[float]:
    cols = list(df1.columns.intersection(df2.columns))
    tvds: List[float] = []
    for combo in list(combinations(cols, degree))[:max_combinations]:
        tvd = multivariate_tvd(df1, df2, combo)
        tvds.append(tvd)
    return [1.0 - t for t in tvds]


def correlation_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    num_cols = df1.select_dtypes(include=np.number).columns.intersection(df2.columns)
    if len(num_cols) == 0:
        return float('nan')
    c1 = df1[num_cols].corr().values
    c2 = df2[num_cols].corr().values
    tri = np.triu_indices_from(c1, k=1)
    v1 = c1[tri].ravel()
    v2 = c2[tri].ravel()
    if v1.std() == 0 or v2.std() == 0:
        return float('nan')
    return float(np.corrcoef(v1, v2)[0, 1])


def pca_similarity(
    df_train: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_holdout: Optional[pd.DataFrame] = None,
) -> float:
    """Cosine similarity between centroids in PCA space."""
    frames = [df_train, df_synth]
    if df_holdout is not None:
        frames.append(df_holdout)

    combined = pd.concat(frames, ignore_index=True)
    encoded = _encode_categorical(combined)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scaled)

    n_train = len(df_train)
    n_synth = len(df_synth)
    train_emb = pcs[:n_train]
    synth_emb = pcs[n_train:n_train + n_synth]
    train_centroid = train_emb.mean(axis=0)
    synth_centroid = synth_emb.mean(axis=0)

    if df_holdout is not None:
        hold_emb = pcs[n_train + n_synth:]
        hold_centroid = hold_emb.mean(axis=0)
        ref = hold_centroid
    else:
        ref = train_centroid

    vec_train = train_centroid - ref
    vec_synth = synth_centroid - ref
    denom = np.linalg.norm(vec_train) * np.linalg.norm(vec_synth)
    if denom == 0:
        return float('nan')
    cos_sim = float(np.dot(vec_train, vec_synth) / denom)
    return cos_sim


def distance_closeness_ratio(
    df_train: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_holdout: pd.DataFrame,
) -> float:
    """Share of synthetic samples closer to train than to holdout data."""
    enc_train = _encode_categorical(df_train)
    enc_synth = _encode_categorical(df_synth)
    enc_hold = _encode_categorical(df_holdout)
    scaler = StandardScaler()
    stacked = np.vstack([enc_train.values, enc_synth.values, enc_hold.values])
    scaled = scaler.fit_transform(stacked)
    n_train = len(df_train)
    n_synth = len(df_synth)
    train_scaled = scaled[:n_train]
    synth_scaled = scaled[n_train:n_train + n_synth]
    hold_scaled = scaled[n_train + n_synth:]
    dist_train = cdist(synth_scaled, train_scaled)
    dist_hold = cdist(synth_scaled, hold_scaled)
    min_train = dist_train.min(axis=1)
    min_hold = dist_hold.min(axis=1)
    return float((min_train < min_hold).mean())


def evaluate_quality(
    original_path: str,
    synthetic_path: str,
    holdout_path: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate synthetic data quality and optionally dump a JSON report."""

    df_orig = pd.read_csv(original_path)
    df_syn = pd.read_csv(synthetic_path)
    df_hol = pd.read_csv(holdout_path) if holdout_path else None
    common_cols = [c for c in df_orig.columns if c in df_syn.columns]
    df_orig = df_orig[common_cols]
    df_syn = df_syn[common_cols]
    if df_hol is not None:
        df_hol = df_hol[common_cols]

    result = {}
    uni_acc = [1.0 - v for v in univariate_tvd(df_orig, df_syn).values()]
    bi_acc = compute_multivariate_accuracies(df_orig, df_syn, degree=2)
    tri_acc = compute_multivariate_accuracies(df_orig, df_syn, degree=3)
    result['univariate_accuracy'] = float(np.mean(uni_acc)) if uni_acc else float('nan')
    result['bivariate_accuracy'] = float(np.mean(bi_acc)) if bi_acc else float('nan')
    result['trivariate_accuracy'] = float(np.mean(tri_acc)) if tri_acc else float('nan')
    accs = [v for v in [result['univariate_accuracy'], result['bivariate_accuracy'], result['trivariate_accuracy']] if not np.isnan(v)]
    result['final_accuracy'] = float(np.mean(accs)) if accs else float('nan')
    result['correlation_similarity'] = correlation_similarity(df_orig, df_syn)
    result['centroid_cosine_similarity'] = pca_similarity(df_orig, df_syn, df_hol)
    if df_hol is not None:
        result['distance_closeness_ratio'] = distance_closeness_ratio(df_orig, df_syn, df_hol)
    if report_path:
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


__all__ = [
    'evaluate_quality',
    'correlation_similarity',
    'univariate_tvd',
    'compute_multivariate_accuracies',
    'pca_similarity',
    'distance_closeness_ratio',
]
