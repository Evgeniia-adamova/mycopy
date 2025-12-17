"""
VK post views prediction (regression) — improved baseline.

This script is intentionally self-contained and does NOT require Spark/S3.
It expects a CSV exported from your Spark step (at minimum):
- col_text
- col_views_count
- col_date (ISO-like string or timestamp)

Improvements vs the original notebook baseline:
- Predicts log1p(views) to handle heavy skew/outliers
- Adds temporal + simple text length features
- Uses stronger loss functions (Huber) and models suitable for sparse text
- Reports multiple metrics on the original scale (MAPE, sMAPE, RMSLE, RMSE, MAE)

Example:
  python task2.py --csv posts/part-00000-....csv --sep ';'
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

__VERSION__ = "0.2.0"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Symmetric MAPE in percent. More stable than MAPE near zero."""
    denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """MAPE in percent with epsilon guard."""
    denom = np.maximum(eps, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared log error on non-negative targets."""
    y_true = np.maximum(0.0, y_true)
    y_pred = np.maximum(0.0, y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def _view_group(views: float) -> str:
    if views <= 200:
        return "low"
    if views <= 500:
        return "medium"
    if views <= 1000:
        return "high"
    return "very_high"


def basic_text_clean(s: str) -> str:
    """Lightweight, dependency-free normalizer (keeps only letters)."""
    if not isinstance(s, str):
        return ""
    s = s.replace("<b>", "").replace("</b>", "")
    s = re.sub(r"ё", "е", s.lower())
    # keep latin/cyrillic letters; collapse everything else to spaces
    s = re.sub(r"[^a-zа-я]+", " ", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def add_time_features(df: pd.DataFrame, date_col: str = "col_date") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek  # Mon=0
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["dayofweek"].isin([5, 6])).astype(int)
    # posting "age" is useful when mixing weeks, but safe if single snapshot too
    out["days_from_first_post"] = (dt - dt.min()).dt.total_seconds() / (3600 * 24)
    return out


def add_basic_text_features(df: pd.DataFrame, text_col: str = "col_text") -> pd.DataFrame:
    out = df.copy()
    txt = out[text_col].fillna("").astype(str)
    out["text_len"] = txt.str.len()
    out["word_count"] = txt.str.split().map(len)
    out["line_count"] = txt.str.count(r"\n") + 1
    out["has_link"] = txt.str.contains(r"https?://|vk\.cc|clck\.ru", regex=True).astype(int)
    out["has_hashtag"] = txt.str.contains(r"#\w+", regex=True).astype(int)
    return out


def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Casts engagement count columns to numeric if present (no target leakage)."""
    out = df.copy()
    for col in ["col_likes_count", "col_comments_count", "col_reposts_count"]:
        if col in out.columns:
            out[col] = out[col].map(_safe_float).astype(float)
    return out


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]


def build_models(
    available_columns: Iterable[str],
    *,
    random_state: int = 2025,
    use_author_ids: bool = True,
    use_engagement: bool = True,
):
    # Local import so the file can be imported even if sklearn isn't installed.
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    available = set(available_columns)
    text_col = "proc"
    num_cols = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "days_from_first_post",
        "text_len",
        "word_count",
        "line_count",
        "has_link",
        "has_hashtag",
    ]

    if use_engagement:
        for c in ["col_likes_count", "col_comments_count", "col_reposts_count"]:
            if c in available:
                num_cols.append(c)

    cat_cols: list[str] = []
    if use_author_ids:
        for c in ["col_owner_id", "col_from_id"]:
            if c in available:
                cat_cols.append(c)

    # Baseline: sparse TF-IDF + numeric scaled -> Ridge (strong baseline for text regression)
    preproc_ridge = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5),
                text_col,
            ),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            *(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        cat_cols,
                    )
                ]
                if cat_cols
                else []
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    ridge = Pipeline(
        steps=[
            ("features", preproc_ridge),
            ("model", Ridge(alpha=2.0, random_state=random_state)),
        ]
    )

    # Robust linear: Huber loss reduces outlier sensitivity (works well with log-target).
    huber = Pipeline(
        steps=[
            ("features", preproc_ridge),
            ("model", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=10_000)),
        ]
    )

    # Dense + non-linear: TF-IDF -> SVD -> HGBR (handles interactions better than linear).
    preproc_hgbr = ColumnTransformer(
        transformers=[
            (
                "text_svd",
                Pipeline(
                    steps=[
                        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)),
                        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
                        ("scale", StandardScaler()),
                    ]
                ),
                text_col,
            ),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            *(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        cat_cols,
                    )
                ]
                if cat_cols
                else []
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense output
    )
    hgbr = Pipeline(
        steps=[
            ("features", preproc_hgbr),
            (
                "model",
                HistGradientBoostingRegressor(
                    # Squared error on log1p target directly optimizes RMSLE-like objective.
                    loss="squared_error",
                    learning_rate=0.08,
                    max_depth=6,
                    max_iter=400,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Alternative: L1 on log-target (sometimes improves MAE/MAPE)
    hgbr_l1 = Pipeline(
        steps=[
            ("features", preproc_hgbr),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.08,
                    max_depth=6,
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "ridge_log1p": ridge,
        "huber_log1p": huber,
        "hgbr_svd_log1p": hgbr,
        "hgbr_svd_log1p_l1": hgbr_l1,
    }


def make_sample_weights(y: np.ndarray, scheme: str) -> np.ndarray | None:
    scheme = scheme.lower().strip()
    if scheme in {"none", "off", "false", "0"}:
        return None
    y = np.asarray(y, dtype=float)
    if scheme in {"inv_sqrt", "inverse_sqrt"}:
        w = 1.0 / np.sqrt(np.maximum(0.0, y) + 1.0)
    elif scheme in {"inv", "inverse"}:
        w = 1.0 / np.maximum(1.0, np.maximum(0.0, y))
    elif scheme in {"inv_log", "inverse_log"}:
        w = 1.0 / np.log1p(np.maximum(0.0, y) + 1.0)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    # Normalize to mean=1 for numerical stability
    w = w * (len(w) / np.sum(w))
    return w


def train_and_evaluate(
    df: pd.DataFrame,
    random_state: int = 2025,
    test_size: float = 0.30,
    sample_weighting: str = "inv_sqrt",
    use_author_ids: bool = True,
    use_engagement: bool = True,
) -> Tuple[pd.DataFrame, Iterable[TrainResult]]:
    from sklearn.model_selection import train_test_split

    # Target
    y = df["col_views_count"].astype(float).values
    y_log = np.log1p(y)

    # Stratify by log-view quantiles (stabilizes splits in skewed data)
    try:
        bins = pd.qcut(y_log, q=10, duplicates="drop")
        strat = bins.astype(str).values
    except Exception:
        strat = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    X_train = train_df
    X_test = test_df
    y_train_log = np.log1p(train_df["col_views_count"].astype(float).values)
    y_test = test_df["col_views_count"].astype(float).values

    sample_weight = make_sample_weights(
        train_df["col_views_count"].astype(float).values,
        scheme=sample_weighting,
    )

    models = build_models(
        available_columns=df.columns,
        random_state=random_state,
        use_author_ids=use_author_ids,
        use_engagement=use_engagement,
    )
    results: list[TrainResult] = []
    preds: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        try:
            model.fit(X_train, y_train_log, **fit_kwargs)
        except TypeError:
            # Estimator doesn't support sample_weight; retry without it.
            model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(0.0, y_pred)

        preds[name] = y_pred
        results.append(
            TrainResult(
                model_name=name,
                metrics={
                    "MAE": float(np.mean(np.abs(y_test - y_pred))),
                    "RMSE": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
                    "RMSLE": _rmsle(y_test, y_pred),
                    "MAPE_%": _mape(y_test, y_pred),
                    "sMAPE_%": _smape(y_test, y_pred),
                },
            )
        )

    # Simple mean ensemble (often improves stability)
    if len(preds) >= 2:
        y_pred_ens = np.mean(np.vstack(list(preds.values())), axis=0)
        results.append(
            TrainResult(
                model_name="mean_ensemble",
                metrics={
                    "MAE": float(np.mean(np.abs(y_test - y_pred_ens))),
                    "RMSE": float(np.sqrt(np.mean((y_test - y_pred_ens) ** 2))),
                    "RMSLE": _rmsle(y_test, y_pred_ens),
                    "MAPE_%": _mape(y_test, y_pred_ens),
                    "sMAPE_%": _smape(y_test, y_pred_ens),
                },
            )
        )

    out_test = test_df.copy()
    for k, v in preds.items():
        out_test[f"pred_{k}"] = v
    if len(preds) >= 2:
        out_test["pred_mean_ensemble"] = np.mean(np.vstack(list(preds.values())), axis=0)

    return out_test, results


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    dfm = pd.DataFrame({"y": y_true, "p": y_pred})
    dfm["group"] = dfm["y"].map(_view_group)
    out: Dict[str, Dict[str, float]] = {}
    for g, sub in dfm.groupby("group"):
        yt = sub["y"].to_numpy(float)
        yp = sub["p"].to_numpy(float)
        out[g] = {
            "n": float(len(sub)),
            "RMSLE": _rmsle(yt, yp),
            "MAPE_%": _mape(yt, yp),
            "sMAPE_%": _smape(yt, yp),
            "MAE": float(np.mean(np.abs(yt - yp))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with posts")
    ap.add_argument("--sep", default=";", help="CSV separator (default: ';')")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")
    ap.add_argument("--random-state", type=int, default=2025)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument(
        "--sample-weighting",
        default="inv",
        help="Sample weighting: none | inv_sqrt | inv | inv_log (default: inv)",
    )
    ap.add_argument(
        "--use-author-ids",
        action="store_true",
        help="If CSV contains col_owner_id/col_from_id, use them as categorical features",
    )
    ap.add_argument(
        "--use-engagement",
        action="store_true",
        help="If CSV contains likes/comments/reposts counts, use them as numeric features",
    )
    args = ap.parse_args()

    print(f"vk-views-regression script version: {__VERSION__}")

    df = pd.read_csv(args.csv, sep=args.sep, encoding=args.encoding)
    needed = {"col_text", "col_views_count", "col_date"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Basic cleaning
    df = df.dropna(subset=["col_views_count"]).copy()
    df["col_views_count"] = df["col_views_count"].map(_safe_float).astype(float)
    df = df[df["col_views_count"].notna()].copy()

    # Text preprocessing (dependency-free; your notebook lemmatization can be plugged in later)
    df["proc"] = df["col_text"].fillna("").astype(str).map(basic_text_clean)

    # Feature engineering
    df = add_time_features(df, date_col="col_date")
    df = add_basic_text_features(df, text_col="col_text")
    df = add_engagement_features(df)

    # Optional categorical IDs if present
    for c in ["col_owner_id", "col_from_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Drop rows with invalid dates (time features become NaN otherwise)
    df = df.dropna(subset=["hour", "dayofweek", "month"]).copy()

    test_scored, results = train_and_evaluate(
        df,
        random_state=args.random_state,
        test_size=args.test_size,
        sample_weighting=args.sample_weighting,
        use_author_ids=args.use_author_ids,
        use_engagement=args.use_engagement,
    )

    # Print results
    print("\nDataset:", df.shape)
    print("Test:", test_scored.shape)
    print("\nMetrics (lower is better):")
    for r in sorted(results, key=lambda x: x.metrics["RMSLE"]):
        m = r.metrics
        print(
            f"- {r.model_name:16s} | "
            f"RMSLE={m['RMSLE']:.4f}  "
            f"RMSE={m['RMSE']:.1f}  "
            f"MAE={m['MAE']:.1f}  "
            f"MAPE={m['MAPE_%']:.1f}%  "
            f"sMAPE={m['sMAPE_%']:.1f}%"
        )

    # Show a few worst errors for quick debugging
    pred_col = "pred_mean_ensemble" if "pred_mean_ensemble" in test_scored.columns else test_scored.filter(like="pred_").columns[0]
    err_pct = np.abs(test_scored[pred_col] - test_scored["col_views_count"]) / np.maximum(1.0, test_scored["col_views_count"]) * 100
    worst = test_scored.assign(error_pct=err_pct).sort_values("error_pct", ascending=False).head(10)
    print("\nWorst 10 by % error (using", pred_col, "):")
    with pd.option_context("display.max_colwidth", 120):
        print(worst[["col_date", "col_views_count", pred_col, "error_pct", "col_text"]].to_string(index=False))

    gm = group_metrics(
        test_scored["col_views_count"].to_numpy(float),
        test_scored[pred_col].to_numpy(float),
    )
    print("\nGroup metrics (using", pred_col, "):")
    for g in ["low", "medium", "high", "very_high"]:
        if g in gm:
            m = gm[g]
            print(
                f"- {g:9s} n={int(m['n']):4d} | "
                f"RMSLE={m['RMSLE']:.4f}  "
                f"MAE={m['MAE']:.1f}  "
                f"MAPE={m['MAPE_%']:.1f}%  "
                f"sMAPE={m['sMAPE_%']:.1f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
