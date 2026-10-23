"""Hardened IQR example; intentionally separate from the manuscript snapshot."""

from numbers import Real
import warnings

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """A declared input or numerical policy prevents invocation."""


def validate_input(df, feature_col, group_col=None, iqr_multiplier=1.5,
                   missing_threshold=0.4):
    """Check the demo's narrower real-valued, float64-representable domain."""
    if not isinstance(df, pd.DataFrame):
        raise ContractError("Input must be a pandas DataFrame")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ContractError("Column names must be unique strings")
    if feature_col not in df.columns:
        raise ContractError("The target column does not exist")
    target = df[feature_col]
    if (not pd.api.types.is_numeric_dtype(target.dtype)
            or pd.api.types.is_bool_dtype(target.dtype)
            or pd.api.types.is_complex_dtype(target.dtype)):
        raise ContractError("The target must contain real, non-Boolean numeric values")
    for name, value in (("iqr_multiplier", iqr_multiplier),
                        ("missing_threshold", missing_threshold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ContractError(f"{name} must be a finite real scalar")
    if iqr_multiplier <= 0 or not 0 <= missing_threshold < 1:
        raise ContractError("Require multiplier > 0 and 0 <= missing threshold < 1")
    valid = target.dropna()
    if len(valid) < 10:
        raise ContractError("At least 10 nonmissing target observations are required")
    # 整数超过 float64 连续精确表示范围时拒绝，避免转换后悄悄改变原始值。
    if pd.api.types.is_integer_dtype(target.dtype) and any(abs(int(v)) > 2**53 for v in valid):
        raise ContractError("Integer magnitude exceeds the exact float64 integer range")
    try:
        numeric = target.to_numpy(dtype=np.float64, na_value=np.nan)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError("Target values cannot be represented as float64") from exc
    if not np.isfinite(numeric[~np.isnan(numeric)]).all():
        raise ContractError("Nonmissing target values must be finite in float64")
    if group_col is not None:
        if group_col not in df.columns or group_col == feature_col:
            raise ContractError("The grouping column is absent or equals the target")
        if df[group_col].isna().any():
            raise ContractError("Grouping values must not be missing")
        try:
            df.groupby(group_col, sort=False, observed=True).indices
        except (TypeError, ValueError) as exc:
            raise ContractError("Grouping values must be supported, hashable labels") from exc
    high_missing = target.isna().mean() > missing_threshold
    if high_missing and f"{feature_col}_was_missing" in df.columns:
        raise ContractError("The missingness-indicator column already exists")
    return numeric, bool(high_missing)


def adaptive_robust_outlier_impute(df, feature_col, group_col=None,
                                  iqr_multiplier=1.5, missing_threshold=0.4):
    """Cap per-group outliers and apply the explicitly stated missingness policy.

    Uses linear quantiles, a global high-missingness decision, minimum group
    count 10, and no capping when IQR is zero. Output target dtype is float64.
    Numerical overflow causes rejection; it never produces an accepted result.
    """
    numeric, high_missing = validate_input(
        df, feature_col, group_col, iqr_multiplier, missing_threshold)
    out = df.copy(deep=True)
    out[feature_col] = numeric
    if high_missing:
        out[f"{feature_col}_was_missing"] = df[feature_col].isna().to_numpy()
        warnings.warn("High missingness: missing entries retained and indicator added", UserWarning)
    col_pos = out.columns.get_loc(feature_col)
    groups = ([np.arange(len(out))] if group_col is None else
              list(out.groupby(group_col, sort=False, observed=True).indices.values()))
    # 按行位置写回，重复索引或交错分组也不会发生标签对齐错误。
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            global_median = float(out[feature_col].dropna().median())
            if not np.isfinite(global_median):
                raise ContractError("Global median is not finite")
            for positions in groups:
                series = out.iloc[positions, col_pos].copy()
                observed = series.dropna()
                if len(observed) >= 10:
                    q1, q3 = observed.quantile([0.25, 0.75], interpolation="linear")
                    spread = q3 - q1
                    if spread > 0:
                        lower = q1 - iqr_multiplier * spread
                        upper = q3 + iqr_multiplier * spread
                        if not np.isfinite([lower, upper]).all():
                            raise ContractError("IQR bounds are not finite")
                        series = series.clip(lower=lower, upper=upper)
                if not high_missing:
                    median = series.dropna().median() if series.notna().any() else global_median
                    if not np.isfinite(median):
                        raise ContractError("Imputation median is not finite")
                    series = series.fillna(median)
                out.iloc[positions, col_pos] = series.to_numpy()
    except (FloatingPointError, OverflowError) as exc:
        raise ContractError("Numerical overflow prevents a reliable float64 result") from exc
    return out
