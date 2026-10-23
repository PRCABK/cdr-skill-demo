"""Unmodified function snapshot from the manuscript; for reproduction only.

Not the hardened implementation used by the offline registry.
Source: CDR-Skill_English_Submission.md, section 5.1.
"""

from typing import Optional
import warnings
import numpy as np
import pandas as pd


def adaptive_robust_outlier_impute(
    df: pd.DataFrame,
    feature_col: str,
    group_col: Optional[str] = None,
    iqr_multiplier: float = 1.5,
    missing_threshold: float = 0.4,
) -> pd.DataFrame:
    """Cap numeric outliers by group, handle missingness, and preserve rows and index."""
    # Preconditions cover columns, types, parameters, sample size, and finite values.
    if not isinstance(df, pd.DataFrame) or feature_col not in df.columns:
        raise TypeError("Input must be a DataFrame containing the target column")
    if not df.columns.is_unique:
        raise ValueError("Input column names must be unique")
    if (not pd.api.types.is_numeric_dtype(df[feature_col])
            or pd.api.types.is_bool_dtype(df[feature_col])):
        raise TypeError("The target column must be numeric and non-Boolean")
    if group_col is not None:
        if group_col not in df.columns or group_col == feature_col:
            raise ValueError("The group column is missing or equals the target column")
        if df[group_col].isna().any():
            raise ValueError("The group column must not contain missing values")
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise ValueError("The IQR multiplier must be positive and finite")
    if not np.isfinite(missing_threshold) or not 0 <= missing_threshold < 1:
        raise ValueError("The missingness threshold must be in [0, 1)")

    valid = df[feature_col].dropna()
    if len(valid) < 10 or not np.isfinite(valid.to_numpy(dtype=float)).all():
        raise ValueError("The target column needs at least 10 finite, nonmissing values")

    df_out = df.copy(deep=True)
    # Convert to float so capped values can be written back to an integer column.
    df_out[feature_col] = df_out[feature_col].astype(float)
    high_missing = df_out[feature_col].isna().mean() > missing_threshold
    if high_missing:
        # Preserve missing entries and add an indicator rather than forcing imputation.
        flag_col = f"{feature_col}_was_missing"
        if flag_col in df_out.columns:
            raise ValueError("The missingness-indicator column already exists")
        df_out[flag_col] = df_out[feature_col].isna()
        warnings.warn("Missingness exceeds the threshold; entries are retained and an indicator is added", UserWarning)

    column_pos = df_out.columns.get_loc(feature_col)
    global_median = valid.median()
    groups = ([np.arange(len(df_out))] if group_col is None else
              list(df_out.groupby(group_col, sort=False).indices.values()))
    for positions in groups:
        series = df_out.iloc[positions, column_pos]
        observed = series.dropna()
        if len(observed) >= 10:
            q25, q75 = observed.quantile([0.25, 0.75])
            iqr = q75 - q25
            # Do not cap when IQR is zero; retain rare valid values in sparse columns.
            if iqr > 0:
                series = series.clip(
                    lower=q25 - iqr_multiplier * iqr,
                    upper=q75 + iqr_multiplier * iqr,
                )
        if not high_missing:
            # Small groups can be imputed; use the global median for empty groups.
            median = series.median()
            series = series.fillna(global_median if pd.isna(median) else median)
        df_out.iloc[positions, column_pos] = series.to_numpy()
    return df_out
