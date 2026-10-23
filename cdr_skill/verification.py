"""Reviewed examples and a scalar numerical oracle, independent of pandas quantile."""

import math

import numpy as np
import pandas as pd


def reference_output(frame, target, group=None, multiplier=1.5, threshold=0.4):
    """Compute expected values for supported inputs using sorted Python lists."""
    result = frame.copy(deep=True)
    values = [float(v) if pd.notna(v) else math.nan for v in frame[target]]
    high = sum(math.isnan(v) for v in values) / len(values) > threshold

    def quantile(sorted_values, fraction):
        # 独立标量插值定义，不调用被测实现的 Pandas 分位数或截断逻辑。
        position = (len(sorted_values) - 1) * fraction
        left, right = math.floor(position), math.ceil(position)
        return sorted_values[left] + (sorted_values[right] - sorted_values[left]) * (position - left)

    def median(observed):
        ordered = sorted(observed)
        n = len(ordered)
        return ordered[n // 2] if n % 2 else ordered[n // 2 - 1] / 2 + ordered[n // 2] / 2

    global_median = median([v for v in values if not math.isnan(v)])
    grouped_positions = {}
    for pos in range(len(frame)):
        label = frame[group].iloc[pos] if group is not None else None
        grouped_positions.setdefault(label, []).append(pos)
    for positions in grouped_positions.values():
        observed = sorted(values[p] for p in positions if not math.isnan(values[p]))
        if len(observed) >= 10:
            q1, q3 = quantile(observed, 0.25), quantile(observed, 0.75)
            spread = q3 - q1
            if spread > 0:
                lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
                for p in positions:
                    if not math.isnan(values[p]):
                        values[p] = min(max(values[p], lower), upper)
        if not high:
            current = [values[p] for p in positions if not math.isnan(values[p])]
            fill = median(current) if current else global_median
            for p in positions:
                if math.isnan(values[p]):
                    values[p] = fill
    result[target] = np.array(values, dtype=float)
    if high:
        result[f"{target}_was_missing"] = frame[target].isna().to_numpy()
    return result


def boundary_cases():
    """The first six fixtures reproduce the manuscript's constructed inputs."""
    return [
        ("low_missing", pd.DataFrame({"x": list(range(1, 11)) + [np.nan, 100],
                                      "tag": list("abcdefghijkl")}, index=range(20, 32)), {}, None),
        ("high_missing", pd.DataFrame({"x": list(range(10)) + [np.nan] * 10}), {}, None),
        ("groupwise", pd.DataFrame({"group": ["A"] * 10 + ["B"] * 10,
                                    "x": list(range(1, 10)) + [100] + list(range(101, 110)) + [200]}),
         {"group_col": "group"}, None),
        ("zero_iqr", pd.DataFrame({"x": [1] * 10 + [100]}), {}, None),
        ("nonnumeric", pd.DataFrame({"x": ["a"] * 12}), {}, (TypeError, ValueError)),
        ("too_few", pd.DataFrame({"x": range(9)}), {}, (TypeError, ValueError)),
    ]


def run_boundary_checks(function):
    """Check exact numerical values, full frame preservation and nonmutation."""
    records = []
    for name, frame, kwargs, expected_error in boundary_cases():
        before = frame.copy(deep=True)
        try:
            try:
                output = function(frame, "x", **kwargs)
            except Exception as exc:
                if expected_error is None or not isinstance(exc, expected_error):
                    raise
            else:
                if expected_error is not None:
                    raise AssertionError("Expected input rejection")
                expected = reference_output(frame, "x", group=kwargs.get("group_col"))
                pd.testing.assert_frame_equal(output, expected, check_exact=False, rtol=1e-12, atol=1e-12)
            pd.testing.assert_frame_equal(frame, before)
        except Exception as exc:
            records.append({"case": name, "passed": False, "detail": str(exc)})
        else:
            records.append({"case": name, "passed": True})
    return records


def zero_replacement(frame, feature_col, group_col=None, **kwargs):
    """Deliberately wrong control used only to demonstrate weak assertions."""
    if not pd.api.types.is_numeric_dtype(frame[feature_col]):
        raise TypeError("Nonnumeric")
    if frame[feature_col].notna().sum() < 10:
        raise ValueError("Too few observations")
    out = frame.copy(deep=True)
    out[feature_col] = out[feature_col].astype(float)
    if out[feature_col].isna().mean() > 0.4:
        out[f"{feature_col}_was_missing"] = out[feature_col].isna()
        return out
    q1, q3 = out[feature_col].dropna().quantile([0.25, 0.75])
    if q3 == q1:
        return out
    out[feature_col] = 0.0
    return out


def run_original_assertions(function):
    """Reproduce precisely the six weak assertion sets, without overwriting history."""
    records = []
    for name, frame, kwargs, expected_error in boundary_cases():
        try:
            if expected_error is not None:
                try:
                    function(frame, "x", **kwargs)
                except (TypeError if name == "nonnumeric" else ValueError):
                    pass
                else:
                    raise AssertionError("Expected manuscript exception type")
            else:
                out = function(frame, "x", **kwargs)
                if name == "low_missing":
                    assert out.index.equals(frame.index) and out["tag"].equals(frame["tag"])
                    assert out["x"].isna().sum() == 0 and out["x"].max() < 100
                elif name == "high_missing":
                    assert out["x"].isna().sum() == 10
                    assert out["x_was_missing"].dtype == bool and out["x_was_missing"].sum() == 10
                elif name == "groupwise":
                    assert out.loc[9, "x"] < 100 and out.loc[19, "x"] < 200
                    assert out["group"].equals(frame["group"])
                else:
                    assert out.loc[10, "x"] == 100
        except Exception as exc:
            records.append({"case": name, "passed": False, "detail": str(exc)})
        else:
            records.append({"case": name, "passed": True})
    return records
