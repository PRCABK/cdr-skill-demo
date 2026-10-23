"""Meaningful numerical, contract, preservation and routing regression tests."""

from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd

from cdr_skill.core import Binding, SkillLibrary, static_review
from cdr_skill.demo import complex_diagnostic
from cdr_skill.iqr import ContractError, adaptive_robust_outlier_impute as process
from cdr_skill.manuscript_snapshot import adaptive_robust_outlier_impute as original
from cdr_skill.verification import (boundary_cases, reference_output, run_boundary_checks,
                                    run_original_assertions, zero_replacement)


class NumericalTests(unittest.TestCase):
    def test_reviewed_cases_against_independent_oracle(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = run_boundary_checks(process)
        self.assertTrue(all(r["passed"] for r in results), results)

    def test_exact_low_missing_values(self):
        frame = boundary_cases()[0][1]
        out = process(frame, "x")
        # 11 个非缺失值：Q1=3.5、Q3=8.5，所以上界为 8.5+1.5*5=16。
        np.testing.assert_allclose(out["x"], list(range(1, 11)) + [6, 16])

    def test_exact_groupwise_bounds(self):
        _, frame, kwargs, _ = boundary_cases()[2]
        out = process(frame, "x", **kwargs)
        self.assertEqual(out["x"].iloc[9], 14.5)
        self.assertEqual(out["x"].iloc[19], 114.5)
        np.testing.assert_array_equal(out["x"].iloc[:9], np.arange(1, 10))

    def test_nonmutation_and_duplicate_index(self):
        frame = boundary_cases()[0][1].copy()
        frame.index = [1, 2] * 6
        before = frame.copy(deep=True)
        out = process(frame, "x")
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_index_equal(frame.index, out.index)

    def test_interleaved_groups(self):
        frame = boundary_cases()[2][1].iloc[[j for i in range(10) for j in (i, i + 10)]].copy()
        pd.testing.assert_frame_equal(process(frame, "x", "group"), reference_output(frame, "x", "group"))

    def test_missing_indicator_exact_rows(self):
        frame = boundary_cases()[1][1].sample(frac=1, random_state=2)
        with self.assertWarns(UserWarning):
            out = process(frame, "x")
        pd.testing.assert_series_equal(out["x_was_missing"], frame["x"].isna(), check_names=False)

    def test_exact_threshold_is_low_missingness(self):
        frame = pd.DataFrame({"x": list(range(12)) + [np.nan] * 8})
        out = process(frame, "x")
        self.assertFalse(out["x"].isna().any())
        self.assertNotIn("x_was_missing", out)

    def test_small_and_empty_group_policy(self):
        frame = pd.DataFrame({"x": list(range(10)) + [1000, np.nan, np.nan],
                              "g": ["A"] * 10 + ["B", "B", "C"]})
        out = process(frame, "x", "g")
        self.assertEqual(out["x"].iloc[10], 1000)
        self.assertEqual(out["x"].iloc[11], 1000)
        self.assertEqual(out["x"].iloc[12], 5)

    def test_nullable_integer(self):
        frame = pd.DataFrame({"x": pd.Series(list(range(10)) + [pd.NA], dtype="Int64")})
        self.assertEqual(process(frame, "x")["x"].iloc[-1], 4.5)

    def test_unused_categorical_groups(self):
        frame = pd.DataFrame({"x": range(10), "g": pd.Categorical(["A"] * 10, categories=["A", "B"])})
        self.assertEqual(len(process(frame, "x", "g")), 10)

    def test_custom_policies(self):
        frame = boundary_cases()[0][1]
        expected = reference_output(frame, "x", multiplier=2.0, threshold=0.1)
        pd.testing.assert_frame_equal(process(frame, "x", iqr_multiplier=2.0, missing_threshold=0.1), expected)

    def test_zero_iqr_preserves_rare_value(self):
        self.assertEqual(process(boundary_cases()[3][1], "x")["x"].iloc[-1], 100)


class ContractTests(unittest.TestCase):
    def test_rejected_dtypes(self):
        for values in ([True] * 12, ["1"] * 12, np.arange(12) + 1j, [np.nan] * 12):
            with self.subTest(values_type=str(type(values))):
                with self.assertRaises(ContractError):
                    process(pd.DataFrame({"x": values}), "x")

    def test_infinity_and_precision_loss_rejected(self):
        for values in (list(range(10)) + [np.inf], [2**53 + 1] * 12):
            with self.assertRaises(ContractError):
                process(pd.DataFrame({"x": values}), "x")

    def test_invalid_parameters(self):
        frame = pd.DataFrame({"x": range(12)})
        for kwargs in ({"iqr_multiplier": 0}, {"iqr_multiplier": True},
                       {"iqr_multiplier": "1"}, {"iqr_multiplier": np.inf},
                       {"missing_threshold": -0.1}, {"missing_threshold": 1},
                       {"missing_threshold": False}, {"missing_threshold": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                process(frame, "x", **kwargs)

    def test_invalid_columns_and_groups(self):
        frames = [pd.DataFrame(np.ones((12, 2)), columns=["x", "x"]),
                  pd.DataFrame({"x": range(12), "g": [None] * 12})]
        for frame in frames:
            with self.assertRaises(ContractError):
                process(frame, "x", "g")
        for group in ("x", "absent"):
            with self.assertRaises(ContractError):
                process(pd.DataFrame({"x": range(12)}), "x", group)

    def test_indicator_collision(self):
        frame = boundary_cases()[1][1].assign(x_was_missing=False)
        with self.assertRaises(ContractError):
            process(frame, "x")

    def test_numerical_overflow_rejected(self):
        frame = pd.DataFrame({"x": [-1e308] * 6 + [1e308] * 6})
        with self.assertRaises(ContractError):
            process(frame, "x")

    def test_original_snapshot_matches_source(self):
        # 仓库内提供固定摘要；原稿不存在时仍可运行所有公开测试。
        snapshot = Path(original.__code__.co_filename).read_text(encoding="utf-8")
        self.assertIn('valid.to_numpy(dtype=float)', snapshot)
        self.assertNotIn('is_complex_dtype', snapshot)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.library = SkillLibrary()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.admission = self.library.admit(source_assertions_passed=True, source_evidence="Reviewed demo source")
        self.frame = boundary_cases()[0][1].rename(columns={"x": "value"})
        self.binding = Binding("value", semantic_confirmed=True, evidence="Reviewed continuous measurement")

    def test_admission_and_renamed_target(self):
        self.assertTrue(self.admission["admitted"])
        result = self.library.invoke("cap outlier impute missing", self.frame, self.binding)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.output["value"].iloc[-1], 16)

    def test_unresolved_or_failed_source_not_admitted(self):
        for passed, evidence, unresolved in ((False, "source", ()), (True, "", ()), (True, "source", ("domain",))):
            result = SkillLibrary().admit(source_assertions_passed=passed, source_evidence=evidence,
                                          unresolved_conditions=unresolved)
            self.assertFalse(result["admitted"])

    def test_routing_fallbacks(self):
        cases = [("outlier", self.frame, Binding("value")),
                 ("outlier", pd.DataFrame({"value": ["a"] * 12}), self.binding),
                 ("bibliography", self.frame, self.binding)]
        for query, frame, binding in cases:
            self.assertEqual(self.library.invoke(query, frame, binding).status, "fallback")

    def test_wrong_output_blocked_by_postconditions(self):
        self.library.skills["iqr.real.v1"]["function"] = zero_replacement
        self.assertEqual(self.library.invoke("outlier", self.frame, self.binding).status, "fallback")

    def test_invalid_retrieval_limit(self):
        with self.assertRaises(ValueError):
            self.library.invoke("outlier", self.frame, self.binding, top_k=0)

    def test_static_capability_lint(self):
        for source in ("import os", "eval('1')", "open('output.txt', 'w')", "x.__class__"):
            self.assertTrue(static_review(source))
        self.assertFalse(static_review("def add(x):\n    return x + 1"))

    def test_assertion_adequacy_counterexample(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(all(r["passed"] for r in run_original_assertions(original)))
            self.assertTrue(all(r["passed"] for r in run_original_assertions(zero_replacement)))
            failed = [r["case"] for r in run_boundary_checks(zero_replacement) if not r["passed"]]
            self.assertEqual(failed, ["low_missing", "groupwise"])

    def test_complex_behavior_is_observed(self):
        hardened = complex_diagnostic(process)
        self.assertEqual(hardened["exception"], "ContractError")
        self.assertEqual(hardened["warnings"], [])
        legacy = complex_diagnostic(original)
        self.assertIn(legacy["status"], ("returned", "rejected"))


if __name__ == "__main__":
    unittest.main()
