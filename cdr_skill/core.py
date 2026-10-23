"""Small offline admission/routing demo; only trusted Python functions are called."""

import ast
from dataclasses import dataclass
import inspect
import re
import warnings

import pandas as pd

from .iqr import ContractError, adaptive_robust_outlier_impute, validate_input
from .verification import reference_output, run_boundary_checks


def static_review(source):
    """A capability lint, not isolation or a proof of safety."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Invalid Python: {exc.msg}"]
    issues = []
    banned = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
              "system", "popen", "run", "Popen", "read_csv", "read_pickle", "to_csv",
              "to_pickle", "to_sql", "read_sql", "load", "save", "connect", "request", "urlopen"}
    allowed_imports = {"__future__", "numbers", "warnings", "numpy", "pandas", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([n.name.split(".")[0] for n in node.names] if isinstance(node, ast.Import)
                     else [(node.module or "").split(".")[0]])
            if any(name not in allowed_imports for name in names):
                issues.append(f"Unapproved import at line {node.lineno}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in banned or name.startswith("__"):
                issues.append(f"Disallowed call {name} at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            issues.append(f"Dunder access at line {node.lineno}")
    return sorted(set(issues))


@dataclass
class Binding:
    feature: str
    group: str | None = None
    semantic_confirmed: bool = False
    evidence: str = ""


@dataclass
class Invocation:
    status: str
    reason: str
    skill_id: str | None = None
    output: pd.DataFrame | None = None


class SkillLibrary:
    """An in-memory registry; JSON reports are ordinary files, not a database."""

    def __init__(self):
        self.skills = {}

    def admit(self, *, source_assertions_passed, source_evidence, unresolved_conditions=()):
        record = {"admitted": False, "skill_id": "iqr.real.v1", "source_evidence": source_evidence}
        if not source_assertions_passed or not source_evidence.strip() or unresolved_conditions:
            return {**record, "reason": "Source evidence is insufficient or applicability is unresolved"}
        # Demo 仅登记内置可信实现，绝不动态执行用户提交的源码。
        function = adaptive_robust_outlier_impute
        issues = static_review(inspect.getsource(inspect.getmodule(function)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            checks = run_boundary_checks(function)
        record.update(static_issues=issues, checks=checks)
        if issues or not all(c["passed"] for c in checks):
            return {**record, "reason": "Static review or boundary assertions failed"}
        self.skills[record["skill_id"]] = {
            "function": function,
            "terms": set("iqr outlier capping cap missing impute imputation numeric clean feature group".split()),
        }
        return {**record, "admitted": True, "reason": "Trusted built-in passed the declared demo admission checks"}

    def invoke(self, query, frame, binding, *, top_k=3):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        tokens = set(re.findall(r"[a-z]+", query.lower()))
        ranked = sorted(((len(tokens & skill["terms"]) / len(tokens | skill["terms"]), sid)
                         for sid, skill in self.skills.items()), key=lambda item: (-item[0], item[1]))
        candidates = [sid for score, sid in ranked[:top_k] if score > 0]
        if not candidates:
            return Invocation("fallback", "No relevant registered skill")
        if not binding.semantic_confirmed or not binding.evidence.strip():
            return Invocation("fallback", "Unresolved semantic binding: human evidence is required")
        reasons = []
        for sid in candidates:
            try:
                validate_input(frame, binding.feature, binding.group)
                before = frame.copy(deep=True)
                working = frame.copy(deep=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    out = self.skills[sid]["function"](working, binding.feature, binding.group)
                # 原始输入单独留存，检查数值答案和输入不变性；不只检查表形状。
                pd.testing.assert_frame_equal(working, before)
                expected = reference_output(before, binding.feature, binding.group)
                pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-12, atol=1e-12)
            except (ContractError, AssertionError, TypeError, ValueError, OverflowError) as exc:
                reasons.append(f"{sid}: {exc}")
                continue
            return Invocation("accepted", "Input contract and independent output assertions satisfied", sid, out)
        return Invocation("fallback", "; ".join(reasons))
