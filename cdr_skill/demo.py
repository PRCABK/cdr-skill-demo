"""Run offline reuse and paper-assertion diagnostics without an API key."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd

from .core import Binding, SkillLibrary
from .iqr import adaptive_robust_outlier_impute
from .manuscript_snapshot import adaptive_robust_outlier_impute as original
from .verification import run_boundary_checks, run_original_assertions, zero_replacement


def complex_diagnostic(function):
    """Record warnings and completion/exception; never assume truncation behavior."""
    frame = pd.DataFrame({"x": np.arange(12) + 1j * np.arange(12)})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            output = function(frame, "x")
        except Exception as exc:
            result = {"status": "rejected", "exception": type(exc).__name__, "message": str(exc)}
        else:
            result = {"status": "returned", "output_dtype": str(output["x"].dtype)}
    result["warnings"] = [{"type": w.category.__name__, "message": str(w.message)} for w in captured]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo_report.json"))
    args = parser.parse_args()
    library = SkillLibrary()
    admission = library.admit(source_assertions_passed=True,
                              source_evidence="Reviewed synthetic source cases; role is a continuous measurement")
    if not admission["admitted"]:
        raise RuntimeError(f"Demo admission failed: {admission['reason']}")
    # 新目标使用不同列名与索引；角色绑定由用户明确确认，不假装由 LLM 推断。
    target = pd.DataFrame({"measurement": list(range(1, 11)) + [np.nan, 100],
                           "row_id": list("abcdefghijkl")}, index=range(200, 212))
    resolved = Binding("measurement", semantic_confirmed=True,
                       evidence="Demo task explicitly permits IQR capping on this continuous measurement")
    invocations = {}
    for name, query, data, binding in [
        ("renamed_target", "cap numeric outlier and impute missing feature", target, resolved),
        ("unresolved_binding", "cap outlier", target, Binding("measurement")),
        ("invalid_type", "cap outlier", pd.DataFrame({"measurement": ["a"] * 12}), resolved),
        ("unrelated_request", "generate document bibliography", target, resolved),
    ]:
        outcome = library.invoke(query, data, binding)
        record = {"status": outcome.status, "reason": outcome.reason, "skill_id": outcome.skill_id}
        if outcome.output is not None:
            csv_path = args.output.parent / f"{name}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            outcome.output.to_csv(csv_path, index=True, index_label="source_index")
            record["output_file"] = csv_path.name
            record["target_values"] = outcome.output["measurement"].tolist()
        invocations[name] = record
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        original_checks = run_original_assertions(original)
        mutant_weak = run_original_assertions(zero_replacement)
        mutant_strong = run_boundary_checks(zero_replacement)
    package_dir = Path(__file__).parent
    report = {
        "scope": "offline synthetic demonstration; not an end-to-end LLM or real-data benchmark",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(package_dir.glob("*.py"))},
        "admission": admission,
        "invocations": invocations,
        "manuscript_original_assertions": original_checks,
        "incorrect_control_original_assertions": mutant_weak,
        "incorrect_control_stronger_assertions": mutant_strong,
        "complex_input_diagnostic": {"manuscript_snapshot": complex_diagnostic(original),
                                     "hardened_demo": complex_diagnostic(adaptive_robust_outlier_impute)},
        "retrieval": "Deterministic English-token Jaccard ranking; a demo substitute for embeddings",
        "runtime": "Trusted in-process functions only; AST lint is not a security sandbox",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")
    print(f"Original assertions: {sum(c['passed'] for c in original_checks)}/6")
    print(f"Incorrect control, original assertions: {sum(c['passed'] for c in mutant_weak)}/6")
    print(f"Incorrect control, stronger assertions: {sum(c['passed'] for c in mutant_strong)}/6")
    for name, result in invocations.items():
        print(f"{name}: {result['status']}")


if __name__ == "__main__":
    main()
