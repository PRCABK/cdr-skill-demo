# CDR-Skill: an offline manuscript companion demo

**CDR means Cross-Dataset Reusable.** This repository accompanies the manuscript
*Contract Guided Generation and Verification of Reusable Data Analysis Skills*.
It provides executable examples of IQR capping, explicit data contracts,
skill admission, target-input screening, and output verification.

This is an **offline, synthetic-data demonstration**. No model API, credentials,
database, or external dataset is required. The original manuscript's function
and assertions are preserved separately from the newly hardened demo.

## Run

Tested locally with Python **3.12.14**, NumPy **2.3.5**, and pandas **3.0.1**.
From this repository's root directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m cdr_skill.demo --output artifacts/demo_report.json
```

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python`.
If the dependencies are already installed, use `python` directly. There is no
need to activate the virtual environment. Optionally install the package with
`python -m pip install .`.

The demo writes a machine-readable JSON report and `renamed_target.csv` into
`artifacts/`. The report includes dependency versions, source hashes, admission
checks, invocation decisions, assertion diagnostics, and observed complex-input
warnings/exceptions. Running it does not modify any historical paper records.

A checked-in local example is available in [`examples/observed_run.json`](examples/observed_run.json).
The 27 regression tests passed in the recorded environment. The original
assertions pass 6/6 for both the original function and the wrong control; the
stronger checks pass 4/6 for the control, rejecting its two wrong numerical
outputs. These counts apply only to these explicitly constructed cases.

## What is implemented

1. **Candidate admission:** a reviewed source-evidence record, module-level AST
   capability lint, and six numerical/structural boundary checks govern the
   registration of one trusted built-in IQR skill.
2. **Reusable interface:** a target feature and optional group can be rebound to
   new column names. A human must explicitly confirm semantic suitability and
   provide a reason. Dtype alone never establishes analytical meaning.
3. **Routing:** deterministic English-token Jaccard ranking proposes a candidate;
   input checks decide applicability; an independent scalar oracle checks the
   output. Failed or unresolved checks return an explicit `fallback` decision.
4. **Reproduction:** the six original assertion sets are run against the original
   function snapshot and against an intentionally wrong zero-output control.
   Stronger assertions then show which incorrect outputs the weak checks missed.

```python
import pandas as pd
from cdr_skill.core import Binding, SkillLibrary

library = SkillLibrary()
admission = library.admit(
    source_assertions_passed=True,
    source_evidence="Reviewed source cases; task permits this cleaning policy",
)
assert admission["admitted"]
frame = pd.DataFrame({"measurement": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 100]})
binding = Binding(
    feature="measurement",
    semantic_confirmed=True,
    evidence="Task defines this column as continuous and permits IQR capping",
)
result = library.invoke("cap outlier and impute missing feature", frame, binding)
assert result.status == "accepted", result.reason
assert result.output["measurement"].iloc[-1] == 16.0
```

## Numerical contract of the hardened demo

- Input: a DataFrame with unique string column names; a real numeric,
  non-Boolean target; at least 10 nonmissing finite observations globally.
- Values must be representable as finite float64 numbers. Integer magnitudes
  above `2**53` are rejected rather than silently rounded. Complex values are
  rejected before any floating-point cast. Output target dtype is float64.
- If grouping is supplied, its column must differ from the target and contain
  supported, nonmissing labels. Row positions and the original index, including
  duplicate index labels, are preserved. Original nontarget columns are preserved.
- A group with at least 10 observed values uses linear quartiles and bounds
  `[Q1 - m*IQR, Q3 + m*IQR]`, with default `m=1.5`. Smaller groups and zero-IQR
  groups are not capped. Missing observations do not enter the quartiles.
- Missingness is decided **globally**, before grouping. If its fraction is
  strictly greater than the default `0.4`, missing values remain and an exact
  rowwise `<target>_was_missing` Boolean column is added. A name collision rejects
  the input. Exactly `0.4` follows the low-missingness branch.
- At low missingness, fill from the group median **after capping**; an entirely
  missing group uses the original global median. The input table is not mutated.
- Nonfinite bounds or numerical overflow reject the invocation. Domain-specific
  permission to cap a measurement must come from the task and human review.

## Relationship to the manuscript

| File | Purpose | Evidence boundary |
| --- | --- | --- |
| `cdr_skill/manuscript_snapshot.py` | Unmodified original function | Retains the original type-checking limitation |
| `cdr_skill/verification.py` | Original six assertions, stronger oracle, wrong control | Synthetic function-level analysis |
| `cdr_skill/iqr.py` | Hardened real-valued implementation | New companion implementation; not the historical function |
| `cdr_skill/core.py` | Admission, explicit binding, routing, screening | Offline workflow demonstration |
| `cdr_skill/demo.py` | Reproducible report and example output | Observed local outcomes only |
| `tests/test_demo.py` | Numerical and workflow regressions | Checks declared examples and policies |

The original function and the incorrect control both pass the six original
assertion sets. The stronger oracle catches the incorrect low-missingness and
groupwise outputs. These results illustrate assertion adequacy; they do not
establish general correctness or performance on real datasets.

The complex-input diagnostic records the actual behavior of the installed
versions. Do not infer successful silent truncation merely from the numeric
dtype check: downstream operations may raise after conversion warnings.

### Deliberate scope limits

- Candidate extraction, code generation, and semantic role discovery are
  manually supplied in this demo. No automatic LLM generation is implemented.
- Token-overlap retrieval is a transparent substitute for the paper's proposed
  embedding retrieval; it is not an embedding-based experiment.
- The registry stores ordinary in-memory Python objects. No database is used.
- The AST check is a capability lint. Transitive dependencies, aliasing and
  general Python behavior are not a security proof. Only the shipped trusted
  function is admitted; no submitted source is dynamically executed. There is
  no claim of operating-system isolation or safe execution of hostile code.
- `fallback` records a refusal and returns no accepted output. It does not
  secretly run another model or analytical procedure.
- No model-token savings, comparative performance, or end-to-end agent benefit
  is measured. Repository results must not be described as historical manuscript
  experiments or as a completed held-out-dataset benchmark.

## 中文说明

本仓库是论文配套的可运行离线 demo，CDR 指“跨数据集可复用”。先运行上面的
安装、测试和 demo 命令即可生成结果，不需要接口或密钥。

`manuscript_snapshot.py` 保留原稿函数；`iqr.py` 是新增修正版，明确拒绝复数并加强
数值检查。原稿六组弱断言与新增精确数值断言分开报告，不能把新增结果冒充原稿
已经完成的实验。目标列的业务含义由使用者确认；仓库不自动推断业务语义。

论文文字、导师批注、个人信息和本机路径不属于这个公开代码包。仓库未附研究
数据集，demo 中的输入均在代码中明确构造。公开仓库不自动授予软件使用许可；
作者确定许可方式后可另行添加 LICENSE。
