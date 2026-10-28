"""LaTeX tables written from summary.json. Without results, the same tables show [todo] cells."""

from __future__ import annotations

from ..config import Config
from ..data import DATASET_SOURCES
from ..methods import METHODS, calls_per_question
from ..prompts import COUNTER_TEMPLATES, DOUBT_TEMPLATES
from .fmt import TODO, auroc_ci, integer, latex_escape, num, pvalue
from .values import setting_values

# Shrinks a column table only when it would be wider than the column.
BOX_OPEN = r"\begin{adjustbox}{max width=\columnwidth}"
BOX_CLOSE = r"\end{adjustbox}"
DATASET_NAMES = {"truthfulqa": "TruthfulQA", "triviaqa": "TriviaQA", "sciq": "SciQ"}


def build_tables(summary: dict | None, config: Config) -> dict[str, str]:
    models = _models(summary, config)
    return {
        "data": _data(config),
        "prompts": _prompts(config),
        "examples": _examples(summary, config),
        "counts": _counts(models, config),
        "main": _main(models, config),
        "datasets": _datasets(models, config),
        "ablation": _ablation(models, config),
    }


def _models(summary: dict | None, config: Config) -> list[tuple[str, dict | None]]:
    out = []
    for spec in config.models[:2]:
        model = (summary or {}).get("models", {}).get(spec.key)
        out.append((latex_escape(spec.display_name), model))
    return out


def _counts(models, config: Config) -> str:
    header = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in models)
    sub = " & ".join("CW & CR & INC" for _ in models)
    rows = []
    for dataset in list(config.datasets) + ["all"]:
        cells = []
        for _, model in models:
            for group in ("CW", "CR", "INC"):
                if model is None:
                    cells.append(TODO)
                elif dataset == "all":
                    cells.append(integer(model["counts"][group]))
                else:
                    cells.append(integer(model["counts"]["per_dataset"].get(dataset, {}).get(group)))
        label = "All" if dataset == "all" else DATASET_NAMES.get(dataset, dataset)
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows.insert(-1, r"\midrule")
    cols = "l" + "rrr" * len(models)
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{Consistently wrong (CW), consistently right (CR) and inconsistent (INC) answers. Refusals are excluded.}",
        r"\label{tab:counts}", BOX_OPEN, rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        f" & {header} \\\\", f"Dataset & {sub} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", BOX_CLOSE,
        r"\end{table}",
    ])


def _main(models, config: Config) -> str:
    calls = calls_per_question(config)
    header = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in models)
    sub = " & ".join(r"CW vs CR [95\% CI] & All & $p_{\mathrm{Holm}}$" for _ in models)
    rows, previous_role = [], "baseline"
    for method in METHODS:
        if method.role != previous_role:
            rows.append(r"\midrule")
            previous_role = method.role
        cells = []
        for _, model in models:
            entry = (model or {}).get("methods", {}).get(method.key)
            comparison = (model or {}).get("comparisons_vs_ours", {}).get(method.key, {})
            compared = method.role == "baseline"
            if entry is None:
                cells += [TODO, "--" if method.scope == "matched" else TODO, TODO if compared else "--"]
                continue
            all_auc = num(entry["auroc_all"]) if entry["auroc_all"] is not None else "--"
            p_holm = pvalue(comparison.get("p_holm")) if compared else "--"
            cells += [auroc_ci(entry["auroc_matched"]), all_auc, p_holm]
        label = latex_escape(method.label)
        access = "logits" if method.access == "logits" else "text"
        rows.append(f"{label} & {access} & {calls[method.key]} & " + " & ".join(cells) + r" \\")
    cols = "lcr" + "lcc" * len(models)
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{AUROC for detecting consistently wrong answers. ``CW vs CR'' is the matched set, the main "
        r"evaluation; ``All'' includes inconsistent answers and uses any wrong answer as the positive class. "
        r"Calls are generations per question. $p_{\mathrm{Holm}}$ is the Holm adjusted DeLong $p$ value for "
        r"\dfr{} against each baseline. The last two rows are other scores computed from the same pressure "
        r"replies. Methods marked ``--'' under All run only on the matched set.}",
        r"\label{tab:main}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        f" & & & {header} \\\\", f"Method & Access & Calls & {sub} \\\\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])


def _datasets(models, config: Config) -> str:
    header = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in models)
    sub = " & ".join(r"CW/CR & \dfr{} & Best" for _ in models)
    rows = []
    for dataset in config.datasets:
        cells = []
        for _, model in models:
            entry = (model or {}).get("per_dataset", {}).get(dataset)
            if entry is None:
                cells += [TODO, TODO, TODO]
                continue
            cells += [f"{entry['n_cw']}/{entry['n_cr']}",
                      num(entry["ours"]) if entry["ours"] is not None else "",
                      num(entry["baseline"]) if entry["baseline"] is not None else ""]
        rows.append(f"{DATASET_NAMES.get(dataset, dataset)} & " + " & ".join(cells) + r" \\")
    cols = "l" + "rcc" * len(models)
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{AUROC by dataset on the matched set. ``Best'' is the strongest text only baseline overall.}",
        r"\label{tab:datasets}", BOX_OPEN, rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        f" & {header} \\\\", f"Dataset & {sub} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", BOX_CLOSE,
        r"\end{table}",
    ])


def _ablation(models, config: Config) -> str:
    def value(model, path: list[str]):
        node = (model or {}).get("ablations")
        for key in path:
            if not isinstance(node, dict):
                return TODO
            node = node.get(key)
        return TODO if model is None else num(node)

    k = config.n_pressure
    variants = [
        (rf"\dfr{{}}, $k={k}$ (main)", ["signal", "dfr"]),
        (r"\dfr{}, $k=1$", ["pressure_count", "1", "dfr"]),
        (r"\dfr{}, $k=2$", ["pressure_count", "2", "dfr"]),
        (r"Counterclaim flip rate $F_{\mathrm{C}}$", ["signal", "f_counter"]),
        (r"\cps{} $=F_{\mathrm{C}}-F_{\mathrm{D}}$", ["signal", "cps"]),
        (r"Strict rule ($\tau=1$)", ["strict", "dfr"]),
        (r"Rank mean of \dfr{} and $X$", ["combination", "rank_mean"]),
        (r"Rank mean of \cps{} and $X$", ["combination", "cps_rank_mean"]),
        (r"Logistic regression (reference)", ["combination", "logistic_cv"]),
    ]
    header = " & ".join(name for name, _ in models)
    rows = [f"{label} & " + " & ".join(value(model, path) for _, model in models) + r" \\"
            for label, path in variants]
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{Ablations. AUROC on the matched set (CW vs CR). The logistic regression uses out of fold "
        r"predictions and is a reference ceiling, not a method.}",
        r"\label{tab:ablation}", BOX_OPEN, r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
        f"Variant & {header} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", BOX_CLOSE, r"\end{table}",
    ])


LICENSES = {"truthfulqa": "Apache 2.0", "triviaqa": "see card", "sciq": "CC BY-NC 3.0"}


def _data(config: Config) -> str:
    sizes = setting_values(config)
    rows = []
    for name in config.datasets:
        source = DATASET_SOURCES[name]
        split = f"{source['name']} / {source['split']}" if source["name"] else source["split"]
        rows.append(f"{DATASET_NAMES.get(name, name)} & {latex_escape(split)} & {sizes.get(f'n.{name}', TODO)} & "
                    f"{LICENSES.get(name, '')} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{Datasets. Questions are asked without context or answer options.}",
        r"\label{tab:data}", r"\begin{tabular}{llrl}", r"\toprule",
        r"Dataset & Config / split & Questions & License \\", r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


def _prompts(config: Config) -> str:
    rows = []
    for index, text in enumerate(DOUBT_TEMPLATES[:config.n_pressure], start=1):
        rows.append(f"D{index} & {_prompt_text(text)} \\\\")
    for index, text in enumerate(COUNTER_TEMPLATES[:config.n_pressure], start=1):
        rows.append(f"C{index} & {_prompt_text(text)} \\\\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{Second turn challenges. D: doubts, used by \dfr{}; C: counterclaims naming the alternative $b$, "
        r"used by the contrastive variant. The first turn is the question and the model's own answer $a$.}",
        r"\label{tab:prompts}", r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.86\columnwidth}@{}}", r"\toprule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def _prompt_text(template: str) -> str:
    return latex_escape(template).replace(r"\{alt\}", "$b$").replace(r"\{answer\}", "$a$")


def _examples(summary: dict | None, config: Config) -> str:
    spec = config.models[0]
    model = (summary or {}).get("models", {}).get(spec.key)
    examples = (model or {}).get("examples", [])
    header = r"Group & Question & Reference & Answer $a$ & Reply to D1 & Alternative $b$ & $F_{\mathrm{D}}$ & $F_{\mathrm{C}}$ \\"
    if not examples:
        rows = [" & ".join([TODO] * 8) + r" \\"]
    else:
        rows = [" & ".join([e["group"], _cut(e["question"], 70), _cut(e["reference"], 25), _cut(e["answer"], 25),
                            _cut(e["reply"], 25), _cut(e["alternative"], 25), num(e["f_doubt"], 2),
                            num(e["f_counter"], 2)]) + r" \\" for e in examples]
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        rf"\caption{{Examples from {latex_escape(spec.display_name)}, chosen by rule: the two consistently wrong "
        r"answers with the highest \dfr{}, the consistently wrong answer with the lowest \dfr{}, and the "
        r"consistently right answer with the highest \dfr{}. D1 is the first doubt prompt. Text is shortened.}",
        r"\label{tab:examples}",
        r"\begin{tabular}{@{}l*{1}{>{\raggedright\arraybackslash}p{4.4cm}}*{4}{>{\raggedright\arraybackslash}p{1.9cm}}cc@{}}", r"\toprule", header, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])


def _cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    return latex_escape(text)
