"""Turns a per-model score table into the numbers the paper reports."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import Config
from .methods import BASELINES, METHODS, PRIMARY, SAMPLING_BASELINES, calls_per_question
from .metrics import (aurac, auroc, bootstrap_auroc, bootstrap_auroc_difference, delong_test, holm,
                      impute)
from .scoring.combine import logistic_out_of_fold, rank_mean

GROUPS = ("CW", "CR", "INC", "EXCL")
MIN_PER_CLASS = 5


def matched_frame(table: pd.DataFrame) -> pd.DataFrame:
    frame = table[table["matched"] & table["group"].isin(["CW", "CR"])].copy()
    frame["label"] = (frame["group"] == "CW").astype(int)
    if frame["cross_family"].notna().any():
        cross, _ = impute(frame["cross_family"].to_numpy())
        frame["dfr_cross"] = rank_mean(impute(frame["dfr"].to_numpy())[0], cross)
        frame["cps_cross"] = rank_mean(impute(frame["cps"].to_numpy())[0], cross)
    else:
        frame["dfr_cross"] = np.nan
        frame["cps_cross"] = np.nan
    return frame


def counts(table: pd.DataFrame) -> dict:
    out = {"total": int(len(table))}
    out.update({g: int((table["group"] == g).sum()) for g in GROUPS})
    out["strict_CW"] = int((table["strict_group"] == "CW").sum())
    out["strict_CR"] = int((table["strict_group"] == "CR").sum())
    out["per_dataset"] = {
        dataset: {g: int((part["group"] == g).sum()) for g in GROUPS} | {"total": int(len(part))}
        for dataset, part in table.groupby("dataset")
    }
    return out


def method_results(config: Config, table: pd.DataFrame, frame: pd.DataFrame) -> dict:
    calls = calls_per_question(config)
    consistent = table[table["group"].isin(["CW", "CR", "INC"])]
    out = {}
    for method in METHODS:
        if frame[method.key].isna().all():
            continue
        scores, missing = impute(frame[method.key].to_numpy())
        interval = bootstrap_auroc(frame["label"].to_numpy(), scores, config.bootstrap_samples, config.seed)
        entry = {"label": method.label, "access": method.access, "is_ours": method.is_ours, "role": method.role,
                 "calls": calls[method.key], "auroc_matched": asdict(interval), "n_missing": missing,
                 "auroc_all": None, "aurac_all": None}
        if method.scope == "all" and consistent[method.key].notna().any():
            all_scores, _ = impute(consistent[method.key].to_numpy())
            wrong = (~consistent["correct"].astype(bool)).astype(int).to_numpy()
            entry["auroc_all"] = auroc(wrong, all_scores)
            entry["aurac_all"] = aurac(consistent["correct"].astype(bool).to_numpy(), all_scores)
        out[method.key] = entry
    return out


def best_baseline(methods: dict, access: str | None = "text") -> str | None:
    candidates = [m.key for m in BASELINES if m.key in methods and (access is None or m.access == access)]
    if not candidates:
        return None
    return max(candidates, key=lambda k: methods[k]["auroc_matched"]["value"])


def comparisons(config: Config, frame: pd.DataFrame, methods: dict, ours: str = PRIMARY) -> dict:
    """Paired DeLong test of the main score against every baseline, Holm corrected over the baselines."""
    labels = frame["label"].to_numpy()
    ours_scores, _ = impute(frame[ours].to_numpy())
    rows, p_values = {}, []
    for method in BASELINES:
        if method.key not in methods:
            continue
        other, _ = impute(frame[method.key].to_numpy())
        _, _, z, p = delong_test(labels, ours_scores, other)
        diff = bootstrap_auroc_difference(labels, ours_scores, other, config.bootstrap_samples, config.seed)
        rows[method.key] = {"z": z, "p": p, "difference": asdict(diff)}
        p_values.append(p)
    for key, adjusted in zip(rows, holm(p_values)):
        rows[key]["p_holm"] = adjusted
    return rows


def flip_rates(config: Config, frame: pd.DataFrame) -> dict:
    out = {}
    rng = np.random.default_rng(config.seed)
    for group in ("CW", "CR"):
        part = frame[frame["group"] == group]
        out[group] = {}
        for column in ("f_counter", "f_doubt", "cps"):
            values = part[column].dropna().to_numpy()
            if len(values) == 0:
                continue
            boots = [rng.choice(values, len(values)).mean() for _ in range(config.bootstrap_samples)]
            low, high = np.percentile(boots, [2.5, 97.5])
            out[group][column] = {"value": float(values.mean()), "low": float(low), "high": float(high),
                                  "n": int(len(values))}
    return out


def per_dataset(frame: pd.DataFrame, methods: dict, baseline_key: str | None) -> dict:
    out = {}
    for dataset, part in frame.groupby("dataset"):
        n_cw, n_cr = int(part["label"].sum()), int((1 - part["label"]).sum())
        entry = {"n_cw": n_cw, "n_cr": n_cr, "ours": None, "baseline": None}
        if n_cw >= MIN_PER_CLASS and n_cr >= MIN_PER_CLASS:
            entry["ours"] = auroc(part["label"].to_numpy(), impute(part[PRIMARY].to_numpy())[0])
            if baseline_key:
                entry["baseline"] = auroc(part["label"].to_numpy(), impute(part[baseline_key].to_numpy())[0])
        out[dataset] = entry
    return out


def ablations(config: Config, table: pd.DataFrame, frame: pd.DataFrame, baseline_key: str | None) -> dict:
    labels = frame["label"].to_numpy()

    def score(column: str, part: pd.DataFrame | None = None) -> float | None:
        part = frame if part is None else part
        if column not in part or part[column].isna().all() or part["label"].nunique() < 2:
            return None
        return auroc(part["label"].to_numpy(), impute(part[column].to_numpy())[0])

    def difference(first: str, second: str) -> dict | None:
        if score(first) is None or score(second) is None:
            return None
        return asdict(bootstrap_auroc_difference(labels, impute(frame[first].to_numpy())[0],
                                                 impute(frame[second].to_numpy())[0],
                                                 config.bootstrap_samples, config.seed))

    out: dict = {"pressure_count": {}, "signal": {}, "strict": {}, "random_alternative": {}, "combination": {}}
    for k, dfr_col, cps_col in ((1, "dfr_k1", "cps_k1"), (2, "dfr_k2", "cps_k2"), (3, "dfr", "cps")):
        out["pressure_count"][str(k)] = {"dfr": score(dfr_col), "cps": score(cps_col), "calls": 1 + k}
    out["signal"] = {"dfr": score("dfr"), "f_counter": score("f_counter"), "cps": score("cps"),
                     "dfr_minus_cps": difference("dfr", "cps"),
                     "dfr_minus_f_counter": difference("dfr", "f_counter")}

    strict = frame[frame["strict_group"].isin(["CW", "CR"])]
    out["strict"] = {"n_cw": int((strict["strict_group"] == "CW").sum()),
                     "n_cr": int((strict["strict_group"] == "CR").sum()),
                     "dfr": score("dfr", strict),
                     "baseline": score(baseline_key, strict) if baseline_key else None}

    with_random = frame[frame["cps_random"].notna()]
    if len(with_random) and with_random["label"].nunique() == 2:
        out["random_alternative"] = {"n": int(len(with_random)), "cps_self": score("cps", with_random),
                                     "cps_random": score("cps_random", with_random)}

    out["alternative_equivalent_count"] = int((frame["alternative_equivalent"] == 1).sum())

    out["combination"] = {"dfr": score("dfr"), "cross_family": score("cross_family"),
                          "rank_mean": score("dfr_cross"), "cps_rank_mean": score("cps_cross")}
    features = frame[["f_counter", "f_doubt", "cross_family"]].astype(float)
    features = features.loc[:, features.notna().any()]
    if not features.empty:
        filled = np.column_stack([impute(features[c].to_numpy())[0] for c in features.columns])
        oof = logistic_out_of_fold(filled, labels, config.cv_folds, config.seed)
        if not np.isnan(oof).all():
            out["combination"]["logistic_cv"] = auroc(labels, oof)
    return out


def hypotheses(methods: dict, comparison: dict, ablation: dict, baseline_key: str | None) -> dict:
    """Criteria fixed after the pilot and before the full run (protocol change log, 27 October 2026)."""
    def interval(key: str) -> dict | None:
        return methods.get(key, {}).get("auroc_matched")

    def includes_half(key: str) -> bool | None:
        found = interval(key)
        return None if found is None else bool(found["low"] <= 0.5 <= found["high"])

    out = {"H1": {"semantic_entropy_ci_includes_half": includes_half("semantic_entropy"),
                  "selfcheck_ci_includes_half": includes_half("selfcheck_nli")}}
    h1_values = [v for v in out["H1"].values() if v is not None]
    out["H1"]["supported"] = bool(h1_values) and all(h1_values)

    ours = interval(PRIMARY)
    if ours is not None:
        sampling = {}
        for key in SAMPLING_BASELINES:
            if key in methods:
                sampling[key] = {"auroc": methods[key]["auroc_matched"]["value"],
                                 "p_holm": comparison.get(key, {}).get("p_holm")}
        beats = [ours["value"] > entry["auroc"] and entry["p_holm"] is not None and entry["p_holm"] < 0.05
                 for entry in sampling.values()]
        out["H2"] = {"dfr": ours["value"], "dfr_low": ours["low"], "sampling": sampling,
                     "supported": bool(sampling) and all(beats) and ours["low"] > 0.5}
        if baseline_key:
            out["H2"]["best_text_baseline"] = baseline_key
            out["H2"]["baseline_auroc"] = methods[baseline_key]["auroc_matched"]["value"]
            out["H2"]["baseline_p_holm"] = comparison.get(baseline_key, {}).get("p_holm")

    signal = ablation.get("signal", {})
    diff = signal.get("dfr_minus_cps")
    if diff is not None:
        out["H3"] = {"dfr": signal.get("dfr"), "cps": signal.get("cps"), "difference": diff,
                     "supported": bool(diff["low"] == diff["low"] and diff["low"] > 0)}
    combo = ablation.get("combination", {})
    if combo.get("rank_mean") is not None and combo.get("cross_family") is not None and combo.get("dfr") is not None:
        out["H4"] = {**combo, "supported": combo["rank_mean"] > max(combo["dfr"], combo["cross_family"])}
    return out
