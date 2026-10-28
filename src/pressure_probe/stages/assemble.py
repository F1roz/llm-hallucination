"""Join every stage output into one table per model: one row per question, one column per score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import read_json, read_jsonl
from ..scoring.baselines import logprob_score, ptrue_score, verbal_score
from ..scoring.cps import contrastive_pressure, flip_rate
from ..scoring.metaqa_score import metaqa_score
from . import paths


def build_table(config: Config, model_key: str) -> pd.DataFrame:
    split = pd.DataFrame(read_jsonl(paths.split(config, model_key))).set_index("qid")
    gens = {r["qid"]: r for r in read_jsonl(paths.generations(config, model_key))}
    samples = pd.DataFrame(read_jsonl(paths.sample_scores(config, model_key))).set_index("qid")
    table = split.join(samples[["semantic_entropy", "selfcheck_nli", "cross_family", "n_clusters"]])
    table["logprob"] = [logprob_score(gens[q]["greedy_logprobs"]) for q in table.index]
    table["ptrue"] = [ptrue_score(gens[q]["ptrue_first_token"]) for q in table.index]
    table["verbal"] = [verbal_score(gens[q]["verbal_raw"]) for q in table.index]
    table["greedy"] = [gens[q]["greedy"] for q in table.index]
    table["matched"] = table.index.isin(read_json(paths.matched(config, model_key)))
    table["cross_family"] = table["cross_family"].astype(float)
    _add_probe_columns(config, model_key, table)
    _add_metaqa(config, model_key, table)
    table.to_csv(paths.scores_table(config, model_key))
    return table


def _add_probe_columns(config: Config, model_key: str, table: pd.DataFrame) -> None:
    columns = ["f_counter", "f_doubt", "dfr", "cps", "cps_k1", "cps_k2", "f_counter_k1", "f_counter_k2",
               "dfr_k1", "dfr_k2", "f_random", "cps_random", "alternative_equivalent"]
    for column in columns:
        table[column] = np.nan
    for row in read_jsonl(paths.probe_scores(config, model_key)):
        qid = row["qid"]
        counter, doubt, rand = row["counter_flips"], row["doubt_flips"], row.get("random_counter_flips", [])
        table.loc[qid, "f_counter"] = flip_rate(counter)
        table.loc[qid, "f_doubt"] = flip_rate(doubt)
        table.loc[qid, "dfr"] = flip_rate(doubt)  # the doubt flip rate is the main score
        table.loc[qid, "cps"] = contrastive_pressure(counter, doubt)
        for k in (1, 2):
            table.loc[qid, f"cps_k{k}"] = contrastive_pressure(counter, doubt, k)
            table.loc[qid, f"f_counter_k{k}"] = flip_rate(counter, k)
            table.loc[qid, f"dfr_k{k}"] = flip_rate(doubt, k)
        if rand:
            table.loc[qid, "f_random"] = flip_rate(rand)
            table.loc[qid, "cps_random"] = contrastive_pressure(rand, doubt)
        table.loc[qid, "alternative_equivalent"] = float(row["alternative_equivalent"])


def _add_metaqa(config: Config, model_key: str, table: pd.DataFrame) -> None:
    table["metaqa"] = np.nan
    path = paths.metaqa(config, model_key)
    if not path.exists():
        return
    for row in read_jsonl(path):
        table.loc[row["qid"], "metaqa"] = metaqa_score(row["synonym_verdicts"], row["antonym_verdicts"])
