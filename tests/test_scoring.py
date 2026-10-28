import math

import numpy as np
import pytest

from pressure_probe.scoring.baselines import (cluster_entropy, greedy_clusters, logprob_score, ptrue_score,
                                              selfcheck_score, verbal_score)
from pressure_probe.scoring.combine import logistic_out_of_fold, rank_mean
from pressure_probe.scoring.cps import contrastive_pressure, flip_rate
from pressure_probe.scoring.metaqa_score import metaqa_score
from pressure_probe.stages.metaqa import parse_numbered, parse_verdict


def test_cps_subtracts_doubt():
    assert contrastive_pressure([True, True, False], [True, False, False]) == pytest.approx(1 / 3)


def test_cps_uses_first_k():
    assert contrastive_pressure([True, False, False], [False, True, True], k=1) == pytest.approx(1.0)


def test_flip_rate_empty_is_nan():
    assert math.isnan(flip_rate([]))


def test_metaqa_mapping_follows_paper():
    assert metaqa_score(["yes", "yes"], ["no", "no"]) == 0.0
    assert metaqa_score(["no"], ["yes"]) == 1.0
    assert metaqa_score(["not_sure"], ["not_sure"]) == 0.5


def test_semantic_entropy_zero_for_one_cluster():
    labels = greedy_clusters(["a", "a", "a"], lambda x, y: x == y)
    assert cluster_entropy(labels) == 0.0


def test_semantic_entropy_two_equal_clusters():
    labels = greedy_clusters(["a", "b", "a", "b"], lambda x, y: x == y)
    assert cluster_entropy(labels) == pytest.approx(math.log(2))


def test_selfcheck_normalises_contradiction():
    assert selfcheck_score([(0.9, 0.1), (0.1, 0.9)]) == pytest.approx(0.5)


def test_ptrue_handles_case_and_spaces():
    score = ptrue_score({" True": math.log(0.8), "false": math.log(0.2)})
    assert score == pytest.approx(0.2)


def test_ptrue_missing_tokens_is_nan():
    assert math.isnan(ptrue_score({"Maybe": -0.1}))


def test_verbal_parsing_and_clipping():
    assert verbal_score("85") == pytest.approx(0.15)
    assert verbal_score("I'd say 120%") == pytest.approx(0.0)
    assert math.isnan(verbal_score("not sure"))


def test_logprob_score_is_negative_mean():
    assert logprob_score([-1.0, -3.0]) == pytest.approx(2.0)


def test_rank_mean_order():
    combined = rank_mean(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert list(np.argsort(combined)) == [0, 1, 2]


def test_logistic_out_of_fold_shape():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 20)
    features = labels[:, None] + rng.normal(0, 0.5, (40, 2))
    predictions = logistic_out_of_fold(features, labels, 5, 0)
    assert predictions.shape == (40,) and not np.isnan(predictions).any()


def test_parse_numbered_and_verdicts():
    assert parse_numbered("1. A\n2) B\nnoise\n3. C", 2) == ["A", "B"]
    assert parse_verdict("Not sure.") == "not_sure"
    assert parse_verdict("Yes, it is") == "yes"
    assert parse_verdict("No") == "no"
    assert parse_verdict("Perhaps") == "not_sure"
