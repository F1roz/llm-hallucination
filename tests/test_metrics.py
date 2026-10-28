import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pressure_probe.metrics import (aurac, auroc, bootstrap_auroc, bootstrap_auroc_difference, delong_test,
                                    holm, impute)


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    strong = labels + rng.normal(0, 0.8, n)
    weak = labels + rng.normal(0, 2.5, n)
    return labels, strong, weak


def test_auroc_matches_sklearn():
    labels, strong, _ = _data()
    assert auroc(labels, strong) == pytest.approx(roc_auc_score(labels, strong))


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.ones(5), np.arange(5)))


def test_delong_auc_values_match_sklearn():
    labels, strong, weak = _data()
    a1, a2, _, _ = delong_test(labels, strong, weak)
    assert a1 == pytest.approx(roc_auc_score(labels, strong))
    assert a2 == pytest.approx(roc_auc_score(labels, weak))


def test_delong_identical_scores_not_significant():
    labels, strong, _ = _data()
    _, _, z, p = delong_test(labels, strong, strong.copy())
    assert z == 0.0 and p == 1.0


def test_delong_detects_large_difference():
    labels, strong, weak = _data(n=600)
    _, _, _, p = delong_test(labels, strong, weak)
    assert p < 0.001


def test_delong_standard_error_agrees_with_bootstrap():
    labels, strong, weak = _data(seed=3, n=400)
    _, _, z, _ = delong_test(labels, strong, weak)
    diff = auroc(labels, strong) - auroc(labels, weak)
    delong_se = diff / z
    boot = bootstrap_auroc_difference(labels, strong, weak, 2000, seed=1)
    boot_se = (boot.high - boot.low) / (2 * 1.96)
    assert delong_se == pytest.approx(boot_se, rel=0.25)


def test_bootstrap_interval_contains_point():
    labels, strong, _ = _data()
    interval = bootstrap_auroc(labels, strong, 500, seed=2)
    assert interval.low <= interval.value <= interval.high


def test_bootstrap_is_deterministic():
    labels, strong, _ = _data()
    assert bootstrap_auroc(labels, strong, 200, 5) == bootstrap_auroc(labels, strong, 200, 5)


def test_holm_known_values():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_is_monotone_and_capped():
    adjusted = holm([0.5, 0.9, 0.01])
    assert max(adjusted) <= 1.0
    assert adjusted[2] <= adjusted[0] <= adjusted[1]


def test_aurac_perfect_ranking_beats_reversed():
    correct = np.array([1, 1, 1, 0, 0])
    good = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert aurac(correct, good) > aurac(correct, -good)
    assert aurac(correct, good) == pytest.approx(np.mean([1, 1, 1, 3 / 4, 3 / 5]))


def test_impute_uses_median_and_counts():
    filled, missing = impute(np.array([0.2, np.nan, 0.6]))
    assert missing == 1
    assert filled[1] == pytest.approx(0.4)


def test_delong_with_too_few_items_per_class_is_missing_not_an_error():
    import math
    labels = np.array([1, 0, 0, 0])
    a1, a2, z, p = delong_test(labels, np.array([0.9, 0.1, 0.2, 0.3]), np.array([0.5, 0.4, 0.3, 0.2]))
    assert a1 == 1.0 and math.isnan(z) and math.isnan(p)


def test_holm_leaves_missing_values_missing():
    import math
    adjusted = holm([0.01, float("nan"), 0.04])
    assert math.isnan(adjusted[1])
    assert adjusted[0] == 0.02 and adjusted[2] == 0.04
