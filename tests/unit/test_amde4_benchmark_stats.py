"""Regression tests verifying exact statistical computations for the AMDE benchmark.

Verifies:
1. Exhaustive sign-flip permutation Wilcoxon signed-rank calculations across 2^N space.
2. Monotonic Holm-Bonferroni step-down adjustment on paired t-tests.
3. Proper handling of zeros (reducing sample size to N_eff) and tied ranks.
4. Non-significance conclusions under family-wise error rate control.
"""

import itertools
import numpy as np
import pytest
from scipy import stats


def exact_sign_flip_wilcoxon(diffs: np.ndarray) -> tuple[float, float, int, int]:
    """Computes exact two-sided Wilcoxon signed-rank test via exhaustive sign-flip permutation.

    Returns:
        (W_observed, p_value, n_eff, n_permutations)
    """
    non_zero = diffs[diffs != 0]
    n_eff = len(non_zero)
    if n_eff == 0:
        return 0.0, 1.0, 0, 1

    abs_diffs = np.abs(non_zero)
    ranks = stats.rankdata(abs_diffs)
    total_rank_sum = float(np.sum(ranks))

    # Observed W
    w_plus = float(np.sum(ranks[non_zero > 0]))
    w_minus = float(np.sum(ranks[non_zero < 0]))
    w_obs = min(w_plus, w_minus)

    # Exhaustive enumeration across 2^N sign flips
    signs_product = list(itertools.product([-1, 1], repeat=n_eff))
    n_perms = len(signs_product)
    count_as_or_more_extreme = 0

    for signs in signs_product:
        perm_w_plus = sum(r for r, s in zip(ranks, signs) if s > 0)
        perm_w_minus = total_rank_sum - perm_w_plus
        perm_w = min(perm_w_plus, perm_w_minus)
        if perm_w <= w_obs + 1e-9:
            count_as_or_more_extreme += 1

    p_val = count_as_or_more_extreme / n_perms
    return w_obs, p_val, n_eff, n_perms


def monotonic_holm_adjust(p_values: list[float]) -> list[float]:
    """Computes monotonic Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = m - rank
        adj_p = min(1.0, multiplier * p)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max

    return adjusted


def test_wilcoxon_native_tpe():
    # Differences: AMDE - Native TPE (in samples correct / 360)
    # [11, 6, 3, 1, 6] / 360
    diffs = np.array([11, 6, 3, 1, 6]) / 360.0
    w_obs, p_val, n_eff, n_perms = exact_sign_flip_wilcoxon(diffs)

    assert n_eff == 5
    assert n_perms == 32
    assert w_obs == 0.0
    assert p_val == pytest.approx(2.0 / 32.0, rel=1e-5)  # 0.06250


def test_wilcoxon_optuna_tpe():
    # Differences: AMDE - Optuna TPE (in samples correct / 360)
    # [4, -1, -1, -1, 4] / 360
    diffs = np.array([4, -1, -1, -1, 4]) / 360.0
    w_obs, p_val, n_eff, n_perms = exact_sign_flip_wilcoxon(diffs)

    assert n_eff == 5
    assert n_perms == 32
    assert w_obs == 6.0
    assert p_val == pytest.approx(20.0 / 32.0, rel=1e-5)  # 0.62500


def test_wilcoxon_random_search_with_zero():
    # Differences: AMDE - Random Search (in samples correct / 360)
    # [1, -6, 3, -1, 0] / 360 (one zero on seed 404)
    diffs = np.array([1, -6, 3, -1, 0]) / 360.0
    w_obs, p_val, n_eff, n_perms = exact_sign_flip_wilcoxon(diffs)

    assert n_eff == 4
    assert n_perms == 16
    assert w_obs == 4.5
    assert p_val == pytest.approx(16.0 / 16.0, rel=1e-5)  # 1.00000


def test_holm_bonferroni_wilcoxon():
    # Raw p-values: [Random Search: 1.0, Native TPE: 0.0625, Optuna: 0.625]
    raw_p = [1.0, 0.0625, 0.625]
    adj_p = monotonic_holm_adjust(raw_p)

    # Native TPE: 0.0625 * 3 = 0.1875
    assert adj_p[1] == pytest.approx(0.1875, rel=1e-5)
    # Optuna: max(0.1875, 0.625 * 2 = 1.25 -> 1.0) = 1.0
    assert adj_p[2] == pytest.approx(1.0, rel=1e-5)
    # Random Search: max(1.0, 1.0 * 1 = 1.0) = 1.0
    assert adj_p[0] == pytest.approx(1.0, rel=1e-5)


def test_monotonic_holm_paired_t_tests():
    # Raw p-values: [Native TPE: 0.03308, Optuna: 0.45955, Random Search: 0.71057]
    raw_p = [0.03308, 0.45955, 0.71057]
    adj_p = monotonic_holm_adjust(raw_p)

    assert adj_p[0] == pytest.approx(0.09924, rel=1e-4)
    assert adj_p[1] == pytest.approx(0.91910, rel=1e-4)
    assert adj_p[2] == pytest.approx(0.91910, rel=1e-4)

    # Monotonicity check
    assert adj_p[0] <= adj_p[1] <= adj_p[2]

    # No comparison is significant at alpha=0.05
    for p in adj_p:
        assert p >= 0.05
