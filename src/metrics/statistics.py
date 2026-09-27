"""Statistical reporting: sample size, confidence intervals, paired significance tests and
effect sizes, so differences between models / representations / constraints can be
reported as more than a difference of means.

describe()  - n, mean, std, median, q80, min, max and the 95% CI of the mean (Student t).
compare()   - paired comparison of two models on the SAME instances: Wilcoxon signed-rank
              test (no normality assumption), paired Cohen's d, and the matched-pairs
              rank-biserial correlation (the Wilcoxon test's own effect size).
required_sample_size() - instances needed to detect a given paired effect size.
"""
import math

import numpy as np
from scipy import stats


def _clean(values):
    return np.array([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)


def describe(values, confidence=0.95):
    x = _clean(values)
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf((1 + confidence) / 2, n - 1)) * std / math.sqrt(n)
    else:
        half = float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(x)),
        "q80": float(np.percentile(x, 80)),
        "min": float(x.min()),
        "max": float(x.max()),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def compare(a, b, alpha=0.05):
    """Paired comparison of a vs b (same instances, same order). Negative mean_diff /
    effect sizes mean a is lower (better, for makespan or relative error)."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))]
    n = len(pairs)
    if n < 2:
        return {"n": n}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    d = x - y
    sd = float(d.std(ddof=1))
    nonzero = d[d != 0]
    if len(nonzero) > 0:
        w = stats.wilcoxon(x, y, zero_method="wilcox")
        p_value = float(w.pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        r_plus = float(ranks[nonzero > 0].sum())
        r_minus = float(ranks[nonzero < 0].sum())
        rank_biserial = (r_plus - r_minus) / (r_plus + r_minus)
    else:
        p_value, rank_biserial = 1.0, 0.0
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "wilcoxon_p": p_value,
        "significant": bool(p_value < alpha),
        "cohens_d": float(d.mean()) / sd if sd > 0 else 0.0,
        "rank_biserial": float(rank_biserial),
        "a_better_count": int((d < 0).sum()),
        "b_better_count": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
    }


def required_sample_size(effect_size, alpha=0.05, power=0.8):
    """Instances needed for a two-sided paired t-test to detect |Cohen's d| = effect_size
    (normal approximation, +1 for small samples). 33 for a medium effect (0.5), 198 for a
    small one (0.2), at alpha=0.05 and power=0.8."""
    if effect_size == 0:
        return None
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_b) / abs(effect_size)) ** 2)) + 1
