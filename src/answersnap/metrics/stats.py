"""Interval arithmetic shared by every rate this package reports."""

import math

# ---------------------------------------------------------------- constants
# Below this many samples an appearance rate says more about our sampling than
# about the brand.
MIN_SAMPLES_FOR_RATE = 10
# Wilson interval; 1.96 is the 95% two-sided normal quantile.
Z_95 = 1.96


def wilson_interval(successes, total, z=Z_95):
    """Confidence interval that stays sane at small n and at rates near 0 or 1."""
    if total <= 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError(f"{successes} successes out of {total} is not possible")
    rate = successes / total
    denominator = 1 + z ** 2 / total
    centre = rate + z ** 2 / (2 * total)
    spread = z * math.sqrt(rate * (1 - rate) / total + z ** 2 / (4 * total ** 2))
    return ((centre - spread) / denominator, (centre + spread) / denominator)
