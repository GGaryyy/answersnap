import pytest

from answersnap.metrics.stats import MIN_SAMPLES_FOR_RATE, wilson_interval


def test_interval_matches_the_wilson_formula():
    low, high = wilson_interval(4, 9)
    assert low == pytest.approx(0.1888, abs=1e-4)
    assert high == pytest.approx(0.7333, abs=1e-4)


def test_interval_stays_inside_zero_and_one_at_the_edges():
    low, high = wilson_interval(0, 6)
    assert low == pytest.approx(0, abs=1e-12) and 0 < high < 0.5
    low, high = wilson_interval(6, 6)
    assert 0.5 < low < 1 and high == pytest.approx(1)


def test_smaller_samples_give_wider_intervals():
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_no_samples_has_no_interval_and_impossible_counts_raise():
    assert wilson_interval(0, 0) == (None, None)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    assert MIN_SAMPLES_FOR_RATE == 10
