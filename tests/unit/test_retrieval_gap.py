"""Whether a quoted sentence really appears on the page it was attributed to.

The export exists to answer one question before v2 commits to this feature: on
honest data, does the score have a clean low end? So the comparison has to be
offered at three strictnesses, and the benign causes of a mismatch have to stay
visible rather than being filtered out.
"""

from answersnap.metrics.retrieval_gap import best_window_score, compare

PAGE = "本院提供植牙、矯正與美白服務。植牙採用瑞典植體，術後保固五年。"


def test_an_exact_quote_matches_at_every_strictness():
    result = compare("術後保固五年", PAGE)
    assert result["exact_match"] is True
    assert result["normalized_match"] is True
    assert result["best_window_score"] == 1.0


def test_whitespace_differences_survive_normalisation_but_not_exactness():
    result = compare("植牙 採用 瑞典植體", PAGE)
    assert result["exact_match"] is False
    # This is why one boolean is not enough: the judgment layer needs to see
    # where along the strictness scale the signal survives.
    assert result["best_window_score"] > 0.9


def test_a_quote_that_is_nowhere_on_the_page_scores_low():
    assert compare("本院提供免費停車與接送服務", PAGE)["best_window_score"] < 0.6


def test_a_paraphrase_lands_between_the_two():
    score = compare("使用瑞典的植體，保固五年", PAGE)["best_window_score"]
    assert 0.5 < score < 1.0


def test_an_empty_side_scores_zero_rather_than_matching():
    assert best_window_score("", PAGE) == 0.0
    assert best_window_score("術後保固五年", "") == 0.0
    assert best_window_score("術後保固五年", None) == 0.0


def test_a_quote_longer_than_the_page_still_scores():
    # Truncated fetches happen; this must not divide by zero or claim a match.
    score = best_window_score(PAGE + PAGE, "植牙")
    assert 0.0 <= score < 0.5


def test_latin_text_is_compared_by_word():
    page = "We offer dental implants with a five year warranty."
    assert compare("five year warranty", page)["best_window_score"] == 1.0
    assert compare("free parking available", page)["best_window_score"] < 0.5


def test_case_differences_do_not_count_as_a_gap():
    assert compare("FIVE YEAR WARRANTY",
                   "a five year warranty")["best_window_score"] == 1.0
