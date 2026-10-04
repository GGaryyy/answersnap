import pytest

from answersnap.metrics import rates
from answersnap.report import format as fmt


def test_metric_text_shows_count_rate_and_interval():
    text = fmt.metric_text(rates.metric(4, 9))
    assert text.startswith("4 of 9 (44%, 95% CI ")
    assert text.endswith(" · low sample")
    assert "low sample" not in fmt.metric_text(rates.metric(5, 10))


@pytest.mark.parametrize("status, words", [
    (rates.STATUS_NOT_OBSERVABLE, "not observable"),
    (rates.STATUS_NOT_MEASURED, "not measured"),
    (rates.STATUS_NO_DATA, "no answers"),
])
def test_missing_numbers_are_words_never_zero(status, words):
    assert fmt.metric_text(rates.metric(0, 0, status_if_empty=status)) == words
    assert fmt.metric_text(None) == "no answers"


def test_yes_blank_distinguishes_no_from_not_asked():
    assert (fmt.yes_blank(True), fmt.yes_blank(False), fmt.yes_blank(None)) == ("yes", "", "–")


@pytest.mark.parametrize("row, text", [
    ({"result": "found", "not_comparable_reason": None}, "found verbatim"),
    ({"result": "not_found", "not_comparable_reason": None, "best_window_score": 0.734},
     "not found verbatim (best overlap 0.73)"),
    ({"result": "not_checkable", "not_comparable_reason": "no_cited_text"}, "no quote returned"),
    ({"result": "not_checkable", "not_comparable_reason": "fetch_paywalled"}, "not checkable: paywalled"),
    ({"result": "not_checkable", "not_comparable_reason": "fetch_new_reason"}, "not checkable: fetch new reason"),
])
def test_faith_row_text(row, text):
    assert fmt.faith_row_text(row) == text


def test_faith_summary_text_names_not_observable():
    assert fmt.faith_summary_text({"status": "not_observable"}).startswith("not observable")
    assert fmt.faith_summary_text(None) == "no cited sources"
    ok = {"status": "ok", "found": 2, "with_quote": 4, "checked": 3, "not_found": 1,
          "not_checkable": 1}
    assert fmt.faith_summary_text(ok) == ("2 of 3 checked quotes found on the page verbatim; "
                                          "1 not found; 1 more could not be checked")


def test_engine_status_text_prefers_the_skip_reason():
    assert fmt.engine_status_text({"status": "skipped_no_credentials",
                                   "reason": "Gemini: not run — GOOGLE_API_KEY is not set"}) \
        == "Gemini: not run — GOOGLE_API_KEY is not set"
    assert fmt.engine_status_text({"status": "partial", "models": ["m"], "done": 2,
                                   "planned": 4}) == "m · partial: 2 of 4"
