import pytest
from builders import citation, make_config, record

from answersnap.metrics import rates
from answersnap.metrics.stats import wilson_interval


def test_metric_carries_count_interval_and_low_sample_status():
    m = rates.metric(3, 6)
    assert (m["count"], m["n"], m["rate"]) == (3, 6, 0.5)
    assert (m["ci_low"], m["ci_high"]) == pytest.approx(wilson_interval(3, 6))
    assert m["status"] == rates.STATUS_LOW_SAMPLE
    assert rates.metric(5, 10)["status"] == rates.STATUS_OK


def test_empty_metric_is_a_status_never_zero():
    m = rates.metric(0, 0, status_if_empty=rates.STATUS_NOT_OBSERVABLE)
    assert m["rate"] is None and m["count"] is None
    assert m["status"] == rates.STATUS_NOT_OBSERVABLE


def test_mention_uses_aliases_deterministically():
    config = make_config()
    row = rates.analyse_record(record(text="I like Trellis Dental."), config)
    assert row["mentioned"] is True
    assert row["brand_sentence"] == "I like Trellis Dental."
    assert row["competitors_named"] == []


def test_named_in_rec_answer_is_none_off_recommendation_questions():
    config = make_config()
    on = rates.analyse_record(record(intent="recommendation", text="no brand"), config)
    off = rates.analyse_record(record(intent="how_to", text="Trellis"), config)
    assert on["named_in_rec_answer"] is False
    assert off["named_in_rec_answer"] is None


def test_owned_citation_counts_cited_sources_on_owned_hosts_only():
    config = make_config()
    row = rates.analyse_record(record(citations=[
        citation("https://blog.trellis.example/post"),           # subdomain: owned
        citation("https://trellis.example/x", is_cited=False),   # retrieved only
        citation("https://nottrellis.example/y"),                # suffix trap
    ]), config)
    assert row["owned_cited"] is True
    assert row["owned_cited_urls"] == ["https://blog.trellis.example/post"]


def test_unobservable_citations_are_none_not_false():
    row = rates.analyse_record(record(observable=False, citations=[]), make_config())
    assert row["citations_observable"] is False
    assert row["owned_cited"] is None


def test_engines_are_never_pooled():
    config = make_config()
    records = [record(engine="anthropic", repeat=r, text="Trellis") for r in range(2)]
    records += [record(engine="openai", repeat=r, text="nothing") for r in range(2)]
    _, per_engine, across = rates.compute(records, config)
    assert per_engine["anthropic"]["mention"]["count"] == 2
    assert per_engine["openai"]["mention"]["count"] == 0
    assert across == {"count": 2, "n": 4, "engines": ["anthropic", "openai"]}


def test_citation_denominator_excludes_unobservable_answers():
    config = make_config()
    records = [record(repeat=0, citations=[citation("https://trellis.example/a")]),
               record(repeat=1, observable=False)]
    _, per_engine, _ = rates.compute(records, config)
    assert (per_engine["anthropic"]["citation"]["count"], per_engine["anthropic"]["citation"]["n"]) == (1, 1)


def test_engine_with_no_observable_citations_reports_not_observable():
    config = make_config()
    _, per_engine, _ = rates.compute([record(observable=False)], config)
    assert per_engine["anthropic"]["citation"]["status"] == rates.STATUS_NOT_OBSERVABLE


def test_not_measured_when_the_config_cannot_measure_it():
    config = make_config(recommendation_intents=["none"], brand={"name": "Trellis"})
    _, per_engine, across = rates.compute([record()], config)
    assert per_engine["anthropic"]["named_in_rec_answers"]["status"] == rates.STATUS_NOT_MEASURED
    assert per_engine["anthropic"]["citation"]["status"] == rates.STATUS_NOT_MEASURED
    assert across["n"] == 0


def test_placeholders_are_left_out_of_every_denominator():
    config = make_config()
    records = [record(text="Trellis"), record(repeat=1, params={"no_fixture": True}, text="x")]
    rows, per_engine, _ = rates.compute(records, config)
    assert len(rows) == 1
    assert per_engine["anthropic"]["mention"]["n"] == 1


def test_competitor_mentions_are_reported_per_engine():
    config = make_config()
    _, per_engine, _ = rates.compute([record(text="Kanbanly and Trellis")], config)
    assert per_engine["anthropic"]["competitors"][0]["name"] == "Kanbanly"
    assert per_engine["anthropic"]["competitors"][0]["mention"]["count"] == 1


def test_engine_without_answers_is_absent_rather_than_zero():
    _, per_engine, _ = rates.compute([record(engine="anthropic")], make_config())
    assert "openai" not in per_engine
