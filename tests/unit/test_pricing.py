import pytest

from answersnap import pricing

USAGE = {"input_tokens": 1_000_000, "output_tokens": 100_000, "reasoning_tokens": 40_000,
         "output_includes_reasoning": True}


def test_known_model_matches_a_hand_calculation():
    usd, basis = pricing.estimate_cost_usd("anthropic", "claude-sonnet-5", USAGE, 4)
    assert usd == pytest.approx(2.00 + 1.00 + 0.04)
    assert basis == {"kind": "list_price", "as_of": pricing.PRICES_AS_OF,
                     "model": "claude-sonnet-5", "searches_priced": True}


def test_a_dated_snapshot_id_finds_its_model_but_a_sibling_does_not():
    assert pricing.list_price("openai", "gpt-5.1-2025-11-13")[0] == "gpt-5.1"
    assert pricing.list_price("openai", "gpt-5.1-mini") == (None, None)
    assert pricing.list_price("google", "claude-sonnet-5") == (None, None)


def test_unknown_model_gets_no_number():
    assert pricing.estimate_cost_usd("openai", "gpt-unknown", USAGE, 1) == (
        None, {"kind": "none", "reason": "no_list_price"})


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 5, "output_tokens": None}])
def test_missing_usage_gets_no_number(usage):
    assert pricing.estimate_cost_usd("anthropic", "claude-sonnet-5", usage, 1)[1]["reason"] \
        == "no_usage"


def test_google_bills_thinking_on_top_of_candidates_and_leaves_search_unpriced():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000, "reasoning_tokens": 100_000,
             "output_includes_reasoning": False}
    usd, basis = pricing.estimate_cost_usd("google", "gemini-3.7-flash", usage, None)
    assert usd == pytest.approx(0.75 + 0.75)
    assert basis["searches_priced"] is False


def test_search_fee_is_added_only_when_the_count_is_known():
    without, basis = pricing.estimate_cost_usd("openai", "gpt-5.1", USAGE, None)
    with_count, _ = pricing.estimate_cost_usd("openai", "gpt-5.1", USAGE, 2)
    assert basis["searches_priced"] is False
    assert with_count == pytest.approx(without + 0.02)
