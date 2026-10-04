"""Alias matching, and the offsets every piece of evidence hangs off."""

import pytest

from answersnap.metrics.mention import (
    extract_entities,
    find_alias_spans,
    find_unknown_entities,
    normalise,
    normalise_with_map,
)

BRAND = ("範例甲牙醫", "brand", ["範例甲", "TRLSD"])
COMPETITOR = ("樣本美學牙醫", "competitor", ["樣本"])


def test_offsets_address_the_raw_text():
    text = "推薦範例甲牙醫，也可以看看樣本美學牙醫。"
    mentions = {m["entity"]: m for m in extract_entities(text, [BRAND, COMPETITOR])}
    # Per entity, not "one of the expected strings": a set-style assertion would
    # pass even if the two offsets were swapped.
    for name in ("範例甲牙醫", "樣本美學牙醫"):
        span = mentions[name]
        assert span["char_start"] == text.index(name)
        assert text[span["char_start"]:span["char_end"]] == name


def test_full_width_forms_match_without_shifting_offsets():
    # NFKC collapses full-width Latin to ASCII, so a match found in a normalised
    # copy sits at a different index than in the original. Every evidence link
    # would point at the wrong characters.
    text = "這家診所叫做ＴＲＬＳＤ，值得看看。"
    spans = find_alias_spans(text, ["TRLSD"])
    assert spans
    start, end, _ = spans[0]
    assert text[start:end] == "ＴＲＬＳＤ"


def test_emoji_before_the_match_do_not_shift_offsets():
    text = "你好🦷 範例甲牙醫 很不錯"
    start, end, _ = find_alias_spans(text, ["範例甲牙醫"])[0]
    assert text[start:end] == "範例甲牙醫"
    # Code points, not bytes or UTF-16 units — the emoji differs in all three.
    assert start == text.index("範例甲牙醫")


def test_matching_is_case_insensitive():
    assert find_alias_spans("visit trlsd today", ["TRLSD"])


def test_the_longest_alias_wins_over_one_it_contains():
    text = "推薦範例甲牙醫診所"
    spans = find_alias_spans(text, ["範例甲", "範例甲牙醫", "範例甲牙醫診所"])
    # Otherwise one clinic counts as three mentions.
    assert len(spans) == 1
    assert text[spans[0][0]:spans[0][1]] == "範例甲牙醫診所"


def test_rank_follows_order_of_appearance():
    text = "首先是樣本美學牙醫，其次才是範例甲牙醫。"
    mentions = {m["entity"]: m["rank"] for m in extract_entities(text, [BRAND, COMPETITOR])}
    assert mentions["樣本美學牙醫"] == 1
    assert mentions["範例甲牙醫"] == 2


def test_an_entity_mentioned_twice_is_one_mention_ranked_by_first_position():
    text = "範例甲牙醫很好。真的，範例甲牙醫值得推薦。"
    mentions = extract_entities(text, [BRAND])
    assert len(mentions) == 1
    assert mentions[0]["char_start"] == text.index("範例甲牙醫")


def test_an_absent_brand_produces_no_mention():
    # This is the row that later means "not mentioned"; inventing one here would
    # quietly erase a suppression signal.
    assert extract_entities("這裡只談樣本美學牙醫。", [BRAND]) == []


def test_sentiment_is_left_unset():
    mentions = extract_entities("範例甲牙醫不錯。", [BRAND])
    assert "sentiment" not in mentions[0]


def test_unknown_clinic_names_are_surfaced_for_confirmation():
    text = "推薦範例甲牙醫，另外示範牙醫診所跟虛構牙醫也不錯。"
    known = extract_entities(text, [BRAND])
    unknown = {u["entity"] for u in find_unknown_entities(text, known)}
    # These become entity_role='unknown' rows: the alias table fills itself in,
    # and unheard-of competitors surface on their own.
    assert "示範牙醫診所" in unknown
    assert "虛構牙醫" in unknown


def test_a_known_entity_is_not_also_reported_as_unknown():
    text = "推薦範例甲牙醫。"
    known = extract_entities(text, [BRAND])
    assert find_unknown_entities(text, known) == []


def test_unknown_spans_carry_usable_offsets():
    text = "另外示範牙醫診所也不錯，他們說「 台北範例牙醫 」也可以。"
    candidates = find_unknown_entities(text, [])
    # Without this the loop below passes vacuously when nothing is found.
    assert candidates
    for candidate in candidates:
        assert text[candidate["char_start"]:candidate["char_end"]] == candidate["entity"]


def test_the_same_unknown_name_is_reported_once_per_answer():
    text = "示範牙醫診所很好，示範牙醫診所真的很好。"
    assert len(find_unknown_entities(text, [])) == 1


@pytest.mark.parametrize("text", ["", "   ", "。。。"])
def test_empty_answers_do_not_break_matching(text):
    assert extract_entities(text, [BRAND]) == []


def test_every_normalised_character_maps_back_to_the_span_it_came_from():
    text = "ＡＢ🦷c"
    normalised, start_map, end_map = normalise_with_map(text)
    assert len(normalised) == len(start_map) == len(end_map)
    for index, folded_char in enumerate(normalised):
        raw_span = text[start_map[index]:end_map[index]]
        # Not just "in range": the character really is what that raw span folds
        # to. Every evidence link in the product rides on this.
        assert folded_char in normalise(raw_span)


def test_a_decomposed_accent_still_matches_a_composed_alias():
    import unicodedata

    text = unicodedata.normalize("NFD", "Visit Café Dental today")
    spans = find_alias_spans(text, ["Café Dental"])
    # Normalising one character at a time can never compose e + U+0301 into é.
    # A missed alias looks exactly like a suppressed brand.
    assert spans
    assert normalise(text[spans[0][0]:spans[0][1]]) == normalise("Café Dental")


def test_a_match_inside_a_single_expanded_character_is_rejected():
    # NFKC turns the ligature into "fi"; an alias of "f" does not mean the
    # answer said "f", and a span over the ligature would point the evidence at
    # something that was never written.
    assert find_alias_spans("ﬁce", ["f"]) == []
    assert find_alias_spans("ﬁce", ["fi"]) == [(0, 1, "fi")]
