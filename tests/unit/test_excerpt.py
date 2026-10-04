from fake_answers import ANSWER_WITH_EMOJI

from answersnap.metrics.excerpt import ELLIPSIS, answer_excerpt, brand_sentence
from answersnap.metrics.mention import extract_entities


def _spans(text, name, aliases=()):
    found = extract_entities(text, [(name, "brand", list(aliases))])
    return [(s, e) for s, e, _ in found[0]["spans"]] if found else []


def test_cjk_and_emoji_sentence_is_cut_on_raw_offsets():
    spans = _spans(ANSWER_WITH_EMOJI, "Trellis")
    sentence, highlights = brand_sentence(ANSWER_WITH_EMOJI, spans)
    assert sentence == "你好🦷 Trellis 是一家診所。"
    start, end = highlights[0]
    assert sentence[start:end] == "Trellis"


def test_abbreviation_does_not_end_a_sentence():
    text = "Try Demo Bean Co. and Trellis for beans. Something else."
    sentence, _ = brand_sentence(text, _spans(text, "Trellis"))
    assert sentence == "Try Demo Bean Co. and Trellis for beans."


def test_sentence_after_an_earlier_one_starts_at_its_own_start():
    text = "Many options exist. Trellis is popular! Others are too."
    sentence, highlights = brand_sentence(text, _spans(text, "Trellis"))
    assert sentence == "Trellis is popular!"
    assert highlights == [(0, 7)]


def test_line_break_ends_a_sentence():
    text = "- Kanbanly\n- Trellis, near the station\n- Other"
    sentence, _ = brand_sentence(text, _spans(text, "Trellis"))
    assert sentence == "- Trellis, near the station"


def test_every_mention_inside_the_sentence_is_highlighted():
    text = "Trellis Dental, also called Trellis, is fine."
    sentence, highlights = brand_sentence(text, _spans(text, "Trellis", ["Trellis Dental"]))
    assert [sentence[s:e] for s, e in highlights] == ["Trellis Dental", "Trellis"]


def test_no_mention_means_no_sentence():
    assert brand_sentence("Nothing here.", []) == (None, [])


def test_excerpt_truncates_with_an_ellipsis_only_when_needed():
    assert answer_excerpt("  short  ") == "short"
    long_text = "word " * 100
    excerpt = answer_excerpt(long_text, limit=20)
    assert excerpt.endswith(ELLIPSIS) and len(excerpt) <= 21


def test_title_abbreviations_do_not_end_a_sentence():
    text = "For implants, try Dr. Trellis today. Others exist."
    sentence, _ = brand_sentence(text, _spans(text, "Trellis"))
    assert sentence == "For implants, try Dr. Trellis today."
    text = "Pick a roaster, e.g. Trellis. Or not."
    assert brand_sentence(text, _spans(text, "Trellis"))[0] == "Pick a roaster, e.g. Trellis."


def test_a_markdown_bullet_after_a_full_stop_starts_a_sentence():
    text = "Here are options. - Trellis is one."
    assert brand_sentence(text, _spans(text, "Trellis"))[0] == "- Trellis is one."
