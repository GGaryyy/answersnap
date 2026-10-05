import pytest

from answersnap.providers.base import AnswerSpan
from answersnap.providers.spans import code_point_at, usage_summary, utf8_offsets, valid_spans
from fake_answers import ANSWER_WITH_EMOJI

TEXT = "你好🦷 Trellis"


def test_utf8_offsets_step_by_encoded_width():
    # 3 bytes per CJK character, 4 for the emoji, 1 for ASCII.
    assert utf8_offsets(TEXT)[:5] == [0, 3, 6, 10, 11]
    assert utf8_offsets(TEXT)[-1] == len(TEXT.encode("utf-8"))


def test_code_point_at_maps_exact_boundaries():
    offsets = utf8_offsets(TEXT)
    assert code_point_at(offsets, 0) == 0
    assert code_point_at(offsets, 10) == 3
    assert code_point_at(offsets, len(TEXT.encode("utf-8"))) == len(TEXT)


@pytest.mark.parametrize("byte_index", [1, 7, 999, -1, None, "3", True])
def test_code_point_at_refuses_anything_off_a_boundary(byte_index):
    assert code_point_at(utf8_offsets(TEXT), byte_index) is None


def test_byte_offsets_round_trip_on_mixed_text():
    target = "Trellis"
    start = ANSWER_WITH_EMOJI.encode().index(target.encode())
    offsets = utf8_offsets(ANSWER_WITH_EMOJI)
    s = code_point_at(offsets, start)
    e = code_point_at(offsets, start + len(target))
    assert ANSWER_WITH_EMOJI[s:e] == target


def test_valid_spans_keeps_good_ones_sorted_and_deduplicates_indexes():
    spans, dropped = valid_spans([(5, 9, [1, 0, 1]), (0, 3, [0])], 10, [True, True])
    assert spans == [AnswerSpan(0, 3, (0,)), AnswerSpan(5, 9, (0, 1))]
    assert dropped == 0


@pytest.mark.parametrize("candidate", [
    (0, 11, [0]),        # past the end
    (-1, 3, [0]),        # negative
    (4, 4, [0]),         # empty
    (5, 2, [0]),         # reversed
    ("0", 3, [0]),       # not an int
    (0, 3, []),          # no source
    (0, 3, [None]),      # source that could not be mapped
    (0, 3, [5]),         # index out of range
    (0, 3, [1]),         # points at a retrieved-only source
])
def test_valid_spans_drops_and_counts_every_invalid_shape(candidate):
    spans, dropped = valid_spans([candidate, (0, 2, [0])], 10, [True, False])
    assert spans == [AnswerSpan(0, 2, (0,))]
    assert dropped == 1


def test_usage_summary_keeps_ints_and_nulls_the_rest():
    usage = usage_summary(10, "20", None, 30.5, True, {"x": 1})
    assert usage == {"input_tokens": 10, "output_tokens": None, "reasoning_tokens": None,
                     "total_tokens": None, "output_includes_reasoning": True, "raw": {"x": 1}}


def test_usage_summary_with_no_counts_is_none():
    assert usage_summary(None, None, None, None, True, {}) is None


@pytest.mark.parametrize("indexes", [[0, None], [0, "1"], [[0]], [0, True], "0", None, {0: 1}])
def test_one_bad_index_drops_its_span_and_never_raises(indexes):
    spans, dropped = valid_spans([(0, 3, indexes), (4, 6, [0])], 10, [True])
    assert spans == [AnswerSpan(4, 6, (0,))]
    assert dropped == 1


def test_usage_summary_rejects_negative_counts():
    assert usage_summary(-5, 10, None, None, True, {})["input_tokens"] is None
