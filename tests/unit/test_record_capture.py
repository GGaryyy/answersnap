"""What each adapter keeps beyond the answer: queries, spans, usage, raw payload.

Payloads are hand-written in each platform's shape with CJK + emoji text, where
byte, character and code-point offsets all disagree.
"""

import pytest

from answersnap.providers.anthropic_provider import AnthropicProvider
from answersnap.providers.base import AnswerSpan, SearchQuery
from answersnap.providers.google_provider import GoogleProvider
from answersnap.providers.openai_provider import OpenAIProvider
from fake_answers import STAMP

FIRST, SECOND = "你好🦷 診所甲很好。", "診所乙也不錯😀。"


def _parse(provider, payload):
    return provider.parse(payload, STAMP, STAMP)


# ---------------------------------------------------------------- anthropic
ANTHROPIC = {
    "model": "claude-sonnet-5",
    "content": [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "診所 推薦"}},
        {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "第二次"}},
        {"type": "web_search_tool_result", "tool_use_id": "s1",
         "content": [{"url": "https://a.example/1"}, {"url": "https://b.example/2"},
                     {"url": "https://r.example/3"}]},
        {"type": "text", "text": "前言。", "citations": None},
        {"type": "text", "text": FIRST,
         "citations": [{"url": "https://a.example/1"}, {"url": "https://b.example/2"}]},
        {"type": "text", "text": "中間"},
        {"type": "text", "text": SECOND, "citations": [{"url": "https://b.example/2"}]},
    ],
    "usage": {"input_tokens": 100, "output_tokens": 40,
              "output_tokens_details": {"thinking_tokens": 7},
              "server_tool_use": {"web_search_requests": 2}},
}


def test_anthropic_places_cited_blocks_by_cumulative_length():
    answer = _parse(AnthropicProvider(client=object()), ANTHROPIC)
    first = answer.answer_text.index(FIRST)
    second = answer.answer_text.index(SECOND)
    assert answer.spans == [AnswerSpan(first, first + len(FIRST), (0, 1)),
                            AnswerSpan(second, second + len(SECOND), (1,))]
    for span in answer.spans:
        assert answer.answer_text[span.start:span.end] in (FIRST, SECOND)
    assert answer.spans_dropped == 0


def test_anthropic_keeps_queries_with_result_counts_and_usage():
    answer = _parse(AnthropicProvider(client=object()), ANTHROPIC)
    assert answer.searches == [SearchQuery("診所 推薦", 3), SearchQuery("第二次", None)]
    assert answer.search_count == 2
    assert answer.usage["input_tokens"] == 100
    assert answer.usage["reasoning_tokens"] == 7
    assert answer.usage["output_includes_reasoning"] is True
    assert answer.raw_payload is ANTHROPIC


def test_anthropic_without_any_citation_channel_has_no_spans():
    payload = {"content": [{"type": "text", "text": "答案", "citations": None}]}
    answer = _parse(AnthropicProvider(client=object()), payload)
    assert answer.spans is None and answer.usage is None


# ---------------------------------------------------------------- openai
def _openai(second_annotations):
    return {
        "model": "gpt-5.1",
        "output": [
            {"type": "web_search_call", "action": {"type": "search", "queries": ["甲", "乙"]}},
            {"type": "web_search_call", "action": {"type": "open_page", "url": "https://x"}},
            {"type": "web_search_call", "action": {"type": "search", "query": "丙"}},
            {"type": "message", "content": [
                {"type": "output_text", "text": FIRST, "annotations": [
                    {"type": "url_citation", "url": "https://a.example/1",
                     "start_index": 0, "end_index": len(FIRST)},
                    {"type": "url_citation", "url": "https://b.example/2",
                     "start_index": 0, "end_index": len(FIRST)}]},
                {"type": "refusal", "refusal": "skipped, not output_text"},
                {"type": "output_text", "text": SECOND, "annotations": second_annotations},
            ]},
        ],
        "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70,
                  "output_tokens_details": {"reasoning_tokens": 3}},
        "tool_usage": {"web_search": {"num_requests": 3}},
    }


def test_openai_offsets_in_a_later_chunk_add_the_earlier_chunks_length():
    second = [{"type": "url_citation", "url": "https://b.example/2",
               "start_index": 0, "end_index": 3}]
    answer = _parse(OpenAIProvider(client=object()), _openai(second))
    assert answer.spans == [AnswerSpan(0, len(FIRST), (0, 1)),
                            AnswerSpan(len(FIRST), len(FIRST) + 3, (1,))]
    assert answer.answer_text[len(FIRST):len(FIRST) + 3] == SECOND[:3]


def test_openai_out_of_range_offsets_are_dropped_not_raised():
    second = [{"type": "url_citation", "url": "https://b.example/2",
               "start_index": 0, "end_index": 999}]
    answer = _parse(OpenAIProvider(client=object()), _openai(second))
    assert len(answer.spans) == 1 and answer.spans_dropped == 1
    # The citation itself still counts; only its placement is unknown.
    assert len(answer.citations) == 2


def test_openai_keeps_search_queries_count_and_usage():
    answer = _parse(OpenAIProvider(client=object()), _openai([]))
    assert [s.query for s in answer.searches] == ["甲", "乙", "丙"]
    assert answer.search_count == 3
    assert answer.usage["total_tokens"] == 70 and answer.usage["reasoning_tokens"] == 3


# ---------------------------------------------------------------- google
def _byte(text, part):
    return len(text[:text.index(part)].encode("utf-8"))


def _google(supports):
    text = FIRST + SECOND
    return {
        "modelVersion": "gemini-3.7-flash",
        "candidates": [{"content": {"parts": [{"text": FIRST}, {"text": SECOND}]},
                        "groundingMetadata": {
                            "webSearchQueries": ["診所"],
                            "groundingChunks": [
                                {"web": {"uri": "https://g.example/a", "title": "a.example"}},
                                {"web": {"uri": "https://g.example/r", "title": "r.example"}},
                                {"web": {"uri": "https://g.example/a", "title": "a.example"}}],
                            "groundingSupports": supports(text)}}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 5,
                          "thoughtsTokenCount": 4, "totalTokenCount": 18},
    }


def _support(text, part, chunks, quoted=None):
    start = _byte(text, part)
    return {"segment": {"startIndex": start, "endIndex": start + len(part.encode()),
                        "text": part if quoted is None else quoted},
            "groundingChunkIndices": chunks}


def test_google_byte_offsets_become_code_points_that_reproduce_the_segment():
    payload = _google(lambda t: [_support(t, SECOND, [2])])
    answer = _parse(GoogleProvider(api_key="x"), payload)
    (span,) = answer.spans
    assert answer.answer_text[span.start:span.end] == SECOND
    # chunk 2 repeats chunk 0's URL, so it maps to the same cited source.
    assert span.citation_indexes == (0,)


def test_google_missing_start_index_means_zero():
    def supports(text):
        support = _support(text, FIRST, [0])
        del support["segment"]["startIndex"]
        return [support]
    answer = _parse(GoogleProvider(api_key="x"), _google(supports))
    assert answer.spans == [AnswerSpan(0, len(FIRST), (0,))]


def test_google_segment_text_mismatch_is_dropped_rather_trusted():
    payload = _google(lambda t: [_support(t, SECOND, [0], quoted="別的句子")])
    answer = _parse(GoogleProvider(api_key="x"), payload)
    assert answer.spans == [] and answer.spans_dropped == 1


def test_google_keeps_queries_and_usage_but_reports_no_search_count():
    answer = _parse(GoogleProvider(api_key="x"), _google(lambda t: []))
    assert answer.searches == [SearchQuery("診所")]
    assert answer.search_count is None
    assert answer.usage["output_tokens"] == 5 and answer.usage["reasoning_tokens"] == 4
    assert answer.usage["output_includes_reasoning"] is False


# ---------------------------------------------------------------- shared
@pytest.mark.parametrize("provider", [AnthropicProvider(client=object()),
                                      OpenAIProvider(client=object()),
                                      GoogleProvider(api_key="x")])
def test_empty_payload_records_nothing_and_raises_nothing(provider):
    answer = _parse(provider, {})
    assert answer.spans in (None, [])
    assert answer.usage is None
    assert answer.search_count is None


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAIProvider, GoogleProvider])
def test_recordings_still_verify_and_spans_never_point_outside_the_text(cls):
    cls.reset_verification_cache()
    assert cls.response_shape_verified()
    payload = cls.load_recorded_response()["payload"]
    answer = cls.__new__(cls).parse(payload, STAMP, STAMP)
    assert answer.usage is not None and answer.searches
    # Redaction shortens the text, so spans past the end must be dropped and
    # counted rather than raise; Claude's block-length spans survive intact.
    assert answer.spans_dropped == {"anthropic": 0, "openai": 12, "google": 23}[cls.platform]
    for span in answer.spans or []:
        assert 0 <= span.start < span.end <= len(answer.answer_text)


# ---------------------------------------------------------------- malformed shapes
def test_anthropic_url_less_citation_beside_a_real_one_keeps_the_answer():
    payload = {"content": [{"type": "text", "text": FIRST,
                            "citations": [{"url": "https://a.example/1"}, {"title": "no url"}]}],
               "usage": {"input_tokens": 1, "output_tokens": 1, "output_tokens_details": "odd",
                         "server_tool_use": ["odd"]}}
    answer = _parse(AnthropicProvider(client=object()), payload)
    assert answer.answer_text == FIRST and answer.spans == [] and answer.spans_dropped == 1
    assert answer.usage["input_tokens"] == 1 and answer.search_count is None


@pytest.mark.parametrize("bad", [[0], "3", None, 1.5])
def test_openai_malformed_offsets_are_each_counted(bad):
    annotations = [{"type": "url_citation", "url": "https://b.example/2",
                    "start_index": bad, "end_index": 3},
                   {"type": "url_citation", "url": "https://b.example/2",
                    "start_index": 0, "end_index": bad}]
    payload = _openai(annotations)
    payload["usage"]["output_tokens_details"] = ["odd"]
    payload["tool_usage"] = "odd"
    answer = _parse(OpenAIProvider(client=object()), payload)
    assert answer.spans_dropped == 2
    assert answer.search_count is None


@pytest.mark.parametrize("chunk_indexes", [[0, 7], [0, None], [[0]], "0"])
def test_google_bad_chunk_indexes_drop_only_that_support(chunk_indexes):
    def supports(text):
        return [_support(text, FIRST, chunk_indexes), _support(text, SECOND, [0])]
    payload = _google(supports)
    payload["candidates"][0]["groundingMetadata"]["groundingChunks"].append({"web": "odd"})
    answer = _parse(GoogleProvider(api_key="x"), payload)
    assert [answer.answer_text[s.start:s.end] for s in answer.spans] == [SECOND]
    assert answer.spans_dropped == 1
