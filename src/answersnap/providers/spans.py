"""Shared helpers for placing cited spans and reporting token usage.

Platforms disagree on offset units (UTF-8 bytes, characters in one text chunk,
or no offsets at all). Everything is converted to code points into the full
answer_text here, and anything that does not land inside the text is dropped
and counted — a parser that raises on a bad offset would mark a whole platform
unverified over one malformed span.
"""

from bisect import bisect_left

from answersnap.providers.base import AnswerSpan

# ---------------------------------------------------------------- constants
USAGE_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def utf8_offsets(text):
    """Byte offset at which each code point starts, plus one past the end."""
    offsets, position = [], 0
    for char in text:
        offsets.append(position)
        position += len(char.encode("utf-8"))
    offsets.append(position)
    return offsets


def code_point_at(offsets, byte_index):
    """Code-point index for a byte offset, or None if it splits a character."""
    if not isinstance(byte_index, int) or isinstance(byte_index, bool):
        return None
    index = bisect_left(offsets, byte_index)
    if index >= len(offsets) or offsets[index] != byte_index:
        return None
    return index


def _is_index(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _span_fits(start, end, indexes, text_len, cited_flags):
    if not (_is_index(start) and _is_index(end) and 0 <= start < end <= text_len):
        return False
    return bool(indexes) and all(
        _is_index(i) and 0 <= i < len(cited_flags) and cited_flags[i] for i in indexes)


def _as_indexes(raw):
    """The indexes as a list, or None if it is not a list at all."""
    return list(raw) if isinstance(raw, (list, tuple)) else None


def valid_spans(candidates, text_len, cited_flags):
    """Keep (start, end, indexes) candidates that land in the text; count the rest.

    cited_flags[i] is whether citation i was cited: a span must never point at
    a source that was only retrieved.
    """
    spans, dropped = [], 0
    for start, end, raw_indexes in candidates:
        indexes = _as_indexes(raw_indexes)
        # Every index is checked before set/sorted: one None or list among them
        # would otherwise raise and lose the whole answer, not just this span.
        if indexes is not None and _span_fits(start, end, indexes, text_len, cited_flags):
            spans.append(AnswerSpan(start=start, end=end,
                                    citation_indexes=tuple(sorted(set(indexes)))))
        else:
            dropped += 1
    spans.sort(key=lambda span: (span.start, span.end))
    return spans, dropped


def _count(value):
    return value if _is_index(value) and value >= 0 else None


def usage_summary(input_tokens, output_tokens, reasoning_tokens, total_tokens,
                  output_includes_reasoning, raw):
    """One key set across platforms; the platform's own block kept under raw."""
    counts = dict(zip(USAGE_KEYS, (_count(input_tokens), _count(output_tokens),
                                   _count(reasoning_tokens), _count(total_tokens))))
    if all(value is None for value in counts.values()):
        return None
    return {**counts, "output_includes_reasoning": output_includes_reasoning,
            "raw": raw}
