"""The sentence that named the brand, cut from the raw answer by offset.

Offsets come from mention.extract_entities and address the RAW answer text, so
the sentence is cut from that same text — never from a normalised copy, whose
positions no longer line up (see mention.py).
"""

import re

# ---------------------------------------------------------------- constants
# A sentence ends at CJK full stops, at a line break, or at Latin . ! ? followed
# by whitespace and something that starts a sentence. The last condition keeps
# "Demo Bean Co. and Sample Roasters" in one piece.
SENTENCE_BREAK = re.compile(r"[。！？]+|\n+|(?<=[.!?])\s+(?=[A-Z0-9\"“'‘(\[*-])")
# A full stop after these is an abbreviation, not the end of a sentence, so
# "Try Dr. Example today" stays whole.
ABBREVIATIONS = ("dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "inc", "ltd", "co",
                 "vs", "etc", "e.g", "i.e")
EXCERPT_CHARS = 200
ELLIPSIS = "…"


def _ends_with_abbreviation(text, stop):
    head = text[:stop].rstrip()
    if not head.endswith("."):
        return False
    word = re.search(r"([A-Za-z.]+)\.$", head)
    return bool(word) and word.group(1).casefold() in ABBREVIATIONS


def _breaks(text):
    for match in SENTENCE_BREAK.finditer(text):
        if text[match.start() - 1:match.start()] == "." and _ends_with_abbreviation(text, match.start()):
            continue
        yield match


def _sentence_bounds(text, position):
    start, end = 0, len(text)
    for match in _breaks(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() >= position:
            end = match.end() if text[match.start()] in "。！？" else match.start()
            break
    return start, end


def brand_sentence(text, spans):
    """(sentence, highlights) for the first brand mention, or (None, []).

    spans: raw (start, end) offsets of brand mentions. highlights are the spans
    that fall inside the sentence, re-based to it, for the report to mark.
    """
    if not spans:
        return None, []
    first = min(start for start, _ in spans)
    start, end = _sentence_bounds(text, first)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    highlights = [(s - start, e - start) for s, e in sorted(spans) if s >= start and e <= end]
    return text[start:end], highlights


def answer_excerpt(text, limit=EXCERPT_CHARS):
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + ELLIPSIS
