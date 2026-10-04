"""Does the quoted sentence actually appear on the page it was attributed to?

`cited_text` changes what this measurement is. Comparing a whole answer against
a fetched page was hopeless — models rewrite, summarise and merge sources, so a
mismatch meant nothing. But `cited_text` is the model's own claim that this
specific sentence came from this specific page, which makes the question nearly
decidable.

It still cannot raise an alert alone: pages get rewritten, render through JS,
sit behind paywalls, and models do invent citations. The point of exporting raw
rows is to find out whether the benign distribution has a clean low end at all —
if honest data already scores badly, the feature has no future; if honest data
clusters high with a few outliers, the threshold goes at the outliers.

Rows are exported one per (answer, citation), never aggregated: the sample is
small enough that the judgment layer can aggregate it however it likes, and raw
rows thrown away cannot be recovered without paying for the sampling again.
"""

import re

# Bump this whenever tokenisation or comparison changes, and never rewrite old
# rows in place: scores from two different methods mixed into one distribution
# are worse than no distribution at all.
METHOD_VERSION = "gap-1"
# A row with a known-good answer, written alongside the real ones. If the
# positive control scores badly, what is broken is the measuring tool, not the
# feature being measured — see the note below.
CONTROL_PAGE_ZH = "本院提供植牙與矯正服務，術後保固五年，採用瑞典植體。"
CONTROL_QUOTE_ZH = "術後保固五年"
CONTROL_PAGE_EN = "We offer dental implants with a five year warranty."
CONTROL_QUOTE_EN = "five year warranty"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
# Scripts written without spaces between words. A word-token regex swallows an unbroken run
# of any of them as a single token, so a quote could never match the sentence it
# came from. Kana, Hangul and the CJK extensions matter here as much as the main
# block — Taiwanese given names routinely live in Extension A/B.
SPACELESS_RANGES = (
    (0x3040, 0x30FF),    # Hiragana, Katakana
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xAC00, 0xD7AF),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK Extensions B onward
)


def _is_spaceless(char):
    code = ord(char)
    return any(low <= code <= high for low, high in SPACELESS_RANGES)


def has_spaceless_script(text):
    return any(_is_spaceless(c) for c in text or "")


def _tokens(text, by_character):
    if not text:
        return []
    if by_character:
        return [c.casefold() for c in text if c.isalnum()]
    return [w.casefold() for w in TOKEN_PATTERN.findall(text)]


def _normalise_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def best_window_score(needle, haystack):
    """Highest overlap between the quote and any same-length window of the page.

    0 when the quote's tokens appear nowhere, 1 when some window contains all of
    them in the same proportions.

    Both sides are split the same way, decided once from the pair. Choosing per
    side means a Latin brand name quoted from a Chinese page is tokenised by
    word on one side and by character on the other — they can then never match,
    and a quote sitting verbatim on the page scores exactly 0.0, which reads as
    a fabricated citation.
    """
    by_character = has_spaceless_script(needle) or has_spaceless_script(haystack)
    needle_tokens = _tokens(needle, by_character)
    haystack_tokens = _tokens(haystack, by_character)
    if not needle_tokens or not haystack_tokens:
        return 0.0

    window = len(needle_tokens)
    if window > len(haystack_tokens):
        window = len(haystack_tokens)

    needed = {}
    for token in needle_tokens:
        needed[token] = needed.get(token, 0) + 1

    best = 0
    current = {}
    for index, token in enumerate(haystack_tokens):
        current[token] = current.get(token, 0) + 1
        if index >= window:
            leaving = haystack_tokens[index - window]
            current[leaving] -= 1
            if current[leaving] == 0:
                del current[leaving]
        overlap = sum(min(count, needed.get(token, 0))
                      for token, count in current.items())
        best = max(best, overlap)
    return best / len(needle_tokens)


def compare(cited_text, fetched_text):
    return {
        "exact_match": bool(cited_text and fetched_text
                            and cited_text in fetched_text),
        "normalized_match": bool(
            cited_text and fetched_text
            and _normalise_whitespace(cited_text) in _normalise_whitespace(fetched_text)),
        "best_window_score": round(best_window_score(cited_text, fetched_text), 4),
    }


def control_rows():
    r"""Rows whose score we already know, so a broken tool is visible at a glance.

    This exists because of a failure that actually happened: `\w+` treats an
    unbroken run of Chinese as one token, so a quote could never match the
    sentence it came from and every score came out 0. The output had nothing
    suspicious about it — no error, valid format, and a distribution that read
    as a perfectly reasonable negative result. We would have run the correct
    process ("this feature has no discriminative power, drop it") to the wrong
    conclusion.

    That the blind spot landed on Chinese is not a coincidence either: the
    defaults were written for English and the data is Chinese, so the tool's
    weak spot sat exactly on the launch market.

    The rule this encodes: before believing a negative measurement, check that
    the instrument works on this kind of data.
    """
    rows = []
    for label, quote, page in (("zh", CONTROL_QUOTE_ZH, CONTROL_PAGE_ZH),
                               ("en", CONTROL_QUOTE_EN, CONTROL_PAGE_EN)):
        row = {"answer_id": None, "citation_id": None, "platform": None,
               "domain": None, "corpus": None, "fetch_status": "ok",
               "comparable": True, "is_control": True, "control_label": label,
               "cited_text": quote, "cited_text_len": len(quote),
               "fetched_text_len": len(page),
               "observed_at": None, "fetched_at": None,
               "fetch_delay_seconds": None, "method_version": METHOD_VERSION,
               "not_comparable_reason": None}
        row.update(compare(quote, page))
        rows.append(row)
    return rows


def gap_rows(answers, citations_by_answer):
    """One row per (answer, citation). Unfetched sources are listed, not dropped.

    Paywalled and JS-rendered pages produce large, entirely benign gaps; leaving
    them out would hide how much of the distribution they account for.
    """
    rows = []
    for answer in answers:
        for citation in citations_by_answer.get(answer.id, []):
            if not citation.is_cited:
                continue
            # cited_text is the claim being checked. Without it there is
            # nothing to verify, and scoring 0.0 would be indistinguishable
            # from the model having invented the citation.
            comparable = bool(citation.fetch_status == "ok"
                              and citation.fetched_text and citation.cited_text)
            if not comparable:
                if not citation.cited_text:
                    reason = "no_cited_text"
                elif citation.fetch_status != "ok":
                    reason = f"fetch_{citation.fetch_status}"
                else:
                    reason = "no_fetched_text"
            else:
                reason = None
            row = {
                "answer_id": str(answer.id),
                "citation_id": str(citation.id),
                "platform": answer.platform,
                "domain": citation.domain,
                "corpus": citation.corpus,
                "fetch_status": citation.fetch_status,
                "comparable": comparable,
                "cited_text": citation.cited_text,
                "cited_text_len": len(citation.cited_text or ""),
                "fetched_text_len": len(citation.fetched_text or ""),
                "observed_at": answer.observed_at.isoformat() if answer.observed_at else None,
                "fetched_at": citation.fetched_at.isoformat() if citation.fetched_at else None,
                # The main benign cause of a gap: the longer the delay, the more
                # chance the page simply changed. Without it, "we fetched late"
                # reads as "the content was swapped".
                "fetch_delay_seconds": (
                    (citation.fetched_at - answer.observed_at).total_seconds()
                    if citation.fetched_at and answer.observed_at else None),
                "method_version": METHOD_VERSION,
                "is_control": False,
                "not_comparable_reason": reason,
            }
            row.update(compare(citation.cited_text, citation.fetched_text)
                       if comparable
                       else {"exact_match": None, "normalized_match": None,
                             "best_window_score": None})
            rows.append(row)
    return rows
