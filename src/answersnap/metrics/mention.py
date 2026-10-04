"""Who the answer mentioned, where, and in what order.

Deterministic alias matching only — no language model anywhere on this path.
This is the step that decides "was the brand mentioned", and a component that
hallucinates here would mean using a hallucinating thing to report on
hallucination.

Offsets point into the RAW `answer_text`, counted in Unicode code points, which
is why matching cannot simply normalise the text first: NFKC changes string
length (full-width forms collapse, ligatures expand), so an offset found in a
normalised copy does not address the same characters in the original. Every
alert's evidence link depends on this being right, so the normalised form is
built alongside a map back to raw positions.
"""

import re
import unicodedata

# Spans that look like a named entity but matched no known alias. They become
# entity_role='unknown' mentions, which serve two purposes at once: the alias
# table fills itself in, and new competitors surface on their own.
ORG_SUFFIXES = ("牙醫診所", "牙醫", "診所", "醫院", "口腔外科", "集團")
UNKNOWN_ENTITY_PATTERNS = (
    # CJK organisation names by suffix — the category's actual shape.
    r"[一-鿿]{2,10}(?:牙醫診所|牙醫|診所|醫院|口腔外科|集團)",
    # Latin proper-noun runs of at least two capitalised words. A single
    # capitalised word is usually just a sentence opening, and a queue full of
    # "The" and "Good" trains people to reject without reading — which damages
    # the alias table this is meant to improve.
    r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,3}\b",
    # Names the answer itself put in CJK quote marks. Straight quotes are left
    # out on purpose: an apostrophe pairs across half a sentence, which would
    # drop answer prose into the metrics file.
    r"[「『]([^」』]{2,30})[」』]",
)
# CJK has no word boundaries, so a suffix match happily swallows the connective
# in front of it ("另外示範牙醫診所"). These are stripped from the front of a
# candidate; getting it wrong means the customer is asked to confirm entities
# whose names carry a stray particle.
LEADING_PARTICLES = (
    "另外", "還有", "以及", "或者", "其次", "再來", "至於", "包括", "例如",
    "推薦", "建議", "可以看看", "像是", "譬如",
    "跟", "和", "與", "或", "及", "的", "是", "有", "在", "去", "找", "看",
)
MIN_UNKNOWN_LENGTH = 2
# Latin runs this short are almost always ordinary words, not names.
MIN_LATIN_UNKNOWN_LENGTH = 3


def _clusters(text):
    """Split into base-character-plus-combining-marks groups.

    Normalising one character at a time can never compose `e` + U+0301 into
    `é`, so an answer written in decomposed form would not match an alias
    written in composed form. A missed alias looks exactly like a suppressed
    brand, which makes this a correctness problem rather than a tidiness one.
    """
    clusters, start = [], 0
    for index, char in enumerate(text):
        if index > start and unicodedata.combining(char) == 0:
            clusters.append((start, index))
            start = index
    if text:
        clusters.append((start, len(text)))
    return clusters


def normalise_with_map(text):
    """Return (normalised text, start map, end map).

    The maps carry each normalised character back to the raw span it came from,
    so a match found in the normalised form addresses the same characters in the
    original — NFKC changes length, and every evidence link depends on this.
    """
    pieces, start_map, end_map = [], [], []
    for raw_start, raw_end in _clusters(text):
        folded = unicodedata.normalize("NFKC", text[raw_start:raw_end]).casefold()
        for folded_char in folded:
            pieces.append(folded_char)
            start_map.append(raw_start)
            end_map.append(raw_end)
    return "".join(pieces), start_map, end_map


def normalise(text):
    return normalise_with_map(text)[0]


def find_alias_spans(text, aliases):
    """All non-overlapping raw-text spans matching any alias, longest first."""
    normalised, start_map, end_map = normalise_with_map(text)
    candidates = []
    for alias in aliases:
        needle = normalise(alias)
        if not needle:
            continue
        start = normalised.find(needle)
        while start != -1:
            end = start + len(needle)
            # A match that begins mid-way through one raw character — NFKC
            # expands `ﬁ` to `fi`, `Ⅻ` to `xii` — is not a match on that
            # character, and reporting a span over it would point the evidence
            # at something the answer never said.
            if start > 0 and start_map[start] == start_map[start - 1]:
                start = normalised.find(needle, start + 1)
                continue
            if end < len(normalised) and end_map[end - 1] == end_map[end]:
                start = normalised.find(needle, start + 1)
                continue
            candidates.append((start_map[start], end_map[end - 1], alias))
            start = normalised.find(needle, start + 1)

    # A longer alias wins over a shorter one it contains, so "示範牙醫診所"
    # is one mention rather than also counting as "示範牙醫".
    candidates.sort(key=lambda span: (span[0], -(span[1] - span[0])))
    chosen, last_end = [], -1
    for raw_start, raw_end, alias in candidates:
        if raw_start >= last_end:
            chosen.append((raw_start, raw_end, alias))
            last_end = raw_end
    return chosen


def extract_entities(text, entities):
    """entities: [(name, role, [aliases])] -> mention dicts, ranked by position.

    Rank is the order of first appearance in the answer, because that is what
    "how prominently did the AI place them" means to a reader.
    """
    # Collect every entity's spans first, then resolve overlaps across entities
    # as well as within one. Otherwise "範例甲牙醫" and "範例甲牙醫診所大安店"
    # both claim the same text, and one clinic counts as two mentions —
    # inflating appearance rate and doubling up the ranks.
    claims = []
    for name, role, aliases in entities:
        for start, end, alias in find_alias_spans(text, [name, *aliases]):
            claims.append((start, end, name, role, alias))
    claims.sort(key=lambda c: (c[0], -(c[1] - c[0])))

    kept, last_end = [], -1
    for start, end, name, role, alias in claims:
        if start >= last_end:
            kept.append((start, end, name, role))
            last_end = end

    found = []
    for name, role, _aliases in entities:
        spans = [(s, e, name) for s, e, n, r in kept if n == name]
        if not spans:
            continue
        found.append({"entity": name, "entity_role": role,
                      "char_start": spans[0][0], "char_end": spans[0][1],
                      "spans": spans})

    found.sort(key=lambda m: m["char_start"])
    for rank, mention in enumerate(found, start=1):
        mention["rank"] = rank
    return found


def _strip_leading_particles(surface):
    """Remove connectives the suffix match swallowed, without eating names.

    A single-character particle is only stripped when what remains still ends in
    a known organisation suffix — otherwise "有範牙醫" loses its first character
    and the customer is asked to confirm a clinic that does not exist.
    """
    changed = True
    while changed:
        changed = False
        for particle in LEADING_PARTICLES:
            if not surface.startswith(particle):
                continue
            remainder = surface[len(particle):]
            if len(remainder) < MIN_UNKNOWN_LENGTH + 1:
                continue
            if len(particle) == 1 and not remainder.endswith(ORG_SUFFIXES):
                continue
            surface = remainder
            changed = True
            break
    return surface


def _claimed_spans(mentions):
    return [(span[0], span[1]) for m in mentions for span in m["spans"]]


def find_unknown_entities(text, known_mentions):
    """Named-looking spans that matched nothing we know about."""
    claimed = _claimed_spans(known_mentions)
    seen, unknown = set(), []

    for pattern in UNKNOWN_ENTITY_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.span(1) if match.groups() else match.span()
            raw = text[start:end]
            stripped = raw.strip()
            # Track both ends: compensating only for the particle leaves the
            # span covering whitespace the entity name does not include, so the
            # evidence highlight and the entity disagree.
            start += len(raw) - len(raw.lstrip())
            end -= len(raw) - len(raw.rstrip())
            trimmed = _strip_leading_particles(stripped)
            start += len(stripped) - len(trimmed)
            surface = trimmed
            if len(surface) < MIN_UNKNOWN_LENGTH:
                continue
            if surface.isascii() and len(surface) < MIN_LATIN_UNKNOWN_LENGTH:
                continue
            if any(start < claimed_end and end > claimed_start
                   for claimed_start, claimed_end in claimed):
                continue
            key = normalise(surface)
            if key in seen:
                continue
            seen.add(key)
            unknown.append({"entity": surface, "entity_role": "unknown",
                            "char_start": start, "char_end": end, "rank": 0,
                            "spans": [(start, end, surface)]})
    return unknown
