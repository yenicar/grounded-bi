"""Deterministic checks that an answer's cited chunks actually support it.

Every answer lists the chunks it was given as citations, so "has citations" says nothing about
whether the answer is supported. These functions split an answer into claims and check each
claim against the text of each cited chunk: most of the claim's content words, and every number
in it, must appear in one chunk. No model call is made, so the check is cheap, repeatable and
cannot be talked into agreeing. It is lexical, so it can miss a correct paraphrase (a false
"unsupported") and cannot catch a claim that reuses a chunk's words with the meaning reversed;
the LLM judge in evaluation.py remains a separate column for that.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# A claim counts as supported by a chunk when this share of its content words appears in the
# chunk and every number in the claim appears there too.
SUPPORT_THRESHOLD = 0.6
# Claims with fewer content words and numbers are skipped (headings, "Yes.", list labels).
MIN_CLAIM_WORDS = 3

REFUSAL_PATTERNS = [
    r"could not find enough grounded evidence",
    r"\binsufficient (evidence|information|context)\b",
    r"\b(evidence|context|information) (is|was) insufficient\b",
    r"\b(do|does|did) not (contain|cover|address|specify|say|mention|include|provide|state)\b",
    r"\bnot (covered|addressed|specified|mentioned|stated|included) in\b",
    r"\bno (information|evidence|mention|details?) (about|on|regarding|of)\b",
    r"\bnot enough (information|evidence|context)\b",
    r"\bunable to (find|answer|determine)\b",
    r"\bcannot (find|answer|determine)\b",
]

_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each either few for
    from further had has have having he her here hers him his how i if in into is it its itself
    just may me might more most must my no nor not of off on once only or other our ours out over
    own per same shall she should so some such than that the their theirs them then there these
    they this those through to too under until up upon very via was we were what when where which
    while who whom why will with within would you your yours
    """.split()
)

_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "ninety": "90", "hundred": "100",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_CONFIDENCE_RE = re.compile(r"^\s*\**\s*confidence\s*:?\s*\**\s*(high|medium|low)\b.*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _stem(word: str) -> str:
    """Crude suffix stripping so "employees"/"employee" and "required"/"requires" compare equal."""
    if word.isdigit() or len(word) <= 3:
        return word
    if word.endswith(("ies", "ied")) and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("ing", "ed", "ly")) and len(word) - (3 if word.endswith("ing") else 2) >= 3:
        word = word[:-3] if word.endswith("ing") else word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    lowered = (text or "").lower().replace("’", "'")
    out: list[str] = []
    for raw in _WORD_RE.findall(lowered):
        raw = raw.replace(",", "")
        out.append(_NUMBER_WORDS.get(raw, raw))
    return out


def content_words(text: str) -> set[str]:
    return {_stem(token) for token in _tokens(text) if token not in _STOPWORDS and not _is_number(token)}


def numbers(text: str) -> set[str]:
    return {_normalise_number(token) for token in _tokens(text) if _is_number(token)}


def _is_number(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def _normalise_number(token: str) -> str:
    return token.rstrip("0").rstrip(".") if "." in token else token.lstrip("0") or "0"


def is_refusal(answer: str) -> bool:
    body = strip_confidence_line(answer).lower()
    return any(re.search(pattern, body) for pattern in REFUSAL_PATTERNS)


def strip_confidence_line(answer: str) -> str:
    lines = [line for line in (answer or "").splitlines() if not _CONFIDENCE_RE.match(line)]
    return "\n".join(lines).strip()


def split_claims(answer: str) -> list[str]:
    """Bullets and sentences of the answer, without the confidence line, headings or fragments."""
    claims: list[str] = []
    for line in strip_confidence_line(answer).splitlines():
        line = _BULLET_RE.sub("", line).replace("**", "").replace("__", "").strip()
        line = line.lstrip("#").strip()
        if not line:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            # numbers count toward the minimum: "They must have worked 1,250 hours." is a claim
            if len(content_words(sentence)) + len(numbers(sentence)) >= MIN_CLAIM_WORDS:
                claims.append(sentence)
    return claims


def claim_support(claim: str, chunk_text: str) -> float:
    """Share of the claim's content words found in the chunk, or 0 if any of its numbers is missing."""
    words = content_words(claim)
    if not words:
        return 0.0
    if not numbers(claim) <= numbers(chunk_text):
        return 0.0
    return len(words & content_words(chunk_text)) / len(words)


def assess_answer(
    answer: str,
    cited_chunks: Iterable[dict[str, Any]],
    threshold: float = SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """Claim-by-claim support of an answer by its cited chunks.

    cited_chunks: dicts with "citation" and "text" (the chunk text the model was given).
    Returns claims with their best supporting citation, claim_support_rate (supported claims /
    claims), citation_precision (cited chunks that support at least one claim / cited chunks),
    and whether the answer is a refusal.
    """
    chunks = [
        {"citation": str(chunk.get("citation", "")), "text": str(chunk.get("text", "") or "")}
        for chunk in cited_chunks
    ]
    refusal = is_refusal(answer)
    claims_out: list[dict[str, Any]] = []
    supporting: set[int] = set()

    for claim in split_claims(answer):
        best_index, best_score = -1, 0.0
        for index, chunk in enumerate(chunks):
            score = claim_support(claim, chunk["text"])
            if score > best_score:
                best_index, best_score = index, score
        supported = best_index >= 0 and best_score >= threshold
        if supported:
            supporting.add(best_index)
        claims_out.append(
            {
                "claim": claim,
                "supported": supported,
                "support": round(best_score, 2),
                "best_citation": chunks[best_index]["citation"] if best_index >= 0 else None,
            }
        )

    supported_count = sum(1 for claim in claims_out if claim["supported"])
    return {
        "refusal": refusal,
        "claims": claims_out,
        "claim_count": len(claims_out),
        "supported_claims": supported_count,
        "claim_support_rate": round(supported_count / len(claims_out), 3) if claims_out else None,
        "cited_chunks": len(chunks),
        "supporting_citations": sorted(chunks[index]["citation"] for index in supporting),
        "citation_precision": round(len(supporting) / len(chunks), 3) if chunks else None,
    }


def expected_source_hit(
    cited_chunks: Iterable[dict[str, Any]],
    expected_doc: str | None,
    expected_pages: Iterable[int] | None = None,
) -> bool | None:
    """Whether any cited chunk is from the expected document (and overlaps an expected page).

    None when the prompt names no expected document.
    """
    if not expected_doc:
        return None
    want_doc = expected_doc.lower()
    pages = {int(page) for page in (expected_pages or [])}
    for chunk in cited_chunks:
        doc = str(chunk.get("doc_name", "")).lower()
        if want_doc not in doc and doc not in want_doc:
            continue
        if not pages:
            return True
        start = int(chunk.get("page_start") or 0)
        end = int(chunk.get("page_end") or start)
        if any(start <= page <= end for page in pages):
            return True
    return False


def expected_facts_found(answer: str, expected_facts: Iterable[str] | None) -> float | None:
    """Share of expected facts whose content words and numbers all appear in the answer."""
    facts = [fact for fact in (expected_facts or []) if fact and fact.strip()]
    if not facts:
        return None
    answer_words, answer_numbers = content_words(answer), numbers(answer)
    found = sum(
        1 for fact in facts if content_words(fact) <= answer_words and numbers(fact) <= answer_numbers
    )
    return round(found / len(facts), 3)
