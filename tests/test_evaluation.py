from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest

from true_rag_bi import evaluation
from true_rag_bi.config import AppConfig, ProjectPaths
from true_rag_bi.evaluation import score_result
from true_rag_bi.grounding_check import (
    assess_answer,
    claim_support,
    expected_facts_found,
    expected_source_hit,
    is_refusal,
    split_claims,
)

FMLA_CHUNK = {
    "citation": "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf p.3 chunk:a",
    "doc_name": "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf",
    "page_start": 3,
    "page_end": 3,
    "text": (
        "Eligible employees may take up to 12 workweeks of leave in a 12-month period for the birth "
        "of a child, to care for a spouse, child, or parent with a serious health condition, or for "
        "their own serious health condition. Employees must have worked for the employer for at "
        "least 12 months and at least 1,250 hours during the 12 months before the leave."
    ),
}
VACATION_CHUNK = {
    "citation": "03_UW_System/UWSystem-1210-VacationPaidLeaveBanksCashPayouts-2026.pdf p.2 chunk:b",
    "doc_name": "03_UW_System/UWSystem-1210-VacationPaidLeaveBanksCashPayouts-2026.pdf",
    "page_start": 2,
    "page_end": 2,
    "text": "Vacation is earned each pay period and may be carried over to the next year subject to limits.",
}


def _result(answer: str, chunks: list[dict]) -> dict:
    return {
        "answer": answer,
        "citations": [chunk["citation"] for chunk in chunks],
        "retrieval_debug": {"selected": chunks},
        "metrics": {"total_seconds": 1.0, "total_tokens": 1000, "remote_calls": 1},
    }


# --- claim splitting and refusal detection -------------------------------------------------


def test_split_claims_drops_confidence_line_headings_and_fragments():
    answer = (
        "**Eligibility:**\n"
        "- Employees can take up to 12 workweeks of FMLA leave in a 12-month period.\n"
        "- They must have worked 1,250 hours. Yes.\n\n"
        "Confidence: High"
    )
    assert split_claims(answer) == [
        "Employees can take up to 12 workweeks of FMLA leave in a 12-month period.",
        "They must have worked 1,250 hours.",
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "I could not find enough grounded evidence in the indexed corpus to answer that confidently.",
        "The retrieved documents do not specify a 401(k) match.\n\nConfidence: Low",
        "The evidence is insufficient to answer this.",
    ],
)
def test_refusals_are_detected(answer):
    assert is_refusal(answer)


def test_an_answer_is_not_a_refusal():
    assert not is_refusal("Employees can take up to 12 workweeks of leave.\n\nConfidence: High")


# --- claim-to-chunk support ----------------------------------------------------------------


def test_claim_supported_by_matching_chunk():
    claim = "Eligible employees can take up to 12 workweeks of leave in a 12-month period."
    assert claim_support(claim, FMLA_CHUNK["text"]) >= 0.6


def test_claim_with_a_number_missing_from_the_chunk_is_not_supported():
    claim = "Eligible employees can take up to 16 workweeks of leave in a 12-month period."
    assert claim_support(claim, FMLA_CHUNK["text"]) == 0.0


def test_number_words_and_digits_compare_equal():
    assert claim_support("Employees may take twelve workweeks of leave.", FMLA_CHUNK["text"]) >= 0.6


def test_assess_answer_counts_supported_claims_and_supporting_citations():
    answer = (
        "- Eligible employees can take up to 12 workweeks of leave in a 12-month period.\n"
        "- Employees must have worked at least 1,250 hours in the 12 months before leave.\n"
        "- Unused leave is paid out in cash at retirement.\n\n"
        "Confidence: Medium"
    )
    report = assess_answer(answer, [FMLA_CHUNK, VACATION_CHUNK])
    assert report["claim_count"] == 3
    assert report["supported_claims"] == 2
    assert report["claim_support_rate"] == pytest.approx(0.667, abs=0.001)
    assert report["supporting_citations"] == [FMLA_CHUNK["citation"]]
    assert report["citation_precision"] == 0.5  # the vacation chunk was cited but supports nothing
    assert [claim["supported"] for claim in report["claims"]] == [True, True, False]


def test_citation_that_exists_but_does_not_support_the_answer_scores_low():
    # Version 1 gave this factual_grounding 5 because a citation existed.
    answer = "Employees receive 20 days of paid parental leave at full salary.\n\nConfidence: High"
    scores, grounding = score_result({"id": "q"}, _result(answer, [VACATION_CHUNK]))
    assert grounding["supported_claims"] == 0
    assert scores["factual_grounding"] == 1
    assert scores["citation_quality"] == 1
    assert scores["citation_precision"] == 0.0


def test_fully_supported_answer_scores_high():
    answer = "Eligible employees can take up to 12 workweeks of leave in a 12-month period.\n\nConfidence: High"
    scores, _ = score_result(
        {
            "id": "q",
            "answerable": True,
            "expected_doc": "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf",
            "expected_pages": [3],
            "expected_facts": ["12 workweeks", "12-month period"],
        },
        _result(answer, [FMLA_CHUNK]),
    )
    assert scores["factual_grounding"] == 5
    assert scores["citation_quality"] == 5
    assert scores["expected_source_hit"] == 1.0
    assert scores["expected_facts_found"] == 1.0
    assert scores["refusal_correct"] == 1.0


def test_supported_answer_from_the_wrong_source_is_capped():
    answer = "Vacation is earned each pay period and may be carried over to the next year."
    scores, _ = score_result(
        {"id": "q", "answerable": True, "expected_doc": "UWSystem-1212-Sick Leave-2026.pdf"},
        _result(answer, [VACATION_CHUNK]),
    )
    assert scores["factual_grounding"] == 5
    assert scores["expected_source_hit"] == 0.0
    assert scores["citation_quality"] == 2


# --- not-covered questions -----------------------------------------------------------------


def test_refusal_on_a_not_covered_question_is_correct():
    answer = "The retrieved documents do not specify a 401(k) employer match.\n\nConfidence: Low"
    scores, _ = score_result({"id": "q", "answerable": False}, _result(answer, [VACATION_CHUNK]))
    assert scores["factual_grounding"] == 5
    assert scores["refusal_correct"] == 1.0
    assert scores["citation_quality"] is None
    assert scores["citation_precision"] is None


def test_answering_a_not_covered_question_is_wrong():
    answer = "Vacation is earned each pay period and may be carried over to the next year."
    scores, _ = score_result({"id": "q", "answerable": False}, _result(answer, [VACATION_CHUNK]))
    assert scores["factual_grounding"] == 1
    assert scores["refusal_correct"] == 0.0


def test_refusing_an_answerable_question_is_wrong():
    answer = "I could not find enough grounded evidence in the indexed corpus to answer that confidently."
    scores, _ = score_result({"id": "q", "answerable": True}, _result(answer, [FMLA_CHUNK]))
    assert scores["factual_grounding"] == 1
    assert scores["refusal_correct"] == 0.0


# --- expected source and facts -------------------------------------------------------------


def test_expected_source_hit_matches_document_and_page_overlap():
    chunks = [{**FMLA_CHUNK, "page_start": 2, "page_end": 4}]
    assert expected_source_hit(chunks, "DOL-FMLA-EmployeeGuide-2026.pdf", [3]) is True
    assert expected_source_hit(chunks, "DOL-FMLA-EmployeeGuide-2026.pdf", [7]) is False
    assert expected_source_hit(chunks, "UWSystem-1213-FMLA-2026.pdf", []) is False
    assert expected_source_hit(chunks, None, [3]) is None


def test_expected_facts_found():
    answer = "You can take 12 workweeks in a 12-month period."
    assert expected_facts_found(answer, ["12 workweeks", "1,250 hours"]) == 0.5
    assert expected_facts_found(answer, []) is None


# --- run_evaluation end to end, with no model calls ----------------------------------------


class _FakeProvider:
    def __init__(self, judge_text: str):
        self.judge_text = judge_text

    def generate_answer(self, prompt: str):
        return SimpleNamespace(text=self.judge_text)


def test_run_evaluation_writes_report_and_skips_missing_scores(monkeypatch):
    root = Path.cwd() / ".tmp_test_evaluation" / f"case-{uuid4().hex}"
    try:
        config = AppConfig(paths=ProjectPaths(root=root))
        prompts_path = root / "prompts.json"
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {"id": "a", "question": "How much FMLA leave?", "answerable": True,
                         "expected_doc": "DOL-FMLA-EmployeeGuide-2026.pdf", "expected_pages": [3]},
                        {"id": "b", "question": "What is the 401(k) match?", "answerable": False},
                    ]
                }
            ),
            encoding="utf-8",
        )
        answers = {
            "How much FMLA leave?": _result(
                "Eligible employees can take up to 12 workweeks of leave in a 12-month period.", [FMLA_CHUNK]
            ),
            "What is the 401(k) match?": _result(
                "The retrieved documents do not specify a 401(k) match.", [VACATION_CHUNK]
            ),
        }
        monkeypatch.setattr(evaluation, "ask_question", lambda _c, _p, question, **_kw: answers[question])

        summary = evaluation.run_evaluation(config, _FakeProvider("not json"), prompts_path=prompts_path)

        assert summary["scoring_version"] == 2
        assert summary["judge_failures"] == 2
        averages = summary["averages"]
        assert averages["factual_grounding"] == 5
        assert averages["refusal_correct"] == 1
        assert averages["citation_quality"] == 5  # the refusal has no citation score and is skipped
        assert "faithfulness_llm" not in averages  # every judge call failed; no fake zeros
        assert all(isinstance(value, float | int) for value in averages.values())
        reports = list(config.paths.eval_dir.glob("eval_report_*.md"))
        assert len(reports) == 1
        assert "Claims supported by a cited chunk: 1 of 1" in reports[0].read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --- the HR question set and alternate sources ----------------------------------------------


def test_also_in_counts_as_the_expected_source():
    answer = "Eligible employees can take up to 12 workweeks of leave in a 12-month period."
    prompt = {
        "id": "q",
        "answerable": True,
        "expected_doc": "03_UW_System/UWSystem-1213-FMLA-2026.pdf",
        "expected_pages": [2],
        "also_in": [{"doc": "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf", "pages": [3]}],
    }
    scores, _ = score_result(prompt, _result(answer, [FMLA_CHUNK]))
    assert scores["expected_source_hit"] == 1.0
    scores, _ = score_result({**prompt, "also_in": []}, _result(answer, [FMLA_CHUNK]))
    assert scores["expected_source_hit"] == 0.0


def test_numbers_with_thousands_separators_match():
    assert expected_facts_found("You need 1,250 hours of service.", ["1250 hours"]) == 1.0


HR_SET = Path(__file__).resolve().parents[1] / "eval" / "hr_policy_questions.json"


def test_hr_question_set_is_well_formed():
    payload = json.loads(HR_SET.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    assert 40 <= len(prompts) <= 50
    assert len({p["id"] for p in prompts}) == len(prompts)
    answerable = [p for p in prompts if p["answerable"]]
    not_covered = [p for p in prompts if not p["answerable"]]
    assert len(answerable) >= 30 and len(not_covered) >= 8
    for p in answerable:
        assert p["question"].strip() and p["expected_doc"].endswith(".pdf")
        assert p["expected_pages"] and all(isinstance(n, int) and n > 0 for n in p["expected_pages"])
        assert 1 <= len(p["expected_facts"]) <= 3
        for alt in p.get("also_in", []):
            assert alt["doc"].endswith(".pdf") and alt["pages"]
    for p in not_covered:
        assert p["expected_doc"] is None and not p["expected_facts"] and p["why_not_covered"].strip()
