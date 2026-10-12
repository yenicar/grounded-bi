from __future__ import annotations

from datetime import datetime, timezone
import json as json_lib
from pathlib import Path
import re
from typing import Any

from .config import AppConfig
from .grounding_check import assess_answer, expected_facts_found, expected_source_hit
from .providers import RAGProvider
from .retrieval import ask_question
from .utils import read_json, write_json


def _default_eval_prompts_path(config: AppConfig) -> Path:
    return config.paths.eval_dir / "default_prompts.json"


SCORING_VERSION = 2
SCORING_NOTES = (
    "factual_grounding and citation_quality check that the cited chunks support the answer "
    "claim by claim (grounding_check.py), not that citations exist. Version 1 scored grounding 5 "
    "for any answer with citations and citation quality by the number of distinct documents."
)


def _cited_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The chunks the answer cites, with their text: the contexts the model was given."""
    return list(result.get("retrieval_debug", {}).get("selected", []) or [])


def _band(value: float, cutoffs: tuple[float, float, float, float]) -> int:
    """5 for value >= cutoffs[0], 4 for >= cutoffs[1], 3, 2, else 1."""
    for score, cutoff in zip((5, 4, 3, 2), cutoffs):
        if value >= cutoff:
            return score
    return 1


def _score_factual_grounding(grounding: dict[str, Any], answerable: bool | None) -> int:
    """1-5 from the share of the answer's claims that a cited chunk supports.

    A refusal scores 5 on a question the corpus does not cover, 1 on one it does, and 2 when the
    prompt does not say. An answer to a question marked not covered scores 1.
    """
    if grounding["refusal"]:
        if answerable is False:
            return 5
        return 1 if answerable is True else 2
    if answerable is False:
        return 1
    rate = grounding["claim_support_rate"]
    if rate is None:
        return 1
    return _band(rate, (0.95, 0.8, 0.6, 0.4))


def _score_citation_quality(grounding: dict[str, Any], source_hit: bool | None) -> int | None:
    """1-5 from the share of cited chunks that support at least one claim; None for a refusal.

    Capped at 2 when the prompt names an expected source and no cited chunk comes from it.
    """
    if grounding["refusal"]:
        return None
    precision = grounding["citation_precision"]
    if not precision:
        return 1
    score = _band(precision, (0.75, 0.5, 0.25, 0.0001))
    if source_hit is False:
        score = min(score, 2)
    return score


def score_result(prompt: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic scores for one answer, and the claim-level grounding detail behind them."""
    cited = _cited_chunks(result)
    grounding = assess_answer(result.get("answer", ""), cited)
    answerable = prompt.get("answerable")
    source_hit = expected_source_hit(cited, prompt.get("expected_doc"), prompt.get("expected_pages"))
    if source_hit is False:
        # other pages that state the same answer (also_in) count as the expected source too
        source_hit = any(
            expected_source_hit(cited, alt.get("doc"), alt.get("pages")) for alt in prompt.get("also_in", [])
        )
    facts = expected_facts_found(result.get("answer", ""), prompt.get("expected_facts"))
    scores: dict[str, Any] = {
        "factual_grounding": _score_factual_grounding(grounding, answerable),
        "citation_quality": _score_citation_quality(grounding, source_hit),
        "claim_support_rate": grounding["claim_support_rate"],
        "citation_precision": None if grounding["refusal"] else grounding["citation_precision"],
        "expected_source_hit": None if source_hit is None else float(source_hit),
        "expected_facts_found": facts,
        "refusal_correct": None if answerable is None else float(grounding["refusal"] == (answerable is False)),
    }
    return scores, grounding


def _average(results: list[dict[str, Any]], key: str) -> float | None:
    values = [item["scores"].get(key) for item in results]
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 3) if values else None


def _score_latency(result: dict[str, Any]) -> int:
    total = result["metrics"]["total_seconds"]
    if total <= 2.0:
        return 5
    if total <= 5.0:
        return 4
    if total <= 8.0:
        return 3
    return 2


def _score_cost(result: dict[str, Any]) -> int:
    total_tokens = result["metrics"]["total_tokens"]
    remote_calls = result["metrics"]["remote_calls"]
    if remote_calls <= 1 and total_tokens <= 2500:
        return 5
    if remote_calls <= 2 and total_tokens <= 5000:
        return 4
    if total_tokens <= 7000:
        return 3
    return 2


def _score_faithfulness_llm(
    provider: RAGProvider,
    question: str,
    answer: str,
    citations: list[str],
    context_texts: list[str],
) -> dict[str, Any]:
    context_block = "\n\n".join(
        f"[{index + 1}] {text[:1500]}"
        for index, text in enumerate(context_texts[:8])
    )
    prompt = f"""You are evaluating a RAG system answer for faithfulness.

Question: {question}

Retrieved context:
{context_block}

Answer given:
{answer}

Visible citations:
{chr(10).join(citations) if citations else "(none)"}

Score the answer on two dimensions from 1-5:
- faithfulness: Does every claim in the answer appear in the retrieved context? (5=fully grounded, 1=fabricated)
- relevance: Does the answer actually address the question? (5=directly answers, 1=off-topic)

Respond with ONLY valid JSON in this exact format:
{{"faithfulness": <int>, "relevance": <int>, "reasoning": "<one sentence>"}}"""
    try:
        result = provider.generate_answer(prompt)
        raw = result.text.strip()
        try:
            return json_lib.loads(raw)
        except json_lib.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json_lib.loads(match.group(0))
            raise
    except Exception:
        # None, not 0, so a failed judge call is left out of the averages instead of dragging them down
        return {"faithfulness": None, "relevance": None, "reasoning": "judge_failed"}


AVERAGED_SCORES = (
    "factual_grounding",
    "citation_quality",
    "claim_support_rate",
    "citation_precision",
    "expected_source_hit",
    "expected_facts_found",
    "refusal_correct",
    "latency",
    "cost_per_query",
    "faithfulness_llm",
    "relevance_llm",
)


def run_evaluation(
    config: AppConfig,
    provider: RAGProvider,
    prompts_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    prompts_path = prompts_path or _default_eval_prompts_path(config)
    prompts_payload = read_json(prompts_path, {"prompts": []})
    prompts = prompts_payload.get("prompts", [])
    if not prompts:
        raise RuntimeError(f"No evaluation prompts found at {prompts_path}")

    results: list[dict[str, Any]] = []
    for prompt in prompts:
        result = ask_question(config, provider, prompt["question"], debug=True, verbose=verbose)
        context_texts = [item.get("text", "") for item in result.get("retrieval_debug", {}).get("selected", [])]
        llm_scores = _score_faithfulness_llm(
            provider,
            question=prompt["question"],
            answer=result["answer"],
            citations=result["citations"],
            context_texts=context_texts,
        )
        scores, grounding = score_result(prompt, result)
        scores.update({
            "latency": _score_latency(result),
            "cost_per_query": _score_cost(result),
            "faithfulness_llm": llm_scores.get("faithfulness"),
            "relevance_llm": llm_scores.get("relevance"),
        })
        results.append(
            {
                "question_id": prompt["id"],
                "question": prompt["question"],
                "category": prompt.get("category", "general"),
                "result": result,
                "scores": scores,
                "judge": llm_scores,
                "grounding": grounding,
                "expected": {
                    key: prompt[key]
                    for key in ("answerable", "expected_doc", "expected_pages", "expected_facts", "also_in")
                    if key in prompt
                },
            }
        )

    judge_failures = sum(1 for item in results if item["judge"].get("reasoning") == "judge_failed")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "generated_at": timestamp,
        "prompt_count": len(results),
        "judge_failures": judge_failures,
        "scoring_version": SCORING_VERSION,
        "scoring_notes": SCORING_NOTES,
        "averages": {
            key: value
            for key in AVERAGED_SCORES
            if (value := _average(results, key)) is not None
        },
        "results": results,
    }

    write_json(config.paths.eval_dir / f"eval_report_{timestamp}.json", summary)
    markdown_lines = [
        "# Business Intelligence RAG Evaluation Report",
        "",
        f"- Generated at: {timestamp}",
        f"- Prompt count: {len(results)}",
        f"- Judge failures: {judge_failures}",
        "",
        f"- Scoring version: {SCORING_VERSION}. {SCORING_NOTES}",
        "",
        "## Average scores",
        "",
        *[f"- {key}: {value}" for key, value in summary["averages"].items()],
        "",
        "## Results",
        "",
    ]
    for item in results:
        markdown_lines.append(f"### {item['question_id']}: {item['question']}")
        markdown_lines.append(f"- Category: {item['category']}")
        markdown_lines.append(f"- Scores: {item['scores']}")
        markdown_lines.append(f"- Judge reasoning: {item['judge'].get('reasoning', 'n/a')}")
        markdown_lines.append(f"- Total seconds: {item['result']['metrics']['total_seconds']}")
        markdown_lines.append(f"- Citations: {len(item['result']['citations'])}")
        unsupported = [claim["claim"] for claim in item["grounding"]["claims"] if not claim["supported"]]
        markdown_lines.append(
            f"- Claims supported by a cited chunk: {item['grounding']['supported_claims']} of {item['grounding']['claim_count']}"
        )
        for claim in unsupported[:5]:
            markdown_lines.append(f"  - unsupported: {claim}")
        markdown_lines.append("")

    (config.paths.eval_dir / f"eval_report_{timestamp}.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )
    return summary
