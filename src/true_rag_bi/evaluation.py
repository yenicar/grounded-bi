from __future__ import annotations

from datetime import datetime, timezone
import json as json_lib
from pathlib import Path
import re
from typing import Any

from .config import AppConfig
from .providers import RAGProvider
from .retrieval import ask_question
from .utils import read_json, write_json


def _default_eval_prompts_path(config: AppConfig) -> Path:
    return config.paths.eval_dir / "default_prompts.json"


def _score_factual_grounding(result: dict[str, Any]) -> int:
    answer = result["answer"].lower()
    if not result["citations"]:
        return 1
    if "insufficient" in answer or "could not find enough grounded evidence" in answer:
        return 2
    return 5 if result["citations"] else 2


def _score_citation_quality(result: dict[str, Any]) -> int:
    unique_docs = {citation.split(" p.", 1)[0].split(" pp.", 1)[0] for citation in result["citations"]}
    if not unique_docs:
        return 1
    if len(unique_docs) == 1:
        return 3
    return 5


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
        f"[{index + 1}] {text[:800]}"
        for index, text in enumerate(context_texts[:4])
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
        return {"faithfulness": 0, "relevance": 0, "reasoning": "judge_failed"}


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
        scores = {
            "factual_grounding": _score_factual_grounding(result),
            "citation_quality": _score_citation_quality(result),
            "latency": _score_latency(result),
            "cost_per_query": _score_cost(result),
            "faithfulness_llm": llm_scores.get("faithfulness", 0),
            "relevance_llm": llm_scores.get("relevance", 0),
        }
        results.append(
            {
                "question_id": prompt["id"],
                "question": prompt["question"],
                "category": prompt.get("category", "general"),
                "result": result,
                "scores": scores,
                "judge": llm_scores,
            }
        )

    judge_failures = sum(1 for item in results if item["judge"].get("reasoning") == "judge_failed")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "generated_at": timestamp,
        "prompt_count": len(results),
        "judge_failures": judge_failures,
        "averages": {
            "factual_grounding": round(sum(item["scores"]["factual_grounding"] for item in results) / len(results), 2),
            "citation_quality": round(sum(item["scores"]["citation_quality"] for item in results) / len(results), 2),
            "latency": round(sum(item["scores"]["latency"] for item in results) / len(results), 2),
            "cost_per_query": round(sum(item["scores"]["cost_per_query"] for item in results) / len(results), 2),
            "faithfulness_llm": round(sum(item["scores"]["faithfulness_llm"] for item in results) / len(results), 2),
            "relevance_llm": round(sum(item["scores"]["relevance_llm"] for item in results) / len(results), 2),
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
        "## Average scores",
        "",
        f"- Factual grounding: {summary['averages']['factual_grounding']}",
        f"- Citation quality: {summary['averages']['citation_quality']}",
        f"- Latency: {summary['averages']['latency']}",
        f"- Cost per query: {summary['averages']['cost_per_query']}",
        f"- Faithfulness (LLM judge): {summary['averages']['faithfulness_llm']}",
        f"- Relevance (LLM judge): {summary['averages']['relevance_llm']}",
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
        markdown_lines.append("")

    (config.paths.eval_dir / f"eval_report_{timestamp}.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )
    return summary
