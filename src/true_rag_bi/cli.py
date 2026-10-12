from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunking import chunk_documents
from .config import build_config
from .evaluation import run_evaluation
from .indexing import index_documents
from .ingest import ingest_documents
from .providers import build_provider
from .retrieval import ask_question, chat_session
from .utils import configure_logging

_MAX_QUERY_LENGTH = 2000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bi-rag", description="Local-first Business Intelligence RAG pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Extract normalized Markdown from PDFs.")
    ingest_parser.add_argument("--verbose", action="store_true")

    index_parser = subparsers.add_parser("index", help="Refresh ingestion, chunking, and Chroma indexing.")
    index_parser.add_argument("--verbose", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question against the indexed corpus.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--debug", action="store_true")
    ask_parser.add_argument("--verbose", action="store_true")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive grounded chat session.")
    chat_parser.add_argument("--debug", action="store_true")
    chat_parser.add_argument("--verbose", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Run the built-in BI evaluation prompts.")
    eval_parser.add_argument("--prompts", type=Path, default=None)
    eval_parser.add_argument("--verbose", action="store_true")

    return parser


def _print_summary(prefix: str, payload: dict) -> None:
    print(prefix)
    for key, value in payload.items():
        print(f"- {key}: {value}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    verbose = getattr(args, "verbose", False)
    configure_logging(verbose=verbose)
    config = build_config()

    try:
        _run(args, config)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print("Run with --verbose for more detail.", file=sys.stderr)
        sys.exit(1)


def _run(args, config) -> None:  # noqa: ANN001
    if args.command == "ingest":
        result = ingest_documents(config, verbose=args.verbose)
        _print_summary("Ingestion complete.", result["cache_stats"])
        print(f"- documents: {len(result['documents'])}")
        return

    if args.command == "index":
        ingest_result = ingest_documents(config, verbose=args.verbose)
        chunk_result = chunk_documents(config, verbose=args.verbose)
        provider = build_provider(config)
        index_result = index_documents(config, provider, verbose=args.verbose)
        print("Index build complete.")
        print(f"- ingestion cache: {ingest_result['cache_stats']}")
        print(f"- chunk cache: {chunk_result['cache_stats']}")
        print(f"- embedding cache: {index_result['cache_stats']}")
        print(f"- corpus fingerprint: {index_result['corpus_fingerprint']}")
        return

    if args.command == "ask" and len(args.question) > _MAX_QUERY_LENGTH:
        raise RuntimeError(
            f"Query is {len(args.question)} characters, which exceeds the {_MAX_QUERY_LENGTH}-character limit. "
            "Please shorten your question."
        )

    provider = build_provider(config)

    if args.command == "ask":
        result = ask_question(config, provider, args.question, debug=args.debug, verbose=args.verbose)
        print(result["answer"])
        if result["citations"]:
            print("\nSources:")
            for citation in result["citations"]:
                print(f"- {citation}")
        print("\nMetrics:")
        print(f"- cache: {result['cache_stats']}")
        print(f"- retrieval_seconds: {result['metrics']['retrieval_seconds']}")
        print(f"- generation_seconds: {result['metrics']['generation_seconds']}")
        print(f"- total_tokens: {result['metrics']['total_tokens']}")
        if args.debug and result["retrieval_debug"]:
            print("\nRetrieval debug:")
            for item in result["retrieval_debug"]["selected"]:
                print(
                    f"- order={item['context_order']} chunk={item['chunk_id']} "
                    f"score={item['effective_score']} fusion={item['fusion_score']} "
                    f"channels={','.join(item['retrieval_channels']) or 'none'} citation={item['citation']} "
                    f"terms={','.join(item['matched_terms']) or 'none'} reason={item['selection_reason']}"
                )
        return

    if args.command == "chat":
        chat_session(config, provider, debug=args.debug, verbose=args.verbose)
        return

    if args.command == "eval":
        result = run_evaluation(config, provider, prompts_path=args.prompts, verbose=args.verbose)
        print("Evaluation complete.")
        print(f"- prompts: {result['prompt_count']}")
        print(f"- averages: {result['averages']}")
        return
