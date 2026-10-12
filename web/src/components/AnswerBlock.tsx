import type { AskResponse } from "../api/client";
import { StructuredMarkdown } from "./StructuredMarkdown";

const confidenceTone: Record<AskResponse["confidence"], string> = {
  High: "badge badge--high",
  Medium: "badge badge--medium",
  Low: "badge badge--low",
};

function metricString(metrics: AskResponse["metrics"], key: string) {
  const value = metrics[key];
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  return "n/a";
}

function isCachedRun(metrics: AskResponse["metrics"]) {
  return metrics.cache_served === true || metrics.cached_response === true || metrics.remote_calls === 0;
}

function totalSeconds(metrics: AskResponse["metrics"]) {
  if (isCachedRun(metrics) && typeof metrics.client_runtime_seconds === "number") {
    return String(metrics.client_runtime_seconds);
  }
  if (isCachedRun(metrics) && typeof metrics.api_response_seconds === "number") {
    return String(metrics.api_response_seconds);
  }
  if (isCachedRun(metrics) && typeof metrics.cache_lookup_seconds === "number") {
    return String(metrics.cache_lookup_seconds);
  }
  if (isCachedRun(metrics) && typeof metrics.retrieval_seconds === "number") {
    return String(metrics.retrieval_seconds);
  }
  return metricString(metrics, "total_seconds");
}

function totalTokens(metrics: AskResponse["metrics"]) {
  if (isCachedRun(metrics)) {
    return "0";
  }
  return metricString(metrics, "total_tokens");
}

export function AnswerBlock({ response, title }: { response: AskResponse; title: string }) {
  return (
    <section className="panel answer-panel">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">{title}</span>
          <h2>Analytic output</h2>
        </div>
        <div className="panel__header-actions">
          <span className={confidenceTone[response.confidence]}>{response.confidence} confidence</span>
        </div>
      </div>

      <StructuredMarkdown answer={response.answer} />

      <div className="metrics-row answer-panel__metrics">
        <span>{response.citations.length} citations</span>
        <span>{totalSeconds(response.metrics)}s total</span>
        <span>{totalTokens(response.metrics)} tokens</span>
        <span>{metricString(response.metrics, "remote_calls")} remote calls</span>
      </div>
    </section>
  );
}
