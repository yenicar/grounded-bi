import type { AskResponse } from "../api/client";

function metricValue(metrics: AskResponse["metrics"], key: string) {
  return metrics[key];
}

function metricString(metrics: AskResponse["metrics"], key: string) {
  const value = metricValue(metrics, key);
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  return "n/a";
}

function isCachedRun(metrics: AskResponse["metrics"]) {
  return metrics.cache_served === true || metrics.cached_response === true || metrics.remote_calls === 0;
}

export function AnswerTelemetry({ response }: { response: AskResponse }) {
  const cachedRun = isCachedRun(response.metrics);
  const grounding = cachedRun ? undefined : response.retrieval_debug?.grounding;
  const verdict = cachedRun
    ? "cached"
    : grounding?.verdict ?? String(metricValue(response.metrics, "grounding_verdict") ?? "unknown");
  const removedClaims = cachedRun
    ? 0
    : grounding?.unsupported_claims_removed?.length ??
      Number(metricValue(response.metrics, "unsupported_claims_removed") ?? 0);
  const totalSeconds =
    cachedRun && typeof response.metrics.client_runtime_seconds === "number"
      ? String(response.metrics.client_runtime_seconds)
      : cachedRun && typeof response.metrics.api_response_seconds === "number"
        ? String(response.metrics.api_response_seconds)
        : cachedRun && typeof response.metrics.cache_lookup_seconds === "number"
          ? String(response.metrics.cache_lookup_seconds)
          : cachedRun && typeof response.metrics.retrieval_seconds === "number"
            ? String(response.metrics.retrieval_seconds)
      : metricString(response.metrics, "total_seconds");
  const groundingSeconds = cachedRun ? "0" : metricString(response.metrics, "grounding_seconds");

  return (
    <section className="panel panel--secondary">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">Answer diagnostics</span>
          <h2>Grounding and execution profile</h2>
        </div>
      </div>

      <div className="summary-grid">
        <div className="metric-card">
          <span>Grounding verdict</span>
          <strong>{String(verdict)}</strong>
          <small>Whether the second-pass checker kept, trimmed, or downgraded the answer.</small>
        </div>
        <div className="metric-card">
          <span>Unsupported claims removed</span>
          <strong>{removedClaims}</strong>
          <small>Higher means the first draft carried more unsupported detail.</small>
        </div>
        <div className="metric-card">
          <span>Prompt evidence tokens</span>
          <strong>{metricString(response.metrics, "prompt_context_tokens")}</strong>
          <small>How much evidence text actually went into generation.</small>
        </div>
        <div className="metric-card">
          <span>Prompt contexts</span>
          <strong>{metricString(response.metrics, "prompt_contexts")}</strong>
          <small>The number of contexts sent to generation after compression.</small>
        </div>
        <div className="metric-card">
          <span>Retrieved contexts</span>
          <strong>{metricString(response.metrics, "retrieved_contexts")}</strong>
          <small>The number of contexts retrieval selected before prompt trimming.</small>
        </div>
        <div className="metric-card">
          <span>Total runtime</span>
          <strong>{totalSeconds}s</strong>
          <small>Grounding pass: {groundingSeconds}s</small>
        </div>
      </div>

      {grounding?.unsupported_claims_removed?.length ? (
        <details className="citation-card citation-card--compact">
          <summary>
            <span className="citation-card__index">!</span>
            <span>
              <strong>Claims removed by the grounding pass</strong>
              <small>{grounding.unsupported_claims_removed.length} items</small>
            </span>
          </summary>
          <div className="simple-list">
            {grounding.unsupported_claims_removed.map((claim, index) => (
              <p key={`${index}-${claim}`}>{claim}</p>
            ))}
          </div>
        </details>
      ) : null}
    </section>
  );
}
