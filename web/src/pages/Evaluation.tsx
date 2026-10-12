import { useEffect, useState } from "react";

import { api, type EvalReportResponse, type EvalReportSummary } from "../api/client";
import { ReportSkeleton } from "../components/LoadingSkeleton";
import { ProgressCard } from "../components/ProgressCard";

const metricLabels: Record<string, string> = {
  faithfulness_llm: "Judge-groundedness",
  relevance_llm: "Judge-relevance",
  latency: "Latency",
  cost_per_query: "Cost pressure",
  factual_grounding: "Claim support",
  citation_quality: "Citation precision",
  claim_support_rate: "Claims supported",
  citation_precision: "Citations used",
  expected_source_hit: "Expected source retrieved",
  expected_facts_found: "Expected facts in answer",
  refusal_correct: "Correct answer or refusal",
};

const metricHelp: Record<string, string> = {
  faithfulness_llm: "LLM judge score. Higher means the answer stays inside the retrieved evidence.",
  relevance_llm: "LLM judge score. Higher means the answer actually addresses the question asked.",
  latency: "A coarse speed score, not raw seconds. Higher is faster.",
  cost_per_query: "A coarse efficiency score based on token and API usage. Higher is cheaper.",
  factual_grounding:
    "1 to 5 from the share of the answer's claims that a cited chunk supports (content words and every number found in the chunk). Refusals score by whether the question was answerable.",
  citation_quality:
    "1 to 5 from the share of cited chunks that support at least one claim; capped at 2 when the expected source was not retrieved.",
  claim_support_rate: "Share of answer claims supported by a cited chunk (0 to 1).",
  citation_precision: "Share of cited chunks that support at least one claim (0 to 1).",
  expected_source_hit: "Share of questions where a cited chunk came from the expected document and page.",
  expected_facts_found: "Share of the expected key facts that appear in the answer.",
  refusal_correct: "Share of questions answered when covered and declined when not covered.",
};

function formatGeneratedAt(value: string) {
  if (!value) {
    return "Unknown";
  }
  const normalized = value.includes("T") ? value : `${value.slice(0, 8)}T${value.slice(9, 15)}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function buildMarkdownReport(report: EvalReportResponse) {
  const lines = [
    "# GroundedBI Evaluation Report",
    "",
    `- Generated at: ${report.generated_at}`,
    `- Prompt count: ${report.prompt_count}`,
    `- Judge failures: ${report.judge_failures}`,
    "",
    "## Average scores",
    "",
    ...Object.entries(report.averages).map(([key, value]) => `- ${key}: ${value}`),
    "",
    "## Results",
    "",
  ];

  for (const item of report.results) {
    lines.push(`### ${item.question_id}: ${item.question}`);
    lines.push(`- Category: ${item.category}`);
    lines.push(`- Scores: ${JSON.stringify(item.scores)}`);
    lines.push(`- Judge reasoning: ${item.judge.reasoning ?? "n/a"}`);
    lines.push(`- Citations: ${item.result.citations.length}`);
    lines.push("");
  }

  return lines.join("\n");
}

function downloadMarkdown(report: EvalReportResponse) {
  const blob = new Blob([buildMarkdownReport(report)], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${report.report_id}.md`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function metricLabel(key: string) {
  return metricLabels[key] ?? key;
}

export function EvaluationPage() {
  const [reports, setReports] = useState<EvalReportSummary[]>([]);
  const [selectedReport, setSelectedReport] = useState<EvalReportResponse | null>(null);
  const [loadingReports, setLoadingReports] = useState(true);
  const [running, setRunning] = useState(false);
  const [runningMessage, setRunningMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadReports() {
    try {
      const next = await api.listEvalReports();
      setReports(next);
      if (next.length > 0 && !selectedReport) {
        setSelectedReport(await api.getEvalReport(next[0].report_id));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load evaluation reports.");
    } finally {
      setLoadingReports(false);
    }
  }

  useEffect(() => {
    void loadReports();
  }, []);

  async function runEvaluation() {
    setRunning(true);
    setError(null);
    setRunningMessage("Queueing evaluation...");
    try {
      const queued = await api.runEval();
      setRunningMessage("Evaluation is running in the background...");
      let reportId: string | null = null;

      while (!reportId) {
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        const job = await api.getEvalJob(queued.job_id);
        if (job.status === "failed") {
          throw new Error(job.error || "Evaluation job failed.");
        }
        if (job.status === "completed" && job.report_id) {
          reportId = job.report_id;
          break;
        }
      }

      const report = await api.getEvalReport(reportId);
      setSelectedReport(report);
      setReports((current) => [
        {
          report_id: report.report_id,
          generated_at: report.generated_at,
          prompt_count: report.prompt_count,
          averages: report.averages,
        },
        ...current.filter((item) => item.report_id !== report.report_id),
      ]);
      setRunningMessage("Evaluation completed.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation run failed.");
      setRunningMessage(null);
    } finally {
      setRunning(false);
    }
  }

  async function openReport(reportId: string) {
    setError(null);
    try {
      setSelectedReport(await api.getEvalReport(reportId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open report.");
    }
  }

  return (
    <section className="page-stack">
      <div className="section-heading">
        <div>
          <span className="panel__eyebrow">Benchmark control</span>
          <h2>Run the benchmark and review model discipline.</h2>
          <p>This workspace is ordered for analyst workflow: select a run, inspect the summary, then drill into per-question results.</p>
        </div>
        <div className="section-heading__actions button-row">
          <button onClick={runEvaluation} disabled={running}>
            {running ? "Running evaluation..." : "Run evaluation"}
          </button>
          <button
            className="ghost-button button--small"
            onClick={() => selectedReport && downloadMarkdown(selectedReport)}
            disabled={!selectedReport}
          >
            Download report
          </button>
        </div>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {runningMessage ? (
        <section className="panel status-panel">
          <strong>Benchmark status</strong>
          <div className="status-panel__body">
            <span>{runningMessage}</span>
          </div>
        </section>
      ) : null}
      <ProgressCard
        active={running}
        title="Executing benchmark run"
        statuses={[
          "Queueing the benchmark and reserving evaluation capacity...",
          "Running prompts and collecting answer telemetry...",
          "Scoring groundedness, relevance, latency, and cost...",
          "Persisting the report and refreshing the board...",
        ]}
        stalledLabel="The benchmark is still running. You can keep working elsewhere in the app and refresh this board when ready."
        actionLabel="Refresh reports"
        onAction={() => {
          void loadReports();
        }}
        stalledAfterMs={14000}
        durationMs={22000}
      />

      <div className="eval-layout">
        <section className="panel report-list">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">Past reports</span>
              <h2>Select a report</h2>
            </div>
          </div>
          {loadingReports ? (
            <div className="empty-state">Loading report inventory...</div>
          ) : reports.length ? (
            reports.map((report) => (
              <button
                key={report.report_id}
                className={`report-list__item${selectedReport?.report_id === report.report_id ? " report-list__item--active" : ""}`}
                onClick={() => void openReport(report.report_id)}
              >
                <strong>{formatGeneratedAt(report.generated_at)}</strong>
                <small>{report.prompt_count} prompts</small>
              </button>
            ))
          ) : (
            <div className="empty-state">No reports saved yet.</div>
          )}
        </section>

        <section className="page-stack">
          <section className="panel summary-bar">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Executive summary</span>
                <h2>{selectedReport ? formatGeneratedAt(selectedReport.generated_at) : "No report selected"}</h2>
              </div>
            </div>

            {loadingReports && !selectedReport ? (
              <ReportSkeleton />
            ) : selectedReport ? (
              <>
                <section className="snapshot-strip">
                  <div className="snapshot-strip__item">
                    <small>Prompts</small>
                    <strong>{selectedReport.prompt_count}</strong>
                  </div>
                  <div className="snapshot-strip__item">
                    <small>Judge failures</small>
                    <strong>{selectedReport.judge_failures}</strong>
                  </div>
                  <div className="snapshot-strip__item">
                    <small>Priority</small>
                    <strong>{metricLabel("faithfulness_llm")} / {metricLabel("relevance_llm")}</strong>
                  </div>
                </section>
                <div className="summary-grid summary-grid--four">
                  {Object.entries(selectedReport.averages).map(([key, value]) => (
                    <div key={key} className="metric-card">
                      <span>{metricLabel(key)}</span>
                      <strong>{value.toFixed(2)}</strong>
                      <small>{metricHelp[key] ?? "No description available."}</small>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">Run or open an evaluation report to inspect the benchmark.</div>
            )}
          </section>

          {selectedReport ? (
            <section className="panel results-table">
              <div className="panel__header">
                <div>
                  <span className="panel__eyebrow">Results</span>
                  <h2>Per-question evaluation breakdown</h2>
                </div>
              </div>

              <div className="results-table__head">
                <span>Question</span>
                <span>Category</span>
                <span>Groundedness</span>
                <span>Relevance</span>
                <span>Judge reasoning</span>
              </div>
              {selectedReport.results.map((item) => (
                <div key={item.question_id} className="results-table__row">
                  <span>{item.question}</span>
                  <span>{item.category}</span>
                  <span>
                    <strong>{item.scores.faithfulness_llm ?? "n/a"}</strong>
                    <small>{String(item.result.metrics.total_seconds ?? "n/a")}s</small>
                  </span>
                  <span>
                    <strong>{item.scores.relevance_llm ?? "n/a"}</strong>
                    <small>{String(item.result.metrics.total_tokens ?? "n/a")} tokens</small>
                  </span>
                  <span>{item.judge.reasoning ?? "n/a"}</span>
                </div>
              ))}
            </section>
          ) : null}

          <section className="panel panel--secondary">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Interpretation guide</span>
                <h2>Why this board matters</h2>
              </div>
            </div>
            <div className="hint-grid">
              <article className="hint-card">
                <strong>Regression control</strong>
                <p>The same prompts are rerun after pipeline changes so groundedness, relevance, latency, and cost regressions show up before release.</p>
              </article>
              <article className="hint-card">
                <strong>Operational signal</strong>
                <p>Enterprise performance means groundedness and relevance trend upward while latency, cost, and judge failures trend down.</p>
              </article>
            </div>
          </section>
        </section>
      </div>
    </section>
  );
}
