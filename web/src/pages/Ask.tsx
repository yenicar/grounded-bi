import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { api, type AskResponse } from "../api/client";
import { AnswerBlock } from "../components/AnswerBlock";
import { AnswerTelemetry } from "../components/AnswerTelemetry";
import { CitationCard } from "../components/CitationCard";
import { Collapsible } from "../components/Collapsible";
import { AnalysisSkeleton } from "../components/LoadingSkeleton";
import { ProgressCard } from "../components/ProgressCard";

export function AskPage() {
  const [question, setQuestion] = useState("");
  const [debug, setDebug] = useState(false);
  const [groundingEnabled, setGroundingEnabled] = useState(false);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  function formatAskError(err: unknown) {
    const message = err instanceof Error ? err.message : "Question failed.";
    if (message.includes("Access is denied")) {
      return "The HR corpus index could not be opened on this machine. I fixed the storage path; rebuild the corpus once if this message persists.";
    }
    if (message.includes("Index manifest not found") || message.includes("Keyword index not found")) {
      return "The active HR corpus is not fully indexed yet. Rebuild the corpus, then retry the question.";
    }
    return message;
  }

  useEffect(() => {
    async function loadRecommendations() {
      try {
        const response = await api.getRecommendations();
        setRecommendations(response.recommendations);
      } catch (err) {
        console.error("Failed to load recommendations", err);
      }
    }
    void loadRecommendations();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      setResponse(await api.ask(question.trim(), debug, groundingEnabled));
    } catch (err) {
      setError(formatAskError(err));
    } finally {
      setIsLoading(false);
    }
  }

  async function handleExport(kind: "pdf" | "pptx") {
    if (!response) {
      return;
    }
    const payload = {
      title: question.trim().slice(0, 80) || "GroundedBI export",
      question: question.trim(),
      answer: response.answer,
      citations: response.citations,
      generated_at: String(response.metrics.generated_at ?? ""),
    };
    const blob = kind === "pdf" ? await api.exportPdf(payload) : await api.exportPptx(payload);
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${payload.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "") || "groundedbi-export"}.${kind}`;
    anchor.click();
    window.URL.revokeObjectURL(url);
  }

  const showCanvas = isLoading || !!response || !!error;

  return (
    <section className="page-stack">
      <div className="section-heading">
        <div>
          <span className="panel__eyebrow">HR assistant</span>
          <h2>Ask HR policy questions against the corpus.</h2>
          <p>Write the question first, then review the answer, citations, and retrieval trace in one workspace.</p>
        </div>
      </div>

      <div className="analytic-layout">
        <aside className="query-panel-stack query-panel-stack--left">
          <Collapsible title="Suggested HR questions" eyebrow="Dynamic prompts" defaultOpen={true}>
            <div className="template-list">
              {recommendations.length > 0 ? (
                recommendations.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    className="template-card"
                    onClick={() => {
                      setQuestion(preset);
                      window.requestAnimationFrame(() => {
                        composerRef.current?.focus();
                        const length = preset.length;
                        composerRef.current?.setSelectionRange(length, length);
                      });
                    }}
                  >
                    <strong>{preset}</strong>
                    <small>Use this question</small>
                  </button>
                ))
              ) : (
                <div className="empty-state" style={{ padding: "1rem" }}>
                  Discovering HR question suggestions...
                </div>
              )}
            </div>
          </Collapsible>
        </aside>

        {showCanvas ? (
          <section className="ask-canvas panel panel--canvas">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Answer workspace</span>
                <h2>Grounded response</h2>
                <p>Review the answer, exports, and evidence only when the assistant has something to show.</p>
              </div>
              <div className="panel__header-actions">
                <button
                  type="button"
                  className="ghost-button button--small"
                  onClick={() => {
                    setResponse(null);
                    setError(null);
                  }}
                >
                  Close workspace
                </button>
              </div>
            </div>

            {error ? <div className="error-banner">{error}</div> : null}
            <ProgressCard
              active={isLoading}
              title="Running HR policy query"
              statuses={[
                "Rewriting the question for cleaner retrieval...",
                "Searching the indexed corpus and compressing evidence...",
                "Drafting the answer against the selected evidence...",
                groundingEnabled
                  ? "Running the grounding pass before release..."
                  : "Preparing the answer without the extra grounding pass...",
              ]}
              stalledLabel="The model is still working. If this keeps hanging, retry the query or simplify the policy question."
              actionLabel="Retry query"
              onAction={() => {
                if (!question.trim()) {
                  return;
                }
                const form = document.querySelector("form.composer.query-dock") as HTMLFormElement | null;
                form?.requestSubmit();
              }}
              stalledAfterMs={11000}
              durationMs={16000}
            />

            {isLoading ? (
              <AnalysisSkeleton />
            ) : response ? (
              <div className="page-stack">
                <AnswerBlock response={response} title="Assessment" />
                <div className="button-row">
                  <button className="ghost-button button--small" type="button" onClick={() => void handleExport("pdf")}>
                    Export PDF
                  </button>
                  <button className="ghost-button button--small" type="button" onClick={() => void handleExport("pptx")}>
                    Export PPTX
                  </button>
                </div>
                <AnswerTelemetry response={response} />
              </div>
            ) : null}
          </section>
        ) : null}

        <aside className="query-panel-stack query-panel-stack--right">
          <Collapsible title="Ask the assistant" eyebrow="Question composer" defaultOpen={true}>
            <form className="panel-flow composer query-dock" onSubmit={handleSubmit}>
              <textarea
                ref={composerRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Example: Compare Federal FMLA guidance with UW System leave policy for intermittent leave."
                rows={8}
              />
              <div className="composer__footer">
                <div className="composer__toggles">
                  <label className="toggle toggle--trace">
                    <input
                      type="checkbox"
                      checked={debug}
                      onChange={(event) => setDebug(event.target.checked)}
                    />
                    Trace review mode
                  </label>
                  <label className="toggle toggle--trace">
                    <input
                      type="checkbox"
                      checked={groundingEnabled}
                      onChange={(event) => setGroundingEnabled(event.target.checked)}
                    />
                    Grounding pass
                  </label>
                </div>
                <button type="submit" disabled={isLoading || !question.trim()}>
                  {isLoading ? "Analyzing..." : "Run query"}
                </button>
              </div>
            </form>
          </Collapsible>

          <Collapsible title="Supporting citations" eyebrow="Evidence lane" defaultOpen={true}>
            {response ? (
              <div className="citation-list">
                {response.citations.map((citation, index) => (
                  <CitationCard
                    key={`${citation.doc_name}-${citation.page_range}-${index}`}
                    citation={citation}
                    index={index + 1}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: "1rem" }}>
                Evidence cards will appear here after the assistant answers.
              </div>
            )}
          </Collapsible>
        </aside>
      </div>
    </section>
  );
}
