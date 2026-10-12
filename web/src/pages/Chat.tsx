import type { FormEvent } from "react";
import { useState } from "react";

import { api, type ChatResponse, type ChatTurn } from "../api/client";
import { AnswerTelemetry } from "../components/AnswerTelemetry";
import { CitationCard } from "../components/CitationCard";
import { ProgressCard } from "../components/ProgressCard";
import { StructuredMarkdown } from "../components/StructuredMarkdown";

interface MessageEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
}

export function ChatPage() {
  const [draft, setDraft] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [groundingEnabled, setGroundingEnabled] = useState(false);

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) {
      return;
    }

    const userMessage: MessageEntry = {
      id: `${Date.now()}-user`,
      role: "user",
      text: message,
    };

    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setLoading(true);
    setError(null);

    try {
      const response = await api.chat(message, history, false, groundingEnabled);
      setHistory(response.history);
      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          text: response.answer,
          response,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed.");
    } finally {
      setLoading(false);
    }
  }

  function resetConversation() {
    setDraft("");
    setHistory([]);
    setMessages([]);
    setError(null);
  }

  async function handleExport(kind: "pdf" | "pptx") {
    if (!latestResponse) {
      return;
    }
    const lastUserQuestion = [...messages].reverse().find((entry) => entry.role === "user")?.text ?? "Conversation export";
    const payload = {
      title: lastUserQuestion.slice(0, 80) || "GroundedBI chat export",
      question: lastUserQuestion,
      answer: latestResponse.answer,
      citations: latestResponse.citations,
      generated_at: String(latestResponse.metrics.generated_at ?? ""),
    };
    const blob = kind === "pdf" ? await api.exportPdf(payload) : await api.exportPptx(payload);
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${payload.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "") || "groundedbi-chat-export"}.${kind}`;
    anchor.click();
    window.URL.revokeObjectURL(url);
  }

  const latestResponse = [...messages].reverse().find((entry) => entry.response)?.response ?? null;

  return (
    <section className="page-stack">
      <div className="section-heading">
        <div>
          <span className="panel__eyebrow">Threaded assessment</span>
          <h2>Continue the HR policy conversation with grounded follow-up questions.</h2>
          <p>Keep the answer moving forward while the latest citations and evidence stay visible beside the thread.</p>
        </div>
        <div className="section-heading__actions">
          <button className="ghost-button button--small" onClick={resetConversation}>
            New conversation
          </button>
        </div>
      </div>

      <div className="chat-layout">
        <aside className="page-stack chat-sidebar">
          {error ? <div className="error-banner">{error}</div> : null}
          <ProgressCard
            active={loading}
            title="Continuing the assessment"
            statuses={[
              "Parsing the follow-up and carrying forward the thread context...",
              "Rewriting the follow-up into a standalone retrieval question...",
              groundingEnabled ? "Searching evidence and grounding the reply..." : "Searching evidence and drafting the reply...",
              "Preparing the updated answer and inline citations...",
            ]}
            stalledLabel="The follow-up is still processing. You can retry, narrow the date filter, or start a fresh conversation."
            actionLabel="Reset conversation"
            onAction={resetConversation}
            stalledAfterMs={11000}
            durationMs={16000}
          />

          {latestResponse ? (
            <>
              <div className="button-row">
                <button className="ghost-button button--small" type="button" onClick={() => void handleExport("pdf")}>
                  Export PDF
                </button>
                <button className="ghost-button button--small" type="button" onClick={() => void handleExport("pptx")}>
                  Export PPTX
                </button>
              </div>
              <AnswerTelemetry response={latestResponse} />

              <section className="panel">
                <div className="panel__header">
                  <div>
                    <span className="panel__eyebrow">Latest evidence</span>
                    <h2>Citations for current answer</h2>
                  </div>
                </div>
                <div className="citation-list">
                  {latestResponse.citations.map((citation, index) => (
                    <CitationCard
                      key={`latest-${citation.doc_name}-${index}`}
                      citation={citation}
                      index={index + 1}
                    />
                  ))}
                </div>
              </section>

            </>
          ) : (
            <section className="panel empty-state">
              Latest answer diagnostics, citations, and retrieval trace will appear here once the conversation starts.
            </section>
          )}
        </aside>

        <div className="page-stack chat-main">
          <section className="panel chat-thread">
            {messages.length === 0 ? (
              <div className="empty-state">
                Start with a broad question, then pressure-test the assessment with targeted follow-ups.
              </div>
            ) : (
              messages.map((entry) => (
                <article className={`chat-message-row chat-message-row--${entry.role}`} key={entry.id}>
                  <div className={`chat-bubble chat-bubble--${entry.role}`}>
                    <div className="chat-bubble__meta">
                      <strong>{entry.role === "user" ? "You" : "GroundedBI"}</strong>
                      {entry.response ? <small>Answer recorded</small> : <small>User prompt</small>}
                    </div>
                    {entry.response ? (
                      <StructuredMarkdown answer={entry.response.answer} />
                    ) : (
                      <p className="chat-bubble__plain">{entry.text}</p>
                    )}
                  </div>
                  {entry.response ? (
                    <div className="chat-artifact">
                      <div className="chat-artifact__section">
                        <small className="chat-bubble__trace-label">Query rewrite used for retrieval</small>
                        <strong className="chat-artifact__rewrite">{entry.response.retrieval_question}</strong>
                      </div>
                      {entry.response.citations.length ? (
                        <div className="chat-artifact__section">
                          <small className="chat-bubble__citations-label">Evidence cards</small>
                          <div className="citation-list">
                            {entry.response.citations.map((citation, index) => (
                              <CitationCard
                                key={`${entry.id}-${citation.doc_name}-${index}`}
                                citation={citation}
                                index={index + 1}
                              />
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </article>
              ))
            )}
            {loading ? (
              <article className="chat-message-row chat-message-row--assistant">
                <div className="chat-bubble chat-bubble--assistant chat-bubble--pending">
                  <div className="chat-bubble__meta">
                    <strong>GroundedBI</strong>
                    <small>Working</small>
                  </div>
                  <div className="skeleton-copy">
                    <span className="skeleton-row" />
                    <span className="skeleton-row" />
                    <span className="skeleton-row skeleton-row--short" />
                  </div>
                </div>
              </article>
            ) : null}
          </section>

          <form className="panel composer" onSubmit={handleSend}>
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Follow-up prompt</span>
                <h2>Continue the assessment</h2>
              </div>
            </div>
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={4}
              placeholder="Ask a follow-up question..."
            />
            <div className="composer__footer">
              <div className="composer__toggles">
                <label className="toggle toggle--trace">
                  <input
                    type="checkbox"
                    checked={groundingEnabled}
                    onChange={(event) => setGroundingEnabled(event.target.checked)}
                  />
                  Grounding pass
                </label>
              </div>
              <button type="submit" disabled={loading || !draft.trim()}>
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}
