import type { Citation } from "../api/client";

function summarizeExcerpt(excerpt: string) {
  const trimmed = excerpt.trim();
  if (!trimmed) {
    return "Source excerpt is available, but no short synopsis could be derived.";
  }

  const firstSentence = trimmed.split(/(?<=[.!?])\s+/)[0] ?? trimmed;
  return firstSentence.length > 140 ? `${firstSentence.slice(0, 137)}...` : firstSentence;
}

export function CitationCard({ citation, index }: { citation: Citation; index: number }) {
  const excerpt = citation.excerpt || "No excerpt available for this citation.";
  const summary = summarizeExcerpt(excerpt);

  return (
    <details className="citation-card">
      <summary>
        <span className="citation-card__index">{String(index).padStart(2, "0")}</span>
        <span className="citation-card__meta">
          <strong>{citation.doc_name}</strong>
          <small>{citation.page_range}</small>
        </span>
        <span className="citation-card__toggle">Expand source</span>
      </summary>
      <div className="citation-card__body">
        <div className="citation-card__section">
          <small className="citation-card__label">Why this source matters</small>
          <p className="citation-card__summary">{summary}</p>
        </div>
        <details className="citation-card__nested">
          <summary className="citation-card__nested-toggle">Full excerpt</summary>
          <p className="citation-card__excerpt">{excerpt}</p>
        </details>
      </div>
    </details>
  );
}
