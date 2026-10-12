import type { DocumentRecord } from "../api/client";

function formatDate(value: string) {
  if (!value) {
    return "Unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function describeDocument(document: DocumentRecord) {
  const summaryParts = [
    document.doc_type ? `${document.doc_type} format` : null,
    document.doc_author ? `author noted as ${document.doc_author}` : null,
    document.doc_date ? `dated ${document.doc_date}` : null,
    document.table_count ? `${document.table_count} extracted table${document.table_count === 1 ? "" : "s"}` : "no extracted tables",
  ].filter(Boolean);

  return summaryParts.length
    ? `${summaryParts.join(", ")}.`
    : "Indexed document with limited metadata available.";
}

export function DocumentCard({
  document,
  deleting,
  onDelete,
}: {
  document: DocumentRecord;
  deleting: boolean;
  onDelete: () => void;
}) {
  const summary = describeDocument(document);

  return (
    <details className="document-card document-card--registry document-card--accordion">
      <summary className="document-card__summary">
        <div className="document-card__top">
          <div className="document-card__title">
            <span className="format-pill">{(document.doc_format ?? "unknown").toUpperCase()}</span>
            <h3>{document.doc_name}</h3>
            <p className="document-card__quiet">{summary}</p>
          </div>
          <div className="document-card__actions">
            <span className="document-card__expand">Expand summary</span>
            <button className="ghost-button ghost-button--danger button--small" onClick={onDelete} disabled={deleting}>
              {deleting ? "Removing..." : "Delete"}
            </button>
          </div>
        </div>
      </summary>

      <div className="document-card__body">
        <div className="document-grid">
          <div>
            <span>Pages</span>
            <strong>{document.page_count}</strong>
          </div>
          <div>
            <span>Tables</span>
            <strong>{document.table_count}</strong>
          </div>
          <div>
            <span>Last ingested</span>
            <strong>{formatDate(document.last_ingested_at)}</strong>
          </div>
        </div>

        <div className="document-card__meta-grid">
          <div>
            <span>Type</span>
            <strong>{document.doc_type ?? "Unknown"}</strong>
          </div>
          <div>
            <span>Date</span>
            <strong>{document.doc_date ?? "Unknown"}</strong>
          </div>
          <div>
            <span>Author</span>
            <strong>{document.doc_author ?? "Unknown"}</strong>
          </div>
        </div>

        {document.doc_summary ? (
          <div className="document-card__ai-summary">
            <span className="panel__eyebrow">AI Content Briefing</span>
            <p>{document.doc_summary}</p>
          </div>
        ) : null}

        <div className="document-card__footer">
          {(document.warnings ?? []).length > 0 ? (
            <div className="warning-list">
              {(document.warnings ?? []).map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </div>
          ) : (
            <span className="document-card__quiet">No extraction warnings.</span>
          )}
        </div>
      </div>
    </details>
  );
}
