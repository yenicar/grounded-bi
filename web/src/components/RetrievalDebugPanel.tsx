import type { RetrievalDebug } from "../api/client";

export function RetrievalDebugPanel({ debug }: { debug: RetrievalDebug | null | undefined }) {
  if (!debug) {
    return null;
  }

  return (
    <section className="panel panel--secondary">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">Retrieval trace</span>
          <h2>Selection trace</h2>
        </div>
      </div>

      <div className="debug-group">
        <div className="debug-group__header">
          <strong>Used in final answer</strong>
          <small>Contexts that survived prompt budgeting and informed generation.</small>
        </div>
        <div className="debug-table">
          <div className="debug-table__head">
            <span>Doc</span>
            <span>Scores</span>
            <span>Signals</span>
            <span>Reason</span>
          </div>
          {debug.selected.map((item) => (
            <div className="debug-table__row" key={`${item.chunk_id}-${item.citation ?? item.doc_name}`}>
              <span>
                <strong>{item.doc_name}</strong>
                <small>
                  pp. {item.page_start}-{item.page_end}
                </small>
              </span>
              <span>
                <strong>{item.effective_score.toFixed(3)}</strong>
                <small>
                  fusion {item.fusion_score.toFixed(3)} | sem {item.similarity_score.toFixed(3)} | kw{" "}
                  {item.keyword_score.toFixed(3)}
                </small>
              </span>
              <span>
                <strong>{item.retrieval_channels.join(", ") || "none"}</strong>
                <small>{item.matched_terms.join(", ") || "No direct terms matched"}</small>
              </span>
              <span>{item.selection_reason}</span>
            </div>
          ))}
        </div>
      </div>

      {debug.retrieved?.length ? (
        <details className="citation-card citation-card--compact">
          <summary>
            <span className="citation-card__index">{debug.retrieved.length}</span>
            <span>
              <strong>Retrieved but trimmed</strong>
              <small>Found by retrieval before the final prompt budget was applied.</small>
            </span>
          </summary>
          <div className="debug-table">
            <div className="debug-table__head">
              <span>Doc</span>
              <span>Scores</span>
              <span>Signals</span>
              <span>Reason</span>
            </div>
            {debug.retrieved.map((item) => (
              <div className="debug-table__row" key={`retrieved-${item.chunk_id}-${item.citation ?? item.doc_name}`}>
                <span>
                  <strong>{item.doc_name}</strong>
                  <small>
                    pp. {item.page_start}-{item.page_end}
                  </small>
                </span>
                <span>
                  <strong>{item.effective_score.toFixed(3)}</strong>
                  <small>
                    fusion {item.fusion_score.toFixed(3)} | sem {item.similarity_score.toFixed(3)} | kw{" "}
                    {item.keyword_score.toFixed(3)}
                  </small>
                </span>
                <span>
                  <strong>{item.retrieval_channels.join(", ") || "none"}</strong>
                  <small>{item.matched_terms.join(", ") || "No direct terms matched"}</small>
                </span>
                <span>{item.selection_reason}</span>
              </div>
            ))}
          </div>
        </details>
      ) : null}
    </section>
  );
}
