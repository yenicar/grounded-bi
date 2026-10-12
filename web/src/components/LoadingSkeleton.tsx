export function AnalysisSkeleton() {
  return (
    <div className="page-stack" aria-hidden="true">
      <section className="panel">
        <div className="skeleton-copy">
          <span className="skeleton-row skeleton-row--short" />
          <span className="skeleton-row" />
          <span className="skeleton-row" />
        </div>
      </section>
      <section className="panel">
        <div className="skeleton-copy">
          <span className="skeleton-row" />
          <span className="skeleton-row" />
          <span className="skeleton-row skeleton-row--short" />
        </div>
      </section>
    </div>
  );
}

export function LibrarySkeleton() {
  return (
    <div className="library-layout" aria-hidden="true">
      <aside className="library-sidebar">
        <section className="panel">
          <div className="skeleton-copy">
            <span className="skeleton-row skeleton-row--short" />
            <span className="skeleton-row" />
            <span className="skeleton-row skeleton-row--short" />
          </div>
        </section>
      </aside>
      <section className="library-canvas panel panel--canvas">
        <div className="skeleton-copy">
          <span className="skeleton-row" />
          <span className="skeleton-row" />
          <span className="skeleton-row" />
        </div>
      </section>
      <aside className="library-sidebar library-sidebar--right">
        <section className="panel">
          <div className="skeleton-copy">
            <span className="skeleton-row skeleton-row--short" />
            <span className="skeleton-row skeleton-row--short" />
            <span className="skeleton-row skeleton-row--short" />
          </div>
        </section>
      </aside>
    </div>
  );
}

export function ReportSkeleton() {
  return (
    <div className="page-stack" aria-hidden="true">
      <section className="panel">
        <div className="skeleton-copy">
          <span className="skeleton-row skeleton-row--short" />
          <span className="skeleton-row skeleton-row--short" />
        </div>
      </section>
      <section className="panel">
        <div className="skeleton-copy">
          <span className="skeleton-row" />
          <span className="skeleton-row" />
          <span className="skeleton-row" />
        </div>
      </section>
    </div>
  );
}
