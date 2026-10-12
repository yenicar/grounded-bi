import { useEffect, useMemo, useState } from "react";

interface ProgressCardProps {
  active: boolean;
  title: string;
  statuses: string[];
  stalledLabel?: string;
  durationMs?: number;
  stalledAfterMs?: number;
  actionLabel?: string;
  onAction?: () => void;
  realProgress?: number;
}

function easeOutCubic(value: number) {
  return 1 - (1 - value) ** 3;
}

export function ProgressCard({
  active,
  title,
  statuses,
  stalledLabel = "This is taking longer than expected.",
  durationMs = 14000,
  stalledAfterMs = 12000,
  actionLabel,
  onAction,
  realProgress,
}: ProgressCardProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => {
      setElapsed(Date.now() - started);
    }, 120);
    return () => window.clearInterval(timer);
  }, [active]);

  const percent = useMemo(() => {
    if (!active) {
      return 0;
    }
    if (typeof realProgress === "number") {
      return Math.round(realProgress);
    }
    const normalized = Math.min(elapsed / durationMs, 1);
    return Math.min(92, Math.round(easeOutCubic(normalized) * 92));
  }, [active, durationMs, elapsed, realProgress]);

  const status = useMemo(() => {
    if (!statuses.length) {
      return "Working...";
    }
    const normalized = Math.min(elapsed / Math.max(durationMs, 1), 0.999);
    const index = Math.min(statuses.length - 1, Math.floor(normalized * statuses.length));
    return statuses[index];
  }, [durationMs, elapsed, statuses]);

  const stalled = active && elapsed >= stalledAfterMs;

  if (!active) {
    return null;
  }

  return (
    <section className="panel progress-card" aria-live="polite">
      <div className="panel__header">
        <div>
          <span className="panel__eyebrow">In progress</span>
          <h2>{title}</h2>
          <p>{status}</p>
        </div>
        <strong className="progress-card__percent">{percent}%</strong>
      </div>
      <div className="progress-bar" aria-hidden="true">
        <div className="progress-bar__fill" style={{ width: `${percent}%` }} />
      </div>
      {stalled ? (
        <div className="progress-card__stalled">
          <span>{stalledLabel}</span>
          {actionLabel && onAction ? (
            <button className="ghost-button button--small" type="button" onClick={onAction}>
              {actionLabel}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
