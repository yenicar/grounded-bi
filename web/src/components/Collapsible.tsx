import type { ReactNode } from "react";
import { useState } from "react";
import { ChevronIcon } from "./Icons";

interface CollapsibleProps {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export function Collapsible({ title, eyebrow, children, defaultOpen = false }: CollapsibleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <section className={`panel ${isOpen ? "panel--open" : "panel--collapsed"}`}>
      <div
        className="panel__header collapsible__header"
        onClick={() => setIsOpen((current) => !current)}
        style={{ cursor: "pointer" }}
      >
        <div>
          {eyebrow ? <span className="panel__eyebrow">{eyebrow}</span> : null}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2>{title}</h2>
            <span className={`chevron ${isOpen ? "chevron--up" : "chevron--down"}`}>
              <ChevronIcon width={14} height={14} />
            </span>
          </div>
        </div>
      </div>
      {isOpen ? <div className="collapsible__content">{children}</div> : null}
    </section>
  );
}
