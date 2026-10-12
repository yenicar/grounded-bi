import type { CSSProperties, ComponentType, SVGProps } from "react";
import { useState } from "react";

import { CloseIcon, MenuIcon } from "./Icons";

type RadialAction = {
  id: string;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  angle: number;
  onClick: () => void;
};

export function RadialActionMenu({
  label,
  actions,
}: {
  label: string;
  actions: RadialAction[];
}) {
  const [open, setOpen] = useState(false);
  const TriggerIcon = open ? CloseIcon : MenuIcon;

  return (
    <div className={`radial-menu${open ? " radial-menu--open" : ""}`}>
      {actions.map((action) => {
        const Icon = action.icon;
        return (
          <button
            key={action.id}
            type="button"
            className="radial-menu__action"
            style={
              {
                "--radial-angle": `${action.angle}deg`,
              } as CSSProperties
            }
            title={action.label}
            aria-label={action.label}
            onClick={() => {
              setOpen(false);
              action.onClick();
            }}
          >
            <Icon width={16} height={16} />
            <span className="radial-menu__tooltip">{action.label}</span>
          </button>
        );
      })}
      <button
        type="button"
        className="radial-menu__trigger"
        aria-expanded={open}
        aria-label={label}
        onClick={() => setOpen((current) => !current)}
      >
        <TriggerIcon width={18} height={18} />
      </button>
    </div>
  );
}
