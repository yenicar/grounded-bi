import type { ComponentType, CSSProperties, SVGProps } from "react";
import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

import {
  AskIcon,
  ChatIcon,
  CloseIcon,
  LibraryIcon,
  MenuIcon,
  SettingsIcon,
  SignOutIcon,
  ThemeIcon,
} from "./Icons";

type DockAction = {
  id: string;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  onClick: () => void;
};

const navItems = [
  { to: "/library", label: "Library", icon: LibraryIcon },
  { to: "/ask", label: "Ask", icon: AskIcon },
  { to: "/chat", label: "Chat", icon: ChatIcon },
];

const navAngles = [169, 203, 237];
const settingsAngle = 271;

export function FloatingCommandDock({
  onTheme,
  onLogout,
}: {
  onTheme: () => void;
  onLogout: () => void;
}) {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsActions: DockAction[] = [
    { id: "theme", label: "Theme", icon: ThemeIcon, onClick: onTheme },
    { id: "logout", label: "Sign out", icon: SignOutIcon, onClick: onLogout },
  ];
  const TriggerIcon = open ? CloseIcon : MenuIcon;

  return (
    <div className={`floating-radial-nav${open ? " floating-radial-nav--open" : ""}`} aria-label="Navigation menu">
      {navItems.map((item, index) => {
        const Icon = item.icon;
        const isActive = location.pathname === item.to || (item.to === "/library" && location.pathname === "/");
        return (
          <NavLink
            key={item.to}
            to={item.to}
            className={`floating-radial-nav__node${isActive ? " floating-radial-nav__node--active" : ""}`}
            title={item.label}
            aria-label={item.label}
            style={{ "--radial-angle": `${navAngles[index]}deg` } as CSSProperties}
            onClick={() => {
              setOpen(false);
              setSettingsOpen(false);
            }}
          >
            <Icon width={18} height={18} />
            <span className="floating-radial-nav__tooltip">{item.label}</span>
          </NavLink>
        );
      })}
      <button
        type="button"
        className={`floating-radial-nav__node floating-radial-nav__node--button${settingsOpen ? " floating-radial-nav__node--active" : ""}`}
        style={{ "--radial-angle": `${settingsAngle}deg` } as CSSProperties}
        title="Settings"
        aria-label="Settings"
        aria-expanded={settingsOpen}
        onClick={() => {
          setOpen(true);
          setSettingsOpen((current) => !current);
        }}
      >
        <SettingsIcon width={18} height={18} />
        <span className="floating-radial-nav__tooltip">Settings</span>
      </button>
      <button
        type="button"
        className="floating-radial-nav__trigger"
        onClick={() => {
          setOpen((current) => {
            const next = !current;
            setSettingsOpen(false);
            return next;
          });
        }}
        aria-expanded={open}
        aria-label="Open navigation menu"
      >
        <TriggerIcon width={20} height={20} />
      </button>

      {settingsActions.map((action, index) => {
        const Icon = action.icon;
        return (
          <button
            key={action.id}
            type="button"
            className={`floating-radial-nav__utility-node${settingsOpen ? " floating-radial-nav__utility-node--open" : ""}`}
            style={{ "--utility-angle": `${252 + index * 24}deg` } as CSSProperties}
            title={action.label}
            aria-label={action.label}
            onClick={() => {
              setOpen(false);
              setSettingsOpen(false);
              action.onClick();
            }}
          >
            <Icon width={18} height={18} />
            <span className="floating-radial-nav__tooltip">{action.label}</span>
          </button>
        );
      })}
    </div>
  );
}
