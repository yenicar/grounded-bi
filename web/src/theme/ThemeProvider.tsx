import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type ThemeId = "system" | "obsidian-dark" | "signal-dark" | "paper-light" | "sand-light";

export interface ThemeOption {
  id: ThemeId;
  label: string;
  family: "system" | "dark" | "light";
  description: string;
}

const THEME_STORAGE_KEY = "groundedbi-theme";

const themeOptions: ThemeOption[] = [
  {
    id: "system",
    label: "System",
    family: "system",
    description: "Follows your operating system preference.",
  },
  {
    id: "obsidian-dark",
    label: "Obsidian",
    family: "dark",
    description: "Low-glare dark workstation palette with amber emphasis.",
  },
  {
    id: "signal-dark",
    label: "Signal",
    family: "dark",
    description: "Cooler tactical dark theme with cyan-green accents.",
  },
  {
    id: "paper-light",
    label: "Paper",
    family: "light",
    description: "High-clarity briefing theme for long reading sessions.",
  },
  {
    id: "sand-light",
    label: "Sand",
    family: "light",
    description: "Warm daylight theme with softer panels and restrained contrast.",
  },
];

interface ThemeContextValue {
  selectedTheme: ThemeId;
  resolvedTheme: Exclude<ThemeId, "system">;
  setSelectedTheme: (theme: ThemeId) => void;
  options: ThemeOption[];
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getSystemTheme(): Exclude<ThemeId, "system"> {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "obsidian-dark";
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "obsidian-dark" : "paper-light";
}

function readStoredTheme(): ThemeId {
  if (typeof window === "undefined") {
    return "system";
  }
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored && themeOptions.some((option) => option.id === stored)) {
    return stored as ThemeId;
  }
  return "system";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [selectedTheme, setSelectedThemeState] = useState<ThemeId>(readStoredTheme);
  const [systemTheme, setSystemTheme] = useState<Exclude<ThemeId, "system">>(getSystemTheme);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      setSystemTheme(media.matches ? "obsidian-dark" : "paper-light");
    };
    handleChange();
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  const resolvedTheme = selectedTheme === "system" ? systemTheme : selectedTheme;

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.dataset.themeSelection = selectedTheme;
    document.documentElement.style.colorScheme = resolvedTheme.endsWith("light") ? "light" : "dark";
  }, [resolvedTheme, selectedTheme]);

  function setSelectedTheme(theme: ThemeId) {
    setSelectedThemeState(theme);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
  }

  const value = useMemo<ThemeContextValue>(
    () => ({
      selectedTheme,
      resolvedTheme,
      setSelectedTheme,
      options: themeOptions,
    }),
    [resolvedTheme, selectedTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used inside ThemeProvider.");
  }
  return context;
}
