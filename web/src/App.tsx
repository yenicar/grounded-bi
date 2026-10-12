import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { useAuth } from "./auth/AuthProvider";
import { FloatingCommandDock } from "./components/FloatingCommandDock";
import { AskPage } from "./pages/Ask";
import { ChatPage } from "./pages/Chat";
import { LibraryPage } from "./pages/Library";
import { useTheme } from "./theme/ThemeProvider";

const navItems = [
  { to: "/library", label: "Library", meta: "Corpus registry" },
  { to: "/ask", label: "Ask", meta: "HR assistant" },
  { to: "/chat", label: "Chat", meta: "Guided follow-up" },
];

export default function App() {
  const location = useLocation();
  const { loading, authenticated, user, availableProviders, login, logout } = useAuth();
  const { selectedTheme, setSelectedTheme, options } = useTheme();
  const activeNav = navItems.find((item) => location.pathname === item.to) ?? navItems[0];
  const selectedThemeOption = options.find((option) => option.id === selectedTheme) ?? options[0];
  const nextTheme = options[(options.findIndex((option) => option.id === selectedTheme) + 1) % options.length];

  if (loading) {
    return (
      <div className="app-shell">
        <section className="panel empty-state">Restoring analyst session...</section>
      </div>
    );
  }

  if (!authenticated) {
    return (
      <div className="app-shell">
        <section className="panel auth-panel">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">Authentication required</span>
              <h2>Sign in to access the HR assistant workspace.</h2>
              <p>
                GroundedBI now protects corpus, query, and evaluation routes behind
                OAuth-backed session auth.
              </p>
            </div>
          </div>

          {availableProviders.length ? (
            <div className="auth-panel__actions">
              {availableProviders.map((provider) => (
                <button key={provider} onClick={() => login(provider)}>
                  {provider === "dev"
                    ? "Continue in local dev mode"
                    : `Sign in with ${provider === "google" ? "Google" : "GitHub"}`}
                </button>
              ))}
            </div>
          ) : (
            <div className="error-banner">
              No OAuth providers are configured yet. Set the Phase 4 auth environment variables
              for Google and/or GitHub to enable sign-in.
            </div>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="workspace-floating-chrome">
        <div className="workspace-pill">
          <span className="eyebrow">GroundedBI / HR Assistant</span>
          <strong>{activeNav.label}</strong>
          <small>{activeNav.meta}</small>
        </div>
      </div>

      <main className="page-frame">
        <Routes>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/eval" element={<Navigate to="/library" replace />} />
        </Routes>
      </main>

      <FloatingCommandDock
        onTheme={() => setSelectedTheme(nextTheme.id)}
        onLogout={() => void logout()}
      />
    </div>
  );
}
