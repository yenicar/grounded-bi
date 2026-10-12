import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { api, type AuthProviderName, type AuthSessionResponse, type CurrentUser } from "../api/client";

interface AuthContextValue {
  loading: boolean;
  authenticated: boolean;
  user: CurrentUser | null;
  availableProviders: AuthProviderName[];
  refresh: () => Promise<void>;
  login: (provider: AuthProviderName) => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

async function loadSession(): Promise<AuthSessionResponse> {
  return api.getAuthSession();
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [availableProviders, setAvailableProviders] = useState<AuthProviderName[]>([]);

  async function refresh() {
    setLoading(true);
    try {
      const session = await loadSession();
      setAuthenticated(session.authenticated);
      setUser(session.user);
      setAvailableProviders(session.available_providers);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  function login(provider: AuthProviderName) {
    window.location.href = api.loginUrl(provider);
  }

  async function logout() {
    const session = await api.logout();
    setAuthenticated(session.authenticated);
    setUser(session.user);
    setAvailableProviders(session.available_providers);
  }

  return (
    <AuthContext.Provider
      value={{
        loading,
        authenticated,
        user,
        availableProviders,
        refresh,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }
  return context;
}
