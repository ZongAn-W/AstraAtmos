import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import {
  apiLogin,
  apiRegister,
  apiGetMe,
  apiChangePassword,
  fetchDataInfo,
  fetchGlobeData,
} from '../services/api';
import {
  beginAuthenticatedPredictionSession,
  endAuthenticatedPredictionSession,
} from '../stores/authPredictionSession';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalTab, setAuthModalTab] = useState('login');
  const prewarmedKeysRef = useRef(new Set());

  useEffect(() => {
    const stored = localStorage.getItem('astraatmos_token');
    if (!stored) {
      endAuthenticatedPredictionSession();
      setIsLoading(false);
      return;
    }

    apiGetMe()
      .then((me) => {
        beginAuthenticatedPredictionSession(me.id);
        setUser(me);
        setToken(stored);
      })
      .catch(() => {
        endAuthenticatedPredictionSession();
        localStorage.removeItem('astraatmos_token');
      })
      .finally(() => setIsLoading(false));
  }, []);

  useEffect(() => {
    const handler = () => {
      endAuthenticatedPredictionSession();
      setUser(null);
      setToken(null);
    };
    window.addEventListener('astraatmos:logout', handler);
    return () => window.removeEventListener('astraatmos:logout', handler);
  }, []);

  useEffect(() => {
    if (!user?.id || !token) return;
    const warmKey = String(user.id);
    if (prewarmedKeysRef.current.has(warmKey)) return;
    prewarmedKeysRef.current.add(warmKey);

    // Keep login-time warmup lightweight.
    // Personal-source warmups are intentionally skipped here to avoid blocking
    // default-source pages immediately after login.
    const warmups = [
      fetchDataInfo({ dataSource: 'default' }),
      fetchGlobeData(27, 0, 'o3col', null, { dataSource: 'default' }),
    ];
    Promise.allSettled(warmups).catch(() => {
      // Do not block login UX on prewarm failure.
    });
  }, [user?.id, token]);

  const login = useCallback(async (email, password) => {
    const data = await apiLogin(email, password);
    beginAuthenticatedPredictionSession(data.user.id);
    localStorage.setItem('astraatmos_token', data.token);
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  const register = useCallback(async (email, username, password, verificationCode) => {
    await apiRegister(email, username, password, verificationCode);
    return login(email, password);
  }, [login]);

  const logout = useCallback(() => {
    endAuthenticatedPredictionSession();
    localStorage.removeItem('astraatmos_token');
    setToken(null);
    setUser(null);
  }, []);

  const changePassword = useCallback(async (oldPassword, newPassword) => {
    return apiChangePassword(oldPassword, newPassword);
  }, []);

  const openAuthModal = useCallback((tab = 'login') => {
    setAuthModalTab(tab);
    setAuthModalOpen(true);
  }, []);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
  }, []);

  return (
    <AuthContext.Provider value={{
      user,
      token,
      isLoading,
      authModalOpen,
      authModalTab,
      login,
      register,
      logout,
      changePassword,
      openAuthModal,
      closeAuthModal,
    }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
