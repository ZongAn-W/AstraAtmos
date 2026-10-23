import { useState, useRef, useEffect, useCallback } from 'react';
import C from '../constants/colors';
import { useT } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import { useSettings } from '../contexts/SettingsContext';
import { useToast } from '../contexts/ToastContext';
import ConfirmDialog from './ConfirmDialog';
import ChangePasswordModal from './ChangePasswordModal';
import NotificationPanel from './NotificationPanel';
import BrandMark from './BrandMark';
import NotificationsNoneRoundedIcon from '@mui/icons-material/NotificationsNoneRounded';
import { Button } from './ui/Controls';
import { getPendingReviews, getUnreadCount } from '../services/api';
import { NOTIFICATION_REFRESH_EVENT } from '../notifications/notificationEvents';

const NAV_IDS = ['home', 'overview', 'training', 'predict', 'explore', 'ai', 'about'];


function NavUserEntry({ t, isLight, onOpenAdmin, onOpenFeedback, pendingCount }) {
  const { user, logout, openAuthModal } = useAuth();
  const { showToast } = useToast();
  const [dropOpen, setDropOpen] = useState(false);
  const [logoutConfirm, setLogoutConfirm] = useState(false);
  const [changePwdOpen, setChangePwdOpen] = useState(false);
  const [hovLogin, setHovLogin] = useState(false);
  const [hovAdmin, setHovAdmin] = useState(false);
  const [hovFeedback, setHovFeedback] = useState(false);
  const wrapRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    if (!dropOpen) return;
    const handler = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setDropOpen(false);
    };
    const timer = setTimeout(() => document.addEventListener('mousedown', handler), 0);
    return () => { clearTimeout(timer); document.removeEventListener('mousedown', handler); };
  }, [dropOpen]);

  useEffect(() => {
    if (!dropOpen) return;
    wrapRef.current?.querySelector('[data-user-menu] button')?.focus();
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') {
        setDropOpen(false);
        wrapRef.current?.querySelector('[aria-expanded]')?.focus();
      }
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [dropOpen]);

  const isAdmin = user?.role === 'admin';
  const closeMenu = () => {
    wrapRef.current?.querySelector('[aria-expanded]')?.focus();
    setDropOpen(false);
  };
  const L = isLight;
  const dropBg     = L ? 'var(--bg-card-strong)' : 'var(--bg-card-strong)';
  const dropBorder = L ? 'var(--border)'         : 'var(--border)';
  const dropShadow = L
    ? '0 18px 44px rgba(15,23,42,0.12), 0 4px 14px rgba(15,23,42,0.08)'
    : '0 18px 44px rgba(0,0,0,0.34), 0 4px 14px rgba(0,0,0,0.20)';
  const labelClr   = 'var(--text)';
  const dimClr     = 'var(--text-60)';
  const hoverBg    = 'var(--bg-muted)';
  const divClr     = 'var(--border)';

  if (!user) {
    return (
      <div style={{ width: 130, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="quiet"
          onClick={() => openAuthModal('login')}
          onMouseEnter={() => setHovLogin(true)}
          onMouseLeave={() => setHovLogin(false)}
          style={{
            background: 'none', border: 'none',
            cursor: 'pointer', padding: '4px 0',
            color: C.brand,
            fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 600,
            textDecoration: hovLogin ? 'underline' : 'none',
            textUnderlineOffset: 3,
            transition: 'color 0.18s, text-decoration 0.18s',
            fontFamily: 'inherit',
            letterSpacing: 0,
          }}
        >
          {t('auth.menuLogin')}
        </Button>
      </div>
    );
  }

  const initial = (user?.username || user?.email || "?")[0]?.toUpperCase() || "?";

  return (
    <>
      <div ref={wrapRef} style={{ width: 130, display: 'flex', justifyContent: 'flex-end', position: 'relative' }}>
        {/* Avatar button */}
        <button
          type="button"
          className="nav-user-button"
          aria-expanded={dropOpen}
          aria-label={user.username || user.email}
          onClick={() => setDropOpen(v => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: dropOpen ? 'var(--bg-muted-strong)' : 'transparent',
            border: `1px solid ${dropOpen ? 'var(--border-strong)' : 'transparent'}`, borderRadius: 10, padding: '5px 10px 5px 6px',
            cursor: 'pointer', transition: 'background 0.15s',
          }}
        >
          {/* Avatar circle */}
          <div style={{
            width: 28, height: 28, borderRadius: '50%',
            background: 'var(--surface-3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 'calc(12px * var(--font-scale, 1))', fontWeight: 700, color: C.brand }}>{initial}</span>
          </div>
          {/* Username */}
          <div style={{ textAlign: 'left', maxWidth: 76, overflow: 'hidden' }}>
            <div style={{
              fontSize: 'calc(11px * var(--font-scale, 1))', fontWeight: 600, color: C.ice,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              lineHeight: 1.2,
            }}>
              {user.username || user.email}
            </div>
            {isAdmin && (
              <div style={{
                display: 'inline-block', fontSize: 'calc(11px * var(--font-scale, 1))', fontWeight: 600,
                color: C.action, letterSpacing: 0,
                background: 'rgba(199,91,57,0.10)',
                borderRadius: 4, padding: '1px 5px', marginTop: 1,
                lineHeight: 1.5,
              }}>
                {t('auth.roleAdmin').toUpperCase()}
              </div>
            )}
          </div>
        </button>

        {/* Dropdown */}
        {dropOpen && (
          <div data-user-menu className="nav-user-menu" style={{
            position: 'absolute', top: '100%', right: 0, marginTop: 6,
            width: 220,
            background: dropBg,
            border: `1px solid ${dropBorder}`,
            borderRadius: 11,
            boxShadow: dropShadow,
            backdropFilter: 'blur(18px)', WebkitBackdropFilter: 'blur(18px)',
            padding: '6px 0',
            zIndex: 3000,
          }}>
            {/* Email display */}
            <div style={{ padding: '8px 16px 10px' }}>
              <div style={{ fontSize: 'calc(11px * var(--font-scale, 1))', color: dimClr, wordBreak: 'break-all' }}>{user.email}</div>
            </div>
            <div style={{ height: 1, background: divClr, margin: '0 10px 4px' }} />

            {/* Admin review — only for admin */}
            {isAdmin && (
              <>
                <button type="button"
                  onClick={() => { closeMenu(); onOpenAdmin?.(); }}
                  onMouseEnter={() => setHovAdmin(true)}
                  onMouseLeave={() => setHovAdmin(false)}
                  style={{
                    padding: '9px 16px', cursor: 'pointer',
                    background: hovAdmin ? hoverBg : 'transparent',
                    transition: 'background 0.1s',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  }}
                >
                  <span style={{ fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 500, color: labelClr, userSelect: 'none' }}>
                    {t('admin.menuItem')}
                  </span>
                  {pendingCount > 0 && (
                    <span style={{
                      minWidth: 18, height: 18,
                      background: C.mars, borderRadius: 9,
                      fontSize: 'calc(10px * var(--font-scale, 1))', fontWeight: 700, color: '#fff',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      padding: '0 5px', lineHeight: 1, flexShrink: 0,
                    }}>
                      {pendingCount > 99 ? '99+' : pendingCount}
                    </span>
                  )}
                </button>
                <button type="button"
                  onClick={() => { closeMenu(); onOpenFeedback?.(); }}
                  onMouseEnter={() => setHovFeedback(true)}
                  onMouseLeave={() => setHovFeedback(false)}
                  style={{
                    padding: '9px 16px', cursor: 'pointer',
                    background: hovFeedback ? hoverBg : 'transparent',
                    transition: 'background 0.1s',
                  }}
                >
                  <span style={{ fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 500, color: labelClr, userSelect: 'none' }}>
                    {t('feedback.adminMenuItem')}
                  </span>
                </button>
                <div style={{ height: 1, background: divClr, margin: '4px 10px' }} />
              </>
            )}

            {/* Change password */}
            <DropItem
              label={t('auth.changePassword')}
              onClick={() => { closeMenu(); setChangePwdOpen(true); }}
              hoverBg={hoverBg} color={labelClr}
            />

            <div style={{ height: 1, background: divClr, margin: '4px 10px' }} />

            {/* Logout */}
            <DropItem
              label={t('auth.menuLogout')}
              onClick={() => { closeMenu(); setLogoutConfirm(true); }}
              hoverBg={hoverBg} color={C.mars}
            />
          </div>
        )}
      </div>

      {/* Logout confirm */}
      {logoutConfirm && (
        <ConfirmDialog
          title={t('auth.logoutConfirmTitle')}
          message={t('auth.logoutConfirmMsg')}
          confirmLabel={t('auth.logoutConfirmBtn')}
          cancelLabel={t('auth.cancelBtn')}
          onConfirm={() => {
            setLogoutConfirm(false);
            logout();
            showToast(t('auth.toastLoggedOut'), 'success');
          }}
          onCancel={() => setLogoutConfirm(false)}
        />
      )}

      {/* Change password modal */}
      {changePwdOpen && <ChangePasswordModal onClose={() => setChangePwdOpen(false)} />}
    </>
  );
}

function DropItem({ label, onClick, hoverBg, color }) {
  const [hov, setHov] = useState(false);
  return (
    <button type="button"
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        padding: '9px 16px', cursor: 'pointer',
        background: hov ? hoverBg : 'transparent',
        transition: 'background 0.1s',
      }}
    >
      <span style={{ fontSize: 'calc(13px * var(--font-scale, 1))', fontWeight: 500, color, userSelect: 'none' }}>{label}</span>
    </button>
  );
}

export default function Navbar({ current, onChange, onOpenAdmin, onOpenFeedback, pendingRefreshSignal }) {
  const t = useT();
  const { settings } = useSettings();
  const { user } = useAuth();
  const isLight = settings.theme === 'light';
  const isAdmin = user?.role === 'admin';

  // 待审核数量（仅 admin 用户拉取）
  const [pendingCount, setPendingCount] = useState(0);

  // 活动项下划指示条：跟随当前页面/语言/字号滑动
  const linksRef = useRef(null);
  const linkRefs = useRef(new Map());
  const [indicator, setIndicator] = useState({ left: 0, width: 0, ready: false });

  useEffect(() => {
    const measure = () => {
      const node = linkRefs.current.get(current);
      const container = linksRef.current;
      if (!node || !container) return;
      setIndicator({ left: node.offsetLeft, width: node.offsetWidth, ready: true });
    };
    measure();
    window.addEventListener('resize', measure);
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null;
    if (observer && linksRef.current) observer.observe(linksRef.current);
    return () => {
      window.removeEventListener('resize', measure);
      observer?.disconnect();
    };
  }, [current, settings.language, settings.fontScale]);

  const fetchPendingCount = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const data = await getPendingReviews();
      setPendingCount(Array.isArray(data) ? data.length : 0);
    } catch {
      // 静默失败，不影响导航栏渲染
    }
  }, [isAdmin]);

  useEffect(() => {
    fetchPendingCount();
  }, [fetchPendingCount, pendingRefreshSignal]);

  const handleOpenAdmin = useCallback(() => {
    onOpenAdmin?.();
    // 打开面板后延迟刷新计数（给面板操作留时间）
    setTimeout(fetchPendingCount, 2000);
  }, [onOpenAdmin, fetchPendingCount]);

  // 未读通知数量（已登录用户拉取，60s 轮询）
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifOpen, setNotifOpen] = useState(false);

  const fetchUnreadCount = useCallback(async () => {
    if (!user) return;
    try {
      const data = await getUnreadCount();
      setUnreadCount(data.count ?? 0);
    } catch {
      // 静默失败
    }
  }, [user]);

  useEffect(() => {
    fetchUnreadCount();
    if (!user) return;
    const timer = setInterval(fetchUnreadCount, 60000);
    return () => clearInterval(timer);
  }, [fetchUnreadCount, user]);

  useEffect(() => {
    window.addEventListener(NOTIFICATION_REFRESH_EVENT, fetchUnreadCount);
    return () => window.removeEventListener(NOTIFICATION_REFRESH_EVENT, fetchUnreadCount);
  }, [fetchUnreadCount]);

  return (
    <nav
      aria-label={settings.language === 'zh' ? '主导航' : 'Main navigation'}
      className={`nav-glass${current === 'home' ? ' home-nav' : ''}`}
      data-page={current}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0,
        zIndex: 2000,
        height: 70,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 48px',
      }}
    >
      {/* Logo — 语义按钮，键盘 Enter / Space 返回首页 */}
      <button
        type="button"
        className="home-nav__brand brand-home-button"
        onClick={() => onChange('home')}
        aria-label={`AstraAtmos · ${t('nav.home')}`}
      >
        <BrandMark />
        <div>
          <div style={{
            fontSize: 'calc(15px * var(--font-scale, 1))',
            fontWeight: 700,
            fontFamily: 'var(--font-display)',
            color: C.ice,
            letterSpacing: 0,
            lineHeight: 1.2,
          }}>
            AstraAtmos
          </div>
          <div className="brand-home-button__subtitle">
            {t('nav.subtitle')}
          </div>
        </div>
      </button>

      {/* Nav Links */}
      <div
        ref={linksRef}
        className="home-nav__links"
        style={{ display: 'flex', gap: 2, alignItems: 'center', marginLeft: 'auto', position: 'relative' }}
      >
        {indicator.ready && (
          <span
            className="nav-indicator"
            aria-hidden="true"
            data-nav-indicator="true"
            style={{ transform: `translateX(${indicator.left}px)`, width: indicator.width }}
          />
        )}
        {NAV_IDS.map((id) => (
          <button
            type="button"
            key={id}
            ref={(node) => {
              if (node) linkRefs.current.set(id, node);
              else linkRefs.current.delete(id);
            }}
            className={`nav-link${current === id ? ' is-active' : ''}`}
            aria-current={current === id ? 'page' : undefined}
            onClick={() => onChange(id)}
          >
            <span>
              {id === 'explore'
                ? (settings.language === 'zh' ? '数据管理' : 'Data Management')
                : t(`nav.${id}`)}
            </span>
          </button>
        ))}

      </div>

      {/* Right — bell + user entry */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 24 }}>
        {/* Bell icon — only for logged-in users */}
        {user && (
          <div style={{ position: 'relative' }}>
            <button
              type="button"
              className="nav-user-button nav-notification-button"
              aria-label={t('notification.title')}
              aria-expanded={notifOpen}
              onClick={() => setNotifOpen(v => !v)}
              title={t('notification.title')}
              style={{
                background: notifOpen ? 'var(--bg-muted-strong)' : 'transparent',
                border: `1px solid ${notifOpen ? 'var(--border-strong)' : 'transparent'}`, borderRadius: 10, padding: '6px 8px',
                cursor: 'pointer', display: 'flex', alignItems: 'center',
                color: 'var(--text)',
                transition: 'background 0.15s',
              }}
            >
              <NotificationsNoneRoundedIcon sx={{ fontSize: 20 }} />
            </button>
            {unreadCount > 0 && (
              <span style={{
                position: 'absolute', top: 4, right: 4,
                width: 8, height: 8, borderRadius: '50%',
                background: C.mars, pointerEvents: 'none',
              }} />
            )}
          </div>
        )}
        <NavUserEntry t={t} isLight={isLight} onOpenAdmin={handleOpenAdmin} onOpenFeedback={onOpenFeedback} pendingCount={pendingCount} />
      </div>

      {/* Notification panel */}
      <NotificationPanel
        open={notifOpen}
        onClose={() => setNotifOpen(false)}
        onReadCountChange={fetchUnreadCount}
        onNavigate={onChange}
      />
    </nav>
  );
}
