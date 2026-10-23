import { useState, useEffect, useRef, useId } from 'react';
import { flushSync } from 'react-dom';
import { withThemeReveal } from './ui/motion';
import SettingsOutlined from '@mui/icons-material/SettingsOutlined';
import ChevronRightOutlined from '@mui/icons-material/ChevronRightOutlined';
import ChevronLeftOutlined from '@mui/icons-material/ChevronLeftOutlined';
import CheckOutlined from '@mui/icons-material/CheckOutlined';
import { useSettings } from '../contexts/SettingsContext';
import { useAuth } from '../contexts/AuthContext';
import { useToast } from '../contexts/ToastContext';
import { useT } from '../i18n';
import ConfirmDialog from './ConfirmDialog';
import FeedbackModal from './FeedbackModal';
import PerformanceMonitor from './PerformanceMonitor';
import './ui/overlay.css';

const COLORMAP_IDS = ['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'jet', 'rdbu'];

export default function SettingsFab({ onOpenSettings }) {
  const { settings, updateSetting } = useSettings();
  const { user, logout, openAuthModal } = useAuth();
  const { showToast } = useToast();
  const t = useT();
  const [menuOpen, setMenuOpen] = useState(false);
  const [logoutConfirm, setLogoutConfirm] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [perfMonitor, setPerfMonitor] = useState(false);
  const [activeSubmenu, setActiveSubmenu] = useState(null);
  const [showTooltip, setShowTooltip] = useState(false);
  const containerRef = useRef(null);
  const triggerRef = useRef(null);
  const mainMenuRef = useRef(null);
  const submenuRef = useRef(null);
  const submenuTriggers = useRef({});
  const focusSubmenuRef = useRef(false);
  const menuId = useId();
  const tooltipId = useId();
  const zh = settings.language === 'zh';
  const tooltipText = zh ? '设置' : 'Settings';

  useEffect(() => {
    if (!menuOpen) { setActiveSubmenu(null); return undefined; }
    mainMenuRef.current?.querySelector('button')?.focus();
    const onPointerDown = event => {
      if (!containerRef.current?.contains(event.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [menuOpen]);

  useEffect(() => {
    if (activeSubmenu && focusSubmenuRef.current) {
      submenuRef.current?.querySelector('[aria-checked="true"]')?.focus();
      focusSubmenuRef.current = false;
    }
  }, [activeSubmenu]);

  const closeMenu = (restoreFocus = true) => {
    setMenuOpen(false);
    setActiveSubmenu(null);
    if (restoreFocus) triggerRef.current?.focus();
  };
  const openSubmenu = (key, focus = false) => {
    focusSubmenuRef.current = focus;
    setActiveSubmenu(key);
    if (focus && activeSubmenu === key) submenuRef.current?.querySelector('[aria-checked="true"]')?.focus();
  };
  const closeSubmenu = () => {
    submenuTriggers.current[activeSubmenu]?.focus();
    setActiveSubmenu(null);
  };
  const onMenuKeyDown = event => {
    if (!menuOpen) return;
    const inSubmenu = event.target.closest('[data-submenu]');
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      if (activeSubmenu) closeSubmenu(); else closeMenu();
      return;
    }
    if (event.key === 'Tab') {
      // Continue the normal tab order from the trigger after closing its menu.
      closeMenu();
      return;
    }
    if (event.key === 'ArrowLeft' && inSubmenu) { event.preventDefault(); closeSubmenu(); return; }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const menu = inSubmenu ? submenuRef.current : mainMenuRef.current;
    const buttons = Array.from(menu?.querySelectorAll('button:not(:disabled)') || []);
    const current = buttons.indexOf(document.activeElement);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (current + (event.key === 'ArrowUp' ? -1 : 1) + buttons.length) % buttons.length;
    buttons[index]?.focus();
  };

  const mainItems = [
    { key: 'language', label: t('settings.language.label'), value: zh ? '中文' : 'English' },
    { key: 'theme', label: t('settings.theme.label'), value: t('settings.theme.' + settings.theme) },
    { key: 'colormap', label: t('settings.colormap.label'), value: settings.colormap.charAt(0).toUpperCase() + settings.colormap.slice(1) },
  ];
  const subOptions = {
    language: [{ value: 'zh', label: '中文' }, { value: 'en', label: 'English' }],
    theme: [{ value: 'dark', label: t('settings.theme.dark') }, { value: 'light', label: t('settings.theme.light') }],
    colormap: COLORMAP_IDS.map(value => ({ value, label: value.charAt(0).toUpperCase() + value.slice(1) })),
  };

  return (
    <div className="settings-fab" ref={containerRef} onKeyDown={onMenuKeyDown}
      onBlur={event => { if (menuOpen && event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) closeMenu(false); }}>
      {menuOpen && (
        <div className="av-quick-menu-wrap">
          <div id={menuId} ref={mainMenuRef} className="av-quick-menu" role="menu" aria-label={tooltipText}>
            {mainItems.map(item => (
              <button key={item.key} ref={element => { submenuTriggers.current[item.key] = element; }}
                type="button" tabIndex={-1} className="av-quick-item" role="menuitem" aria-haspopup="menu"
                aria-expanded={activeSubmenu === item.key} aria-controls={menuId + '-' + item.key}
                onMouseEnter={() => openSubmenu(item.key)} onClick={() => openSubmenu(item.key, true)}
                onKeyDown={event => {
                  if (event.key === 'ArrowRight') { event.preventDefault(); openSubmenu(item.key, true); }
                }}>
                <span>{item.label}</span>
                <span className="av-quick-item-value">{item.value}<ChevronRightOutlined /></span>
              </button>
            ))}
            <div className="av-quick-separator" role="separator" />
            <button type="button" tabIndex={-1} role="menuitem" className="av-quick-item" onMouseEnter={() => setActiveSubmenu(null)}
              onClick={() => { closeMenu(); onOpenSettings(); }}>{zh ? '更多设置' : 'More Settings'}</button>
            <button type="button" tabIndex={-1} role="menuitem" className="av-quick-item"
              onClick={() => { closeMenu(); setFeedbackOpen(true); }}>{t('feedback.menuItem')}</button>
            <button type="button" tabIndex={-1} role="menuitemcheckbox" aria-checked={perfMonitor} className="av-quick-item"
              onClick={() => setPerfMonitor(value => !value)}>
              <span>{t('settings.perfMonitor')}</span><span className="av-quick-item-value">{perfMonitor ? 'ON' : 'OFF'}</span>
            </button>
            <div className="av-quick-separator" role="separator" />
            {user ? (
              <>
                <div className="av-quick-user">
                  <span className="av-quick-avatar" aria-hidden="true">{(user.username || user.email || '?')[0].toUpperCase()}</span>
                  <div style={{ minWidth: 0 }}>
                    <div className="av-quick-user-name">{user.username || user.email}</div>
                    <div className="av-helper">{user.role === 'admin' ? t('auth.roleAdmin') : t('auth.roleUser')}</div>
                  </div>
                </div>
                <button type="button" tabIndex={-1} role="menuitem" className="av-quick-item av-quick-item--danger"
                  onClick={() => { closeMenu(); setLogoutConfirm(true); }}>{t('auth.menuLogout')}</button>
              </>
            ) : (
              <button type="button" tabIndex={-1} role="menuitem" className="av-quick-item"
                onClick={() => { closeMenu(); openAuthModal('login'); }}>{t('auth.menuLogin')}</button>
            )}
          </div>
          {activeSubmenu && (
            <div id={menuId + '-' + activeSubmenu} ref={submenuRef} data-submenu
              className="av-quick-menu av-quick-submenu" role="menu"
              aria-label={mainItems.find(item => item.key === activeSubmenu)?.label}>
              <button type="button" tabIndex={-1} role="menuitem" className="av-quick-item" onClick={closeSubmenu}>
                <ChevronLeftOutlined fontSize="small" /><span>{zh ? '返回' : 'Back'}</span>
              </button>
              {subOptions[activeSubmenu].map(option => (
                <button key={option.value} type="button" tabIndex={-1} role="menuitemradio"
                  aria-checked={settings[activeSubmenu] === option.value} className="av-quick-item"
                  onClick={(event) => {
                    const rect = event.currentTarget.getBoundingClientRect();
                    const apply = () => flushSync(() => updateSetting(activeSubmenu, option.value));
                    if (activeSubmenu === 'theme') {
                      withThemeReveal(apply, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
                    } else {
                      apply();
                    }
                  }}>
                  <span>{option.label}</span>
                  {settings[activeSubmenu] === option.value && <CheckOutlined fontSize="small" />}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {showTooltip && !menuOpen && <div id={tooltipId} role="tooltip" className="av-quick-tooltip">{tooltipText}</div>}
      <button ref={triggerRef} type="button" className="av-quick-trigger" aria-label={tooltipText}
        title={tooltipText} aria-haspopup="menu" aria-expanded={menuOpen} aria-controls={menuId}
        aria-describedby={showTooltip && !menuOpen ? tooltipId : undefined}
        onClick={() => { if (menuOpen) closeMenu(); else setMenuOpen(true); setShowTooltip(false); }}
        onMouseEnter={() => setShowTooltip(true)} onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)} onBlur={() => setShowTooltip(false)}
        onKeyDown={event => {
          if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { event.preventDefault(); setMenuOpen(true); }
        }}>
        <SettingsOutlined />
      </button>
      <PerformanceMonitor visible={perfMonitor} />
      <FeedbackModal open={feedbackOpen} onClose={() => setFeedbackOpen(false)} />
      {logoutConfirm && (
        <ConfirmDialog title={t('auth.logoutConfirmTitle')} message={t('auth.logoutConfirmMsg')}
          confirmLabel={t('auth.logoutConfirmBtn')} cancelLabel={t('auth.cancelBtn')}
          onConfirm={() => { setLogoutConfirm(false); logout(); showToast(t('auth.toastLoggedOut'), 'success'); }}
          onCancel={() => setLogoutConfirm(false)} />
      )}
    </div>
  );
}
