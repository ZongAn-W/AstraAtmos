import { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import { useToast } from '../contexts/ToastContext';
import { useSettings } from '../contexts/SettingsContext';
import C from '../constants/colors';

const EXIT_MS = 190;

function CheckIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path
        d="M2.5 7.5L5.5 10.5L12.5 3.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path
        d="M3.5 3.5L11.5 11.5M11.5 3.5L3.5 11.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ToastContent() {
  const { toast, hideToast } = useToast();
  const { settings } = useSettings();
  // 关闭后保留一份快照，等退场动画播放完再卸载
  const [snapshot, setSnapshot] = useState(null);
  const exitTimer = useRef(null);

  useEffect(() => {
    if (toast) {
      clearTimeout(exitTimer.current);
      setSnapshot(toast);
      return undefined;
    }
    if (!snapshot) return undefined;
    exitTimer.current = setTimeout(() => setSnapshot(null), EXIT_MS);
    return () => clearTimeout(exitTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast]);

  useEffect(() => () => clearTimeout(exitTimer.current), []);

  const active = toast || snapshot;
  if (!active) return null;

  const exiting = !toast && Boolean(snapshot);
  const isLight = settings.theme === 'light';
  const message = String(active.message ?? '').trim();
  if (!message) return null;

  const isSuccess = active.type === 'success';
  const accentColor = isSuccess ? C.blue : C.mars;
  const bgColor = isLight ? 'rgba(255,255,255,0.96)' : 'rgba(18,24,34,0.94)';
  const borderColor = isSuccess ? 'rgba(74,158,255,0.32)' : 'rgba(199,91,57,0.34)';
  const textColor = isLight ? '#1f2b3a' : '#eaf1ff';
  const shadow = isLight
    ? '0 8px 28px rgba(0,0,0,0.10), 0 2px 8px rgba(0,0,0,0.06)'
    : '0 8px 28px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.32)';

  return (
    <>
      <style>{`
        @keyframes _tslide {
          from { opacity: 0; transform: translateX(-50%) translateY(-12px) scale(0.98); }
          to   { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
        }
        @keyframes _tslideout {
          from { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
          to   { opacity: 0; transform: translateX(-50%) translateY(-10px) scale(0.98); }
        }
      `}</style>
      <div
        key={active.id}
        onClick={exiting ? undefined : hideToast}
        data-toast="true"
        data-toast-exiting={exiting ? 'true' : 'false'}
        style={{
          position: 'fixed',
          top: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 10000,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '11px 20px',
          background: bgColor,
          border: `1px solid ${borderColor}`,
          borderLeft: `3px solid ${accentColor}`,
          borderRadius: 10,
          boxShadow: shadow,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          cursor: exiting ? 'default' : 'pointer',
          pointerEvents: exiting ? 'none' : 'auto',
          animation: exiting
            ? `_tslideout ${EXIT_MS}ms cubic-bezier(0.4, 0, 0.2, 1) forwards`
            : '_tslide 0.2s ease-out',
          maxWidth: 460,
          whiteSpace: 'pre-wrap',
          userSelect: 'none',
        }}
      >
        <span style={{ color: accentColor, flexShrink: 0, display: 'flex', alignItems: 'center' }}>
          {isSuccess ? <CheckIcon /> : <XIcon />}
        </span>
        <span style={{ fontSize: 'calc(13.5px * var(--font-scale, 1))', color: textColor, fontWeight: 500 }}>{message}</span>
      </div>
    </>
  );
}

export default function Toast() {
  return ReactDOM.createPortal(<ToastContent />, document.body);
}
