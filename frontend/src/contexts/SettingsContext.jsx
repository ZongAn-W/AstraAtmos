import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { DEFAULT_FONT_SCALE, FONT_SCALE_VAR, normalizeFontScale } from '../utils/fontScale';
import { DEFAULT_TRAINING_DEFAULTS, normalizeTrainingDefaults } from '../utils/trainingDefaults';

const STORAGE_KEY = 'astraatmos_settings';

export const DEFAULT_SETTINGS = {
  language: 'zh',       // 'zh' | 'en'
  colormap: 'inferno',  // 'inferno' | 'viridis' | 'plasma' | 'magma' | 'cividis' | 'jet' | 'rdbu'
  units: {
    ozone: 'um-atm',    // 'um-atm' | 'DU'
    temperature: 'K',   // 'K' | 'C'
    wind: 'm/s',        // 'm/s' | 'km/h'
    pressure: 'Pa',     // 'Pa' | 'hPa'
  },
  precision: 2,         // 2 | 4 | 6 | 'full'
  theme: 'dark',        // 'dark' | 'light'
  export: {
    format: 'png',      // 'png' | 'svg' | 'pdf'
    dpi: 300,           // 150 | 300 | 600
    includeTitle: true,
    fontSize: 10,       // 8 | 10 | 12
  },
  appearance: {
    uiScale: DEFAULT_FONT_SCALE,         // font scale only
  },
  trainingDefaults: {
    ...DEFAULT_TRAINING_DEFAULTS,
  },
};

/** 深合并：以 defaults 结构为准，用 saved 中的值覆盖，忽略 saved 中多余的键 */
function deepMerge(defaults, saved) {
  const result = { ...defaults };
  for (const key of Object.keys(defaults)) {
    if (
      key in saved &&
      typeof defaults[key] === 'object' &&
      defaults[key] !== null &&
      !Array.isArray(defaults[key])
    ) {
      result[key] = deepMerge(defaults[key], saved[key]);
    } else if (key in saved) {
      result[key] = saved[key];
    }
  }
  if (defaults === DEFAULT_SETTINGS && result.trainingDefaults) {
    const normalizedTrainingDefaults = normalizeTrainingDefaults(result.trainingDefaults);
    for (const key of ['trainRatio', 'validationRatio', 'testRatio']) {
      const rawRatio = result.trainingDefaults[key];
      const ratio = Number(rawRatio);
      const minimum = key === 'validationRatio' ? 0 : 0.01;
      if (rawRatio === '' || (Number.isFinite(ratio) && ratio >= minimum && ratio <= 0.98)) {
        normalizedTrainingDefaults[key] = rawRatio === '' ? '' : ratio;
      }
    }
    result.trainingDefaults = normalizedTrainingDefaults;
  }
  return result;
}

const SettingsContext = createContext(null);

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return deepMerge(DEFAULT_SETTINGS, JSON.parse(raw));
    } catch {
      // 解析失败时回退到默认值
    }
    return DEFAULT_SETTINGS;
  });

  // 跟踪 colormap 是否由主题切换自动设置（用户未手动选择）
  const isColormapAutoSet = useRef(false);

  // 持久化
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  // 应用主题到 <html> 元素
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', settings.theme);
    // 清除 body 上可能残留的内联背景色，让 CSS 变量生效
    document.body.style.removeProperty('background');
  }, [settings.theme]);

  // 应用 UI 缩放 (用于调节字体和整体布局大小)
  useEffect(() => {
    const scale = normalizeFontScale(settings.appearance?.uiScale);
    document.documentElement.style.setProperty(FONT_SCALE_VAR, String(scale));
  }, [settings.appearance?.uiScale]);

  // 主题变化时自动切换 colormap
  useEffect(() => {
    if (settings.theme === 'light' && settings.colormap === 'inferno') {
      isColormapAutoSet.current = true;
      setSettings(prev => ({ ...prev, colormap: 'jet' }));
    } else if (settings.theme === 'dark' && isColormapAutoSet.current) {
      isColormapAutoSet.current = false;
      setSettings(prev => ({ ...prev, colormap: 'inferno' }));
    }
  }, [settings.theme]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * 更新设置，支持点分路径（如 'units.ozone' 或 'export.dpi'）
   * @param {string} path - 点分路径
   * @param {*} value - 新值
   */
  const updateSetting = useCallback((path, value) => {
    // 用户手动选择 colormap，清除自动设置标记
    if (path === 'colormap') {
      isColormapAutoSet.current = false;
    }
    setSettings(prev => {
      const next = { ...prev };
      const keys = path.split('.');
      let obj = next;
      for (let i = 0; i < keys.length - 1; i++) {
        obj[keys[i]] = { ...obj[keys[i]] };
        obj = obj[keys[i]];
      }
      obj[keys[keys.length - 1]] = value;
      return next;
    });
  }, []);

  return (
    <SettingsContext.Provider value={{ settings, updateSetting }}>
      {children}
    </SettingsContext.Provider>
  );
}
