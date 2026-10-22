import { useId, useRef } from 'react';
import CloseOutlined from '@mui/icons-material/CloseOutlined';
import { useSettings } from '../contexts/SettingsContext';
import { useT } from '../i18n';
import { useScrollLock } from '../hooks/useScrollLock';
import { useDialogFocus } from '../hooks/useDialogFocus';
import { DEFAULT_TRAINING_DEFAULTS } from '../utils/trainingDefaults';
import JellyToggle from './ui/JellyToggle';
import './ui/overlay.css';

/* ─── 小工具组件 ─── */

function SectionHeader({ label }) {
  return <h3 className="av-settings-section">{label}</h3>;
}

function Divider() {
  return <div className="av-settings-divider" />;
}

/** 二选一 / 多选一 pill 按钮组 */
function ChipGroup({ options, value, onChange, cols = 2 }) {
  return (
    <div className="av-segmented" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {options.map(opt => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            className="av-segment"
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
            title={opt.desc || opt.label}
          >
            {opt.label}
            {opt.sub && (
              <small>{opt.sub}</small>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** 行内标签 + 控件 */
function SettingRow({ label, children }) {
  return (
    <div className="av-setting-row">
      <span>{label}</span>
      <div>
        {children}
      </div>
    </div>
  );
}

function NumberSetting({ label, value, min, max, step = 1, onChange }) {
  return (
    <SettingRow label={label}>
      <input
        type="number"
        className="av-number-input"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value ?? ''}
        onChange={onChange}
      />
    </SettingRow>
  );
}

/** 紧凑型 pill 切换（用于行内 unit 选择等） */
function InlinePill({ options, value, onChange }) {
  return (
    <div className="av-segmented av-segmented--inline">
      {options.map(opt => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            className="av-segment"
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

/** Checkbox：与全站一致改用果冻滑块，点击文字同样可切换 */
function Checkbox({ label, checked, onChange }) {
  return (
    <div className="av-checkbox av-checkbox--switch">
      <JellyToggle checked={Boolean(checked)} onChange={onChange} label={label} />
      <span className="av-checkbox__text" onClick={() => onChange(!checked)}>{label}</span>
    </div>
  );
}

/* ─── 主组件 ─── */

export default function SettingsPanel({ open, onClose }) {
  const { settings, updateSetting } = useSettings();
  const t = useT();
  const panelRef = useRef(null);
  const titleId = useId();
  const closeLabel = settings.language === 'zh' ? '关闭设置' : 'Close preferences';
  const trainingDefaults = settings.trainingDefaults || DEFAULT_TRAINING_DEFAULTS;
  const ratioTotal = ['trainRatio', 'validationRatio', 'testRatio']
    .reduce((total, key) => total + (Number(trainingDefaults[key]) || 0), 0);
  const ratioTotalPercent = Math.round(ratioTotal * 10000) / 100;
  const ratioTotalValid = Math.abs(ratioTotal - 1) < 0.000001;
  const updateTrainingNumber = (key, rawValue) => {
    updateSetting(`trainingDefaults.${key}`, rawValue);
  };
  const updateRatio = (key, rawValue) => {
    if (rawValue === '') {
      updateSetting(`trainingDefaults.${key}`, '');
      return;
    }
    updateSetting(`trainingDefaults.${key}`, String(Number(rawValue) / 100));
  };
  useScrollLock(open);
  useDialogFocus(open, panelRef, onClose);

  return (
    <>
      {/* 遮罩 */}
      <div
        className="av-overlay-backdrop"
        onClick={onClose}
        style={{
          zIndex: 2999,
          opacity: open ? 1 : 0,
          pointerEvents: open ? 'auto' : 'none',
          transition: 'opacity 0.25s',
        }}
      />

      {/* 面板主体 */}
      <div
        ref={panelRef}
        className="av-drawer av-settings"
        data-open={open}
        role="dialog"
        aria-modal={open ? true : undefined}
        aria-labelledby={titleId}
        tabIndex={-1}
        inert={!open}
        aria-hidden={!open}
        style={{
          zIndex: 3000,
          boxShadow: open ? 'var(--shadow-panel)' : 'none',
        }}
      >
        {/* 面板头部 */}
        <div className="av-drawer-header">
          <h2 id={titleId} className="av-dialog-title">{t('settings.title')}</h2>
          <button
            type="button"
            className="av-icon-button"
            aria-label={closeLabel}
            title={closeLabel}
            data-dialog-autofocus
            onClick={onClose}
          >
            <CloseOutlined />
          </button>
        </div>

        {/* 可滚动内容区 */}
        <div className="av-drawer-body">

          {/* ── 显示偏好 ── */}
          <SectionHeader label={t('settings.appearance.label')} />
          <p style={{ fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
            {t('settings.appearance.desc')}
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8, marginBottom: 12 }}>
            <div style={{
              width: 44,
              fontSize: 'calc(var(--type-label) * var(--font-scale, 1))',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              textAlign: 'right',
            }}>
              A
            </div>
            
            <input
              type="range"
              aria-label={t('settings.appearance.label')}
              min="0.7"
              max="1.5"
              step="0.1"
              value={settings.appearance?.uiScale || 1}
              onChange={e => updateSetting('appearance.uiScale', parseFloat(e.target.value))}
              style={{
                flex: 1,
                cursor: 'pointer',
                accentColor: 'var(--line-active)',
                minHeight: 32,
              }}
            />
            
            <div style={{
              width: 44,
              fontSize: 'calc(var(--type-body) * var(--font-scale, 1))',
              fontWeight: 600,
              color: 'var(--text-primary)',
              textAlign: 'left',
            }}>
              A
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 8px' }}>
            <button
              onClick={() => updateSetting('appearance.uiScale', 1)}
              type="button"
              className="av-link-button"
            >
              {t('settings.appearance.scaleMedium') || 'Reset'}
            </button>
            <div style={{
              fontSize: 'calc(var(--type-label) * var(--font-scale, 1))',
              color: 'var(--brand-ice)',
              background: 'var(--surface-2)',
              padding: '2px 8px',
              borderRadius: 4,
              fontWeight: 600,
              fontVariantNumeric: 'tabular-nums'
            }}>
              {Math.round((settings.appearance?.uiScale || 1) * 100)}%
            </div>
          </div>
          <Divider />

          {/* ── 训练默认值 ── */}
          <SectionHeader label={t('settings.training.label')} />
          <p style={{ fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
            {t('settings.training.desc')}
          </p>
          {[
            ['epochs', t('settings.training.epochs'), 1, 1000, 1],
            ['batchSize', t('settings.training.batchSize'), 1, 64, 1],
            ['learningRate', t('settings.training.learningRate'), 0.000001, 1, 0.0001],
            ['window', t('settings.training.window'), 1, 30, 1],
            ['horizon', t('settings.training.horizon'), 1, 30, 1],
          ].map(([key, label, min, max, step]) => (
            <NumberSetting
              key={key}
              label={label}
              min={min}
              max={max}
              step={step}
              value={trainingDefaults[key]}
              onChange={event => updateTrainingNumber(key, event.target.value)}
            />
          ))}

          <Divider />
          <SectionHeader label={t('settings.training.strategyLabel')} />
          {[
            ['trainRatio', t('settings.training.trainRatio'), 1],
            ['validationRatio', t('settings.training.validationRatio'), 0],
            ['testRatio', t('settings.training.testRatio'), 1],
          ].map(([key, label, min]) => (
            <NumberSetting
              key={key}
              label={label}
              min={min}
              max={98}
              step={1}
              value={trainingDefaults[key] === '' ? '' : Math.round(Number(trainingDefaults[key]) * 100)}
              onChange={event => updateRatio(key, event.target.value)}
            />
          ))}
          <p className="av-helper" role="status" data-valid={ratioTotalValid ? 'true' : 'false'} style={{ color: ratioTotalValid ? 'var(--text-secondary)' : 'var(--status-danger)', margin: '0 0 12px' }}>
            {ratioTotalValid
              ? t('settings.training.ratioTotalValid', { total: ratioTotalPercent })
              : t('settings.training.ratioTotalInvalid', { total: ratioTotalPercent })}
          </p>
          <NumberSetting
            label={t('settings.training.seed')}
            min={0}
            max={2147483647}
            value={trainingDefaults.seed}
            onChange={event => updateTrainingNumber('seed', event.target.value)}
          />
          <NumberSetting
            label={t('settings.training.earlyStoppingPatience')}
            min={0}
            max={200}
            value={trainingDefaults.earlyStoppingPatience}
            onChange={event => updateTrainingNumber('earlyStoppingPatience', event.target.value)}
          />
          <Checkbox label={t('settings.training.transferEnabled')} checked={trainingDefaults.transferEnabled === true} onChange={value => updateSetting('trainingDefaults.transferEnabled', value)} />
          <SettingRow label={t('settings.training.freezeMode')}>
            <select aria-label={t('settings.training.freezeMode')} value={trainingDefaults.transferFreezeMode || 'none'} onChange={event => updateSetting('trainingDefaults.transferFreezeMode', event.target.value)}>
              <option value="none">{t('settings.training.freezeNone')}</option>
              <option value="backbone">{t('settings.training.freezeBackbone')}</option>
              <option value="head">{t('settings.training.freezeHead')}</option>
            </select>
          </SettingRow>
          <NumberSetting
            label={t('settings.training.finetuneLearningRate')}
            min={0.000001}
            max={1}
            step={0.00001}
            value={trainingDefaults.finetuneLearningRate}
            onChange={event => updateTrainingNumber('finetuneLearningRate', event.target.value)}
          />

          <Divider />

          {/* ── 单位制 ── */}
          <SectionHeader label={t('settings.units.label')} />
          <SettingRow label={t('settings.units.ozone')}>
            <InlinePill
              options={[
                { value: 'um-atm', label: 'μm-atm' },
                { value: 'DU',     label: 'DU' },
              ]}
              value={settings.units.ozone}
              onChange={v => updateSetting('units.ozone', v)}
            />
          </SettingRow>
          <SettingRow label={t('settings.units.temperature')}>
            <InlinePill
              options={[
                { value: 'K', label: 'K' },
                { value: 'C', label: '°C' },
              ]}
              value={settings.units.temperature}
              onChange={v => updateSetting('units.temperature', v)}
            />
          </SettingRow>
          <SettingRow label={t('settings.units.wind')}>
            <InlinePill
              options={[
                { value: 'm/s',  label: 'm/s' },
                { value: 'km/h', label: 'km/h' },
              ]}
              value={settings.units.wind}
              onChange={v => updateSetting('units.wind', v)}
            />
          </SettingRow>
          <SettingRow label={t('settings.units.pressure')}>
            <InlinePill
              options={[
                { value: 'Pa',  label: 'Pa' },
                { value: 'hPa', label: 'hPa' },
              ]}
              value={settings.units.pressure}
              onChange={v => updateSetting('units.pressure', v)}
            />
          </SettingRow>

          <Divider />

          {/* ── 数据精度 ── */}
          <SectionHeader label={t('settings.precision.label')} />
          <p style={{ fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
            {t('settings.precision.desc')}
          </p>
          <ChipGroup
            cols={4}
            options={[
              { value: 2,      label: t('settings.precision.opt2') },
              { value: 4,      label: t('settings.precision.opt4') },
              { value: 6,      label: t('settings.precision.opt6') },
              { value: 'full', label: t('settings.precision.optFull') },
            ]}
            value={settings.precision}
            onChange={v => updateSetting('precision', v)}
          />

          <Divider />

          {/* ── 导出偏好 ── */}
          <SectionHeader label={t('settings.export.label')} />
          <p style={{ fontSize: 'calc(var(--type-helper) * var(--font-scale, 1))', color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
            {t('settings.export.desc')}
          </p>

          <div style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)', marginBottom: 6 }}>{t('settings.export.format')}</div>
          <ChipGroup
            cols={3}
            options={[
              { value: 'png', label: t('settings.export.png'), sub: t('settings.export.pngSub') },
              { value: 'svg', label: t('settings.export.svg'), sub: t('settings.export.svgSub') },
              { value: 'pdf', label: t('settings.export.pdf'), sub: t('settings.export.pdfSub') },
            ]}
            value={settings.export.format}
            onChange={v => updateSetting('export.format', v)}
          />

          <div style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)', margin: '12px 0 6px' }}>{t('settings.export.dpi')}</div>
          <ChipGroup
            cols={3}
            options={[
              { value: 150, label: '150', sub: t('settings.export.dpi150Sub') },
              { value: 300, label: '300', sub: t('settings.export.dpi300Sub') },
              { value: 600, label: '600', sub: t('settings.export.dpi600Sub') },
            ]}
            value={settings.export.dpi}
            onChange={v => updateSetting('export.dpi', v)}
          />

          <div style={{ fontSize: 'calc(var(--type-label) * var(--font-scale, 1))', color: 'var(--text-secondary)', margin: '12px 0 6px' }}>{t('settings.export.fontSize')}</div>
          <ChipGroup
            cols={3}
            options={[
              { value: 8,  label: '8 pt' },
              { value: 10, label: '10 pt' },
              { value: 12, label: '12 pt' },
            ]}
            value={settings.export.fontSize}
            onChange={v => updateSetting('export.fontSize', v)}
          />

          <div style={{ marginTop: 12 }}>
            <Checkbox
              label={t('settings.export.includeTitle')}
              checked={settings.export.includeTitle}
              onChange={v => updateSetting('export.includeTitle', v)}
            />
          </div>

        </div>

        {/* 面板底部 */}
        <div className="av-drawer-footer">
          <span className="av-helper">{t('settings.autoSave')}</span>
          <button
            onClick={() => {
              if (window.confirm(t('settings.resetConfirm'))) {
                localStorage.removeItem('aresvision_settings');
                window.location.reload();
              }
            }}
            type="button"
            className="av-button av-button--compact av-button--danger"
          >
            {t('settings.reset')}
          </button>
        </div>
      </div>
    </>
  );
}
