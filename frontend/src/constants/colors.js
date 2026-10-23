/**
 * AstraAtmos 设计 Token — 颜色常量
 * 所有组件统一从这里引用颜色，方便全局调整
 * 主题感知颜色使用 CSS 变量，accent 色保持固定
 */
const C = {
  bg:          'var(--bg)',
  bgElevated:  'var(--bg-elevated)',
  bgCard:      'var(--bg-card)',
  bgCardStrong:'var(--bg-card-strong)',
  bgMuted:     'var(--bg-muted)',
  bgMutedStrong:'var(--bg-muted-strong)',
  mars:        '#ff8f68',
  marsLight:   '#ffb08e',
  blue:        '#79bbff',
  blueGlow:    'rgba(121,187,255,0.28)',
  ice:         'var(--text)',
  ice80:       'var(--text-80)',
  ice70:       'color-mix(in srgb, var(--text) 70%, transparent)',
  ice60:       'var(--text-60)',
  ice50:       'color-mix(in srgb, var(--text) 50%, transparent)',
  ice40:       'var(--text-40)',
  ice30:       'var(--text-30)',
  border:      'var(--border)',
  borderStrong:'var(--border-strong)',
  borderHover: 'var(--border-hover)',
  green:       '#63e8bf',
  purple:      '#b39bff',
  brand:       'var(--brand-ice)',
  action:      'var(--brand-orange)',
  success:     'var(--status-success)',
  warning:     'var(--status-warning)',
  danger:      'var(--status-danger)',
  diagnostic:  'var(--status-diagnostic)',
  ice45:       'var(--text-muted)',
};

export default C;
