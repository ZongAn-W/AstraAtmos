/**
 * AstraAtmos 色阶工具模块
 * 所有组件统一从这里导入，不再各自重复定义色阶函数
 */

// 通用线性插值器（stops: [[r,g,b], ...], t: 0-1）
function interp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const i = Math.min(Math.floor(idx), stops.length - 2);
  const f = idx - i;
  const c0 = stops[i], c1 = stops[i + 1];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * f),
    Math.round(c0[1] + (c1[1] - c0[1]) * f),
    Math.round(c0[2] + (c1[2] - c0[2]) * f),
  ];
}

const STOPS = {
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [72, 39, 120], [62, 82, 160], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [96, 199, 101], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [204, 70, 120], [229, 107, 93], [248, 148, 65], [240, 249, 33],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [252, 253, 191],
  ],
  cividis: [
    [0, 32, 77], [0, 59, 115], [34, 87, 119], [66, 112, 121],
    [98, 135, 122], [133, 161, 122], [170, 187, 122], [254, 233, 55],
  ],
  jet: [
    [0, 0, 143], [0, 0, 255], [0, 127, 255], [0, 255, 255],
    [127, 255, 127], [255, 255, 0], [255, 127, 0], [255, 0, 0], [128, 0, 0],
  ],
  rdbu: [
    [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
    [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 169, 128],
    [214, 96, 77], [178, 24, 43], [103, 0, 31],
  ],
};

/** 获取 [r,g,b] 数组（t: 0-1），name 无效时回退 inferno */
export function getRgb(name, t) {
  return interp(STOPS[name] ?? STOPS.inferno, t);
}

/** 获取 'rgb(r,g,b)' 字符串（t: 0-1） */
export function getRgbStr(name, t) {
  const [r, g, b] = getRgb(name, t);
  return `rgb(${r},${g},${b})`;
}

/** 专用于残差场的 RdBu 色阶（始终固定，与 colormap 设置无关） */
export function rdbuRgb(t) {
  return interp(STOPS.rdbu, t);
}

/**
 * 生成 CSS linear-gradient 字符串（高值在上，低值在下）
 * 用于 colorbar 背景
 */
export function makeGradient(name) {
  const n = 10;
  // t 从 1 递减到 0：高值(黄)在顶部 0%，低值(黑)在底部 100%
  const pts = Array.from({ length: n }, (_, i) => {
    const t = 1 - i / (n - 1);
    const [r, g, b] = getRgb(name, t);
    return `rgb(${r},${g},${b}) ${(i / (n - 1) * 100).toFixed(0)}%`;
  });
  return `linear-gradient(180deg, ${pts.join(', ')})`;
}

/** Plotly 内置色阶名称映射 */
export const PLOTLY_SCALE = {
  inferno: 'Inferno',
  viridis: 'Viridis',
  plasma:  'Plasma',
  magma:   'Magma',
  cividis: 'Cividis',
  jet:     'Jet',
  rdbu:    'RdBu',
};

/** 已知色阶名（不含残差专用 rdbu，它由 colorMode 直接指定） */
const NAMED = new Set(['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'jet']);

/** 有符号的变量：风分量在火星与地球都用发散色阶（地球的 `earthColormap` 同样如此）。 */
const DIVERGING_VARIABLES = new Set(['U_Wind', 'V_Wind', 'U10M', 'V10M']);

/** 变量与设置解析出的色阶描述：`{ mode, name, min, max }`；数值缺失时范围落在 0–1 之外，调用方据此回退。 */
export function makeColorSpec({ variable, centeredOnZero, colormap, range } = {}) {
  const diverging = Boolean(centeredOnZero) || DIVERGING_VARIABLES.has(variable);
  const mode = diverging ? 'rdbu' : (NAMED.has(colormap) ? colormap : 'inferno');
  const finite = Number.isFinite(range?.min) && Number.isFinite(range?.max) && range.max > range.min;
  return {
    mode,
    name: mode === 'rdbu' ? 'rdbu' : mode,
    min: finite ? range.min : null,
    max: finite ? range.max : null,
  };
}

/** 按已知范围把数值转成色阶坐标 t∈[0,1]；范围缺失或退化时返回 null。 */
export function colorFraction(spec, value) {
  if (!Number.isFinite(value)) return null;
  const min = spec?.min;
  const max = spec?.max;
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}
