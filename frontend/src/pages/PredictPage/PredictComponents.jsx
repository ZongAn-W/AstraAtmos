import { useRef, useEffect } from 'react'; // Re-triggering vite cache
import C from '../../constants/colors';
import { useT } from '../../i18n';
import { useSettings } from '../../contexts/SettingsContext';
import { buildCanvasFont, normalizeFontScale } from '../../utils/fontScale';
import { getRgb, rdbuRgb } from '../../utils/colormaps';
import { ozoneLabel, ozoneDeltaLabel, convertOzone } from '../../utils/units';
import { marsPredictionGrid } from './marsPredictionGrid';
import { fmtNum } from '../../utils/fmt';

export function fmtVal(v, precision = 3) { return fmtNum(v, precision); }

// ─── Canvas 场热力图（带坐标轴 + Colorbar） ───

export function FieldCanvas({ fieldData, colorMode = 'inferno', h = 240, colorRange }) {
  const canvasRef = useRef(null);
  const t = useT();
  const { settings } = useSettings();
  const colormapName = settings.colormap;
  const ozoneUnit = settings.units.ozone;
  const precision = settings.precision;
  const theme = settings.theme;
  const fontScale = normalizeFontScale(settings.appearance?.uiScale);
  const isLight = theme === 'light';

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const grid = marsPredictionGrid(fieldData);
    if (!grid) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    
    // 动态计算内部绘图区，确保维持约 2:1 的物理宽高比，防止地图变形
    const CH = canvas.height;  // h (e.g. 220 or 400)
    const ML = 56, MR = precision === 'full' ? 144 : Number(precision) > 3 ? 96 : 72, MT = 22, MB = 48;
    const plotH = CH - MT - MB;
    const plotW = plotH * 2;
    const CW = plotW + ML + MR;
    
    // 将逻辑宽度的计算回馈给 canvas 元素，防止拉伸。设置 width 会清空 context，需放在获取 ctx 之前。
    if (canvas.width !== Math.round(CW)) {
        canvas.width = Math.round(CW);
    }
    
    const ctx = canvas.getContext('2d');

    const isLight = theme === 'light';
    const axisTextColor  = isLight ? '#000000' : '#ffffff';
    const axisTitleColor = isLight ? '#000000' : '#ffffff';
    const axisLineColor  = isLight ? '#000000' : '#ffffff';
    const borderColor    = isLight ? '#000000' : '#ffffff';
    const cbBorderColor  = isLight ? '#000000' : '#ffffff';
    const cbLabelColor   = isLight ? '#000000' : '#ffffff';
    const cbTitleColor   = isLight ? '#000000' : '#ffffff';

    // 只填充绘图区背景，边距保持透明（浅色主题下边距显示卡片白色背景）
    ctx.clearRect(0, 0, CW, CH);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(ML, MT, plotW, plotH);

    const { field, minVal, maxVal } = fieldData;
    const nLat = field.length;    // 36
    const nLon = field[0].length; // 72

    // 计算色阶范围
    let dMin = colorRange?.min ?? minVal, dMax = colorRange?.max ?? maxVal;
    let absMax = 0;
    if (colorMode === 'rdbu') {
      for (let li = 0; li < nLat; li++)
        for (let lj = 0; lj < nLon; lj++)
          absMax = Math.max(absMax, Math.abs(field[li][lj]));
      absMax = absMax || 1;
      dMin = -absMax;
      dMax = absMax;
    }
    const range = dMax - dMin || 1;

    // 绘制热力图主体（ImageData）
    const imgData = ctx.createImageData(plotW, plotH);
    const pixels = imgData.data;
    for (let k = 0; k < pixels.length; k += 4) {
      pixels[k] = 10; pixels[k + 1] = 10; pixels[k + 2] = 15; pixels[k + 3] = 255;
    }
    for (let li = 0; li < nLat; li++) {
      const pyStart = Math.round(grid.latCells[li].start * plotH);
      const pyEnd = Math.round(grid.latCells[li].end * plotH);
      for (let lj = 0; lj < nLon; lj++) {
        const val = field[li][lj];
        if (val == null || isNaN(val)) continue;
        const t = (val - dMin) / range;
        const rgb = colorMode === 'rdbu' ? rdbuRgb(t) : getRgb(colormapName, Math.max(0, Math.min(1, t)));
        const pxStart = Math.round(grid.lonCells[lj].start * plotW);
        const pxEnd = Math.round(grid.lonCells[lj].end * plotW);
        for (let py = pyStart; py < pyEnd; py++) {
          for (let px = pxStart; px < pxEnd; px++) {
            if (px >= plotW || py >= plotH || px < 0 || py < 0) continue;
            const idx = (py * plotW + px) * 4;
            pixels[idx] = rgb[0]; pixels[idx + 1] = rgb[1]; pixels[idx + 2] = rgb[2]; pixels[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imgData, ML, MT);

    // 图框边框
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(ML, MT, plotW, plotH);

    // X 轴（经度）
    ctx.strokeStyle = axisLineColor;
    ctx.lineWidth = 1;
    ctx.fillStyle = axisTextColor;
    ctx.font = buildCanvasFont(11, { scale: fontScale });
    ctx.textAlign = 'center';
    grid.lonTicks.forEach(({ value: lonV, position }) => {
      const fx = ML + position * plotW;
      ctx.beginPath(); ctx.moveTo(fx, MT + plotH); ctx.lineTo(fx, MT + plotH + 4); ctx.stroke();
      ctx.fillText(`${Number(lonV.toFixed(2))}°`, fx, MT + plotH + 16);
    });
    ctx.fillStyle = axisTitleColor;
    ctx.font = buildCanvasFont(11, { weight: 'bold', scale: fontScale });
    ctx.fillText(`${t('overview.controls.longitude')} (°)`, ML + plotW / 2, CH - 6);

    // Y 轴（纬度）
    ctx.textAlign = 'right';
    ctx.fillStyle = axisTextColor;
    ctx.font = buildCanvasFont(11, { scale: fontScale });
    grid.latTicks.forEach(({ value: latV, position }) => {
      const fy = MT + position * plotH;
      ctx.beginPath(); ctx.moveTo(ML, fy); ctx.lineTo(ML - 4, fy); ctx.stroke();
      ctx.fillText(`${Number(latV.toFixed(2))}°`, ML - 8, fy + 3);
    });
    // Y 轴旋转标签
    ctx.save();
    ctx.translate(14, MT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = axisTitleColor;
    ctx.font = buildCanvasFont(11, { weight: 'bold', scale: fontScale });
    ctx.fillText(`${t('overview.controls.latitude')} (°)`, 0, 0);
    ctx.restore();


    // Colorbar
    const cbX = ML + plotW + 10;
    const cbW = 14;
    const cbH = plotH;
    const cbImgData = ctx.createImageData(cbW, cbH);
    const cbPx = cbImgData.data;
    for (let py = 0; py < cbH; py++) {
      const t = 1 - py / cbH; // 顶部=高值
      const rgb = colorMode === 'rdbu' ? rdbuRgb(t) : getRgb(colormapName, t);
      for (let px = 0; px < cbW; px++) {
        const idx = (py * cbW + px) * 4;
        cbPx[idx] = rgb[0]; cbPx[idx + 1] = rgb[1]; cbPx[idx + 2] = rgb[2]; cbPx[idx + 3] = 255;
      }
    }
    ctx.putImageData(cbImgData, cbX, MT);
    ctx.strokeStyle = cbBorderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, MT, cbW, cbH);

    // Colorbar 刻度标签
    const lbX = cbX + cbW + 3;
    ctx.textAlign = 'left';
    ctx.fillStyle = cbLabelColor;
    ctx.font = buildCanvasFont(11, { scale: fontScale });
    const topLabel = colorMode === 'rdbu' ? `+${fmtVal(convertOzone(absMax, ozoneUnit), precision)}` : fmtVal(convertOzone(dMax, ozoneUnit), precision);
    const midLabel = colorMode === 'rdbu' ? fmtVal(0, precision) : fmtVal(convertOzone((dMin + dMax) / 2, ozoneUnit), precision);
    const botLabel = colorMode === 'rdbu' ? `-${fmtVal(convertOzone(absMax, ozoneUnit), precision)}` : fmtVal(convertOzone(dMin, ozoneUnit), precision);
    ctx.fillText(topLabel, lbX, MT + 8);
    ctx.fillText(midLabel, lbX, MT + cbH / 2 + 3);
    ctx.fillText(botLabel, lbX, MT + cbH);

    // Colorbar 单位
    ctx.save();
    ctx.translate(cbX + cbW / 2, MT + cbH + 22);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = cbTitleColor;
    ctx.font = buildCanvasFont(11, { scale: fontScale });
    ctx.fillText(colorMode === 'rdbu' ? ozoneDeltaLabel(ozoneUnit) : ozoneLabel(ozoneUnit), 0, 0);
    ctx.restore();

  }, [fieldData, colorMode, h, colormapName, ozoneUnit, precision, theme, fontScale, colorRange?.min, colorRange?.max, t]);

  return (
    <div style={{ borderRadius: isLight ? 10 : 0, overflow: 'hidden', background: 'transparent', display: 'flex', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        width={400} // 这只是个初始占位值，useEffect 中会根据 h 计算精确的 2:1 宽高
        height={h}
        className="observation-window"
        style={{ width: '100%', height: 'auto', maxWidth: (h - 70) * 2 + 56 + (precision === 'full' ? 144 : Number(precision) > 3 ? 96 : 72), display: 'block', background: 'transparent' }}
      />
    </div>
  );
}

// ─── 辅助组件 ───

export function LoadingBox({ h = 240 }) {
  const t = useT();
  return (
    <div
      data-predict-loading="true"
      style={{
        height: h, display: 'flex', flexDirection: 'column',
        alignItems: 'stretch', justifyContent: 'center', gap: 10,
        padding: '0 18%',
        background: 'rgba(255,255,255,0.02)', borderRadius: 8,
      }}
    >
      {/* 用骨架块表示「正在计算」，比转圈更贴近最终内容的形状 */}
      <span className="ui-skeleton" style={{ height: '58%', borderRadius: 8 }} />
      <span className="ui-skeleton ui-skeleton--line" style={{ width: '46%', alignSelf: 'center' }} />
      <div style={{ alignSelf: 'center', fontSize: 'calc(12px * var(--font-scale, 1))', color: C.ice30 }}>{t('predict.computing')}</div>
    </div>
  );
}

export function EmptyBox({ h = 240 }) {
  const t = useT();
  return (
    <div className="prediction-field-empty" style={{ minHeight: h }}>
      {t('predict.clickToStart')}
    </div>
  );
}

// ─── 常量 ───

export const VARIABLE_DEFS = [
  { id: 'Temperature',        icon: '🌡',  color: '#ff6b4a' },
  { id: 'Dust_Optical_Depth', icon: '🌫',  color: '#d4a06a' },
  { id: 'Solar_Flux_DN',      icon: '☀️', color: '#ffd740' },
  { id: 'U_Wind',             icon: '💨',  color: '#4a9eff' },
  { id: 'V_Wind',             icon: '🌬',  color: '#7c5cbf' },
];

export const METRIC_META = [
  { key: 'rmse', name: 'RMSE', unit: 'μm-atm', better: '↓', color: C.mars },
  { key: 'mae', name: 'MAE', unit: 'μm-atm', better: '↓', color: C.mars },
  { key: 'ssim', name: 'SSIM', unit: '', better: '↑', color: '#4acfac' },
  { key: 'r2', name: 'R²', unit: '', better: '↑', color: '#4acfac' },
];

export const VIEW_MODE_IDS = ['triptych', 'original', 'prediction', 'diff'];

export const TRIPTYCH_PANEL_DEFS = [
  { key: 'truth',      color: C.blue,    mode: 'inferno' },
  { key: 'prediction', color: C.mars,    mode: 'inferno' },
  { key: 'residual',   color: '#9c7bea', mode: 'rdbu' },
];
