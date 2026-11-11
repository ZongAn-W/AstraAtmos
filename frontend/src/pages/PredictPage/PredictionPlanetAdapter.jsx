import { FieldCanvas } from './PredictComponents';
import SphericalFieldCanvas from '../../components/SphericalFieldCanvas';
import EarthMap2D from '../DataOverviewPage/EarthOverview/EarthMap2D';
import { fmtNum } from '../../utils/fmt';
import { makeGradient } from '../../utils/colormaps';
import { createEarthPredictionAdapter, createMarsPredictionAdapter } from './singleModelAdapters';
import { createEarthPredictionPresentation, createMarsPredictionPresentation } from './predictionPresentation';

export function PredictionOriginControl({ adapter, value, onChange, disabled }) {
  const origin = adapter.origin;
  return <label className="prediction-origin-control">
    <span>{origin.label}{origin.kind === 'range' ? ` · ${value}°` : ''}</span>
    <input type={origin.kind} min={origin.toInput(origin.min)} max={origin.toInput(origin.max)} step={origin.step}
      value={origin.toInput(value)} disabled={disabled}
      onChange={event => onChange(origin.fromInput(event.target.value))}
      data-prediction-origin={adapter.id} aria-label={origin.label} />
  </label>;
}

function EarthPredictionField({ fieldData, colorMode, colorRange, precision, colormap, height }) {
  const range = colorRange || fieldData.payload.color_range;
  const name = colorMode === 'rdbu' ? 'rdbu' : colormap;
  const payload = { ...fieldData.payload, color_range: range, colormap: name };
  const unit = payload.unit || 'DU';
  return <div className="prediction-earth-field" data-unit={unit} data-colormap={name}
    style={{ minHeight: typeof height === 'number' ? Math.min(height, 340) : undefined }}>
    <EarthMap2D field={payload} colormap={name} selectedPoint={null} />
    <div className="prediction-colorbar" aria-label={`${unit} ${fmtNum(range.min, precision)} – ${fmtNum(range.max, precision)}`}>
      <span>{fmtNum(range.min, precision)}</span>
      <i style={{ background: makeGradient(name).replace('180deg', '270deg') }} />
      <span>{fmtNum(range.max, precision)} {colorMode === 'rdbu' ? `Δ${unit}` : unit}</span>
    </div>
  </div>;
}

/** All planet-specific rendering is chosen here; shared UI consumes callbacks. */
export function earthPredictionAdapter(options) {
  return { ...createEarthPredictionAdapter(options), presentation: createEarthPredictionPresentation(options), renderField: props => (
    <EarthPredictionField {...props} colormap={options.colormap} />
  ) };
}

export function marsPredictionAdapter(options) {
  const adapter = createMarsPredictionAdapter(options);
  return { ...adapter, presentation: createMarsPredictionPresentation(options), renderField: ({ fieldData, colorMode, colorRange, height, fullscreen }) => {
    const grid = adapter.grid(fieldData);
    if (!grid) return null;
    const displayField = colorRange ? { ...grid.sphericalFieldData, minVal: colorRange.min, maxVal: colorRange.max } : grid.sphericalFieldData;
    return fullscreen ? <SphericalFieldCanvas planet="mars" fieldData={displayField}
      geometry={{ latCenters: grid.latitude, lonCenters: grid.longitude }} colorMode={colorMode}
      h="100%" zoom={3.25} showMars={false} />
      : <FieldCanvas fieldData={fieldData} colorMode={colorMode} h={height} colorRange={colorRange} presentation={adapter} />;
  } };
}
