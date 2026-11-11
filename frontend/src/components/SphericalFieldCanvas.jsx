import React, { useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { getRgb, rdbuRgb } from '../utils/colormaps';
import { useSettings } from '../contexts/SettingsContext';
import { buildCanvasFont, normalizeFontScale } from '../utils/fontScale';
import { buildSeasonalSunLight } from './sphericalLighting';
import { localPointToLatLng } from './sphericalPicking';
import {
  buildGridParticleSamples,
  buildPointParticleSamples,
  updateGridParticleBuffers,
  updatePointParticleBuffers,
} from './sphericalFieldParticles';
import {
  buildRegionalCellSampleValues,
  buildCoastlineSphereLines,
  geographicToCartesian,
  mapRegionalRgb,
} from './sphericalRegionalGrid';
import {
  EARTH_GLOBE_RADIUS,
  createEarthGlobeMaterial,
  fetchCoastlineGeoJson,
} from './sphericalEarthBaseMap';
import { OVERVIEW_GLOBE } from '../pages/DataOverviewPage/workbench/overviewVisualContract.js';
import {
  buildRegionalParticleGeometry,
  updateRegionalParticlePositions,
} from './sphericalRegionalParticles.js';
import {
  buildContourSegments,
  buildWindVectors,
} from './sphericalFieldLayers.js';

/** 场单元壳层半径：略高于地球底球与海岸线以外的可分层级由调用方决定。 */
const EARTH_FIELD_RADIUS = 0.872;
const EARTH_COASTLINE_LINE_RADIUS = 0.878;

/**
 * 相机姿态缓存：星球切换会卸载并重建整个 Three 场景，若不保存视角，
 * 每次切回来都会跳回默认朝向。键由调用方给出的 `poseKey` 决定（每个星球一个）。
 * 显式“重置视角”调用 `dropCameraPose()` 删除该键，并抑制随之而来的那一次保存
 * （重置会重建画布，旧画布的卸载清理否则会把重置前的视角又写回来）。
 * 只保存数值，不持有 Three 对象，卸载后不会泄漏。
 */
const cameraPoseCache = new Map();
const suppressedPoseSaves = new Set();
const CAMERA_POSE_CACHE_LIMIT = 8;

/** 丢弃某个视角键的缓存，并让下一次该键的保存被忽略一次。 */
export function dropCameraPose(poseKey) {
  if (!poseKey) return;
  cameraPoseCache.delete(poseKey);
  suppressedPoseSaves.add(poseKey);
}

// --- 全局缓存贴图 ---
let cachedMarsTexture = null;
let cachedCircleTexture = null;
const earthTextureCache = new Map();
let cachedCoastlineGeoJson = null;

function paintEarthTexture(entry, isLight) {
  if (!entry?.context || !entry?.canvas) return;
  const { canvas, context } = entry;
  const width = canvas.width;
  const height = canvas.height;
  // Local NASA Blue Marble imagery supplies land/ocean colours independently of data.
  if (entry.image) {
    context.clearRect(0, 0, width, height);
    context.globalAlpha = 1;
    context.drawImage(entry.image, 0, 0, width, height);
    entry.texture.needsUpdate = true;
    return;
  }
  const oceanTop = isLight ? '#a9c9e0' : '#06182b';
  const oceanBottom = isLight ? '#dbeaf3' : '#0c3d63';
  const ocean = context.createLinearGradient(0, 0, 0, height);
  ocean.addColorStop(0, oceanTop);
  ocean.addColorStop(0.48, isLight ? '#cfe4ef' : '#0a2b49');
  ocean.addColorStop(1, oceanBottom);
  context.clearRect(0, 0, width, height);
  context.fillStyle = ocean;
  context.fillRect(0, 0, width, height);

  // Soft latitude bands keep the continuous texture readable behind the data shell.
  context.globalAlpha = isLight ? 0.16 : 0.2;
  for (let latitude = -60; latitude <= 60; latitude += 30) {
    const y = ((90 - latitude) / 180) * height;
    context.fillStyle = latitude % 60 === 0
      ? (isLight ? '#f4fbff' : '#4da5c7')
      : (isLight ? '#ffffff' : '#164b6d');
    context.fillRect(0, Math.max(0, y - height * 0.045), width, height * 0.09);
  }

  context.globalAlpha = 1;
  entry.texture.needsUpdate = true;
}

function getEarthTexture(isLight) {
  if (typeof document === 'undefined') return null;
  const key = isLight ? 'light' : 'dark';
  const cached = earthTextureCache.get(key);
  if (cached) return cached.texture;
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 1024;
  const context = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  // SphereGeometry's U increases toward -Z; geographic east increases toward +Z.
  texture.repeat.x = -1;
  texture.offset.x = 1;
  const entry = { canvas, context, texture };
  earthTextureCache.set(key, entry);
  paintEarthTexture(entry, isLight);
  const image = new Image();
  image.onload = () => {
    entry.image = image;
    paintEarthTexture(entry, isLight);
  };
  // Keep the generated ocean texture if the optional imagery cannot load.
  image.onerror = () => {};
  image.src = '/earth/blue-marble-2048.png';
  return texture;
}

function latLonToVec3(latDeg, lonDeg, radius) {
  const phi = (90 - latDeg) * (Math.PI / 180);
  const theta = lonDeg * (Math.PI / 180);
  return new THREE.Vector3(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

function createLabelSprite(text, isLight, fontScale = 1) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const fontSize = 30;
  const padX = 20;
  const padY = 8;
  const strokeWidth = 4;
  const labelFontFamily = '"Segoe UI Symbol", "Segoe UI", "Arial Unicode MS", "Noto Sans", sans-serif';
  ctx.font = buildCanvasFont(fontSize, { family: labelFontFamily, weight: 600, scale: fontScale });
  const textWidth = Math.ceil(ctx.measureText(text).width);

  canvas.width = Math.max(110, textWidth + padX * 2 + strokeWidth * 2 + 6);
  canvas.height = fontSize + padY * 2 + strokeWidth;

  ctx.font = buildCanvasFont(fontSize, { family: labelFontFamily, weight: 600, scale: fontScale });
  ctx.fillStyle = isLight ? '#203042' : '#d5e8ff';
  ctx.strokeStyle = isLight ? 'rgba(255,255,255,0.9)' : 'rgba(8,12,20,0.9)';
  ctx.lineWidth = strokeWidth;
  ctx.lineJoin = 'round';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2 + 1);
  ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  texture.minFilter = THREE.LinearFilter;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: true,
    depthWrite: false,
  });

  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  const baseScale = 0.095;
  sprite.scale.set(baseScale * aspect, baseScale, 1);
  sprite.renderOrder = 20;
  return sprite;
}

function updateGeoLabelVisibility(overlay, globeGroup, camera) {
  if (!overlay || !globeGroup || !camera) return;

  const cameraLocal = globeGroup
    .worldToLocal(camera.getWorldPosition(new THREE.Vector3()))
    .normalize();
  overlay.children.forEach((child) => {
    const normal = child.userData?.geoLabelNormal;
    if (!normal) return;
    const isPole = child.userData.geoLabelRole === 'latitude-pole';
    child.visible = isPole || normal.dot(cameraLocal) > 0.22;
  });
}

function buildGeoOverlay(isLight, fontScale = 1) {
  const group = new THREE.Group();
  group.name = 'geo-overlay';

  const minorColor = isLight ? 0x3b4f66 : 0x89a8c8;
  const majorColor = isLight ? 0x1e293b : 0xc7e1ff;
  const lineRadius = 0.902;
  const latStep = 30;
  const lonStep = 60;

  const makeLine = (points, color, opacity) => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    });
    return new THREE.Line(geometry, material);
  };

  for (let lat = -90; lat <= 90; lat += latStep) {
    if (Math.abs(lat) === 90) continue; // 极点为退化点，网线使用标签表达更清晰
    const points = [];
    for (let lon = 0; lon <= 360; lon += 4) {
      points.push(latLonToVec3(lat, lon, lineRadius));
    }
    const isMajor = lat === 0;
    group.add(makeLine(points, isMajor ? majorColor : minorColor, isMajor ? 0.5 : 0.23));
  }

  for (let lon = 0; lon < 360; lon += lonStep) {
    const points = [];
    for (let lat = -90; lat <= 90; lat += 4) {
      points.push(latLonToVec3(lat, lon, lineRadius));
    }
    const isMajor = lon % 90 === 0;
    group.add(makeLine(points, isMajor ? majorColor : minorColor, isMajor ? 0.48 : 0.2));
  }

  const latLabels = [-90, -60, -30, 30, 60, 90];
  const degree = '\u00B0';
  const formatLatLabel = (lat) => (lat === 0 ? `0${degree}` : `${Math.abs(lat)}${degree}${lat > 0 ? 'N' : 'S'}`);
  latLabels.forEach((lat) => {
    const sprite = createLabelSprite(formatLatLabel(lat), isLight, fontScale);
    const p = latLonToVec3(lat, 8, lat === 90 || lat === -90 ? 1.06 : 1.03);
    sprite.position.set(p.x, p.y, p.z);
    sprite.userData.geoLabelNormal = p.clone().normalize();
    if (Math.abs(lat) === 90) sprite.userData.geoLabelRole = 'latitude-pole';
    group.add(sprite);
  });

  const lonLabels = Array.from({ length: 360 / lonStep }, (_, index) => index * lonStep);
  const lonLabelLat = 0;
  const formatLonLabel = (lon) => {
    if (lon === 0) return `0${degree}`;
    if (lon === 180) return `180${degree}`;
    if (lon > 0 && lon < 180) return `${lon}${degree}E`;
    return `${360 - lon}${degree}W`;
  };
  lonLabels.forEach((lon) => {
    const text = formatLonLabel(lon);
    const sprite = createLabelSprite(text, isLight, fontScale);
    const p = latLonToVec3(lonLabelLat, lon, 1.06);
    sprite.position.set(p.x, p.y, p.z);
    sprite.userData.geoLabelNormal = p.clone().normalize();
    group.add(sprite);
  });

  return group;
}

function disposeObject3D(root) {
  if (!root) return;
  root.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    if (node.material) {
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach((material) => {
        if (
          material?.map
          && material.map !== cachedMarsTexture
          && material.map !== cachedCircleTexture
          && !Array.from(earthTextureCache.values()).some((entry) => entry.texture === material.map)
        ) {
          material.map.dispose();
        }
        material?.dispose?.();
      });
    }
  });
}

function createCircleTexture() {
  if (cachedCircleTexture) return cachedCircleTexture;
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 32, 32);
  cachedCircleTexture = new THREE.CanvasTexture(canvas);
  return cachedCircleTexture;
}

function getParticleLayerKey(layerConfig, index) {
  return `${index}:${layerConfig?.id || layerConfig?.source || (layerConfig?.renderAsPoints ? 'points' : 'grid')}`;
}

function getParticleSeed(key) {
  let hash = 2166136261;
  for (let i = 0; i < key.length; i += 1) {
    hash ^= key.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function getGridLayerSignature(fieldData, geometry, particleDensity, radiusOffset) {
  const nLat = fieldData?.field?.length || 0;
  const nLon = fieldData?.field?.[0]?.length || 0;
  const latCenters = fieldData?.latCenters || geometry?.latCenters;
  const lonCenters = fieldData?.lonCenters || geometry?.lonCenters;
  const lat = latCenters?.length === nLat ? latCenters.join(',') : '';
  const lon = lonCenters?.length === nLon ? lonCenters.join(',') : '';
  return `grid:${nLat}x${nLon}:${particleDensity}:${radiusOffset}:${lat}:${lon}`;
}

function getPointLayerSignature(points, radiusOffset) {
  const parts = (points || []).map((point) => [
    Number(point?.lat || 0).toFixed(3),
    Number(point?.lng || 0).toFixed(3),
    Math.round(16 * Math.min(3, Math.max(1, Math.sqrt(Math.max(1, point?.count || 1))))),
  ].join(':'));
  return `points:${radiusOffset}:${parts.join('|')}`;
}

function mapParticleColor(colorMode, colormap, t) {
  const rgb = colorMode === 'rdbu' ? rdbuRgb(t) : getRgb(colormap, t);
  return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255];
}

function disposeParticleMesh(mesh) {
  if (!mesh) return;
  if (mesh.geometry) mesh.geometry.dispose();
  if (mesh.material) mesh.material.dispose();
}

function overlayCoordinates(fieldData, geometry, isEarth) {
  const field = fieldData?.field;
  const rows = field?.length || 0;
  const cols = field?.[0]?.length || 0;
  const latCenters = geometry?.latCenters?.length === rows
    ? geometry.latCenters
    : Array.from({ length: rows }, (_, index) => (90 - (index / Math.max(1, rows - 1)) * 180));
  const lonCenters = geometry?.lonCenters?.length === cols
    ? geometry.lonCenters
    : Array.from({ length: cols }, (_, index) => -180 + (index / Math.max(1, cols)) * 360);
  return { latCenters, lonCenters };
}

function setOverlayGroup(parent, ref, next) {
  if (ref.current && parent) {
    parent.remove(ref.current);
    disposeObject3D(ref.current);
  }
  ref.current = next || null;
  if (next && parent) parent.add(next);
}

function buildGeoLineGroup(records, { color, opacity, radius = 0.925, name }) {
  if (!records?.length) return null;
  const group = new THREE.Group();
  group.name = name;
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const positions = [];
  records.forEach((record) => {
    const points = record.points || [];
    if (points.length < 2) return;
    for (let index = 0; index + 1 < points.length; index += 1) {
      const start = latLonToVec3(points[index][0], points[index][1], radius);
      const end = latLonToVec3(points[index + 1][0], points[index + 1][1], radius);
      positions.push(start.x, start.y, start.z, end.x, end.y, end.z);
    }
  });
  if (positions.length < 6) {
    material.dispose();
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(geometry, material);
  lines.name = name;
  lines.renderOrder = 7;
  group.add(lines);
  return group;
}

function buildWindOverlay(vectors, isLight) {
  if (!vectors?.length) return null;
  const positions = [];
  vectors.forEach((vector) => {
    const radius = 1.135;
    const start = latLonToVec3(vector.start[0], vector.start[1], radius);
    const end = latLonToVec3(vector.end[0], vector.end[1], radius);
    const tangent = end.clone().sub(start);
    const length = tangent.length();
    if (length < 1e-5) return;
    tangent.divideScalar(length);
    const normal = end.clone().normalize();
    const side = new THREE.Vector3().crossVectors(normal, tangent);
    if (side.lengthSq() < 1e-8) return;
    side.normalize();
    const headLength = Math.min(0.052, Math.max(0.018, length * 0.34));
    const halfWidth = headLength * 0.56;
    const wingCenter = end.clone().addScaledVector(tangent, -headLength);
    const wingA = wingCenter.clone().addScaledVector(side, halfWidth);
    const wingB = wingCenter.clone().addScaledVector(side, -halfWidth);
    // Three independent line segments per arrow: shaft + two arrow-head wings.
    positions.push(
      start.x, start.y, start.z, end.x, end.y, end.z,
      end.x, end.y, end.z, wingA.x, wingA.y, wingA.z,
      end.x, end.y, end.z, wingB.x, wingB.y, wingB.z,
    );
  });
  if (!positions.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: isLight ? 0x155e75 : 0x9ad9ef,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.name = 'wind-vector-overlay';
  lines.renderOrder = 8;
  return lines;
}

function buildTerminatorOverlay(direction, isLight) {
  if (!direction) return null;
  const normal = new THREE.Vector3(direction.x, direction.y, direction.z).normalize();
  const basis = Math.abs(normal.y) < 0.92
    ? new THREE.Vector3(0, 1, 0)
    : new THREE.Vector3(1, 0, 0);
  const tangent = new THREE.Vector3().crossVectors(normal, basis).normalize();
  const bitangent = new THREE.Vector3().crossVectors(normal, tangent).normalize();
  const points = [];
  for (let index = 0; index <= 128; index += 1) {
    const angle = (index / 128) * Math.PI * 2;
    const point = tangent.clone().multiplyScalar(Math.cos(angle) * 1.145)
      .add(bitangent.clone().multiplyScalar(Math.sin(angle) * 1.145));
    points.push(point);
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: isLight ? 0xc76543 : 0xf19a78,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });
  const line = new THREE.Line(geometry, material);
  line.name = 'solar-terminator-overlay';
  line.renderOrder = 9;
  return line;
}

function applyBaseMapStrength(material, strength) {
  if (!material) return;
  const normalized = Math.max(0, Math.min(1, Number(strength) || 0));
  // Keep the reference sphere opaque at every strength.  Scaling the albedo
  // preserves depth and lets the data shell remain readable when the map is dimmed.
  const base = material.userData?.baseColor || material.color.clone();
  material.userData = material.userData || {};
  material.userData.baseColor = base;
  const albedo = 0.18 + normalized * 0.82;
  material.color.copy(base).multiplyScalar(albedo);
  material.opacity = 1;
  material.transparent = false;
  material.depthWrite = true;
  material.needsUpdate = true;
}

const SphericalFieldCanvas = forwardRef(({
  fieldData,
  fieldLayers,
  colorMode = 'inferno',
  h = 240,
  forceFullscreen = false,
  autoRotate = true,
  zoom = 4.5,
  showMars = true,
  showConcentration = true,
  showGeoAnnotations = true,
  showBaseMap = true,
  showSurfaceTexture = true,
  baseMapStrength = 1,
  showContours = false,
  showWindVectors = false,
  showTerminator = false,
  windFields = null,
  sunDirection = null,
  offsetX = 0,
  solarLongitudeLs = 0,
  onGlobeClick,
  // ── 共享工作台新增的显式接口 ────────────────────────────────────────
  // planet="mars" 且不传 field/geometry 时行为与旧调用完全一致。
  planet = 'mars',
  field = null,
  geometry = null,
  selection = null,
  lighting = null,
  particlePalette = null,
  particleDensity = OVERVIEW_GLOBE.particleDensity,
  particleSize = OVERVIEW_GLOBE.particleSize,
  pointParticleSize = OVERVIEW_GLOBE.pointParticleSize,
  globeMaterial = 'shared',
  lightingMode = null,
  // 视角记忆：同一个 poseKey 重新挂载时恢复上次的相机与球体朝向。
  poseKey = null,
  restoreCameraPose = true,
}, ref) => {
  const { settings } = useSettings();
  const fontScale = normalizeFontScale(settings.appearance?.uiScale);
  const isLight = settings.theme === 'light';
  const isEarth = planet === 'earth';
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sphereMeshRef = useRef(null);
  const particlesMeshRef = useRef(null);
  const particleLayersRef = useRef([]);
  const particleLayerCacheRef = useRef(new Map());
  const controlsRef = useRef(null);
  const autoRotateRef = useRef(autoRotate);
  const starMeshRef = useRef(null);
  const offsetXRef = useRef(offsetX);
  const geoOverlayRef = useRef(null);
  const marsMeshRef = useRef(null);
  const earthGlobeRef = useRef(null);
  const earthCoastlineRef = useRef(null);
  const regionalMeshRef = useRef(null);
  const regionalGeometryRef = useRef(null);
  const selectionMarkerRef = useRef(null);
  const contourOverlayRef = useRef(null);
  const windOverlayRef = useRef(null);
  const terminatorRef = useRef(null);
  const directionalLightRef = useRef(null);
  const pickingMeshRef = useRef(null);
  const onGlobeClickRef = useRef(onGlobeClick);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerRef = useRef(new THREE.Vector2());
  const isEarthRef = useRef(isEarth);
  isEarthRef.current = isEarth;
  // 待应用的视角：init effect 读取缓存并设置相机，图层 effect 在创建球体组时
  // 消费一次并清空，避免之后重建几何时重复套用旧朝向。
  const restorePoseRef = useRef(null);

  useEffect(() => {
    onGlobeClickRef.current = onGlobeClick;
  }, [onGlobeClick]);

  const addMarsMesh = (globeGroup) => {
    if (!globeGroup || marsMeshRef.current) return;
    const marsRadius = 0.86;
    const marsGeometry = new THREE.SphereGeometry(marsRadius, 64, 64);
    if (!cachedMarsTexture) {
      cachedMarsTexture = new THREE.TextureLoader().load('/mars_texture.jpg');
    }
    const marsMaterial = new THREE.MeshPhongMaterial({
      map: cachedMarsTexture,
      shininess: 5,
    });
    const marsMesh = new THREE.Mesh(marsGeometry, marsMaterial);
    globeGroup.add(marsMesh);
    marsMeshRef.current = marsMesh;
  };

  const removeMarsMesh = (globeGroup) => {
    if (!globeGroup || !marsMeshRef.current) return;
    const marsMesh = marsMeshRef.current;
    globeGroup.remove(marsMesh);
    if (marsMesh.geometry) marsMesh.geometry.dispose();
    const materials = Array.isArray(marsMesh.material) ? marsMesh.material : [marsMesh.material];
    materials.forEach((material) => {
      if (material?.map && material.map !== cachedMarsTexture) {
        material.map.dispose();
      }
      material?.dispose?.();
    });
    marsMeshRef.current = null;
  };

  const ensurePickingMesh = (globeGroup) => {
    if (!globeGroup || pickingMeshRef.current) return;
    const pickingGeometry = new THREE.SphereGeometry(0.9, 64, 64);
    const pickingMaterial = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const pickingMesh = new THREE.Mesh(pickingGeometry, pickingMaterial);
    pickingMesh.name = 'globe-picking-sphere';
    pickingMesh.renderOrder = -1;
    globeGroup.add(pickingMesh);
    pickingMeshRef.current = pickingMesh;
  };

  // ── 地球：连续彩色纹理底球、真实单元场与选中点 ────────────────
  // 底球纹理不参与数据色带；未着色区域表示该位置没有数据，而不是"零值"。
  const addEarthGlobe = (globeGroup) => {
    if (!globeGroup || earthGlobeRef.current) return;
    const options = createEarthGlobeMaterial({ isLight }) || {};
    const globeGeometry = new THREE.SphereGeometry(EARTH_GLOBE_RADIUS, 96, 64);
    const earthTexture = showSurfaceTexture ? getEarthTexture(isLight) : null;
    const material = new THREE.MeshPhongMaterial({
      map: earthTexture,
      color: earthTexture ? 0xffffff : (options.color ?? (isLight ? OVERVIEW_GLOBE.lightGlobeColor : OVERVIEW_GLOBE.darkGlobeColor)),
      emissive: options.emissive,
      emissiveIntensity: options.emissiveIntensity,
      shininess: globeMaterial === 'shared' ? OVERVIEW_GLOBE.globeShininess : 6,
      transparent: false,
      opacity: 1,
    });
    applyBaseMapStrength(material, baseMapStrength);
    const globeMesh = new THREE.Mesh(globeGeometry, material);
    globeMesh.name = 'earth-globe-base';
    globeGroup.add(globeMesh);
    earthGlobeRef.current = globeMesh;
  };

  const removeEarthGlobe = (globeGroup) => {
    if (!globeGroup) return;
    for (const ref of [earthGlobeRef, earthCoastlineRef, regionalMeshRef, selectionMarkerRef]) {
      const mesh = ref.current;
      if (!mesh) continue;
      globeGroup.remove(mesh);
      disposeObject3D(mesh);
      ref.current = null;
    }
    regionalGeometryRef.current = null;
  };

  const applyEarthGlobeColor = () => {
    if (!earthGlobeRef.current) return;
    const options = createEarthGlobeMaterial({ isLight }) || {};
    const earthTexture = showSurfaceTexture ? getEarthTexture(isLight) : null;
    earthGlobeRef.current.material.map = earthTexture;
    earthGlobeRef.current.material.color.setHex(earthTexture ? 0xffffff : (options.color ?? (isLight ? OVERVIEW_GLOBE.lightGlobeColor : OVERVIEW_GLOBE.darkGlobeColor)));
    earthGlobeRef.current.material.userData = earthGlobeRef.current.material.userData || {};
    earthGlobeRef.current.material.userData.baseColor = earthGlobeRef.current.material.color.clone();
    earthGlobeRef.current.material.emissive.set(options.emissive || '#000000');
    earthGlobeRef.current.material.emissiveIntensity = options.emissiveIntensity ?? 0;
    if (earthTexture) earthTexture.needsUpdate = true;
    applyBaseMapStrength(earthGlobeRef.current.material, baseMapStrength);
  };

  const ensureEarthCoastline = (globeGroup, isActive) => {
    if (!globeGroup || earthCoastlineRef.current) return;
    const build = (geojson) => {
      if (!isActive() || !geojson || earthCoastlineRef.current) return;
      const lines = buildCoastlineSphereLines(geojson, { radius: EARTH_COASTLINE_LINE_RADIUS });
      const group = new THREE.Group();
      group.name = 'earth-coastline';
      const material = new THREE.LineBasicMaterial({
        color: isLight ? 0x2f4a68 : 0xcfe4ff,
        transparent: true,
        opacity: isLight ? 0.72 : 0.62,
        depthWrite: false,
      });
      for (const positions of lines) {
        const lineGeometry = new THREE.BufferGeometry();
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const line = new THREE.Line(lineGeometry, material);
        line.renderOrder = 3;
        group.add(line);
      }
      globeGroup.add(group);
      earthCoastlineRef.current = group;
    };
    if (cachedCoastlineGeoJson) {
      build(cachedCoastlineGeoJson);
      return;
    }
    fetchCoastlineGeoJson({})
      .then((geojson) => {
        cachedCoastlineGeoJson = geojson;
        build(geojson);
      })
      .catch(() => {});
  };

  const ensureRegionalMesh = (globeGroup, geometrySpec) => {
    if (!globeGroup || !geometrySpec) return null;
    const key = JSON.stringify([
      geometrySpec.latCenters, geometrySpec.lonCenters,
      geometrySpec.latBounds, geometrySpec.lonBounds, particleDensity,
    ]);
    if (regionalGeometryRef.current?.key !== key || !regionalMeshRef.current) {
      if (regionalMeshRef.current) {
        globeGroup.remove(regionalMeshRef.current);
        disposeParticleMesh(regionalMeshRef.current);
      }
      const built = buildRegionalParticleGeometry({ ...geometrySpec, particleDensity, radius: EARTH_FIELD_RADIUS });
      const buffer = new THREE.BufferGeometry();
      buffer.setAttribute('position', new THREE.BufferAttribute(built.positions, 3));
      buffer.setAttribute('color', new THREE.BufferAttribute(new Float32Array(built.vertexCount * 3), 3));
      const material = new THREE.PointsMaterial({
        vertexColors: true, map: createCircleTexture(), transparent: true, depthWrite: false,
      });
      const mesh = new THREE.Points(buffer, material);
      mesh.name = 'earth-regional-particles';
      mesh.renderOrder = 2;
      globeGroup.add(mesh);
      regionalMeshRef.current = mesh;
      regionalGeometryRef.current = { key, built };
    }
    Object.assign(regionalMeshRef.current.material, {
      size: particleSize,
      opacity: isLight ? 0.7 : 0.9,
      blending: isLight ? THREE.NormalBlending : THREE.AdditiveBlending,
    });
    regionalMeshRef.current.material.needsUpdate = true;
    return regionalMeshRef.current;
  };

  const applyRegionalField = (fieldSpec) => {
    const built = regionalGeometryRef.current?.built;
    const mesh = regionalMeshRef.current;
    if (!built || !mesh || !fieldSpec?.values) return;
    const values = buildRegionalCellSampleValues(fieldSpec.values, built.cellCount);
    const position = mesh.geometry.getAttribute('position');
    const color = mesh.geometry.getAttribute('color');
    updateRegionalParticlePositions({
      geometry: built,
      values,
      colorRange: fieldSpec.colorRange || { min: 0, max: 1 },
      positions: position.array,
      colors: color.array,
      colorMapper: (t) => mapRegionalRgb(colorMode, colorMode === 'rdbu' ? settings.colormap : colorMode, t),
      baseRadius: EARTH_FIELD_RADIUS,
      heightScale: 0.225,
      signed: colorMode === 'rdbu',
    });
    position.needsUpdate = true;
    color.needsUpdate = true;
    mesh.geometry.computeBoundingSphere();
  };

  const applySelectionMarker = (globeGroup, point) => {
    if (!globeGroup) return;
    if (!point || !Number.isFinite(point.lat) || !Number.isFinite(point.lon)) {
      if (selectionMarkerRef.current) {
        globeGroup.remove(selectionMarkerRef.current);
        disposeObject3D(selectionMarkerRef.current);
        selectionMarkerRef.current = null;
      }
      return;
    }
    const position = geographicToCartesian(point.lat, point.lon, EARTH_COASTLINE_LINE_RADIUS + 0.006);
    if (!selectionMarkerRef.current) {
      const markerGeometry = new THREE.SphereGeometry(0.018, 16, 16);
      const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xffd166 });
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.name = 'earth-selection-marker';
      marker.renderOrder = 5;
      globeGroup.add(marker);
      selectionMarkerRef.current = marker;
    }
    selectionMarkerRef.current.position.set(position.x, position.y, position.z);
  };

  const ensureParticleEntry = ({
    key,
    signature,
    samples,
    materialOptions,
    globeGroup,
  }) => {
    const cache = particleLayerCacheRef.current;
    let entry = cache.get(key);

    if (!entry || entry.signature !== signature) {
      if (entry?.mesh) {
        globeGroup.remove(entry.mesh);
        disposeParticleMesh(entry.mesh);
      }

      const positions = new Float32Array(samples.count * 3);
      const colors = new Float32Array(samples.count * 3);
      const geometry = new THREE.BufferGeometry();
      const positionAttribute = new THREE.BufferAttribute(positions, 3);
      const colorAttribute = new THREE.BufferAttribute(colors, 3);
      positionAttribute.setUsage(THREE.DynamicDrawUsage);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('position', positionAttribute);
      geometry.setAttribute('color', colorAttribute);
      const material = new THREE.PointsMaterial({
        vertexColors: true,
        map: createCircleTexture(),
        transparent: true,
        depthWrite: false,
        ...materialOptions,
      });
      const mesh = new THREE.Points(geometry, material);
      globeGroup.add(mesh);
      entry = { key, signature, samples, mesh, positions, colors };
      cache.set(key, entry);
    } else {
      Object.assign(entry.mesh.material, materialOptions);
      entry.mesh.material.needsUpdate = true;
    }

    return entry;
  };

  const removeUnusedParticleEntries = (activeKeys, globeGroup) => {
    const cache = particleLayerCacheRef.current;
    for (const [key, entry] of cache.entries()) {
      if (activeKeys.has(key)) continue;
      globeGroup.remove(entry.mesh);
      disposeParticleMesh(entry.mesh);
      cache.delete(key);
    }
  };

  useEffect(() => {
    offsetXRef.current = offsetX;
  }, [offsetX]);

  const pickGlobeAtClientPoint = (clientX, clientY) => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const globe = sphereMeshRef.current;
    const pickingMesh = pickingMeshRef.current;
    if (!renderer || !camera || !globe || !pickingMesh) return null;

    const rect = renderer.domElement.getBoundingClientRect();
    const xRatio = (clientX - rect.left) / rect.width;
    const yRatio = (clientY - rect.top) / rect.height;
    if (xRatio < 0 || xRatio > 1 || yRatio < 0 || yRatio > 1) return null;

    const pointer = pointerRef.current;
    pointer.x = xRatio * 2 - 1;
    pointer.y = -(yRatio * 2 - 1);
    raycasterRef.current.setFromCamera(pointer, camera);
    const [hit] = raycasterRef.current.intersectObject(pickingMesh, false);
    if (!hit) return null;

    const localPoint = globe.worldToLocal(hit.point.clone());
    return localPointToLatLng(localPoint);
  };

  // Expose imperative API for gesture control
  useImperativeHandle(ref, () => ({
    applyGestureRotation: (dx, dy) => {
      if (sphereMeshRef.current && cameraRef.current) {
        // 模型旋转：不要直接修改固定的 Euler 旋转（会产生万向节锁或方向反转）
        // 改为绕着相机空间内的世界轴（Up和Right）进行旋转
        // 放大倍率提高体验灵敏度

        // 算出相机在世界空间中的向上和向右向量
        const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(cameraRef.current.quaternion).normalize();
        const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(cameraRef.current.quaternion).normalize();

        // 绕着视角的Y（Up）轴左右转，绕X（Right）轴上下转
        sphereMeshRef.current.rotateOnWorldAxis(cameraUp, dx * 3.0);
        sphereMeshRef.current.rotateOnWorldAxis(cameraRight, dy * 3.0);
      }
    },
    applyGestureZoom: (dDist) => {
      if (cameraRef.current) {
        // 向内捏合变小 (-dDist): 视距变大 (离远); 向外张开 (+dDist): 视距变小 (凑近)
        const step = -dDist * 8.0;

        // 因为用户可能用鼠标（TrackballControls）转动过视角，相机的坐标不再是在纯正的 Z 轴上
        // 正确做法是直接缩放相机所在坐标向量的长度（维持到原点方向不变）
        const currentDist = cameraRef.current.position.length();
        const newDist = Math.max(1.2, Math.min(12.0, currentDist + step));
        cameraRef.current.position.setLength(newDist);
      }
    },
    pickGlobeAtClientPoint: (clientX, clientY) => pickGlobeAtClientPoint(clientX, clientY),
  }));

  // Update ref when prop changes so animation loop catches it
  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  // 1. 初始化 Three.js 场景、相机、渲染器和控制器（仅执行一次）
  useEffect(() => {
    if (!containerRef.current) return;

    // 清理可能存在的旧 Canvas
    containerRef.current.innerHTML = '';

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    if (offsetX !== 0) {
      camera.setViewOffset(width, height, offsetX, 0, width, height);
    }
    camera.position.set(0, 0, zoom);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // 恢复同一 poseKey 的上次视角；不存在时保持默认朝向。
    const restorePose = restoreCameraPose && poseKey ? cameraPoseCache.get(poseKey) : null;
    restorePoseRef.current = restorePose || null;
    if (restorePose) {
      camera.position.set(
        restorePose.cameraPosition[0], restorePose.cameraPosition[1], restorePose.cameraPosition[2],
      );
      camera.quaternion.set(
        restorePose.cameraQuaternion[0], restorePose.cameraQuaternion[1],
        restorePose.cameraQuaternion[2], restorePose.cameraQuaternion[3],
      );
      // TrackballControls 以 target 为旋转中心；两者必须一起恢复，否则视角会漂。
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 3.0; // 适当降低些旋转的抽搐
    controls.zoomSpeed = 0.5; // 降低缩放灵敏度
    controls.panSpeed = 0.2; // 显著降低右键平移的灵敏度
    controls.noZoom = false;
    // 禁止右键平移，固定球体在这个中心位置
    controls.noPan = true;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.ZOOM,
    };
    controls.staticMoving = false; // true可以去掉阻尼
    controls.dynamicDampingFactor = 0.15; // 阻尼系数

    // 让球体固定在画面中央
    controls.target.set(0, 0, 0);

    controlsRef.current = controls;

    let pointerStart = null;

    const handlePointerDown = (event) => {
      if (event.button !== 0) return;
      pointerStart = { x: event.clientX, y: event.clientY, time: performance.now() };
    };

    const handlePointerUp = (event) => {
      if (event.button !== 0 || !pointerStart || !cameraRef.current || !sphereMeshRef.current || !pickingMeshRef.current) {
        pointerStart = null;
        return;
      }
      const moved = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y);
      const elapsed = performance.now() - pointerStart.time;
      pointerStart = null;
      if (moved > 5 || elapsed > 850 || typeof onGlobeClickRef.current !== 'function') return;

      const coord = pickGlobeAtClientPoint(event.clientX, event.clientY);
      if (coord) onGlobeClickRef.current(coord);
    };

    renderer.domElement.addEventListener('pointerdown', handlePointerDown);
    renderer.domElement.addEventListener('pointerup', handlePointerUp);

    // 光照对于 Points 材质不生效，但可用于内部火星球体
    const ambientLight = new THREE.AmbientLight(0xffffff, isLight ? 0.8 : 0.2);
    scene.add(ambientLight);

    const seasonalSunlight = isEarthRef.current
      // Earth 的日平均日期不是某一时刻的太阳位置：使用固定展示照明，
      // 不调用 buildSeasonalSunLight，也不画晨昏线。
      ? { position: { x: 3, y: 1.6, z: 4 } }
      : buildSeasonalSunLight(solarLongitudeLs);
    const dirLight = new THREE.DirectionalLight(0xffffff, isLight ? 1.0 : 1.5);
    dirLight.position.set(
      seasonalSunlight.position.x,
      seasonalSunlight.position.y,
      seasonalSunlight.position.z,
    );
    scene.add(dirLight);
    directionalLightRef.current = dirLight;

    // --- 背景星星特效（恒定不变，在此初始化）---
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 500;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      // 随机散布在半宽 10 的立方体内，挖空中间半径 2 的核心（避免挡住主星）
      let r = 2.5 + Math.random() * 8.0;
      let theta = Math.random() * Math.PI * 2;
      let phi = Math.acos(2 * Math.random() - 1);

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.cos(phi);
      const z = r * Math.sin(phi) * Math.sin(theta);

      starPositions[i * 3] = x;
      starPositions[i * 3 + 1] = y;
      starPositions[i * 3 + 2] = z;
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
      color: isLight ? 0x1e293b : 0xffffff,
      size: 0.02,
      transparent: true,
      opacity: isLight ? 0 : 0.6, // 浅色模式初始就不显示
      depthWrite: false,
    });
    const stars = new THREE.Points(starGeometry, starMaterial);
    stars.visible = !isLight; // 初始可见性
    scene.add(stars);
    starMeshRef.current = stars;

    let reqId;
    const animate = () => {
      reqId = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      if (sphereMeshRef.current && autoRotateRef.current) {
        sphereMeshRef.current.rotateY(0.001); // 绕模型本身的极点（局部 Y 轴）自转，即使手势倾斜了球体也始终按纬度线旋转
      }
      updateGeoLabelVisibility(geoOverlayRef.current, sphereMeshRef.current, cameraRef.current);
      stars.rotateY(0.0003); // 星空背景微弱伴走
      if (rendererRef.current) rendererRef.current.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current) return;
      const w = containerRef.current.clientWidth;
      const h2 = containerRef.current.clientHeight;
      if (!w || !h2) return;
      camera.aspect = w / h2;
      if (offsetXRef.current !== 0) {
        camera.setViewOffset(w, h2, offsetXRef.current, 0, w, h2);
      } else {
        camera.clearViewOffset();
      }
      camera.updateProjectionMatrix();
      if (controlsRef.current) {
        controlsRef.current.handleResize();
      }
      rendererRef.current.setSize(w, h2);
    };
    const sceneObserver = new ResizeObserver(handleResize);
    sceneObserver.observe(containerRef.current);
    window.addEventListener('resize', handleResize);

    return () => {
      sceneObserver.disconnect();
      controls.dispose();
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp);
      cancelAnimationFrame(reqId);
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current.forceContextLoss();
      }
      starGeometry.dispose();
      starMaterial.dispose();
    };
  }, [forceFullscreen]); // 仅在尺寸模式切换时重新初始化控制台

  // 主题变化：更新场景背景色和星星可见性
  useEffect(() => {
    if (!sceneRef.current) return;
    sceneRef.current.background = null;
    if (starMeshRef.current) {
      starMeshRef.current.visible = !isLight; // 浅色模式下隐藏星星，保持画面纯净
      starMeshRef.current.material.color.setHex(isLight ? 0x1e293b : 0xffffff);
    }
    sceneRef.current.children.forEach(child => {
      if (child instanceof THREE.AmbientLight) {
        child.intensity = isLight ? 0.8 : 0.2;
      }
    });
    if (directionalLightRef.current) {
      directionalLightRef.current.intensity = isLight ? 1.0 : 1.5;
    }
  }, [isLight]);

  useEffect(() => {
    if (isEarth || lightingMode === 'fixed') {
      // Earth 使用固定展示照明，Ls 不参与光照。
      if (directionalLightRef.current) {
        directionalLightRef.current.intensity = isLight ? 1.0 : 1.5;
        directionalLightRef.current.position.set(3, 1.6, 4);
      }
      applyEarthGlobeColor();
      return;
    }
    const seasonalSunlight = buildSeasonalSunLight(solarLongitudeLs);
    if (directionalLightRef.current) {
      directionalLightRef.current.intensity = isLight ? 1.0 : 1.5;
      directionalLightRef.current.position.set(
        seasonalSunlight.position.x,
        seasonalSunlight.position.y,
        seasonalSunlight.position.z,
      );
    }
  }, [baseMapStrength, isEarth, isLight, lightingMode, showSurfaceTexture, solarLongitudeLs]);

  useEffect(() => {
    if (sphereMeshRef.current) {
      if (geoOverlayRef.current) {
        sphereMeshRef.current.remove(geoOverlayRef.current);
        disposeObject3D(geoOverlayRef.current);
        geoOverlayRef.current = null;
      }
      const overlay = buildGeoOverlay(isLight, fontScale);
      overlay.visible = showGeoAnnotations;
      sphereMeshRef.current.add(overlay);
      geoOverlayRef.current = overlay;
    }
  }, [fontScale, isLight]);

  useEffect(() => {
    if (geoOverlayRef.current) {
      geoOverlayRef.current.visible = showGeoAnnotations;
    }
  }, [showGeoAnnotations]);

  useEffect(() => {
    if (particlesMeshRef.current) {
      particlesMeshRef.current.visible = showConcentration;
    }
    particleLayersRef.current.forEach((layer) => {
      layer.visible = showConcentration;
    });
  }, [showConcentration]);

  // 当外部动态调整窗口边界宽度时，实时保持地球在可用中间区域的正中心
  useEffect(() => {
    if (!cameraRef.current || !containerRef.current) return;
    const w = containerRef.current.clientWidth;
    const h2 = containerRef.current.clientHeight;
    if (offsetX !== 0) {
      cameraRef.current.setViewOffset(w, h2, offsetX, 0, w, h2);
    } else {
      cameraRef.current.clearViewOffset();
    }
    cameraRef.current.updateProjectionMatrix();
  }, [offsetX]);

  useEffect(() => {
    if (!sceneRef.current) return;
    const sourceLayers = Array.isArray(fieldLayers) && fieldLayers.length
      ? fieldLayers
      : [{ id: 'fieldData', source: 'fieldData', fieldData, colorMode }];
    const drawableLayers = sourceLayers.filter((layer) => layer?.fieldData?.field || (layer?.renderAsPoints && layer?.points?.length));

    const scene = sceneRef.current;
    if (!sphereMeshRef.current) {
      const globeGroup = new THREE.Group();
      globeGroup.rotation.y = -Math.PI / 2;
      const pendingPose = restorePoseRef.current;
      if (pendingPose?.globeQuaternion) {
        // 手势与自动旋转都会改球体朝向，和相机一起恢复才是一致的视角。
        globeGroup.quaternion.set(
          pendingPose.globeQuaternion[0], pendingPose.globeQuaternion[1],
          pendingPose.globeQuaternion[2], pendingPose.globeQuaternion[3],
        );
      }
      // 只消费一次：之后重建几何不应再次套用旧朝向。
      restorePoseRef.current = null;
      scene.add(globeGroup);
      sphereMeshRef.current = globeGroup;

      const overlay = buildGeoOverlay(isLight, fontScale);
      overlay.visible = showGeoAnnotations;
      globeGroup.add(overlay);
      geoOverlayRef.current = overlay;
    }

    const globeGroup = sphereMeshRef.current;

    // Legacy local coordinates put east at +Z. Reflect the whole Earth frame so
    // north-up views show east on the right, keeping every layer aligned.
    // worldToLocal() also undoes this reflection before geographic picking.
    globeGroup.scale.z = isEarth ? -1 : 1;

    if (isEarth) {
      // Earth particles keep their true coordinates; only visual height/colour interpolate.
      removeMarsMesh(globeGroup);
      addEarthGlobe(globeGroup);
      if (showBaseMap) ensureEarthCoastline(globeGroup, () => isEarthRef.current && sphereMeshRef.current === globeGroup);
      if (earthCoastlineRef.current) {
        earthCoastlineRef.current.visible = showBaseMap;
        earthCoastlineRef.current.children.forEach((line) => {
          line.material.color.setHex(isLight ? 0x2f4a68 : 0xcfe4ff);
          line.material.opacity = isLight ? 0.72 : 0.62;
        });
      }
      ensurePickingMesh(globeGroup);
      particleLayerCacheRef.current.forEach((entry) => { entry.mesh.visible = false; });
      particleLayersRef.current = [];
      particlesMeshRef.current = null;

      const mesh = ensureRegionalMesh(globeGroup, geometry);
      if (mesh) {
        mesh.visible = Boolean(showConcentration && field?.values);
        if (mesh.visible) applyRegionalField(field);
      }
      applySelectionMarker(globeGroup, selection?.point || null);
      return;
    }

    if (showMars) addMarsMesh(globeGroup);
    else removeMarsMesh(globeGroup);

    if (!drawableLayers.length) {
      removeUnusedParticleEntries(new Set(), globeGroup);
      particleLayersRef.current = [];
      particlesMeshRef.current = null;
      return;
    }

    if (!showConcentration) {
      particleLayerCacheRef.current.forEach((entry) => {
        entry.mesh.visible = false;
      });
      particleLayersRef.current = Array.from(particleLayerCacheRef.current.values()).map((entry) => entry.mesh);
      particlesMeshRef.current = particleLayersRef.current[0] || null;
      return;
    }

    const isLayeredMode = Array.isArray(fieldLayers) && fieldLayers.length > 1;
    const activeKeys = new Set();
    const nextLayers = [];

    drawableLayers.forEach((layerConfig, index) => {
      const key = getParticleLayerKey(layerConfig, index);
      activeKeys.add(key);

      if (layerConfig.renderAsPoints) {
        const layerColorMode = layerConfig.layerColorMode || layerConfig.colorMode || colorMode;
        const radiusOffset = layerConfig.radiusOffset || 0;
        const signature = getPointLayerSignature(layerConfig.points, radiusOffset);
        const cachedEntry = particleLayerCacheRef.current.get(key);
        const samples = cachedEntry?.signature === signature
          ? cachedEntry.samples
          : buildPointParticleSamples(layerConfig.points, {
            radiusOffset,
            seed: getParticleSeed(key),
          });
        const entry = ensureParticleEntry({
          key,
          signature,
          samples,
          globeGroup,
          materialOptions: {
            size: pointParticleSize,
            opacity: isLight ? 0.86 : 0.96,
            blending: isLight ? THREE.NormalBlending : THREE.AdditiveBlending,
          },
        });

        updatePointParticleBuffers({
          samples: entry.samples,
          points: layerConfig.points,
          colorMode: layerColorMode,
          positions: entry.positions,
          colors: entry.colors,
          colorMapper: mapParticleColor,
          tint: particlePalette?.pointTint || '#34d399',
          radiusOffset,
        });
        entry.mesh.geometry.attributes.position.needsUpdate = true;
        entry.mesh.geometry.attributes.color.needsUpdate = true;
        entry.mesh.visible = true;
        nextLayers.push(entry.mesh);
        return;
      }

      const layerFieldData = layerConfig.fieldData;
      const layerColorMode = layerConfig.colorMode || colorMode;
      const radiusOffset = layerConfig.radiusOffset || 0;
      const layerParticleDensity = isLayeredMode ? Math.max(24, Math.round(particleDensity * 0.46)) : particleDensity;
      const signature = getGridLayerSignature(layerFieldData, geometry, layerParticleDensity, radiusOffset);
      const cachedEntry = particleLayerCacheRef.current.get(key);
      const samples = cachedEntry?.signature === signature
        ? cachedEntry.samples
        : buildGridParticleSamples(layerFieldData.field, {
          particleDensity: layerParticleDensity,
          radiusOffset,
          seed: getParticleSeed(key),
          latCenters: layerFieldData.latCenters || geometry?.latCenters,
          lonCenters: layerFieldData.lonCenters || geometry?.lonCenters,
        });
      const entry = ensureParticleEntry({
        key,
        signature,
        samples,
        globeGroup,
        materialOptions: {
          size: particleSize,
          opacity: isLayeredMode ? (isLight ? 0.58 : 0.78) : (isLight ? 0.7 : 0.9),
          blending: isLight ? THREE.NormalBlending : THREE.AdditiveBlending,
        },
      });

      updateGridParticleBuffers({
        samples: entry.samples,
        fieldData: layerFieldData,
        colorMode: layerColorMode,
        colormap: settings.colormap,
        positions: entry.positions,
        colors: entry.colors,
        colorMapper: mapParticleColor,
        tint: layerConfig.tint || particlePalette?.fieldTint || null,
        radiusOffset,
        equatorHighlight: !isLayeredMode,
      });
      entry.mesh.geometry.attributes.position.needsUpdate = true;
      entry.mesh.geometry.attributes.color.needsUpdate = true;
      entry.mesh.visible = true;
      nextLayers.push(entry.mesh);
    });

    removeUnusedParticleEntries(activeKeys, globeGroup);
    particleLayersRef.current = nextLayers;
    particlesMeshRef.current = nextLayers[0] || null;
  }, [field, fieldData, fieldLayers, colorMode, geometry, isEarth, selection, settings.colormap, showBaseMap, showConcentration, showMars, showSurfaceTexture, baseMapStrength, isLight, fontScale, showGeoAnnotations, particleDensity, particlePalette, particleSize, pointParticleSize]);

  // Analysis overlays intentionally live in their own disposable groups so they can be
  // toggled without rebuilding the data field or the camera.  They are derived from
  // the same current slice shown by the globe; the optional wind pair is supplied by
  // the page controller only when the user requests it.
  useEffect(() => {
    const globeGroup = sphereMeshRef.current;
    if (!globeGroup) return;
    const firstLayer = Array.isArray(fieldLayers) && fieldLayers.length
      ? fieldLayers.find((layer) => layer?.fieldData?.field)
      : null;
    const activeFieldData = isEarth && field?.values
      ? {
        field: field.values,
        minVal: field.colorRange?.min,
        maxVal: field.colorRange?.max,
      }
      : firstLayer?.fieldData || fieldData;
    const activeGeometry = isEarth
      ? { latCenters: field?.latCenters || geometry?.latCenters, lonCenters: field?.lonCenters || geometry?.lonCenters, wrapLongitude: geometry?.wrapLongitude }
      : geometry;
    const { latCenters, lonCenters } = overlayCoordinates(activeFieldData, activeGeometry);

    if (showContours && activeFieldData?.field && latCenters.length && lonCenters.length) {
      const contours = buildContourSegments(activeFieldData, {
        levels: 6,
        latCenters,
        lonCenters,
        wrapLongitude: Boolean(activeGeometry?.wrapLongitude),
      });
      setOverlayGroup(globeGroup, contourOverlayRef, buildGeoLineGroup(contours, {
        color: isLight ? 0x17324d : 0xd5f5ff,
        opacity: isLight ? 0.52 : 0.56,
        // Analysis lines sit just above the tallest particle shell, so they
        // remain legible for both sequential and anomaly (signed) fields.
        radius: 1.13,
        name: 'field-contour-overlay',
      }));
    } else {
      setOverlayGroup(globeGroup, contourOverlayRef, null);
    }

    const wind = windFields?.u && windFields?.v && showWindVectors
      ? buildWindVectors(windFields.u, windFields.v, {
        latCenters: windFields.latCenters || latCenters,
        lonCenters: windFields.lonCenters || lonCenters,
        stride: windFields.stride || 2,
        scale: windFields.scale || 0.24,
      })
      : [];
    setOverlayGroup(globeGroup, windOverlayRef, buildWindOverlay(wind, isLight));

    const direction = showTerminator ? sunDirection : null;
    setOverlayGroup(globeGroup, terminatorRef, buildTerminatorOverlay(direction, isLight));
  }, [field, fieldData, fieldLayers, geometry, isEarth, isLight, sunDirection, showContours, showTerminator, showWindVectors, windFields]);


  // 组件完全卸载时，清空 sphereMeshRef / 材质资源
  useEffect(() => {
    return () => {
      // 先保存视角：星球切换会卸载整个场景，不保存就会跳回默认朝向。
      if (poseKey && cameraRef.current && sphereMeshRef.current) {
        if (suppressedPoseSaves.delete(poseKey)) {
          // 这次卸载来自“重置视角”触发的画布重建：不要写回重置前的视角。
        } else {
          if (cameraPoseCache.size >= CAMERA_POSE_CACHE_LIMIT && !cameraPoseCache.has(poseKey)) {
            // 有界缓存：淘汰最早的一条，避免长期停留时无限增长。
            cameraPoseCache.delete(cameraPoseCache.keys().next().value);
          }
          cameraPoseCache.set(poseKey, {
            cameraPosition: cameraRef.current.position.toArray(),
            cameraQuaternion: cameraRef.current.quaternion.toArray(),
            globeQuaternion: sphereMeshRef.current.quaternion.toArray(),
          });
        }
      }
      if (sphereMeshRef.current && sceneRef.current) {
        sceneRef.current.remove(sphereMeshRef.current);
        disposeObject3D(sphereMeshRef.current);
        sphereMeshRef.current = null;
        particlesMeshRef.current = null;
        particleLayersRef.current = [];
        particleLayerCacheRef.current.clear();
        geoOverlayRef.current = null;
        marsMeshRef.current = null;
        earthGlobeRef.current = null;
        earthCoastlineRef.current = null;
        regionalMeshRef.current = null;
        regionalGeometryRef.current = null;
        selectionMarkerRef.current = null;
        contourOverlayRef.current = null;
        windOverlayRef.current = null;
        terminatorRef.current = null;
        directionalLightRef.current = null;
        pickingMeshRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (sphereMeshRef.current) {
      ensurePickingMesh(sphereMeshRef.current);
    }
  }, [field, fieldData, fieldLayers, geometry, showConcentration, showMars]);

  if (forceFullscreen) {
    return (
      <div
      ref={containerRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
      }}
      />
    );
  }

  // 内嵌状态（如果以后还需要作为内嵌卡片的话）
  return (
    <div
      ref={containerRef}
      className="observation-window"
      style={{
        width: '100%',
        height: h,
        background: isLight ? '#f5f6f8' : 'rgba(0,0,0,0.3)',
        cursor: 'zoom-in',
        overflow: 'hidden',
      }}
    />
  );
});

export default SphericalFieldCanvas;
