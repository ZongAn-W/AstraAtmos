"""Matplotlib figures and editable MATLAB bundles from immutable result snapshots."""
from io import BytesIO
import json
import textwrap
import threading
import warnings
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

RENDER_LOCK = threading.Lock()  # Matplotlib's font/rc caches are process global.
MAX_CELLS = 1_000_000
MAX_POINTS = 250_000
FEATURE_ZH = {"Ozone": "臭氧", "Temperature": "温度", "Dust_Optical_Depth": "沙尘光学厚度",
              "Solar_Flux_DN": "太阳辐射通量", "U_Wind": "纬向风", "V_Wind": "经向风"}


def _axis(values, size):
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size != size or size < 2 or not np.isfinite(axis).all():
        raise ValueError("Real grid coordinates are missing or invalid; refresh the prediction.")
    delta = np.diff(axis)
    if not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError("Grid coordinates must be strictly monotonic and stay paired with data.")
    return axis


def cell_edges(axis):
    # Half-cell boundaries from real centres; no interpolation or reordering.
    return np.r_[axis[0] - (axis[1] - axis[0]) / 2,
                 (axis[:-1] + axis[1:]) / 2, axis[-1] + (axis[-1] - axis[-2]) / 2]


def prepare_figure(sources, request):
    kind, step, opt = request.kind, request.step, request.options
    analysis = "metrics" if kind == "step_curves" else "pfi" if kind == "pfi" else "prediction"
    if any(source.ref["analysis"] != analysis for source in sources):
        raise ValueError("This result cannot be used for the selected figure type")
    if (kind == "step_curves" and len(sources) < 2) or (kind != "step_curves" and len(sources) != 1):
        raise ValueError("Select two to eight models for curves, or one model for other figures")
    if len({s.ref["planet"] for s in sources}) != 1:
        raise ValueError("Compared models must belong to the same planet")
    if kind not in ("triptych", "step_curves") and any(s.ref["planet"] != "mars" for s in sources):
        raise ValueError("Earth supports triptych and test-set step curves")
    if kind == "step_curves" and sources[0].ref["planet"] == "earth":
        identities = [(s.metadata["dataset_fingerprint"], s.metadata["split_meta"]) for s in sources]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("Earth curves require the same dataset and test windows")
    if len({s.ref["task_id"] for s in sources}) != len(sources):
        raise ValueError("Duplicate models are not allowed")
    if len({s.ref["horizon"] for s in sources}) != 1:
        raise ValueError("Compared results must have the same forecast horizon")
    unit = "DU" if sources[0].ref["planet"] == "earth" else opt.unit
    factor = 0.1 if sources[0].ref["planet"] == "mars" and unit == "DU" else 1.0
    zh = opt.language == "zh"
    labels = dict(longitude="经度 (°)" if zh else "Longitude (°)", latitude="纬度 (°)" if zh else "Latitude (°)",
                  reference="参考" if zh else "Reference", prediction="预测" if zh else "Prediction",
                  residual="预测减参考" if zh else "Prediction - reference", step="预测步" if zh else "Forecast step",
                  unit="μm-atm" if unit == "um-atm" else "DU")
    value_unit = "dimensionless" if kind == "pfi" or (kind == "step_curves" and request.metric in ("r2", "ssim")) else unit
    result = dict(kind=kind, unit=unit, value_unit=value_unit, labels=labels, metadata=[s.metadata for s in sources],
                  options=opt.model_dump(), step=step + 1, metric=request.metric)
    source = sources[0]
    if kind in ("triptych", "scatter"):
        data = source.data
        reference_key = "reference" if source.ref["planet"] == "earth" else "ground_truth"
        if step >= len(data["prediction"]) or step >= source.ref["horizon"]:
            raise ValueError("Selected forecast step is not present in this result")
        reference = data[reference_key][step] * factor
        prediction = data["prediction"][step] * factor
        residual = data["residual"][step] * factor
        if reference.ndim != 2 or reference.size > MAX_CELLS or prediction.shape != reference.shape or residual.shape != reference.shape:
            raise ValueError("Grid shape mismatch or export grid exceeds one million cells")
        if not all(np.isfinite(x).all() for x in (reference, prediction, residual)):
            raise ValueError("Prediction contains missing or non-finite values")
        if not np.allclose(residual, prediction - reference, rtol=1e-5, atol=max(1e-7, np.max(np.abs(prediction)) * 1e-7)):
            raise ValueError("Residual direction is inconsistent; expected prediction minus reference")
        mars = source.ref["planet"] == "mars"
        latitude = _axis(data["latitude"][step] if mars else data["latitude"], reference.shape[0])
        longitude = _axis(data["longitude"][step] if mars else data["longitude"], reference.shape[1])
        if mars:
            for key in ("prediction", "residual"):
                if not np.array_equal(latitude, data[f"{key}_latitude"][step]) or not np.array_equal(longitude, data[f"{key}_longitude"][step]):
                    raise ValueError("Reference, prediction and residual coordinates do not match")
        if np.max(np.abs(latitude)) > 90 or np.max(np.abs(longitude)) > 360:
            raise ValueError("Grid coordinates are outside geographic limits")
        times = source.metadata.get("ls_values") if mars else source.metadata.get("target_timestamps") or source.metadata.get("target_dates")
        if not times or step >= len(times):
            raise ValueError("Real target time/Ls is missing")
        result["target_time"] = f"Ls={float(times[step]):.3f}°" if mars else str(times[step])
        result.update(reference=reference, prediction=prediction, residual=residual,
                      latitude=latitude, longitude=longitude, latitude_edges=cell_edges(latitude), longitude_edges=cell_edges(longitude))
        low, high = min(reference.min(), prediction.min()), max(reference.max(), prediction.max())
        if low == high:
            pad = max(abs(low) * .01, 1e-6 * factor)
            low, high = low - pad, high + pad
        magnitude = float(np.max(np.abs(residual))) or factor
        result.update(physical_limits=np.asarray([low, high]), residual_limits=np.asarray([-magnitude, magnitude]))
        if kind == "scatter":
            # Export every current-step grid cell, or fail explicitly; never silently sample.
            if reference.size > MAX_POINTS:
                raise ValueError("Scatter exceeds 250000 points; select a smaller grid")
            result.update(x=reference.ravel(order="C"), y=prediction.ravel(order="C"), point_order="latitude rows then longitude columns (C order)")
    elif kind == "step_curves":
        curves = []
        for source in sources:
            rows = source.data.get("per_step") or []
            if any(request.metric not in row for row in rows):
                raise ValueError("The selected metric is unavailable for this model")
            x = np.asarray([row["step"] for row in rows], dtype=float)
            y = np.asarray([row[request.metric] for row in rows], dtype=float)
            if len(x) != source.ref["horizon"] or not np.isfinite(x).all() or not np.isfinite(y).all() or not (np.diff(x) > 0).all():
                raise ValueError("Per-step metrics are incomplete or unordered")
            if curves and not np.array_equal(x, curves[0]["x"]):
                raise ValueError("Compared models have different step coordinates")
            curves.append(dict(name=source.metadata["model"], x=x, y=y * (factor if request.metric in ("rmse", "mae") else 1)))
        result["curves"] = curves
    else:
        if source.data.get("baseline_metric") != "r2" or not source.data.get("items"):
            raise ValueError("PFI requires existing R² importance results")
        result["names"] = [FEATURE_ZH.get(row["name"], row["name"]) if zh else row["name"].replace("_", " ") for row in source.data["items"]]
        result["values"] = np.asarray([row["importance"] for row in source.data["items"]], dtype=float)
        if not np.isfinite(result["values"]).all():
            raise ValueError("PFI contains invalid importance values")
        result["baseline_value"] = source.data["baseline_value"]
    return result


def resolve_fonts(choice="auto", language="en"):
    from matplotlib import font_manager as fm
    candidates = {"auto": ["Arial", "DejaVu Sans"], "sans": ["Arial", "DejaVu Sans"],
                  "serif": ["Times New Roman", "DejaVu Serif"]}[choice]
    available = {f.name for f in fm.fontManager.ttflist}
    primary = next(name for name in candidates if name in available)
    cjk = next((name for name in ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Zen Hei"] if name in available), None)
    if language == "zh" and not cjk:
        raise ValueError("A Chinese font is unavailable on this server; use English or configure a CJK font")
    return list(dict.fromkeys([primary] + ([cjk] if cjk else [])))


def _draw(data):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import colormaps
    opt, lab, kind = data["options"], data["labels"], data["kind"]
    fig = Figure(figsize=(opt["width_mm"] / 25.4, opt["height_mm"] / 25.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    width = opt["line_width"]
    colors = ["#0072BD", "#D95319", "#009E73", "#7E2F8E", "#A6761D", "#56B4E9", "#4D4D4D", "#CC79A7"]
    styles = ["-", "--", "-.", ":"]
    title = ""
    if kind == "triptych":
        horizontal = opt["width_mm"] >= 140
        axes = np.asarray(fig.subplots(1, 3) if horizontal else fig.subplots(3, 1)).ravel()
        cmap = "RdBu_r" if opt["colormap"] == "rdbu" else opt["colormap"]
        for index, (ax, key) in enumerate(zip(axes, ["reference", "prediction", "residual"])):
            limits = data["residual_limits"] if key == "residual" else data["physical_limits"]
            mesh = ax.pcolormesh(data["longitude_edges"], data["latitude_edges"], data[key],
                                 cmap="RdBu_r" if key == "residual" else cmap,
                                 vmin=limits[0], vmax=limits[1], shading="flat", rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(lab["longitude"])
            ax.set_ylabel(lab["latitude"])
            ax.set_title((f"({chr(97 + index)}) " if opt["panel_labels"] else "") + lab[key])
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=.9, pad=.06, aspect=25)
            cb.set_label(lab["unit"])
        title = data["target_time"]
        data["colormap_rgb"] = colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
        data["residual_colormap_rgb"] = colormaps["RdBu_r"](np.linspace(0, 1, 256))[:, :3]
    else:
        ax = fig.subplots()
        if kind == "scatter":
            ax.scatter(data["x"], data["y"], s=3, c=colors[0], alpha=.45, linewidths=0, rasterized=len(data["x"]) > 3000)
            limits = data["physical_limits"]
            ax.plot(limits, limits, color=".25", linestyle="--", linewidth=width, label="y = x")
            ax.set(xlim=limits, ylim=limits, xlabel=f'{lab["reference"]} ({lab["unit"]})', ylabel=f'{lab["prediction"]} ({lab["unit"]})')
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper left", frameon=False)
            title = data["target_time"]
        elif kind == "step_curves":
            for index, curve in enumerate(data["curves"]):
                ax.plot(curve["x"], curve["y"], color=colors[index], linestyle=styles[index % 4],
                        marker="o", markersize=3, linewidth=width,
                        label="\n".join(textwrap.wrap(curve["name"], 28)))
            metric = "R²" if data["metric"] == "r2" else data["metric"].upper()
            ax.set_xlabel(lab["step"])
            ax.set_ylabel(metric + (f' ({lab["unit"]})' if data["metric"] in ("rmse", "mae") else ""))
            ax.set_xticks(data["curves"][0]["x"])
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=min(2, len(data["curves"])), frameon=False)
            title = "完整测试集" if opt["language"] == "zh" else "Full test set"
        else:
            ax.barh(data["names"], data["values"], color=colors[0], height=.65)
            ax.invert_yaxis()
            ax.axvline(0, color=".25", linewidth=width)
            ax.set_xlabel("ΔR²")
            title = "抽样测试集 PFI" if opt["language"] == "zh" else "Sampled test-set PFI"
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=width)
    if opt["include_title"]:
        fig.suptitle(title, fontsize=opt["font_size"])
    data["plot_title"] = title
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(width)
    return fig


def render_figure(data, *, preview=False):
    import matplotlib as mpl
    from matplotlib.text import Text
    from matplotlib.ft2font import FT2Font
    from matplotlib import font_manager as fm
    opt = data["options"]
    families = resolve_fonts(opt["font"], opt["language"])
    data["font_families"] = families
    with RENDER_LOCK, mpl.rc_context({"font.family": families, "font.size": opt["font_size"],
                                    "axes.labelsize": opt["font_size"], "axes.titlesize": opt["font_size"],
                                    "xtick.labelsize": opt["font_size"] - 1, "ytick.labelsize": opt["font_size"] - 1,
                                    "legend.fontsize": opt["font_size"] - 1, "pdf.fonttype": 42, "svg.fonttype": "none",
                                    "axes.unicode_minus": False, "text.usetex": False}):
        fig = _draw(data)
        try:
            glyphs = set()
            for family in families:
                glyphs.update(FT2Font(fm.findfont(fm.FontProperties(family=family))).get_charmap())
            texts = fig.findobj(Text)
            if any(ord(char) not in glyphs for artist in texts for char in artist.get_text() if not char.isspace()):
                raise ValueError("Selected fonts cannot render all figure labels; choose another font/language")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig.canvas.draw()
                if any("constrained_layout" in str(w.message) or "Glyph" in str(w.message) for w in caught):
                    raise ValueError("Labels do not fit this size/font. Increase figure dimensions or reduce font size.")
            renderer = fig.canvas.get_renderer()
            bounds = fig.bbox
            # Locators keep extra tick artists outside view limits. Matplotlib
            # does not draw those, so they must not trigger a clipping error.
            hidden_tick_labels = set()
            for ax in fig.axes:
                for axis in (ax.xaxis, ax.yaxis):
                    low, high = sorted(axis.get_view_interval())
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                        if tick.get_loc() < low or tick.get_loc() > high:
                            hidden_tick_labels.update((id(tick.label1), id(tick.label2)))
            for artist in fig.findobj(Text):
                if not artist.get_visible() or not artist.get_text() or id(artist) in hidden_tick_labels:
                    continue
                box = artist.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width + 1 or box.y1 > bounds.height + 1:
                    raise ValueError("Labels exceed the figure boundary. Increase figure size or reduce font size.")
            buffer = BytesIO()
            fmt = "png" if preview else opt["format"]
            fig.savefig(buffer, format=fmt, dpi=120 if preview else opt["dpi"], facecolor="white")
            return buffer.getvalue()
        finally:
            fig.clear()


MATLAB_SCRIPT = r'''% AstraAtmos scientific figure reconstruction (MATLAB R2020b+).
% Run this script in the extracted folder. No training/inference is performed.
% MATLAB execution has not been verified by the exporter.
folder = fileparts(mfilename('fullpath'));
S = load(fullfile(folder, 'figure.mat')); D = S.figure_data;
o = D.options; L = D.labels;
f = figure('Color','w','Units','centimeters', ...
    'Position',[2 2 double(o.width_mm)/10 double(o.height_mm)/10]);
set(f,'PaperUnits','centimeters','PaperSize',[double(o.width_mm)/10 double(o.height_mm)/10]);
set(f,'PaperPosition',[0 0 double(o.width_mm)/10 double(o.height_mm)/10],'PaperPositionMode','manual');
set(f,'DefaultAxesFontName',D.font_families{1},'DefaultTextFontName',D.font_families{1}, ...
    'DefaultAxesFontSize',double(o.font_size),'DefaultTextFontSize',double(o.font_size));
% For Chinese labels select an installed CJK font (e.g. Microsoft YaHei).
if strcmp(o.language,'zh') && numel(D.font_families)>1
    set(f,'DefaultAxesFontName',D.font_families{end},'DefaultTextFontName',D.font_families{end});
end
colors = [0 .447 .741; .85 .325 .098; 0 .62 .45; .494 .184 .556; .65 .46 .11; .337 .706 .914; .3 .3 .3; .8 .475 .655];
styles = {'-','--','-.',':'};
switch D.kind
    case 'triptych'
        if double(o.width_mm)>=140, T=tiledlayout(1,3); else, T=tiledlayout(3,1); end
        T.TileSpacing='compact'; T.Padding='compact';
        fields={D.reference,D.prediction,D.residual}; labels={L.reference,L.prediction,L.residual};
        for k=1:3
            ax=nexttile(T); C=double(fields{k});
            % Flat cells on true (possibly irregular/descending) coordinate edges.
            [X,Y]=meshgrid(double(D.longitude_edges),double(D.latitude_edges));
            C=[C C(:,end); C(end,:) C(end,end)];
            surface(ax,X,Y,zeros(size(X)),C,'FaceColor','flat','EdgeColor','none'); view(ax,2);
            axis(ax,'equal'); axis(ax,'tight'); set(ax,'YDir','normal');
            if k==3, caxis(ax,double(D.residual_limits)); colormap(ax,D.residual_colormap_rgb);
            else, caxis(ax,double(D.physical_limits)); colormap(ax,D.colormap_rgb); end
            cb=colorbar(ax,'southoutside'); cb.Label.String=L.unit;
            xlabel(ax,L.longitude); ylabel(ax,L.latitude);
            prefix=''; if o.panel_labels, prefix=sprintf('(%s) ',char(96+k)); end
            title(ax,[prefix labels{k}],'Interpreter','none');
        end
        if o.include_title, title(T,D.target_time,'Interpreter','none'); end
    case 'scatter'
        ax=axes(f); scatter(ax,double(D.x),double(D.y),3,colors(1,:),'filled','MarkerFaceAlpha',.45); hold(ax,'on');
        h=plot(ax,D.physical_limits,D.physical_limits,'--','Color',[.25 .25 .25],'LineWidth',double(o.line_width));
        xlim(ax,D.physical_limits); ylim(ax,D.physical_limits); axis(ax,'square');
        xlabel(ax,[L.reference ' (' L.unit ')']); ylabel(ax,[L.prediction ' (' L.unit ')']);
        legend(ax,h,'y = x','Location','northwest','Box','off');
        if o.include_title, title(ax,D.target_time,'Interpreter','none'); end
    case 'step_curves'
        ax=axes(f); hold(ax,'on'); names=cell(1,numel(D.curves));
        for k=1:numel(D.curves)
            c=D.curves(k); if iscell(D.curves), c=D.curves{k}; end
            plot(ax,c.x,c.y,'Color',colors(k,:),'LineStyle',styles{mod(k-1,4)+1}, ...
                'Marker','o','MarkerSize',3,'LineWidth',double(o.line_width)); names{k}=c.name;
        end
        xlabel(ax,L.step); metric=upper(D.metric); if strcmp(D.metric,'r2'), metric='R²'; end
        if ismember(D.metric,{'rmse','mae'}), metric=[metric ' (' L.unit ')']; end
        ylabel(ax,metric); xticks(ax,D.curves(1).x);
        legend(ax,names,'Location','southoutside','NumColumns',2,'Interpreter','none');
        if o.include_title, title(ax,D.plot_title,'Interpreter','none'); end
    case 'pfi'
        ax=axes(f); barh(ax,double(D.values),'FaceColor',colors(1,:));
        set(ax,'YTick',1:numel(D.names),'YTickLabel',D.names,'YDir','reverse','TickLabelInterpreter','none');
        xline(ax,0,'Color',[.25 .25 .25],'LineWidth',double(o.line_width)); xlabel(ax,'ΔR²');
        if o.include_title, title(ax,D.plot_title,'Interpreter','none'); end
end
set(findall(f,'Type','axes'),'LineWidth',double(o.line_width),'TickDir','out','Box','off');
set(findall(f,'Type','text'),'Interpreter','none');
% Edit fonts, legend, axes and tiledlayout here, then export:
% print(f,'adjusted.pdf','-dpdf','-painters');
% print(f,'adjusted.png','-dpng',sprintf('-r%d',double(o.dpi)));
'''


def build_bundle(data, figure_bytes):
    from scipy.io import savemat

    # JSON retains complete metadata; MAT fields contain exactly plotted values.
    public_meta = dict(schema="astraatmos_research_figure_v1", kind=data["kind"], unit=data["unit"], value_unit=data["value_unit"],
                       selected_step=data["step"] if data["kind"] in ("triptych", "scatter") else None,
                       target_time=data.get("target_time"), metric=data["metric"],
                       sources=data["metadata"], options=data["options"], font_families=data["font_families"],
                       residual_definition="prediction - reference", matlab_execution_verified=False)
    mat_data = {key: value for key, value in data.items() if key != "metadata"}
    # MATLAB loads homogeneous curve structures as a struct array.
    if "curves" in mat_data:
        curves = np.empty((1, len(data["curves"])), dtype=[("name", "O"), ("x", "O"), ("y", "O")])
        for index, curve in enumerate(data["curves"]):
            curves[0, index] = (curve["name"], curve["x"], curve["y"])
        mat_data["curves"] = curves
    mat_data["font_families"] = np.asarray(data["font_families"], dtype=object)
    if "names" in mat_data:
        mat_data["names"] = np.asarray(mat_data["names"], dtype=object)
    captions = []
    for meta in data["metadata"]:
        captions.append(f"Model: {meta['model']}; dataset: {meta['dataset_id']}; window: {meta['window']}; "
                        f"horizon: {meta['horizon']}; scope: {meta['scope']}; "
                        f"origin: {meta.get('forecast_origin', meta.get('requested_ls_start', 'not applicable (test set)'))}; "
                        f"target: {data.get('target_time', 'test-set forecast steps')}; "
                        f"selected step: {public_meta['selected_step'] or 'all test-set forecast steps'}; value unit: {data['value_unit']}. "
                        + (f"PFI sampling: {json.dumps(meta.get('sampling'), ensure_ascii=False)}." if data["kind"] == "pfi" else ""))
    public_meta["suggested_caption"] = "\n".join(captions)
    mat_data["metadata_json"] = json.dumps(public_meta, ensure_ascii=False)
    mat = BytesIO()
    savemat(mat, {"figure_data": mat_data}, do_compression=True, long_field_names=True, oned_as="row")
    note = public_meta["suggested_caption"] + "\n\n"
    if data["kind"] == "triptych":
        note += "Residual = prediction - reference. Reference and prediction share color limits.\n"
    note += "Scatter uses all current-step grid cells, with equal axes and y=x; no density estimate.\n" if data["kind"] == "scatter" else ""
    note += "Coordinates retain source order; flat cells use half-cell boundaries from real centres, without smoothing/interpolation.\n"
    note += "Open figure.mat or run rebuild.m in MATLAB R2020b+ to edit the figure. Font availability and layout can vary by MATLAB version.\nMATLAB script execution was not verified. metadata.json contains the suggested caption and complete scope information.\n"
    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        archive.writestr(f"figure.{data['options']['format']}", figure_bytes)
        archive.writestr("figure.mat", mat.getvalue())
        archive.writestr("rebuild.m", MATLAB_SCRIPT.encode("utf-8"))
        archive.writestr("README.txt", note.encode("utf-8"))
        archive.writestr("metadata.json", json.dumps(public_meta, ensure_ascii=False, indent=2).encode("utf-8"))
    return out.getvalue()
