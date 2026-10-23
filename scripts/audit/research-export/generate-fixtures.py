"""Generate clearly labelled synthetic figures for file/layout QA, never inference.

Run with the project backend interpreter and pass an output directory outside Git.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "AstraAtmos_backend" / "backend"))
from schemas.research_export import ResearchExportRequest
from services.research_export_sources import ExportSourceStore
from services.research_figure import prepare_figure, render_figure, build_bundle


def fixtures(curve_models=3):
    store = ExportSourceStore()
    lat = np.linspace(87.5, -87.5, 36)
    lon = np.linspace(-177.5, 177.5, 72)
    x, y = np.meshgrid(lon, lat)
    reference = 8 + 3 * np.cos(np.deg2rad(y * 2)) + np.sin(np.deg2rad(x))
    prediction = reference + .8 * np.sin(np.deg2rad(x * 2)) * np.cos(np.deg2rad(y))
    data = {"ground_truth": [reference], "prediction": [prediction], "residual": [prediction - reference],
            "latitude": [lat], "longitude": [lon],
            **{f"{kind}_{axis}": [lat if axis == "latitude" else lon]
               for kind in ("prediction", "residual") for axis in ("latitude", "longitude")}}
    meta = dict(model="SYNTHETIC LAYOUT FIXTURE (no model inference)", dataset_id="synthetic_layout_fixture",
                planet="mars", scope="current_prediction_window", window=3, horizon=1, ls_values=[93.2],
                input_ls_values=[90], requested_ls_start=90, source_unit="um-atm", synthetic=True)
    mars = store.register(user_id=7, task_id=1, planet="mars", analysis="prediction", horizon=1, origin="90.0",
                          variables=["Temperature"], guard="synthetic", data=data, metadata=meta)
    earth = store.register(user_id=7, task_id=2, planet="earth", analysis="prediction", horizon=1,
                           origin="2021-12-01T00:00:00Z", variables=[], guard="synthetic",
                           data={"reference": [reference * 30], "prediction": [prediction * 30],
                                 "residual": [(prediction - reference) * 30], "latitude": lat, "longitude": lon},
                           metadata={**meta, "planet": "earth", "source_unit": "DU",
                                     "forecast_origin": "2021-12-01T00:00:00Z", "frequency_hours": 3,
                                     "target_timestamps": ["2021-12-01T03:00:00Z"], "origin_split": "test"})
    pfi = store.register(user_id=7, task_id=1, planet="mars", analysis="pfi", horizon=3, origin="",
                         variables=["Temperature"], guard="synthetic",
                         data=dict(items=[dict(name=name, importance=value) for name, value in zip(
                             ["Ozone", "Temperature", "Dust_Optical_Depth", "U_Wind", "V_Wind"], [.23, .11, .045, .007, -.026])],
                                   baseline_metric="r2", baseline_value=.65),
                         metadata={**meta, "horizon": 3, "scope": "sampled_test_set", "sampling": {"sample_size": 40, "test_windows": 300, "method": "synthetic_values_not_measured"}})
    curves = []
    for index in range(curve_models):
        curves.append(store.register(user_id=7, task_id=index + 1, planet="mars", analysis="metrics", horizon=6,
                                      origin="", variables=[], guard="synthetic",
                                      data=dict(per_step=[dict(step=i, rmse=(i * .3 + .6) * (1 + .2 * index),
                                                               mae=(i * .2 + .4) * (1 + .2 * index), r2=.85 - i * .05, ssim=.8) for i in range(1, 7)]),
                                      metadata={**meta, "model": f"Synthetic curve {index + 1}", "horizon": 6, "scope": "full_test_set"}))
    return store, mars, earth, pfi, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--curve-models", type=int, choices=(3, 8), default=3,
                        help="Number of synthetic curves; 8 checks the per-figure limit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store, mars, earth, pfi, curves = fixtures(args.curve_models)
    specs = [
        ("mars-triptych", "triptych", [mars], "en", 180, 60, 10),
        ("earth-triptych-zh", "triptych", [earth], "zh", 180, 60, 10),
        ("mars-triptych-single-zh", "triptych", [mars], "zh", 85, 215, 8),
        ("mars-scatter", "scatter", [mars], "en", 85, 85, 10),
        ("step-curves", "step_curves", curves, "en", 180, 100, 10),
        ("pfi-zh", "pfi", [pfi], "zh", 85, 85, 8),
    ]
    for name, kind, refs, lang, width, height, font in specs:
        for fmt in ("png", "svg", "pdf"):
            req = ResearchExportRequest(sources=refs, kind=kind, options=dict(format=fmt, dpi=300,
                width_mm=width, height_mm=height, language=lang, font_size=font, colormap="viridis"))
            data = prepare_figure([store.get(ref, 7) for ref in refs], req)
            if "target_time" in data:
                data["target_time"] = "SYNTHETIC · " + data["target_time"]
            content = render_figure(data)
            (args.output / f"synthetic-{name}.{fmt}").write_bytes(content)
            if fmt == "pdf":
                (args.output / f"synthetic-{name}-matlab.zip").write_bytes(build_bundle(data, content))
    (args.output / "README.txt").write_text("Every figure here uses synthetic layout data, not real model results.\nMATLAB script execution has not been verified.\n", encoding="utf-8")
    # Browser fixture excludes arrays; UI API mocks use these references.
    (args.output / "browser-refs.json").write_text(json.dumps(dict(mars=mars, earth=earth, pfi=pfi, curves=curves)), encoding="utf-8")
    print(f"Generated 18 synthetic figures and 6 MATLAB bundles in {args.output}")


if __name__ == "__main__":
    main()
