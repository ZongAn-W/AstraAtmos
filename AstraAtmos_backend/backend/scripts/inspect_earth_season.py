"""Print the derived Earth seasonal coordinate for the published three-hour axis.

Read-only inspection of :mod:`services.earth_season`: no dataset package, no
checkpoint, no contract. The defaults mirror the released
``earth_merra2_3hourly_v1`` geometry (01:30 UTC centre labels, three-hour
cadence, 5848 steps = 731 days), so the printed series is the series a model
would actually receive.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.earth_season import SEASON_ALGORITHM, solar_declination, solar_longitude

DEFAULT_START = "2020-01-01T01:30"
DEFAULT_COUNT = 5848
DEFAULT_STEP_HOURS = 3
DEFAULT_WINDOW = 56
QUADRANTS = ((0.0, "March equinox"), (90.0, "June solstice"),
             (180.0, "September equinox"), (270.0, "December solstice"))


def iso_z(value):
    return np.datetime_as_string(np.datetime64(value, "s"), unit="s") + "Z"


def timeline(start=DEFAULT_START, count=DEFAULT_COUNT, step_hours=DEFAULT_STEP_HOURS):
    """Build the axis from whole seconds, so a fractional step is still exact."""
    seconds = int(round(float(step_hours) * 3600.0))
    if seconds < 1:
        raise ValueError("step must be at least one second")
    return np.datetime64(start, "s") + np.arange(count) * np.timedelta64(seconds, "s")


def season_boundaries(dates, longitude):
    """Report the first sample inside each calendar quadrant of the solar year."""
    quadrant = np.floor(longitude.astype("float64") / 90.0).astype("int64")
    changes = np.flatnonzero(np.diff(quadrant) != 0) + 1
    boundaries = []
    for index in changes.tolist():
        degree, label = QUADRANTS[int(quadrant[index])]
        boundaries.append({
            "season": label, "longitude_deg": degree,
            "utc": iso_z(dates[index]), "step_index": int(index),
            "longitude_at_sample": round(float(longitude[index]), 4),
        })
    return boundaries


def monthly_summary(dates, longitude):
    """Summarise each month without ever averaging across the equinox wrap.

    An arithmetic mean of longitudes is meaningless for a month that contains the
    359 -> 0 crossing (it would report roughly 180 for late March), so each month
    reports its first, middle and last sample instead.
    """
    labels = np.datetime_as_string(dates, unit="D").reshape(-1).tolist()
    months = np.array([label[:7] for label in labels])
    rows = []
    for month in sorted(set(months.tolist())):
        selected = np.flatnonzero(months == month)
        rows.append({
            "month": month,
            "steps": int(selected.size),
            "first_utc": iso_z(dates[selected[0]]),
            "last_utc": iso_z(dates[selected[-1]]),
            "longitude_start_deg": round(float(longitude[selected[0]]), 3),
            "longitude_mid_deg": round(float(longitude[selected[selected.size // 2]]), 3),
            "longitude_end_deg": round(float(longitude[selected[-1]]), 3),
        })
    return rows


def build_report(args):
    dates = timeline(args.start, args.count, args.step_hours)
    longitude = solar_longitude(dates)
    window = min(int(args.window), len(dates))
    first_window = longitude[:window]
    span = float(first_window.max() - first_window.min())
    days = float((dates[-1] - dates[0]) / np.timedelta64(1, "D"))
    return {
        "algorithm": SEASON_ALGORITHM,
        "unit": "degree",
        "definition": "apparent solar longitude; 0 = March equinox, 90 = June solstice",
        "timeline": {
            "start_utc": iso_z(dates[0]), "end_utc": iso_z(dates[-1]),
            "steps": int(len(dates)), "step_hours": float(args.step_hours),
            "span_days": round(days, 4),
        },
        "range": {
            "dtype": str(longitude.dtype),
            "min": round(float(longitude.min()), 4),
            "max": round(float(longitude.max()), 4),
            "wraps": int((np.diff(longitude) < 0).sum()),
        },
        "first_step": {"utc": iso_z(dates[0]), "longitude_deg": round(float(longitude[0]), 4),
                       "declination_deg": round(float(solar_declination(dates[0])), 4)},
        "last_step": {"utc": iso_z(dates[-1]), "longitude_deg": round(float(longitude[-1]), 4),
                      "declination_deg": round(float(solar_declination(dates[-1])), 4)},
        "window": {
            "steps": window,
            "span_hours": round((window - 1) * float(args.step_hours), 3),
            "longitude_span_deg": round(span, 4),
        },
        "season_boundaries": season_boundaries(dates, longitude),
        "months": monthly_summary(dates, longitude),
        "head": [
            {"index": int(index), "utc": iso_z(dates[index]),
             "longitude_deg": round(float(longitude[index]), 4)}
            for index in range(min(int(args.head), len(dates)))
        ],
        "tail": [
            {"index": int(index), "utc": iso_z(dates[index]),
             "longitude_deg": round(float(longitude[index]), 4)}
            for index in range(max(0, len(dates) - int(args.head)), len(dates))
        ],
    }


def print_table(report):
    timeline_info = report["timeline"]
    print(f"Earth seasonal coordinate (derived) - {report['algorithm']}")
    print(f"definition      {report['definition']}  [{report['unit']}]")
    print(f"timeline        {timeline_info['start_utc']} .. {timeline_info['end_utc']}")
    print(f"steps           {timeline_info['steps']} x {timeline_info['step_hours']:g} h "
          f"= {timeline_info['span_days']} days")
    print(f"range           [{report['range']['min']}, {report['range']['max']}] "
          f"{report['range']['dtype']}, wraps {report['range']['wraps']}")
    print(f"first step      {report['first_step']['utc']}  ls={report['first_step']['longitude_deg']:>8.4f}"
          f"  dec={report['first_step']['declination_deg']:>7.4f}")
    print(f"last step       {report['last_step']['utc']}  ls={report['last_step']['longitude_deg']:>8.4f}"
          f"  dec={report['last_step']['declination_deg']:>7.4f}")
    print(f"window          {report['window']['steps']} steps "
          f"({report['window']['span_hours']:g} h) spans "
          f"{report['window']['longitude_span_deg']:.4f} deg")
    print("\nseason boundaries (first sample inside each quadrant)")
    for row in report["season_boundaries"]:
        print(f"  {row['season']:<18} ls={row['longitude_deg']:>5.1f}  {row['utc']}  "
              f"index {row['step_index']}")
    print("\nmonths (start / middle / end sample; never averaged across the wrap)")
    for row in report["months"]:
        bar = "#" * int(round(row["longitude_mid_deg"] / 360.0 * 48))
        print(f"  {row['month']}  {row['longitude_start_deg']:>7.3f} -> "
              f"{row['longitude_mid_deg']:>7.3f} -> {row['longitude_end_deg']:>7.3f}  "
              f"{row['steps']:>3} steps  {bar}")
    print(f"\nhead ({len(report['head'])} steps)")
    for row in report["head"]:
        print(f"  {row['index']:>5}  {row['utc']}  {row['longitude_deg']:>8.4f}")
    print(f"tail ({len(report['tail'])} steps)")
    for row in report["tail"]:
        print(f"  {row['index']:>5}  {row['utc']}  {row['longitude_deg']:>8.4f}")


def write_csv(path, dates, longitude):
    lines = ["step_index,utc,solar_longitude_deg,solar_declination_deg"]
    declination = solar_declination(dates)
    for index in range(len(dates)):
        lines.append(f"{index},{iso_z(dates[index])},{float(longitude[index]):.6f},"
                     f"{float(declination[index]):.6f}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=DEFAULT_START,
                        help="first UTC centre label (default: the released axis)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--step-hours", type=float, default=DEFAULT_STEP_HOURS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="input window length used for the span summary")
    parser.add_argument("--head", type=int, default=8,
                        help="rows printed for the head and tail of the series")
    parser.add_argument("--csv", help="write the full series to this path")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    report = build_report(args)
    if args.csv:
        dates = timeline(args.start, args.count, args.step_hours)
        report["csv"] = {"path": args.csv,
                         "rows": write_csv(args.csv, dates, solar_longitude(dates))}
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_table(report)
        if args.csv:
            print(f"\ncsv             {report['csv']['rows']} rows -> {report['csv']['path']}")


if __name__ == "__main__":
    main()
