"""Earth seasonal coordinate: external anchors, release geometry, input contract.

The derivation is checked against published equinox and solstice instants rather
than against itself, and against the geometry of the real 5848-step release axis
(01:30 UTC centre labels, three-hour cadence, 731 days).
"""

import numpy as np
import pytest

from services.earth_season import (
    EARTH_OBLIQUITY_DEG,
    SEASON_ALGORITHM,
    solar_declination,
    solar_longitude,
)

RELEASE_START = np.datetime64("2020-01-01T01:30", "s")
RELEASE_COUNT = 5848
RELEASE_STEP = np.timedelta64(3, "h")
SECONDS_PER_DAY = 86400

#: Published equinox and solstice instants (UTC), northern-hemisphere convention:
#: 0 = March equinox, 90 = June solstice, 180 = September equinox, 270 = December
#: solstice. The tolerance is 0.05 degrees, about 73 minutes of solar motion, so
#: this stays robust to the minute-level precision of the published times while
#: still failing a formula that is a season off or badly phased.
ANCHORS = (
    ("2020-03-20T03:50", 0.0),
    ("2020-06-20T21:44", 90.0),
    ("2020-09-22T13:31", 180.0),
    ("2020-12-21T10:02", 270.0),
    ("2021-03-20T09:37", 0.0),
    ("2021-06-21T03:32", 90.0),
    ("2021-09-22T19:21", 180.0),
    ("2021-12-21T15:59", 270.0),
)
ANCHOR_TOLERANCE_DEG = 0.05


def release_timeline(count=RELEASE_COUNT):
    """The published three-hourly axis, as the task split module builds it."""
    return RELEASE_START + np.arange(count) * RELEASE_STEP


def signed_longitude(instant):
    """Return the longitude on a continuous branch, so the wrap stays sign-safe."""
    return ((float(solar_longitude(instant)) + 180.0) % 360.0) - 180.0


def circular_delta(value, expected):
    """Distance on the circle: 359.9979 is 0.0021 away from 0, not 359.9979."""
    return abs((float(value) - float(expected) + 180.0) % 360.0 - 180.0)


def find_zero_crossing(start, span_seconds):
    """Bisect the March-equinox crossing of the continuous branch, to one minute."""
    base = np.datetime64(start, "s")
    lo, hi = 0, int(span_seconds)
    assert signed_longitude(base) < 0.0 < signed_longitude(
        base + np.timedelta64(hi, "s")
    )
    while hi - lo > 60:
        mid = (lo + hi) // 2
        if signed_longitude(base + np.timedelta64(mid, "s")) < 0.0:
            lo = mid
        else:
            hi = mid
    return base + np.timedelta64((lo + hi) // 2, "s")


@pytest.mark.parametrize("instant,expected_deg", ANCHORS)
def test_published_equinox_and_solstice_instants_anchor_the_series(instant, expected_deg):
    """Every anchor lands within tolerance of the longitude it defines."""
    delta = circular_delta(solar_longitude(instant), expected_deg)
    assert delta <= ANCHOR_TOLERANCE_DEG, f"{instant}: {delta:.4f} deg from {expected_deg}"


def test_march_equinox_zero_crossing_matches_the_published_instant():
    """The crossing itself, not just a sample near it, matches the almanac."""
    crossing = find_zero_crossing("2020-03-19T00:00:00", 2 * SECONDS_PER_DAY)
    published = np.datetime64("2020-03-20T03:50", "s")
    offset_minutes = abs(float((crossing - published) / np.timedelta64(1, "m")))
    assert offset_minutes <= 25.0, f"crossing {crossing} vs published {published}"


def test_release_axis_geometry_and_range():
    """The real release axis keeps the documented shape, dtype and half-open range."""
    longitude = solar_longitude(release_timeline())
    assert longitude.shape == (RELEASE_COUNT,)
    assert longitude.dtype == np.float32
    assert np.isfinite(longitude).all()
    assert float(longitude.min()) >= 0.0 and float(longitude.max()) < 360.0

    # 2020-01-01 is ten days past the December solstice: deep northern winter.
    assert 279.8 <= float(longitude[0]) <= 280.5
    # 731 days minus one three-hour step is almost exactly two tropical years, so
    # the series returns close to where it started, having wrapped twice.
    assert abs(float(longitude[-1]) - float(longitude[0])) < 1.0
    assert int((np.diff(longitude) < 0).sum()) == 2


def test_seven_day_window_spans_about_seven_degrees():
    """A default 56-step window advances by the annual rate over its 55 intervals."""
    window = solar_longitude(
        np.datetime64("2020-06-01T01:30", "s") + np.arange(56) * RELEASE_STEP
    )
    span = float(window.max() - window.min())  # June: nowhere near the wrap
    assert 6.5 <= span <= 7.0, f"56-step window spans {span:.4f} deg"


def test_values_advance_by_the_annual_rate_not_by_day_of_year():
    """One year later is one tropical year minus a day, never a repeated phase."""
    one_year_apart = solar_longitude(["2020-03-01T01:30", "2021-03-01T01:30"])
    signed = ((float(one_year_apart[1]) - float(one_year_apart[0]) + 180.0) % 360.0) - 180.0
    # 365 days of a 365.2422-day year: short of a full turn by about 0.24 degrees.
    assert -0.35 <= signed <= -0.15, f"signed advance {signed:.4f} deg"
    # A day-of-year formula would return exactly zero here.
    assert abs(signed) > 0.15


def test_declination_follows_the_obliquity():
    """Declination is zero at the equinoxes and the obliquity at the solstices."""
    june = float(solar_declination("2020-06-20T21:44"))
    december = float(solar_declination("2020-12-21T10:02"))
    assert 23.35 <= june <= 23.50
    assert -23.50 <= december <= -23.35
    assert abs(june - EARTH_OBLIQUITY_DEG) <= 0.10
    for equinox in ("2020-03-20T03:50", "2020-09-22T13:31"):
        assert abs(float(solar_declination(equinox))) <= 0.05


def test_half_open_range_survives_the_wrap():
    """Sampling densely across the crossing never yields 360 or a negative value."""
    fan = np.datetime64("2020-03-20T03:00:00", "s") + np.arange(0, 7200, 30) * np.timedelta64(1, "s")
    longitude = solar_longitude(fan)
    assert float(longitude.min()) >= 0.0 and float(longitude.max()) < 360.0
    # Both sides of the wrap are present, so the guard above was actually exercised.
    assert float(longitude.max()) > 359.9 and float(longitude.min()) < 0.1


def test_shape_dtype_and_scalar_are_preserved():
    """A window keeps its shape; a scalar input stays scalar; strings convert."""
    window = solar_longitude(release_timeline(56)[None, :].repeat(2, axis=0))
    assert window.shape == (2, 56)
    assert window.dtype == np.float32

    scalar = solar_longitude(np.datetime64("2020-06-20T21:44", "s"))
    assert scalar.shape == ()
    assert scalar.dtype == np.float32
    assert circular_delta(scalar, 90.0) <= ANCHOR_TOLERANCE_DEG

    from_string = solar_longitude(["2020-06-20T21:44Z", "2020-12-21T10:02Z"])
    assert from_string.shape == (2,)
    assert circular_delta(from_string[0], 90.0) <= ANCHOR_TOLERANCE_DEG
    assert circular_delta(from_string[1], 270.0) <= ANCHOR_TOLERANCE_DEG


def test_utc_marker_is_accepted_without_a_deprecation_warning():
    """The project writes instants as ``...Z``; numpy's tz parsing must not run."""
    import warnings

    from datetime import datetime, timezone

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        marked = solar_longitude("2020-06-20T21:44Z")
        bare = solar_longitude("2020-06-20T21:44")
        objects = solar_longitude([datetime(2020, 6, 20, 21, 44, tzinfo=timezone.utc)])
    assert float(marked) == float(bare)
    assert circular_delta(marked, 90.0) <= ANCHOR_TOLERANCE_DEG
    assert circular_delta(objects[0], 90.0) <= ANCHOR_TOLERANCE_DEG


def test_repeated_calls_are_deterministic():
    """The derivation is pure: identical inputs give bit-identical output."""
    timeline = release_timeline(512)
    first = solar_longitude(timeline)
    second = solar_longitude(timeline)
    assert np.array_equal(first, second)
    assert first.tobytes() == second.tobytes()


def test_invalid_timestamps_are_rejected():
    """A missing or unparsable instant fails loudly instead of becoming a value."""
    with pytest.raises(ValueError):
        solar_longitude(np.datetime64("NaT"))
    with pytest.raises(TypeError):
        solar_longitude("not-a-date")
    with pytest.raises(TypeError):
        solar_longitude([object()])


def test_algorithm_identity_is_stable():
    """The producer identity is persisted by future contracts, so it is frozen here."""
    assert SEASON_ALGORITHM == "solar_longitude_meeus_low_precision_v1"
