"""Earth seasonal coordinate: the solar longitude behind a UTC timestamp.

Earth's seasonal phase is the Sun's apparent ecliptic longitude ``lambda``, which
is the *same* quantity the Mars pipeline stores as ``Ls``: 0 degrees at the
northern spring equinox, 90 at the northern summer solstice, 180 at the autumn
equinox and 270 at the winter solstice. Both planets relate it to solar
declination through one shared expression with different obliquity
(``delta = asin(sin(eps) * sin(lambda))``, 23.44 degrees for Earth and 25.19 for
Mars), so a value produced here is directly comparable with a Mars ``Ls``.

The three-hourly release ships no such coordinate -- its variables are
``TO3``/``U10M``/``V10M``/``T2M``/``SWGDN`` -- so Earth's value is *derived* from
the UTC timestamp rather than read from the package.

This module is deliberately read-only and free of side effects. Nothing in the
training, prediction, cache or checkpoint contracts consumes it yet: handing it
to a model is a separate, contract-changing step that must also freeze
:data:`SEASON_ALGORITHM`, because a derived feature whose formula changes
silently shifts the input distribution of an already trained checkpoint.

Accuracy of the low-precision series is about 0.01 degrees, far below any
physically meaningful resolution at these grid and time scales. The value is
computed from the absolute instant (days since J2000) rather than from a
day-of-year count, so leap years cannot introduce a phase step.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

import numpy as np

#: Producer identity for the derivation. Any contract that persists a seasonal
#: coordinate must record this string, not just the numbers.
SEASON_ALGORITHM = "solar_longitude_meeus_low_precision_v1"

#: Mean obliquity of the Earth's ecliptic at J2000, in degrees.
EARTH_OBLIQUITY_DEG = 23.4392911

#: Terrestrial Time epoch the low-precision series is referenced to.
J2000_EPOCH = np.datetime64("2000-01-01T12:00:00", "s")
SECONDS_PER_DAY = 86400.0
JULIAN_CENTURY_DAYS = 36525.0
FULL_TURN_DEG = 360.0


def _normalize_instant(value: Any) -> Any:
    """Remove the timezone annotations numpy can only parse on a deprecated path.

    Every timestamp in this project is UTC and callers write it as ``...Z``, so
    that marker (and an explicit ``+00:00``) is stripped before conversion. An
    aware ``datetime`` names one unambiguous instant, so it is converted to UTC
    rather than reinterpreted. A string carrying any *other* offset is handed to
    numpy untouched on purpose: a local wall-clock time must never be silently
    treated as UTC.
    """
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii", "replace")
    if isinstance(value, str):
        text = value.strip()
        for marker in ("Z", "z", "+00:00"):
            if text.endswith(marker):
                return text[: -len(marker)]
        return text
    if getattr(value, "tzinfo", None) is not None and hasattr(value, "astimezone"):
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _as_datetime64(when: Any) -> np.ndarray:
    """Convert supported inputs to ``datetime64[s]`` without guessing."""
    raw = np.asarray(when)
    if raw.dtype.kind in ("U", "S", "O"):
        raw = np.array(
            [_normalize_instant(item) for item in raw.reshape(-1)], dtype=object
        ).reshape(raw.shape)
    try:
        dates = raw.astype("datetime64[s]")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Seasonal coordinates require timestamps convertible to datetime64 "
            f"(numpy datetime64, ISO strings or datetime objects), got {when!r}"
        ) from exc
    if dates.dtype.kind != "M":
        raise TypeError(
            "Seasonal coordinates require timestamps convertible to datetime64 "
            f"(numpy datetime64, ISO strings or datetime objects), got {when!r}"
        )
    if np.isnat(dates).any():
        raise ValueError("Seasonal coordinates require every timestamp to be a real instant")
    return dates


def solar_longitude(when: Any) -> np.ndarray:
    """Return the Sun's apparent ecliptic longitude, in degrees ``[0, 360)``.

    Accepts any array-like of timestamps (``numpy.datetime64``, ISO strings or
    ``datetime`` objects) and returns ``float32`` with the input's shape, so a
    ``[batch, window]`` window keeps its shape and matches the existing Mars
    ``ls`` auxiliary-input contract (``[batch, window]``, ``float32``, degree).
    """
    dates = _as_datetime64(when)
    elapsed_days = (
        (dates - J2000_EPOCH) / np.timedelta64(1, "s")
    ) / SECONDS_PER_DAY
    centuries = elapsed_days / JULIAN_CENTURY_DAYS

    # Geometric mean longitude and mean anomaly of the Sun (Meeus, chapter 25).
    mean_longitude = (
        280.46646 + 36000.76983 * centuries + 0.0003032 * centuries * centuries
    )
    mean_anomaly = (
        357.52911 + 35999.05029 * centuries - 0.0001537 * centuries * centuries
    )
    anomaly_rad = np.deg2rad(mean_anomaly)
    equation_of_center = (
        (1.914602 - 0.004817 * centuries - 0.000014 * centuries * centuries)
        * np.sin(anomaly_rad)
        + (0.019993 - 0.000101 * centuries) * np.sin(2.0 * anomaly_rad)
        + 0.000289 * np.sin(3.0 * anomaly_rad)
    )
    true_longitude = mean_longitude + equation_of_center

    # Aberration and nutation in longitude: the apparent-vs-true correction is
    # small, but keeping it means the equinox anchors below hold to 0.01 deg.
    ascending_node = 125.04 - 1934.136 * centuries
    apparent = (
        true_longitude - 0.00569 - 0.00478 * np.sin(np.deg2rad(ascending_node))
    )

    # Wrap in float64 first, then once more in float32: the narrowing cast can
    # round a value just below 360 up to exactly 360, which would break the
    # documented half-open range for a model input.
    wrapped = np.mod(apparent, FULL_TURN_DEG).astype(np.float32)
    return np.mod(wrapped, np.float32(FULL_TURN_DEG))


def solar_declination(when: Any) -> np.ndarray:
    """Return the solar declination in degrees, derived from :func:`solar_longitude`.

    Positive values put the Sun over the northern hemisphere. This is the same
    expression the Mars page uses for its terminator, with Earth's obliquity, so
    it lets a caller cross-check the two planets' seasonal geometry.
    """
    longitude_rad = np.deg2rad(solar_longitude(when).astype(np.float64))
    obliquity_rad = np.deg2rad(EARTH_OBLIQUITY_DEG)
    declination_rad = np.arcsin(np.sin(obliquity_rad) * np.sin(longitude_rad))
    return np.rad2deg(declination_rad).astype(np.float32)


__all__ = [
    "EARTH_OBLIQUITY_DEG",
    "SEASON_ALGORITHM",
    "solar_declination",
    "solar_longitude",
]
