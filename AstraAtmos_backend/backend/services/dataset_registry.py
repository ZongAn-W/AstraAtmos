"""Public server dataset catalog.

The registry is the single authority for dataset identity: fixed ids, release
versions, artifact fingerprints and availability. Daily Earth releases retain
pinned hashes; three-hour releases verify their configured manifest and content.
Results are cached per file signature; Mars entries stay
``unverified`` because this stage does not probe the large Mars directories.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from services.dataset_identity import (
    ACTIVE_DATASET_IDS,
    DATASET_IDS,
    EARTH_DATASET_ID,
    EARTH_DATASET_V2_ID,
    EARTH_DATASET_3HOURLY_ID,
    EARTH_DATASET_3HOURLY_5DEG_ID,
    EARTH_THREE_HOURLY_DATASET_IDS,
    EARTH_DATASET_IDS,
    IDENTITY_STATUS_UNVERSIONED,
    IDENTITY_STATUS_VERIFIED,
    REGISTERED_DATASET_IDS,
    REGISTRY_BINDING_BASIS,
    UNVERSIONED_VERSION_STATUS,
    DatasetRequestError,
    build_identity_snapshot,
    require_training_dataset,
    require_active_dataset,
)
from services.earth_training_contract import earth_training_profile
from services.earth_dataset_metadata import (
    DATA_FILE_NAME,
    REASON_PACKAGE_CHANGED,
    REASON_PACKAGE_MISSING,
    REASON_PACKAGE_UNREADABLE,
    EarthPackageError,
    VerifiedEarthRelease,
    package_signature,
    read_earth_release,
    read_earth_3hourly_release,
)

logger = logging.getLogger("astraatmos.datasets")

# Pinned release of data/earth/merra2_daily_v1 (731 days, 2020-01-01..2021-12-31).
EXPECTED_MANIFEST_SHA256 = "1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e"
EXPECTED_DATA_SHA256 = "c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74"

# Pinned global v2 release generated from raw hourly MERRA-2.  Keeping hashes in source makes a
# release immutable: changing either file makes the registry report invalid.
EXPECTED_V2_MANIFEST_SHA256 = "935ca37bd3064772a370db6873b605e12b85c031281c2b34d8fbc49ab11f0702"
EXPECTED_V2_DATA_SHA256 = "d280a17cb291e1568ccedd1638cb2fadcc5cb8d89bb8ed0d4d51987e8e181396"

EARTH_DATASET_VERSION = "v1"
EARTH_DISPLAY_NAME = "MERRA-2 daily ozone compact v1"
EARTH_V2_DATASET_VERSION = "v2"
EARTH_V2_DISPLAY_NAME = "MERRA-2 daily global 5 degree compact v2"
EARTH_SCHEMA = "astraatmos_earth_daily_v1"
EARTH_3HOURLY_SCHEMA = "astraatmos_earth_3hourly_v1"
EARTH_3HOURLY_DATA_FILE = "earth_merra2_3hourly.nc"
EARTH_3HOURLY_DISPLAY_NAME = "MERRA-2 3-hourly global 0.75 degree v1"
EARTH_3HOURLY_5DEG_DISPLAY_NAME = "MERRA-2 3-hourly global 5 degree v1"


def earth_3hourly_training_profile() -> dict:
    """Publish the official DLinear contract for the independent release."""
    return earth_training_profile(EARTH_DATASET_3HOURLY_ID)

LEGACY_DATASET_REASON = "legacy_dataset_not_probed"

MARS_DISPLAY_NAMES = {
    "openmars_mcd": "OpenMARS + MCD ozone",
    "mcd_overview": "MCD overview fields",
}

# Entrance wiring, not data availability: both Mars datasets can already start a
# training run and drive trained-model prediction.
MARS_CAPABILITIES = {
    "metadata": True,
    "training": True,
    "web_overview": False,
    "trained_prediction": True,
}
EARTH_DESCRIPTOR_CAPABILITIES = {
    "metadata": True,
    # The 2D overview entry is wired for this dataset. Data requests still
    # require availability=available, so a missing package keeps returning 503.
    "web_overview": True,
    # Earth training and historical prediction are wired for both Earth releases
    # (the v1 legacy package and the default v2). Entrance wiring is independent
    # of whether the package is currently verifiable: `availability` stays the
    # separate gate, exactly as it is for the overview.
    "training": True,
    "trained_prediction": True,
}

MARS_TIME = {
    "kind": "mars_ls",
    "calendar": None,
    "start": None,
    "end": None,
    "count": None,
    "step": None,
    "step_unit": None,
}

MARS_LIMITATIONS = [
    "Legacy server directory; no immutable release version or content fingerprint",
    "Availability is not probed by the dataset registry in this stage",
]

# A published manifest may describe the state of the application at build time.
# Those sentences go stale once the entry is wired, so they are filtered out of
# the public descriptor; the physical data limitations are kept verbatim.
_STALE_LIMITATION_MARKERS = (
    "web overview",
    "web/api dataset registry",
    "not connected yet",
    "training and prediction remain disabled",
    "training and prediction are not connected",
    "尚未接入",
    "未接入",
)
EARTH_APPLICATION_LIMITATIONS = [
    "Coverage mean refers to the published covered area; it is not a total atmospheric mass",
    "Earth prediction is limited to historical forecast origins that have published reference days",
    "Earth and Mars results cannot be compared in one request yet",
]


def _physical_limitations(limitations: Any) -> list[str]:
    if not isinstance(limitations, list):
        return []
    kept = []
    for item in limitations:
        text = str(item)
        lowered = text.lower()
        if any(marker.lower() in lowered for marker in _STALE_LIMITATION_MARKERS):
            logger.debug("Dropping stale manifest limitation: %s", text)
            continue
        kept.append(text)
    return kept


def _blank_dynamic_earth_fields() -> dict:
    """Dynamic Earth fields for a package that is missing or not verifiable."""
    return {
        "schema": EARTH_SCHEMA,
        "manifest_sha256": None,
        "data_sha256": None,
        "dataset_fingerprint": None,
        "time": {
            "kind": "date",
            "calendar": None,
            "start": None,
            "end": None,
            "count": None,
            "step": None,
            "step_unit": None,
        },
        "grid": None,
        "channel_order": [],
        "variables": [],
        "splits": None,
        "limitations": [],
    }


class DatasetRegistry:
    """Read-only catalog over the server managed datasets."""

    def __init__(
        self,
        earth_package_dir: Any,
        *,
        expected_manifest_sha256: Optional[str] = None,
        expected_data_sha256: Optional[str] = None,
        data_file_name: str = DATA_FILE_NAME,
        earth_dataset_id: Optional[str] = None,
        legacy_earth_package_dir: Any = None,
        earth_3hourly_package_dir: Any = None,
        earth_3hourly_5deg_package_dir: Any = None,
        verification_cache_dir: Any = None,
        progress=None,
    ):
        self._earth_package_dir = Path(earth_package_dir).expanduser()
        primary_id = earth_dataset_id or (
            EARTH_DATASET_V2_ID if self._earth_package_dir.name == "merra2_daily_v2" else EARTH_DATASET_ID
        )
        if primary_id not in (EARTH_DATASET_ID, EARTH_DATASET_V2_ID):
            raise ValueError("Unknown Earth release identity")
        if expected_manifest_sha256 is None:
            expected_manifest_sha256 = EXPECTED_V2_MANIFEST_SHA256 if primary_id == EARTH_DATASET_V2_ID else EXPECTED_MANIFEST_SHA256
        if expected_data_sha256 is None:
            expected_data_sha256 = EXPECTED_V2_DATA_SHA256 if primary_id == EARTH_DATASET_V2_ID else EXPECTED_DATA_SHA256
        self._earth_specs = {
            primary_id: {
                "path": self._earth_package_dir,
                "manifest": expected_manifest_sha256,
                "data": expected_data_sha256,
                "version": "v1" if primary_id == EARTH_DATASET_ID else EARTH_V2_DATASET_VERSION,
                "display": EARTH_DISPLAY_NAME if primary_id == EARTH_DATASET_ID else EARTH_V2_DISPLAY_NAME,
            }
        }
        secondary_id = EARTH_DATASET_V2_ID if primary_id == EARTH_DATASET_ID else EARTH_DATASET_ID
        secondary_path = self._earth_package_dir.parent / (
            "merra2_daily_v2" if secondary_id == EARTH_DATASET_V2_ID else "merra2_daily_v1"
        )
        if secondary_id == EARTH_DATASET_ID and legacy_earth_package_dir is not None:
            secondary_path = Path(legacy_earth_package_dir).expanduser()
        secondary_manifest = EXPECTED_V2_MANIFEST_SHA256 if secondary_id == EARTH_DATASET_V2_ID else EXPECTED_MANIFEST_SHA256
        secondary_data = EXPECTED_V2_DATA_SHA256 if secondary_id == EARTH_DATASET_V2_ID else EXPECTED_DATA_SHA256
        self._earth_specs[secondary_id] = {
            "path": secondary_path, "manifest": secondary_manifest, "data": secondary_data,
            "version": "v2" if secondary_id == EARTH_DATASET_V2_ID else "v1",
            "display": EARTH_V2_DISPLAY_NAME if secondary_id == EARTH_DATASET_V2_ID else EARTH_DISPLAY_NAME,
        }
        self._data_file_name = data_file_name
        self._verification_cache_dir = (Path(verification_cache_dir).expanduser()
                                       if verification_cache_dir is not None else None)
        self._verification_progress = progress
        for spec in self._earth_specs.values():
            spec["data_file_name"] = data_file_name
        self._earth_specs[EARTH_DATASET_3HOURLY_ID] = {
            "path": Path(earth_3hourly_package_dir).expanduser()
            if earth_3hourly_package_dir is not None
            else self._earth_package_dir.parent / "merra2_3hourly_v1",
            "version": "v1", "display": EARTH_3HOURLY_DISPLAY_NAME,
            "data_file_name": EARTH_3HOURLY_DATA_FILE,
        }
        self._earth_specs[EARTH_DATASET_3HOURLY_5DEG_ID] = {
            "path": Path(earth_3hourly_5deg_package_dir).expanduser()
            if earth_3hourly_5deg_package_dir is not None
            else self._earth_package_dir.parent / "merra2_3hourly_5deg_v1",
            "version": "v1", "display": EARTH_3HOURLY_5DEG_DISPLAY_NAME,
            "data_file_name": EARTH_3HOURLY_DATA_FILE,
        }
        # NetCDF reads are serialized: one verification at a time protects the
        # library handle and keeps the cache consistent.
        self._lock = threading.Lock()
        # At most one snapshot per release ID is held per process: the verified
        # release (arrays) and the descriptor derived from the same verification.
        self._earth_cache_signature: dict[str, tuple] = {}
        self._earth_cache_release: dict[str, VerifiedEarthRelease] = {}
        self._earth_cache_descriptor: dict[str, dict] = {}
        # Observability: unchanged in-process snapshots do not count as a read;
        # cross-process proof reads do not count as a full verification.
        self.verification_count = 0
        self.verification_cache_hits = 0

    # ── public protocol ────────────────────────────────────────────────
    def list_datasets(self) -> list[dict]:
        return [self.get_dataset(dataset_id) for dataset_id in ACTIVE_DATASET_IDS]

    def get_dataset(self, dataset_id: str) -> dict:
        if not isinstance(dataset_id, str) or dataset_id.strip().lower() not in REGISTERED_DATASET_IDS:
            raise DatasetRequestError(
                "unknown_dataset", "Unknown dataset id", status_code=404
            )
        normalized = dataset_id.strip().lower()
        require_active_dataset(normalized)
        if normalized in EARTH_DATASET_IDS:
            return copy.deepcopy(self._earth_descriptor(normalized))
        return copy.deepcopy(self._mars_descriptor(normalized))

    def build_training_binding(self, dataset_id: str) -> dict:
        """Return the five task identity column values for a trainable dataset.

        Earth returns a *verified* binding built from the same verified release
        that the run will read, so the task, the checkpoint and the prediction
        request all agree on one published identity. Mars keeps the legacy
        unversioned binding.
        """
        normalized = self._require_trainable(dataset_id)
        if normalized in EARTH_DATASET_IDS:
            return self._earth_training_binding(normalized)
        return {
            "dataset_id": normalized,
            "dataset_version": None,
            "dataset_fingerprint": None,
            "dataset_identity_status": IDENTITY_STATUS_UNVERSIONED,
            "dataset_snapshot": build_identity_snapshot({
                "dataset_id": normalized,
                "planet": "mars",
                "dataset_version": None,
                "binding_basis": REGISTRY_BINDING_BASIS,
                "version_status": UNVERSIONED_VERSION_STATUS,
            }),
        }

    def _require_trainable(self, dataset_id: Any) -> str:
        """Accept a registered Earth id or a legacy Mars training id, else raise."""
        if isinstance(dataset_id, str):
            normalized = dataset_id.strip().lower()
            require_active_dataset(normalized)
            if normalized in EARTH_DATASET_IDS:
                return normalized
        return require_training_dataset(dataset_id)

    def _earth_training_binding(self, dataset_id: str) -> dict:
        release = self.get_earth_snapshot(dataset_id)
        metadata = release.metadata
        snapshot = {
            "dataset_id": dataset_id,
            "planet": "earth",
            "dataset_version": metadata.get("dataset_version"),
            "schema": metadata.get("schema"),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "data_sha256": metadata.get("data_sha256"),
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "time": copy.deepcopy(metadata.get("time")),
            "grid": copy.deepcopy(metadata.get("grid")),
            "variables": copy.deepcopy(metadata.get("variables")),
            "channel_order": list(metadata.get("channel_order") or []),
            "splits": copy.deepcopy(metadata.get("splits")),
            "binding_basis": REGISTRY_BINDING_BASIS,
            "version_status": IDENTITY_STATUS_VERIFIED,
        }
        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            snapshot.update({
                "manifest_content_sha256": metadata.get("manifest_content_sha256"),
                "frequency_hours": 3, "step_unit": "hour", "step": 3,
                "training_profile": earth_training_profile(dataset_id),
            })
        return {
            "dataset_id": dataset_id,
            "dataset_version": metadata.get("dataset_version"),
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "dataset_identity_status": IDENTITY_STATUS_VERIFIED,
            "dataset_snapshot": build_identity_snapshot(snapshot),
        }

    def get_earth_snapshot(
        self, dataset_id: str, expected_fingerprint: Optional[str] = None,
        *, force_full: bool = False,
    ) -> VerifiedEarthRelease:
        """Return the verified release for any supported Earth purpose.

        This is the neutral entry point shared by the overview API, training and
        historical prediction. Only this registry touches the package path and
        the release hashes. Daily callers get immutable field arrays that remain
        readable after the NetCDF handle closes. The three-hour snapshot contains
        metadata and coordinates; its large fields stay on disk.

        ``expected_fingerprint=None`` takes the currently verified release.
        """
        normalized = self._require_registered_earth(dataset_id)
        release = self._earth_release(normalized, force_full=force_full)
        if release is None:
            with self._lock:
                descriptor = self._earth_cache_descriptor.get(normalized, {})
            raise DatasetRequestError(
                "dataset_unavailable",
                "The registered Earth dataset is not available",
                status_code=503,
                availability_reason=descriptor.get("availability_reason"),
            )
        if expected_fingerprint is not None and expected_fingerprint != release.metadata["dataset_fingerprint"]:
            raise DatasetRequestError(
                "dataset_version_changed",
                "The dataset version changed; reload the catalog",
                status_code=409,
            )
        return release

    def get_earth_overview_snapshot(
        self, dataset_id: str, expected_fingerprint: str
    ) -> VerifiedEarthRelease:
        """Return the verified release arrays for read-only overview queries.

        Kept as a thin wrapper over :meth:`get_earth_snapshot` so the overview
        error contract (an explicit expected fingerprint) does not change.
        """
        if dataset_id == EARTH_DATASET_3HOURLY_5DEG_ID:
            raise DatasetRequestError("dataset_overview_not_supported", "The 5 degree Earth release is a training and backtesting dataset", status_code=409)
        return self.get_earth_snapshot(dataset_id, expected_fingerprint=expected_fingerprint)

    @staticmethod
    def _require_registered_earth(dataset_id: Any) -> str:
        if not isinstance(dataset_id, str) or dataset_id.strip().lower() not in REGISTERED_DATASET_IDS:
            raise DatasetRequestError(
                "unknown_dataset", "Unknown dataset id", status_code=404
            )
        normalized = dataset_id.strip().lower()
        require_active_dataset(normalized)
        if normalized not in EARTH_DATASET_IDS:
            raise DatasetRequestError(
                "dataset_overview_not_supported",
                "This dataset has no Earth overview",
                status_code=409,
            )
        return normalized

    # ── descriptors ───────────────────────────────────────────────────
    def _mars_descriptor(self, dataset_id: str) -> dict:
        return {
            "dataset_id": dataset_id,
            "display_name": MARS_DISPLAY_NAMES.get(dataset_id, dataset_id),
            "planet": "mars",
            "dataset_version": None,
            "schema": None,
            "availability": "unverified",
            "availability_reason": LEGACY_DATASET_REASON,
            "manifest_sha256": None,
            "data_sha256": None,
            "dataset_fingerprint": None,
            "capabilities": dict(MARS_CAPABILITIES),
            "time": dict(MARS_TIME),
            "grid": None,
            "channel_order": [],
            "variables": [],
            "splits": None,
            "training_profile": None,
            "limitations": list(MARS_LIMITATIONS),
        }

    def _base_earth_descriptor(self, dataset_id: str, availability: str, reason: Optional[str]) -> dict:
        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            lowres = dataset_id == EARTH_DATASET_3HOURLY_5DEG_ID
            profile = earth_training_profile(dataset_id)
            descriptor = {
                "dataset_id": dataset_id, "display_name": EARTH_3HOURLY_5DEG_DISPLAY_NAME if lowres else EARTH_3HOURLY_DISPLAY_NAME,
                "planet": "earth", "dataset_version": "v1", "schema": EARTH_3HOURLY_SCHEMA,
                "availability": availability, "availability_reason": reason,
                "manifest_sha256": None, "manifest_content_sha256": None,
                "data_sha256": None, "dataset_fingerprint": None,
                "capabilities": {"metadata": True, "training": True,
                                 "web_overview": not lowres, "trained_prediction": True},
                "time": {"kind": "datetime", "calendar": None, "start": None, "end": None,
                         "count": None, "step": 3, "step_unit": "hour",
                         "frequency_hours": 3, "time_zone": "UTC"},
                "frequency_hours": 3, "step_unit": "hour", "step": 3,
                "grid_shape": list(profile["grid_shape"]), "grid": None,
                "channel_order": [], "variables": [], "splits": None,
                "training_profile": profile,
                "limitations": (["This 5 degree release uses full global 36x72 windows for official DLinear and explicitly compatible uploaded models",
                                 "The release is generated by spherical cell-area weighted overlap from the native 0.75 degree package"]
                                if lowres else
                                ["Uploaded three-hour models require the independent versioned contract and dry-run",
                                 "Overview fields use a server display grid; point queries use the native 240 by 480 grid"]),
            }
            return descriptor
        is_v2 = dataset_id == EARTH_DATASET_V2_ID
        descriptor = {
            "dataset_id": dataset_id,
            "display_name": EARTH_V2_DISPLAY_NAME if is_v2 else EARTH_DISPLAY_NAME,
            "planet": "earth",
            "dataset_version": EARTH_V2_DATASET_VERSION if is_v2 else EARTH_DATASET_VERSION,
            "availability": availability,
            "availability_reason": reason,
            "capabilities": dict(EARTH_DESCRIPTOR_CAPABILITIES),
            # Additive catalog fields leave the existing daily time/profile
            # contract intact and make cadence explicit for both Earth families.
            "frequency_hours": 24, "step_unit": "day", "step": 1,
            "grid_shape": [36, 72] if is_v2 else [31, 49],
        }
        descriptor.update(_blank_dynamic_earth_fields())
        # The profile describes the wired entry point, so it is published even
        # when the package is currently missing or invalid: the client can show
        # real limits and still disable the button on `availability`.
        descriptor["training_profile"] = earth_training_profile()
        return descriptor

    def _primary_earth_id(self) -> str:
        for dataset_id, spec in self._earth_specs.items():
            if spec["path"] == self._earth_package_dir:
                return dataset_id
        return EARTH_DATASET_V2_ID

    def _earth_descriptor(self, dataset_id: str) -> dict:
        self._earth_release(dataset_id)
        with self._lock:
            return copy.deepcopy(self._earth_cache_descriptor.get(dataset_id) or self._unavailable_earth_descriptor(dataset_id))

    def _earth_release(self, dataset_id: str, *, force_full: bool = False) -> Optional[VerifiedEarthRelease]:
        """Load, verify and cache the release once per unchanged package.

        The cache tag is the signature that this very verification confirmed, so
        a newer ``stat()`` never gets attached to older arrays. A package that
        changes mid verification drops the whole cache.
        """
        spec = self._earth_specs[dataset_id]
        with self._lock:
            signature = package_signature(spec["path"], data_file_name=spec["data_file_name"])
            if (
                not force_full and dataset_id in self._earth_cache_release
                and signature == self._earth_cache_signature.get(dataset_id)
            ):
                return self._earth_cache_release[dataset_id]
            if (not force_full and dataset_id in EARTH_THREE_HOURLY_DATASET_IDS
                    and signature == self._earth_cache_signature.get(dataset_id)
                    and dataset_id in self._earth_cache_descriptor):
                return None  # An unchanged missing/invalid new package is already classified.
            self._earth_cache_release.pop(dataset_id, None)
            self._earth_cache_descriptor.pop(dataset_id, None)
            self._earth_cache_signature.pop(dataset_id, None)
            self.verification_count += 1
            if dataset_id not in EARTH_THREE_HOURLY_DATASET_IDS and (not spec["manifest"] or not spec["data"]):
                self._earth_cache_descriptor[dataset_id] = self._unavailable_earth_descriptor(dataset_id, REASON_PACKAGE_MISSING)
                return None
            try:
                if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
                    kwargs = {"verification_cache_dir": self._verification_cache_dir}
                    if force_full:
                        kwargs["force_full"] = True
                    if self._verification_progress is not None:
                        kwargs["progress"] = self._verification_progress
                    release = read_earth_3hourly_release(
                        spec["path"], **kwargs
                    )
                    if release.verification_cached:
                        self.verification_count -= 1
                        self.verification_cache_hits += 1
                else:
                    release = read_earth_release(
                        spec["path"], expected_manifest_sha256=spec["manifest"],
                        expected_data_sha256=spec["data"], dataset_id=dataset_id,
                        dataset_version=spec["version"],
                        data_file_name=spec["data_file_name"],
                    )
            except EarthPackageError as exc:
                reason = exc.reason
                logger.warning(
                    "Earth dataset is not available: reason=%s detail=%s",
                    reason,
                    exc.detail,
                )
                self._earth_cache_descriptor[dataset_id] = self._unavailable_earth_descriptor(dataset_id, reason)
                if reason != REASON_PACKAGE_CHANGED:
                    # Settle the negative result for the current signature; a
                    # package that changed mid verification is retried at once.
                    current = package_signature(spec["path"], data_file_name=spec["data_file_name"])
                    if current == signature:
                        self._earth_cache_signature[dataset_id] = signature
                return None
            except Exception:
                # Unexpected failures stay visible instead of being reported as
                # data state, so programming errors do not become "no data".
                logger.exception("Unexpected Earth dataset verification failure")
                raise
            # Carry the registration identity in the release so every consumer
            # answers with the same id, version and fingerprint.
            if release.metadata.get("dataset_id", dataset_id) != dataset_id:
                self._earth_cache_descriptor[dataset_id] = self._unavailable_earth_descriptor(dataset_id, "invalid_dataset")
                return None
            release.metadata.setdefault("dataset_id", dataset_id)
            release.metadata.setdefault("dataset_version", spec["version"])
            descriptor = self._available_earth_descriptor(dataset_id, release.metadata)
            self._earth_cache_release[dataset_id] = release
            self._earth_cache_descriptor[dataset_id] = descriptor
            # The tag is the signature the verification itself confirmed.
            self._earth_cache_signature[dataset_id] = release.signature
            return release

    def _available_earth_descriptor(self, dataset_id: str, metadata: dict) -> dict:
        descriptor = self._base_earth_descriptor(dataset_id, "available", None)
        descriptor.update(copy.deepcopy(metadata))
        descriptor["grid_shape"] = list(metadata["grid"]["shape"])
        # ``manifest`` is verified provenance that the analysis services read from
        # the release, not part of the public descriptor: it lists source files
        # and is far larger than the descriptor contract. The response model would
        # drop it anyway, so it never reaches a client.
        descriptor.pop("manifest", None)
        if dataset_id in EARTH_THREE_HOURLY_DATASET_IDS:
            descriptor["training_profile"] = earth_training_profile(dataset_id)
            if dataset_id == EARTH_DATASET_3HOURLY_5DEG_ID:
                descriptor["limitations"] = [
                    "This 5 degree release uses full global 36x72 windows for official DLinear and explicitly compatible uploaded models",
                    "The release is generated by spherical cell-area weighted overlap from the native 0.75 degree package",
                ]
                return descriptor
            descriptor["limitations"] = [
                "Uploaded three-hour models require the independent versioned contract and dry-run",
                "The three-hourly release is configured on the server and verified against its manifest",
                "Overview fields use a server display grid; point queries use the native 240 by 480 grid",
            ]
            return descriptor
        descriptor["limitations"] = _physical_limitations(metadata.get("limitations"))
        descriptor["limitations"].extend(EARTH_APPLICATION_LIMITATIONS)
        return descriptor
    def _unavailable_earth_descriptor(self, dataset_id: str, reason: Optional[str] = None) -> dict:
        if reason == REASON_PACKAGE_MISSING:
            availability = "missing"
        elif reason == REASON_PACKAGE_UNREADABLE:
            availability = "unverified"
        else:
            availability = "invalid"
        return self._base_earth_descriptor(dataset_id, availability, reason)
