"""Server-owned Earth derived-input preparation shared by every runtime path."""

from __future__ import annotations

import copy

from services.dataset_identity import DatasetRequestError


def declaration_for_reference(reference):
    """Read the declaration from hash-verified, safety-checked uploaded source.

    This does not build or probe the model. It also handles source declarations
    composed from constants, just as the upload validator does.
    """
    from services.earth_model_source import _load_uploaded_module
    from services.uploaded_model_source import resolve_source_text, hash_source_bytes
    from training_backbones.earth_3hourly_uploaded_contract import normalize_derived_inputs

    source, _ = resolve_source_text(reference)
    if hash_source_bytes(source.encode("utf-8")) != reference.get("content_hash"):
        raise ValueError("Earth derived-input source digest does not match the frozen model")
    module = _load_uploaded_module(source, "earth_derived_declaration.py")
    return normalize_derived_inputs(getattr(module, "MODEL_SPEC", None))


def freeze_topography_for_reference(reference, release):
    declaration = declaration_for_reference(reference)
    if "topography" not in declaration:
        return None
    from services.earth_topography import load_earth_topography

    asset = load_earth_topography(release=release)
    return {"declaration": copy.deepcopy(declaration), "identity": copy.deepcopy(asset.identity)}


def topography_for_reference(reference, release, frozen):
    """Reject queued tasks whose declaration or terrain no longer matches."""
    declaration = declaration_for_reference(reference)
    if "topography" not in declaration:
        if frozen is not None:
            raise ValueError("An Earth task without topography must not carry a terrain identity")
        return None
    if (not isinstance(frozen, dict) or set(frozen) != {"declaration", "identity"}
            or frozen["declaration"] != declaration):
        raise ValueError("Earth task is missing its frozen topography declaration and identity")
    from services.earth_topography import load_earth_topography

    return load_earth_topography(release=release, expected_identity=frozen["identity"])


def topography_for_checkpoint(checkpoint, release):
    """Check current content and actual release grid before any result cache."""
    config = getattr(checkpoint, "model_config", {})
    if "topography" not in (config.get("derived_inputs") or []):
        return None
    from services.earth_topography import load_earth_topography

    try:
        return load_earth_topography(
            release=release, expected_identity=config.get("topography_identity"))
    except (ValueError, OSError, RuntimeError) as exc:
        raise DatasetRequestError("earth_topography_invalid", str(exc), status_code=409) from exc


def topography_kwargs(asset, inputs, *, spatial_slices=None):
    """Prepare the same location-specific tensor for training, prediction and PFI."""
    if asset is None:
        return {}
    from services.earth_topography import prepare_topography_batch

    return {"topography": prepare_topography_batch(asset, inputs, spatial_slices=spatial_slices)}
