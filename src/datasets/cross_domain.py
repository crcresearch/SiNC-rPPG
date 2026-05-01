"""Optional validation dataset different from training."""

from __future__ import annotations

from copy import copy


def validation_arg_obj(arg_obj):
    """Return a shallow copy of ``arg_obj`` with ``dataset`` (and optional ``fps``) for validation only."""
    vd = getattr(arg_obj, "validation_dataset", None)
    if vd is None:
        return arg_obj
    if isinstance(vd, str) and vd.strip().lower() in ("", "none", "null"):
        return arg_obj

    o = copy(arg_obj)
    o.dataset = str(vd).strip()

    vf = getattr(arg_obj, "validation_fps", None)
    if vf is not None:
        if isinstance(vf, str) and vf.strip().lower() in ("", "none", "null"):
            return o
        try:
            o.fps = int(float(vf))
        except (TypeError, ValueError):
            pass
    return o


def is_cross_domain(arg_obj) -> bool:
    vd = getattr(arg_obj, "validation_dataset", None)
    if vd is None or (isinstance(vd, str) and vd.strip().lower() in ("", "none", "null")):
        return False
    return str(vd).strip().lower() != str(getattr(arg_obj, "dataset", "")).strip().lower()
