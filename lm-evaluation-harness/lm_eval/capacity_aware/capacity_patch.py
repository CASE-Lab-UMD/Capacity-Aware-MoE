"""Compatibility wrapper for the repository-level capacity-aware patch."""

from importlib import util
from pathlib import Path


_ROOT_PATCH = Path(__file__).resolve().parents[3] / "capacity_aware" / "capacity_patch.py"
_SPEC = util.spec_from_file_location("_capacity_aware_root_patch", _ROOT_PATCH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load capacity-aware patch from {_ROOT_PATCH}")

_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if _name not in {"__builtins__", "__cached__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
        globals()[_name] = _value

__all__ = getattr(_MODULE, "__all__", ["apply_capacity_aware_moe_patch"])
