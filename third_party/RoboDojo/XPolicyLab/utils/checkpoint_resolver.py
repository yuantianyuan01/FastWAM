"""Unified checkpoint-directory resolution shared by all policy adapters.

Every adapter resolves the checkpoint (run) directory from the deploy config in
the same way. Resolution precedence (highest first):

  1. An explicit path key in the deploy config
     (``model_path`` / ``checkpoint_path`` / ``ckpt_path`` / ``model_dir`` /
     ``pretrained_path``).
  2. ``ckpt_name`` given as a *path* (absolute, or a relative path containing a
     path separator). Relative paths resolve against the policy directory.
  3. The concatenated run-dir name built from the eval args:
     ``{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}-{seed}``.
  4. ``checkpoints/<ckpt_name>`` (the verbatim full run-dir name) as a fallback,
     so checkpoints saved/named directly by their run id still load.

Training scripts write to the same concatenated name
(``{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}-{seed}``), so eval finds
the checkpoint without the caller having to know the exact template.

Adapters with an extra naming layer (e.g. an ``aloha_<name>`` prefix, timestamp
subdirs, or DeepSpeed nesting) should call :func:`build_run_dir_name` for the
shared base name and/or iterate :func:`candidate_checkpoint_roots` and apply
their per-root logic on top.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Deploy-config keys that carry an explicit checkpoint path (highest priority).
DEFAULT_EXPLICIT_KEYS = (
    "model_path",
    "checkpoint_path",
    "ckpt_path",
    "model_dir",
    "pretrained_path",
)


def _clean(value: Any) -> Optional[str]:
    """Return a stripped non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ckpt_name_is_path(ckpt_name: Any) -> bool:
    """True when ``ckpt_name`` should be treated as a filesystem path.

    A value counts as a path when it is absolute or contains a path separator
    (``/`` or the OS-specific ``os.sep``). A bare identifier such as
    ``RoboDojo-cotrain-arx_x5-joint-0`` is not a path.
    """
    text = _clean(ckpt_name)
    if text is None:
        return False
    expanded = os.path.expanduser(text)
    if Path(expanded).is_absolute():
        return True
    if "/" in text or os.sep in text:
        return True
    return bool(os.altsep and os.altsep in text)


def build_run_dir_name(model_cfg: dict, *, include_seed: bool = True) -> Optional[str]:
    """Build ``{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}[-{seed}]``.

    Returns None when any required component is missing. ``bench_name`` falls
    back to the legacy ``dataset_name`` key for backward compatibility.
    """
    bench = _clean(model_cfg.get("bench_name") or model_cfg.get("dataset_name"))
    ckpt = _clean(model_cfg.get("ckpt_name"))
    env_cfg_type = _clean(model_cfg.get("env_cfg_type"))
    action_type = _clean(model_cfg.get("action_type"))
    parts = [bench, ckpt, env_cfg_type, action_type]
    if include_seed:
        parts.append(_clean(model_cfg.get("seed")))
    if any(part is None for part in parts):
        return None
    return "-".join(parts)


def _as_path(value: Any, base_dir: Optional[Path]) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def candidate_checkpoint_roots(
    model_cfg: dict,
    checkpoints_dir: os.PathLike | str,
    *,
    policy_dir: os.PathLike | str | None = None,
    explicit_keys: tuple[str, ...] = DEFAULT_EXPLICIT_KEYS,
    include_seed: bool = True,
    resolve: bool = True,
) -> list[Path]:
    """Return ordered, de-duplicated candidate checkpoint-root paths.

    See the module docstring for the precedence. ``policy_dir`` is the base for
    resolving relative path inputs (defaults to ``checkpoints_dir``'s parent).
    """
    checkpoints_dir = Path(checkpoints_dir)
    base_dir = Path(policy_dir) if policy_dir is not None else checkpoints_dir.parent

    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: os.PathLike | str) -> None:
        candidate = Path(path)
        if resolve:
            try:
                candidate = candidate.resolve()
            except OSError:
                pass
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            roots.append(candidate)

    # 1. explicit path keys
    for key in explicit_keys:
        value = _clean(model_cfg.get(key))
        if value:
            add(_as_path(value, base_dir))

    ckpt_name = _clean(model_cfg.get("ckpt_name"))
    is_path = ckpt_name is not None and ckpt_name_is_path(ckpt_name)

    # 2. ckpt_name given as a path (short-circuits the name-based candidates)
    if is_path:
        add(_as_path(ckpt_name, base_dir))
    else:
        # 3. concatenated 5-tuple run-dir name
        run_dir_name = build_run_dir_name(model_cfg, include_seed=include_seed)
        if run_dir_name:
            add(checkpoints_dir / run_dir_name)

        # 4. fallback: checkpoints/<ckpt_name> (verbatim full run-dir name)
        if ckpt_name:
            add(checkpoints_dir / ckpt_name)

    return roots


def resolve_checkpoint_root(
    model_cfg: dict,
    checkpoints_dir: os.PathLike | str,
    *,
    policy_dir: os.PathLike | str | None = None,
    explicit_keys: tuple[str, ...] = DEFAULT_EXPLICIT_KEYS,
    include_seed: bool = True,
    must_exist: bool = True,
) -> Path:
    """Resolve the checkpoint root directory (or file) per the shared precedence.

    Returns the first existing candidate. When ``must_exist`` is False and no
    candidate exists, returns the highest-priority candidate so callers can raise
    their own error. Raises ``FileNotFoundError`` (listing what was checked) when
    ``must_exist`` is True and nothing resolves.
    """
    candidates = candidate_checkpoint_roots(
        model_cfg,
        checkpoints_dir,
        policy_dir=policy_dir,
        explicit_keys=explicit_keys,
        include_seed=include_seed,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not must_exist and candidates:
        return candidates[0]
    checked = "\n  ".join(str(path) for path in candidates) or "  <no checkpoint candidates>"
    raise FileNotFoundError(
        "Could not resolve a checkpoint directory. Pass a valid ckpt_name "
        "(full run-dir name or a path), or set an explicit checkpoint path in "
        f"deploy.yml ({'/'.join(explicit_keys)}).\nChecked:\n  {checked}"
    )
