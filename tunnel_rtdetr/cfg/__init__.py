
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tunnel_rtdetr import __version__
from tunnel_rtdetr.utils import DEFAULT_CFG_DICT, FLOAT_OR_INT, LOGGER, RANK, ROOT, RUNS_DIR, STR_OR_PATH, TESTS_RUNNING, YAML, IterableSimpleNamespace, colorstr, deprecation_warn

# Define valid solutions
MODES = frozenset({"train", "val", "predict", "track", "export"})
TASKS = frozenset({"detect"})

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
CLI_HELP_MSG = "tunnel-rtdetr [train|val|predict|track|export] model=LOCAL_PATH key=value ..."

# Define keys for arg type checks
CFG_FLOAT_KEYS = frozenset(['batch', 'degrees', 'shear', 'time', 'warmup_epochs'])
CFG_FRACTION_KEYS = frozenset(['bgr', 'conf', 'cutmix', 'fliplr', 'flipud', 'fraction', 'hsv_h', 'hsv_s', 'hsv_v', 'iou', 'lr0', 'lrf', 'mixup', 'momentum', 'mosaic', 'multi_scale', 'perspective', 'scale', 'translate', 'warmup_bias_lr', 'warmup_momentum', 'weight_decay'])
CFG_INT_KEYS = frozenset(['close_mosaic', 'epochs', 'line_width', 'max_det', 'nbs', 'patience', 'save_period', 'seed', 'vid_stride', 'workers'])
CFG_BOOL_KEYS = frozenset(['augment', 'cos_lr', 'deterministic', 'dnn', 'dynamic', 'exist_ok', 'half', 'plots', 'rect', 'save', 'save_conf', 'save_crop', 'save_frames', 'save_json', 'save_txt', 'show', 'show_boxes', 'show_conf', 'show_labels', 'single_cls', 'val', 'verbose', 'visualize'])


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    """Convert a configuration object to a dictionary."""
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace:
    """Load and merge configuration data from a file or dictionary, with optional overrides."""
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = str(cfg.get("model", "")).partition(".")[0]
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg: dict, hard: bool = True) -> None:
    """Check configuration argument types and values for the Tunnel RT-DETR library."""
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    """Return the directory path for saving outputs, derived from arguments or default settings."""
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from tunnel_rtdetr.utils.files import increment_path

        project = args.project or ""
        if not Path(project).is_absolute():
            project = (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task / project
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir).resolve()  # resolve to display full path in console


def _handle_deprecation(custom: dict) -> dict:
    """Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings."""
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom


def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None:
    """Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error."""
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    # Allow 'augmentations' as a valid custom parameter for custom Albumentations transforms
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations", "save_dir"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid RT-DETR argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: list[str]) -> list[str]:
    """Merge arguments around isolated '=' in a list of strings and join fragments with brackets."""
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle equals sign merging
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # Handle bracket joining
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # Append any remaining current string
    if current:
        new_args.append(current)

    return new_args










def parse_key_value_pair(pair: str = "key=value") -> tuple:
    """Parse a key-value pair string into separate key and value components."""
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v: str) -> Any:
    """Convert a string representation of a value to its appropriate Python type."""
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            return v


def entrypoint(debug=""):
    from tunnel_rtdetr import RTDETR
    args = merge_equals_args(debug.split() if debug else sys.argv[1:])
    if not args or args[0] in {"help", "--help", "-h"}:
        print("tunnel-rtdetr [train|val|predict|track|export] model=LOCAL_PATH key=value ...")
        return
    if args[0] in {"version", "--version"}:
        print(__version__)
        return
    mode = args.pop(0)
    if mode not in {"train", "val", "predict", "track", "export"}:
        raise ValueError("Supported modes: train, val, predict, track, export")
    options = dict(parse_key_value_pair(arg) for arg in args)
    model = options.pop("model", None)
    if model is None:
        raise ValueError("A model=LOCAL_PATH argument is required.")
    getattr(RTDETR(model), mode)(**options)


# Special modes --------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # Run the local RT-DETR CLI.
    entrypoint(debug="")
