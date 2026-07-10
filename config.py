"""
TOML config support with per-channel sections.

Layout (see config.example.toml):

    [defaults]
    diff_threshold = 0.5
    trim_speech = true
    workers = 1

    [channels.ChannelName]
    channel_url = "https://www.youtube.com/@ChannelName"
    channel_id = "ChannelName"
    reference_intros = ["intro.m4a"]
    cutoff_date = "2022-10-01"   # don't process videos uploaded before this

Precedence (highest wins): CLI flag > [channels.X] > [defaults] > built-in
argparse default. A CLI flag "wins" only when it was actually changed from
the parser default — so config values fill in everything you didn't type.
"""
import os
import datetime
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from logger import logger

DEFAULT_CONFIG_FILE = "config.toml"

# config key -> argparse dest. Most are 1:1; the exceptions are named.
CONFIG_TO_ARG = {
    "reference_intros": "reference_intros",
    "channel_url": "channel",
    "urls_file": "urls_file",
    "channel_id": "channel_id",
    "cutoff_date": "cutoff_date",
    "intro_duration": "intro_duration",
    "manual_approval": "manual_approval",
    "clipboard": "clipboard",
    "trim_speech": "trim_speech",
    "peak_height": "peak_height",
    "weighted_threshold": "weighted_threshold",
    "search_limit": "search_limit",
    "early_exit": "early_exit",
    "workers": "workers",
    "user_id": "user_id",
    "db": "db",
    "diff_threshold": "diff_threshold",
    "cookies_file": "cookies_file",
    "cookies_from_browser": "cookies_from_browser",
}


def load_config(path: str = None) -> dict:
    """
    Load a TOML config file. If `path` is None, tries ./config.toml next to
    this script, then the CWD. Returns {} when no file exists.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir / DEFAULT_CONFIG_FILE)
        candidates.append(Path(os.getcwd()) / DEFAULT_CONFIG_FILE)

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "rb") as f:
                cfg = tomllib.load(f)
            logger.info(f"Loaded config: {candidate}")
            return cfg
        if path:
            raise FileNotFoundError(f"Config file not found: {path}")
    return {}


def resolve_channel_settings(cfg: dict, channel_name: str = None) -> dict:
    """
    Merge [defaults] with the selected [channels.X] section (channel wins).

    Channel selection:
      - explicit `channel_name`, or
      - the ONLY channel in the config (convenience), or
      - defaults only.
    """
    settings = dict(cfg.get("defaults", {}))
    channels = cfg.get("channels", {})

    selected = None
    if channel_name:
        if channel_name not in channels:
            raise KeyError(
                f"Channel '{channel_name}' not found in config. "
                f"Available: {', '.join(channels) or '(none)'}"
            )
        selected = channel_name
    elif len(channels) == 1:
        selected = next(iter(channels))
        logger.info(f"Config has a single channel — using [channels.{selected}]")

    if selected:
        settings.update(channels[selected])
        # A channel section implies its own name as channel_id unless set
        settings.setdefault("channel_id", selected)

    return settings


def apply_settings_to_args(args, parser, settings: dict):
    """
    Copy config settings onto the parsed args, but ONLY for options the user
    left at their argparse default — typed CLI flags always win.
    """
    for key, value in settings.items():
        dest = CONFIG_TO_ARG.get(key)
        if dest is None:
            logger.warning(f"Unknown config key ignored: {key}")
            continue
        if getattr(args, dest, None) == parser.get_default(dest):
            setattr(args, dest, value)
    return args


def parse_cutoff_date(value) -> "datetime.date | None":
    """Accept TOML date objects or YYYY-MM-DD / YYYY-MM / YYYY strings."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Unsupported cutoff date format: {value!r}. Use YYYY-MM-DD, YYYY-MM or YYYY."
    )
