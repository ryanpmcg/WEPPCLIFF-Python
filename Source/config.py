"""WEPPCLIFF configuration — replaces R global flag variables."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Sequence, Union

# Universal line return (R ``lr``).
LR = "\n"

# CLI flags from WEPPCLIFF.R lines 2748-2758 plus PyMICE extensions.
FLAG_NAMES: tuple[str, ...] = (
    "fr",
    "la",
    "d",
    "o",
    "e",
    "p",
    "f",
    "fn",
    "delim",
    "fsp",
    "isc",
    "rs",
    "u",
    "qc",
    "id",
    "pd",
    "ed",
    "alt",
    "pmd",
    "cp",
    "pi",
    "ai",
    "ei",
    "ee",
    "sid",
    "tth",
    "dth",
    "os",
    "sod",
    "qcop",
    "chkth",
    "spkth",
    "strth",
    "stkth",
    "rp",
    "im",
    "io",
    "qi",
    "iv",
    "pm",
    "ns",
    "kg",
    "st",
    "jm",
    "gf",
    "tauqc",
    "ir",
    "tz",
    "sdt",
    "edt",
    "rtb",
    "ipb",
    "dtf1",
    "dtf2",
    "dtf3",
    "cv",
    "sm",
    "bf",
    "wi",
    "ctr",
    "cow",
    "mp",
    "prof",
    "pint",
    "mepr",
    "gcpr",
    "lnpr",
    "warn",
    "verb",
)

_RUNTIME_FIELDS = frozenset(
    {
        "home_dir",
        "in_dir",
        "out_dir",
        "ex_dir",
        "plot_dir",
        "days_count",
        "args",
        "u_loc",
        "lat",
        "lon",
        "impute_granularity",
        "fill_dates",
        "cligen_monthly",
        "phigi_report",
    }
)

# Module-level active config for modules that call get_config() during migration.
_active_config: Optional["WeppcliffConfig"] = None


def get_config() -> "WeppcliffConfig":
    """Return the active runtime configuration."""
    if _active_config is None:
        raise RuntimeError("WeppcliffConfig has not been initialized; call set_config() first.")
    return _active_config


def set_config(config: "WeppcliffConfig") -> None:
    """Set the active runtime configuration."""
    global _active_config
    _active_config = config


@dataclass
class WeppcliffConfig:
    """Configuration object passed through the WEPPCLIFF pipeline."""

    # --- Installation arguments ---
    fr: str = "f"
    la: Optional[str] = None

    # --- Input/output arguments ---
    d: Optional[str] = None
    o: Optional[str] = None
    e: Optional[str] = None
    p: Optional[str] = None
    f: Optional[str] = None
    fn: Optional[str] = None
    delim: Optional[str] = None
    fsp: Optional[str] = None
    isc: str = "f"
    rs: str = "f"

    # --- Functionality arguments ---
    u: str = "m"
    qc: str = "f"
    id: str = "f"
    pd: str = "f"
    ed: Any = 0
    alt: str = "f"
    pmd: Any = "f"

    # --- Precipitation arguments ---
    cp: str = "f"
    pi: str = "f"
    ai: str = "f"
    ei: str = "f"
    ee: str = "R2"

    # --- Storm control arguments ---
    sid: Any = 1
    tth: float = 6.0
    dth: float = 1.27
    os: str = "f"
    sod: float = 0.5

    # --- Quality control arguments ---
    qcop: str = "b"
    chkth: float = 0.9
    spkth: float = 0.9
    strth: float = 2.0
    stkth: float = 4.0
    rp: int = 1000

    # --- Impute control arguments ---
    im: str = "DEFAULT"
    io: int = 100
    qi: str = "f"
    iv: str = "f"
    pm: str = "t"
    ns: str = "f"
    kg: str = "AUTO"
    st: str = "t"
    jm: str = "f"
    gf: str = "legacy"
    tauqc: str = "auto"
    ir: str = "f"

    # --- Datetime control arguments ---
    tz: str = "GMT"
    sdt: str = "start"
    edt: str = "end"
    rtb: str = "m"
    ipb: str = "f"
    dtf1: str = "%Y-%m-%d %H:%M:%S"
    dtf2: Optional[str] = None
    dtf3: Optional[str] = None

    # --- CLIGEN file format arguments ---
    cv: str = "0.0"
    sm: int = 1
    bf: int = 1
    wi: int = 1
    ctr: str = "all"
    cow: str = "none"

    # --- Optimization arguments ---
    mp: str = "t"

    # --- Developer arguments ---
    prof: str = "f"
    pint: float = 0.02
    mepr: str = "f"
    gcpr: str = "f"
    lnpr: str = "f"
    warn: str = "f"
    verb: str = "f"

    # --- Parsed CLI state ---
    args: List[str] = field(default_factory=list)
    u_loc: Optional[int] = None

    # --- Runtime paths ---
    home_dir: Optional[str] = None
    in_dir: Optional[str] = None
    out_dir: Optional[str] = None
    ex_dir: Optional[str] = None
    plot_dir: Optional[str] = None

    # --- Runtime counters ---
    days_count: Optional[float] = None

    # --- Gap-fill runtime (PyMICE 6b) ---
    lat: float = 45.0
    lon: float = 0.0
    impute_granularity: str = "seasonal"
    fill_dates: Optional[List[str]] = None
    cligen_monthly: Optional[dict[str, list[float]]] = None
    phigi_report: Any = None

    # --- Cached runtime flags (set by apply_runtime_flags) ---
    arg_locations: Optional[dict[str, int]] = None
    cv_off: bool = False
    do_qc: bool = False
    do_id: bool = False
    do_ei: bool = False
    do_ai: bool = False
    do_pd: bool = False
    do_alt: bool = False
    do_ipb: bool = False
    do_verb: bool = False

    def is_true(self, flag: str) -> bool:
        """Return True when a flag is set to 't' (case-insensitive), matching R."""
        value = getattr(self, flag, None)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).upper() == "T"

    def copy(self) -> "WeppcliffConfig":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)

    def for_worker(self) -> "WeppcliffConfig":
        """Slim config for multiprocessing workers (drops verbose argv list)."""
        worker = self.copy()
        worker.args = []
        worker.arg_locations = None
        return worker

    def get_flag(self, flag: str) -> Any:
        """Get the value of a named CLI flag."""
        if flag not in FLAG_NAMES:
            raise KeyError(f"Unknown flag: {flag}")
        return getattr(self, flag)

    def set_flag(self, flag: str, value: Any) -> None:
        """Set the value of a named CLI flag."""
        if flag not in FLAG_NAMES:
            raise KeyError(f"Unknown flag: {flag}")
        setattr(self, flag, value)

    def na_rm(self) -> bool:
        """R ``na.rm = !pmd`` — omit NAs in aggregation when pmd is false."""
        return not self.is_true("pmd")

    def verbose(self, message: str) -> None:
        """Print a status line when verbosity is enabled."""
        if self.is_true("verb"):
            print(f"{LR}{message}", end="")

    def ensure_datetime_formats(self) -> None:
        """Apply R defaults for dtf2/dtf3 when unset."""
        if not self.dtf2:
            self.dtf2 = self.dtf1
        if not self.dtf3:
            self.dtf3 = self.dtf1

    def apply_runtime_flags(self) -> None:
        """Cache frequently checked boolean flags after assign_final_args."""
        self.cv_off = str(self.cv).upper() == "F"
        self.do_qc = self.is_true("qc")
        self.do_id = self.is_true("id")
        self.do_ei = self.is_true("ei")
        self.do_ai = self.is_true("ai")
        self.do_pd = self.is_true("pd")
        self.do_alt = self.is_true("alt")
        self.do_ipb = self.is_true("ipb")
        self.do_verb = self.is_true("verb")
        self.do_phigi = str(self.gf).lower() == "phigi"
        self.do_ir = self.is_true("ir")

    @classmethod
    def flag_field_names(cls) -> tuple[str, ...]:
        """Return dataclass field names that correspond to CLI flags."""
        return tuple(f.name for f in fields(cls) if f.name not in _RUNTIME_FIELDS)

    @classmethod
    def from_args(cls, args: Sequence[str], home_dir: str = "") -> "WeppcliffConfig":
        """Build config from a flat CLI argument list (``-flag value`` pairs)."""
        from Source.args import assign_empty_args, parse_args

        config = parse_args(list(args))
        config.args = list(args)
        if home_dir:
            config.home_dir = home_dir
        assign_empty_args(config)
        config.ensure_datetime_formats()
        return config

    def parse_unit_vars(self) -> Optional[List[str]]:
        """Parse ``-u`` exception variable list from CLI args (R ``convert_units``)."""
        if self.u_loc is None or self.u_loc >= len(self.args):
            return None
        dash_positions = [idx for idx, arg in enumerate(self.args) if arg.startswith("-")]
        dash_positions = [idx for idx in dash_positions if idx >= self.u_loc]
        if not dash_positions:
            return None
        # R: start = x[1]+2, end = x[2]-1 (inclusive); Python slice uses exclusive end.
        start = dash_positions[0] + 2
        end = len(self.args)
        if len(dash_positions) > 1:
            end = min(dash_positions[1], len(self.args))
        if start >= end:
            return None
        return [str(v).upper() for v in self.args[start:end]]

    def parse_user_equation(self) -> Optional[str]:
        """Extract USER erosion equation from CLI args."""
        lowered = [a.lower() for a in self.args]
        for marker in ("all", "user"):
            if marker in lowered:
                idx = lowered.index(marker)
                if idx + 1 < len(self.args):
                    return self.args[idx + 1]
        return None