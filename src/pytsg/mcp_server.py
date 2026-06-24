"""MCP server for pytsg — exposes TSG hyperspectral data operations as AI-callable tools.

Three operating modes are supported; call get_agent_capabilities first to
discover which modes are available in the current environment:

Mode 1 — pytsg only (no TSG licence required)
    Read .tsg/.bip packages directly in Python and run analysis.
    Start with read_tsg_package or read_tsg_bip_pair.

Mode 2 — pytsg + TSG Desktop (manual)
    The user operates TSG Desktop by hand; the agent provides step-by-step
    instructions via get_tsg_manual_instructions and reads the saved results
    with the standard pytsg tools once the user confirms the save.

Mode 3 — pytsg + TSG Desktop (headless / automated)
    The agent drives TSGHeadless.exe directly via run_tsg_headless_scalars
    and run_tsg_headless_export.  Requires TSG Pro (headless edition) with
    a valid CSIRO licence installed on the host.  Use configure_tsg_desktop
    to supply the path to the executable if auto-detection fails.

Run the server with:
    pytsg-mcp          # stdio transport (for MCP-compatible clients)
    pytsg-mcp --http   # streamable-HTTP transport (for MCP-compatible web clients)

The server keeps an in-process cache of loaded TSG packages so that a
conversation can load a dataset once and then call analysis tools
repeatedly without re-reading from disk each time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from mcp.server.fastmcp import FastMCP

from pytsg import parse_tsg
from pytsg.feature import band_extractor, fit_gaussian, sqm
from pytsg import tsg_desktop

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "pytsg",
    instructions=(
        "You are an agent for TSG (The Spectral Geologist) hyperspectral "
        "drill-core analysis. "
        "Always start a session by calling get_agent_capabilities to "
        "understand which operating modes are available. "
        "Three modes exist: "
        "(1) pytsg only - pure Python, no TSG licence needed; "
        "(2) pytsg + TSG Desktop manual - you give the user step-by-step "
        "GUI instructions via get_tsg_manual_instructions; "
        "(3) pytsg + TSG headless - you drive TSGHeadless automatically "
        "via run_tsg_headless_scalars / run_tsg_headless_export. "
        "After loading data use the analysis tools "
        "(get_spectra_summary, get_scalars, extract_band_features, etc.) "
        "to query spectra, depth headers, and mineral-classification scalars."
    ),
)

# ---------------------------------------------------------------------------
# TSG desktop installation (mutable singleton, set via configure_tsg_desktop)
# ---------------------------------------------------------------------------

_tsg_installation: Union[tsg_desktop.TsgInstallation, None] = None

# ---------------------------------------------------------------------------
# In-process dataset cache  { handle -> TSG | Spectra }
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _new_handle(prefix: str) -> str:
    """Generate a unique cache key."""
    idx = sum(1 for k in _cache if k.startswith(prefix))
    return f"{prefix}_{idx}"


def _require(handle: str) -> Any:
    if handle not in _cache:
        raise KeyError(
            f"Handle '{handle}' not found. "
            "Use read_tsg_package or read_tsg_bip_pair first."
        )
    return _cache[handle]


def _spectra_for(obj: Any, sensor: str) -> parse_tsg.Spectra:
    """Return a Spectra object from a TSG package or directly."""
    if isinstance(obj, parse_tsg.TSG):
        s = getattr(obj, sensor, None)
        if not isinstance(s, parse_tsg.Spectra):
            raise ValueError(
                f"Sensor '{sensor}' is not available in this package. "
                "Available sensors: nir, tir."
            )
        return s
    if isinstance(obj, parse_tsg.Spectra):
        return obj
    raise TypeError(f"Unexpected cached object type: {type(obj)}")


def _df_to_dict(df: pd.DataFrame, max_rows: int = 200) -> dict:
    truncated = len(df) > max_rows
    return {
        "columns": list(df.columns),
        "rows": df.head(max_rows).to_dict(orient="records"),
        "total_rows": len(df),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Tools — data loading
# ---------------------------------------------------------------------------


@mcp.tool()
def read_tsg_package(
    folder: str,
    read_cras: bool = False,
) -> dict:
    """Load a TSG package directory into the cache and return a handle.

    A TSG package folder contains paired .tsg/.bip files for NIR (and
    optionally TIR) spectra, a linescan image (.cras.bip), and a lidar
    height profile (*_hires.dat).

    Args:
        folder: Absolute or relative path to the TSG package directory
                (e.g. "example_data/ETG0187").
        read_cras: Set to True to also load the linescan core image.
                   This can be slow for large files.

    Returns:
        A dict with a ``handle`` key. Pass this handle to other tools.
    """
    p = Path(folder).expanduser().resolve()
    tsg = parse_tsg.read_package(str(p), read_cras_file=read_cras)
    handle = _new_handle(p.name)
    _cache[handle] = tsg

    sensors = []
    if isinstance(tsg.nir, parse_tsg.Spectra):
        sensors.append("nir")
    if isinstance(tsg.tir, parse_tsg.Spectra):
        sensors.append("tir")

    return {
        "handle": handle,
        "folder": str(p),
        "available_sensors": sensors,
        "has_lidar": tsg.lidar is not None,
        "has_cras": isinstance(tsg.cras, parse_tsg.Cras),
    }


@mcp.tool()
def read_tsg_bip_pair(
    tsg_file: str,
    bip_file: str,
    sensor_name: str = "nir",
) -> dict:
    """Load a single .tsg / .bip file pair into the cache.

    Use this when you have individual files rather than a full package
    directory.

    Args:
        tsg_file: Path to the .tsg header file.
        bip_file: Path to the matching .bip spectral data file.
        sensor_name: Label for the sensor type (e.g. "nir", "tir").

    Returns:
        A dict with a ``handle`` key. Pass this handle to analysis tools.
    """
    spectra = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, sensor_name)
    handle = _new_handle(sensor_name)
    _cache[handle] = spectra
    return {
        "handle": handle,
        "sensor": sensor_name,
        "n_samples": spectra.spectra.shape[0],
        "n_bands": spectra.spectra.shape[1],
        "wavelength_start_nm": float(spectra.wavelength[0]),
        "wavelength_end_nm": float(spectra.wavelength[-1]),
    }


@mcp.tool()
def list_loaded_datasets() -> dict:
    """List all datasets currently held in the in-process cache.

    Returns:
        A dict mapping each handle to a brief description.
    """
    out = {}
    for handle, obj in _cache.items():
        if isinstance(obj, parse_tsg.TSG):
            sensors = []
            if isinstance(obj.nir, parse_tsg.Spectra):
                sensors.append("nir")
            if isinstance(obj.tir, parse_tsg.Spectra):
                sensors.append("tir")
            out[handle] = {"type": "TSG package", "sensors": sensors}
        elif isinstance(obj, parse_tsg.Spectra):
            out[handle] = {
                "type": "Spectra",
                "sensor": obj.spectrum_name,
                "n_samples": obj.spectra.shape[0],
            }
    return out


@mcp.tool()
def unload_dataset(handle: str) -> dict:
    """Remove a dataset from the cache to free memory.

    Args:
        handle: The handle returned by read_tsg_package or read_tsg_bip_pair.

    Returns:
        Confirmation dict.
    """
    _require(handle)
    del _cache[handle]
    return {"removed": handle}


# ---------------------------------------------------------------------------
# Tools — introspection
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spectra_summary(handle: str, sensor: str = "nir") -> dict:
    """Return metadata about a loaded spectra dataset.

    Args:
        handle: Dataset handle from a load tool.
        sensor: Which sensor to inspect: "nir" or "tir".
                Ignored when the handle points to a single Spectra object.

    Returns:
        Dict with shape, wavelength range, sample count, scalar names, and
        available class names.
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)

    scalar_cols = list(s.scalars.columns) if s.scalars is not None else []
    header_cols = list(s.sampleheaders.columns) if s.sampleheaders is not None else []
    class_names = [
        c.name for c in s.classes.values() if isinstance(c, parse_tsg.ClassHeaders)
    ] if isinstance(s.classes, dict) else []

    return {
        "sensor": s.spectrum_name,
        "n_samples": s.spectra.shape[0],
        "n_bands": s.spectra.shape[1],
        "wavelength_start_nm": float(s.wavelength[0]),
        "wavelength_end_nm": float(s.wavelength[-1]),
        "wavelength_step_nm": float(s.wavelength[1] - s.wavelength[0])
        if len(s.wavelength) > 1
        else 0.0,
        "scalar_columns": scalar_cols,
        "sample_header_columns": header_cols,
        "class_names": class_names,
    }


@mcp.tool()
def get_wavelengths(handle: str, sensor: str = "nir") -> dict:
    """Return the full wavelength axis (in nm) for a sensor.

    Args:
        handle: Dataset handle.
        sensor: "nir" or "tir".

    Returns:
        Dict with a ``wavelengths`` list (nm).
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)
    return {"wavelengths_nm": s.wavelength.tolist()}


@mcp.tool()
def get_sample_headers(
    handle: str,
    sensor: str = "nir",
    max_rows: int = 200,
) -> dict:
    """Return the per-sample depth / position headers.

    Common columns include: sample, T (tray), L (line), P (position),
    D (depth m), X (distance mm), H (hole name).

    Args:
        handle: Dataset handle.
        sensor: "nir" or "tir".
        max_rows: Maximum number of rows to return (default 200).

    Returns:
        Dict with ``columns``, ``rows``, ``total_rows``, and ``truncated``.
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)
    return _df_to_dict(s.sampleheaders, max_rows)


@mcp.tool()
def get_scalars(
    handle: str,
    sensor: str = "nir",
    max_rows: int = 200,
) -> dict:
    """Return the TSG-derived scalar / mineral-classification columns.

    These are the band-header derived values computed by TSG during
    processing, e.g. mineral group assignments and band-depth scalars.

    Args:
        handle: Dataset handle.
        sensor: "nir" or "tir".
        max_rows: Maximum rows to return.

    Returns:
        Dict with ``columns``, ``rows``, ``total_rows``, and ``truncated``.
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)
    return _df_to_dict(s.scalars, max_rows)


# ---------------------------------------------------------------------------
# Tools — spectral data access
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spectra_at_index(
    handle: str,
    indices: list[int],
    sensor: str = "nir",
) -> dict:
    """Retrieve raw reflectance spectra for the given sample indices.

    Args:
        handle: Dataset handle.
        indices: List of integer sample indices (0-based).
        sensor: "nir" or "tir".

    Returns:
        Dict with ``wavelengths_nm`` and ``spectra`` (list of lists,
        one row per requested sample).
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)
    n = s.spectra.shape[0]
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise IndexError(
            f"Indices {bad} are out of range for dataset with {n} samples."
        )
    data = s.spectra[indices, :].tolist()
    return {
        "wavelengths_nm": s.wavelength.tolist(),
        "spectra": data,
        "indices": indices,
    }


@mcp.tool()
def get_spectra_at_depth_range(
    handle: str,
    depth_from_m: float,
    depth_to_m: float,
    sensor: str = "nir",
    max_spectra: int = 500,
) -> dict:
    """Return spectra for all samples within a depth interval.

    Depth is read from the ``D`` column of the sample headers.  If that
    column is absent, returns an error message.

    Args:
        handle: Dataset handle.
        depth_from_m: Start depth in metres (inclusive).
        depth_to_m: End depth in metres (inclusive).
        sensor: "nir" or "tir".
        max_spectra: Safety cap on the number of spectra returned.

    Returns:
        Dict with ``wavelengths_nm``, ``spectra``, ``depths_m``,
        ``sample_indices``, and ``n_returned``.
    """
    obj = _require(handle)
    s = _spectra_for(obj, sensor)

    if "D" not in s.sampleheaders.columns:
        return {
            "error": (
                "Depth column 'D' not found in sample headers. "
                f"Available columns: {list(s.sampleheaders.columns)}"
            )
        }

    depths = pd.to_numeric(s.sampleheaders["D"], errors="coerce")
    mask = (depths >= depth_from_m) & (depths <= depth_to_m)
    indices = np.where(mask.values)[0]

    if len(indices) > max_spectra:
        indices = indices[:max_spectra]
        truncated = True
    else:
        truncated = False

    return {
        "wavelengths_nm": s.wavelength.tolist(),
        "spectra": s.spectra[indices, :].tolist(),
        "depths_m": depths.iloc[indices].tolist(),
        "sample_indices": indices.tolist(),
        "n_returned": len(indices),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Tools — analysis
# ---------------------------------------------------------------------------


@mcp.tool()
def extract_band_features(
    handle: str,
    start_band: int = 0,
    end_band: int = -1,
    sensor: str = "nir",
    statistics: list[str] = ["argmin", "min", "sum"],
    sample_indices: Union[list[int], None] = None,
) -> dict:
    """Compute band statistics (position, depth, area) across a wavelength range.

    Args:
        handle: Dataset handle.
        start_band: First band index (0-based, inclusive).
        end_band: Last band index (exclusive). -1 means last band.
        sensor: "nir" or "tir".
        statistics: List of NumPy reduction names to apply along the
                    band axis.  Supported: "argmin", "argmax", "min",
                    "max", "sum", "mean".
        sample_indices: Optional list of sample indices to restrict
                        the analysis.  If None, all samples are used.

    Returns:
        Dict with ``results`` (list of rows, one per sample) and
        ``statistic_names``.
    """
    _stat_map = {
        "argmin": np.argmin,
        "argmax": np.argmax,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
        "mean": np.mean,
    }
    bad_stats = [s for s in statistics if s not in _stat_map]
    if bad_stats:
        raise ValueError(
            f"Unknown statistics: {bad_stats}. "
            f"Supported: {list(_stat_map.keys())}"
        )

    obj = _require(handle)
    sp = _spectra_for(obj, sensor)

    data = sp.spectra
    if sample_indices is not None:
        data = data[sample_indices, :]

    callables = [_stat_map[s] for s in statistics]
    result = band_extractor(data, start=start_band, end=end_band, statistic=callables)
    return {
        "statistic_names": statistics,
        "results": result.tolist(),
        "n_samples": result.shape[0],
        "band_range": [start_band, end_band],
    }


@mcp.tool()
def fit_gaussian_to_spectra(
    handle: str,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    sensor: str = "nir",
    sample_indices: Union[list[int], None] = None,
    initial_amplitude: float = 1.0,
    initial_centre_nm: float = 0.0,
    initial_width_nm: float = 10.0,
) -> dict:
    """Fit a Gaussian absorption feature to a subset of spectra.

    Args:
        handle: Dataset handle.
        start_wavelength_nm: Start of the fitting window (nm).
        end_wavelength_nm: End of the fitting window (nm).
        sensor: "nir" or "tir".
        sample_indices: Optional list of sample indices.  Defaults to all.
        initial_amplitude: Initial guess for Gaussian amplitude.
        initial_centre_nm: Initial guess for centre wavelength (nm).
                           0.0 auto-selects the midpoint of the window.
        initial_width_nm: Initial guess for Gaussian width (std dev, nm).

    Returns:
        Dict with ``parameters`` (amplitude, centre_nm, width_nm per sample)
        and ``wavelengths_nm`` used for fitting.
    """
    obj = _require(handle)
    sp = _spectra_for(obj, sensor)

    wl = sp.wavelength
    idx_start = int(np.searchsorted(wl, start_wavelength_nm))
    idx_end = int(np.searchsorted(wl, end_wavelength_nm, side="right"))

    data = sp.spectra
    if sample_indices is not None:
        data = data[sample_indices, :]

    wl_window = wl[idx_start:idx_end]
    data_window = data[:, idx_start:idx_end]

    if initial_centre_nm == 0.0:
        initial_centre_nm = float(wl_window.mean())

    x0 = np.array([initial_amplitude, initial_centre_nm, initial_width_nm])
    params = fit_gaussian(wl_window, data_window, x0=x0)

    return {
        "parameters": [
            {"amplitude": float(r[0]), "centre_nm": float(r[1]), "width_nm": float(r[2])}
            for r in params
        ],
        "wavelengths_nm": wl_window.tolist(),
        "n_samples": params.shape[0],
    }


@mcp.tool()
def sqm_band_analysis(
    handle: str,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    sensor: str = "nir",
    sample_indices: Union[list[int], None] = None,
) -> dict:
    """Apply the Simple Quadratic Method (SQM) to extract absorption features.

    SQM fits a second-degree polynomial to a spectral window and reports
    the position and depth of the absorption minimum.

    Reference: https://doi.org/10.1016/j.rse.2011.11.025

    Args:
        handle: Dataset handle.
        start_wavelength_nm: Start of the analysis window (nm).
        end_wavelength_nm: End of the analysis window (nm).
        sensor: "nir" or "tir".
        sample_indices: Optional list of sample indices.

    Returns:
        Dict with per-sample ``centre_nm``, ``depth``, and ``width_nm``.
    """
    obj = _require(handle)
    sp = _spectra_for(obj, sensor)

    data = sp.spectra
    if sample_indices is not None:
        data = data[sample_indices, :]

    params, _coefs = sqm(
        sp.wavelength,
        data,
        start_wavelength=start_wavelength_nm,
        end_wavelength=end_wavelength_nm,
    )

    return {
        "results": [
            {"centre_nm": float(r[0]), "depth": float(r[1]), "width_nm": float(r[2])}
            for r in params
        ],
        "n_samples": params.shape[0],
    }


@mcp.tool()
def get_lidar_profile(handle: str, max_samples: int = 1000) -> dict:
    """Return the lidar (profilometer) height profile for a TSG package.

    Args:
        handle: Dataset handle (must be a TSG package, not a bare Spectra).
        max_samples: Maximum number of values to return.

    Returns:
        Dict with ``lidar_mm`` list and ``n_total``.
    """
    obj = _require(handle)
    if not isinstance(obj, parse_tsg.TSG):
        return {"error": "Handle does not point to a full TSG package."}
    if obj.lidar is None:
        return {"error": "This package does not contain a lidar profile."}

    lidar = obj.lidar
    truncated = len(lidar) > max_samples
    return {
        "lidar_mm": lidar[:max_samples].tolist(),
        "n_total": len(lidar),
        "truncated": truncated,
    }




# ---------------------------------------------------------------------------
# Tools — capability discovery and TSG Desktop configuration
# ---------------------------------------------------------------------------


@mcp.tool()
def get_agent_capabilities(
    tsg_exe_path: Union[str, None] = None,
) -> dict:
    """Detect which operating modes are available in the current environment.

    Always call this tool at the start of a session to understand what is
    possible before deciding which tools to use.

    **Mode 1** — pytsg only (always available, no TSG licence needed).
    **Mode 2** — pytsg + TSG Desktop manual guidance (always available;
        requires the user to have TSG Desktop installed and a licence).
    **Mode 3** — pytsg + TSG headless automated (available only when
        TSGHeadless is installed on the host and reachable on PATH, or via
        the ``TSG_HEADLESS_EXE`` environment variable, or via
        ``tsg_exe_path``).

    Args:
        tsg_exe_path: Optional explicit path to the TSGHeadless executable.
                      Leave empty for auto-detection.

    Returns:
        Dict describing each mode with ``available`` (bool), ``description``
        (str), and for Mode 3 the detected executable path and version.
    """
    global _tsg_installation
    caps = tsg_desktop.get_capabilities(tsg_exe_path)
    if caps["mode_3_pytsg_plus_tsg_headless"]["available"] and not _tsg_installation:
        _tsg_installation = tsg_desktop.detect_tsg_installation(tsg_exe_path)
    return caps


@mcp.tool()
def configure_tsg_desktop(
    exe_path: str,
    headless_cmd_template: Union[list[str], None] = None,
) -> dict:
    """Set the path to the TSGHeadless executable (Mode 3 configuration).

    Call this tool when TSGHeadless is installed at a non-standard location
    that was not picked up by auto-detection.  You only need to call it once
    per session.

    Args:
        exe_path: Absolute path to the ``TSGHeadless`` (or
            ``TSGHeadless.exe``) binary.
        headless_cmd_template: Optional custom argument template.  Each
            element is a string token; the following placeholders are
            substituted at run time:
            ``{exe}`` ``{dataset}`` ``{output_dir}`` ``{export_format}``
            ``{log_file}``.
            Leave empty to use the default template.

            Default template::

                ["{exe}", "-t", "{dataset}", "-o", "{output_dir}",
                 "-s", "-e", "{export_format}", "-l", "{log_file}"]

    Returns:
        Dict confirming the configured path and whether the file exists.
    """
    global _tsg_installation
    p = Path(exe_path).expanduser().resolve()
    version = tsg_desktop._probe_version(p) if p.exists() else None
    inst = tsg_desktop.TsgInstallation(exe_path=p, version=version)
    if headless_cmd_template is not None:
        inst.headless_cmd_template = headless_cmd_template
    _tsg_installation = inst
    return {
        "configured": True,
        "exe_path": str(p),
        "exe_exists": p.exists(),
        "version": version,
        "cmd_template": inst.headless_cmd_template,
    }


# ---------------------------------------------------------------------------
# Tools — Mode 2: manual TSG Desktop guidance
# ---------------------------------------------------------------------------


@mcp.tool()
def get_tsg_manual_instructions(
    operation: str,
    dataset_folder: str,
    output_path: Union[str, None] = None,
    algorithm_name: Union[str, None] = None,
    class_name: Union[str, None] = None,
) -> dict:
    """Return step-by-step instructions for performing a TSG Desktop operation
    manually (Mode 2).

    Use this tool when the user has TSG Desktop installed but you cannot or
    should not drive it programmatically.  The instructions are returned as
    markdown text that you should present to the user.  After the user
    confirms they have completed the steps, reload the dataset with
    ``read_tsg_package`` to pick up any updated scalars.

    Args:
        operation: One of:
            ``"run_scalars"`` — compute all registered scalar algorithms,
            ``"export_csv"``  — export scalar results to a CSV file,
            ``"view_classification"`` — open a mineral-classification map,
            ``"apply_algorithm"`` — apply a named processing algorithm.
        dataset_folder: Path to the TSG package folder the user should open.
        output_path: Required for ``"export_csv"``.  Suggested output file path.
        algorithm_name: Required for ``"apply_algorithm"``.  Name of the
            TSG algorithm to apply (e.g. ``"Feature Extraction"``).
        class_name: Required for ``"view_classification"``.  Name of the
            classification scalar to display.

    Returns:
        Dict with ``operation``, ``instructions`` (markdown), and a
        ``next_step`` hint for the agent.
    """
    op = operation.lower().strip()

    if op == "run_scalars":
        text = tsg_desktop.manual_guidance_run_scalars(dataset_folder)
        next_step = (
            "Wait for the user to confirm they have finished running scalars "
            "and saving the dataset, then call read_tsg_package to reload "
            "the updated results."
        )
    elif op == "export_csv":
        if output_path is None:
            output_path = str(Path(dataset_folder) / "scalars_export.csv")
        text = tsg_desktop.manual_guidance_export_csv(dataset_folder, output_path)
        next_step = (
            f"Wait for the user to confirm they have exported the CSV, "
            f"then read the file at '{output_path}' with a file-read tool "
            "or ask the user to share its contents."
        )
    elif op == "view_classification":
        if class_name is None:
            class_name = "(specify class name)"
        text = tsg_desktop.manual_guidance_open_class_map(dataset_folder, class_name)
        next_step = (
            "Ask the user to describe what they see in the classification map, "
            "or to export and share the image."
        )
    elif op == "apply_algorithm":
        if algorithm_name is None:
            algorithm_name = "(specify algorithm name)"
        text = tsg_desktop.manual_guidance_apply_algorithm(dataset_folder, algorithm_name)
        next_step = (
            "Wait for the user to confirm they have applied the algorithm and "
            "saved the dataset, then call read_tsg_package to reload the results."
        )
    else:
        return {
            "error": (
                f"Unknown operation '{operation}'.  "
                "Valid values: run_scalars, export_csv, "
                "view_classification, apply_algorithm."
            )
        }

    return {
        "operation": operation,
        "dataset_folder": dataset_folder,
        "instructions": text,
        "next_step": next_step,
    }


@mcp.tool()
def check_dataset_updated(
    dataset_folder: str,
    since_iso: Union[str, None] = None,
) -> dict:
    """Check whether the .tsg / .bip files in a dataset folder have been
    modified recently (useful in Mode 2 to detect when the user has saved
    from TSG Desktop).

    Args:
        dataset_folder: Path to the TSG package folder.
        since_iso: ISO-8601 datetime string (e.g. ``"2024-06-01T12:00:00"``).
            If supplied, only files modified **after** this time are reported.
            If omitted, all file modification times are returned.

    Returns:
        Dict with ``files`` (list of dicts with ``name`` and ``modified``),
        ``any_updated`` (bool), and ``folder``.
    """
    import datetime

    p = Path(dataset_folder).expanduser().resolve()
    if not p.exists():
        return {"error": f"Folder not found: {dataset_folder}"}

    since_dt: Union[datetime.datetime, None] = None
    if since_iso is not None:
        try:
            since_dt = datetime.datetime.fromisoformat(since_iso)
        except ValueError:
            return {"error": f"Invalid ISO datetime: {since_iso}"}

    tsg_extensions = {".tsg", ".bip", ".dat"}
    files = []
    any_updated = False
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in tsg_extensions:
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            updated = since_dt is None or mtime > since_dt
            if updated:
                any_updated = True
            files.append(
                {
                    "name": f.name,
                    "modified": mtime.isoformat(),
                    "updated_since_check": updated,
                }
            )

    return {"folder": str(p), "files": files, "any_updated": any_updated}


# ---------------------------------------------------------------------------
# Tools — Mode 3: TSG headless (automated)
# ---------------------------------------------------------------------------


@mcp.tool()
def run_tsg_headless_scalars(
    dataset_folder: str,
    output_dir: Union[str, None] = None,
    timeout_seconds: int = 600,
) -> dict:
    """Run TSG scalar processing on a dataset folder using TSGHeadless
    (Mode 3 — automated).

    TSGHeadless will open the dataset, execute all registered scalar
    algorithms, write a log file, and exit.  Requires a valid TSG Pro
    licence.

    Prerequisites:
        Call ``get_agent_capabilities`` first to confirm Mode 3 is available.
        If TSGHeadless was not auto-detected, call ``configure_tsg_desktop``
        to supply its path.

    Args:
        dataset_folder: Path to the TSG package folder to process.
        output_dir: Directory for TSGHeadless output files.  Defaults to a
            subdirectory ``tsg_output`` inside ``dataset_folder``.
        timeout_seconds: Kill the process after this many seconds (default
            600).  Increase for large datasets.

    Returns:
        Dict with ``success``, ``returncode``, ``stdout``, ``stderr``,
        ``exported_files``, and ``log_file``.  On success, call
        ``read_tsg_package`` to reload the updated dataset.
    """
    global _tsg_installation

    if _tsg_installation is None:
        _tsg_installation = tsg_desktop.detect_tsg_installation()

    if _tsg_installation is None or not _tsg_installation.is_available():
        return {
            "error": (
                "TSGHeadless executable not found.  "
                "Set the TSG_HEADLESS_EXE environment variable, place "
                "TSGHeadless on PATH, or call configure_tsg_desktop first."
            ),
            "mode": "3",
            "fallback": (
                "Use Mode 2 instead: call get_tsg_manual_instructions with "
                "operation='run_scalars' to guide the user through running "
                "scalars in TSG Desktop manually."
            ),
        }

    p = Path(dataset_folder).expanduser().resolve()
    if output_dir is None:
        out = p / "tsg_output"
    else:
        out = Path(output_dir).expanduser().resolve()

    result = tsg_desktop.run_headless(
        _tsg_installation,
        dataset=p,
        output_dir=out,
        export_format="csv",
        timeout_seconds=timeout_seconds,
    )
    return result.to_dict()


@mcp.tool()
def run_tsg_headless_export(
    dataset_folder: str,
    output_dir: Union[str, None] = None,
    export_format: str = "csv",
    timeout_seconds: int = 600,
) -> dict:
    """Export TSG scalar results using TSGHeadless (Mode 3 — automated).

    Runs TSGHeadless to export the current scalar values from a dataset to
    the specified format.  Use this after scalars have already been computed
    (either via a prior ``run_tsg_headless_scalars`` call or by the user
    running TSG Desktop manually).

    Args:
        dataset_folder: Path to the TSG package folder.
        output_dir: Directory for the exported files.  Defaults to
            ``<dataset_folder>/tsg_output``.
        export_format: ``"csv"`` (default) or ``"envi"``.
        timeout_seconds: Process timeout (default 600).

    Returns:
        Dict with ``success``, ``exported_files``, and ``log_file``.
    """
    global _tsg_installation

    if _tsg_installation is None:
        _tsg_installation = tsg_desktop.detect_tsg_installation()

    if _tsg_installation is None or not _tsg_installation.is_available():
        return {
            "error": (
                "TSGHeadless executable not found.  "
                "Use Mode 2: call get_tsg_manual_instructions with "
                "operation='export_csv' to guide the user through "
                "exporting from TSG Desktop manually."
            ),
        }

    p = Path(dataset_folder).expanduser().resolve()
    if output_dir is None:
        out = p / "tsg_output"
    else:
        out = Path(output_dir).expanduser().resolve()

    result = tsg_desktop.run_headless(
        _tsg_installation,
        dataset=p,
        output_dir=out,
        export_format=export_format,
        timeout_seconds=timeout_seconds,
    )
    return result.to_dict()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pytsg MCP server — expose TSG spectral-data tools to AI agents"
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use streamable-HTTP transport instead of stdio.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to when using --http (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on when using --http (default: 8765).",
    )
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
