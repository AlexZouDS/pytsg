"""MCP server for pytsg — exposes TSG hyperspectral data operations as AI-callable tools.

Run with:
    pytsg-mcp          # stdio transport (VS Code / Claude Desktop)
    pytsg-mcp --http   # streamable-HTTP transport (web clients)

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

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "pytsg",
    instructions=(
        "Use this server to load and analyse TSG (The Spectral Geologist) "
        "hyperspectral drill-core datasets. "
        "Start by calling read_tsg_package or read_tsg_bip_pair to load data, "
        "then call the analysis tools to query spectra, depth headers, and "
        "mineral-classification scalars."
    ),
)

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
