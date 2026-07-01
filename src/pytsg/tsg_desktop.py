"""tsg_desktop — helpers for interacting with TSG Desktop software.

TSG (The Spectral Geologist) by CSIRO has two desktop-integration paths
that complement the pure-Python pytsg reader:

**Mode 2 — manual guidance**
The agent cannot drive TSG itself, but can produce step-by-step
instructions for the user to carry out in the TSG GUI, then read the
results once the user has saved.

**Mode 3 — headless execution**
TSG ships a headless executable (``TSGHeadless.exe`` on Windows,
``TSGHeadless`` on macOS/Linux) that can run processing jobs from the
command line without the graphical interface.  A valid TSG Pro licence
is required.

.. note::

    CSIRO has not published the TSGHeadless CLI specification publicly.
    The flag names and task-file format used in this module are derived
    from community knowledge and may vary across TSG versions.  If you
    observe a mismatch, override the command template via
    :meth:`TsgInstallation.headless_cmd_template`.  Verified corrections
    are very welcome as pull-requests.

Reference for the TSG file formats parsed by pytsg:
    https://research.csiro.au/thespectralgeologist/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Known installation locations (searched in order)
# ---------------------------------------------------------------------------

_WINDOWS_DEFAULT_PATHS: list[str] = [
    r"C:\Program Files\CSIRO\TSG\TSGHeadless.exe",
    r"C:\Program Files\TSG\TSGHeadless.exe",
    r"C:\Program Files (x86)\CSIRO\TSG\TSGHeadless.exe",
    r"C:\Program Files (x86)\TSG\TSGHeadless.exe",
    r"C:\TSG\TSGHeadless.exe",
]

_MACOS_DEFAULT_PATHS: list[str] = [
    "/Applications/TSG.app/Contents/MacOS/TSGHeadless",
    "/usr/local/bin/TSGHeadless",
]

_LINUX_DEFAULT_PATHS: list[str] = [
    "/usr/local/bin/TSGHeadless",
    "/opt/TSG/TSGHeadless",
]

# env-var that users can set to point directly at the executable
_ENV_VAR = "TSG_HEADLESS_EXE"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TsgInstallation:
    """Represents a located TSG headless installation.

    Attributes:
        exe_path: Absolute path to the ``TSGHeadless`` executable.
        version: Version string reported by the executable, or ``None``
            if it could not be determined.
        headless_cmd_template: Command-line argument template used when
            building headless invocations.  The template is a list of
            tokens where the following placeholders are substituted:

            * ``{dataset}`` — absolute path to the TSG package folder
            * ``{output_dir}`` — absolute path for output files
            * ``{export_format}`` — export format string (``csv``, ``envi``)
            * ``{log_file}`` — absolute path for the log file

            The default template reflects the most commonly observed
            TSG headless convention.  Override it if your TSG version uses
            different flags.
    """

    exe_path: Path
    version: Union[str, None] = None
    headless_cmd_template: list[str] = field(
        default_factory=lambda: [
            "{exe}",
            "-t",
            "{dataset}",
            "-o",
            "{output_dir}",
            "-s",         # run scalars
            "-e",
            "{export_format}",
            "-l",
            "{log_file}",
        ]
    )

    def is_available(self) -> bool:
        return self.exe_path.exists()

    def build_command(
        self,
        dataset: Path,
        output_dir: Path,
        export_format: str = "csv",
        log_file: Union[Path, None] = None,
    ) -> list[str]:
        """Substitute placeholders and return a ready-to-run command list."""
        if log_file is None:
            log_file = output_dir / "tsg_headless.log"
        subs = {
            "exe": str(self.exe_path),
            "dataset": str(dataset),
            "output_dir": str(output_dir),
            "export_format": export_format,
            "log_file": str(log_file),
        }
        return [token.format(**subs) for token in self.headless_cmd_template]


@dataclass
class HeadlessResult:
    """Result of a TSG headless invocation.

    Attributes:
        success: ``True`` when the process exited with code 0.
        returncode: The process return-code.
        stdout: Standard output captured from the process.
        stderr: Standard error captured from the process.
        output_dir: Directory where TSG wrote its output files.
        log_file: Path to the TSG log file (if created).
        exported_files: List of paths created in ``output_dir``.
    """

    success: bool
    returncode: int
    stdout: str
    stderr: str
    output_dir: Path
    log_file: Union[Path, None]
    exported_files: list[Path]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "returncode": self.returncode,
            "stdout": self.stdout[-4000:] if len(self.stdout) > 4000 else self.stdout,
            "stderr": self.stderr[-4000:] if len(self.stderr) > 4000 else self.stderr,
            "output_dir": str(self.output_dir),
            "log_file": str(self.log_file) if self.log_file else None,
            "exported_files": [str(f) for f in self.exported_files],
        }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_tsg_installation(
    exe_path_override: Union[str, Path, None] = None,
) -> Union[TsgInstallation, None]:
    """Search for a TSG headless installation and return a
    :class:`TsgInstallation`, or ``None`` if not found.

    Search order:
    1. ``exe_path_override`` argument (if supplied).
    2. ``TSG_HEADLESS_EXE`` environment variable.
    3. ``TSGHeadless`` / ``TSGHeadless.exe`` on ``PATH`` (via ``shutil.which``).
    4. Platform-specific default install paths.

    Args:
        exe_path_override: Explicit path to the TSGHeadless executable.

    Returns:
        :class:`TsgInstallation` if found, otherwise ``None``.
    """
    candidates: list[Path] = []

    if exe_path_override is not None:
        candidates.append(Path(exe_path_override).expanduser().resolve())

    env_val = os.environ.get(_ENV_VAR)
    if env_val:
        candidates.append(Path(env_val).expanduser().resolve())

    exe_name = "TSGHeadless.exe" if sys.platform == "win32" else "TSGHeadless"
    which_result = shutil.which(exe_name)
    if which_result:
        candidates.append(Path(which_result))

    if sys.platform == "win32":
        platform_paths = _WINDOWS_DEFAULT_PATHS
    elif sys.platform == "darwin":
        platform_paths = _MACOS_DEFAULT_PATHS
    else:
        platform_paths = _LINUX_DEFAULT_PATHS

    for p in platform_paths:
        candidates.append(Path(p))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            version = _probe_version(candidate)
            return TsgInstallation(exe_path=candidate, version=version)

    return None


def _probe_version(exe: Path) -> Union[str, None]:
    """Try to get the TSGHeadless version string by running ``--version``."""
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        if output:
            return output[:200]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Headless execution
# ---------------------------------------------------------------------------


def run_headless(
    installation: TsgInstallation,
    dataset: Union[str, Path],
    output_dir: Union[str, Path],
    export_format: str = "csv",
    timeout_seconds: int = 600,
) -> HeadlessResult:
    """Run TSG headless processing on a dataset package directory.

    TSG will:
    1. Open the dataset at ``dataset``.
    2. Compute all registered scalar algorithms (``-s`` flag).
    3. Export results to ``output_dir`` in the requested format.

    .. note::

        This function requires a valid TSG Pro licence.  The scalar
        algorithms that will run are those configured in the TSG dataset
        itself (saved in the ``.tsg`` file by the TSG GUI or a prior
        headless run).

    Args:
        installation: A :class:`TsgInstallation` obtained from
            :func:`detect_tsg_installation`.
        dataset: Path to the TSG package folder (contains the ``.tsg``
            and ``.bip`` files).
        output_dir: Directory where TSG will write exported CSV/ENVI files.
            Created automatically if it does not exist.
        export_format: Export format — ``"csv"`` (default) or ``"envi"``.
        timeout_seconds: Maximum seconds to wait before killing the
            process (default 600).

    Returns:
        :class:`HeadlessResult` with process outcome and output file list.
    """
    dataset_path = Path(dataset).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "tsg_headless.log"
    cmd = installation.build_command(
        dataset_path, output_path, export_format, log_file
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        success = proc.returncode == 0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = f"TSGHeadless timed out after {timeout_seconds}s."
        proc = None
    except FileNotFoundError:
        success = False
        stdout = ""
        stderr = f"Executable not found: {cmd[0]}"
        proc = None

    returncode = proc.returncode if proc is not None else -1

    exported: list[Path] = sorted(output_path.iterdir()) if output_path.exists() else []
    log_path = log_file if log_file.exists() else None

    return HeadlessResult(
        success=success,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        output_dir=output_path,
        log_file=log_path,
        exported_files=exported,
    )


# ---------------------------------------------------------------------------
# Mode 2 — Manual GUI guidance
# ---------------------------------------------------------------------------


def manual_guidance_run_scalars(dataset_folder: str) -> str:
    """Return step-by-step instructions for running TSG scalar processing
    manually in the TSG Desktop GUI.

    Args:
        dataset_folder: Path to the TSG package folder.

    Returns:
        Multi-line instruction string suitable for display to the user.
    """
    return textwrap.dedent(f"""
        ## TSG Desktop — Run Scalar Processing (Manual Steps)

        Perform these steps in the **TSG Desktop** application:

        1. **Open TSG Desktop** on your computer.

        2. **Open the dataset**
           File → Open Dataset… → navigate to:
               {dataset_folder}
           Select the folder and click **Open**.

        3. **Run scalars**
           Processing → Run Scalars (or press **F5**).
           Wait for the progress bar to complete.

        4. **Save the dataset**
           File → Save (Ctrl+S / Cmd+S).
           This writes updated scalar values back into the `.bip` / `.tsg`
           files in the dataset folder.

        5. **Notify the agent**
           Once the save is complete, tell the agent:
               "I have finished running scalars and saved the dataset."
           The agent will then reload the file and analyse the results.

        > **Tip:** You can also use *Processing → Batch Process* if you
        > need to run scalars across multiple dataset folders at once.
    """).strip()


def manual_guidance_export_csv(dataset_folder: str, output_path: str) -> str:
    """Return step-by-step instructions for exporting TSG scalar results
    to CSV from the TSG Desktop GUI.

    Args:
        dataset_folder: Path to the open TSG dataset.
        output_path: Suggested path for the exported CSV file.

    Returns:
        Multi-line instruction string.
    """
    return textwrap.dedent(f"""
        ## TSG Desktop — Export Scalars to CSV (Manual Steps)

        1. **Open (or confirm) the dataset is open in TSG Desktop:**
               {dataset_folder}

        2. **Export scalars to CSV**
           File → Export → Scalars / Band Results…
           Set the output file to:
               {output_path}
           Click **Export**.

        3. **Notify the agent**
           Tell the agent: "I have exported the scalars to {output_path}."
           The agent will read the file and summarise the results.
    """).strip()


def manual_guidance_open_class_map(dataset_folder: str, class_name: str) -> str:
    """Return instructions for viewing a mineral classification map in TSG."""
    return textwrap.dedent(f"""
        ## TSG Desktop — View Classification Map (Manual Steps)

        1. **Open the dataset in TSG Desktop:**
               {dataset_folder}

        2. **Open the mineralogy map**
           View → Scalar Profiles → {class_name}
           (Or select the scalar from the drop-down in the main toolbar.)

        3. **Optionally export the map image**
           File → Export → Image… and save to a folder of your choice.
           Tell the agent where you saved it to include it in the analysis.
    """).strip()


def manual_guidance_apply_algorithm(
    dataset_folder: str, algorithm_name: str
) -> str:
    """Return instructions for applying a named TSG algorithm manually."""
    return textwrap.dedent(f"""
        ## TSG Desktop — Apply Algorithm: {algorithm_name} (Manual Steps)

        1. **Open the dataset in TSG Desktop:**
               {dataset_folder}

        2. **Apply the algorithm**
           Processing → Algorithms → {algorithm_name}
           Configure the parameters as desired and click **Run**.

        3. **Save the dataset** (Ctrl+S / Cmd+S).

        4. **Notify the agent**
           Tell the agent: "I have applied {algorithm_name} and saved."
           The agent will reload the scalars and summarise the outputs.
    """).strip()


# ---------------------------------------------------------------------------
# Capability summary
# ---------------------------------------------------------------------------


def get_capabilities(
    exe_path_override: Union[str, Path, None] = None,
) -> dict:
    """Return a dict describing the available agent operating modes.

    This is used by the MCP server's ``get_agent_capabilities`` tool to
    tell the AI which operations are possible in the current environment.

    Returns:
        Dict with keys ``mode_1``, ``mode_2``, and ``mode_3``, each a
        sub-dict with ``available`` (bool) and ``description`` (str).
    """
    installation = detect_tsg_installation(exe_path_override)

    modes: dict = {
        "mode_1_pytsg_only": {
            "available": True,
            "description": (
                "Read and analyse TSG file packages using pure Python (pytsg). "
                "No TSG licence required.  Supports spectra loading, depth "
                "filtering, band-feature extraction, Gaussian fitting, SQM "
                "analysis, and lidar profiles."
            ),
        },
        "mode_2_pytsg_plus_manual_tsg": {
            "available": True,
            "description": (
                "Use pytsg to read TSG files PLUS step-by-step guidance so the "
                "user can run TSG Desktop manually.  The agent issues instructions "
                "(e.g. 'run scalars', 'export CSV') and reads the results once the "
                "user saves them.  Requires the user to have TSG Desktop installed "
                "and a valid licence."
            ),
        },
        "mode_3_pytsg_plus_tsg_headless": {
            "available": installation is not None and installation.is_available(),
            "description": (
                "Use pytsg to read TSG files AND drive TSGHeadless from the "
                "command line to run scalar processing and exports automatically "
                "without any manual steps.  Requires TSG Pro (headless edition) "
                "installed with a valid licence."
            ),
            "tsg_exe": str(installation.exe_path) if installation else None,
            "tsg_version": installation.version if installation else None,
        },
    }

    active_modes = [k for k, v in modes.items() if v["available"]]
    modes["recommended_mode"] = active_modes[-1]  # prefer the most capable available
    modes["tsg_headless_env_var"] = _ENV_VAR

    return modes
