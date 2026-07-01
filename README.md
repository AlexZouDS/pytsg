# pytsg
## Rationale
The spectral geologist (TSG) is an industry standard software for hyperspectral data analysis
https://research.csiro.au/thespectralgeologist/

pytsg is an open source one function utility that imports the spectral geologist file package into a simple object.

## Installation
Installation is via pip

    pip install pytsg

To also install the AI-agent MCP server:

    pip install "pytsg[agent]"

## Usage

If using the top level importer the data is assumed to follow this structure
```
\HOLENAME
         \HOLEMAME_tsg.bip
         \HOLENAME_tsg.tsg
         \HOLENAME_tsg_tir.bip
         \HOLENAME_tsg_tir.tsg
         \HOLENAME_tsg_hires.dat
         \HOLENAME_tsg_cras.bip

```

```python
from matplotlib import pyplot as plt
from pytsg import parse_tsg

data = parse_tsg.read_package('example_data/ETG0187')

plt.plot(data.nir.wavelength, data.nir.spectra[0, 0:10, :].T)
plt.plot(data.tir.wavelength, data.tir.spectra[0, 0:10, :].T)
plt.xlabel('Wavelength nm')
plt.ylabel('Reflectance')
plt.title('pytsg reads tsg files')
plt.show()

```

If you would prefer to have full control over importing individual files the following syntax is what you need

```python

# bip files
nir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg.tsg','ETG0187_tsg.bip','nir')
tir = parse_tsg.read_tsg_bip_pair('ETG0187_tsg_tir.tsg','ETG0187_tsg_tir.bip','tir')

# cras file
cras = parse_tsg.read_cras('ETG0187_tsg_cras.bip')

# hires dat file
lidar = parse_tsg.read_lidar('ETG0187_tsg_hires.dat')


```
For convienience 

---

## AI Agent / MCP Server

pytsg ships a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
server (`pytsg-mcp`) so that any MCP-compatible AI assistant or client can load,
query, and analyse TSG hyperspectral drill-core datasets through natural language.

### Three operating modes

The agent automatically adapts to what is installed on the host.  Always start
by asking the agent to call `get_agent_capabilities`.

| Mode | Requirements | What the agent can do |
|------|-------------|----------------------|
| **1 — pytsg only** | `pip install "pytsg[agent]"` | Read `.tsg`/`.bip` files in pure Python; query spectra, depth headers, scalars; run band-feature extraction, Gaussian fitting, SQM analysis |
| **2 — pytsg + TSG Desktop (manual)** | TSG Desktop installed + licence | All of Mode 1, plus the agent issues step-by-step GUI instructions for the user to run inside TSG Desktop (run scalars, export CSV, apply algorithms); agent reads the results after the user saves |
| **3 — pytsg + TSG headless (automated)** | TSG Pro headless edition + licence | All of Modes 1 & 2, plus the agent drives `TSGHeadless` from the command line automatically — no user interaction required |

> **No TSG licence?**  Mode 1 is fully open-source and requires only `pytsg`.
>
> **Have TSG Desktop?**  Mode 2 lets the agent guide you through TSG operations
> without any extra setup.
>
> **Have TSG Pro (headless)?**  Mode 3 is fully automated.  Set the
> `TSG_HEADLESS_EXE` environment variable to the path of `TSGHeadless.exe`
> (Windows) or `TSGHeadless` (macOS/Linux), or call `configure_tsg_desktop` in
> the agent session.

---

### Installation

```bash
pip install "pytsg[agent]"
```

### Starting the server

**stdio transport** (for MCP clients that launch local commands):

```bash
pytsg-mcp
```

**Streamable-HTTP transport** (for web-based clients):

```bash
pytsg-mcp --http --host 127.0.0.1 --port 8765
```

### Client setup

Any MCP-compatible client can connect to `pytsg-mcp` by registering the
following stdio server definition in its MCP configuration:

```json
{
  "mcpServers": {
    "pytsg": {
      "command": "pytsg-mcp",
      "args": []
    }
  }
}
```

This repository also ships the same server definition at `.vscode/mcp.json`,
which can be reused by MCP-aware tools that read workspace configuration files.

To verify the connection, ask your client to:

> Load the TSG package at `example_data/ETG0187` and tell me about the NIR spectra.

### TSGHeadless setup (Mode 3 only)

Obtain TSG Pro (headless edition) from CSIRO:
<https://research.csiro.au/thespectralgeologist/tsg/compare-versions/>

Then point the agent at the executable via the environment variable:

```bash
# Windows (PowerShell)
$env:TSG_HEADLESS_EXE = "C:\Program Files\CSIRO\TSG\TSGHeadless.exe"
pytsg-mcp

# macOS / Linux
export TSG_HEADLESS_EXE="/usr/local/bin/TSGHeadless"
pytsg-mcp
```

Or configure it at runtime inside the agent session:

```
User:  The TSGHeadless binary is at D:	ools\TSGHeadless.exe

Agent: [calls configure_tsg_desktop with that path]
       Mode 3 is now active. I can run processing automatically.
```

> **Note:** TSGHeadless flags can vary by version. Use
> `configure_tsg_desktop(headless_cmd_template=[…])` to override the
> template if needed.

---

### Available MCP tools

#### All modes (Mode 1+)

| Tool | Description |
|------|-------------|
| `get_agent_capabilities` | **Start here** — detect available modes |
| `read_tsg_package` | Load a full TSG package directory |
| `read_tsg_bip_pair` | Load a single .tsg / .bip file pair |
| `list_loaded_datasets` | List datasets currently in memory |
| `unload_dataset` | Remove a dataset to free memory |
| `get_spectra_summary` | Metadata: shape, wavelength range, column names |
| `get_wavelengths` | Full wavelength axis (nm) |
| `get_sample_headers` | Per-sample depth / position headers |
| `get_scalars` | TSG-derived mineral-classification scalars |
| `get_spectra_at_index` | Raw reflectance for specific sample indices |
| `get_spectra_at_depth_range` | Spectra filtered by depth interval |
| `extract_band_features` | Band statistics (position, depth, area) |
| `fit_gaussian_to_spectra` | Gaussian absorption-feature fitting |
| `sqm_band_analysis` | Simple Quadratic Method absorption analysis |
| `get_lidar_profile` | Profilometer height profile |

#### Mode 2 — manual TSG Desktop guidance

| Tool | Description |
|------|-------------|
| `get_tsg_manual_instructions` | Get step-by-step GUI instructions for an operation (`run_scalars`, `export_csv`, `view_classification`, `apply_algorithm`) |
| `check_dataset_updated` | Poll dataset folder to detect when the user has saved from TSG Desktop |

#### Mode 3 — TSG headless (automated)

| Tool | Description |
|------|-------------|
| `configure_tsg_desktop` | Set TSGHeadless path and optional command template |
| `run_tsg_headless_scalars` | Run scalar processing via TSGHeadless |
| `run_tsg_headless_export` | Export results to CSV/ENVI via TSGHeadless |

---

### Example agent sessions

**Mode 1 — pytsg only**

```
User:  I don't have TSG. Load example_data/ETG0187 and show the NIR wavelength range.

Agent: [calls get_agent_capabilities → Mode 1 only]
       [calls read_tsg_package, then get_spectra_summary]
       The NIR sensor covers 380–2500 nm across 512 bands with 1247 samples.

User:  Extract the 2200 nm clay absorption feature for samples 0–99.

Agent: [calls extract_band_features]
       Band minimum position ranges from 2195–2210 nm across those samples.
```

**Mode 2 — manual TSG Desktop**

```
User:  I have TSG Desktop. Can you help me run scalars on my dataset at C:\data\HOLE001?

Agent: [calls get_agent_capabilities → Mode 2 available]
       [calls get_tsg_manual_instructions(operation="run_scalars", dataset_folder="C:\data\HOLE001")]
       Here are the steps to run in TSG Desktop:
       1. Open TSG Desktop
       2. File → Open Dataset → C:\data\HOLE001
       ...

User:  Done, I saved it.

Agent: [calls read_tsg_package to reload the updated scalars]
       Great! I can see TSG has computed 8 scalar bands including KAOLINITE
       and WHITE MICA classifications.
```

**Mode 3 — TSG headless (fully automated)**

```
User:  Run scalars on C:\data\HOLE001 automatically.

Agent: [calls get_agent_capabilities → Mode 3 available (TSGHeadless found)]
       [calls run_tsg_headless_scalars(dataset_folder="C:\data\HOLE001")]
       TSGHeadless completed successfully. Output written to C:\data\HOLE001	sg_output\.
       [calls read_tsg_package to load the updated results]
       Scalars computed: KAOLINITE, WHITE MICA, CHLORITE at 1247 sample depths.
```

## Thanks
Thanks to CSIRO and in particular Andrew Rodger for his assistance in decoding the file structures.