# pytsg
## Rationale
The spectral geologist (TSG) is an industry standard software for hyperspectral data analysis
https://research.csiro.au/thespectralgeologist/

pytsg is an open source one function utility that imports the spectral geologist file package into a simple object.

## Installation
Installation is via pip
```pip install pytsg```

To also install the AI-agent MCP server:
```pip install "pytsg[agent]"```

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
server so that AI coding assistants — including **VS Code GitHub Copilot** (agent mode),
**Claude Desktop**, and any other MCP-compatible client — can operate TSG data directly
through natural language.

### Prerequisites

Install the optional `agent` extra:

```bash
pip install "pytsg[agent]"
```

### Starting the server

**stdio transport** (used by VS Code and Claude Desktop):

```bash
pytsg-mcp
```

**Streamable-HTTP transport** (for web-based clients):

```bash
pytsg-mcp --http --host 127.0.0.1 --port 8765
```

### VS Code setup

This repository ships a `.vscode/mcp.json` file that registers the server
automatically. Once `pytsg-mcp` is on your `PATH`, open the repository in VS
Code and enable **GitHub Copilot agent mode** — the pytsg tools will appear
in Copilot's tool list.

To verify, open the Copilot Chat panel, switch to **Agent** mode, and type:

> Load the TSG package at `example_data/ETG0187` and tell me about the NIR spectra.

### Claude Desktop setup

Add the following block to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or the equivalent path on your OS:

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

Restart Claude Desktop and the pytsg tools will be available automatically.

### Available MCP tools

| Tool | Description |
|------|-------------|
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

### Example agent session

```
User:  Load example_data/ETG0187 and show me the NIR wavelength range.

Agent: [calls read_tsg_package then get_spectra_summary]
       The NIR sensor covers 380–2500 nm across 512 bands with 1247 samples.

User:  Extract the 2200 nm clay absorption feature for samples 0–99.

Agent: [calls extract_band_features with start/end wavelengths and indices]
       Band minimum position ranges from 2195–2210 nm across those samples…
```

## Thanks
Thanks to CSIRO and in particular Andrew Rodger for his assistance in decoding the file structures.