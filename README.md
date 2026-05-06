# NeuroTrace

A Julia package for reading, analyzing, and visualizing neuronal data from [NWB](https://www.nwb.org/) (Neurodata Without Borders) files.

> Work in progress — built as a first Julia package. Feedback welcome.

---

## Features

- **Session overview** — quickly inspect the contents and structure of any NWB file
- **Raster + PSTH** — peri-event spike rasters and firing-rate histograms for individual units
- **Z-scored heatmap** — population-level view of responses aligned to behavioral events
- **3D brain viewer** — probe placement visualized in the Allen Brain Atlas CCFv3
- **ZETA test** — statistical detection of unit responsiveness via [ZetaJu](https://github.com/PierreLeMerre/ZetaJu)

---

## Requirements

- Julia ≥ 1.9
- NWB files (`.nwb`) from extracellular electrophysiology recordings

---

## Installation

```julia
import Pkg
Pkg.activate("path/to/Neurotrace")
Pkg.instantiate()
```

> `ZetaJu` is a local dependency — make sure it lives at `../ZetaJu` relative to this repo, or update the path in `Project.toml`.

---

## Quick start

### Explore any NWB file (no config needed)

```bash
julia --project=. scripts/explore_nwb.jl path/to/file.nwb
```

Or from the REPL:

```julia
include("scripts/explore_nwb.jl")
explore("path/to/file.nwb")
```

Automatically detects whether the file contains spike-sorted units or a continuous electrical series and saves a figure (`nwb_plot.png`).

### Run the analysis scripts

All scripts share a single configuration file: [`scripts/config.toml`](scripts/config.toml).  
Edit it once to point to your data, then run any script without touching the code.

| Script | What it does |
|---|---|
| `explore_nwb.jl` | File-level overview: structure, metadata, raster or trace |
| `Explore_data.jl` | Session-level overview across one or multiple NWB files |
| `unit_raster_psth.jl` | Raster + PSTH for a single unit, one column per event type |
| `Unit_zeta_test.jl` | ZETA responsiveness test + IFR for a single unit |
| `population_heatmap.jl` | Z-scored population heatmap aligned to behavioral events |
| `Brain3D.jl` | 3D probe placement in the Allen CCFv3 atlas |

Scripts can be run cell-by-cell in VS Code (`Shift+Enter`) or as whole files (`Ctrl+F5`).

---

## Configuration

```toml
# scripts/config.toml

data_path = "/path/to/data.nwb"   # single file or directory of .nwb files

[[events]]
label = "Trial start"
start = "intervals/trials/start_time"
stop  = "intervals/trials/stop_time"
color = "#000000"

win_start = -1.0    # seconds before event
win_stop  =  2.0    # seconds after event
psth_bin  =  0.005  # bin width (s)

min_firing_rate = 0.01   # Hz — set to 0 to keep all units
regions = []             # leave empty for all regions
```

Multiple `[[events]]` blocks produce multiple columns in the raster/PSTH and heatmap plots.

---

## Gallery

### Session overview

<p align="center">
    <img width="800" src="images/overview.png">
</p>

### Raster + PSTH

<p align="center">
    <img width="400" src="images/psth_raster.png">
</p>

### Z-scored population heatmap

<p align="center">
    <img width="400" src="images/heatmap.png">
</p>

### 3D probe placement (Allen CCFv3)

<p align="center">
    <img width="600" src="images/3DBrain.png">
</p>
