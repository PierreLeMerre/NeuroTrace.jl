"""
    NeuroTrace

A Julia package for reading, analyzing, and visualizing neuronal data
stored in NWB (Neurodata Without Borders) files.

# Sub-modules
- `NeuroTrace.IO`      – Reading NWB/HDF5 files and extracting typed data structures.
- `NeuroTrace.Viz`     – Plotting functions (spike rasters, signal traces) via Plots.jl/GR.
- `NeuroTrace.Analysis`– Signal-processing utilities (firing rates, filtering, etc.).

# Quick start

```julia
using NeuroTrace

# Inspect an NWB file and auto-plot whatever it contains
NeuroTrace.explore("path/to/data.nwb")

# Or work with the sub-modules directly
nwb = NeuroTrace.IO.load("path/to/data.nwb")
NeuroTrace.Viz.raster(nwb.units)
```
"""
module NeuroTrace

# ---------------------------------------------------------------------------
# Sub-module includes
# Each file defines its own sub-module and is included here. Julia resolves
# them lazily, so the order matters only when there are cross-dependencies.
# ---------------------------------------------------------------------------
include("io/nwb_reader.jl")
include("viz/plots.jl")
include("analysis/signals.jl")

# Re-export the sub-modules so users can write `using NeuroTrace` and then
# call `NeuroTrace.IO.load(...)`, or `using NeuroTrace.IO` for shorter names.
using .IO
using .Viz
using .Analysis

# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

"""
    explore(path::String; output::String = "nwb_plot.png")

High-level entry point. Opens the NWB file at `path`, auto-detects whether it
contains spike-sorted units or a continuous electrical series, and saves an
exploratory figure to `output`.

# Arguments
- `path`   – Absolute or relative path to an `.nwb` file.
- `output` – Destination path for the saved figure (PNG by default).

# Example
```julia
NeuroTrace.explore("session_001.nwb"; output = "raster.png")
```
"""
function explore(path::String; output::String = "nwb_plot.png")
    nwb = IO.load(path)
    fig = Viz.autoplot(nwb)
    Viz.save_figure(fig, output)
    println("Figure saved to: $output")
    return fig
end

end  # module NeuroTrace
