"""
explore_nwb.jl — Standalone NWB file explorer.

This script does NOT require the NeuroTrace package to be installed.
It uses only HDF5.jl and Plots.jl (GR backend) directly, so you can run it
immediately from the Julia REPL or command line while the package is
still under development.

Usage (command line):
    julia --project=.. scripts/explore_nwb.jl path/to/file.nwb

Usage (REPL):
    include("scripts/explore_nwb.jl")
    explore("path/to/file.nwb")

Output:
    nwb_plot.png  – saved in the current working directory
"""

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
# `@__DIR__` is the directory containing this script file.
# By activating the parent project (`..`), we pick up Project.toml
# without needing to install NeuroTrace globally.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()   # download any missing packages (runs only once)

using HDF5
using Plots
using Statistics
gr()   # activate GR backend (default, but explicit is clearer)

# ---------------------------------------------------------------------------
# Utility: print HDF5 tree
# ---------------------------------------------------------------------------

"""
    print_h5_tree(obj, indent = "")

Recursively walk an HDF5 file and print every group and dataset.
Useful for understanding the layout of an unfamiliar NWB file.
"""
function print_h5_tree(obj, indent = "")
    for key in keys(obj)
        child = obj[key]
        if isa(child, HDF5.Group)
            n_children = length(keys(child))
            println(indent * "📁  $(key)/  ($n_children children)")
            print_h5_tree(child, indent * "    ")
        else  # HDF5.Dataset
            sz = size(child)
            println(indent * "📄  $(key)  size=$(sz)")
        end
    end
end

# ---------------------------------------------------------------------------
# NWB reader helpers
# ---------------------------------------------------------------------------

"""
    read_spike_units(grp::HDF5.Group) -> (ids, spike_times)

Extract spike-sorted unit data from the NWB `/units` group.

NWB stores variable-length spike trains using a "ragged array" encoding:
  - `spike_times`       – one flat array of ALL spike timestamps.
  - `spike_times_index` – cumulative end-indices; element i gives the
                          exclusive end of unit i's spikes in the flat array.

For example, if units 1 and 2 have 3 and 5 spikes respectively:
  flat  = [t1, t2, t3, t4, t5, t6, t7, t8]
  index = [3, 8]
"""
function read_spike_units(grp::HDF5.Group)
    ids = haskey(grp, "id") ? Int.(read(grp["id"])) : Int[]

    spike_times = Vector{Vector{Float64}}()
    if haskey(grp, "spike_times") && haskey(grp, "spike_times_index")
        flat  = Float64.(read(grp["spike_times"]))
        idx   = Int.(read(grp["spike_times_index"]))
        starts = vcat(1, idx[1:end-1] .+ 1)
        for (s, e) in zip(starts, idx)
            push!(spike_times, flat[s:e])
        end
    end

    isempty(ids) && (ids = collect(1:length(spike_times)))
    return ids, spike_times
end

"""
    find_electrical_series(acq::HDF5.Group) -> (name, data, timestamps, unit_str) or nothing

Search the `/acquisition` group for the first ElectricalSeries dataset.
Returns `nothing` if none is found.
"""
function find_electrical_series(acq::HDF5.Group)
    for name in keys(acq)
        grp = acq[name]
        grp isa HDF5.Group || continue
        haskey(grp, "data") && haskey(grp, "timestamps") || continue

        raw  = read(grp["data"])
        # NWB stores data as (channels × samples) or (samples,); normalise to
        # (samples × channels) so we index as data[sample, channel].
        data = ndims(raw) == 1 ? reshape(Float64.(raw), :, 1) :
               Float64.(ndims(raw) == 2 ? permutedims(raw) : raw[:, :, 1])
        ts   = Float64.(read(grp["timestamps"]))
        unit_str = haskey(attributes(grp), "unit") ?
                   read(attributes(grp)["unit"]) : "a.u."

        return name, data, ts, unit_str
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

"""
    plot_raster(ids, spike_times; t_window = nothing) -> Plots.Plot

Draw a spike raster plot. Each row is a neuron; each tick is a spike.

# Arguments
- `ids`         – Vector of unit IDs (used as y-axis labels).
- `spike_times` – Vector of spike-time vectors (one per unit).
- `t_window`    – Optional `(t_start, t_stop)` tuple; defaults to full range.
"""
function plot_raster(ids, spike_times; t_window = nothing)
    all_spikes = vcat(spike_times...)
    isempty(all_spikes) && error("No spikes found to plot.")

    t_start, t_stop = isnothing(t_window) ?
                      (minimum(all_spikes), maximum(all_spikes)) : t_window
    n_units = length(ids)

    # Build one NaN-separated polyline for ALL spikes across ALL units.
    # A single series renders far faster than one series per unit.
    TICK_H = 0.36
    xs = Float64[]
    ys = Float64[]
    for (row, (id, spikes)) in enumerate(zip(ids, spike_times))
        in_win = filter(t -> t_start ≤ t ≤ t_stop, spikes)
        for t in in_win
            push!(xs, t,          t,          NaN)
            push!(ys, row-TICK_H, row+TICK_H, NaN)
        end
    end

    p = plot(xs, ys;
             seriestype = :path,
             color      = :black,
             linewidth  = 0.8,
             label      = false,
             xlabel     = "Time (s)",
             ylabel     = "Unit",
             title      = "Spike Raster  ($n_units units, " *
                          "$(round(t_stop - t_start; digits=1)) s)",
             yticks     = (1:n_units, string.(ids)),
             xlims      = (t_start, t_stop),
             ylims      = (0.5, n_units + 0.5),
             size       = (900, clamp(250 + 22*n_units, 350, 1100)),
             legend     = false)
    return p
end

"""
    plot_trace(name, data, timestamps, unit_str;
               channel = 1, t_window = nothing,
               max_points = 60_000) -> Plots.Plot

Plot a continuous voltage trace. Down-samples if necessary to keep the
output file small and rendering fast (no visual loss at typical zoom levels).
"""
function plot_trace(name, data, timestamps, unit_str;
                    channel    = 1,
                    t_window   = nothing,
                    max_points = 60_000)

    t_start, t_stop = isnothing(t_window) ?
                      (timestamps[1], timestamps[end]) : t_window
    mask = (timestamps .>= t_start) .& (timestamps .<= t_stop)
    ts_w = timestamps[mask]
    y_w  = data[mask, min(channel, size(data, 2))]

    # Down-sample by averaging adjacent blocks
    if length(ts_w) > max_points
        factor  = ceil(Int, length(ts_w) / max_points)
        n_bins  = length(ts_w) ÷ factor
        ts_w    = [mean(ts_w[(i-1)*factor+1 : i*factor]) for i in 1:n_bins]
        y_w     = [mean(y_w[(i-1)*factor+1  : i*factor]) for i in 1:n_bins]
        @info "Display down-sampled by $(factor)× for performance."
    end

    duration = round(t_stop - t_start; digits = 2)
    p = plot(ts_w, y_w;
             color     = :steelblue,
             linewidth = 0.7,
             label     = false,
             xlabel    = "Time (s)",
             ylabel    = unit_str,
             title     = "$(name)  [ch $(channel)]  — $(duration) s",
             size      = (1000, 300),
             legend    = false)
    return p
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    explore(path::String;
            output::String     = "nwb_plot.png",
            t_window           = nothing,
            channel::Int       = 1)

Open an NWB file, auto-detect its contents, produce a plot, and save it.

# Arguments
- `path`     – Path to the `.nwb` file.
- `output`   – Output image path (extension sets format: .png / .svg / .pdf).
- `t_window` – `(t_start, t_stop)` to zoom in; `nothing` plots everything.
- `channel`  – Which electrode channel to display for continuous data.

# Example
```julia
explore("sub-01_ses-001_ecephys.nwb"; t_window = (0.0, 10.0))
```
"""
function explore(path::String;
                 output::String  = "nwb_plot.png",
                 t_window        = nothing,
                 channel::Int    = 1)

    isfile(path) || error("File not found: $path")

    fig = h5open(path, "r") do fid
        println("\n═══════════════════════════════════════")
        println("  NWB file: $(basename(path))")
        println("═══════════════════════════════════════")

        # Print session metadata from root-level attributes
        attrs = keys(attributes(fid))
        if !isempty(attrs)
            println("\n── Session metadata ──")
            for a in attrs
                println("   $(a) = $(read(attributes(fid)[a]))")
            end
        end

        println("\n── File structure ──")
        print_h5_tree(fid)

        # --- Decide what to plot -------------------------------------------
        has_units  = haskey(fid, "units")
        has_series = haskey(fid, "acquisition")

        if has_units
            println("\n── Reading spike units ─────────────────")
            ids, spikes = read_spike_units(fid["units"])
            println("   $(length(ids)) units, " *
                    "$(sum(length.(spikes))) total spikes")
            if !isempty(spikes) && !isempty(vcat(spikes...))
                t_range = extrema(vcat(spikes...))
                println("   Time range: $(round.(t_range; digits=3)) s")
            end
            return plot_raster(ids, spikes; t_window)

        elseif has_series
            println("\n── Reading acquisition ─────────────────")
            result = find_electrical_series(fid["acquisition"])
            if isnothing(result)
                error("Found /acquisition but no ElectricalSeries with 'data' + 'timestamps'.")
            end
            name, data, ts, unit_str = result
            println("   Series: $(name)")
            println("   Shape: $(size(data))  (samples × channels)")
            println("   Duration: $(round(ts[end] - ts[1]; digits=2)) s  " *
                    "at $(round(length(ts)/(ts[end]-ts[1]); digits=1)) Hz")
            return plot_trace(name, data, ts, unit_str;
                              channel, t_window)

        else
            @warn "No recognized data groups found. Showing file structure only."
            return nothing
        end
    end  # h5open closes here

    if !isnothing(fig)
        savefig(fig, output)
        println("\n✓  Figure saved → $(abspath(output))")
    end

    return fig
end

# ---------------------------------------------------------------------------
# Run if called from the command line
# ---------------------------------------------------------------------------
# `abspath(PROGRAM_FILE)` matches this script's path when run via `julia script.jl`,
# but is empty when the file is `include()`-d from the REPL.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("""
        Usage:
            julia --project=.. scripts/explore_nwb.jl  <path_to_file.nwb>  [output.png]

        Options:
            Second argument sets the output filename (default: nwb_plot.png).
        """)
    else
        nwb_path = ARGS[1]
        out_path = length(ARGS) >= 2 ? ARGS[2] : "nwb_plot.png"
        explore(nwb_path; output = out_path)
    end
end
