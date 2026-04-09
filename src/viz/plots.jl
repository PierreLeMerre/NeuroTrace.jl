"""
    NeuroTrace.Viz

Plotting functions built on Plots.jl with the GR backend.
All functions return a `Plots.Plot` object so the caller can further
customise or re-save it with `savefig`.
"""
module Viz

using Plots
using Statistics
# Import the IO structs by reaching up to the parent module.
# The `..` syntax means "go up two module levels from here".
using ..IO: NWBSession, SpikeUnits, ElectricalSeries

export autoplot, raster, trace, save_figure

# Activate GR explicitly. It is the default backend but being explicit makes
# the choice visible and easy to swap (change `gr()` to `pyplot()`, etc.).
gr()

# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------

"""
    autoplot(nwb::NWBSession) -> Plots.Plot

Inspect `nwb` and delegate to the most appropriate plot:
- If spike units are available, draw a raster plot.
- If a continuous series is available, draw a signal trace.
- If both are available, combine them in a two-panel figure.
"""
function autoplot(nwb::NWBSession)
    has_units  = !isnothing(nwb.units)
    has_series = !isnothing(nwb.series)

    if has_units && has_series
        return _combined_plot(nwb.units, nwb.series)
    elseif has_units
        return raster(nwb.units)
    elseif has_series
        return trace(nwb.series)
    else
        error("NWBSession contains no plottable data (no units, no ElectricalSeries).")
    end
end

# ---------------------------------------------------------------------------
# Spike raster
# ---------------------------------------------------------------------------

"""
    raster(units::SpikeUnits;
           t_start::Float64   = 0.0,
           t_stop::Float64    = Inf,
           tick_height::Float64 = 0.8) -> Plots.Plot

Draw a classic spike raster: each row is a neuron; each vertical tick
marks a spike. Time is on the x-axis, unit index on the y-axis.

# Arguments
- `units`       – A `SpikeUnits` struct from `NeuroTrace.IO`.
- `t_start`     – Show only spikes after this time (seconds).
- `t_stop`      – Show only spikes before this time (seconds).
- `tick_height` – Vertical extent of each tick (fraction of row height).

# Example
```julia
p = NeuroTrace.Viz.raster(nwb.units; t_stop = 10.0)
savefig(p, "raster.png")
```
"""
function raster(units::SpikeUnits;
                t_start::Float64    = 0.0,
                t_stop::Float64     = Inf,
                tick_height::Float64 = 0.8)

    n_units = length(units.ids)
    all_spikes = vcat(units.spike_times...)
    isempty(all_spikes) && error("No spikes found.")

    t_max = isinf(t_stop) ? maximum(all_spikes) : t_stop
    TICK_H = tick_height / 2

    # Build one set of NaN-separated polylines across ALL units in a single
    # pass. Plotting a single series is much faster than one series per unit,
    # because GR re-renders the legend and axes on every plot!() call.
    xs = Float64[]
    ys = Float64[]
    for (row, spikes) in enumerate(units.spike_times)
        in_win = filter(t -> t_start ≤ t ≤ t_max, spikes)
        for t in in_win
            push!(xs, t,         t,         NaN)
            push!(ys, row-TICK_H, row+TICK_H, NaN)
        end
    end

    p = plot(xs, ys;
             seriestype  = :path,
             color       = :black,
             linewidth   = 0.8,
             label       = false,          # suppress legend entry
             xlabel      = "Time (s)",
             ylabel      = "Unit",
             title       = "Spike Raster  ($n_units units)",
             yticks      = (1:n_units, string.(units.ids)),
             xlims       = (t_start, t_max),
             ylims       = (0.5, n_units + 0.5),
             size        = (900, clamp(250 + 22*n_units, 350, 1100)),
             legend      = false)
    return p
end

# ---------------------------------------------------------------------------
# Continuous signal trace
# ---------------------------------------------------------------------------

"""
    trace(series::ElectricalSeries;
          channel::Int       = 1,
          t_start::Float64   = 0.0,
          t_stop::Float64    = Inf,
          n_samples_max::Int = 50_000) -> Plots.Plot

Plot a continuous voltage trace for one channel of an `ElectricalSeries`.

# Arguments
- `series`        – An `ElectricalSeries` struct from `NeuroTrace.IO`.
- `channel`       – 1-based channel index to plot.
- `t_start`       – Start of the displayed time window (seconds).
- `t_stop`        – End of the displayed time window (seconds).
- `n_samples_max` – Down-sample by block-averaging if the window exceeds this
                    many points. Keeps file size and render time manageable.

# Example
```julia
p = NeuroTrace.Viz.trace(nwb.series; channel = 2, t_stop = 5.0)
```
"""
function trace(series::ElectricalSeries;
               channel::Int       = 1,
               t_start::Float64   = 0.0,
               t_stop::Float64    = Inf,
               n_samples_max::Int = 50_000)

    ts   = series.timestamps
    mask = (ts .>= t_start) .& (ts .<= (isinf(t_stop) ? ts[end] : t_stop))
    ts_w = ts[mask]
    y_w  = series.data[mask, min(channel, size(series.data, 2))]

    # Block-average down-sampling: take the mean of every `factor` points.
    # This preserves the amplitude envelope while shrinking the array.
    if length(ts_w) > n_samples_max
        factor = ceil(Int, length(ts_w) / n_samples_max)
        n_bins = length(ts_w) ÷ factor
        ts_w   = [mean(ts_w[(i-1)*factor+1 : i*factor]) for i in 1:n_bins]
        y_w    = [mean(y_w[(i-1)*factor+1  : i*factor]) for i in 1:n_bins]
        @info "Display down-sampled by $(factor)× for performance."
    end

    label_str = "$(series.name)  [ch $channel, $(series.unit)]"
    p = plot(ts_w, y_w;
             color     = :steelblue,
             linewidth = 0.7,
             label     = false,
             xlabel    = "Time (s)",
             ylabel    = series.unit,
             title     = label_str,
             size      = (1000, 300),
             legend    = false)
    return p
end

# ---------------------------------------------------------------------------
# Combined two-panel figure
# ---------------------------------------------------------------------------

function _combined_plot(units::SpikeUnits, series::ElectricalSeries)
    p1 = raster(units)
    p2 = trace(series)
    # plot(p1, p2, layout=...) composes existing plots into a single figure.
    return plot(p1, p2; layout = (2, 1), size = (1000, 700))
end

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

"""
    save_figure(p, path::String)

Save a Plots.jl figure to disk using `savefig`.
The format is inferred from the file extension: `.png`, `.svg`, `.pdf`.
"""
function save_figure(p, path::String)
    savefig(p, path)
    println("Saved: $path")
end

end  # module Viz
