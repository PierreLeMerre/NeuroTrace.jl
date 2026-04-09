"""
    NeuroTrace.Viz

Plotting functions built on CairoMakie. All functions return a `Figure`
object so the caller can further customise or re-save it.

CairoMakie is chosen over GLMakie/WGLMakie because it renders entirely
off-screen (no window required) and exports crisp vector graphics —
ideal for scripts running on servers or in CI pipelines.
"""
module Viz

using CairoMakie
# Import the IO structs by reaching up to the parent module.
# The `..` syntax means "go up two module levels from here".
using ..IO: NWBSession, SpikeUnits, ElectricalSeries

export autoplot, raster, trace, save_figure

# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------

"""
    autoplot(nwb::NWBSession) -> Figure

Inspect `nwb` and delegate to the most appropriate plot:
- If spike units are available, draw a raster plot.
- If a continuous series is available, draw a signal trace.
- If both are available, combine them in a two-panel figure.
"""
function autoplot(nwb::NWBSession)::Figure
    has_units  = !isnothing(nwb.units)
    has_series = !isnothing(nwb.series)

    if has_units && has_series
        return _combined_figure(nwb.units, nwb.series)
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
           t_start::Float64 = 0.0,
           t_stop::Float64  = Inf,
           tick_height::Float64 = 0.8) -> Figure

Draw a classic spike raster: each row is a neuron; each vertical tick
marks a spike. Time is on the x-axis, unit ID on the y-axis.

# Arguments
- `units`       – A `SpikeUnits` struct from `NeuroTrace.IO`.
- `t_start`     – Clip the plot to spikes after this time (seconds).
- `t_stop`      – Clip the plot to spikes before this time (seconds).
- `tick_height` – Vertical extent of each tick (fraction of row height).

# Example
```julia
fig = NeuroTrace.Viz.raster(nwb.units; t_stop = 10.0)
NeuroTrace.Viz.save_figure(fig, "raster.png")
```
"""
function raster(units::SpikeUnits;
                t_start::Float64 = 0.0,
                t_stop::Float64  = Inf,
                tick_height::Float64 = 0.8)::Figure

    n_units = length(units.ids)
    fig = Figure(size = (900, 400 + 20 * n_units))
    ax  = Axis(fig[1, 1];
               xlabel = "Time (s)",
               ylabel = "Unit ID",
               title  = "Spike Raster  ($n_units units)",
               yticks = (1:n_units, string.(units.ids)),
               yreversed = false)

    for (row, (id, spikes)) in enumerate(zip(units.ids, units.spike_times))
        # Filter by time window
        in_window = filter(t -> t_start ≤ t ≤ t_stop, spikes)
        isempty(in_window) && continue

        # Each spike becomes a vertical line segment
        xs = repeat(in_window; inner = 3)        # x: spike, spike, NaN
        ys = repeat([row - tick_height/2,
                     row + tick_height/2,
                     NaN]; outer = length(in_window))
        lines!(ax, xs, ys; color = (:black, 0.8), linewidth = 0.8)
    end

    # Clean up axes
    t_max = isinf(t_stop) ?
        maximum(maximum(s) for s in units.spike_times if !isempty(s)) :
        t_stop
    xlims!(ax, t_start, t_max)
    ylims!(ax, 0.5, n_units + 0.5)

    return fig
end

# ---------------------------------------------------------------------------
# Continuous signal trace
# ---------------------------------------------------------------------------

"""
    trace(series::ElectricalSeries;
          channel::Int    = 1,
          t_start::Float64 = 0.0,
          t_stop::Float64  = Inf,
          n_samples_max::Int = 50_000) -> Figure

Plot a continuous voltage trace for one channel of an `ElectricalSeries`.

# Arguments
- `series`        – An `ElectricalSeries` struct from `NeuroTrace.IO`.
- `channel`       – 1-based channel index to plot.
- `t_start`       – Start of the displayed time window (seconds).
- `t_stop`        – End of the displayed time window (seconds).
- `n_samples_max` – Down-sample (by averaging) if more samples are requested.
                    This keeps the SVG/PNG file small and rendering fast.

# Example
```julia
fig = NeuroTrace.Viz.trace(nwb.series; channel = 2, t_stop = 5.0)
```
"""
function trace(series::ElectricalSeries;
               channel::Int     = 1,
               t_start::Float64 = 0.0,
               t_stop::Float64  = Inf,
               n_samples_max::Int = 50_000)::Figure

    # Time-window mask
    ts   = series.timestamps
    mask = (ts .>= t_start) .& (ts .<= t_stop)
    ts_w = ts[mask]
    y_w  = series.data[mask, channel]

    # Optional down-sampling: if the window contains more points than
    # n_samples_max, bin and take the mean. This preserves the envelope
    # of the signal without plotting millions of invisible points.
    if length(ts_w) > n_samples_max
        factor  = ceil(Int, length(ts_w) / n_samples_max)
        n_bins  = length(ts_w) ÷ factor
        ts_w    = [mean(ts_w[(i-1)*factor+1 : i*factor]) for i in 1:n_bins]
        y_w     = [mean(y_w[(i-1)*factor+1  : i*factor]) for i in 1:n_bins]
        @info "Down-sampled trace by factor $factor for display."
    end

    label = "$(series.name)  [ch $(channel), $(series.unit)]"
    fig = Figure(size = (1000, 300))
    ax  = Axis(fig[1, 1];
               xlabel = "Time (s)",
               ylabel = series.unit,
               title  = label)

    lines!(ax, ts_w, y_w; color = :steelblue, linewidth = 0.6)

    return fig
end

# ---------------------------------------------------------------------------
# Combined two-panel figure
# ---------------------------------------------------------------------------

function _combined_figure(units::SpikeUnits, series::ElectricalSeries)::Figure
    fig_r = raster(units)
    fig_t = trace(series)

    # Merge both axes into a new figure with two rows
    fig = Figure(size = (1000, 700))
    ax1 = Axis(fig[1, 1]; title = "Spike Raster",
               xlabel = "Time (s)", ylabel = "Unit ID")
    ax2 = Axis(fig[2, 1]; title = "Continuous Signal",
               xlabel = "Time (s)", ylabel = series.unit)

    # Re-draw into the combined figure
    for (row, (id, spikes)) in enumerate(zip(units.ids, units.spike_times))
        isempty(spikes) && continue
        xs = repeat(spikes; inner = 3)
        ys = repeat([row - 0.4, row + 0.4, NaN]; outer = length(spikes))
        lines!(ax1, xs, ys; color = (:black, 0.7), linewidth = 0.8)
    end
    ylims!(ax1, 0.5, length(units.ids) + 0.5)

    lines!(ax2, series.timestamps, series.data[:, 1];
           color = :steelblue, linewidth = 0.6)

    return fig
end

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

"""
    save_figure(fig::Figure, path::String)

Save a Makie figure to disk. The file format is inferred from the extension:
`.png`, `.svg`, and `.pdf` are all supported by CairoMakie.
"""
function save_figure(fig::Figure, path::String)
    CairoMakie.save(path, fig; px_per_unit = 2)   # 2× for retina-quality PNG
    println("Saved: $path")
end

end  # module Viz
