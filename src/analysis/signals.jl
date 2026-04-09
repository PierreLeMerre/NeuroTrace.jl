"""
    NeuroTrace.Analysis

Signal-processing and event-analysis utilities for neural data.
"""
module Analysis

using Statistics
using ..IO: SpikeUnits, ElectricalSeries

export firing_rate, bin_spikes,
       find_spks_in_window, simple_raster, simple_raster_units,
       simple_PSTH

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _histcount(data, edges) -> Vector{Int}

Fast integer histogram without external dependencies.
Uses binary search (O(n log m)) so it is efficient even for large arrays.
`edges` must be sorted; spikes outside [edges[1], edges[end]) are ignored.
"""
function _histcount(data::AbstractVector{<:Real},
                    edges::AbstractVector{<:Real})
    counts = zeros(Int, length(edges) - 1)
    for x in data
        i = searchsortedlast(edges, x)
        if 1 ≤ i < length(edges)
            @inbounds counts[i] += 1
        end
    end
    return counts
end

# ============================================================================
# Window finding
# ============================================================================

"""
    find_spks_in_window(spk_times, event_time, start, stop) -> Vector{Float64}

Return spike times **centred around `event_time`** that fall in the open
interval `(start, stop)` seconds relative to the event.

This is the core primitive used by `simple_raster` and `simple_PSTH`.

# Improvement over the original
A single boolean mask `(centered .> start) .& (centered .< stop)` replaces
the `findall` + `intersect` pattern, avoiding an O(n log n) sort and a
temporary index array.

# Example
```julia
# spikes of unit 3 in a [−0.5, 1.0] s window around each event
for ev in event_times
    ts = find_spks_in_window(spk_times[3], ev, -0.5, 1.0)
end
```
"""
function find_spks_in_window(spk_times::AbstractVector{<:Real},
                              event_time::Real,
                              start::Real,
                              stop::Real)
    centered = spk_times .- event_time
    return centered[(centered .> start) .& (centered .< stop)]
end

# ============================================================================
# Raster helpers
# ============================================================================

"""
    simple_raster_units(spk_times, event_time, start, stop) -> (X, Y)

Build scatter coordinates for a **multi-unit raster around a single event**.

Each element of `spk_times` is one unit's spike-time vector.  Returns `X`
(centred spike times) and `Y` (unit-row indices, 1-based) as typed
`Float64` / `Int` vectors — ready to pass directly to `scatter(X, Y)`.

# Improvement over the original
- `X` and `Y` are pre-typed (`Float64[]`, `Int[]`) instead of `Any[]`,
  avoiding costly type-inference on every `append!` call.
- Uses `find_spks_in_window` (boolean mask) instead of `findall + intersect`.
- `fill(u, n)` replaces `ones(n) .* u` (no floating-point allocation).

# Example
```julia
X, Y = simple_raster_units(spk_times, 0.0, 0.0, 600.0)
scatter(X, Y; ms=1, mc=:black, markerstrokewidth=0)
```
"""
function simple_raster_units(spk_times::AbstractVector,
                              event_time::Real,
                              start::Real,
                              stop::Real)
    X = Float64[]
    Y = Int[]
    for (u, unit_spks) in enumerate(spk_times)
        ts = find_spks_in_window(unit_spks, event_time, start, stop)
        isempty(ts) && continue
        append!(X, ts)
        append!(Y, fill(u, length(ts)))
    end
    return X, Y
end

"""
    simple_raster(spk_times, event_times, start, stop) -> (X, Y)

Build scatter coordinates for a **single-unit raster across multiple trials**.

`spk_times` is a single unit's spike-time vector; `event_times` is the vector
of trial-onset times.  Returns centred spike times `X` and trial indices `Y`.

# Example
```julia
X, Y = simple_raster(spk_times[unit_id], trial_onsets, -0.5, 1.5)
scatter(X, Y; ms=1, mc=:black, markerstrokewidth=0,
        xlabel="time re event (s)", ylabel="trial")
```
"""
function simple_raster(spk_times::AbstractVector{<:Real},
                       event_times::AbstractVector{<:Real},
                       start::Real,
                       stop::Real)
    X = Float64[]
    Y = Int[]
    for (t, ev) in enumerate(event_times)
        ts = find_spks_in_window(spk_times, ev, start, stop)
        isempty(ts) && continue
        append!(X, ts)
        append!(Y, fill(t, length(ts)))
    end
    return X, Y
end

# ============================================================================
# PSTH
# ============================================================================

"""
    simple_PSTH(spk_times, event_times, bin_size, start, stop)
        -> (rate::Vector{Float64}, bin_centers::Vector{Float64})

Compute an **event-aligned peri-stimulus time histogram** (PSTH).

Returns the mean firing rate (Hz) in each bin and the corresponding bin-centre
times — so the caller does not need to reconstruct the time axis separately.

# Arguments
- `spk_times`   – spike times for **one unit** (seconds, absolute).
- `event_times` – onset times of the events / stimuli (seconds, absolute).
- `bin_size`    – histogram bin width (seconds).
- `start`       – window start relative to each event (e.g. `-0.5` for 500 ms before).
- `stop`        – window stop  relative to each event (e.g. `1.0`  for 1 s after).

# Returns
- `rate`        – firing rate in Hz, length `round(Int, (stop-start)/bin_size)`.
- `bin_centers` – time axis (seconds re event) matching `rate`.

# Improvement over the original
The original loops over events and calls `histcounts` once per trial, then
sums the results.  This version collects **all** centred spike times in a
single pass and calls the histogram function only once — reducing allocations
and `N_events` histogram calls to one.

It also returns `bin_centers`, avoiding the mismatch that occurred when the
caller created `timevec2` with a different length/offset.

# Example
```julia
rate, t = simple_PSTH(spk_times[5], trial_onsets, 0.01, -0.5, 1.5)
plot(t, rate; xlabel="time re event (s)", ylabel="spikes/s")
```
"""
function simple_PSTH(spk_times::AbstractVector{<:Real},
                     event_times::AbstractVector{<:Real},
                     bin_size::Real,
                     start::Real,
                     stop::Real)

    edges = collect(range(start, stop; step=bin_size))
    # Ensure the last edge exactly reaches stop (floating-point range may fall short)
    edges[end] < stop && push!(edges, stop)

    # --- Single-pass spike collection ---
    # Collect ALL centred spike times from every trial at once.
    # Pre-allocating with sizehint! avoids repeated array resizing.
    all_centered = Float64[]
    sizehint!(all_centered, length(spk_times) * length(event_times) ÷ max(1,
              round(Int, (spk_times[end] - spk_times[1]) / (stop - start))))
    for ev in event_times
        centered = spk_times .- ev
        append!(all_centered, centered[(centered .> start) .& (centered .≤ stop)])
    end

    # One histogram call for all trials
    counts      = _histcount(all_centered, edges)
    rate        = Float64.(counts) ./ length(event_times) ./ bin_size
    bin_centers = edges[1:end-1] .+ bin_size / 2

    return rate, bin_centers
end

# ============================================================================
# Lower-level utilities (kept from v0.1)
# ============================================================================

"""
    firing_rate(units::SpikeUnits; t_start, t_stop) -> Vector{Float64}

Mean firing rate (Hz) for each unit in [t_start, t_stop].
"""
function firing_rate(units::SpikeUnits;
                     t_start::Float64 = 0.0,
                     t_stop::Float64)::Vector{Float64}
    duration = t_stop - t_start
    duration > 0 || error("t_stop must be greater than t_start")
    return [count(t_start .≤ s .≤ t_stop) / duration for s in units.spike_times]
end

"""
    bin_spikes(units::SpikeUnits; bin_size, t_start, t_stop)
        -> (Matrix{Int}, Vector{Float64})

Bin spike times into a `(n_units × n_bins)` count matrix plus bin-centre times.
"""
function bin_spikes(units::SpikeUnits;
                    bin_size::Float64 = 0.05,
                    t_start::Float64  = 0.0,
                    t_stop::Float64)

    edges     = t_start : bin_size : t_stop
    n_bins    = length(edges) - 1
    n_units   = length(units.ids)
    counts    = zeros(Int, n_units, n_bins)
    bin_times = [t_start + (i - 0.5) * bin_size for i in 1:n_bins]

    for (u, spikes) in enumerate(units.spike_times)
        for t in spikes
            t < t_start || t > t_stop && continue
            b = clamp(ceil(Int, (t - t_start) / bin_size), 1, n_bins)
            counts[u, b] += 1
        end
    end
    return counts, bin_times
end

end  # module Analysis
