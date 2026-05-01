"""
    NeuroTrace.Analysis

Signal-processing and event-analysis utilities for neural data.
"""
module Analysis

using Statistics
using ImageFiltering: imfilter, Kernel
using ProgressMeter
using ZetaJu
using ..IO: SpikeUnits, ElectricalSeries, UnitInfo, EventInfo

export firing_rate, bin_spikes,
       find_spks_in_window, simple_raster, simple_raster_units,
       simple_PSTH,
       population_psth, population_psth_multi,
       zscore_psth, peak_sort, smooth_psth,
       zeta_pvalues

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
# Population-level analysis
# ============================================================================

"""
    population_psth(spk_times, event_times, bin_size, win_start, win_stop)
        -> (mat::Matrix{Float64}, bin_centers::Vector{Float64})

Compute a PSTH for every unit and stack the results into a
`(n_units × n_bins)` matrix.  Each row is one unit's mean firing rate
(Hz) aligned to `event_times`.

Thin wrapper around `simple_PSTH` — the real work happens there.

# Example
```julia
mat, t = population_psth(spk_sorted, trial_onsets, 0.025, -0.5, 1.5)
```
"""
function population_psth(spk_times::AbstractVector,
                         event_times::AbstractVector{<:Real},
                         bin_size::Real,
                         win_start::Real,
                         win_stop::Real)

    # Compute the first unit to learn n_bins, then pre-allocate
    rate1, bin_centers = simple_PSTH(spk_times[1], event_times,
                                     bin_size, win_start, win_stop)
    n_units = length(spk_times)
    n_bins  = length(bin_centers)
    mat     = zeros(Float64, n_units, n_bins)
    mat[1, :] = rate1

    for u in 2:n_units
        rate, _ = simple_PSTH(spk_times[u], event_times,
                               bin_size, win_start, win_stop)
        mat[u, :] = rate
    end
    return mat, bin_centers
end

"""
    population_psth_multi(units, events, bin_size, win_start, win_stop)
        -> (mat::Matrix{Float64}, bin_centers::Vector{Float64})

Multi-session version of `population_psth`.

Accepts a `Vector{UnitInfo}` and a `Vector{EventInfo}` — one element per
NWB file — matches them by `session_id`, computes a PSTH for each session's
units against that session's events, and stacks all rows into a single
`(total_units × n_bins)` matrix.

Row order is session-then-unit (the natural stacking order).  To sort across
sessions, apply `sortperm` on the merged region labels:

```julia
all_regions  = vcat([u.regions for u in units]...)
region_sort  = sortperm(all_regions)
mat_sorted   = mat[region_sort, :]
```

Works identically for a single file (length-1 vectors) and for a directory
full of sessions.

# Example
```julia
units  = load_units("/data/SC19/")
events = load_events("/data/SC19/", "intervals/trials/start_time")
mat, t = population_psth_multi(units, events, 0.025, -0.5, 1.5)
```
"""
function population_psth_multi(units    ::AbstractVector,
                                events   ::AbstractVector,
                                bin_size ::Real,
                                win_start::Real,
                                win_stop ::Real)

    # Build session_id → event_times lookup for O(1) matching
    event_dict = Dict(e.session_id => e.t_start for e in events)

    rows        = Vector{Vector{Float64}}()
    bin_centers = Float64[]

    for ui in units
        ev = get(event_dict, ui.session_id, nothing)
        isnothing(ev) &&
            error("No matching EventInfo for session: $(ui.session_id). " *
                  "Available sessions: $(keys(event_dict))")

        mat, bc = population_psth(ui.spike_times, ev, bin_size, win_start, win_stop)
        bin_centers = bc
        for u in axes(mat, 1)
            push!(rows, mat[u, :])
        end
    end

    isempty(rows) && error("No units found across all sessions.")
    # Stack: each row-vector becomes a 1×n_bins matrix, vcat gives total×n_bins
    return reduce(vcat, permutedims.(rows)), bin_centers
end

"""
    zscore_psth(mat, bin_centers; baseline_stop = 0.0)
        -> Matrix{Float64}

Z-score each row of a PSTH matrix against its own pre-event baseline.

The baseline is defined as all bins where `bin_centers < baseline_stop`
(default: everything before the event at t = 0).  For each unit:
  - subtract the baseline mean
  - divide by the baseline standard deviation

Units with a baseline std below `eps()` (silent or near-silent cells)
are given std = 1 so they produce all-zero rows rather than NaN.

# Example
```julia
z = zscore_psth(mat, t)               # baseline = all t < 0
z = zscore_psth(mat, t; baseline_stop = -0.1)  # exclude 100 ms before event
```
"""
function zscore_psth(mat::Matrix{Float64},
                     bin_centers::Vector{Float64};
                     baseline_stop::Real = 0.0)

    baseline_mask = bin_centers .< baseline_stop
    any(baseline_mask) ||
        error("No baseline bins found before t = $baseline_stop. " *
              "Check win_start and baseline_stop.")

    bl_mean = mean(mat[:, baseline_mask], dims=2)   # (n_units × 1)
    bl_std  = std(mat[:,  baseline_mask], dims=2)

    # Protect against division by zero for silent cells
    bl_std[bl_std .< eps(Float64)] .= 1.0

    return (mat .- bl_mean) ./ bl_std
end

"""
    peak_sort(z_mat, bin_centers; post_event_start = 0.0) -> Vector{Int}

Return a permutation vector that sorts units by their **peak response time**
in the post-event window (`bin_centers >= post_event_start`).

Applied to the z-scored matrix so peak detection is not biased by
differences in baseline firing rate across units.

The result can be passed directly as a row index:
```julia
idx = peak_sort(z_mat, t)
heatmap(t, 1:n_units, z_mat[idx, :])
```
"""
function peak_sort(z_mat::Matrix{Float64},
                   bin_centers::Vector{Float64};
                   post_event_start::Real = 0.0)

    post_mask = bin_centers .>= post_event_start
    any(post_mask) ||
        error("No post-event bins found at t >= $post_event_start.")

    # argmax returns the *index within the masked slice* for each unit
    peak_bin = [argmax(z_mat[u, post_mask]) for u in axes(z_mat, 1)]
    return sortperm(peak_bin)
end

"""
    smooth_psth(mat; σ = 1.0) -> Matrix{Float64}

Apply a **1-D Gaussian smooth along the time axis** (columns) of a PSTH
matrix, independently for each unit (row).

`σ` is the kernel width in **bins** — so the actual time smoothing depends
on your bin size: σ = 1 bin at 25 ms/bin ≈ 25 ms FWHM.
Increase `σ` for noisier data or coarser smoothing.

The Gaussian kernel is computed by `ImageFiltering.Kernel.gaussian`, which
uses a truncated kernel of width ≈ 6σ and automatically handles edge padding.

# Example
```julia
s_mat = smooth_psth(mat;   σ = 1.5)   # smooth raw Hz matrix
s_z   = smooth_psth(z_mat; σ = 1.5)   # smooth z-scored matrix
```
"""
function smooth_psth(mat::Matrix{Float64}; σ::Real = 1.0)
    ker     = Kernel.gaussian((σ,))           # 1-D kernel, width in bins
    newdata = [Float64.(imfilter(mat[u, :], ker)) for u in axes(mat, 1)]
    # hcat stacks row-vectors as columns → (n_bins × n_units); ' transposes back
    return collect(hcat(newdata...)')
end

# ============================================================================
# ZETA test (ZetaJu integration)
# ============================================================================

"""
    zeta_pvalues(units, events;
                 intResampNum = 100,
                 dblUseMaxDur = nothing)
        -> (pvals::Vector{Float64}, latencies::Vector{Float64})

Run the ZETA responsiveness test (Montijn et al. 2021) on every unit across
all sessions and return flat vectors of p-values and peak latencies.

Units and events are matched by `session_id`.  Any unit whose session has no
matching `EventInfo` is skipped with a warning.  Units for which `zetatest`
throws (too few spikes, degenerate data, etc.) receive `p = 1.0` and
`latency = NaN` so the rest of the pipeline is not interrupted.

# Arguments
- `units`        – `Vector{UnitInfo}` as returned by `load_units` / `filter_units`.
- `events`       – `Vector{EventInfo}` as returned by `load_events`.
- `intResampNum` – number of jitter resamplings (default 100; raise for publication).
- `dblUseMaxDur` – analysis window in seconds (default: min inter-event interval).

# Returns
- `pvals`     – one p-value per unit, same order as the rows of a population matrix.
- `latencies` – IFR peak latency (s re event) per unit; `NaN` when unavailable.
               This is the time of peak instantaneous firing rate (from `dRate["dblLatencyPeak"]`),
               which is more meaningful for heatmap sorting than the ZETA cumulative-deviation
               latency (`dblLatencyZETA`), which can be biased toward the end of the trial
               for tonically responsive units.

# Example
```julia
pvals, lats = zeta_pvalues(units, events; intResampNum = 200)
sig_mask = pvals .< 0.05
```
"""
function zeta_pvalues(units    ::AbstractVector,
                      events   ::AbstractVector;
                      intResampNum::Int     = 100,
                      dblUseMaxDur          = nothing)

    event_dict = Dict(e.session_id => e.t_start for e in events)

    pvals     = Float64[]
    latencies = Float64[]

    n_total = sum(length(ui.spike_times) for ui in units)
    prog = Progress(n_total; desc="ZETA tests: ", showspeed=true, color=:cyan)

    for ui in units
        ev = get(event_dict, ui.session_id, nothing)
        if isnothing(ev)
            @warn "zeta_pvalues: no EventInfo for session $(ui.session_id) — skipping."
            continue
        end

        ev_f64 = Float64.(ev)   # ev is already Vector{Float64} from event_dict

        for spk in ui.spike_times
            spk_f64 = Float64.(spk)
            p, lat = try
                kwargs = (intResampNum = intResampNum, boolReturnRate = true)
                if !isnothing(dblUseMaxDur)
                    kwargs = merge(kwargs, (dblUseMaxDur = Float64(dblUseMaxDur),))
                end
                dblZetaP, _, dRate = zetatest(spk_f64, ev_f64; kwargs...)
                # Use IFR peak latency (instantaneous firing rate peak) rather than
                # ZETA latency (cumulative deviation peak), which can be biased toward
                # the end of the trial for tonically responsive units.
                raw_lat = dRate["dblLatencyPeak"]
                lat_out = isnothing(raw_lat) ? NaN : Float64(raw_lat)
                (Float64(dblZetaP), lat_out)
            catch e
                @warn "zetatest failed for a unit in $(ui.session_id): $e"
                (1.0, NaN)
            end
            push!(pvals,     p)
            push!(latencies, lat)
            next!(prog)
        end
    end

    return pvals, latencies
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
