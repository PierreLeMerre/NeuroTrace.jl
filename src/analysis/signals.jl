"""
    NeuroTrace.Analysis

Basic signal-processing utilities for neural data.
This module is intentionally minimal at v0.1 — it is a placeholder
for firing-rate estimation, filtering, and event-triggered averaging.
"""
module Analysis

using Statistics
using ..IO: SpikeUnits, ElectricalSeries

export firing_rate, bin_spikes

"""
    firing_rate(units::SpikeUnits; t_start::Float64 = 0.0, t_stop::Float64) -> Vector{Float64}

Compute the mean firing rate (spikes per second) for each unit over
the interval [t_start, t_stop].

# Example
```julia
rates = NeuroTrace.Analysis.firing_rate(nwb.units; t_stop = 60.0)
```
"""
function firing_rate(units::SpikeUnits;
                     t_start::Float64 = 0.0,
                     t_stop::Float64)::Vector{Float64}
    duration = t_stop - t_start
    duration > 0 || error("t_stop must be greater than t_start")

    return [count(t_start .≤ spikes .≤ t_stop) / duration
            for spikes in units.spike_times]
end

"""
    bin_spikes(units::SpikeUnits; bin_size::Float64 = 0.05,
               t_start::Float64 = 0.0, t_stop::Float64) -> Matrix{Int}, Vector{Float64}

Bin spike times into a (n_units × n_bins) count matrix.
Returns the matrix and the bin-centre time vector.

# Example
```julia
counts, bin_times = NeuroTrace.Analysis.bin_spikes(nwb.units;
                        bin_size = 0.01, t_stop = 10.0)
```
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
            bin = ceil(Int, (t - t_start) / bin_size)
            bin = clamp(bin, 1, n_bins)
            counts[u, bin] += 1
        end
    end

    return counts, bin_times
end

end  # module Analysis
