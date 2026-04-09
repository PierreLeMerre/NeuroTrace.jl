"""
    NeuroTrace.IO

Functions for opening NWB files and extracting structured data.

NWB files are HDF5 archives with a standardised internal layout.
We use HDF5.jl to traverse the hierarchy and return plain Julia structs,
keeping I/O concerns cleanly separated from analysis and plotting.
"""
module IO

using HDF5
using Statistics

export load, NWBSession, SpikeUnits, ElectricalSeries

# ---------------------------------------------------------------------------
# Data types
# Each struct mirrors one NWB "neurodata type". Using plain structs (rather
# than mutable ones) encourages immutable, functional-style processing.
# ---------------------------------------------------------------------------

"""
    SpikeUnits

Holds spike-sorted unit data extracted from the `/units` NWB group.

# Fields
- `ids`         – Integer unit IDs (one per neuron).
- `spike_times` – Vector of vectors; `spike_times[i]` contains all spike
                  timestamps (in seconds) for unit `i`.
"""
struct SpikeUnits
    ids         :: Vector{Int}
    spike_times :: Vector{Vector{Float64}}
end

"""
    ElectricalSeries

Holds a continuous voltage trace from the `/acquisition` NWB group.

# Fields
- `name`        – Dataset name within the NWB file.
- `data`        – 2-D array, shape (n_samples × n_channels).
- `timestamps`  – Time axis in seconds (length n_samples).
- `unit`        – Physical unit string (e.g. `"volts"`, `"microvolts"`).
- `channel_ids` – Optional electrode IDs.
"""
struct ElectricalSeries
    name        :: String
    data        :: Matrix{Float64}
    timestamps  :: Vector{Float64}
    unit        :: String
    channel_ids :: Vector{Int}
end

"""
    NWBSession

Top-level container returned by `load`. Holds all data extracted from a
single NWB file. Fields not found in the file are `nothing`.

# Fields
- `path`    – Source file path.
- `units`   – `SpikeUnits` if `/units` group is present, otherwise `nothing`.
- `series`  – `ElectricalSeries` if continuous data is found, otherwise `nothing`.
- `metadata`– Dict of session-level attributes (e.g. subject, lab, date).
"""
struct NWBSession
    path     :: String
    units    :: Union{SpikeUnits, Nothing}
    series   :: Union{ElectricalSeries, Nothing}
    metadata :: Dict{String, Any}
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    load(path::String) -> NWBSession

Open the NWB file at `path` and extract all supported data types.
Prints a short summary of what was found.

# Example
```julia
nwb = NeuroTrace.IO.load("subject01_session001.nwb")
```
"""
function load(path::String)::NWBSession
    isfile(path) || error("File not found: $path")

    units    = nothing
    series   = nothing
    metadata = Dict{String, Any}()

    h5open(path, "r") do fid
        println("── Opening: $path")
        println("   Top-level keys: ", join(keys(fid), ", "))

        # --- Session metadata from root attributes --------------------------
        for attr in keys(attributes(fid))
            metadata[attr] = read(attributes(fid)[attr])
        end

        # --- Spike units ----------------------------------------------------
        if haskey(fid, "units")
            units = _read_units(fid["units"])
            println("   ✓ Found spike units: $(length(units.ids)) neurons")
        end

        # --- Continuous acquisition -----------------------------------------
        if haskey(fid, "acquisition")
            series = _read_first_electrical_series(fid["acquisition"])
            if !isnothing(series)
                n_s, n_c = size(series.data)
                println("   ✓ Found ElectricalSeries '$(series.name)': " *
                        "$n_s samples × $n_c channels")
            end
        end
    end

    return NWBSession(path, units, series, metadata)
end

# ---------------------------------------------------------------------------
# Private helpers  (underscore prefix = internal convention in Julia)
# ---------------------------------------------------------------------------

"""
    _read_units(grp::HDF5.Group) -> SpikeUnits

Parse the NWB `/units` compound dataset. NWB stores spike times in a
ragged (variable-length) array encoded as a flat vector + an index vector.
"""
function _read_units(grp::HDF5.Group)::SpikeUnits
    ids = Int[]
    spike_times = Vector{Vector{Float64}}()

    # NWB unit IDs
    if haskey(grp, "id")
        ids = Int.(read(grp["id"]))
    end

    # Spike times are stored as:
    #   spike_times          – flat concatenated array of all timestamps
    #   spike_times_index    – cumulative end-indices per unit
    if haskey(grp, "spike_times") && haskey(grp, "spike_times_index")
        flat   = Float64.(read(grp["spike_times"]))
        idx    = Int.(read(grp["spike_times_index"]))

        starts = vcat(1, idx[1:end-1] .+ 1)
        for (s, e) in zip(starts, idx)
            push!(spike_times, flat[s:e])
        end
    end

    # Ensure ids and spike_times are aligned
    if isempty(ids)
        ids = collect(1:length(spike_times))
    end

    return SpikeUnits(ids, spike_times)
end

"""
    _read_first_electrical_series(acq::HDF5.Group) -> Union{ElectricalSeries, Nothing}

Look through the `/acquisition` group for the first dataset whose NWB type
attribute is `"ElectricalSeries"` and return it.
"""
function _read_first_electrical_series(acq::HDF5.Group)
    for name in keys(acq)
        grp = acq[name]
        grp isa HDF5.Group || continue

        # NWB types are tagged with a "neurodata_type" attribute
        ntype = get(attributes(grp), "neurodata_type", nothing)
        if isnothing(ntype) || read(ntype) != "ElectricalSeries"
            # Fallback: accept any group that has "data" and "timestamps"
            haskey(grp, "data") && haskey(grp, "timestamps") || continue
        end

        raw  = read(grp["data"])
        # NWB data can be (samples,) for single channel or (samples, channels)
        data = ndims(raw) == 1 ? reshape(Float64.(raw), :, 1) : Float64.(permutedims(raw))

        ts   = Float64.(read(grp["timestamps"]))
        unit = haskey(attributes(grp), "unit") ? read(attributes(grp)["unit"]) : "unknown"

        ch_ids = haskey(grp, "electrodes") ? Int.(read(grp["electrodes"])) :
                 collect(1:size(data, 2))

        return ElectricalSeries(name, data, ts, unit, ch_ids)
    end
    return nothing
end

end  # module IO
