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

export load, read_ragged, NWBSession, SpikeUnits, ElectricalSeries

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
    read_ragged(fid, data_key, index_key) -> Vector{Vector{Float64}}

Read an NWB **ragged (variable-length) array** stored as two flat datasets:

- `data_key`  — the concatenated values (e.g. `"units/spike_times"`).
- `index_key` — cumulative end-indices, one per row (e.g. `"units/spike_times_index"`).

NWB's ragged encoding packs variable-length rows into a single flat array to
avoid HDF5 overhead. `index_key[i]` is the *exclusive* end of row `i`, so
row `i` occupies `data[ index_key[i-1]+1 : index_key[i] ]`, with
`index_key[0] = 0` by convention.

Returns a `Vector{Vector{Float64}}` — one inner vector per row.

# Why a utility?
This pattern appears in every NWB file that stores ragged data:
spike times, spike waveforms, electrode groups, trial-interval tables, etc.
Centralising it here means the fix for the off-by-one is applied once.

# Example
```julia
h5open("session.nwb", "r") do fid
    spk_times = read_ragged(fid, "units/spike_times",
                                  "units/spike_times_index")
end
```
"""
function read_ragged(fid,
                     data_key::String,
                     index_key::String)::Vector{Vector{Float64}}
    flat = Float64.(read(fid[data_key]))
    idx  = Int.(read(fid[index_key]))

    # starts[i] = first element of row i in `flat` (1-based)
    starts = [1; idx[1:end-1] .+ 1]
    return [flat[starts[i]:idx[i]] for i in eachindex(idx)]
end

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

    # Spike times use NWB's ragged-array encoding — delegate to read_ragged.
    if haskey(grp, "spike_times") && haskey(grp, "spike_times_index")
        spike_times = read_ragged(grp, "spike_times", "spike_times_index")
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
