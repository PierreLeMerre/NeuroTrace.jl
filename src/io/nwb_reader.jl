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
using TOML

export load, read_ragged, NWBSession, SpikeUnits, ElectricalSeries,
       UnitInfo, EventInfo, EventSpec, load_units, load_events, filter_units,
       NTConfig, load_config,
       RegionAtlas, RegionNode, load_atlas, descendants,
       region_display_labels, region_color_map,
       ProbeInfo, load_probes

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
    UnitInfo

Lightweight container for all unit-level data from **one** NWB session.
Returned (one per file) by `load_units`.

# Fields
- `spike_times` – `Vector{Vector{Float64}}`: one inner vector per unit (seconds).
- `regions`     – `Vector{String}`: brain-region label for each unit.
- `session_id`  – `String`: source filename (`basename`), used to match `EventInfo`.
"""
struct UnitInfo
    spike_times :: Vector{Vector{Float64}}
    regions     :: Vector{String}
    depths      :: Vector{Float64}   # DV coordinate per unit (mm, from electrode table)
    session_id  :: String
end

"""
    EventSpec

Specification for one event type, loaded from `config.toml`.

# Fields
- `label` – human-readable name (e.g. `"Trial"`, `"Reward"`).
- `start` – HDF5 path to start timestamps.
- `stop`  – HDF5 path to stop timestamps; empty string if not available.
- `color` – color name or hex string for plotting (e.g. `"black"`, `"#2196F3"`).
"""
struct EventSpec
    label :: String
    start :: String
    stop  :: String
    color :: String
end

"""
    EventInfo

Event timestamps from **one** NWB session for **one** event type.
Returned (one per file per event type) by `load_events`.

# Fields
- `t_start`    – `Vector{Float64}`: event start timestamps (seconds).
- `t_stop`     – `Vector{Float64}`: event stop timestamps; `NaN`-filled when no stop path exists.
- `label`      – `String`: event type label (from `EventSpec`).
- `color`      – `String`: color for plotting (from `EventSpec`).
- `session_id` – `String`: source filename, used to match `UnitInfo`.
"""
struct EventInfo
    t_start    :: Vector{Float64}
    t_stop     :: Vector{Float64}
    label      :: String
    color      :: String
    session_id :: String
end

"""
    ProbeInfo

Electrode-level geometry for one Neuropixels probe, extracted from the
`general/extracellular_ephys/electrodes` table.

# Fields
- `session_id` – source filename (matches `UnitInfo.session_id`).
- `probe_id`   – electrode-group name (e.g. `"probe0"`, `"imec0"`).
- `xyz`        – `(n_channels × 3)` matrix of raw electrode coordinates as
                 stored in the NWB file (x = AP, y = DV, z = ML).
                 Units depend on the recording pipeline — call `load_probes`
                 with the appropriate `coord_scale` to convert to CCF µm.
- `regions`    – per-channel brain-region string (same source as `UnitInfo`).

# Coordinate scaling
Allen CCF meshes use µm. Common NWB conventions and the scale to apply:

| Pipeline              | NWB unit | coord_scale |
|-----------------------|----------|-------------|
| NWB spec (SI)         | meters   | 1e6         |
| ibllib / ONE          | µm       | 1.0         |
| Custom mm pipeline    | mm       | 1e3         |

Run `Brain3D.jl` once and check the printed raw coordinate range to confirm.
"""
struct ProbeInfo
    session_id :: String
    probe_id   :: String
    xyz        :: Matrix{Float64}   # (n_channels × 3)
    regions    :: Vector{String}
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

# ---------------------------------------------------------------------------
# Configuration (loaded from a TOML file)
# ---------------------------------------------------------------------------

"""
    NTConfig

Experiment-wide configuration loaded from a TOML file via `load_config`.
All fields have sensible defaults so you only need to specify what differs
from experiment to experiment.

# Fields
- `data_path`       – path to a single `.nwb` file or a directory of files.
- `events`          – `Vector{EventSpec}`: one entry per event type, defined via
                      `[[events]]` blocks in `config.toml`.
- `win_start`       – peri-event window start (s, relative to event start).
- `win_stop`        – peri-event window stop  (s, relative to event start).
- `psth_bin`        – PSTH bin width (s).
- `baseline_stop`   – z-score baseline end (s); all bins before this are baseline.
- `smooth_sigma`    – Gaussian kernel width for `smooth_psth` (bins).
- `zlim`            – colour saturation for z-scored heatmap panels (±σ).
- `min_firing_rate` – minimum mean firing rate (Hz) to keep a unit; 0 = no filter.
- `regions`         – list of region names to keep; empty = keep all regions.
- `region_colors`   – custom hex colors (`"#RRGGBB"`) for region display; empty = use
                      Allen atlas colors.  Colors are cycled if fewer than the number
                      of regions.
- `save_path`       – directory where figures are saved; empty string = don't save.
- `save_format`     – file format for saved figures: `"png"`, `"pdf"`, or `"svg"`.
"""
struct NTConfig
    data_path       :: String
    events          :: Vector{EventSpec}
    win_start       :: Float64
    win_stop        :: Float64
    psth_bin        :: Float64
    baseline_stop   :: Float64
    smooth_sigma    :: Float64
    zlim            :: Float64
    min_firing_rate :: Float64
    regions         :: Vector{String}
    region_colors   :: Vector{String}
    save_path       :: String
    save_format     :: String
end

"""
    load_config(path) -> NTConfig

Parse a TOML configuration file and return an `NTConfig`.
Missing keys fall back to the defaults shown below.

# Minimal example config.toml
```toml
data_path = "/data/SC19/"

[[events]]
label = "Trial"
start = "intervals/trials/start_time"
stop  = "intervals/trials/stop_time"
color = "black"
```

# Backward-compatible: single `event_path` key is accepted and converted to one EventSpec.

# Full example with multiple event types
```toml
data_path = "/data/SC19/"

[[events]]
label = "Trial"
start = "intervals/trials/start_time"
stop  = "intervals/trials/stop_time"
color = "black"

[[events]]
label = "Reward"
start = "intervals/trials/rw_start"
stop  = "intervals/trials/rw_stop"
color = "#2196F3"
```
"""
function load_config(path::AbstractString)::NTConfig
    isfile(path) || error("Config file not found: $path")
    d = TOML.parsefile(path)

    # Required
    haskey(d, "data_path") || error("config.toml: missing required key 'data_path'")

    # Events: prefer [[events]] array-of-tables; fall back to legacy event_path key
    events = EventSpec[]
    if haskey(d, "events")
        for ev in d["events"]
            push!(events, EventSpec(
                get(ev, "label", "event"),
                get(ev, "start", ""),
                get(ev, "stop",  ""),
                get(ev, "color", "black"),
            ))
        end
    elseif haskey(d, "event_path")
        push!(events, EventSpec("event", d["event_path"], "", "black"))
    end
    isempty(events) &&
        error("config.toml: no events defined. Add one or more [[events]] blocks.")

    fmt = lowercase(strip(get(d, "save_format", "png")))
    fmt in ("png", "pdf", "svg") ||
        error("config.toml: save_format must be one of: png, pdf, svg (got \"$fmt\")")

    return NTConfig(
        d["data_path"],
        events,
        Float64(get(d, "win_start",        -0.5)),
        Float64(get(d, "win_stop",          1.5)),
        Float64(get(d, "psth_bin",          0.025)),
        Float64(get(d, "baseline_stop",     0.0)),
        Float64(get(d, "smooth_sigma",      1.5)),
        Float64(get(d, "zlim",             7.5)),
        Float64(get(d, "min_firing_rate",   0.0)),
        String.(get(d, "regions",           String[])),
        String.(get(d, "region_colors",     String[])),
        strip(get(d, "save_path",           "")),
        fmt,
    )
end

# ---------------------------------------------------------------------------
# Unit filtering
# ---------------------------------------------------------------------------

"""
    filter_units(units; min_firing_rate, regions) -> Vector{UnitInfo}
    filter_units(units, cfg::NTConfig)            -> Vector{UnitInfo}

Filter units in each session by firing rate and/or brain region.

- `min_firing_rate` – drop units whose mean firing rate (spikes ÷ session
  duration) is below this threshold (Hz).  Default `0` = no filter.
- `regions` – keep only units whose region label is in this list.
  An empty list means **keep all regions**.

Session duration is estimated as the span between the earliest and latest
spike across all units in the same session.

# Example
```julia
units = load_units(cfg.data_path)
units = filter_units(units, cfg)   # apply both filters from config
```
"""
function filter_units(units::Vector{UnitInfo};
                      min_firing_rate::Real    = 0.0,
                      regions::Vector{String}  = String[])::Vector{UnitInfo}
    result = map(units) do ui
        n = length(ui.spike_times)
        keep = trues(n)

        # ── firing rate filter ──────────────────────────────────────────────
        if min_firing_rate > 0
            all_spk = vcat(ui.spike_times...)
            dur     = isempty(all_spk) ? 1.0 :
                      max(all_spk[end] - all_spk[1], 1.0)
            for u in 1:n
                keep[u] &= (length(ui.spike_times[u]) / dur) >= min_firing_rate
            end
        end

        # ── region filter ───────────────────────────────────────────────────
        if !isempty(regions)
            for u in 1:n
                keep[u] &= ui.regions[u] ∈ regions
            end
        end

        n_kept = sum(keep)
        n_kept < n &&
            println("  $(ui.session_id): kept $n_kept / $n units after filtering")

        UnitInfo(ui.spike_times[keep], ui.regions[keep], ui.depths[keep], ui.session_id)
    end

    # Drop sessions where every unit was filtered out
    return filter(ui -> !isempty(ui.spike_times), result)
end

# Convenience overload: unpack directly from an NTConfig
filter_units(units::Vector{UnitInfo}, cfg::NTConfig) =
    filter_units(units; min_firing_rate = cfg.min_firing_rate,
                        regions         = cfg.regions)

# ---------------------------------------------------------------------------
# Multi-session loaders (accept a single file path OR a directory)
# ---------------------------------------------------------------------------

"""
    load_units(path) -> Vector{UnitInfo}

Load spike times and electrode-region labels from one NWB file **or** from
every `.nwb` file inside a directory.  Returns one `UnitInfo` per file, in
alphabetical order.

# Example
```julia
# single session
units = load_units("/data/session01.nwb")

# whole experiment folder
units = load_units("/data/SC19/")
```
"""
function load_units(path::AbstractString)::Vector{UnitInfo}
    files = _nwb_files(path)
    println("Loading units from $(length(files)) file(s)…")
    return [_load_units_file(f) for f in files]
end

"""
    load_events(path, event_specs) -> Vector{Vector{EventInfo}}

Load event timestamps for each `EventSpec` from one NWB file **or** from
every `.nwb` file inside a directory.

Returns a `Vector{Vector{EventInfo}}` where:
- outer index = event type (one per `EventSpec`)
- inner index = session file (one per `.nwb` file)

Each `EventInfo` carries `t_start`, `t_stop` (NaN-filled if no stop path),
`label`, `color`, and `session_id`.

# Example
```julia
all_events = load_events("/data/SC19/", cfg.events)
# all_events[1] → trial-start EventInfo for every session
# all_events[2] → reward EventInfo for every session
```
"""
function load_events(path::AbstractString,
                     event_specs::Vector{EventSpec})::Vector{Vector{EventInfo}}
    files = _nwb_files(path)
    return [[_load_events_file(f, spec) for f in files] for spec in event_specs]
end

"""
    load_probes(path) -> Vector{ProbeInfo}

Extract electrode positions and region labels from one NWB file **or** from
every `.nwb` file inside a directory.  Returns one `ProbeInfo` per probe
per file.

Positions are read from `general/extracellular_ephys/electrodes/x`, `y`, `z`
and are returned **as-is** (no scaling).  Pass the result to `add_probes!`
with the matching `coord_scale` keyword to convert to CCF µm.

If x/y/z columns are absent the file is silently skipped (an informational
message is printed), so the function is safe to call on files recorded with
pipelines that don't write coordinates.

# Example
```julia
probes = load_probes(cfg.data_path)
```
"""
function load_probes(path::AbstractString)::Vector{ProbeInfo}
    files = _nwb_files(path)
    result = ProbeInfo[]
    for f in files
        append!(result, _load_probes_file(f))
    end
    return result
end

# ---------- private helpers -------------------------------------------------

"""Return a sorted list of `.nwb` file paths from a file or directory."""
function _nwb_files(path::AbstractString)::Vector{String}
    if isfile(path)
        return [path]
    elseif isdir(path)
        files = filter(readdir(path; join=true)) do f
            b = basename(f)
            endswith(b, ".nwb") && !startswith(b, "._")   # skip macOS resource forks
        end
        isempty(files) && error("No .nwb files found in directory: $path")
        return sort(files)
    else
        error("Path not found: $path")
    end
end

function _load_units_file(filepath::AbstractString)::UnitInfo
    h5open(filepath, "r") do fid
        spk_times  = read_ragged(fid, "units/spike_times",
                                       "units/spike_times_index")
        grp        = fid["general/extracellular_ephys/electrodes"]
        ede_labels = String.(read(grp["location"]))
        main_ch    = Int.(read(fid["units/electrodes"])) .+ 1
        regions    = ede_labels[main_ch]

        # DV depth per unit — try DV column first (mm re bregma), then y (µm or raw)
        depths = if haskey(grp, "DV")
            Float64.(read(grp["DV"]))[main_ch]
        elseif haskey(grp, "y")
            Float64.(read(grp["y"]))[main_ch]
        else
            fill(NaN, length(spk_times))
        end

        println("  $(basename(filepath)): $(length(spk_times)) units")
        return UnitInfo(spk_times, regions, depths, basename(filepath))
    end
end

function _load_events_file(filepath::AbstractString,
                            spec::EventSpec)::EventInfo
    h5open(filepath, "r") do fid
        haskey(fid, spec.start) ||
            error("Event start key '$(spec.start)' not found in $(basename(filepath))")
        t_start  = Float64.(read(fid[spec.start]))
        has_stop = !isempty(spec.stop) && haskey(fid, spec.stop)
        t_stop   = has_stop ? Float64.(read(fid[spec.stop])) :
                              fill(NaN, length(t_start))
        dur_str  = has_stop ?
            "  median dur=$(round(median(t_stop .- t_start); digits=2))s" : ""
        println("  $(basename(filepath)): $(length(t_start)) × '$(spec.label)'$dur_str")
        return EventInfo(t_start, t_stop, spec.label, spec.color, basename(filepath))
    end
end

function _load_probes_file(filepath::AbstractString)::Vector{ProbeInfo}
    h5open(filepath, "r") do fid
        grp = fid["general/extracellular_ephys/electrodes"]

        _col(name) = haskey(grp, name) ? Float64.(read(grp[name])) : nothing

        # ── Coordinates ──────────────────────────────────────────────────────
        # Case 1: AP / DV / ML columns (mm relative to bregma, CCFv3 convention).
        #   CCF µm = (bregma_voxel + coord_mm × ±100) × 10
        #   DV is negated: positive anatomy (dorsal) → positive CCF (ventral).
        #   Bregma in 10 µm CCF voxels: AP=540, DV=65, ML=570
        ap_raw = _col("AP")
        dv_raw = _col("DV")
        ml_raw = _col("ML")

        # Case 2: standard NWB x / y / z columns (units unknown — stored raw).
        x_raw = _col("x")
        y_raw = _col("y")
        z_raw = _col("z")

        local xyz_all::Matrix{Float64}

        if !isnothing(ap_raw) && !isnothing(dv_raw) && !isnothing(ml_raw)
            bregma_ap, bregma_dv, bregma_ml = 540.0, 65.0, 570.0
            x_um = (bregma_ap .+ ap_raw .*  100.0) .* 10.0
            y_um = (bregma_dv .+ dv_raw .* -100.0) .* 10.0
            z_um = (bregma_ml .+ ml_raw .*  100.0) .* 10.0
            xyz_all = hcat(x_um, y_um, z_um)
            println("  $(basename(filepath)): AP/DV/ML (mm) → CCF µm  " *
                    "(AP=$(round(ap_raw[1],digits=3)) → x=$(round(Int,x_um[1])) µm)")

        elseif !isnothing(x_raw) && !isnothing(y_raw) && !isnothing(z_raw)
            xyz_all = hcat(x_raw, y_raw, z_raw)
            println("  $(basename(filepath)): x/y/z columns (raw — units unverified, " *
                    "x=$(round(x_raw[1],digits=4)))")

        else
            avail = join(sort(keys(grp)), ", ")
            error("$(basename(filepath)): no probe coordinate columns found.\n" *
                  "  Expected: AP+DV+ML  or  x+y+z\n" *
                  "  Available electrode columns: $avail")
        end

        # ── Regions ──────────────────────────────────────────────────────────
        regions_all = String.(read(grp["location"]))

        # ── Probe grouping ────────────────────────────────────────────────────
        # Try "group_name" first (SpikeInterface convention), then "group".
        # Fall back to treating the whole file as a single probe.
        probe_ids = if haskey(grp, "group_name")
            String.(read(grp["group_name"]))
        elseif haskey(grp, "group")
            # "group" stores HDF5 object references — read the referenced name
            try
                String.(read(grp["group"]))
            catch
                fill("probe0", size(xyz_all, 1))
            end
        else
            fill("probe0", size(xyz_all, 1))
        end

        # ── Split by probe ────────────────────────────────────────────────────
        probes = ProbeInfo[]
        for pid in unique(probe_ids)
            mask = probe_ids .== pid
            push!(probes, ProbeInfo(
                basename(filepath),
                pid,
                xyz_all[mask, :],
                regions_all[mask],
            ))
            println("    $pid: $(sum(mask)) channels")
        end
        return probes
    end
end

# ---------------------------------------------------------------------------
# Atlas (Allen Brain region tree + display helpers)
# Included here so it lives inside module IO without a separate module block.
# ---------------------------------------------------------------------------
include("atlas.jl")

end  # module IO
