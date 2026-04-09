# ============================================================================
#  Explore_data.jl  —  NeuroTrace session explorer
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
#  Each `# %%` comment marks a VSCode cell boundary.
# ============================================================================

# %% ── Configuration (edit before running) ──────────────────────────────────

NWB_FILE   = "/Volumes/T7/NWB_Alicante/NWB/SC19_20250529.nwb"   # ← set this

TIME_IN    = 000.0        # session start to display (s)
TIME_OUT   = 1400.0      # session end   to display (s)
BIN_SZ     = 0.10       # population firing-rate bin width (s)

# Events to overlay on the raster.  Add as many as you need.
# Each entry is a NamedTuple: (label, path, color).
# Paths that don't exist in the file are silently skipped.
EVENTS = [
    (label="Reward delivery",    path="intervals/trials/rw_start", color=:blue),
    (label="Trial start",  path="intervals/trials/start_time",color=:black),
    # (label="Lever press", path="stimulus/presentation/LeverPress/timestamps", color=:blue),
]

# One colour per brain region (cycles if there are more regions than colours).
REGION_COLORS = [
    "#5A8DAF",   # steel blue
    "#CE6161",   # red
    "#4B6A2E",   # dark green
    "#e5a106",   # amber
    "#9F453B",   # brick
    "#92A691",   # sage
    "#7B5EA7",   # purple
    "#afd576",   # light green
]

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HDF5
using Plots
using Statistics
gr()

# Load the Analysis utilities from the NeuroTrace package
using NeuroTrace.IO:       read_ragged
using NeuroTrace.Analysis: find_spks_in_window, simple_raster_units,
                           simple_raster, simple_PSTH, _histcount

# %% ── Open NWB file ─────────────────────────────────────────────────────────

nwb = h5open(NWB_FILE, "r")
println("Opened  : ", basename(NWB_FILE))
println("Keys    : ", join(keys(nwb), ", "))

# %% ── Load spike times (NWB ragged / jagged array) ─────────────────────────
#
# NWB stores variable-length spike trains as TWO flat arrays:
#
#   units/spike_times        – all spike timestamps concatenated
#   units/spike_times_index  – spike_times_index[i] is the EXCLUSIVE end
#                              of unit i's data in the flat array.
#
# So unit i's spikes live at:
#   spike_times[ spike_times_index[i-1]+1 : spike_times_index[i] ]
# with the convention that spike_times_index[0] = 0.
#
# This replaces the original pushfirst!/manual-offset approach which
# accidentally skipped the first spike of unit 1.

unit_ids  = Int.(read(nwb["units/id"]))
spk_times = read_ragged(nwb, "units/spike_times", "units/spike_times_index")
n_units   = length(spk_times)

all_spk   = vcat(spk_times...)
println("Units   : $n_units")
println("Total spikes: $(length(all_spk))")
println("Session duration: $(round(all_spk[end]; digits=1)) s")

# %% ── Electrode regions ─────────────────────────────────────────────────────

ede_labels = String.(read(nwb["general/extracellular_ephys/electrodes/location"]))
main_ch    = Int.(read(nwb["units/peak_channel_id"])) .+ 1   # 0-based → 1-based
ede_region = ede_labels[main_ch]

# Sort units by brain region so same-region units appear together in the raster
sort_idx    = sortperm(ede_region)
spk_sorted  = spk_times[sort_idx]
ylab        = ede_region[sort_idx]          # region label for each row
pfc_regions = unique(ylab)                  # ordered list of unique regions

println("Regions : ", join(pfc_regions, ", "))

# Map each region to a colour (cycle through REGION_COLORS if needed)
color_map = Dict(r => REGION_COLORS[mod1(i, length(REGION_COLORS))]
                 for (i, r) in enumerate(pfc_regions))

# Row index boundaries between regions (for separator lines)
region_sizes = [count(==(r), ylab) for r in pfc_regions]
boundaries   = cumsum(region_sizes)[1:end-1]   # boundary AFTER each region

# %% ── Event timestamps ──────────────────────────────────────────────────────
#
# Load each entry in EVENTS, skipping any path that doesn't exist in the file.
# `loaded_events` is a Vector of NamedTuples with the same fields as EVENTS
# plus a `times` field containing the actual timestamps.

loaded_events = map(EVENTS) do ev
    if haskey(nwb, ev.path)
        times = Float64.(read(nwb[ev.path]))
        println("Events  : $(length(times)) × '$(ev.label)'  ($(ev.path))")
        (label=ev.label, path=ev.path, color=ev.color, times=times)
    else
        @warn "Event path not found, skipping: $(ev.path)"
        nothing
    end
end |> x -> filter(!isnothing, x)

# Keep the first event set as the default for peri-event analysis below
tstart = isempty(loaded_events) ? Float64[] : loaded_events[1].times

# %% ── Session-wide raster ───────────────────────────────────────────────────
#
# Pass event_time = 0 and window [TIME_IN, TIME_OUT] so that the centred
# spike times equal the absolute times — giving us the full-session scatter.

X, Y = simple_raster_units(spk_sorted, 0.0, TIME_IN, TIME_OUT)
Y    = Int.(Y)
println("Scatter points: $(length(X))")

# %% ── Population firing rate ────────────────────────────────────────────────
#
# Histogram of all spike times in the session window, divided by the number
# of units and the bin width → mean firing rate per unit in Hz.

edges    = collect(range(TIME_IN, TIME_OUT; step=BIN_SZ))
pop_cnt  = _histcount(X, edges)
pop_rate = Float64.(pop_cnt) ./ n_units ./ BIN_SZ
timevec  = edges[1:end-1] .+ BIN_SZ / 2

# %% ── Plot ──────────────────────────────────────────────────────────────────

## ── Panel 1: event markers ──
event_plt = plot(;
    title         = basename(NWB_FILE),
    xlims         = (TIME_IN, TIME_OUT),
    yticks        = false,
    ylabel        = "events",
    legend        = :topright,
    bottom_margin = -2Plots.mm)

for ev in loaded_events
    in_win = filter(t -> TIME_IN ≤ t ≤ TIME_OUT, ev.times)
    isempty(in_win) && continue
    # Draw the first vline with a label so it appears in the legend;
    # the rest are unlabelled to avoid a legend entry per timestamp.
    vline!(event_plt, [in_win[1]];  color=ev.color, lw=0.6, alpha=0.6, label=ev.label)
    length(in_win) > 1 &&
        vline!(event_plt, in_win[2:end]; color=ev.color, lw=0.6, alpha=0.6, label=false)
end

## ── Panel 2: raster ──
sc = scatter(X, Y;
    mc               = :black,
    ms               = 0.8,
    markerstrokewidth = 0,
    xlims            = (TIME_IN, TIME_OUT),
    ylims            = (0.5, n_units + 0.5),
    yticks           = (1:n_units, ylab),
    ylabel           = "region",
    legend           = false)

# Overlay one scatter series per region for colour (one call per region,
# not one per unit — much faster for large recordings)
for (R, reg) in enumerate(pfc_regions)
    col  = color_map[reg]
    rows = findall(==(reg), ylab)   # unit-row indices belonging to this region
    row_set = Set(rows)
    mask = [yi in row_set for yi in Y]
    isempty(mask) && continue
    scatter!(sc, X[mask], Y[mask];
        mc=col, ms=0.8, markerstrokewidth=0, label=false)
end

# Separator lines between regions
for b in boundaries
    hline!(sc, [b + 0.5]; color=:black, lw=1.0, label=false)
end

## ── Panel 3: population firing rate ──
rate_plt = plot(timevec, pop_rate;
    color        = :black,
    lw           = 0.8,
    xlims        = (TIME_IN, TIME_OUT),
    ylabel       = "spikes/s",
    xlabel       = "time (s)",
    legend       = false,
    top_margin   = -2Plots.mm)

## ── Combine ──
plt = plot(event_plt, sc, rate_plt;
    layout = grid(3, 1; heights=[0.06, 0.80, 0.14]),
    size   = (1200, 800),
    link   = :x)      # ← links all three x-axes so zooming one zooms all

display(plt)

# %% ── (Optional) Save figure ────────────────────────────────────────────────
# Uncomment to save:
# savefig(plt, joinpath(@__DIR__, "session_overview.png"))

# ============================================================================
#  Peri-event raster + PSTH  (single unit)
#  Run this cell after setting UNIT_IDX and confirming tstart is loaded.
# ============================================================================

# %% ── Peri-event: configuration ─────────────────────────────────────────────

UNIT_IDX   = 1       # which unit to inspect (index in spk_sorted)
WIN_START  = -0.5    # window start re event (s)
WIN_STOP   =  1.5    # window stop  re event (s)
PSTH_BIN   =  0.02   # PSTH bin size (s)

# %% ── Peri-event: compute ───────────────────────────────────────────────────

isempty(tstart) && error("No events loaded — check EVENT_PATH.")

Xr, Yr = simple_raster(spk_sorted[UNIT_IDX], tstart, WIN_START, WIN_STOP)
Yr     = Int.(Yr)

rate, t_psth = simple_PSTH(spk_sorted[UNIT_IDX], tstart,
                            PSTH_BIN, WIN_START, WIN_STOP)

unit_label = "unit $(sort_idx[UNIT_IDX])  [$(ylab[UNIT_IDX])]"

# %% ── Peri-event: plot ───────────────────────────────────────────────────────

rast_plt = scatter(Xr, Yr;
    mc               = :black,
    ms               = 1.5,
    markerstrokewidth = 0,
    xlims            = (WIN_START, WIN_STOP),
    ylims            = (0, length(tstart) + 1),
    ylabel           = "trial",
    legend           = false,
    title            = unit_label)
vline!(rast_plt, [0.0]; color=:red, lw=1, ls=:dash, label=false)

psth_plt = plot(t_psth, rate;
    color  = :black,
    lw     = 1.2,
    xlims  = (WIN_START, WIN_STOP),
    xlabel = "time re event (s)",
    ylabel = "spikes/s",
    legend = false)
vline!(psth_plt, [0.0]; color=:red, lw=1, ls=:dash, label=false)

peri_plt = plot(rast_plt, psth_plt;
    layout = grid(2, 1; heights=[0.70, 0.30]),
    size   = (700, 600),
    link   = :x)

display(peri_plt)
