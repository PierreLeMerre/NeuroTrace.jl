# ============================================================================
#  Unit_raster_psth.jl  —  Peri-event raster + PSTH for a single unit
#  Standalone: runs independently of Explore_data.jl.
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Configuration ─────────────────────────────────────────────────────────

NWB_FILE  = "/Volumes/T7/NWB_Alicante/NWB/SC19_20250529.nwb"   # ← same file as Explore_data

UNIT_IDX  = 100        # row index in the region-sorted unit list (1 = first unit)
                     # run Explore_data first to see the full sorted list

EVENT_PATH = "intervals/trials/start_time"   # NWB path to event timestamps
                                             # other option: "intervals/trials/rw_start"

WIN_START = -0.5     # window start relative to each event (s)
WIN_STOP  =  1.5     # window stop  relative to each event (s)
PSTH_BIN  =  0.025   # PSTH bin width (s)

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HDF5
using Plots
using Statistics
gr()

using NeuroTrace.IO:       read_ragged
using NeuroTrace.Analysis: simple_raster, simple_PSTH

# %% ── Load spike times ───────────────────────────────────────────────────────

nwb       = h5open(NWB_FILE, "r")
unit_ids  = Int.(read(nwb["units/id"]))
spk_times = read_ragged(nwb, "units/spike_times", "units/spike_times_index")

# %% ── Sort units by region (mirrors Explore_data sort order) ─────────────────

ede_labels = String.(read(nwb["general/extracellular_ephys/electrodes/location"]))
main_ch    = Int.(read(nwb["units/peak_channel_id"])) .+ 1
ede_region = ede_labels[main_ch]

sort_idx   = sortperm(ede_region)
spk_sorted = spk_times[sort_idx]
ylab       = ede_region[sort_idx]

println("Unit $UNIT_IDX / $(length(spk_sorted))  →  region: $(ylab[UNIT_IDX])")
println("Spike count: $(length(spk_sorted[UNIT_IDX]))")

# %% ── Load event timestamps ──────────────────────────────────────────────────

haskey(nwb, EVENT_PATH) || error("Event path not found in file: $EVENT_PATH")
event_times = Float64.(read(nwb[EVENT_PATH]))
println("Events: $(length(event_times))  ($(EVENT_PATH))")

# %% ── Compute raster + PSTH ──────────────────────────────────────────────────

spk = spk_sorted[UNIT_IDX]

Xr, Yr   = simple_raster(spk, event_times, WIN_START, WIN_STOP)
Yr       = Int.(Yr)

rate, t_psth = simple_PSTH(spk, event_times, PSTH_BIN, WIN_START, WIN_STOP)

unit_label = "unit $(unit_ids[sort_idx[UNIT_IDX]])  [$(ylab[UNIT_IDX])]"

# %% ── Plot ───────────────────────────────────────────────────────────────────

rast_plt = scatter(Xr, Yr;
    mc                = :black,
    ms                = 1.5,
    markerstrokewidth = 0,
    xlims             = (WIN_START, WIN_STOP),
    ylims             = (0, length(event_times) + 1),
    ylabel            = "trial",
    title             = unit_label,
    legend            = false)
vline!(rast_plt, [0.0]; color=:red, lw=1, ls=:dash, label=false)

psth_plt = plot(t_psth, rate;
    color   = :black,
    lw      = 1.5,
    xlims   = (WIN_START, WIN_STOP),
    xlabel  = "time re event (s)",
    ylabel  = "spikes/s",
    legend  = false)
vline!(psth_plt, [0.0]; color=:red, lw=1, ls=:dash, label=false)

plt = plot(rast_plt, psth_plt;
    layout = grid(2, 1; heights=[0.72, 0.28]),
    size   = (700, 650),
    link   = :x)

display(plt)

# %% ── (Optional) Save ───────────────────────────────────────────────────────
# savefig(plt, joinpath(@__DIR__, "unit_$(UNIT_IDX)_raster_psth.png"))
