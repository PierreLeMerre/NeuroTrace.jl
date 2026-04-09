# ============================================================================
#  Population_heatmap.jl  —  Population response heatmap aligned to an event
#
#  Produces one figure with two columns (raw Hz | z-scored).
#  Neurons are grouped by region; within each region they are peak-sorted.
#
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Configuration ─────────────────────────────────────────────────────────

NWB_FILE   = "/Volumes/T7/NWB_Joana/NWB/999770_20251111_probe01.nwb"

#EVENT_PATH = "intervals/trials/start_time"
#EVENT_PATH = "intervals/trials/lick_time"
EVENT_PATH = "intervals/trials/imec_blue_led_on"
#EVENT_PATH = "intervals/trials/imec_stim_on"
#EVENT_PATH = "intervals/trials/imec_lick"
#EVENT_PATH = "intervals/trials/imec_reward_on"
#EVENT_PATH = "intervals/trials/imec_punishment_on"

#EVENT_PATH = "intervals/trials/first_move_time"
#EVENT_PATH = "intervals/trials/area_entry_time"
#EVENT_PATH = "intervals/trials/area_exit_time"
#EVENT_PATH = "intervals/trials/reward_time"
#EVENT_PATH = "intervals/trials/distractor"

WIN_START  = -1.0    # window start relative to event (s)
WIN_STOP   =  3.0    # window stop  relative to event (s)
PSTH_BIN   =  0.01  # bin width (s)

BASELINE_STOP = 0.0  # z-score baseline: bins where t < this value
SMOOTH_SIGMA  = 5.0  # Gaussian kernel width in bins (1 bin = PSTH_BIN seconds)
ZLIM          = 3.0  # colour saturation for z-scored panels (±σ)

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HDF5
using Plots
using Statistics
gr()

using NeuroTrace.IO:       read_ragged
using NeuroTrace.Analysis: population_psth, zscore_psth, smooth_psth, peak_sort

# %% ── Load spike times ───────────────────────────────────────────────────────

nwb       = h5open(NWB_FILE, "r")
spk_times = read_ragged(nwb, "units/spike_times", "units/spike_times_index")
n_units   = length(spk_times)
println("Loaded $n_units units")

# %% ── Electrode regions ─────────────────────────────────────────────────────

ede_labels = String.(read(nwb["general/extracellular_ephys/electrodes/location"]))
main_ch    = Int.(read(nwb["units/peak_channel_id"])) .+ 1
ede_region = ede_labels[main_ch]

region_sort  = sortperm(ede_region)
spk_sorted   = spk_times[region_sort]
ylab         = ede_region[region_sort]
pfc_regions  = unique(ylab)

region_sizes = [count(==(r), ylab) for r in pfc_regions]
boundaries   = cumsum(region_sizes)
region_mids  = [div(get(boundaries, i-1, 0) + boundaries[i], 2)
                for i in eachindex(pfc_regions)]

println("Regions : ", join(pfc_regions, ", "))

# %% ── Load events ────────────────────────────────────────────────────────────

haskey(nwb, EVENT_PATH) || error("Event path not found: $EVENT_PATH")
event_times = Float64.(read(nwb[EVENT_PATH]))
println("Events  : $(length(event_times))")

# %% ── Compute + smooth matrices ─────────────────────────────────────────────
#
# Order matters:
#   1. population_psth  → raw Hz matrix
#   2. zscore_psth      → z-score against pre-event baseline (unsmoothed data)
#   3. smooth_psth      → smooth both for display
#   4. peak_sort        → within-region: sort neurons by peak response time

println("Computing PSTHs …")
mat_reg, t = population_psth(spk_sorted, event_times, PSTH_BIN, WIN_START, WIN_STOP)
z_reg      = zscore_psth(mat_reg, t; baseline_stop = BASELINE_STOP)

println("Smoothing (σ = $SMOOTH_SIGMA bins = $(round(SMOOTH_SIGMA * PSTH_BIN * 1000; digits=1)) ms) …")
s_mat_reg  = smooth_psth(mat_reg; σ = SMOOTH_SIGMA)
s_z_reg    = smooth_psth(z_reg;   σ = SMOOTH_SIGMA)

# Within-region peak sort: keep region blocks intact, sort neurons by peak
# response time inside each block.
intra_idx = Int[]
let offset = 0
    for r in pfc_regions
        block_size  = count(==(r), ylab)
        block_range = offset+1 : offset+block_size
        local_order = peak_sort(s_z_reg[block_range, :], t)   # ranks within block
        append!(intra_idx, collect(block_range)[local_order])
        offset += block_size
    end
end

s_mat_sorted = s_mat_reg[intra_idx, :]
s_z_sorted   = s_z_reg[intra_idx, :]

println("Done. Matrix: $(size(mat_reg))  (units × bins)")

# %% ── Plot helpers ───────────────────────────────────────────────────────────

function _event_line!(p; color=:white)
    vline!(p, [0.0]; color=color, lw=1.2, ls=:dash, label=false)
end

function _region_lines!(p, bs; color=:white)
    for b in bs
        hline!(p, [b + 0.5]; color=color, lw=0.8, label=false)
    end
end

# %% ── Figure 1: region-sorted ───────────────────────────────────────────────

p1_raw = heatmap(t, 1:n_units, s_mat_sorted;
    color          = :inferno,
    xlabel         = "time re event (s)",
    ylabel         = "unit",
    title          = "raw (Hz)",
    yticks         = (region_mids, pfc_regions),
    colorbar_title = "Hz")
_event_line!(p1_raw)
_region_lines!(p1_raw, boundaries[1:end-1])

p1_z = heatmap(t, 1:n_units, s_z_sorted;
    color          = cgrad(:RdBu, rev=true),
    clim           = (-ZLIM, ZLIM),
    xlabel         = "time re event (s)",
    ylabel         = "",
    title          = "z-scored",
    yticks         = (region_mids, pfc_regions),
    colorbar_title = "z")
_event_line!(p1_z, color=:black)
_region_lines!(p1_z, boundaries[1:end-1], color=:black)

#fig_region = plot(p1_raw, p1_z;
#    layout = grid(1, 2),
#    size   = (1000, 1500),
#    plot_title = "Region + peak-sorted  ·  $(basename(NWB_FILE))")

fig_region = plot(p1_z;
    size   = (1000, 1500),
    plot_title = "Region + peak-sorted  ·  $(basename(NWB_FILE))")


display(fig_region)

# %% ── (Optional) Save ───────────────────────────────────────────────────────
# savefig(fig_region, joinpath(@__DIR__, "heatmap_region_sorted.png"))
