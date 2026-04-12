# ============================================================================
#  Population_heatmap.jl  —  Population response heatmap aligned to an event
#
#  Produces one figure with two columns (raw Hz | z-scored).
#  Neurons are grouped by region; within each region they are peak-sorted.
#
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Configuration ─────────────────────────────────────────────────────────
# All parameters live in config.toml (same directory as this script).
# Edit that file — do not hardcode values here.

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Plots
using Statistics
gr()

using NeuroTrace.IO:       load_config, load_units, load_events, filter_units,
                           load_atlas, region_display_labels, region_color_map
using NeuroTrace.Analysis: population_psth_multi, zscore_psth, smooth_psth, peak_sort

# %% ── Load config ────────────────────────────────────────────────────────────

cfg = load_config(joinpath(@__DIR__, "config.toml"))

WIN_START     = cfg.win_start
WIN_STOP      = cfg.win_stop
PSTH_BIN      = cfg.psth_bin
BASELINE_STOP = cfg.baseline_stop
SMOOTH_SIGMA  = cfg.smooth_sigma
ZLIM          = cfg.zlim

# %% ── Load atlas, filter units and events ───────────────────────────────────

atlas   = load_atlas()
units   = filter_units(load_units(cfg.data_path), cfg, atlas)
events  = load_events(cfg.data_path, cfg.event_path)
n_units = sum(length(u.spike_times) for u in units)

println("Total units : $n_units across $(length(units)) session(s)")

# %% ── Region sort ────────────────────────────────────────────────────────────
#
# raw_regions   – exact labels from NWB (e.g. "ACA1", "ACA2/3")
# disp_labels   – display labels (e.g. "ACA" when user wrote regions=["ACA"])
#                 equals raw_regions when cfg.regions is empty
# color_map     – Dict(display_label => "#RRGGBB") from the Allen atlas

all_raw     = vcat([u.regions for u in units]...)
disp_labels = region_display_labels(atlas, all_raw, cfg.regions)
color_map   = region_color_map(atlas, disp_labels; custom_colors = cfg.region_colors)

region_sort = sortperm(disp_labels)
ylab        = disp_labels[region_sort]
pfc_regions = unique(ylab)

region_sizes = [count(==(r), ylab) for r in pfc_regions]
boundaries   = cumsum(region_sizes)
region_mids  = [div(get(boundaries, i-1, 0) + boundaries[i], 2)
                for i in eachindex(pfc_regions)]

println("Regions : ", join(pfc_regions, ", "))

# %% ── Compute + smooth matrices ─────────────────────────────────────────────
#
# Order matters:
#   1. population_psth  → raw Hz matrix
#   2. zscore_psth      → z-score against pre-event baseline (unsmoothed data)
#   3. smooth_psth      → smooth both for display
#   4. peak_sort        → within-region: sort neurons by peak response time

println("Computing PSTHs …")
mat, t  = population_psth_multi(units, events, PSTH_BIN, WIN_START, WIN_STOP)
mat_reg = mat[region_sort, :]     # apply region sort (based on display labels)
z_reg   = zscore_psth(mat_reg, t; baseline_stop = BASELINE_STOP)

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

println("Done. Matrix: $(size(mat_reg)) (units × bins)")

# %% ── Save helper ───────────────────────────────────────────────────────────

function _save(fig, stem)
    isempty(cfg.save_path) && return
    mkpath(cfg.save_path)
    out = joinpath(cfg.save_path, "$(stem).$(cfg.save_format)")
    savefig(fig, out)
    println("Saved: $out")
end

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
#    plot_title = "Region + peak-sorted  ·  $(join([u.session_id for u in units], ", "))")

fig_region = plot(p1_z;
    size   = (1000, 1500),
    plot_title = "Region + peak-sorted  ·  $(join([u.session_id for u in units], ", "))")


display(fig_region)
_save(fig_region, "heatmap_$(replace(cfg.event_path, '/' => '_'))")
