# ============================================================================
#  Population_heatmap.jl  —  Population response heatmap aligned to events
#
#  Produces one z-scored heatmap column per event type defined in config.toml.
#  Neurons are grouped by region; within each region they are peak-sorted
#  using the first event type — the same ordering is applied to all columns
#  so cross-event comparisons are straightforward.
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
                           load_atlas, region_display_labels, region_color_map,
                           UnitInfo
using NeuroTrace.Analysis: population_psth_multi, zscore_psth, smooth_psth, peak_sort,
                           zeta_pvalues

# %% ── Load config ────────────────────────────────────────────────────────────

cfg = load_config(joinpath(@__DIR__, "config.toml"))

WIN_START     = cfg.win_start
WIN_STOP      = cfg.win_stop
PSTH_BIN      = cfg.psth_bin
BASELINE_STOP = cfg.baseline_stop
SMOOTH_SIGMA  = cfg.smooth_sigma
ZLIM          = cfg.zlim

# ── ZETA options ──────────────────────────────────────────────────────────────
USE_ZETA    = false   # false → skip ZETA entirely (faster, keeps all units)
ZETA_ALPHA  = 0.05   # significance threshold for unit filtering
ZETA_RESAMP = 100    # jitter resamplings (100 = fast / exploratory, 200+ = publication)

# %% ── Load atlas, filter units and events ───────────────────────────────────

atlas = load_atlas()
units = filter_units(load_units(cfg.data_path), cfg, atlas)

# Load all event types.
# all_ev_by_type[k]    → Vector{EventInfo} for event type k (one per session)
# all_ev_by_type[k][j] → EventInfo for event k, session j
all_ev_by_type = load_events(cfg.data_path, cfg.events)
n_ev_types     = length(all_ev_by_type)
n_units_total  = sum(length(u.spike_times) for u in units)

println("Total units : $n_units_total across $(length(units)) session(s)")
println("Event types : $n_ev_types  ($(join([s.label for s in cfg.events], ", ")))")

# %% ── ZETA filtering (optional, computed on event type 1) ───────────────────

if USE_ZETA
    println("Running ZETA tests on '$(cfg.events[1].label)' " *
            "(intResampNum=$ZETA_RESAMP, α=$ZETA_ALPHA) …")
    pvals, latencies = zeta_pvalues(units, all_ev_by_type[1]; intResampNum = ZETA_RESAMP)

    n_total  = length(pvals)
    sig_mask = pvals .< ZETA_ALPHA
    n_sig    = sum(sig_mask)
    println("  Significant units : $n_sig / $n_total  (α=$ZETA_ALPHA)")

    units_filtered = let flat_idx = 0
        out = UnitInfo[]
        for ui in units
            n = length(ui.spike_times)
            mask_chunk = sig_mask[flat_idx+1 : flat_idx+n]
            flat_idx  += n
            filtered = UnitInfo(ui.spike_times[mask_chunk],
                                ui.regions[mask_chunk],
                                ui.depths[mask_chunk],
                                ui.session_id)
            isempty(filtered.spike_times) || push!(out, filtered)
        end
        out
    end

    sig_latencies = latencies[sig_mask]
else
    println("ZETA disabled — using all $n_units_total units.")
    units_filtered = units
    n_total        = n_units_total
    n_sig          = n_units_total
    sig_latencies  = fill(NaN, n_units_total)
end

# %% ── Region sort (depth-first, matching Explore_data.jl) ───────────────────
#
# Sort all units by DV depth (superficial → deep) so the heatmap rows follow
# probe order rather than alphabetical region order.  Contiguous region runs
# are then identified exactly as in the session explorer.

all_raw     = vcat([u.regions for u in units_filtered]...)
all_depths  = vcat([u.depths  for u in units_filtered]...)
disp_labels = region_display_labels(atlas, all_raw, cfg.regions)
color_map   = region_color_map(atlas, disp_labels; custom_colors = cfg.region_colors)

region_sort = sortperm(all_depths)          # NaN depths sort to end
ylab        = disp_labels[region_sort]

# Contiguous region runs in depth order: (label, first_row, last_row)
region_runs = Tuple{String,Int,Int}[]
let cur = ylab[1], lo = 1
    for i in 2:length(ylab)
        if ylab[i] != cur
            push!(region_runs, (cur, lo, i-1))
            cur = ylab[i]; lo = i
        end
    end
    push!(region_runs, (cur, lo, length(ylab)))
end

# Boundary rows (last row of each block) — used for hline separators
boundaries = [row_hi for (_, _, row_hi) in region_runs]

println("Regions (depth order) : ", join(unique(ylab), " → "))

# %% ── Compute PSTH matrices for every event type ────────────────────────────
#
# Each event type gets its own z-score baseline (pre-event window).
# Neuron ordering is always derived from event type 1 so columns are
# directly comparable row-by-row.

println("Computing PSTHs for $n_ev_types event type(s) …")

all_mat_reg = Matrix{Float64}[]
all_z_reg   = Matrix{Float64}[]
all_t       = Vector{Float64}[]

for (k, ev_list) in enumerate(all_ev_by_type)
    spec = cfg.events[k]
    println("  '$(spec.label)' …")
    mat, t  = population_psth_multi(units_filtered, ev_list, PSTH_BIN, WIN_START, WIN_STOP)
    mat_reg = mat[region_sort, :]
    z_reg   = zscore_psth(mat_reg, t; baseline_stop = BASELINE_STOP)
    push!(all_mat_reg, mat_reg)
    push!(all_z_reg,   z_reg)
    push!(all_t,       t)
end

n_units = size(all_mat_reg[1], 1)

# %% ── Smooth + intra-block peak sort (ordering always from event type 1) ────

println("Smoothing (σ = $SMOOTH_SIGMA bins = " *
        "$(round(SMOOTH_SIGMA * PSTH_BIN * 1000; digits=1)) ms) …")

all_s_z_reg = [smooth_psth(z; σ = SMOOTH_SIGMA) for z in all_z_reg]

# Peak sort within each contiguous depth block using event type 1.
# Loop over region_runs (not unique region names) so probe passes that
# re-enter a region are treated as separate blocks.
# intra_idx is then applied identically to all event types.
intra_idx = Int[]
for (_, row_lo, row_hi) in region_runs
    block_range = row_lo:row_hi
    local_order = peak_sort(all_s_z_reg[1][block_range, :], all_t[1])
    append!(intra_idx, collect(block_range)[local_order])
end

all_s_z_sorted = [s_z[intra_idx, :] for s_z in all_s_z_reg]

# Re-order ZETA latencies to match final display order
lat_reg    = sig_latencies[region_sort]
lat_sorted = lat_reg[intra_idx]

println("Done. Matrix size: $(size(all_mat_reg[1])) (units × bins)")

# %% ── Save helper ───────────────────────────────────────────────────────────

function _save(fig, stem)
    isempty(cfg.save_path) && return
    mkpath(cfg.save_path)
    out = joinpath(cfg.save_path, "$(stem).$(cfg.save_format)")
    savefig(fig, out)
    println("Saved: $out")
end

# %% ── Plot helpers ───────────────────────────────────────────────────────────

_event_line!(p; color=:black) =
    vline!(p, [0.0]; color=color, lw=1.2, ls=:dash, label=false)

_region_lines!(p, bs; color=:black) =
    (for b in bs; hline!(p, [b + 0.5]; color=color, lw=0.8, label=false); end)

# %% ── Region bar geometry ────────────────────────────────────────────────────
#
# The bar is embedded directly inside each heatmap's coordinate space, to the
# left of WIN_START.  Because it shares the same y-axis as the image, row
# alignment is guaranteed — no separate subplot needed.
#
#   x_lo ──[  colored bar  ]── WIN_START ──[  heatmap image  ]── WIN_STOP
#            ← bar_dx →  gap

bar_dx  = clamp(0.20 * abs(WIN_START), 0.05, 0.35)  # bar width (s)
bar_gap = max(PSTH_BIN * 3, 0.015)                    # gap to WIN_START edge
x_lo    = WIN_START - bar_dx - bar_gap                # shared left xlim for all panels

_tick_step  = (WIN_STOP - WIN_START) ≤ 4 ? 0.5 : 1.0
_xtick_vals = collect(WIN_START:_tick_step:WIN_STOP)

# %% ── Heatmap panels (one per event type) ───────────────────────────────────

panels = Plots.Plot[]

for k in 1:n_ev_types
    spec = cfg.events[k]
    t    = all_t[k]
    smat = all_s_z_sorted[k]

    title_str = if USE_ZETA && k == 1
        "$(spec.label)  [$n_sig/$n_total, ZETA p<$ZETA_ALPHA]"
    else
        spec.label
    end

    p = heatmap(t, 1:n_units, smat;
        color          = cgrad(:RdBu, rev=true),
        clim           = (-ZLIM, ZLIM),
        xlims          = (x_lo, WIN_STOP),
        xlabel         = "time re event (s)",
        ylabel         = "",
        title          = title_str,
        yticks         = false,
        colorbar_title = k == n_ev_types ? "z" : "")

    xticks!(p, _xtick_vals)

    # ── Region bar: colored blocks left of WIN_START ──────────────────────────
    x_bar_lo = x_lo
    x_bar_hi = x_lo + bar_dx
    for (reg, row_lo, row_hi) in region_runs
        col  = color_map[reg]
        y_lo = row_lo - 0.5
        y_hi = row_hi + 0.5
        plot!(p, Shape([x_bar_lo, x_bar_hi, x_bar_hi, x_bar_lo],
                       [y_lo, y_lo, y_hi, y_hi]);
              fillcolor=col, fillalpha=0.85, linecolor=:white, lw=0.4, label=false)
        annotate!(p, (x_bar_lo + x_bar_hi) / 2, (y_lo + y_hi) / 2,
                  text(reg, :center, 6, :black))
    end

    # Region boundary lines (heatmap area only — bar section is self-contained)
    for b in boundaries[1:end-1]
        plot!(p, [WIN_START, WIN_STOP], [b + 0.5, b + 0.5];
              color=:black, lw=0.8, label=false)
    end

    # Thin vertical separator between bar and heatmap
    vline!(p, [WIN_START]; color=:grey50, lw=0.5, label=false)

    # ── Event line ────────────────────────────────────────────────────────────
    _event_line!(p)

    # ── ZETA latency dots on all panels (same unit ordering) ─────────────────
    if USE_ZETA
        rows   = findall(isfinite, lat_sorted)
        dot_t  = lat_sorted[rows]
        dot_y  = rows
        in_win = (dot_t .>= WIN_START) .& (dot_t .<= WIN_STOP)
        if any(in_win)
            scatter!(p, dot_t[in_win], dot_y[in_win];
                mc=:black, ms=2.5, markerstrokewidth=0, alpha=0.8, label=false)
        end
    end

    push!(panels, p)
end

# %% ── Compose: N equal-width heatmap columns (region bar embedded in each) ───

session_str = join(unique([u.session_id for u in units_filtered]), ", ")

fig = plot(panels...;
    layout     = grid(1, n_ev_types),
    size       = (200 + 900 * n_ev_types, 1500),
    plot_title = "Depth + peak-sorted (by '$(cfg.events[1].label)')  ·  $session_str")

display(fig)

ev_labels = join([replace(s.label, " " => "-") for s in cfg.events], "_")
_save(fig, "heatmap_$ev_labels")
