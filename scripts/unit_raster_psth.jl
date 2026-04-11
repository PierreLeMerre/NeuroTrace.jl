# ============================================================================
#  Unit_raster_psth.jl  —  Peri-event raster + PSTH for a single unit
#  Standalone: runs independently of Explore_data.jl.
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Configuration ─────────────────────────────────────────────────────────
# data_path, event_path and unit filters come from config.toml.

SESSION_IDX = 4    # which session to use (1 = first file, 2 = second, …)
UNIT_IDX    = 101  # row index in the region-sorted, filtered unit list of that session

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Plots
using Statistics
gr()

using NeuroTrace.IO:       load_config, load_units, load_events, filter_units,
                           load_atlas, region_display_labels, region_color_map
using NeuroTrace.Analysis: simple_raster, simple_PSTH

# %% ── Load config, atlas, units and events ───────────────────────────────────

cfg   = load_config(joinpath(@__DIR__, "config.toml"))
atlas = load_atlas()

WIN_START = cfg.win_start
WIN_STOP  = cfg.win_stop
PSTH_BIN  = cfg.psth_bin

# Load all sessions, then pick the one requested by SESSION_IDX
all_units  = filter_units(load_units(cfg.data_path), cfg, atlas)
all_events = load_events(cfg.data_path, cfg.event_path)

SESSION_IDX ≤ length(all_units) ||
    error("SESSION_IDX=$SESSION_IDX but only $(length(all_units)) session(s) loaded.")

ui = all_units[SESSION_IDX]
# Match event file by session_id so index stays correct even after unit filtering
ev = first(filter(e -> e.session_id == ui.session_id, all_events))

# Sort by display label (parent region if cfg.regions given, else raw)
disp_labels = region_display_labels(atlas, ui.regions, cfg.regions)
sort_idx    = sortperm(disp_labels)
spk_sorted  = ui.spike_times[sort_idx]
ylab        = disp_labels[sort_idx]
event_times = ev.times

# Region color for this unit
unit_region = ylab[UNIT_IDX]
unit_color  = region_color_map(atlas, [unit_region])[unit_region]

println("Session : $(ui.session_id)  ($(SESSION_IDX) / $(length(all_units)))")
println("Unit    : $UNIT_IDX / $(length(spk_sorted))  →  $unit_region  ($unit_color)")
println("Spikes  : $(length(spk_sorted[UNIT_IDX]))")
println("Events  : $(length(event_times))  ($(cfg.event_path))")

# %% ── Compute raster + PSTH ──────────────────────────────────────────────────

spk          = spk_sorted[UNIT_IDX]
Xr, Yr       = simple_raster(spk, event_times, WIN_START, WIN_STOP)
Yr           = Int.(Yr)
rate, t_psth = simple_PSTH(spk, event_times, PSTH_BIN, WIN_START, WIN_STOP)

unit_label = "$(ui.session_id)  ·  unit $UNIT_IDX  [$unit_region]"

# %% ── Plot ───────────────────────────────────────────────────────────────────

rast_plt = scatter(Xr, Yr;
    mc                = unit_color,
    ms                = 1.5,
    markerstrokewidth = 0,
    xlims             = (WIN_START, WIN_STOP),
    ylims             = (0, length(event_times) + 1),
    ylabel            = "trial",
    title             = unit_label,
    legend            = false)
vline!(rast_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

psth_plt = plot(t_psth, rate;
    color   = unit_color,
    lw      = 1.5,
    xlims   = (WIN_START, WIN_STOP),
    xlabel  = "time re event (s)",
    ylabel  = "spikes/s",
    legend  = false)
vline!(psth_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

plt = plot(rast_plt, psth_plt;
    layout = grid(2, 1; heights=[0.72, 0.28]),
    size   = (700, 650),
    link   = :x)

display(plt)

# %% ── (Optional) Save ───────────────────────────────────────────────────────
# savefig(plt, joinpath(@__DIR__, "$(ui.session_id)_unit$(UNIT_IDX)_raster_psth.png"))
