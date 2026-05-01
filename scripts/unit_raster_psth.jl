# ============================================================================
#  Unit_raster_psth.jl  —  Peri-event raster + PSTH for a single unit
#
#  Produces one column per event type defined in config.toml:
#    row 1 → raster   (trials × time)
#    row 2 → PSTH     (firing rate vs time)
#
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Configuration ─────────────────────────────────────────────────────────
# data_path, events and unit filters come from config.toml.

SESSION_IDX = 1    # which session to use (1 = first file, 2 = second, …)
UNIT_IDX    = 10   # row index in the region-sorted, filtered unit list of that session

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

_save(fig, stem) = isempty(cfg.save_path) ? nothing :
    (mkpath(cfg.save_path);
     out = joinpath(cfg.save_path, "$(stem).$(cfg.save_format)");
     savefig(fig, out); println("Saved: $out"))

WIN_START = cfg.win_start
WIN_STOP  = cfg.win_stop
PSTH_BIN  = cfg.psth_bin

all_units      = filter_units(load_units(cfg.data_path), cfg, atlas)
all_ev_by_type = load_events(cfg.data_path, cfg.events)
n_ev_types     = length(all_ev_by_type)

SESSION_IDX ≤ length(all_units) ||
    error("SESSION_IDX=$SESSION_IDX but only $(length(all_units)) session(s) loaded.")

ui = all_units[SESSION_IDX]

# Sort by display label (parent region if cfg.regions given, else raw)
disp_labels = region_display_labels(atlas, ui.regions, cfg.regions)
sort_idx    = sortperm(disp_labels)
spk_sorted  = ui.spike_times[sort_idx]
ylab        = disp_labels[sort_idx]

unit_region = ylab[UNIT_IDX]
unit_color  = region_color_map(atlas, [unit_region];
                               custom_colors = cfg.region_colors)[unit_region]

println("Session : $(ui.session_id)  ($SESSION_IDX / $(length(all_units)))")
println("Unit    : $UNIT_IDX / $(length(spk_sorted))  →  $unit_region  ($unit_color)")
println("Spikes  : $(length(spk_sorted[UNIT_IDX]))")
println("Events  : $n_ev_types type(s)  " *
        "($(join([s.label for s in cfg.events], ", ")))")

spk        = spk_sorted[UNIT_IDX]
unit_label = "$(ui.session_id)  ·  unit $UNIT_IDX  [$unit_region]"

# %% ── Compute raster + PSTH for each event type ─────────────────────────────

rast_panels = Plots.Plot[]
psth_panels = Plots.Plot[]

for (k, ev_list) in enumerate(all_ev_by_type)
    spec = cfg.events[k]

    # Match this session's EventInfo by session_id
    ev_idx = findfirst(e -> e.session_id == ui.session_id, ev_list)
    isnothing(ev_idx) &&
        error("No events for session '$(ui.session_id)' in event type '$(spec.label)'")
    ev          = ev_list[ev_idx]
    event_times = ev.t_start

    println("  '$(spec.label)': $(length(event_times)) trials")

    Xr, Yr       = simple_raster(spk, event_times, WIN_START, WIN_STOP)
    Yr           = Int.(Yr)
    rate, t_psth = simple_PSTH(spk, event_times, PSTH_BIN, WIN_START, WIN_STOP)

    rast_plt = scatter(Xr, Yr;
        mc                = unit_color,
        ms                = 1.5,
        markerstrokewidth = 0,
        xlims             = (WIN_START, WIN_STOP),
        ylims             = (0, length(event_times) + 1),
        ylabel            = k == 1 ? "trial" : "",
        title             = k == 1 ? unit_label : spec.label,
        legend            = false)
    vline!(rast_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

    psth_plt = plot(t_psth, rate;
        color   = unit_color,
        lw      = 1.5,
        xlims   = (WIN_START, WIN_STOP),
        xlabel  = "time re event (s)",
        ylabel  = k == 1 ? "spikes/s" : "",
        legend  = false)
    vline!(psth_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

    push!(rast_panels, rast_plt)
    push!(psth_panels, psth_plt)
end

# %% ── Compose: [rast1 rast2 … ; psth1 psth2 …] ──────────────────────────────
# grid(2, N) fills row-major, so passing rast panels first then psth panels
# gives the raster row on top and PSTH row on the bottom across all columns.

plt = plot(rast_panels..., psth_panels...;
    layout = grid(2, n_ev_types; heights=[0.72, 0.28]),
    size   = (700 * n_ev_types, 650),
    link   = :x)

display(plt)
_save(plt, "$(replace(ui.session_id, ".nwb" => ""))_unit$(UNIT_IDX)_$(unit_region)")
