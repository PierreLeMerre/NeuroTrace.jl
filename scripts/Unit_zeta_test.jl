# ============================================================================
#  Unit_zeta_test.jl  —  ZETA responsiveness test for a single unit
#
#  Mirrors Unit_raster_psth.jl: same SESSION_IDX / UNIT_IDX controls.
#  Produces one column per event type defined in config.toml, each with
#  three vertically stacked panels:
#    row 1 → peri-event raster
#    row 2 → ZETA deviation curve  (raw + mean null ± 2 SD)
#    row 3 → instantaneous firing rate (IFR, baseline + post-event)
#
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
# ============================================================================

# %% ── Controls ───────────────────────────────────────────────────────────────

SESSION_IDX = 1    # which session (1 = first file, 2 = second, …)
UNIT_IDX    = 10   # row in the region-sorted, filtered unit list for that session
RESAMP_NUM  = 200  # jitter resamplings (100 = fast / exploratory, 200+ = publication)

# %% ── Dependencies ───────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Plots
using Statistics
gr()

using NeuroTrace.IO:       load_config, load_units, load_events, filter_units,
                           load_atlas, region_display_labels, region_color_map
using NeuroTrace.Analysis: simple_raster
using ZetaJu: zetatest, ifr

# %% ── Load config, atlas, units and events ───────────────────────────────────

cfg   = load_config(joinpath(@__DIR__, "config.toml"))
atlas = load_atlas()

_save(fig, stem) = isempty(cfg.save_path) ? nothing :
    (mkpath(cfg.save_path);
     out = joinpath(cfg.save_path, "$(stem).$(cfg.save_format)");
     savefig(fig, out); println("Saved: $out"))

WIN_START = cfg.win_start
WIN_STOP  = cfg.win_stop

all_units      = filter_units(load_units(cfg.data_path), cfg, atlas)
all_ev_by_type = load_events(cfg.data_path, cfg.events)
n_ev_types     = length(all_ev_by_type)

SESSION_IDX ≤ length(all_units) ||
    error("SESSION_IDX=$SESSION_IDX but only $(length(all_units)) session(s) loaded.")

ui = all_units[SESSION_IDX]

# Region-sort units within this session
disp_labels = region_display_labels(atlas, ui.regions, cfg.regions)
sort_idx    = sortperm(disp_labels)
spk_sorted  = ui.spike_times[sort_idx]
ylab        = disp_labels[sort_idx]

UNIT_IDX ≤ length(spk_sorted) ||
    error("UNIT_IDX=$UNIT_IDX but session has only $(length(spk_sorted)) units.")

unit_region = ylab[UNIT_IDX]
unit_color  = region_color_map(atlas, [unit_region];
                               custom_colors = cfg.region_colors)[unit_region]

println("Session : $(ui.session_id)  ($SESSION_IDX / $(length(all_units)))")
println("Unit    : $UNIT_IDX / $(length(spk_sorted))  →  $unit_region  ($unit_color)")
println("Spikes  : $(length(spk_sorted[UNIT_IDX]))")
println("Events  : $n_ev_types type(s)  " *
        "($(join([s.label for s in cfg.events], ", ")))")

spk        = Float64.(spk_sorted[UNIT_IDX])
unit_label = "$(ui.session_id)  ·  unit $UNIT_IDX  [$unit_region]"

# %% ── Build panels for each event type ──────────────────────────────────────

rast_panels = Plots.Plot[]
dev_panels  = Plots.Plot[]
ifr_panels  = Plots.Plot[]

for (k, ev_list) in enumerate(all_ev_by_type)
    spec    = cfg.events[k]
    ev_idx  = findfirst(e -> e.session_id == ui.session_id, ev_list)
    isnothing(ev_idx) &&
        error("No events for session '$(ui.session_id)' in type '$(spec.label)'")
    ev          = ev_list[ev_idx]
    event_times = ev.t_start
    ev_f64      = Float64.(event_times)

    println("\n── '$(spec.label)' : $(length(ev_f64)) trials ──")

    # ── Raster ────────────────────────────────────────────────────────────────
    Xr, Yr = simple_raster(spk, ev_f64, WIN_START, WIN_STOP)

    rast_plt = scatter(Xr, Int.(Yr);
        mc                = unit_color,
        ms                = 1.5,
        markerstrokewidth = 0,
        xlims             = (WIN_START, WIN_STOP),
        ylims             = (0, length(ev_f64) + 1),
        ylabel            = k == 1 ? "trial" : "",
        title             = k == 1 ? unit_label : spec.label,
        legend            = false)
    vline!(rast_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

    # ── ZETA test ─────────────────────────────────────────────────────────────
    println("  Running ZETA (intResampNum=$RESAMP_NUM) …")
    dblZetaP, dZETA, dRate = zetatest(spk, ev_f64;
        intResampNum   = RESAMP_NUM,
        boolReturnRate = true)

    lat_zeta = dZETA["dblLatencyZETA"]
    println("  p = $(round(dblZetaP; sigdigits=3))  |  " *
            "ZETA latency: $(isnothing(lat_zeta) ? "n/a" : round(lat_zeta; digits=4)) s")

    # ── ZETA deviation curve ──────────────────────────────────────────────────
    vec_t     = dZETA["vecSpikeT"]
    vec_dev   = dZETA["vecRealDeviation"]
    rand_devs = dZETA["cellRandDeviation"]

    zeta_dur = !isnothing(dZETA["dblUseMaxDur"]) ? Float64(dZETA["dblUseMaxDur"]) :
               (!isnothing(vec_t) && !isempty(vec_t) ? maximum(vec_t) : WIN_STOP)

    dev_plt = plot(; xlabel = k == n_ev_types ? "time re event onset (s)" : "",
                     ylabel = k == 1 ? "Δ cum. fraction" : "",
                     title  = "p = $(round(dblZetaP; sigdigits=3))",
                     legend = false,
                     xlims  = (0.0, zeta_dur))

    if !isnothing(vec_t) && !isnothing(vec_dev)
        ref_level = 0.0
        if !isnothing(rand_devs) && length(rand_devs) > 0
            null_peaks = [maximum(abs.(d)) for d in rand_devs if !isempty(d)]
            ref_level  = mean(null_peaks) + 2 * std(null_peaks)
        end
        y_abs = max(maximum(abs.(vec_dev)), ref_level)
        ylims!(dev_plt, -1.15 * y_abs, 1.15 * y_abs)

        hline!(dev_plt, [0.0]; color=:grey30, lw=0.8, ls=:dash, label=false)
        if ref_level > 0
            hline!(dev_plt, [ ref_level]; color=:grey65, lw=1.0, ls=:dot, label=false)
            hline!(dev_plt, [-ref_level]; color=:grey65, lw=1.0, ls=:dot, label=false)
        end

        plot!(dev_plt, vec_t, vec_dev;
              fillrange = zeros(length(vec_dev)),
              fillalpha = 0.25, fillcolor = unit_color,
              color     = unit_color, lw = 1.5)

        if !isnothing(lat_zeta)
            idx_peak = argmin(abs.(vec_t .- lat_zeta))
            vline!(dev_plt, [lat_zeta]; color=:red, lw=1.0, ls=:dash, label=false)
            scatter!(dev_plt, [lat_zeta], [vec_dev[idx_peak]];
                     mc=:red, ms=5, markerstrokewidth=0, label=false)
        end
    end

    # ── IFR: baseline + post-event ────────────────────────────────────────────
    ev_baseline = ev_f64 .+ WIN_START
    bl_dur      = Float64(abs(WIN_START))
    bl_t, bl_rate, _ = ifr(spk, ev_baseline; dblUseMaxDur = bl_dur)
    bl_t_shifted = bl_t .+ WIN_START

    ifr_plt = plot(; xlabel = "time re event (s)",
                     ylabel = k == 1 ? "spikes/s" : "",
                     legend = false,
                     xlims  = (WIN_START, WIN_STOP))
    vline!(ifr_plt, [0.0]; color=:grey40, lw=1, ls=:dash, label=false)

    isempty(bl_t_shifted) ||
        plot!(ifr_plt, bl_t_shifted, bl_rate; color=unit_color, lw=1.5)

    ifr_t    = dRate["vecT"]
    ifr_rate = dRate["vecRate"]
    if !isnothing(ifr_t) && !isnothing(ifr_rate)
        plot!(ifr_plt, ifr_t, ifr_rate; color=unit_color, lw=1.5)
        lat_peak = dRate["dblLatencyPeak"]
        if !isnothing(lat_peak)
            idx_p = argmin(abs.(ifr_t .- lat_peak))
            scatter!(ifr_plt, [lat_peak], [ifr_rate[idx_p]];
                     mc=:red, ms=5, markerstrokewidth=0, label=false)
        end
    end

    push!(rast_panels, rast_plt)
    push!(dev_panels,  dev_plt)
    push!(ifr_panels,  ifr_plt)
end

# %% ── Compose: [rast1 rast2 … ; dev1 dev2 … ; ifr1 ifr2 …] ─────────────────
# The deviation row has its own x-axis (stitched trial time 0→dblUseMaxDur)
# and cannot be linked to the raster/IFR rows — no link=:x.

plt = plot(rast_panels..., dev_panels..., ifr_panels...;
    layout = grid(3, n_ev_types; heights=[0.45, 0.30, 0.25]),
    size   = (700 * n_ev_types, 850))

display(plt)
ev_labels = join([replace(s.label, " " => "-") for s in cfg.events], "_")
_save(plt, "$(replace(ui.session_id, ".nwb" => ""))_unit$(UNIT_IDX)_$(unit_region)_zeta_$ev_labels")
