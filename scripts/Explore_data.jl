# ============================================================================
#  Explore_data.jl  —  NeuroTrace session explorer
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
#  Each `# %%` comment marks a VSCode cell boundary.
# ============================================================================

# %% ── Configuration (edit before running) ──────────────────────────────────
# data_path and unit filters come from config.toml.
# The settings below are specific to the session-overview view.

TIME_IN    = 0.0    # session start to display (s)
TIME_OUT   = 100.0  # session end   to display (s)
BIN_SZ     = 0.10   # population firing-rate bin width (s)

# Events to overlay on the raster.  Add as many as you need.
# Each entry is a NamedTuple: (label, path, color).
# Paths that don't exist in the file are silently skipped.
EVENTS = [
    (label="Reward delivery", path="intervals/trials/rw_start",  color=:blue),
    (label="Trial start",     path="intervals/trials/start_time", color=:black),
    # (label="Lever press", path="stimulus/presentation/LeverPress/timestamps", color=:blue),
]

# Region colors come from the Allen atlas — no need to specify them manually.

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HDF5
using Plots
using Statistics
gr()

using NeuroTrace.IO:       load_config, load_units, filter_units,
                           load_atlas, region_display_labels, region_color_map
using NeuroTrace.Analysis: find_spks_in_window, simple_raster_units,
                           simple_raster, simple_PSTH, _histcount

# %% ── Load config, atlas and all sessions ───────────────────────────────────

cfg       = load_config(joinpath(@__DIR__, "config.toml"))
atlas     = load_atlas()
all_units = filter_units(load_units(cfg.data_path), cfg, atlas)

println("$(length(all_units)) session(s) loaded.")

# Helper: reconstruct the NWB file path for a given UnitInfo.
# Works whether cfg.data_path is a single file or a directory.
_file_path(ui) = isfile(cfg.data_path) ? cfg.data_path :
                 joinpath(cfg.data_path, ui.session_id)

# %% ── Plot one figure per session ───────────────────────────────────────────

for ui in all_units

    spk_times = ui.spike_times
    n_units   = length(spk_times)
    all_spk   = vcat(spk_times...)

    println("\n── $(ui.session_id) ──")
    println("Units   : $n_units")
    println("Total spikes: $(length(all_spk))")
    println("Session duration: $(round(all_spk[end]; digits=1)) s")

    # ── Region sort ──────────────────────────────────────────────────────────
    disp_labels  = region_display_labels(atlas, ui.regions, cfg.regions)
    sort_idx     = sortperm(disp_labels)
    spk_sorted   = spk_times[sort_idx]
    ylab         = disp_labels[sort_idx]
    pfc_regions  = unique(ylab)
    color_map    = region_color_map(atlas, pfc_regions)
    region_sizes = [count(==(r), ylab) for r in pfc_regions]
    boundaries   = cumsum(region_sizes)[1:end-1]

    println("Regions : ", join(pfc_regions, ", "))

    # ── Events (open the individual session file) ─────────────────────────────
    loaded_events = h5open(_file_path(ui), "r") do nwb
        map(EVENTS) do ev
            if haskey(nwb, ev.path)
                times = Float64.(read(nwb[ev.path]))
                println("Events  : $(length(times)) × '$(ev.label)'")
                (label=ev.label, path=ev.path, color=ev.color, times=times)
            else
                nothing
            end
        end |> x -> filter(!isnothing, x)
    end

    # ── Session-wide raster ───────────────────────────────────────────────────
    X, Y = simple_raster_units(spk_sorted, 0.0, TIME_IN, TIME_OUT)
    Y    = Int.(Y)

    # ── Population firing rate ────────────────────────────────────────────────
    edges    = collect(range(TIME_IN, TIME_OUT; step=BIN_SZ))
    pop_cnt  = _histcount(X, edges)
    pop_rate = Float64.(pop_cnt) ./ n_units ./ BIN_SZ
    timevec  = edges[1:end-1] .+ BIN_SZ / 2

    # ── Panel 1: event markers ────────────────────────────────────────────────
    event_plt = plot(;
        title         = ui.session_id,
        xlims         = (TIME_IN, TIME_OUT),
        yticks        = false,
        ylabel        = "events",
        legend        = :topright,
        bottom_margin = -2Plots.mm)

    for ev in loaded_events
        in_win = filter(t -> TIME_IN ≤ t ≤ TIME_OUT, ev.times)
        isempty(in_win) && continue
        vline!(event_plt, [in_win[1]];   color=ev.color, lw=0.6, alpha=0.6, label=ev.label)
        length(in_win) > 1 &&
            vline!(event_plt, in_win[2:end]; color=ev.color, lw=0.6, alpha=0.6, label=false)
    end

    # ── Panel 2: raster ───────────────────────────────────────────────────────
    sc = scatter(X, Y;
        mc                = :black,
        ms                = 0.8,
        markerstrokewidth = 0,
        xlims             = (TIME_IN, TIME_OUT),
        ylims             = (0.5, n_units + 0.5),
        yticks            = (1:n_units, ylab),
        ylabel            = "region",
        legend            = false)

    for reg in pfc_regions
        col     = color_map[reg]
        row_set = Set(findall(==(reg), ylab))
        mask    = [yi in row_set for yi in Y]
        any(mask) || continue
        scatter!(sc, X[mask], Y[mask]; mc=col, ms=0.8, markerstrokewidth=0, label=false)
    end

    for b in boundaries
        hline!(sc, [b + 0.5]; color=:black, lw=1.0, label=false)
    end

    # ── Panel 3: population firing rate ───────────────────────────────────────
    rate_plt = plot(timevec, pop_rate;
        color       = :black,
        lw          = 0.8,
        xlims       = (TIME_IN, TIME_OUT),
        ylabel      = "spikes/s",
        xlabel      = "time (s)",
        legend      = false,
        top_margin  = -2Plots.mm)

    # ── Combine and display ───────────────────────────────────────────────────
    plt = plot(event_plt, sc, rate_plt;
        layout = grid(3, 1; heights=[0.06, 0.80, 0.14]),
        size   = (1200, 800),
        link   = :x)

    display(plt)

    # Uncomment to save each session figure automatically:
    # savefig(plt, joinpath(@__DIR__, "$(ui.session_id)_overview.png"))

end  # for ui in all_units
