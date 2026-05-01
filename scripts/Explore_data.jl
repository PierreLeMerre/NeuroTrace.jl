# ============================================================================
#  Explore_data.jl  —  NeuroTrace session explorer
#  Run cell-by-cell in VSCode (Shift+Enter) or whole file (Ctrl+F5).
#  Each `# %%` comment marks a VSCode cell boundary.
# ============================================================================

# %% ── Configuration (edit before running) ──────────────────────────────────
# data_path and unit filters come from config.toml.
# Events are defined in config.toml via [[events]] blocks — edit there.
# The settings below are specific to the session-overview view.

TIME_IN    = 1800.0    # session start to display (s)
TIME_OUT   = 2500.0   # session end   to display (s)
BIN_SZ     = 0.10   # population firing-rate bin width (s)

# Region colors come from the Allen atlas — no need to specify them manually.

# %% ── Dependencies ──────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Plots
using Statistics
gr()

using NeuroTrace.IO:       load_config, load_units, load_events, filter_units,
                           load_atlas, region_display_labels, region_color_map
using NeuroTrace.Analysis: simple_raster_units, _histcount

# %% ── Load config, atlas and all sessions ───────────────────────────────────

cfg       = load_config(joinpath(@__DIR__, "config.toml"))
atlas     = load_atlas()
all_units = filter_units(load_units(cfg.data_path), cfg, atlas)

_save(fig, stem) = isempty(cfg.save_path) ? nothing :
    (mkpath(cfg.save_path);
     out = joinpath(cfg.save_path, "$(stem).$(cfg.save_format)");
     savefig(fig, out); println("Saved: $out"))

println("$(length(all_units)) session(s) loaded.")

# Load all event types once up front.
# all_ev_by_type[k]    → Vector{EventInfo} for event type k (one per session)
# all_ev_by_type[k][j] → EventInfo with t_start, t_stop, label, color, session_id
all_ev_by_type = load_events(cfg.data_path, cfg.events)

# %% ── Plot one figure per session ───────────────────────────────────────────

for ui in all_units

    spk_times = ui.spike_times
    n_units   = length(spk_times)
    all_spk   = vcat(spk_times...)

    println("\n── $(ui.session_id) ──")
    println("Units        : $n_units")
    println("Total spikes : $(length(all_spk))")
    println("Session dur  : $(round(all_spk[end]; digits=1)) s")

    # ── Depth sort (DV ascending = superficial → deep) ───────────────────────
    disp_labels = region_display_labels(atlas, ui.regions, cfg.regions)
    sort_idx    = sortperm(ui.depths)          # NaN depths go to end
    spk_sorted  = spk_times[sort_idx]
    ylab        = disp_labels[sort_idx]
    color_map   = region_color_map(atlas, unique(ylab); custom_colors = cfg.region_colors)

    # Identify contiguous region runs in depth order.
    # Each run = (region label, first unit row, last unit row).
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

    println("Regions (depth order) : ", join(unique(ylab), " → "))

    # ── Events: pick this session's EventInfo for each event type ────────────
    loaded_events = filter(!isnothing, [
        begin
            idx = findfirst(e -> e.session_id == ui.session_id, ev_list)
            if isnothing(idx)
                @warn "No events for session '$(ui.session_id)' in type '$(ev_list[1].label)'"
                nothing
            else
                ev = ev_list[idx]
                has_interval = any(isfinite, ev.t_stop)
                println("Events  : $(length(ev.t_start)) × '$(ev.label)'" *
                        (has_interval ? "  [intervals]" : "  [ticks]"))
                ev
            end
        end
        for ev_list in all_ev_by_type
    ])

    xtick_step = (TIME_OUT - TIME_IN) <= 120 ? 10 : 30

    # ── Session-wide raster ───────────────────────────────────────────────────
    X, Y = simple_raster_units(spk_sorted, 0.0, TIME_IN, TIME_OUT)
    Y    = Int.(Y)

    # ── Population firing rate ────────────────────────────────────────────────
    edges    = collect(range(TIME_IN, TIME_OUT; step=BIN_SZ))
    pop_cnt  = _histcount(X, edges)
    pop_rate = Float64.(pop_cnt) ./ n_units ./ BIN_SZ
    timevec  = edges[1:end-1] .+ BIN_SZ / 2

    # ── Panel 1: event interval rectangles ───────────────────────────────────
    # One row per event type; each trial drawn as a filled rectangle
    # spanning [t_start, t_stop].  Falls back to a thin tick if no stop path.
    n_ev_rows = max(length(loaded_events), 1)

    event_plt = plot(;
        title         = ui.session_id,
        xlims         = (TIME_IN, TIME_OUT),
        ylims         = (0.5, n_ev_rows + 0.5),
        yticks        = (1:n_ev_rows, [ev.label for ev in loaded_events]),
        yflip         = false,
        legend        = false,
        grid          = false,
        bottom_margin = -2Plots.mm,
        left_margin   = 2Plots.mm,
        tickfontsize  = 7)

    for (row_idx, ev) in enumerate(loaded_events)
        col          = ev.color          # String: "black", "#2196F3", etc.
        has_interval = any(isfinite, ev.t_stop)
        for k in eachindex(ev.t_start)
            t0 = ev.t_start[k]
            t1 = (has_interval && isfinite(ev.t_stop[k])) ? ev.t_stop[k] : t0
            t0 > TIME_OUT && continue
            t1 < TIME_IN  && continue
            t0c = clamp(t0, TIME_IN, TIME_OUT)
            t1c = clamp(t1, TIME_IN, TIME_OUT)
            if has_interval && (t1c - t0c) > 0
                rect = Shape([t0c, t1c, t1c, t0c],
                             [row_idx - 0.40, row_idx - 0.40,
                              row_idx + 0.40, row_idx + 0.40])
                plot!(event_plt, rect;
                      fillcolor=col, fillalpha=0.35,
                      linecolor=col, lw=0.3, label=false)
            else
                plot!(event_plt, [t0c, t0c],
                      [row_idx - 0.40, row_idx + 0.40];
                      color=col, lw=0.8, label=false)
            end
        end
    end

    # ── Panel 2a: region column ───────────────────────────────────────────────
    # Normalised x [0,1]: colored bar from 0.55→1.0, label right-aligned at 0.50.
    reg_plt = plot(;
        xlims    = (0.0, 1.0),
        ylims    = (0.5, n_units + 0.5),
        legend   = false,
        grid     = false,
        ticks    = nothing,
        showaxis = false)

    for (reg, row_lo, row_hi) in region_runs
        col  = color_map[reg]
        y_lo = row_lo - 0.5
        y_hi = row_hi + 0.5
        plot!(reg_plt, Shape([0.55, 1.0, 1.0, 0.55], [y_lo, y_lo, y_hi, y_hi]);
              fillcolor=col, fillalpha=0.85, linecolor=:white, lw=0.4, label=false)
        annotate!(reg_plt, 0.50, (y_lo + y_hi) / 2, text(reg, :right, 7, :black))
    end
    for (_, _, row_hi) in region_runs[1:end-1]
        hline!(reg_plt, [row_hi + 0.5]; color=:grey50, lw=0.6, label=false)
    end

    # ── Panel 2b: raster ─────────────────────────────────────────────────────
    sc = scatter(X, Y;
        mc                = :black,
        ms                = 0.8,
        markerstrokewidth = 0,
        xlims             = (TIME_IN, TIME_OUT),
        ylims             = (0.5, n_units + 0.5),
        yticks            = false,
        yshowaxis         = false,
        legend            = false,
        grid              = false)

    for (reg, row_lo, row_hi) in region_runs
        col     = color_map[reg]
        row_set = Set(row_lo:row_hi)
        mask    = [yi in row_set for yi in Y]
        any(mask) || continue
        scatter!(sc, X[mask], Y[mask]; mc=col, ms=0.8, markerstrokewidth=0, label=false)
    end
    for (_, _, row_hi) in region_runs[1:end-1]
        hline!(sc, [row_hi + 0.5]; color=:grey50, lw=0.6, label=false)
    end
    xticks!(sc, collect(TIME_IN:xtick_step:TIME_OUT))

    # ── Panel 3: population firing rate ──────────────────────────────────────
    rate_plt = plot(timevec, pop_rate;
        color      = :black,
        lw         = 0.8,
        xlims      = (TIME_IN, TIME_OUT),
        xticks     = collect(TIME_IN:xtick_step:TIME_OUT),
        ylabel     = "spikes/s",
        xlabel     = "time (s)",
        legend     = false,
        top_margin = -2Plots.mm)

    # ── Combine: 3×2 grid, blanks in left column top and bottom ──────────────
    # Row-major order: blank | event_plt
    #                  reg_plt | sc
    #                  blank   | rate_plt
    blank = plot(; axis=false, ticks=nothing, legend=false, grid=false)

    plt = plot(blank, event_plt, reg_plt, sc, blank, rate_plt;
        layout = grid(3, 2; heights=[0.06, 0.80, 0.14], widths=[0.07, 0.93]),
        size   = (1200, 800))

    display(plt)
    _save(plt, "$(replace(ui.session_id, ".nwb" => ""))_overview")

end  # for ui in all_units
