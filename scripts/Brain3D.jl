# ============================================================================
#  Brain3D.jl — 3D brain visualization: Allen CCF meshes + probe tracks
#
#  Uses its own isolated environment (scripts/brain3d_env/) so that GLMakie
#  does not pollute the 2D-plotting scripts.
#
#  First-time setup (run once):
#      cd scripts/brain3d_env
#      julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
#  Then run normally:
#      julia scripts/Brain3D.jl
#
#  Electrode coordinates (AP/DV/ML in mm) are read from each NWB file and
#  converted to CCF µm automatically — no manual scale factor needed.
# ============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "brain3d_env"))
Pkg.instantiate()   # no-op if already installed, safe to leave in

# 3D packages — use `import` (not `using`) for FileIO so that FileIO.load()
# stays qualified and does not clash with IO.load.
using GLMakie, Colors
import FileIO        # FileIO.load() used explicitly in brain3d.jl
using HDF5, Statistics, TOML

GLMakie.activate!(inline = false)

# ---------------------------------------------------------------------------
# Pull in NeuroTrace IO directly (no package activation needed — just include).
# nwb_reader.jl defines `module IO` at the Main level.
# We do NOT `using .IO` — that would clash if NeuroTrace.IO was already loaded
# in this REPL session (e.g. by running another script first).
# All IO calls below are explicitly qualified: IO.load_config(...), etc.
# atlas.jl is included inside IO automatically via the include() at its bottom.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "src", "io", "nwb_reader.jl"))

# 3D visualization helpers (requires GLMakie + Colors + FileIO already loaded)
include(joinpath(@__DIR__, "..", "src", "viz", "brain3d.jl"))

# ── MAIN ─────────────────────────────────────────────────────────────────────
# Wrapped in a function to sidestep Julia 1.12 world-age restrictions on
# top-level calls to functions defined via include().
# Called via Base.invokelatest so Julia uses the latest method world.

function main()
    cfg_path    = joinpath(@__DIR__, "config.toml")
    cfg         = IO.load_config(cfg_path)
    atlas       = IO.load_atlas()

    # Read 3D-specific keys directly from the raw TOML (not in NTConfig struct)
    cfg_raw     = TOML.parsefile(cfg_path)
    mesh_dir    = get(cfg_raw, "mesh_path",
                      joinpath(@__DIR__, "..", "assets", "structure_meshes"))
    brain_alpha = Float64(get(cfg_raw, "brain_alpha", 0.15))

    println("Mesh directory : $mesh_dir")
    println("Regions        : $(cfg.regions)")

    # ── Probe tracks ─────────────────────────────────────────────────────────
    probes = IO.load_probes(cfg.data_path)
    println("Loaded $(length(probes)) probe(s) across all sessions")

    # ── 3D figure ─────────────────────────────────────────────────────────────
    # invokelatest needed for functions defined by include() in Julia 1.12
    fig, ax = Base.invokelatest(brain3d, cfg.regions, atlas, mesh_dir;
                                alpha         = brain_alpha,
                                custom_colors = cfg.region_colors,
                                azimuth       = -0.5π,
                                elevation     = -0.3π,
                                fig_size      = (1200, 800))

    Base.invokelatest(add_probes!, ax, probes; color = :black, linewidth = 5)

    display(fig)
end

Base.invokelatest(main)
