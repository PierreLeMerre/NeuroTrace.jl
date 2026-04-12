# ============================================================================
#  brain3d.jl — 3D brain visualization using GLMakie + Allen CCF meshes
#
#  This file is NOT wrapped in a module and is NOT included by NeuroTrace.jl.
#  It is loaded on-demand by scripts/Brain3D.jl via:
#
#      include(joinpath(@__DIR__, "..", "src", "viz", "brain3d.jl"))
#
#  Requires (loaded by the calling script before include):
#      using GLMakie, FileIO, Colors
#      using NeuroTrace.IO   # for RegionAtlas, ProbeInfo, region_color_map
#
#  Keeping it separate from the main package means GLMakie is not pulled into
#  every Julia session — only when doing 3D visualization.
# ============================================================================

import Logging  # used to silence the benign .mtl-not-found warning from MeshIO

# ---------------------------------------------------------------------------
# Default mesh directory (bundled with the package)
# ---------------------------------------------------------------------------

const _MESH_DIR = joinpath(@__DIR__, "..", "..", "assets", "structure_meshes")

# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

"""
    load_region_mesh(atlas, acronym, mesh_dir) -> (mesh, hex_color)

Load the Allen CCF `.obj` mesh for `acronym`.

Looks up the integer region ID from `atlas`, then loads
`{mesh_dir}/{id}.obj` via FileIO.  Returns the GeometryBasics mesh object
and the hex color string (with leading `#`) from the atlas.

# Example
```julia
mesh_obj, color = load_region_mesh(atlas, "PL", cfg.mesh_path)
```
"""
function load_region_mesh(atlas, acronym::String, mesh_dir::String)
    node = get(atlas.by_acronym, acronym, nothing)
    isnothing(node) && error("Acronym '$acronym' not found in atlas")

    obj_path = joinpath(mesh_dir, "$(node.id).obj")
    isfile(obj_path) || error("Mesh file not found: $obj_path\n" *
                              "Check that mesh_path in config.toml points to " *
                              "the folder containing the .obj files.")

    # Silence the benign "no .mtl file found" warning from MeshIO —
    # the Allen .obj files reference material files that were never shipped.
    # We supply colors programmatically so the missing .mtl has no effect.
    mesh_obj = Logging.with_logger(Logging.NullLogger()) do
        FileIO.load(obj_path)
    end
    hex_color = isempty(node.color) ? "#808080" : "#" * node.color
    return mesh_obj, hex_color
end

# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

"""
    brain3d(regions, atlas, mesh_dir;
            alpha         = 0.3,
            custom_colors = String[],
            azimuth       = -0.5π,
            elevation     = -0.3π,
            fig_size      = (1200, 800)) -> (fig, ax)

Create a 3D GLMakie figure populated with transparent Allen CCF region meshes.

Each region in `regions` is loaded from `mesh_dir` and rendered with its
atlas color (or a custom color if `custom_colors` is non-empty — same cycling
logic as the 2D scripts).  Returns the `Figure` and `Axis3` so the caller
can add probes, neurons, or further annotations.

# Arguments
- `regions`       – vector of Allen acronyms to render (e.g. `cfg.regions`).
- `atlas`         – `RegionAtlas` from `load_atlas()`.
- `mesh_dir`      – path to folder containing `{id}.obj` files.
- `alpha`         – mesh transparency (0 = invisible, 1 = opaque).
- `custom_colors` – if non-empty, override atlas colors with this palette
                    (cycled over sorted unique acronyms, same as heatmap).
- `azimuth`       – camera azimuth angle in radians.
- `elevation`     – camera elevation angle in radians.
- `fig_size`      – figure pixel dimensions `(width, height)`.

# Example
```julia
fig, ax = brain3d(cfg.regions, atlas, cfg.mesh_path;
                  alpha=cfg.brain_alpha,
                  custom_colors=cfg.region_colors)
```
"""
function brain3d(regions::Vector{String},
                 atlas,
                 mesh_dir::String;
                 alpha::Float64         = 0.3,
                 custom_colors::Vector{String} = String[],
                 azimuth::Float64       = -0.5π,
                 elevation::Float64     = -0.3π,
                 fig_size::Tuple{Int,Int} = (1200, 800))

    color_map = IO.region_color_map(atlas, regions; custom_colors = custom_colors)

    fig = Figure(size = fig_size)
    ax  = Axis3(fig[1, 1]; aspect = :data, azimuth = azimuth, elevation = elevation)

    # Always render the whole-brain outline first so it sits behind all regions.
    # Uses the Allen atlas color for "root" (white/gray) at very low alpha.
    root_mesh, root_hex = load_region_mesh(atlas, "root", mesh_dir)
    mesh!(ax, root_mesh; color = parse(Colorant, root_hex), transparency = true, alpha = 0.1)
    println("  rendered: root (brain outline)")

    for region in regions
        mesh_obj, _ = load_region_mesh(atlas, region, mesh_dir)
        c = parse(Colorant, color_map[region])
        mesh!(ax, mesh_obj; color = c, transparency = true, alpha = alpha)
        println("  rendered: $region")
    end

    hidedecorations!(ax)
    hidespines!(ax)

    return fig, ax
end

# ---------------------------------------------------------------------------
# Probe track overlay
# ---------------------------------------------------------------------------

"""
    add_probes!(ax, probes;
                color     = :black,
                linewidth = 5) -> nothing

Draw Neuropixels probe tracks on a 3D `Axis3`.

Each probe is rendered as a straight line from its shallowest to deepest
electrode site.  `ProbeInfo.xyz` is already in CCF µm (the loader applies
the AP/DV/ML → CCF µm transform internally), so no further scaling is needed.

# Arguments
- `ax`        – the `Axis3` returned by `brain3d`.
- `probes`    – `Vector{ProbeInfo}` from `load_probes`.
- `color`     – line color (any Makie-compatible color spec).
- `linewidth` – line thickness in pixels.

# Example
```julia
add_probes!(ax, probes; color = :black, linewidth = 5)
```
"""
function add_probes!(ax,
                     probes::Vector;
                     color      = :black,
                     linewidth::Int = 5)

    for probe in probes
        xyz_um = probe.xyz   # already in CCF µm

        # Endpoints: shallowest and deepest site (sorted by DV = column 2)
        order = sortperm(xyz_um[:, 2])
        top   = xyz_um[order[1],   :]
        bot   = xyz_um[order[end], :]

        # CCF axes: x=AP, y=DV, z=ML  (same orientation as the .obj meshes)
        lines!(ax,
               [top[1], bot[1]],
               [top[2], bot[2]],
               [top[3], bot[3]];
               color       = color,
               linewidth   = linewidth,
               transparency = true)
    end
end
