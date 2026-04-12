# ============================================================================
#  atlas.jl — Allen Brain Atlas region tree
#  Included inside module IO (from nwb_reader.jl).
#  Provides region hierarchy, descendant expansion, display labels, and colors.
# ============================================================================

const _ATLAS_CSV = joinpath(@__DIR__, "..", "..", "assets",
                             "structure_tree_safe_2017.csv")

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

"""
    RegionNode

One row from the Allen Brain Atlas structure tree.

# Fields
- `id`        – unique integer ID used in `structure_id_path`.
- `acronym`   – short name (e.g. `"ACA"`, `"ACA1"`).
- `color`     – 6-character hex string without `#` (e.g. `"268F45"`).
- `path`      – slash-delimited ancestor path, e.g. `"/997/8/.../32/"`.
- `parent_id` – ID of the immediate parent (0 for root).
"""
struct RegionNode
    id        :: Int
    acronym   :: String
    color     :: String
    path      :: String
    parent_id :: Int
end

"""
    RegionAtlas

Parsed Allen Brain Atlas structure tree, indexed by acronym and by integer ID.
Construct with `load_atlas()`.
"""
struct RegionAtlas
    by_acronym :: Dict{String, RegionNode}
    by_id      :: Dict{Int,    RegionNode}
end

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

"""
    load_atlas(csv_path = <bundled>) -> RegionAtlas

Parse the Allen Brain Atlas structure-tree CSV and return a `RegionAtlas`.
With no argument the bundled `assets/structure_tree_safe_2017.csv` is used.

# Example
```julia
atlas = load_atlas()
atlas = load_atlas("/path/to/custom_atlas.csv")
```
"""
function load_atlas(csv_path::AbstractString = _ATLAS_CSV)::RegionAtlas
    isfile(csv_path) || error("Atlas CSV not found: $csv_path")

    by_acronym = Dict{String, RegionNode}()
    by_id      = Dict{Int,    RegionNode}()

    open(csv_path, "r") do io
        readline(io)   # skip header
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = _parse_csv_row(line)
            length(fields) < 14 && continue

            id_str  = strip(fields[1])
            acr     = strip(fields[4])
            pid_str = strip(fields[9])
            path    = strip(fields[13])
            color   = strip(fields[14])

            (isempty(id_str) || isempty(acr)) && continue
            id = tryparse(Int, id_str)
            isnothing(id) && continue
            pid = something(tryparse(Int, pid_str), 0)

            node = RegionNode(id, acr, color, path, pid)
            by_acronym[acr] = node
            by_id[id]       = node
        end
    end

    println("Atlas loaded: $(length(by_acronym)) regions")
    return RegionAtlas(by_acronym, by_id)
end

# ---------------------------------------------------------------------------
# Hierarchy queries
# ---------------------------------------------------------------------------

"""
    descendants(atlas, acronym) -> Vector{String}

Return all acronyms that are descendants of `acronym` **including itself**,
at any depth.  Uses `structure_id_path` prefix matching — the same approach
the Allen Institute uses internally.

If `acronym` is not found in the atlas it is returned as-is (leaf behaviour),
so calling code never has to special-case missing regions.

# Example
```julia
descendants(atlas, "ACA")   # → ["ACA", "ACA1", "ACA2/3", ...]
descendants(atlas, "ACA1")  # → ["ACA1"]         (leaf)
```
"""
function descendants(atlas::RegionAtlas,
                     acronym::AbstractString)::Vector{String}
    node = get(atlas.by_acronym, acronym, nothing)
    isnothing(node) && return [String(acronym)]
    prefix = node.path
    return [r.acronym for r in values(atlas.by_acronym)
            if startswith(r.path, prefix)]
end

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

"""
    region_display_labels(atlas, raw_regions, user_parents) -> Vector{String}

Map each raw region string (as stored in the NWB file) to a *display label*:

- If `user_parents` is **empty** → return `raw_regions` unchanged (all regions,
  raw labels).
- Otherwise each raw region is matched against the descendant sets of the
  user-specified parent acronyms.  The first matching parent becomes the
  display label.  Raw regions that don't match any parent keep their raw label.

# Example
```julia
# config.toml: regions = ["ACA", "SC"]
disp = region_display_labels(atlas, ["ACA1","ACA2/3","SCdg"], ["ACA","SC"])
# → ["ACA", "ACA", "SC"]
```
"""
function region_display_labels(atlas::RegionAtlas,
                               raw_regions::Vector{String},
                               user_parents::Vector{String})::Vector{String}
    isempty(user_parents) && return copy(raw_regions)

    # Build leaf → parent lookup  (one pass per parent)
    leaf_to_parent = Dict{String, String}()
    for parent in user_parents
        for leaf in descendants(atlas, parent)
            leaf_to_parent[leaf] = parent
        end
    end

    return [get(leaf_to_parent, r, r) for r in raw_regions]
end

"""
    region_color_map(atlas, acronyms; custom_colors = String[]) -> Dict{String, String}

Return `Dict(acronym => "#RRGGBB")` for each unique acronym.

- If `custom_colors` is **empty** (default): colors come from the Allen atlas.
  Acronyms not found in the atlas default to `"#808080"` (grey).
- If `custom_colors` is **non-empty**: those hex strings are assigned in order to
  the unique acronyms (sorted for reproducibility) and **cycled** when there are
  more regions than colors.

# Example
```julia
# Allen atlas colors
color_map = region_color_map(atlas, pfc_regions)

# Custom palette, cycles if more than 3 regions
color_map = region_color_map(atlas, pfc_regions;
                             custom_colors = ["#E64B35", "#4DBBD5", "#00A087"])
```
"""
function region_color_map(atlas::RegionAtlas,
                          acronyms;
                          custom_colors::Vector{String} = String[])::Dict{String, String}
    uniq = unique(acronyms)
    if isempty(custom_colors)
        return Dict(acr => begin
                        node = get(atlas.by_acronym, acr, nothing)
                        (isnothing(node) || isempty(node.color)) ? "#808080" : "#" * node.color
                    end
                    for acr in uniq)
    else
        return Dict(acr => custom_colors[mod1(i, length(custom_colors))]
                    for (i, acr) in enumerate(sort(uniq)))
    end
end

# ---------------------------------------------------------------------------
# filter_units overload (atlas-aware region expansion)
# ---------------------------------------------------------------------------

"""
    filter_units(units, cfg, atlas) -> Vector{UnitInfo}

Atlas-aware version of `filter_units`.  The `regions` list in `cfg` is treated
as **parent acronyms**: every descendant in the atlas is automatically included
and grouped under the parent label for display.

If `cfg.regions` is empty, only the firing-rate filter is applied (all regions
are kept).

# Example
```julia
atlas = load_atlas()
units = filter_units(load_units(cfg.data_path), cfg, atlas)
```
"""
function filter_units(units::Vector{UnitInfo},
                      cfg::NTConfig,
                      atlas::RegionAtlas)::Vector{UnitInfo}
    if isempty(cfg.regions)
        return filter_units(units; min_firing_rate = cfg.min_firing_rate)
    end

    # Expand each parent to its full descendant set
    expanded = Set{String}()
    for parent in cfg.regions
        union!(expanded, descendants(atlas, parent))
    end

    return filter_units(units;
                        min_firing_rate = cfg.min_firing_rate,
                        regions         = collect(expanded))
end

# ---------------------------------------------------------------------------
# Minimal quoted-CSV parser  (stdlib only — no CSV.jl needed)
# ---------------------------------------------------------------------------

function _parse_csv_row(line::AbstractString)::Vector{String}
    fields = String[]
    i = firstindex(line)
    n = lastindex(line)

    while i <= n
        if line[i] == '"'
            # Quoted field — scan to closing quote
            buf = Char[]
            j   = nextind(line, i)
            while j <= n
                ch = line[j]
                ch == '"' && (j = nextind(line, j); break)
                push!(buf, ch)
                j = nextind(line, j)
            end
            push!(fields, String(buf))
            i = j
            i <= n && line[i] == ',' && (i = nextind(line, i))
        else
            j = findnext(',', line, i)
            if isnothing(j)
                push!(fields, line[i:n])
                break
            else
                push!(fields, line[i:prevind(line, j)])
                i = nextind(line, j)
            end
        end
    end
    return fields
end
