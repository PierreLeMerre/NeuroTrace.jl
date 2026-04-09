"""
NeuroTrace test suite.

Run with:
    cd NeuroTrace.jl
    julia --project=. -e "using Pkg; Pkg.test()"

Or from the Julia REPL after `] activate .`:
    julia> using Pkg; Pkg.test()

The tests here are "unit tests" — they test individual functions in
isolation with synthetic data, so no real NWB file is needed.
"""

using Test
using NeuroTrace
using NeuroTrace.IO
using NeuroTrace.Analysis

# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic data structures
# ---------------------------------------------------------------------------

function make_test_units()
    ids = [1, 2, 3]
    spike_times = [
        [0.1, 0.5, 1.2, 1.8, 2.3],   # unit 1
        [0.2, 0.8, 1.0, 1.5],         # unit 2
        [0.05, 0.95, 1.7, 2.1, 2.8],  # unit 3
    ]
    return SpikeUnits(ids, spike_times)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "NeuroTrace" begin

    @testset "SpikeUnits struct" begin
        units = make_test_units()
        @test length(units.ids) == 3
        @test length(units.spike_times) == 3
        @test units.spike_times[1][1] ≈ 0.1
    end

    @testset "firing_rate" begin
        units = make_test_units()
        rates = Analysis.firing_rate(units; t_stop = 3.0)

        @test length(rates) == 3
        # Unit 1 has 5 spikes in 3 s → ~1.67 Hz
        @test rates[1] ≈ 5 / 3.0 atol = 1e-10
        @test all(rates .>= 0.0)
    end

    @testset "bin_spikes" begin
        units = make_test_units()
        counts, bin_times = Analysis.bin_spikes(units; bin_size = 1.0, t_stop = 3.0)

        @test size(counts, 1) == 3          # n_units rows
        @test size(counts, 2) == 3          # 3 bins of width 1 s
        @test sum(counts[1, :]) == 5        # unit 1 total spikes
        @test all(counts .>= 0)
        @test length(bin_times) == 3
    end

end
