
using LinearAlgebra: I, norm
using Statistics: mean, cor
using Random: seed!

using InformativeSampling
using .Maps: Map, generateAxes
using .Samples: Sample, MapsSampler
using .SampleCosts: EIGF
using .Missions: Mission

using InformativeSamplingUtils
using .DataIO: GaussGroundTruth, Peak
using .Visualization: vis

function synMission(; seed_val=0, num_samples=30, num_peaks=3, priors=Bool[1,1,1])
    seed!(seed_val) # make random values deterministic

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    ## initialize ground truth

    # synthetic
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower
        Σ = 0.02*(rand()+0.5)*mean(bounds.upper-bounds.lower)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    ggt = GaussGroundTruth(peaks)
    _, points = generateAxes(occupancy)
    mat = ggt(points)
    map0 = Map(mat./maximum(mat), bounds)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # multiplicative
    m = Map(abs.(map0 .* randn()), bounds)
    push!(prior_maps, m)

    # additive
    m = Map(abs.(map0 .+ 0.2 .* randn(size(map0))), bounds)
    push!(prior_maps, m)

    # # both
    # push!(prior_maps, Map(abs.(map0 .* randn() + 0.1 .* randn(size(map0))), bounds))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(map0,1),t) map0[:,1:end-t]]) # shift

    # random peaks
    peaks = map(1:num_peaks) do _
        μ = rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower
        Σ = 0.02*(rand()+0.5)*mean(bounds.upper-bounds.lower)^2*I
        h = rand()
        Peak(μ, Σ, h)
    end
    tggt = GaussGroundTruth(peaks)
    tmat = tggt(points)
    m = Map(tmat./maximum(tmat), bounds)
    push!(prior_maps, m)

    # # purely random values
    # m = Map(rand(size(map0)...), bounds)
    # push!(prior_maps, m)

    sampler = MapsSampler(map0)

    sampleCostType = EIGF

    ## initialize alg values
    weights = (; μ=1, σ=1e2, τ=1, d=0) # others
    start_locs = [[1.0, 0.0]] # starting location

    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(bounds..., n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    noise = (value=0.0, learned=true)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples,
        noise
    )

    return mission, prior_maps

end


#* Run

## initialize data for mission
mission, prior_maps = synMission(num_samples=5)

vis(mission.sampler..., prior_maps...;
    titles=["QOI", "Scaling Factor", "Additive Noise", "Random Map"],
    points=first.(getfield.(mission.prior_samples, :x)))

## run search alg
@time samples, beliefs = mission(
    vis;
    sleep_time=0.0
);
