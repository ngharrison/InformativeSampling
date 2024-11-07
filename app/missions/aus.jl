
using Logging: global_logger, ConsoleLogger, Info, Debug
using LinearAlgebra: norm
using DelimitedFiles: readdlm
using Statistics: cor
using Random: seed!

using InformativeSampling
using .Maps: Map, getBounds
using .Samples: Sample, MapsSampler, selectSampleLocation
using .SampleCosts: MIPT, EIGF, DistScaledEIGF, OnlyVar, DerivVar, DistScaledDerivVar, LogLikelihood
using .Missions: Mission
using .Kernels: multiKernel, mtoKernel

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: normalize, maps_dir, imgToMap

function ausMission(; seed_val=0, num_samples=30,
                    priors=Bool[1, 1, 1],
                    sampleCostType=DistScaledEIGF, kernel=multiKernel,
                    use_means=true, noise_learned=true, use_cond_pdf=false,
                    use_hyp_drop=false, weights = (; μ=1, σ=1e2, τ=1, d=1))

    # have it run around australia

    seed!(seed_val)

    file_names = [
        "vege_ave_aus.csv",
        "topo_ave_aus.csv",
        "temp_ave_aus.csv",
        "rain_ave_aus.csv"
    ]

    images = readdlm.(maps_dir .* file_names, ',')

    bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

    map0 = imgToMap(normalize(images[1]), bounds)
    sampler = MapsSampler(map0)

    prior_maps = [imgToMap(normalize(img), bounds) for img in images[2:end]]

    occupancy = imgToMap(Matrix{Bool}(reduce(.|, [isnan.(i)
                                                  for i in images])), bounds)

    ## initialize alg values
    # weights = [1e-1, 6, 5e-1, 3e-3] # mean, std, dist, prox
    # weights = (; μ=1, σ=5e3, τ=1, d=1) # others
    # weights = (; μ=1, σ=1e2, τ=1, d=1) # others
    start_locs = [[0.8, 0.6]] # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations

    # maximize minimum distance between samples
    points_sp = Vector{Float64}[]
    sampleCost = x -> occupancy(x) ? Inf : -minimum(norm(loc - x) for loc in points_sp; init=Inf)
    for _ in 1:25
        x = selectSampleLocation(sampleCost, getBounds(occupancy))
        push!(points_sp, x)
        # x, v = rand(occupancy)
        # !v && push!(points_sp, x)
    end
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    [cor(vec(map0[.!occupancy]), vec(d[.!occupancy])) for d in prior_maps]
    # scatter(vec(map0[.!occupancy]), [vec(d[.!occupancy]) for d in prior_maps], layout=3)

    noise = (value = zeros(length(sampler) + sum(priors)), learned = noise_learned)
    hyp_drop = (dropout=use_hyp_drop, start=10, num=5, threshold=0.4)

    means = (use = use_means, learned = true)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples,
        kernel,
        means,
        noise,
        use_cond_pdf,
        hyp_drop,
    )

    return mission, prior_maps
end


#* Run

# # set the logging level: Info or Debug
# global_logger(ConsoleLogger(stderr, Info))
#
# ## initialize data for mission
#
# options = (
#     kernel = multiKernel,
#     use_means = true,
#     noise_learned = true,
#     use_cond_pdf = false,
#     use_hyp_drop = false,
#     sampleCostType = DistScaledEIGF,
#     weights = (; μ=1, σ=1e2, τ=1, d=1)
# )
#
# mission, prior_maps = ausMission(; num_samples=30, priors=Bool[1,1,1], options...)
#
# vis(mission.sampler..., prior_maps...;
#     titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
#     points=first.(getfield.(mission.prior_samples, :x)))
#
# ## run search alg
# @time samples, beliefs = mission(
#     vis;
#     sleep_time=0.0
# );


#* Compilation run
global_logger(ConsoleLogger(stderr, Info))
mission, = ausMission(num_samples=4, priors=Bool[0,0,0])
mission();

#* Runs

runs = [

    # only var
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = OnlyVar
    ),

    # no means
    (
        kernel = multiKernel,
        use_means = false,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = OnlyVar
    ),

    # eigf
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = EIGF
    ),

    # # no noise
    # (
    #     kernel = multiKernel,
    #     use_means = true,
    #     noise_learned = false,
    #     use_cond_pdf = false,
    #     use_hyp_drop = false,
    #     sampleCostType = EIGF
    # ),

    # deriv var
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = DerivVar
    ),

    # dist-scaled eigf
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = DistScaledEIGF
    ),

    # # dist-scaled deriv var
    # (
    #     kernel = multiKernel,
    #     use_means = true,
    #     noise_learned = true,
    #     use_cond_pdf = false,
    #     use_hyp_drop = false,
    #     sampleCostType = DistScaledDerivVar
    # ),

    # many-to-one
    (
        kernel = mtoKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = false,
        sampleCostType = DistScaledEIGF
    ),

    # conditional likelihood
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = true,
        use_hyp_drop = false,
        sampleCostType = DistScaledEIGF
    ),

    # hypothesis dropout
    (
        kernel = multiKernel,
        use_means = true,
        noise_learned = true,
        use_cond_pdf = false,
        use_hyp_drop = true,
        sampleCostType = DistScaledEIGF
    ),


]

for options in runs


#* Pair

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .MultiQuantityGPs: outputCorMat

using .Metrics: calcMetrics
using .DataIO: save

# options = runs[3]

# # LogLikelihood
# options = (
#     kernel = multiKernel,
#     use_means = true,
#     noise_learned = true,
#     use_cond_pdf = false,
#     use_hyp_drop = false,
#     sampleCostType = LogLikelihood
# )

k = options.kernel
m = (options.use_means ? "means" : "zeromean")
n = (options.noise_learned ? "noises" : "zeronoise")
c = (options.use_cond_pdf ? "condpdf" : "fullpdf")
h = (options.use_hyp_drop ? "hypdrop" : "nodrop")
s = options.sampleCostType

dir = "new_aus/aus_$(k)_$(m)_$(n)_$(c)_$(h)_$(s)"

all_metrics = Array{Any}(undef, 2)

@time for (i, priors) in enumerate([(0,0,0), (1,1,1)])
    ## initialize data for mission
    # priors = (0,0,0)
    mission, _ = ausMission(; priors=collect(Bool, priors),
                            weights = (; μ=1, σ=5e1, τ=1, d=1),
                            options...)
    # empty!(mission.prior_samples)

    ## run search alg
    @time samples, beliefs, cors, times = mission(sleep_time=0.0);
    @debug "output correlation matrix:" outputCorMat(beliefs[end])
    # save(mission, samples, beliefs; animation=true)

    ## calculate errors
    metrics = calcMetrics(mission, samples, beliefs, times, 1)
    all_metrics[i] = metrics

    ## save outputs
    save(; file_name="$(dir)/data_$(join(priors))", mission, samples, beliefs, metrics)
end

#* Data analysis

using InformativeSampling

using InformativeSamplingUtils
using .DataIO: output_dir, output_ext

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm

maes = stack(metrics.mae for metrics in all_metrics)
mxaes = stack(metrics.mxae for metrics in all_metrics)
dists = stack(cumsum(metrics.dists) for metrics in all_metrics)
times = stack(cumsum(metrics.times) for metrics in all_metrics)

cors = all_metrics[2].cors
dets = [u.^2 for u in cors]

width, height = 1200, 800

p_cors = plot(
    hcat((c[2:end] for c in cors[1:30,:])...)',
    title="Estimated Correlation to Vegetation",
    labels=["Elevation" "Ground Temperature" "Rainfall"],
    xlabel="Sample Number",
    ylabel="Correlation",
    seriescolors=[RGB(0.4, 0.2, 0.1) RGB(0.9, 0.4, 0.0) RGB(0.1,0.3,1)],
    legend=(0.74,0.15),
    # legend=(0.74,0.45),
    # legend=(0.74,0.75),
    # legend=(0.11,0.15),
    framestyle=:box,
    marker=true,
    ylim=(-1,1),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/correlations.png")

p_errs = plot(
    maes[1:30,:],
    title="Mean Prediction Errors",
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    labels=["No Priors" "All Priors"],
    seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    ylim=(0,.3),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/errors.png")

p_max_errs = plot(
    mxaes[1:30,:],
    title="Max Prediction Errors",
    xlabel="Sample Number",
    ylabel="Max Absolute Map Error",
    labels=["No Priors" "All Priors"],
    seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    ylim=(0,1),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/max_errors.png")

p_dists = plot(
    dists[1:30,:],
    title="Distance Traveled",
    xlabel="Sample Number",
    ylabel="Cumulative Distance",
    labels=["No Priors" "All Priors"],
    seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    ylim=(0,25),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/distances.png")

times_per_sample = times[end,:]/size(times,1)
p_comp = bar(
    ["No Priors", "All Priors"],
    times_per_sample,
    xlabel="Sample Number",
    ylabel="Average Computation Time (s)",
    title="Computation Time per Sample",
    ylim=(0,3),
    seriescolors=[:black, RGB(0.1,0.7,0.2)],
    framestyle=:box,
    markers=true,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legend=nothing,
    margin=5mm,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/computation_times.png")

plot(
    times,
    title="Average Computation Time",
    xlabel="Sample Number",
    ylabel="Average Computation Time (s)",
    labels=["No Priors" "All Priors"],
    seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    # ylim=(0,25),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/computation_times_full_run.png")


#* End Runs

end
