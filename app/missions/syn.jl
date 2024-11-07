
using Logging: global_logger, ConsoleLogger, Info, Debug
using LinearAlgebra: I, norm
using Statistics: mean, cor
using Random: seed!

using InformativeSampling
using .Maps: Map, generateAxes
using .Samples: Sample, MapsSampler
using .SampleCosts: MIPT, EIGF, DistScaledEIGF, OnlyVar,
                    DerivVar, DistScaledDerivVar, LogLikelihood,
                    LogLikelihoodFull, DistLogEIGF
using .Missions: Mission
using .Kernels: multiKernel, mtoKernel
using .MultiQuantityGPs: quantityCorMat

using InformativeSamplingUtils
using .DataIO: GaussGroundTruth, Peak
using .Visualization: vis

function synMission(; seed_val=0, num_samples=30,
                    num_peaks=3, priors=Bool[1, 1, 1],
                    sampleCostType=DistScaledEIGF, kernel=multiKernel,
                    use_means=true, noise_learned=true, use_cond_pdf=false,
                    use_hyp_drop=false)

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
    m = Map(map0 .* 2*(2*rand() - 1) .+ 2*(2*rand() - 1), bounds)
    push!(prior_maps, m)

    # additive
    m = Map(map0 .+ 0.2 .* randn(size(map0)), bounds)
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

    ## initialize alg values
    # weights = (; μ=17, σ=1.5, τ=7)
    # weights = (; μ=3, σ=1, τ=.5, d=1)
    # weights = (; μ=1, σ=1e1, τ=1, d=1) # sogp
    weights = (; μ=1, σ=1e2, τ=1, d=1) # others
    # weights = (; μ=1, σ=1, τ=.1, d=1)
    start_locs = [] # starting location

    # n = (4,4) # number of samples in each dimension
    # axs_sp = range.(bounds..., n)
    # start_locs = vec(collect.(Iterators.product(axs_sp...))) # starting locations


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(bounds..., n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(sampler)), d(x))
                     for (i, d) in enumerate(prior_maps[priors])
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    @debug [cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
    # @debug [cor(vec(map0), vec(d)) for d in prior_maps]

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
# global_logger(ConsoleLogger(stderr, Debug))
#
# options = (
#     use_means=true,
#     noise_learned=true,
#     sampleCostType=OnlyVar
# )
#
# ## initialize data for mission
# mission, prior_maps = synMission(; num_samples=30, priors=Bool[1,1,1], seed_val=0, options...)
#
# vis(mission.sampler..., prior_maps...;
#     titles=["QOI", "Scaling Factor", "Additive Noise", "Random Map"],
#     points=first.(getfield.(mission.prior_samples, :x)))
#
# ## run search alg
# @time samples, beliefs, cors, times = mission(vis; sleep_time=0.0);
#
# for bm in beliefs
#     println(quantityCorMat(bm)[1,:])
# end
# println(beliefs[end].θ.σn)
# println(beliefs[end].θ.μ)

#* Compilation run
global_logger(ConsoleLogger(stderr, Info))
mission, = synMission(num_samples=4, priors=Bool[0,0,0])
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

# for options in runs


#* Batch

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Info))

using .Metrics: calcMetrics
using .DataIO: save

# options = (
#     kernel = multiKernel,
#     use_means = true,
#     noise_learned = true,
#     use_cond_pdf = false,
#     use_hyp_drop = false,
#     sampleCostType = DistScaledEIGF
# )

options = runs[1]

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

dir = "new_syn/syn_$(k)_$(m)_$(n)_$(c)_$(h)_$(s)"
mission, _ = synMission(; options...)
save(; file_name="$(dir)/mission", mission)

mission_peaks = [3,3,4,4,5,5]
num_runs = 3
missions = Array{Any, 2}(undef, (length(mission_peaks), num_runs))
metrics = Array{Any, 2}(undef, (length(mission_peaks), num_runs))
# pick all the prior data combinations
@time for priors in Iterators.product(fill(0:1,3)...)
    for (i, num_peaks) in enumerate(mission_peaks)
        ## initialize data for mission
        mission, _ = synMission(; seed_val=i, num_peaks, priors=collect(Bool, priors), options...)
        for j in 1:num_runs
            println()
            println("Priors ", priors)
            println("Peaks number ", i)
            println("Run number ", j)
            println()
            ## run search alg
            @time samples, beliefs, cors, times = mission(seed_val=j, sleep_time=0.0);
            @debug "output correlation matrix:" quantityCorMat(beliefs[end])
            ## calculate errors
            missions[i,j] = (; mission, samples, beliefs, times)
            metrics[i,j] = calcMetrics(mission, samples, beliefs, times, 1)
        end
    end
    ## save outputs
    save(; file_name="$(dir)/data_$(join(priors))", missions, metrics)
end

#* Data analysis

using InformativeSampling

using InformativeSamplingUtils
using .DataIO: output_dir, output_ext

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm

"""
Return the sum and number of elements that are not NaN. Accepts an iterable.
"""
function sumCountSkip(a; skip=isnan)
    tot = 0.0
    cnt = 0
    for x in a
        if !skip(x)
            tot += x
            cnt += 1
        end
    end
    return tot, cnt
end

"""
Calculates the mean and standard deviation across elements of an array when the
elements are arrays of arrays. Also skip any NaN values and only give a value if
2 or more elements were not NaN.
"""
function meanAndStdM(dets)
    m = [zeros(size(dets[1][1])) for _ in eachindex(dets[1])]
    s = [zeros(size(dets[1][1])) for _ in eachindex(dets[1])]
    for i in eachindex(m), j in eachindex(m[1])
        tot, cnt = sumCountSkip(v[i][j] for v in dets)
        m[i][j] = tot/cnt
        tot, cnt = sumCountSkip((v[i][j] - m[i][j])^2 for v in dets)
        s[i][j] = sqrt(tot/(cnt-1))
        if isnan(s[i][j])
            m[i][j] = NaN
        end
    end
    return m, s
end


p_err = Vector{Any}(undef, 8)
p_cor = Vector{Any}(undef, 8)

err_means = zeros(30, 8)
err_stds = zeros(30, 8)

max_err_means = zeros(30, 8)
max_err_stds = zeros(30, 8)

det_means = Vector{Any}(undef, 8)
det_stds = Vector{Any}(undef, 8)

dist_means = zeros(30, 8)
dist_stds = zeros(30, 8)

time_means = zeros(30, 8)
time_stds = zeros(30, 8)

priors = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1)
]
chars = "HML"
colors = [(1,0,0), (0,1,0), (0,0,1)]

for (i, p) in enumerate(priors)
    file_name = output_dir * "$dir/data_$(join(p))" * output_ext

    data = load(file_name)
    maes = [run.mae for run in data["metrics"]]
    mxaes = [run.mxae for run in data["metrics"]]
    cors = [run.cors for run in data["metrics"]]
    dets = [[v.^2 for v in u] for u in cors]
    dists = [cumsum(run.dists) for run in data["metrics"]]
    times = [cumsum(run.times) for run in data["metrics"]]

    err_means[:,i] .= mean(maes)
    err_stds[:,i] .= std(maes)

    max_err_means[:,i] .= mean(mxaes)
    max_err_stds[:,i] .= std(mxaes)

    det_means[i], det_stds[i] = meanAndStdM(dets)

    dist_means[:,i] .= mean(dists)
    dist_stds[:,i] .= std(dists)

    time_means[:,i] .= mean(times)
    time_stds[:,i] .= std(times)

    plt = plot(hcat((c[2:end] for c in det_means[i])...)',
               ribbon=hcat((c[2:end] for c in det_stds[i])...)',
               framestyle=:box,
               labels=false,
               markers=true,
               left_margin=(i∈(2,4,6,8) ? 5mm : 0mm),
               bottom_margin=(i∈(7,8) ? 5mm : 0mm),
               xticks=(i∈(7,8) ? (0:5:30) : false),
               yticks=(i∈(2,4,6,8) ? (0:0.5:1) : false),
               xlabel=(i∈(7,8) ? "Sample Number" : ""),
               # ylabel=(i∈(2,4,6,8) ? "\$ \\mathbf{\\rho^2} \$" : ""),
               seriescolors=[(RGB((c.*0.8)...) for (i, c) in zip(p, colors) if i==1)...;;])
    p_cor[i] = plt
end

width, height = 1200, 800

p_cor[1] = plot([2 2 2],
                framestyle=:none,
                legend=(0.4, 0.7),
                linewidth=4,
                xticks=false,
                yticks=false,
                markers=false,
                legendfontsize=16,
                labels=[" High" " Medium" " Low"],
                seriescolors=[(RGB((c.*0.8)...) for c in colors)...;;])
plot(
    p_cor...,
    plot_title="Hypothesis Scores",
    plot_titlefontsize=24,
    ylim=(0,1),
    grid=false,
    tickfontsize=15,
    labelfontsize=20,
    markersize=6,
    # legendcolumns=3,
    fg_legend=:transparent,
    layout=grid(4,2),
    # layout=@layout([grid(3,2){0.75h}; _ a{0.5w} b]),
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/hypothesis_scores.png")

plot(
    err_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    title="Prediction Errors",
    ylim=(0,.4),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
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

plot(
    max_err_means,
    # ribbon=max_err_stds,
    xlabel="Sample Number",
    ylabel="Max Absolute Map Error",
    title="Max Prediction Errors",
    ylim=(0,1.2),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
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

plot(
    dist_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Cumulative Distance",
    title="Distance Traveled",
    ylim=(0,25),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    legend=:topleft,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/distances.png")

idxs = [1,3,5,2,7,4,6,8]
times_per_sample = time_means[end,idxs]/size(time_means,1)
bar(
    replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors][idxs], ""=>"none"),
    times_per_sample,
    xlabel="Sample Number",
    ylabel="Average Computation Time (s)",
    title="Computation Time per Sample",
    ylim=(0,2.25),
    seriescolors=[RGB((p.*0.8)...) for p in priors][idxs],
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
    time_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Cumulative Computation Time (s)",
    title="Cumulative Computation Time",
    # ylim=(0,25),
    seriescolors=[(RGB((p.*0.8)...) for p in priors)...;;],
    labels=[replace([join(c for (p, c) in zip(p, chars) if p==1) for p in priors], ""=>"none")...;;],
    framestyle=:box,
    markers=true,
    legendcolumns=2, # OR layout=2,
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    legend=:topleft,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(output_dir * "$dir/computation_times_full_run.png")



#* End Runs

# end
