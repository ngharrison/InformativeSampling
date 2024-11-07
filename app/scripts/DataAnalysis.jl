# script for analyzing data from missions
# can be run after Main.jl or by opening a saved file

using InformativeSampling

using .Missions: Mission
using .MultiQuantityGPs: MQGP
using .Samples: Sample

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

#* Aus
# for dir in readdir(output_dir * "new_aus", join=true)
dir = output_dir * "new_aus/aus_multiKernel_means_noises_fullpdf_nodrop_LogLikelihood"
# file_name = output_dir * "2023-08-28-16-58-29_metrics" * output_ext
file_name_s = "$dir/data_000" * output_ext
file_name_m = "$dir/data_111" * output_ext

data = load(file_name_s)
maes = data["metrics"].mae
mxaes = data["metrics"].mxae
dists = cumsum(data["metrics"].dists)
times = cumsum(data["metrics"].times)

data = load(file_name_m)
maes = [maes data["metrics"].mae]
mxaes = [mxaes data["metrics"].mxae]
dists = [dists cumsum(data["metrics"].dists)]
times = [times cumsum(data["metrics"].times)]
cors = data["metrics"].cors
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

savefig("$dir/correlations.png")

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

savefig("$dir/errors.png")

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

savefig("$dir/max_errors.png")

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

savefig("$dir/distances.png")

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

savefig("$dir/computation_times.png")

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

savefig("$dir/computation_times_full_run.png")

# end

#* Batch
for dir in readdir(output_dir * "new_syn", join=true)
# dir = output_dir * "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_DistLogEIGF"

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
    file_name = "$dir/data_$(join(p))" * output_ext

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

savefig("$dir/hypothesis_scores.png")

plot(
    err_means,
    # ribbon=err_stds,
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    title="Mean Prediction Errors",
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

savefig("$dir/errors.png")

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

savefig("$dir/max_errors.png")

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

savefig("$dir/distances.png")

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

savefig("$dir/computation_times.png")

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

savefig("$dir/computation_times_full_run.png")

# end

#* Calculating statistical significance of difference between means
using Distributions
p = 4; a = 8; b = 2; n = 18;
t = abs(err_means[p,a] - err_means[p,b])/sqrt(err_stds[p,a]^2/n + err_stds[p,b]^2/n)
cdf(TDist(n-1), t) # t-test

# calc number of samples from t-score
t^2*(err_stds[p,a]^2 + err_stds[p,b]^2)/(err_means[p,a] - err_means[p,b])^2


#* True mean coefficients

# This is old code and would need to be re-written
mission_peaks = [3,3,4,4,5,5]
cors = Array{Any, 1}(undef, length(mission_peaks))
for (i, num_peaks) in enumerate(mission_peaks)
    mission, c = synMission(; seed_val=i, num_peaks)
    cors[i] = c
end
s = [abs.(c) for c in cors]
mean(s)
std(s)

# From the map abs correlations:
#
# julia> mean(s)
# 3-element Vector{Float64}:
#  0.9999999999999996
#  0.7786774403828388
#  0.21356037266496505
#
# julia> std(s)
# 3-element Vector{Float64}:
#  2.5316980181136773e-16
#  0.01919544968657839
#  0.13245078231003993

# Determination coefficients:
#
# julia> mean(s)
# 3-element Vector{Float64}:
#  0.9999999999999991
#  0.6066456105683947
#  0.060227374218250064
#
# julia> std(s)
# 3-element Vector{Float64}:
#  5.063396036227355e-16
#  0.029810722128705513
#  0.05812149041402016
