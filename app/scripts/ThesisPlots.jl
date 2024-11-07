#* packages and functions

using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples, .ROSInterface, .Kernels, .SampleCosts

using InformativeSamplingUtils
using .DataIO

using LinearAlgebra: norm
using Statistics: mean, std, cor, quantile
using FileIO: load
using Plots
using Plots: mm
using Printf
using DelimitedFiles: readdlm

# r is the range, contains the min and max values
function createColorbarTicks((l, h))
    ticks = [ceil(l, sigdigits=3),
        round((h - l) / 2 + l, sigdigits=3),
        floor(h, sigdigits=3)]
    return ticks, [@sprintf("%.1g", x) for x in ticks]
end

"""
Takes a two-argument function `f(x, key)` that returns true or false for each
element-key pair, an iterator of elements to group, and a set of keys.
"""
function group(f, a, keys)
    result = Dict(key=>eltype(a)[] for key in keys)
    for x in a, key in keys
        if f(x, key)
            push!(result[key], x)
        end
    end
    return result
end

"""
Takes a one-argument function `f(x)`, and an iterator of elements to group `a`.
"""
function group(f, a)
    val_type = eltype(a)
    # this next line actually makes it slower, requires compilation every time
    # key_type = promote_type(Base.return_types(f, (eltype(a),))...)
    groups = Dict{Any, Vector{val_type}}()
    for x in a
        key = f(x)
        if haskey(groups, key)
            push!(groups[key], x)
        else
            groups[key] = [x]
        end
    end
    return groups
end

num_samples = 30

# NOTE: crop of depth image went from 5% to 15% from each side
# NOTE: ground depth tuning went from 1082 to 1012 on Apr 23 2024 (diff = 70)
# NOTE: ground depth tuning went from 1012 to 962 for new bracket on Jun 7 2024

depth_diff = 70 # remove this much from each height


#* load mission
dir = "new_syn/syn_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_000" * output_ext

mkpath(output_dir * "thesis/syn_000")

data = load(file_name)
maes = data["metrics"][end-2].mae
mxaes = data["metrics"][end-2].mxae
dists = cumsum(data["metrics"][end-2].dists)
times = cumsum(data["metrics"][end-2].times)

mission = data["missions"][end-2].mission
samples = data["missions"][end-2].samples
beliefs = data["missions"][end-2].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for bm in @view beliefs[5:5:end]
    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(vec(points), 1))
    mask = vec(.! mission.occupancy)

    global pred_range = (min(minimum(pred_map[mask]), pred_range[1]),
                         max(maximum(pred_map[mask]), pred_range[2]))
    global err_range = (min(minimum(err_map[mask]), err_range[1]),
                        max(maximum(err_map[mask]), err_range[2]))
end

err_range = (0.0, err_range[2])

pred_ticks = createColorbarTicks(pred_range)
err_ticks = createColorbarTicks(err_range)

#* metrics

gr()

width, height = 1200, 800

p = plot(
    [maes mxaes],
    title="Prediction Errors",
    xlabel="Sample Number",
    ylabel="Prediction Error",
    labels=["Mean Absolute Error" "Max Absolute Error"],
    # seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    ylim=(0, 1.2),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/syn_000/errors.png")

p = plot(
    dists,
    title="Distance Traveled",
    xlabel="Sample Number",
    ylabel="Cumulative Distance",
    labels=false,
    seriescolors=[:black RGB(0.1, 0.7, 0.2)],
    framestyle=:box,
    marker=true,
    # ylim=(0,.5),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/syn_000/distances.png")

#* hyperparameters

ls[[15,20]]

gr()

width, height = 1200, 800

ls = map(beliefs) do bm
    abs(bm.θ.ℓ)
end

ss = map(beliefs) do bm
    abs(only(bm.θ.σ))
end

ns = map(beliefs) do bm
    abs(only(bm.θ.σn))
end

p = plot(
    ls,
    title="Length Scales",
    xlabel="Sample Number",
    ylabel="Length Scale",
    labels=false,
    # seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    ylim=(0, 2),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/syn_000/length_scales.png")

p = plot(
    ss,
    title="Signal Standard Deviations",
    xlabel="Sample Number",
    ylabel="Signal Standard Deviation",
    labels=false,
    # seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    # ylim=(0, 1.2),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/syn_000/signal_std.png")

p = plot(
    ns,
    title="Noise Standard Deviations",
    xlabel="Sample Number",
    ylabel="Noise Standard Deviation",
    labels=false,
    # seriescolors=[:black RGB(0.1,0.7,0.2)],
    framestyle=:box,
    marker=true,
    # ylim=(0, 1.2),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# p |> display
savefig(output_dir * "thesis/syn_000/noise_std.png")


#* ground truth

pyplot()

axs, _ = generateAxes(mission.sampler[1])

p = heatmap(axs..., mission.sampler[1]';
    title="Vegetation Ground Truth",
    framestyle=:none,
    titlefontsize=21,
    colorbar_tickfontsize=17,
)
display(p)
savefig(output_dir * "thesis/syn_000/ground_truth.png")


#* frames

pyplot()

mkpath(output_dir * "thesis/syn_000/belief_frames")
mkpath(output_dir * "thesis/syn_000/uncertainty_frames")

for (i, bm) in enumerate(beliefs)
axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

pred_map[occ] .= NaN
err_map[occ] .= NaN

p1 = heatmap(axs..., pred_map';
    title="Predicted Values",
    framestyle=:none,
    titlefontsize=21,
    colorbar_tickfontsize=17,
    clim=(0, 1),
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
savefig(output_dir * "thesis/syn_000/belief_frames/$(lpad(i,2,'0')).png")

p2 = heatmap(axs..., err_map';
    title="Uncertainties",
    framestyle=:none,
    titlefontsize=21,
    colorbar_tickfontsize=17,
    # clim=err_range,
    # colorbar_ticks=err_ticks,
)
scatter!(x1[1:i], x2[1:i];
    label=false,
    color=:green,
    markersize=8)
savefig(output_dir * "thesis/syn_000/uncertainty_frames/$(lpad(i,2,'0')).png")
end

#* full run comparison

pyplot()

i=5
plots = map(5:5:length(beliefs)) do i
    axs, _ = generateAxes(mission.sampler[1])

    gt_title = i == 5 ? "Ground Truth" : ""
    p0 = heatmap(axs..., mission.sampler[1]';
        title=gt_title,
        ylabel="$i Samples",
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        labelfontsize=17,
    )

    bm = beliefs[i]

    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(points, 1))

    pred_map[occ] .= NaN
    err_map[occ] .= NaN

    pred_title = i == 5 ? "Predicted Values" : ""
    p1 = heatmap(axs..., pred_map';
        title=pred_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=(0, 1),
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=6)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    err_title = i == 5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
        # colorbar_ticks=err_ticks,
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=6)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    sampleCost = mission.sampleCostType(
        occ, samples[1:i], bm, quantities, mission.weights
    )
    obj_map = -sampleCost.(points)

    lessNotInf = x -> x === -Inf ? Inf : x
    greaterNotInf = x -> x === Inf ? -Inf : x

    obj_range = (0.0, maximum(greaterNotInf, obj_map))
    obj_ticks = createColorbarTicks(obj_range)

    obj_title = i == 5 ? "Sample Utility" : ""
    p3 = heatmap(axs..., obj_map';
        title=obj_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=obj_range,
        colorbar_ticks=obj_ticks,
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=6)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    return p0, p1, p2, p3
end

p = plot(Iterators.flatten(plots)...,
    layout=(6, 4),
    size=(1100, 1300))

savefig(output_dir * "thesis/syn_000/full_run_comparison.png")

#* distance scaling for objective function

using Plots

# gradually-delayed distance scaling

n_scale = n -> 2/(1 + exp(1 - n)) - 1
plot(
    n_scale, 1, 8,
    title="Sample Scaling",
    xlabel="Sample Number",
    ylabel="Scaling Factor",
    framestyle=:box,
    legend=false,
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    linewidth=3,
    color=2,
    size=(700,500),
    dpi=150
)|>display

savefig(output_dir * "thesis/sample_scaling.png")

d_scale = d -> 1/(1 + 1*d^2)
plot(
    d_scale, 0, 2,
    title="Distance Scaling",
    xlabel="Distance",
    ylabel="Scaling Factor",
    framestyle=:box,
    legend=false,
    ylim=(-0.03,1.03),
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    linewidth=3,
    color=2,
    size=(700,500),
    dpi=150
)|>display

savefig(output_dir * "thesis/distance_scaling.png")

full_scale = (n,d) -> 1/(1 + n_scale(n)*d^2)
n, d = 1:8, 0:0.01:2
# z = Surface((x,y)->rosenbrock([x,y]), x, y)
surface(
    n, d, full_scale,
    xlabel="Samples",
    ylabel="Distance",
    zlabel="Scaling",
    camera = (60, 20)
)|>display

#* distance to uncertainty

gr()

uncertainty = d -> σf^2*(1 - σf^2*exp(-(d/σl)^2)/(σf^2+σn^2))
error = d -> (1 - σf^2*exp(-(d/σl)^2/2)/(σf^2+σn^2))^2*(y - μf)^2
combined = d -> error(d) + α*uncertainty(d)
d_scale = d -> 1/(1 + (d/d0)^2)
utility = d -> combined(d)*d_scale(d)

σf = √.5+.05; σn = 0; d0 = 3; σl = 1
y = √.5-.05; μf = 0; α = 1
plot(
    [uncertainty, error, combined], 0, 2d0,
    title="Uncertainty vs Distance",
    xlabel="Distance/length-scale",
    ylabel="Uncertainty",
    labels=["GP posterior variance" "nearest-sample difference" "combined uncertainty"],
    framestyle=:box,
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    legend=:bottomright,
    legendfontsize=15,
    linewidth=3,
    color=[4 5 1],
    size=(700,500),
    dpi=150
)|>display

savefig(output_dir * "thesis/uncertainty_vs_distance.png")

# uncertainty and change in uncertainty vs distance
p = plot([d->(1-exp(-d^2)),
          d->2d*exp(-d^2),
          d->(1-exp(-d^2))/(1+(d/10)^2)], 0, 5)
display(p)

# # uncertainty and change in uncertainty vs distance for log
# p = plot([d->log(1-exp(-d^2)),
#           d->log(1-exp(-d^2))/(1+(d/10)^2)], 0, 5)
# display(p)
#
# d->2d*exp(-d^2)/(1-exp(-d^2))

σf = √.5+.05; σn = 0; d0 = 3; σl = 1
y = √.5-.05; μf = 0; α = 1
plot(
    [combined, d_scale, utility], 0, 2d0,
    title="Effect of Distance Scaling on Utility",
    xlabel="Distance/length-scale",
    ylabel="Sample Utility",
    framestyle=:box,
    labels=["usual utility" "distance scaling" "scaled utility"],
    titlefontsize=19,
    tickfontsize=13,
    labelfontsize=17,
    legend=(.66,.77),
    legendfontsize=15,
    linewidth=3,
    size=(700,500),
    dpi=150
)
vline!(
    [d0],
    color=:grey,
    line=:dash,
    label="region width"
)|>display

savefig(output_dir * "thesis/distance_scaled_utility.png")

#* load mission
dir = "new_syn/syn_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
maes = data["metrics"][1].mae
mxaes = data["metrics"][1].mxae
dists = cumsum(data["metrics"][1].dists)
times = cumsum(data["metrics"][1].times)

mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for bm in beliefs
axs, points = generateAxes(occ)
pred, err = bm(tuple.(vec(points), 1))
global pred_range = (min(minimum(pred), pred_range[1]), max(maximum(pred), pred_range[2]))
global err_range = (min(minimum(err), err_range[1]), max(maximum(err), err_range[2]))
end

err_range = (0.0, err_range[2])

using .SampleCosts, .Samples, .Maps

pyplot()

# mean, var, derivvar, eigf

i = 20

bm = beliefs[i]
occ = mission.occupancy

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

pred_plt = heatmap(axs..., pred_map',
    title="GP Mean",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)

err_plt = heatmap(axs..., err_map',
    title="GP Variance",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)

weights = (1, 1e1, 1, 1)

sampleCost = DerivVar(
    occ, samples[1:i], bm, quantities, weights
)

new_loc = selectSampleLocation(sampleCost, occ.bounds)

data = -sampleCost.(points)

data[occ] .= NaN

der_plt = heatmap(axs..., data',
    title="Gradient Norm",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc[1]], [new_loc[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)

sampleCost = EIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc = selectSampleLocation(sampleCost, occ.bounds)

data = -sampleCost.(points)

data[occ] .= NaN

eigf_plt = heatmap(axs..., data',
    title="Nearest-Sample Diff",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc[1]], [new_loc[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)

plt = plot(
    pred_plt, err_plt,
    der_plt, eigf_plt,
    layout = 4,
    size=(1000,800)
)
# display(plt)

savefig(output_dir * "thesis/obj_func_comp.png")

#* dist-scaled objective function plots

# eigf, dist-scaled

using .SampleCosts, .Samples, .Maps

pyplot()

# mean, var, derivvar, eigf

i = 20

bm = beliefs[i]

axs, points = generateAxes(occ)

weights = (1, 1e2, 1, 1)


sampleCost = EIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc_eigf = selectSampleLocation(sampleCost, occ.bounds)

data_eigf = -sampleCost.(points)

data_eigf[occ] .= NaN

sampleCost = DistScaledEIGF(
    occ, samples[1:i], bm, quantities, weights
)

new_loc_dist = selectSampleLocation(sampleCost, occ.bounds)

data_dist = -sampleCost.(points)

data_dist[occ] .= NaN


eigf_plt = heatmap(axs..., data_eigf',
    title="No Distance Scaling",
    framestyle=:none,
    titlefontsize=24,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc_eigf[1]], [new_loc_eigf[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)


dist_plt = heatmap(axs..., data_dist',
    title="Distance Scaling",
    framestyle=:none,
    titlefontsize=24,
    colorbar_tickfontsize=18,
    # clim=(0, 1),
)
scatter!(x1[1:i-1], x2[1:i-1];
    label=false,
    color=:green,
    markersize=8)
scatter!(x1[i:i], x2[i:i];
    label=false,
    color=:royalblue,
    shape=:utriangle,
    markersize=14)
scatter!([new_loc_dist[1]], [new_loc_dist[2]],
    label=false,
    color=:red,
    shape=:xcross,
    markersize=14)


plt = plot(
    eigf_plt, dist_plt,
    layout=2,
    size=(1000, 400)
)
# display(plt)

savefig(output_dir * "thesis/dist_scaled_comp.png")

# eigf, dist-scaled

#* dist-scaled run sample order

pyplot()

dir = "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_EIGF"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

bm = beliefs[end]

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

eigf_plt = heatmap(axs..., pred_map',
    title="Predictions",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1, x2;
    label=false,
    color=colormap("Greens", length(xp)),
    markersize=10)
plot!(x1, x2, label=false, color=:gray, line=:dash)

dxs = map(xp) do x
    dx1, dx2 = 0, 1
    if abs(x[1] - 0) < 0.05
        dx1 = 1
    end
    if abs(x[1] - 1) < 0.05
        dx1 = -1
    end
    if abs(x[2] - 0) < 0.05
        dx2 = 1
    end
    if abs(x[2] - 1) < 0.05
        dx2 = -1.25
    end
    n = 1
    dx1/20/n, dx2/20/n
end
dx1 = getindex.(dxs, 1)
dx2 = getindex.(dxs, 2)
for (i, (x1, x2)) in enumerate(xp)
    annotate!(x1 .+ dx1[i], x2 .+ dx2[i], Plots.text(string(i), :lightgray, 16))
end

dir = "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"
file_name = output_dir * "$dir/data_000" * output_ext

data = load(file_name)
mission = data["missions"][1].mission
samples = data["missions"][1].samples
beliefs = data["missions"][1].beliefs

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

bm = beliefs[end]

axs, points = generateAxes(occ)
pred_map, err_map = bm(tuple.(points, 1))

# blocked points
pred_map[occ] .= NaN
err_map[occ] .= NaN

palette(:greens, length(xp))

dist_plt = heatmap(axs..., pred_map',
    title="Predictions",
    framestyle=:none,
    titlefontsize=26,
    colorbar_tickfontsize=18,
    clim=(0, 1),
)
scatter!(x1, x2;
    label=false,
    color=colormap("Greens", length(xp)),
    markersize=10)
plot!(x1, x2, label=false, color=:gray, line=:dash)

dxs = map(xp) do x
    dx1, dx2 = 0, 1
    if abs(x[1] - 0) < 0.05
        dx1 = 1
    end
    if abs(x[1] - 1) < 0.05
        dx1 = -1
    end
    if abs(x[2] - 0) < 0.05
        dx2 = 1
    end
    if abs(x[2] - 1) < 0.05
        dx2 = -1.25
    end
    n = 1
    dx1/20/n, dx2/20/n
end
dx1 = getindex.(dxs, 1)
dx2 = getindex.(dxs, 2)
for (i, (x1, x2)) in enumerate(xp)
    annotate!(x1 .+ dx1[i], x2 .+ dx2[i], Plots.text(string(i), :lightgray, 16))
end

plt = plot(
    eigf_plt, dist_plt,
    layout=2,
    size=(1000, 400)
)
# display(plt)

savefig(output_dir * "thesis/dist_scaled_run_comp.png")

#* mto single runs

#** load

gr()

dir = "new_syn/syn_mtoKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"
file_name = output_dir * "$dir/data_101" * output_ext
data = load(file_name)
missions = data["missions"]

#** run

plts = map(missions) do mission
    beliefs = mission.beliefs
    cors = map(beliefs) do bm
        outputCorMat(bm)[2:end, 1]
    end
    plot(stack(cors, dims=1), ylim=(-1, 1))
end

plot(plts..., layout=(3,6), size=(1200, 800)) |> display


#* combine runs into single plots (this isn't currently used, very little going on)

#** load

gr()

comp_dir = output_dir * "new_aus/aus_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"

temp = [
    "new_aus/aus_multiKernel_means_noises_fullpdf_nodrop_OnlyVar",
    "new_aus/aus_multiKernel_means_noises_fullpdf_nodrop_EIGF",
    "new_aus/aus_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF",
    # "new_aus/aus_multiKernel_means_noises_condpdf_nodrop_DistScaledEIGF",
    # "new_aus/aus_mtoKernel_means_noises_fullpdf_nodrop_DistScaledEIGF",
    "new_aus/aus_multiKernel_means_noises_fullpdf_hypdrop_DistScaledEIGF",
]
dirs =  output_dir .* temp

comp_name = comp_dir * "/data_000" * output_ext
names = dirs .* "/data_111" .* output_ext

all_names = [comp_name; names]

datas = load.(all_names)

maes = stack(data["metrics"].mae for data in datas)

#** run

labels = [
    "Zero Mean";;
    "MQGP + Variance";;
    "Nearest Difference";;
    "Distance-Scaled";;
    # "Conditional Likelihood";;
    # "Many-To-One";;
    "Hypothesis Dropout"
]

colors = reduce(hcat, colormap("Blues", length(labels)))

p_errs = plot(maes;
    title="Prediction Errors",
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    labels,
    color=colors,
    framestyle=:box,
    ylim=(0, 0.4),
    titlefontsize=24,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    marker=true,
    markersize=8,
    linewidth=4,
    margin=5mm,
    size=(width, height)
)
# for data in datas
#     metrics = data["metrics"]
#     plot!(
#         metrics.mae,
#     )
# end

display(p_errs)

#* combined satellite, elevation, and learned height maps

pyplot()

bounds = (lower=[0.0, 0.0], upper=[50.0, 50.0])
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap((elev_img.-minimum(elev_img)).*100, bounds)

sat_img = load(maps_dir * "satellite_50x50.tif")

# # sample sparsely from the prior maps
# # currently all data have the same sample numbers and locations
# n = (7,7) # number of samples in each dimension
# axs_sp = range.(bounds..., n)
# points_sp = vec(collect.(Iterators.product(axs_sp...)))

x0, y0, w, h = 10 .* (20, 50-15, 15, 15)
x, y = (x0 .+ [0,w,w,0,0], y0 .+ [0,0,h,h,0])

file_name = output_dir * "pye_farm_trial2/packaged/50x50_dense_grid/avg_height.csv"
pred_map = imgToMap(readdlm(file_name, ','), bounds)

dims = size(pred_map)
axs, points = generateAxes(bounds, dims)

# file_name = output_dir * "pye_farm_trial2/packaged/50x50_dense_grid/samples.txt"
# samples = map(eachline(file_name)) do line
#     Sample(eval(Meta.parse(line))...) # read each line as code and turn into Sample
# end
#
# xp = [s.x[1] for s in samples]
# x1 = getindex.(xp, 1)
# x2 = getindex.(xp, 2)

p1 = plot(sat_img;
          title="Satellite Image",
          aspect_ratio=:equal,
          # left_margin=-10mm
          )
# plot!(x, y,
#       legend=false,
#       linewidth=3
#       )
scatter!([], [];
         label=false,
         markersize=5)
p2 = heatmap(elevMap';
             title="Elevation Change (cm)",
             aspect_ratio=:equal,
             colorbar_tickfontsize=14,
             c=:oslo
             )
# plot!(x./10, 50 .- y./10,
#       legend=false,
#       linewidth=3
#       )
scatter!([], [];
         label=false,
         markersize=5)
p3 = heatmap(pred_map';
             title="Height (mm)",
             aspect_ratio=:equal,
             colorbar_tickfontsize=14,
             right_margin=-5Plots.mm,
             # clim=pred_range
             )
scatter!([], [];
         label=false,
         color=:green,
         markersize=5)
p = plot(p1, p2, p3;
         layout=(1,3),
         size=(1050, 300),
         titlefontsize=16,
         framestyle=:none,
         )
savefig(output_dir * "thesis/sat_elev_height_50x50.png")
# display(p)

#* field trial data

#** 2nd

gr()

width, height = 1200, 800

m = 10
n = 25
s = round.(Int, range(1, n, m))

dir_name = "pye_farm_trial2"

gt_dir = output_dir * dir_name * "/packaged/50x50_dense_grid/"

file_name = gt_dir * "avg_height.csv"
gt_map = imgToMap(readdlm(file_name, ','))

file_name = gt_dir * "avg_height_belief.csv"
gt_pred = imgToMap(readdlm(file_name, ','))

file_name = gt_dir * "avg_height_uncertainty.csv"
gt_err = imgToMap(readdlm(file_name, ','))

gt_mean = mean(gt_map)


dir = output_dir * dir_name * "/packaged/"
extent = "50x50"
runs = dir * extent .* ["", "_prior", "_dist", "_prior_dist"]
save_dir = output_dir * "thesis/field_trial_2/$(extent)/"
mkpath(save_dir)

boundsn = Dict()

# # 15x15 meter sub-patch (alt3)
lower = [284745.0, 6241345.0]
upper = [284760.0, 6241360.0]
boundsn["15x15"] = (; lower, upper)

# 50x50 meter space (alt2)
lower = [284725.0, 6241345.0]
upper = [284775.0, 6241395.0]
boundsn["50x50"] = (; lower, upper)

#*** mean error plot

maes = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    frames = readdir(run * "/avg_height_beliefs_25x25", join=true)
    for (i, frame) in pairs(frames)
        data_map = imgToMap(readdlm(frame, ','))
        mean_err = mean(abs.(data_map .- gt_map))
        maes[i,j] = mean_err
    end
end

p_errs = plot(
    maes,
    title="Mean Prediction Errors",
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error (mm)",
    labels=["No Prior or Scaling" "With Prior" "Distance Scaling" "Prior and Scaling"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:circle :circle :square :square],
    framestyle=:box,
    ylim=(27,40),
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

savefig(save_dir * "mean_errors.png")

#*** max error plot

mxaes = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    frames = readdir(run * "/avg_height_beliefs_25x25", join=true)
    for (i, frame) in pairs(frames)
        data_map = imgToMap(readdlm(frame, ','))
        max_err = maximum(abs.(data_map .- gt_map))
        mxaes[i,j] = max_err
    end
end

p_max_errs = plot(
    mxaes,
    title="Max Prediction Errors",
    xlabel="Sample Number",
    ylabel="Max Absolute Map Error (mm)",
    labels=["No Prior or Scaling" "With Prior" "Distance Scaling" "Prior and Scaling"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:circle :circle :square :square],
    framestyle=:box,
    ylim=(110,230),
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

savefig(save_dir * "max_errors.png")

#*** distance plot

dists = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    file_name = run * "/samples.txt"
    samples = map(eachline(file_name)) do line
        Sample(eval(Meta.parse(line))...) # read each line as code and turn into Sample
    end
    filter!(s->s.x[2]==1, samples) # only looking at height
    for i in eachindex(samples)[2:end]
        # calculate the cumulative distance traveled
        dists[i,j] = dists[i-1,j] + norm(samples[i].x[1] - samples[i-1].x[1])
    end
end

p_dists = plot(
    dists,
    title="Distance Traveled",
    xlabel="Sample Number",
    ylabel="Cumulative Distance (m)",
    labels=["No Prior or Scaling" "With Prior" "Distance Scaling" "Prior and Scaling"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:circle :circle :square :square],
    framestyle=:box,
    # ylim=(-30,1150),
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

savefig(save_dir * "distances.png")

#*** sample times

times = Dict(
    "15x15" => ["06:11", "06:41", "05:54", "04:24"],
    "50x50" => ["14:35", "15:20", "07:24", "09:58"]
)

stimes = map(times[extent]) do time
    m, s = split(time, ':')
    parse(Int, m)*60 + parse(Int, s)
end

atimes = stimes ./ num_samples

p_comp = bar(
    ["No Prior or Scaling", "With Prior", "Distance Scaling", "Prior and Scaling"],
    atimes,
    xlabel="Sample Number",
    ylabel="Average Time per Sample (s)",
    title="Average Time per Sample",
    seriescolors=[:black, RGB(0.1,0.7,0.2), :black, RGB(0.1,0.7,0.2)],
    framestyle=:box,
    # ylim=(0,25),
    titlefontsize=24,
    tickfontsize=15,
    labelfontsize=20,
    legend=nothing,
    margin=5mm,
    size=(width, height)
)
gui()

savefig(save_dir * "sample_times.png")

#*** correlation plot

runs_with_prior = [
    output_dir * "pye_farm_trial2/packaged/" * extent .* ["_prior", "_prior_dist"]...,
    output_dir * "pye_farm_trial_orig/packaged/" * "30samples_" * extent .* ["_priors"]...
]

# correlations
cors = zeros(num_samples, length(runs_with_prior))
for (j, run) in pairs(runs_with_prior)
    file_name = run * "/correlations_new.txt"
    for (i, line) in enumerate(eachline(file_name))
        cors[i,j] = only(eval(Meta.parse(line))[2:end])
    end
end

# full dense correlation
bounds = (lower=[0.0, 0.0], upper=[50.0, 50.0])
elev_img = Float64.(gray.(load(maps_dir * "dem_$(extent).tif")))
elevMap = imgToMap(elev_img, bounds)

axs, points = generateAxes(bounds, (25, 25))
elev_map = elevMap.(points)
gt_cor = cor(vec(elev_map), vec(gt_map))

p_cors = plot(
    cors,
    title="Correlation Between Pasture Height and Elevation",
    labels=["Without Distance Scaling" "With Distance Scaling" "Using Zero Mean"],
    xlabel="Sample Number",
    ylabel="Estimated Correlation",
    seriescolors=[RGB(0.1,0.7,0.2) RGB(0.1,0.7,0.2) :darkred],
    markers=[:circle :square],
    framestyle=:box,
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
hline!([gt_cor],
    label="Ground Truth Correlation",
    width=4,
    color=:royalblue
)
gui()

savefig(save_dir * "correlations_new.png")

#*** final height maps

#**** set all ranges

t_extent = "50x50"

pred_range = extrema(gt_pred)
err_range = extrema(gt_err)

# pred_range = -Inf, Inf
# err_range = -Inf, Inf

all_runs = [
               output_dir * "pye_farm_trial2" * "/packaged/" * t_extent .* ["", "_prior", "_dist", "_prior_dist"],
               # output_dir * "pye_farm_trial_adjust" * "/packaged/" * "30samples_" * t_extent .* ["", "_priors"]
           ] |> Iterators.flatten |> collect

for run in runs
    pred = imgToMap(readdlm(run * "/avg_height_belief.csv", ','), boundsn[t_extent])
    err = imgToMap(readdlm(run * "/avg_height_uncertainty.csv", ','), boundsn[t_extent])
    global pred_range = (min(minimum(pred), pred_range[1]),
                         max(maximum(pred), pred_range[2]))
    global err_range = (min(minimum(err), err_range[1]),
                        max(maximum(err), err_range[2]))
end

pred_range = (0.0, pred_range[2])
err_range = (0.0, err_range[2])

#**** load all data

full_data = map(runs) do run
    final_belief = imgToMap(readdlm(run * "/avg_height_belief.csv", ','), boundsn[extent])
    final_uncertainty = imgToMap(readdlm(run * "/avg_height_uncertainty.csv", ','), boundsn[extent])

    samples = map(eachline(run * "/samples.txt")) do line
        Sample(eval(Meta.parse(line))...)
    end
    filter!(s->s.x[2]==1, samples) # only looking at height
    xp = [s.x[1] for s in samples]
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    return (final_belief, final_uncertainty), (x1, x2)
end

#**** make plots

pyplot()

mkpath(save_dir * "final_maps")

axs, _ = generateAxes(boundsn[extent], size(full_data[1][1][1]))

# i = 1
# run = runs[i]
# run_data = full_data[i]
for (run, run_data) in zip(runs, full_data)
    pred_map, err_map = run_data[1]
    x1, x2 = run_data[2]

    p = heatmap(axs..., pred_map';
        title="Average Height (mm)",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=26,
        colorbar_tickfontsize=18,
        size=(600, 515),
        right_margin=-10mm,
        clim=pred_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=8)

    savefig(save_dir * "final_maps/"
            * run[findlast('/', run)+1:end]
            * "_belief.png")

    p = heatmap(axs..., err_map';
        title="Uncertainty (mm)",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=26,
        colorbar_tickfontsize=18,
        size=(600, 515),
        right_margin=-10mm,
        clim=err_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=8)

    savefig(save_dir * "final_maps/"
            * run[findlast('/', run)+1:end]
            * "_uncertainty.png")
end

#**** make combined plots

pyplot()

mkpath(save_dir * "final_maps")

axs, _ = generateAxes(boundsn[extent], size(gt_pred))

p01 = heatmap(gt_pred';
    title="Average Height (mm)",
    ylabel="Dense Sampling",
    labelfontsize=19,
    aspect_ratio=:equal,
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    clim=pred_range,
)
scatter!([], [];
         label=false,
         color=:green,
         markersize=6)

p02 = heatmap(gt_err';
    title="Uncertainty (mm)",
    aspect_ratio=:equal,
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    clim=err_range,
)
scatter!([], [];
         label=false,
         color=:green,
         markersize=6)

ylabels = ["No Prior or Scaling", "With Prior", "Distance Scaling", "Prior and Scaling"]

i = 1
run = runs[i]
run_data = full_data[i]
plots = map(runs, full_data, ylabels) do run, run_data, ylabel
    pred_map, err_map = run_data[1]
    x1, x2 = run_data[2]

    p1 = heatmap(axs..., pred_map';
        title="",
        ylabel,
        labelfontsize=19,
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=pred_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=6)

    p2 = heatmap(axs..., err_map';
        title="",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=6)

    return p1, p2
end

p = plot(p01, p02, Iterators.flatten(plots)...,
    layout=(5, 2),
    size=(700, 1300))

savefig(save_dir * "final_maps/combined.png")

#**** final example map

pyplot()

axs, _ = generateAxes(boundsn[extent], size(gt_pred))

i = 1
run = runs[i]
run_data = full_data[i]

pred_map, err_map = run_data[1]

p1 = heatmap(axs..., pred_map';
    title="",
    labelfontsize=19,
    aspect_ratio=:equal,
    framestyle=:none,
    color=:YlGn,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    colorbar=false,
    margin=0mm,
    size=(400, 400)
)

savefig(output_dir * "thesis/example_field_map$i.png")

i = 2
run = runs[i]
run_data = full_data[i]

pred_map, err_map = run_data[1]

p1 = heatmap(axs..., reverse(-pred_map, dims=1)';
    title="",
    labelfontsize=19,
    aspect_ratio=:equal,
    framestyle=:none,
    color=:Purples,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    colorbar=false,
    margin=0mm,
    size=(400, 400)
)

savefig(output_dir * "thesis/example_field_map$i.png")

i = 3
run = runs[i]
run_data = full_data[i]

pred_map, err_map = run_data[1]

p1 = heatmap(axs..., -pred_map';
    title="",
    labelfontsize=19,
    aspect_ratio=:equal,
    framestyle=:none,
    color=:Blues,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    colorbar=false,
    margin=0mm,
    size=(400, 400)
)

savefig(output_dir * "thesis/example_field_map$i.png")

#** 1st

gr()

width, height = 1200, 800


dir_name = "pye_farm_trial_adjust"

gt_dir = output_dir * dir_name * "/packaged/100samples_50x50_grid/"

file_name = gt_dir * "avg_height.csv"
gt_map = imgToMap(readdlm(file_name, ','))

file_name = gt_dir * "avg_height_belief.csv"
gt_pred = imgToMap(readdlm(file_name, ','))

file_name = gt_dir * "avg_height_uncertainty.csv"
gt_err = imgToMap(readdlm(file_name, ','))

gt_mean = mean(gt_map)


dir = output_dir * dir_name * "/packaged/"
extent = "50x50"
runs = dir * "30samples_" * extent .* ["", "_priors"]
save_dir = output_dir * "thesis/field_trial_1_adjust/$(extent)/"
mkpath(save_dir)

boundsn = Dict()

# # 15x15 meter sub-patch (alt3)
lower = [284745.0, 6241345.0]
upper = [284760.0, 6241360.0]
boundsn["15x15"] = (; lower, upper)

# 50x50 meter space (alt2)
lower = [284725.0, 6241345.0]
upper = [284775.0, 6241395.0]
boundsn["50x50"] = (; lower, upper)

#*** mean error plot

maes = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    frames = readdir(run * "/avg_height_beliefs_10x10", join=true)
    for (i, frame) in pairs(frames)
        data_map = imgToMap(readdlm(frame, ','))
        mean_err = mean(abs.(data_map .- gt_map))
        maes[i,j] = mean_err
    end
end

p_errs = plot(
    maes,
    title="Prediction Errors",
    xlabel="Sample Number",
    ylabel="Mean Absolute Map Error",
    labels=["Without Prior" "With Prior" "Without Prior 2"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:utriangle :utriangle :square :square],
    framestyle=:box,
    ylim=(23,37),
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

savefig(save_dir * "mean_errors.png")

#*** max error plot

mxaes = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    frames = readdir(run * "/avg_height_beliefs_10x10", join=true)
    for (i, frame) in pairs(frames)
        data_map = imgToMap(readdlm(frame, ','))
        max_err = maximum(abs.(data_map .- gt_map))
        mxaes[i,j] = max_err
    end
end

p_max_errs = plot(
    mxaes,
    title="Max Prediction Errors",
    xlabel="Sample Number",
    ylabel="Max Absolute Map Error",
    labels=["Without Prior" "With Prior" "Without Prior 2"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:utriangle :utriangle :square :square],
    framestyle=:box,
    # ylim=(110,230),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    legend=:topright,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
gui()

savefig(save_dir * "max_errors.png")

#*** distance plot

dists = zeros(num_samples, length(runs))
for (j, run) in pairs(runs)
    file_name = run * "/samples.txt"
    samples = map(eachline(file_name)) do line
        Sample(eval(Meta.parse(line))...) # read each line as code and turn into Sample
    end
    filter!(s->s.x[2]==1, samples) # only looking at height
    for i in eachindex(samples)[2:end]
        # calculate the cumulative distance traveled
        dists[i,j] = dists[i-1,j] + norm(samples[i].x[1] - samples[i-1].x[1])
    end
end

p_dists = plot(
    dists,
    title="Distance Traveled",
    xlabel="Sample Number",
    ylabel="Cumulative Distance (m)",
    labels=["Without Prior" "With Prior" "Without Prior 2"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:utriangle :utriangle :square :square],
    framestyle=:box,
    ylim=(-30,1150),
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

savefig(save_dir * "distances.png")

#*** correlation plot

runs_with_prior = dir * "30samples_" * extent .* ["_priors"]

# correlations
cors = zeros(num_samples, length(runs_with_prior))
for (j, run) in pairs(runs_with_prior)
    file_name = run * "/correlations.txt"
    for (i, line) in enumerate(eachline(file_name))
        cors[i,j] = only(eval(Meta.parse(line))[2:end])
    end
end

# full dense correlation
bounds = (lower=[0.0, 0.0], upper=[50.0, 50.0])
elev_img = Float64.(gray.(load(maps_dir * "dem_$(extent).tif")))
elevMap = imgToMap(elev_img, bounds)

axs, points = generateAxes(bounds, (25, 25))
elev_map = elevMap.(points)
gt_cor = cor(vec(elev_map), vec(gt_map))

p_cors = plot(
    cors,
    title="Estimated Correlation Between Plant Height and Elevation",
    labels=:none,
    xlabel="Sample Number",
    ylabel="Correlation",
    seriescolors=[RGBA(0.4, 0.2, 0.1, 0.5) RGBA(0.4, 0.2, 0.1, 1.0)],
    markers=[:utriangle :square],
    framestyle=:box,
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
hline!([gt_cor],
    label="Ground Truth Correlation",
    width=4,
    color=:royalblue
)
gui()

savefig(save_dir * "correlations.png")

#*** final height maps

#**** load all data

full_data = map(runs) do run
    final_belief = imgToMap(readdlm(run * "/avg_height_belief.csv", ','), boundsn[extent])
    final_uncertainty = imgToMap(readdlm(run * "/avg_height_uncertainty.csv", ','), boundsn[extent])

    samples = map(eachline(run * "/samples.txt")) do line
        Sample(eval(Meta.parse(line))...)
    end
    filter!(s->s.x[2]==1, samples) # only looking at height
    xp = [s.x[1] for s in samples]
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    return (final_belief, final_uncertainty), (x1, x2)
end

#**** make plots

pyplot()

mkpath(save_dir * "final_maps")

axs, _ = generateAxes(boundsn[extent], size(full_data[1][1][1]))

# i = 1
# run = runs[i]
# run_data = full_data[i]
for (run, run_data) in zip(runs, full_data)
    pred_map, err_map = run_data[1]
    x1, x2 = run_data[2]

    p = heatmap(axs..., pred_map';
        title="Average Height (mm)",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=26,
        colorbar_tickfontsize=18,
        size=(600, 515),
        right_margin=-10mm,
        clim=pred_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=8)

    savefig(save_dir * "final_maps/"
            * run[findlast('/', run)+1:end]
            * "_belief.png")

    p = heatmap(axs..., err_map';
        title="Uncertainty (mm)",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=26,
        colorbar_tickfontsize=18,
        size=(600, 515),
        right_margin=-10mm,
        clim=err_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=8)

    savefig(save_dir * "final_maps/"
            * run[findlast('/', run)+1:end]
            * "_uncertainty.png")
end

#**** make combined plots

pyplot()

mkpath(save_dir * "final_maps")

axs, _ = generateAxes(boundsn[extent], size(gt_pred))

p01 = heatmap(gt_pred';
    title="Average Height (mm)",
    ylabel="Dense Sampling",
    labelfontsize=19,
    aspect_ratio=:equal,
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    clim=pred_range,
)
scatter!([], [];
         label=false,
         color=:green,
         markersize=6)

p02 = heatmap(gt_err';
    title="Uncertainty (mm)",
    aspect_ratio=:equal,
    framestyle=:none,
    titlefontsize=19,
    colorbar_tickfontsize=17,
    clim=err_range,
)
scatter!([], [];
         label=false,
         color=:green,
         markersize=6)

ylabels=["Without Prior" "With Prior" "Without Prior 2"]

i = 1
run = runs[i]
run_data = full_data[i]
plots = map(runs, full_data, ylabels) do run, run_data, ylabel
    pred_map, err_map = run_data[1]
    x1, x2 = run_data[2]

    p1 = heatmap(axs..., pred_map';
        title="",
        ylabel,
        labelfontsize=19,
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=pred_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=6)

    p2 = heatmap(axs..., err_map';
        title="",
        aspect_ratio=:equal,
        framestyle=:none,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
    )
    scatter!(x1, x2;
        label=false,
        color=:green,
        markersize=6)

    return p1, p2
end

p = plot(p01, p02, Iterators.flatten(plots)...,
    layout=(3, 2),
    size=(700, 780))

savefig(save_dir * "final_maps/combined.png")

#* full run comparison

pyplot()

j = 1
run = runs[j]
name = run[findlast('/', run)+1:end]

samples = map(eachline(run * "/samples.txt")) do line
    Sample(eval(Meta.parse(line))...)
end
filter!(s -> s.x[2] == 1, samples) # only looking at height
xp = [s.x[1] for s in samples]
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = extrema(gt_pred)

err_max = mapreduce(max, 5:5:num_samples) do i
    si = lpad(i,2,'0')
    err_map = imgToMap(readdlm(run * "/avg_height_uncertainties/$(si).csv", ','), boundsn[extent])
    maximum(err_map)
end
err_range = (0.0, err_max)

pred_ticks = createColorbarTicks(pred_range)
err_ticks = createColorbarTicks(err_range)

i=20
plots = map(5:5:num_samples) do i

    si = lpad(i,2,'0')
    pred_map = imgToMap(readdlm(run * "/avg_height_beliefs/$(si).csv", ','), boundsn[extent])
    err_map = imgToMap(readdlm(run * "/avg_height_uncertainties/$(si).csv", ','), boundsn[extent])
    obj_map = imgToMap(readdlm(run * "/sample_utilities/$(si).csv", ','), boundsn[extent])

    axs, _ = generateAxes(pred_map)

    gt_title = i == 5 ? "Dense Sampling" : ""
    p0 = heatmap(axs..., gt_pred';
        title=gt_title,
        ylabel="$i Samples",
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        labelfontsize=17,
        clim=pred_range,
    )

    pred_title = i == 5 ? "Predicted Values" : ""
    p1 = heatmap(axs..., pred_map';
        title=pred_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=pred_range,
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=7)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    err_title = i == 5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=(0, 43),
        # colorbar_ticks=err_ticks,
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=7)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    lessNotInf = x -> x === -Inf ? Inf : x
    greaterNotInf = x -> x === Inf ? -Inf : x

    obj_range = (0.0, maximum(greaterNotInf, obj_map))
    obj_ticks = createColorbarTicks(obj_range)

    obj_title = i == 5 ? "Sample Utility" : ""
    p3 = heatmap(axs..., obj_map';
        title=obj_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=obj_range,
        colorbar_ticks=obj_ticks,
    )
    scatter!(x1[1:i-1], x2[1:i-1];
        label=false,
        color=:green,
        markersize=7)
    scatter!(x1[i:i], x2[i:i];
        label=false,
        color=:royalblue,
        shape=:utriangle,
        markersize=12)
    if i < num_samples
        scatter!(x1[i+1:i+1], x2[i+1:i+1],
            label=false,
            color=:red,
            shape=:xcross,
            markersize=10)
    end

    return p0, p1, p2, p3
end

p = plot(Iterators.flatten(plots)...,
    layout=(6, 4),
    size=(1100, 1300))

mkpath(save_dir * "full_runs")
savefig(save_dir * "full_runs/$(name).png")

#* scatter comparison

m = 10
n = 25
s = round.(Int, range(1, n, m))

50/24*(1/3)*√2 # farthest away the points will be from each other, ~1 meter

p = plot()

axs, points = generateAxes((0,1), (n,n))
x1, x2 = getindex.(points[s,s], 1), getindex.(points[s,s], 2)
scatter!(x1, x2,
    color=:red,
    legend=false
)

axs, points = generateAxes((0,1), (m,m))
x1, x2 = getindex.(points, 1), getindex.(points, 2)
scatter!(x1, x2,
    color=:blue,
    legend=false
)

display(p)

#* scatter comparison

dir_name = "pye_farm_trial2"

gt_dir = output_dir * dir_name * "/packaged/50x50_dense_grid/"

file_name = gt_dir * "avg_height.csv"
gt_map_2 = imgToMap(readdlm(file_name, ','))[s,s]

gt_mean_2 = mean(gt_map_2)

dir_name = "pye_farm_trial_adjust"

gt_dir = output_dir * dir_name * "/packaged/100sample_50x50_grid/"

file_name = gt_dir * "avg_height.csv"
gt_map_1 = imgToMap(readdlm(file_name, ','))

gt_mean_1 = mean(gt_map_1)

mean(abs.(gt_map_2 .- gt_map_1))

clim = extrema(hcat(gt_map_1, gt_map_2, abs.(gt_map_2 .- gt_map_1)))
extrema(gt_map_1), extrema(gt_map_2)
mean(gt_map_1), mean(gt_map_2)

plot(
    heatmap(gt_map_1; clim),
    heatmap(gt_map_2; clim),
    heatmap(abs.(gt_map_2 .- gt_map_1)),
    layout=(3,1),
    # aspect_ratio=:equal,
    framestyle=:none,
    size=(500,1200),
)|>display

cor(vec(gt_map_1), vec(gt_map_2))
mean(abs.(gt_map_1 .- gt_map_2))/mean((gt_map_1 .+ gt_map_2)/2)

#* load mission

region = "aus"
priors = "000"
dir = "new_$(region)/$(region)_multiKernel_zeromean_noises_fullpdf_nodrop_OnlyVar"
file_name = output_dir * "$dir/data_$(priors)" * output_ext

mkpath(output_dir * "thesis/$(region)_$(priors)")

data = load(file_name)
maes = data["metrics"].mae
mxaes = data["metrics"].mxae
dists = cumsum(data["metrics"].dists)
times = cumsum(data["metrics"].times)

mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for bm in @view beliefs[5:5:end]
    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(vec(points), 1))
    mask = vec(.! mission.occupancy)

    global pred_range = (min(minimum(pred_map[mask]), pred_range[1]),
                         max(maximum(pred_map[mask]), pred_range[2]))
    global err_range = (min(minimum(err_map[mask]), err_range[1]),
                        max(maximum(err_map[mask]), err_range[2]))
end

err_range = (0.0, err_range[2])

pred_ticks = createColorbarTicks(pred_range)
err_ticks = createColorbarTicks(err_range)

#* full run comparison

pyplot()

i=5
plots = map(5:5:length(beliefs)) do i
    axs, _ = generateAxes(mission.sampler[1])

    gt_title = i == 5 ? "Ground Truth" : ""
    p0 = heatmap(axs..., mission.sampler[1]';
        title=gt_title,
        ylabel="$i Samples",
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        labelfontsize=17,
    )

    bm = beliefs[i]

    axs, points = generateAxes(occ)
    pred_map, err_map = bm(tuple.(points, 1))

    pred_map[occ] .= NaN
    err_map[occ] .= NaN

    pred_title = i == 5 ? "Predicted Values" : ""
    p1 = heatmap(axs..., pred_map';
        title=pred_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=(0, 1),
    )
    scatter!(x1[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=7)

    err_title = i == 5 ? "Uncertainties" : ""
    p2 = heatmap(axs..., err_map';
        title=err_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        clim=err_range,
        # colorbar_ticks=err_ticks,
    )
    scatter!(x1[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=7)

    sampleCost = mission.sampleCostType(
        occ, samples[1:i], bm, quantities, mission.weights
    )
    obj_map = -sampleCost.(points)

    obj_title = i == 5 ? "Sample Utility" : ""
    p3 = heatmap(axs..., obj_map';
        title=obj_title,
        framestyle=:none,
        # aspect_ratio=:equal,
        right_margin=-10mm,
        titlefontsize=19,
        colorbar_tickfontsize=17,
        # clim=obj_range,
        # colorbar_ticks=obj_ticks,
    )
    scatter!(x1[1:i], x2[1:i];
        label=false,
        color=:green,
        markersize=7)

    return p0, p1, p2, p3
end

p = plot(Iterators.flatten(plots)...,
    layout=(6, 4),
    size=(1100, 1300))

savefig(output_dir * "thesis/$(region)_$(priors)/full_run_comparison.png")


#* combined data maps

samples = Iterators.flatmap([gt_dir]) do run
    Iterators.filter(s -> s.x[2] == 1,
        Iterators.map(eachline(run * "/samples.txt")) do line
            Sample(eval(Meta.parse(line))...)
        end)
end |> collect

argmax(s->s.y, samples)

bm = MQGP(samples, boundsn[extent];
            means=(use=true, learned=true),
            noise=(value=0.0, learned=true))

axs, _ = generateAxes(boundsn[extent], (100, 100))
pred_map, err_map = produceMaps(bm, boundsn[extent], (100, 100))

p = plot(
    heatmap(axs..., pred_map';
            clim=(0,240),
            framestyle=:none,
            aspect_ratio=:equal),
    heatmap(axs..., err_map';
            clim=(0,40),
            framestyle=:none,
            aspect_ratio=:equal),
)
xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)
display(p)

bm

# xp = first.(getfield.(samples, :x))
# x1 = getindex.(xp, 1)
# x2 = getindex.(xp, 2)
# scatter(x1, x2;
#          legend=false)|>display

#* length scales

#** synthetic

dir = "new_syn/syn_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"

file_name = output_dir * "$dir/data_000" * output_ext
data = load(file_name)

ls = mean(data["missions"]) do mission
    [belief.θ.ℓ for belief in mission.beliefs]
end

# ls = [belief.θ.ℓ for belief in data["missions"][1].beliefs]

plot(ls)|>display

ls[end]

#** satellite

# dir = "new_aus/aus_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"
dir = "new_nsw/nsw_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF"

file_name = output_dir * "$dir/data_111" * output_ext
data = load(file_name)

occ = data["mission"].occupancy
ll = (occ |> count)/(occ |> size |> prod) |> sqrt

ls = [belief.θ.ℓ for belief in data["beliefs"]]

plot(ls)|>display

ls[end-1]

#** field

run = runs[1]

ls = map(eachline(run * "/belief_params.txt")) do line
    eval(Meta.parse(line)).σn[1]
end

plot(abs.(ls))|>display

abs(ls[end])

#* other

ll = 1 # length of region
b = count(.!occupancy)/length(occupancy) # fractional free area
ln = ll*sqrt(b) # new length of region

ls = 0.47 # original aus length-scale
ls/ln # new aus length-scale

#* other

a = [
    [0.45, 0.21, 0.0025],
    [0.61, 0.24, 0.02],
    [0.44, 0.16, 0.00],
    [1.14/50, 38/41, 1.45/41],
    [2.63/50, 36/41, 0.1/41],
    [2.37/50, 27/41, 0.1/41],
    [2.49/50, 45/41, 0.1/41],
    [1.65/50, 41/41, 0.1/41]
]

b = map(a) do x
    (x[2] - x[3])/x[1]
end

#** dense field

samples = Iterators.flatmap([gt_dir]) do run
    Iterators.filter(s -> s.x[2] == 1,
        Iterators.map(eachline(run * "/samples.txt")) do line
            Sample(eval(Meta.parse(line))...)
        end)
end |> collect

bm = MQGP(samples, boundsn[extent]; noise=(value=0.0, learned=true))
bm.θ.σn

ls = map(eachline(run * "/belief_params.txt")) do line
    eval(Meta.parse(line)).σn[1]
end

plot(abs.(ls))|>display

abs(ls[end])

#* recalculate correlations

using DelimitedFiles

for run in filter(contains("prior"), runs)
    N = 2
    beliefs = map(enumerate(eachline(run * "/belief_params.txt"))) do (i, line)
        params = (; eval(Meta.parse(line))..., μ = [0, 0])
        MQGP([Sample(([0.0,0.0],1),0.0)], params; N)
    end
    cors = [outputCorMat(bm)[:, 1] for bm in beliefs]
    writedlm(run * "/correlations_new.txt", [cors], "\n")
end

#*** aus distance plot

save_dir = output_dir * "thesis/satellite/"
mkpath(save_dir)

width, height = 1200, 800

region = "nsw"
priors = ["000", "111"]
types = ["no", "hyp"]

file_names = [
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_EIGF/data_000" * output_ext,
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_EIGF/data_111" * output_ext,
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF/data_000" * output_ext,
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF/data_111" * output_ext,
]

dists = map(file_names) do file_name
    data = load(file_name)
    cumsum(data["metrics"].dists)
end |> vec |> stack

p = plot(
    dists,
    title="Distance Traveled",
    xlabel="Sample Number",
    ylabel="Cumulative Distance",
    labels=["Without Distance Scaling" "Without Distance Scaling" "With Distance Scaling" "With Distance Scaling"],
    seriescolors=[:black RGB(0.1,0.7,0.2) :black RGB(0.1,0.7,0.2)],
    markers=[:circle :circle :square :square],
    framestyle=:box,
    # ylim=(-30,1150),
    titlefontsize=24,
    markersize=8,
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)
# display(p)

savefig(save_dir * "$(region)_distances.png")


file_names = [
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF/data_000" * output_ext,
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_nodrop_DistScaledEIGF/data_111" * output_ext,
    output_dir * "new_$(region)/$(region)_multiKernel_means_noises_fullpdf_hypdrop_DistScaledEIGF/data_111" * output_ext,
]

times = map(file_names) do file_name
    data = load(file_name)
    cumsum(data["metrics"].times)
end |> vec |> stack

p = plot(
    times,
    title="Computation Time",
    xlabel="Sample Number",
    ylabel="Cumulative Computation Time (s)",
    labels=["No Priors" "All Priors" "With Hypothesis Dropout"],
    seriescolors=[:black RGB(0.1,0.7,0.2) RGB(0.1,0.7,0.2)],
    markers=[:circle :circle :dtriangle],
    framestyle=:box,
    # ylim=(-30,1150),
    titlefontsize=24,
    markersize=[8 8 10],
    tickfontsize=15,
    labelfontsize=20,
    legendfontsize=16,
    margin=5mm,
    linewidth=4,
    size=(width, height)
)

savefig(save_dir * "$(region)_computation_times.png")

