using AdaptiveSampling: Maps, Missions, BeliefModels, Samples, Outputs, Kernels

using .Maps: res, generateAxes
using .Missions: Mission
using .BeliefModels: BeliefModel
using .Kernels: fullyConnectedCovMat
using .Samples: Sample
using .Outputs: output_dir, output_ext

using Statistics: mean, std
using StatsBase: mean_and_cov, AnalyticWeights
using FileIO: load
using Plots

const CI = CartesianIndex

struct LinearModel
    a::Float64
    b::Float64
end

"""
Returns a linear model from second variable to first variable.
Requires mean of each variable, the covariance between them,
and the variance of the second variable.
"""
function LinearModel(mean_y, mean_x, cov_xy, var_x)
    b = cov_xy/var_x
    a = mean_y - b*mean_x # y-intercept
    return LinearModel(a,b)
end

(lm::LinearModel)(x) = lm.a + lm.b*x

## load mission
dir = "aus_ave"
mname = "111"
file_name = output_dir * "$dir/mission_$(mname)" * output_ext

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
occ = mission.occupancy

all_samples = [samples; mission.prior_samples]
num_quant = maximum(s->s.x[2], all_samples)
# split samples by quantity and get just the values
vals = [getfield.(filter(s->s.x[2]==i, all_samples), :y)
        for i in 1:num_quant]
extrema.(vals)
beliefModel = beliefs[end]
A = fullyConnectedCovMat(beliefModel.θ.σ)

## linear model --- measurements

fs1 = []
for k in 1:num_quant
    ixs = [1,k]
    means = mean.(vals[ixs])
    stds = std.(vals[ixs])
    r = A[ixs...] / √(A[ixs[1],ixs[1]] * A[ixs[2],ixs[2]])
    push!(fs1, LinearModel(means..., r*prod(stds), stds[2]^2))
    println("r^2 = $(sub_cov_mat[1,2]/sqrt(sub_cov_mat[1,1]*sub_cov_mat[2,2]))")
end
fs1

plot([x->f(x) for f in fs1], 0, 1;
     title="Modeling Vegetation from Others",
     xlabel="Other Quantity",
     ylabel="Vegetation",
     labels=["Vegetation" "Elevation" "Ground Temperature" "Rainfall"])

savefig(output_dir * "linear_models/from_samples.svg")

## linear model --- GP cov mat

fs2 = []
for k in 1:num_quant
    ixs = [1,k]
    means = mean.(vals[ixs])
    sub_cov_mat = A[CI.((i,j) for i in ixs, j in ixs)]
    push!(fs2, LinearModel(means..., sub_cov_mat[1,2], sub_cov_mat[2,2]))
    println("r^2 = $(sub_cov_mat[1,2]/sqrt(sub_cov_mat[1,1]*sub_cov_mat[2,2]))")
end
fs2

plot([x->f(x) for f in fs2], 0, 1;
     title="Modeling Vegetation from Others",
     xlabel="Other Quantity",
     ylabel="Vegetation",
     labels=["Vegetation" "Elevation" "Ground Temperature" "Rainfall"])

savefig(output_dir * "linear_models/from_samples_and_kernel.svg")

## linear model --- GP predictions
axs, points = generateAxes(occ)
dims = Tuple(length.(axs))

weights = σ -> @. 1/σ^2 # inverse variance
# weights = σ -> maximum(σ) .- σ # difference from max
# note: little difference between the two, can't say one is better

# using Visualization: visualize
# k=1
# visualize(beliefModel, filter(s->s.x[2]==k, all_samples), occ, k)

fs3 = []
for k in 1:num_quant
    μ1, σ1 = beliefModel(tuple.(vec(points), 1))
    w1 = weights(σ1)
    μk, σk = beliefModel(tuple.(vec(points), k))
    wk = weights(σk)
    means, sub_cov_mat = mean_and_cov([μ1 μk], AnalyticWeights(w1 .* wk))
    push!(fs3, LinearModel(means..., sub_cov_mat[1,2], sub_cov_mat[2,2]))
    println("r^2 = $(sub_cov_mat[1,2]/sqrt(sub_cov_mat[1,1]*sub_cov_mat[2,2]))")
end
fs3

# scatter(μk[begin:100:end], μ1[begin:100:end])

plot([x->f(x) for f in fs3], 0, 1;
     title="Modeling Vegetation from Others",
     xlabel="Other Quantity",
     ylabel="Vegetation",
     labels=["Vegetation" "Elevation" "Ground Temperature" "Rainfall"])

savefig(output_dir * "linear_models/from_predictions.svg")

## table
# 1. save csv
#    coefficients
#    r^2 values
#    plots

## plots
# not functional yet
all_samples[1]
p1 = heatmap(axs..., map0', title="Vegetation (QOI)", ticks=false, framestyle=:none)
p2 = heatmap(axs..., prior_maps[1]', title="Elevation", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p3 = heatmap(axs..., prior_maps[2]', title="Ground Temperature", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p4 = heatmap(axs..., prior_maps[3]', title="Rainfall", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
cy = 0:.001:1
cbar = heatmap([0], cy, reshape(cy, :, 1),
               # aspect_ratio=1/100,
               xticks=false,
               yticks=0:0.5:1,
               mirror=true
               )
plot(p1, p2, p3, p4, cbar,
     layout=@layout([grid(2,2){0.95w} c]),
     # clim=(0,1),
     colorbar=false,
     size=(1000, 800),
     titlefontsize=24,
     tickfontsize=20,
     legendfontsize=14,
     margin=4Plots.mm
)
