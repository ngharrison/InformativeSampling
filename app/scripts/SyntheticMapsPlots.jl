
using Random: seed!
using LinearAlgebra: I
using Statistics: cor
using Plots

using InformativeSampling

using .Maps: Map, generateAxes
using .Samples: Sample, MapsSampler

using InformativeSamplingUtils
using .DataIO: output_dir, GaussGroundTruth, Peak

#* data
seed_val=3; num_peaks=4; priors=Bool[1,1,1];

seed!(seed_val) # make random values deterministic

bounds = (lower = [0.0, 0.0], upper = [1.0, 1.0])

occupancy = Map(zeros(Bool, 100, 100), bounds)

#* initialize ground truth

# synthetic
peaks = [Peak(rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower, 0.02*(rand()+0.5)*I, rand())
         for i in 1:num_peaks]
ggt = GaussGroundTruth(peaks)
axs, points = generateAxes(occupancy)
mat = ggt(points)
map0 = Map(mat./maximum(mat), bounds)

#* Create prior prior_samples

# none -- leave uncommented
prior_maps = []

# multiplicative
m = Map(abs.(map0 .* randn()), bounds)
push!(prior_maps, m)

# additive
m = Map(abs.(map0 .+ 0.2 .* randn(size(map0))), bounds)
push!(prior_maps, m)

# random peaks
peaks = [Peak(rand(2).*(bounds.upper-bounds.lower) .+ bounds.lower, 0.02*(rand()+0.5)*I, rand())
         for i in 1:num_peaks]
tggt = GaussGroundTruth(peaks)
tmat = tggt(points)
m = Map(tmat./maximum(tmat), bounds)
push!(prior_maps, m)

sampler = MapsSampler(map0)

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (5,5) # number of samples in each dimension
axs_sp = range.(bounds..., n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample((x, i+length(sampler)), d(x))
                 for (i, d) in enumerate(prior_maps[priors])
                     for x in points_sp if !isnan(d(x))]

# Calculate correlation coefficients
[cor(map0.(points_sp), d.(points_sp)) for d in prior_maps]
[cor(vec(map0), vec(d)) for d in prior_maps]

#* plot maps
p1 = heatmap(axs..., map0', title="QOI", ticks=false, framestyle=:none)
p2 = heatmap(axs..., prior_maps[1]', title="High Dependence", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p3 = heatmap(axs..., prior_maps[2]', title="Medium Dependence", ticks=false)
scatter!(first.(points_sp), last.(points_sp);
         framestyle=:none,
         label="Samples",
         color=:green,
         legend=(0.15, 0.87),
         markersize=6)
p4 = heatmap(axs..., prior_maps[3]', title="Low Dependence", ticks=false)
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
p = plot(p1, p2, p3, p4, cbar,
     layout=@layout([grid(2,2){0.95w} c]),
     clim=(0,1),
     colorbar=false,
     size=(1000, 800),
     titlefontsize=24,
     tickfontsize=20,
     legendfontsize=14,
     margin=4Plots.mm
)

savefig(output_dir * "paper/qoi_priors_alt.png")
