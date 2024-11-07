
using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples, .ROSInterface, .Kernels

using InformativeSamplingUtils
using .DataIO, .Visualization

using Statistics, FileIO, Plots, Images, Logging
using Plots: mm

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

# pyplot()

#* data
file_name = output_dir * "2024-02-15-12-02-46_mission.jld2"
file_name = output_dir * "2024-02-15-12-22-16_mission.jld2"
file_name = output_dir * "2024-02-15-12-56-06_mission.jld2"

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
quantities = eachindex(mission.sampler)
sampleCost = mission.sampleCostType(
    mission.occupancy, samples, beliefs[end], quantities, mission.weights
)
vis(mission, samples, beliefs[end], nothing; quantity=1)


#* concatenate split missions
file_name1 = output_dir * "pye_farm_trial/2024-02-15-17-26-06_mission.jld2"
file_name2 = output_dir * "pye_farm_trial/2024-02-15-18-26-15_mission.jld2"

data1 = load(file_name1)
data2 = load(file_name2)
samples = [data1["samples"]; data2["samples"]]
mission = data1["mission"]
bounds = getBounds(mission.occupancy)
beliefs = map(1:length(samples)) do i
    MQGP(samples[1:i], bounds)
end
save(mission, samples, beliefs; sub_dir_name="pye_farm_trial_gen")

#* elevation maps

elev_img = load(maps_dir * "dem_50x50.tif")
elevMap = imgToMap(gray.(elev_img), bounds)
vis(elevMap)

elev_img = load(maps_dir * "dem_15x15.tif")
elevMap = imgToMap(gray.(elev_img), bounds)
vis(elevMap)

#* load mission data
name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
bounds = getBounds(mission.occupancy)

global_logger(ConsoleLogger(stderr, Debug))

#* replay mission
replay(vis, mission, samples, beliefs; sleep_time=1.0)

#* make png from proxy ground truth
saveBeliefMapToPng(beliefs[end], mission.occupancy, name)

#* look at every sample collected together
all_samples = map(readdir(output_dir * "pye_farm_trial", join=true)[2:end]) do file_name
    data = load(file_name)
    data["samples"]
end |> Iterators.flatten |> collect

x = first.(first.(getfield.(all_samples, :x)))
y = last.(first.(getfield.(all_samples, :x)))
z = getfield.(all_samples, :y)

display(scatter3d(x, y, z, legend=false))

mission = load(readdir(output_dir * "pye_farm_trial", join=true)[end-1])["mission"]
bounds = getBounds(mission.occupancy)
beliefModel = MQGP(all_samples, bounds)

vis(beliefModel, [], mission.occupancy)


#* with different prior samples

name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
bounds = getBounds(mission.occupancy)

elev_img = load(maps_dir * "dem_15x15.tif")
elevMap = imgToMap(gray.(elev_img), bounds)
prior_maps = [elevMap]

n = (7,7)
axs_sp = range.(bounds..., n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))
prior_samples = [Sample{Float64}((x, i+length(mission.sampler)), d(x))
                 for (i, d) in enumerate(prior_maps)
                     for x in points_sp if !isnan(d(x))]

beliefs = map(1:length(samples)) do i
    MQGP([prior_samples; samples[1:i]], bounds)
end

vis(elevMap; points=points_sp)

#*
cors = map(beliefs) do b
    outputCorMat(b)
end

using LinearAlgebra

i = 10
outputCorMat(beliefs[i])
for i in 1:30
    vis(mission, samples[1:i], beliefs[i], nothing)
    sleep(1.5)
end

prior_belief = MQGP(prior_samples, bounds)
vis(mission, prior_samples, prior_belief, nothing; quantity=2)


#* combined points

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

all_samples = map(names) do name
    file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    data = load(file_name)
    data["samples"]
end |> Iterators.flatten |> collect

x = first.(first.(getfield.(all_samples, :x)))
y = last.(first.(getfield.(all_samples, :x)))
z = getfield.(all_samples, :y)

display(scatter(x, y, legend=false))

display(scatter3d(x, y, z, legend=false))

name = names[end]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
bounds = getBounds(mission.occupancy)

beliefs[end]

beliefModel = MQGP(all_samples, bounds)

vis(beliefs[end], [], mission.occupancy)

beliefModel = MQGP([prior_samples; samples], bounds)

outputCorMat(beliefModel)

#*

gt_name = "100samples_50x50_grid"
gt_data = load(output_dir * "pye_farm_trial_named/" * gt_name * output_ext)
gt_belief = gt_data["beliefs"][end]
mission = gt_data["mission"]
occupancy = mission.occupancy
bounds = getBounds(occupancy)

axes, points = generateAxes(occupancy)
μ, σ = gt_belief(tuple.(points, 1))
gt_pred_map = Map(μ, bounds)

name = "30samples_50x50_priors"
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
beliefs = data["beliefs"]

for i in 1:30
    μ, σ = beliefs[i](tuple.(points, 1))
    pred_map = Map(μ, bounds)
    err_map = abs.(pred_map .- gt_pred_map)
    vis(gt_pred_map, pred_map; titles=["gt", "belief"])
    println(mean(err_map))
    sleep(1.0)
end

maes = map(beliefs) do belief
    pred_map, err_map = belief(tuple.(points, 1))
    mean(abs.(pred_map .- gt_pred_map))
end

display(scatter(maes))

#* other plots
using Statistics, FileIO, GLMakie, Images

x = range(0, 10, length=100)
y = sin.(x)
lines(x, y)
lines(sin)

name = "pye_farm_trial_named/30samples_50x50_priors"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
occupancy = mission.occupancy
bounds = getBounds(occupancy)
axes, points = generateAxes(occupancy)
μ, σ = beliefs[15](tuple.(vec(points), 1))
pred_map = Map(μ, bounds)

fig, ax, hm = heatmap(pred_map')
Colorbar(fig[1, 2], hm, width=20)
scatter!([20], [40], color=:black, markersize=10)
display(fig)

using GLMakie
f = Figure()
ax = GLMakie.Axis(f[1,1])

heatmap.([pred_map, pred_map])

#* combined satellite, elevation, and learned height maps
bounds = [0.0, 0.0], [50.0, 50.0]
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap((elev_img.-minimum(elev_img)).*100, bounds)

sat_img = load(maps_dir * "satellite_50x50.tif")

# sample sparsely from the prior maps
# currently all data have the same sample numbers and locations
n = (7,7) # number of samples in each dimension
axs_sp = range.(bounds..., n)
points_sp = vec(collect.(Iterators.product(axs_sp...)))

x0, y0, w, h = 10 .* (20, 50-15, 15, 15)
x, y = (x0 .+ [0,w,w,0,0], y0 .+ [0,0,h,h,0])

name = names[6]
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
bm = MQGP([mission.prior_samples; samples], bounds)

occupancy = mission.occupancy
bounds = getBounds(occupancy)
axs, points = generateAxes(occupancy)
pred_map, err_map = bm(tuple.(points, 1))

p1 = plot(sat_img;
          title="Satellite Image",
          left_margin=-10mm
          )
# plot!(p1, x, y,
#       legend=false,
#       linewidth=3
#       )
p2 = heatmap(range.(bounds..., size(elevMap))..., elevMap;
             title="Elevation Change (cm)",
             aspect_ratio=:equal,
             # left_margin=-5mm,
             ytickfontsize=12,
             c=:oslo
             )
# scatter!(getindex.(points_sp, 1), getindex.(points_sp, 2);
#          label=false,
#          color=:green,
#          markersize=8)
p3 = heatmap(axs..., pred_map';
             title="Estimated Height (cm)",
             aspect_ratio=:equal,
             # left_margin=-5mm,
             ytickfontsize=12,
             )
p = plot(p1, p2, p3;
         layout=(1,3),
         size=(1050, 300),
         titlefontsize=16,
         framestyle=:none,
         )
# savefig(output_dir * "iros_2024/sat_elev_50x50.png")
display(p)

#* test correlations
axs, points = generateAxes(elevMap)
ss = vec(Sample.(tuple.(points, 1), elevMap))
bm = MQGP(ss, bounds)
vis(bm, [], Map(zeros(Bool,size(elev_img)), bounds))

name = "pye_farm_trial_named/100samples_50x50_grid"
file_name = output_dir * "$(name)" * output_ext
data = load(file_name)
samples = data["samples"]
ss = vec(Sample.(tuple.(points, 2), elevMap))
bm = MQGP([ss; samples], bounds)
outputCorMat(bm)
vis(bm, [], Map(zeros(Bool,size(elev_img)), bounds); quantity=1)

# still not getting consistent or expected results

#* averages predicted values over entire maps

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

names15 = names[1:3]
names50 = names[4:6]

# names for 15x15, load each, combine samples, calculate average
all_vals15 = Float64[]
all_locs15 = Vector{Float64}[]
pred_means15 = Vector{Float64}[]
for name in names15
    file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    data = load(file_name)
    for sample in data["samples"]
        if all(norm(sample.x[1] .- loc) > 0.3 for loc in all_locs15)
            push!(all_locs15, sample.x[1])
            push!(all_vals15, sample.y)
        end
    end
    mission = data["mission"]
    beliefs = map(1:mission.num_samples) do i
        MQGP([mission.prior_samples; samples[1:i]], bounds)
    end
    occ = mission.occupancy
    axs, points = generateAxes(occ)
    ss = vec(tuple.(points, 1))
    push!(pred_means15, [mean(belief(ss)[1]) for belief in beliefs])
end

all_vals15 |> mean

all_vals15 |> histogram |> display

all_vals50 = Float64[]
all_locs50 = Vector{Float64}[]
pred_means50 = Vector{Float64}[]
for name in names50
    file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
    data = load(file_name)
    for sample in data["samples"]
        if all(norm(sample.x[1] .- loc) > 0.3 for loc in all_locs50)
            push!(all_locs50, sample.x[1])
            push!(all_vals50, sample.y)
        end
    end
    mission = data["mission"]
    beliefs = map(1:mission.num_samples) do i
        MQGP([mission.prior_samples; samples[1:i]], bounds)
    end
    occ = mission.occupancy
    axs, points = generateAxes(occ)
    ss = vec(tuple.(points, 1))
    push!(pred_means50, [mean(belief(ss)[1]) for belief in beliefs])
end

# plot belief map average after each sample

p = hline([mean(all_vals15)],
       label="Sample Ave",
       width=4)
plot!(
    pred_means15[2:3],
    title="Mean Predicted Map Values",
    xlabel="Sample Number",
    ylabel="Height (mm)",
    marker=true,
    # ylim=(0,.5),
    titlefontsize=18,
    markersize=8,
    tickfontsize=14,
    labelfontsize=16,
    legendfontsize=14,
    margin=5mm,
    linewidth=4,
    framestyle=:box,
    labels=["Priors" "No Priors"],
    # size=(width, height)
)
# savefig(output_dir * "iros_2024/aves_15x15.png")
display(p)

p = hline([mean(all_vals50)],
          label="Sample Ave",
          width=4)
plot!(
    pred_means50[1:2],
    title="Mean Predicted Map Values",
    xlabel="Sample Number",
    ylabel="Height (mm)",
    marker=true,
    # ylim=(0,.5),
    titlefontsize=18,
    markersize=8,
    tickfontsize=14,
    labelfontsize=16,
    legendfontsize=14,
    margin=5mm,
    linewidth=4,
    framestyle=:box,
    labels=["Priors" "No Priors"],
    # size=(width, height)
)
# savefig(output_dir * "iros_2024/aves_50x50.png")
display(p)

#* estimated correlations with elevation

name_run15 = "30samples_15x15_priors"
name_run50 = "30samples_50x50_priors"

name = name_run15
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
cors15 = map(1:mission.num_samples) do i
    bm = MQGP([mission.prior_samples; samples[1:i]], bounds)
    outputCorMat(bm)[2,1]
end

name = name_run50
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
cors50 = map(1:mission.num_samples) do i
    bm = MQGP([mission.prior_samples; samples[1:i]], bounds)
    outputCorMat(bm)[2,1]
end

name = names[6]
file_name = output_dir * "pye_farm_trial_named/" * name * output_ext
data = load(file_name)
mission_grid = data["mission"]
samples = data["samples"]
bm = MQGP([mission.prior_samples; samples], bounds)
comp_cor = outputCorMat(bm)[2,1]

p = plot(
    cors15,
    title="Estimated Correlation to Elevation",
    xlabel="Sample Number",
    marker=true,
    ylim=(-1.05, 0),
    titlefontsize=18,
    markersize=8,
    tickfontsize=14,
    labelfontsize=16,
    legendfontsize=14,
    margin=5mm,
    linewidth=4,
    framestyle=:box,
    legend=false,
    # size=(width, height)
)
savefig(output_dir * "iros_2024/cors_15x15.png")
display(p)

p = hline([comp_cor],
          label="Sample Ave",
          width=4)
p = plot!(
    cors50,
    title="Estimated Correlation to Elevation",
    xlabel="Sample Number",
    marker=true,
    ylim=(-1.05, 0),
    titlefontsize=18,
    markersize=8,
    tickfontsize=14,
    labelfontsize=16,
    legendfontsize=14,
    margin=5mm,
    linewidth=4,
    framestyle=:box,
    legend=false,
    # size=(width, height)
)
# savefig(output_dir * "iros_2024/cors_50x50.png")
display(p)

#* look at data

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

name = names[5]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
beliefs = data["beliefs"]
bounds = getBounds(mission.occupancy)

samples[1:2]
beliefs[2]
mission.prior_samples

vis(beliefs[2], [], mission.occupancy; quantity=1)
outputCorMat(beliefs[2])

#* re-run with randomized and negative initial hyperparameters

gr()

name = names[5]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)

global_logger(ConsoleLogger(stderr, Debug))

new_beliefs = map(1:mission.num_samples) do i
    bm = MQGP([mission.prior_samples; samples[1:i]], bounds; σn=1)
    vis(bm, samples[1:i], mission.occupancy)
    @debug bm.θ.σ
    @debug outputCorMat(bm)
    sleep(1)
    bm
end

beliefModel = MQGP([mission.prior_samples[[1,end]]; samples[1:2]], bounds)

vis(beliefModel, samples[1:2], mission.occupancy)

outputCorMat(beliefModel)

T = floor(Int, sqrt(length(a)*2)) # (T+1)*T/2 in matrix

# cholesky factorization technique to create a free-form covariance matrix
# that is positive semidefinite
L = [(v<=u ? a[u*(u-1)÷2 + v] : 0.0) for u in 1:T, v in 1:T]
A = L*L' # lower triangular times upper

cov_mat = A + √eps()*I

cov_mat = fullyConnectedCovMat(beliefModel.θ.σ)
vars = diag(cov_mat)
cor_mat = @. cov_mat / √(vars * vars') # broadcast shorthand
# correlation matrix calculation is fine

using LinearAlgebra: diag
using AbstractGPs: logpdf

function outputCorMatVec(a)
    cov_mat = fullyConnectedCovMat(a)
    vars = diag(cov_mat)
    return @. cov_mat / √(vars * vars') # broadcast shorthand
end

mission.occupancy.bounds.lower
mission.occupancy.bounds.upper

X = getfield.([mission.prior_samples[[1,end]]; samples[1:2]], :x)
Y_vals = getfield.([mission.prior_samples[[1,end]]; samples[1:2]], :y)
Y_errs = 0.0
kernel = multiKernel

θ = (σ = [55.82592833071618, 32.44570139050765, 6.617717042512063],
     ℓ = -3.963171016628482,
     σn = 1.2337243317000441)
outputCorMatVec(θ.σ)
fx = MultiQuantityGPs.buildPriorGP(X, Y_errs, kernel, θ)
-logpdf(fx, Y_vals)

# negate parameter to make it negatively correlated
θ = (σ = [55.82592833071618, -32.44570139050765, 6.617717042512063],
     ℓ = -3.963171016628482,
     σn = 1.2337243317000441)
outputCorMatVec(θ.σ)
fx = MultiQuantityGPs.buildPriorGP(X, Y_errs, kernel, θ)
-logpdf(fx, Y_vals)
# unexpectedly, positively-correlated hyperparameters have greater marginal likelihood

#* plot the functions
plot(-100, 100) do v
    θ = (σ = [55.82592833071618, v, 6.617717042512063],
         ℓ = -3.963171016628482,
         σn = 1.2337243317000441)
    fx = MultiQuantityGPs.buildPriorGP(X, Y_errs, kernel, θ)
    -logpdf(fx, Y_vals)
end |> display

#* now look at straight correlation between points

bounds = getBounds(mission.occupancy)
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap(elev_img, bounds)

elev_vals = [elevMap(s.x[1]) for s in samples]

cor(getfield.(samples, :y), elev_vals)

bm = MQGP([mission.prior_samples; samples], bounds)
vis(bm, samples, mission.occupancy)
bm
outputCorMat(bm)

belief_vals = bm(tuple.(first.(getfield.(mission.prior_samples, :x)), 1))[1]

cor(belief_vals, getfield.(mission.prior_samples, :y))

# both are slightly negative

#* now train a gp on the matching points

elev_samples = Sample.(tuple.(first.(getfield.(samples, :x)), 2), elev_vals)

bm1 = MQGP([elev_samples; samples], bounds)
outputCorMat(bm1)

bm2 = MQGP([mission.prior_samples; samples], bounds)
outputCorMat(bm2)

# and with only two samples each

plot(
    visualize(bm1, samples, mission.occupancy; quantity=1),
    visualize(bm2, samples, mission.occupancy; quantity=1),
    layout=(2,1)
) |> display

bm1 = MQGP([elev_samples[1:2]; samples[1:2]], bounds)
outputCorMat(bm1)

bm2 = MQGP([mission.prior_samples[[1,end]]; samples[1:2]], bounds)
outputCorMat(bm2)

plot(
    visualize(bm1, samples[1:2], mission.occupancy; quantity=1),
    visualize(bm2, samples[1:2], mission.occupancy; quantity=1),
    layout=(2,1)
) |> display

#* belief model with multi-mean

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

name = names[4]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)
bm = MQGP([mission.prior_samples; samples], bounds)
vis(bm, samples, mission.occupancy)
axs, points = generateAxes(mission.occupancy)
pred, _ = bm(tuple.(vec(points), 1))
mean(s.y for s in samples), mean(pred)

# saveBeliefMapToPng(bm, Map(zeros(200, 200), bounds), "food_for_munch")

outputCorMat(bm)

@info "start"
new_beliefs = map(1:mission.num_samples) do i
    @info "sample" i
    ss = [mission.prior_samples; samples[1:i]]
    # bm = MQGP(ss, bounds)
    bm = data["beliefs"][i]
    vis(bm, samples[1:i], mission.occupancy)
    # X = getfield.(ss, :x)
    # Y = getfield.(ss, :y)
    # m = MultiQuantityGPs.multiMeanAve(X, Y)
    # @info "calculated means" m.f.(tuple.(0, 1:2))
    # @info "learned means" bm.θ.β
    @info "matrix" outputCorMat(bm)
    sleep(0.5)
    bm
end

ss = [mission.prior_samples; samples[1:end-1]]
X = getfield.(ss, :x)
Y = getfield.(ss, :y)
m = multiMeanAve(X, Y)
m.f.(tuple.(0, 1:2))

plot()

plot(getindex.(outputCorMat.(new_beliefs), 2), ylim=(-1.05,1.05))|>display

plot([abs(bm.θ.ℓ) for bm in new_beliefs])|>display

replay(vis, mission, samples; sleep_time=1.0)

# things are much better with that: multi-mean, no noise

#* measurement stability

a = [141.010421753, 141.062576294, 141.064041138, 141.022872925, 141.020812988,
     141.058609009, 141.023162842, 141.041168213, 141.050064087, 141.04574585,
     141.030960083, 141.018539429, 141.020690918, 140.999923706, 141.004730225,
     140.985870361, 141.024597168, 141.011825562, 140.938491821, 140.956359863,
     140.965713501, 140.994094849, 140.975067139, 140.945053101, 140.970718384,
     140.907211304, 140.852355957, 140.840408325, 140.847793579, 140.889160156,
     140.877670288, 140.902404785, 140.853591919, 140.818237305, 140.831817627,
     140.822967529, 140.830337524, 140.827209473, 140.794799805, 140.794189453,
     140.802505493, 140.788803101, 140.778030396, 140.805252075, 140.811309814,
     140.833129883, 140.78062439, 140.760787964, 140.822097778, 140.869232178,
     140.854690552, 140.810287476, 140.758300781, 140.774017334, 140.771362305,
     140.764022827, 140.720077515, 140.762039185, 140.72227478, 140.730651855,
     140.748580933, 140.75453186, 140.70854187, 140.699707031, 140.742675781,
     140.695007324, 140.707443237, 140.702133179, 140.723373413, 140.666244507,
     140.665435791, 140.660736084, 140.648406982, 140.667831421, 140.675964355,
     140.666213989]

l, h = extrema(a)
(h, l), h-l, std(a)
# over about 8 seconds, the most it changed was less than half a millimeter

#* correlation convergence over time compared to grid sampling

name = names[6]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)

new_beliefs = map(1:mission.num_samples) do i
    MQGP([mission.prior_samples; samples[1:i]], bounds)
end

[abs(bm.θ.ℓ) for bm in new_beliefs]

plot(getindex.(outputCorMat.(new_beliefs), 2), ylim=(-1.05,1.05))|>display

#* Uncertainty over time

name = names[5]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)

vals = map(1:mission.num_samples) do i
    ss = [mission.prior_samples; samples[1:i]]
    # bm = MQGP(ss, bounds)
    bm = data["beliefs"][i]
    axs, points = generateAxes(mission.occupancy)
    _, err = bm(tuple.(vec(points), 1))
    mean(err), maximum(err)
end

ave_unc = first.(vals)
max_unc = last.(vals)

plot([ave_unc, max_unc], marker=true, labels=["mean" "max"],
     xlabel="sample", ylabel="uncertainty")|>display

# savefig(output_dir * "iros_2024/$(name)_uncertainties.png")

# average end pasture heights

hs = map(1:6) do i
    name = names[i]
    data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
    mission = data["mission"]
    samples = data["samples"]
    bounds = getBounds(mission.occupancy)
    bm = MQGP([mission.prior_samples; samples], bounds)

    axs, points = generateAxes(mission.occupancy)
    pred, err = bm(tuple.(vec(points), 1))
    (mean.((pred, err)), maximum.((pred, err)))
end

((126.9648162088205, 11.547336363604536), (175.15609447238796, 20.00858466066348))

((137.2042787545911, 14.545342357661497), (196.37456355834598, 22.302127805035617))
((129.2400653745223, 16.15926177447316), (192.31228673481823, 23.77683393410874))
((135.60407199028992, 17.7953446994974), (210.10372920367024, 29.868852065807257))
((138.309657102973, 19.412740424002084), (270.1021874083624, 34.700488903150074))

((129.44429739368377, 12.32834854494204), (243.3937835676651, 18.041388383729156))
