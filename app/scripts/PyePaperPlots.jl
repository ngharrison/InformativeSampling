
using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples, .ROSInterface, .Kernels

using InformativeSamplingUtils
using .DataIO, .Visualization

using Statistics, FileIO, Images, Logging, Printf

using Plots

pyplot()

# r is the range, contains the min and max values
function createColorbarTicks(r)
    ticks = [ceil(r[1], sigdigits=3),
             round((r[2]-r[1])/2 + r[1], sigdigits=3),
             floor(r[2], sigdigits=3)]
    return ticks, [@sprintf("%.0f", x) for x in ticks]
end

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

#* data
name = names[2]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)

i = 2

pred_range = (Inf, -Inf)

for i in [1,3,4,5,6]
    name = names[i]
    data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
    mission = data["mission"]
    bounds = getBounds(mission.occupancy)

    axs, points = generateAxes(mission.occupancy)
    pred, _ = data["beliefs"][end](tuple.(vec(points), 1))
    global pred_range = (min(minimum(pred), pred_range[1]), max(maximum(pred), pred_range[2]))
end

pred_ticks = createColorbarTicks(pred_range)

titles = Dict(1=>"15x15 No Priors",
              3=>"15x15 With Priors",
              4=>"50x50 No Priors",
              5=>"50x50 With Priors")

#* plots
plts = map([1,3,4,5]) do i
    name = names[i]
    data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
    mission = data["mission"]
    samples = data["samples"]
    bounds = getBounds(mission.occupancy)
    # bm = MQGP([mission.prior_samples; samples], bounds)
    bm = data["beliefs"][end]

    axs, points = generateAxes(mission.occupancy)
    pred_map, err = bm(tuple.(points, 1))
    xp = first.(getfield.(samples, :x))
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    plt = heatmap(axs..., pred_map';
                        title=titles[i],
                        clim=pred_range,
                        # colorbar_ticks=pred_ticks,
                 # colorbar=false,
                 # title="Average Height (mm)",
                 # colorbar_titlefontsize=17
                 )
    scatter!(x1, x2;
             label=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)
    # gui(plt)
    plt
end

plot(plts[1], plts[3], plts[2], plts[4],
     framestyle=:none,
     ticks=false,
     size=(1000, 800),
     titlefontsize=21,
     colorbar_tickfontsize=20,
     legendfontsize=14,
     aspect_ratio=:equal
     )|>display

savefig(output_dir * "iros_2024/full_results.png")

bm = MQGP([mission.prior_samples; samples], bounds)

vis(bm, samples, mission.occupancy)
outputCorMat(bm)

#* static sampling

name = names[6]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)
# bm = MQGP([mission.prior_samples; samples], bounds)
bm = data["beliefs"][end]

axs, points = generateAxes(mission.occupancy)
pred_map, err_map = bm(tuple.(points, 1))
xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

plt = Plots.heatmap(axs..., pred_map';
                    title="50x50 Grid Sampling",
                    framestyle=:none,
                    ticks=false,
                    size=(600, 500),
                    titlefontsize=21,
                    colorbar_tickfontsize=20,
                    legendfontsize=14,
                    aspect_ratio=:equal
                    )
Plots.scatter!(x1, x2;
               label=false,
               color=:green,
               legend=(0.15, 0.87),
               markersize=8)
savefig(output_dir * "iros_2024/grid_sampling.png")
# display(plt)


#* now with makie

using GLMakie

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

name = names[2]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]
bounds = getBounds(mission.occupancy)

bm = data["beliefs"][end]
axs = range.(bounds..., size(mission.occupancy).+1)
points = collect.(Iterators.product(axs...))
pred_map, err_map = bm(tuple.(points, 1))
xp = (first.(getfield.(samples, :x)) .- Ref(bounds.lower)) .* Ref(bounds.upper .- bounds.lower)
xp = [(s.x[1] .- bounds.lower) .* () ./ (bounds.upper .- bounds.lower) for s in samples]
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)


f, ax, plt = heatmap((axes(pred_map).+1)..., pred_map, colormap=:plasma)
# Colorbar(f[1, 2], plt, width=20)
scatter!(ax, x1, x2, color=:green, markersize=15)
# hidedecorations!(ax)
# hidespines!(ax)
display(f)


# display(GLMakie.Screen(), f)
# GLMakie.closeall()
