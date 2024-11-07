
using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples, .ROSInterface

using InformativeSamplingUtils
using .DataIO

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm
using Printf

pyplot()

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors"]

#* load mission
dir = "pye_farm_trial_named"
mname = names[5]
file_name = output_dir * "$dir/$(mname)" * output_ext

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

i = 30
# GP maps
axs, points = generateAxes(occ)
pred_map, err_map = beliefs[i](tuple.(points, 1))

#* plots
p1 = heatmap(axs..., pred_map';
             title="Average Height (mm)",
             colorbar_titlefontsize=17
             )
scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
         label=false,
         color=:green,
         legend=(0.15, 0.87),
         markersize=8)

p2 = heatmap(axs..., err_map';
             title="Uncertainty (mm)",
             colorbar_titlefontsize=17
             )
scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
         label=false,
         color=:green,
         legend=(0.15, 0.87),
         markersize=8)

p = plot(p1, p2,
         layout=(1,2),
         framestyle=:none,
         ticks=false,
         size=(800, 350),
         titlefontsize=21,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         aspect_ratio=:equal
         )
display(p)

# p = plot(p1, p2,
#          layout=(2,1),
#          framestyle=:none,
#          ticks=false,
#          size=(450, 800),
#          titlefontsize=21,
#          colorbar_tickfontsize=20,
#          legendfontsize=14,
#          aspect_ratio=:equal
#          )
# display(p)
