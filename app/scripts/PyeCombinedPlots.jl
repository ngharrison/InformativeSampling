#* starter stuff

using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples

using InformativeSamplingUtils
using .DataIO, .Visualization

using Statistics, FileIO, Plots, Images, Logging

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

pyplot()


#* create value ranges

pred_range = (Inf, -Inf)
err_range = (Inf, -Inf)

for i in 4:6
    name = names[i]
    data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
    mission = data["mission"]
    samples = data["samples"]
    bounds = getBounds(mission.occupancy)

    axs, points = generateAxes(mission.occupancy)
    bm = MQGP([mission.prior_samples; samples], bounds)
    pred, err = bm(tuple.(vec(points), 1))
    global pred_range = (min(minimum(pred), pred_range[1]), max(maximum(pred), pred_range[2]))
    global err_range = (min(minimum(err), err_range[1]), max(maximum(err), err_range[2]))
end

#* combined satellite, elevation, and learned height maps

bounds = [0.0, 0.0], [50.0, 50.0]
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap((elev_img.-minimum(elev_img)).*100, bounds)

sat_img = load(maps_dir * "satellite_50x50.tif")
satMap = imgToMap(sat_img, bounds)

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
xp = [s.x[1] for s in samples]
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

p1 = plot(sat_img;
          title="Satellite Image",
          aspect_ratio=:equal,
          )
plot!(x, y,
      legend=false,
      linewidth=3
      )
scatter!([], [];
         label=false,
         markersize=5)
p2 = heatmap(elevMap';
             title="Elevation Change (cm)",
             aspect_ratio=:equal,
             colorbar_tickfontsize=14,
             c=:oslo
             )
plot!(x./10, 50 .- y./10,
      legend=false,
      linewidth=3
      )
scatter!([], [];
         label=false,
         markersize=5)
p3 = heatmap(axs..., pred_map';
             title="Average Height (cm)",
             aspect_ratio=:equal,
             colorbar_tickfontsize=14,
             right_margin=-5Plots.mm,
             clim=pred_range
             )
scatter!(x1, x2;
         label=false,
         color=:green,
         markersize=5)
p = plot(p1, p2, p3;
         layout=@layout([[a; b] c]),
         size=(700, 600),
         titlefontsize=16,
         framestyle=:none,
         )
savefig(output_dir * "iros_2024/sat_elev_height_50x50_triangle.png")
# display(p)


#* two 50x50 runs

name = names[4]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]

bm = MQGP([mission.prior_samples; samples], bounds)
axs_4, points = generateAxes(mission.occupancy)
pred_map_4, err_map_4 = bm(tuple.(points, 1))
xp = first.(getfield.(samples, :x))
x1_4 = getindex.(xp, 1)
x2_4 = getindex.(xp, 2)

p1_4 = heatmap(axs_4..., pred_map_4';
             title="Average Height (mm)",
             clim=pred_range
             )
scatter!(x1_4, x2_4;
         label=false,
         color=:green,
         markersize=8)

p2_4 = heatmap(axs_4..., err_map_4';
             title="Uncertainties (mm)",
             clim=err_range
             )
scatter!(x1_4, x2_4;
         label=false,
         color=:green,
         markersize=8)

p = plot(p1_4, p2_4;
         layout=(1,2),
         framestyle=:none,
         ticks=false,
         size=(800, 350),
         titlefontsize=21,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         aspect_ratio=:equal
         )
savefig(output_dir * "iros_2024/$(name)_new.png")
# display(p)

name = names[5]
data = load(output_dir * "pye_farm_trial_named/" * name * output_ext)
mission = data["mission"]
samples = data["samples"]

bm = MQGP([mission.prior_samples; samples], bounds)
axs_5, points = generateAxes(mission.occupancy)
pred_map_5, err_map_5 = bm(tuple.(points, 1))
xp = first.(getfield.(samples, :x))
x1_5 = getindex.(xp, 1)
x2_5 = getindex.(xp, 2)

p1_5 = heatmap(axs_5..., pred_map_5';
             title="Average Height (mm)",
             clim=pred_range
             )
scatter!(x1_5, x2_5;
         label=false,
         color=:green,
         markersize=8)

p2_5 = heatmap(axs_5..., err_map_5';
             title="Uncertainties (mm)",
             clim=err_range
             )
scatter!(x1_5, x2_5;
         label=false,
         color=:green,
         markersize=8)

p = plot(p1_5, p2_5;
         layout=(1,2),
         framestyle=:none,
         ticks=false,
         size=(800, 350),
         titlefontsize=21,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         aspect_ratio=:equal
         )
savefig(output_dir * "iros_2024/$(name)_new.png")
# display(p)


#* all four in a grid

p1_5 = heatmap(axs_5..., pred_map_5';
               title="Average Height (mm)",
               ylabel="With Prior",
               clim=pred_range
               )
scatter!(x1_5, x2_5;
         label=false,
         color=:green,
         markersize=8)
p2_5 = heatmap(axs_5..., err_map_5';
               title="Uncertainties (mm)",
               clim=err_range
               )
scatter!(x1_5, x2_5;
         label=false,
         color=:green,
         markersize=8)

p1_4 = heatmap(axs_4..., pred_map_4';
               ylabel="Without Prior",
               clim=pred_range
               )
scatter!(x1_4, x2_4;
         label=false,
         color=:green,
         markersize=8)
p2_4 = heatmap(axs_4..., err_map_4';
               clim=err_range
               )
scatter!(x1_4, x2_4;
         label=false,
         color=:green,
         markersize=8)

p = plot(p1_5, p2_5, p1_4, p2_4;
         framestyle=:none,
         ticks=false,
         size=(850, 700),
         titlefontsize=21,
         labelfontsize=21,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         aspect_ratio=:equal
         )
savefig(output_dir * "iros_2024/50x50_results_grid.png")
# display(p)
