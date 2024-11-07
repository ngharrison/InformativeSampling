
using InformativeSampling
using .Maps, .Missions, .MultiQuantityGPs, .Samples, .Kernels

using InformativeSamplingUtils
using .DataIO, .Visualization

using Statistics, FileIO, GLMakie, Images, Logging
using GLMakie: Axis, save

names = ["30samples_15x15_1", "30samples_15x15_2", "30samples_15x15_priors",
         "30samples_50x50", "30samples_50x50_priors", "100samples_50x50_grid"]

#* combined satellite, elevation, and learned height maps
bounds = [0.0, 0.0], [50.0, 50.0]
elev_img = Float64.(gray.(load(maps_dir * "dem_50x50.tif")))
elevMap = imgToMap((elev_img.-minimum(elev_img)).*100, bounds)

sat_img = load(maps_dir * "satellite_50x50.tif")
satMap = imgToMap(sat_img, bounds)

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
xp = [(s.x[1] .- bounds.lower) .* size(occupancy) ./ (bounds.upper .- bounds.lower) for s in samples]
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

f = Figure(size=(1040, 320))
rowsize!(f.layout, 1, Aspect(1, 1.0))
# colgap!(f.layout, 10)
ax = Any[nothing for _ in 1:3] # for heatmaps
ax[1], hm = heatmap(
    f[1,1], satMap;
    axis=(title="Satellite Image", titlesize=24),
)
Box(f[1,2]; width=15, visible=false)
ax[2], hm = heatmap(
    f[1,3], elevMap;
    axis=(title="Elevation Change (mm)", titlesize=24),
    colormap=:oslo,
)
Colorbar(f[1,4], hm)
Box(f[1,5]; width=5, visible=false)
ax[3], hm = heatmap(
    f[1,6], pred_map;
    axis=(title="Estimated Height (mm)", titlesize=24),
    colormap=:plasma,
)
scatter!(ax[3], x1, x2; color=:green, strokewidth=1)
Colorbar(f[1,7], hm)
hidedecorations!.(ax)
hidespines!.(ax)
# display(f)
save(output_dir * "iros_2024/sat_elev_height_50x50.png", f)
