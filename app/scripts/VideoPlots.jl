
using InformativeSampling

using .Maps: generateAxes
using .Missions: Mission
using .MultiQuantityGPs: MQGP
using .Samples: Sample

using InformativeSamplingUtils
using .DataIO: output_dir, output_ext

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm
using Printf

pyplot()

# r is the range, contains the min and max values
function createColorbarTicks(r)
    ticks = [ceil(r[1], sigdigits=2),
             round((r[2]-r[1])/2 + r[1], sigdigits=2),
             floor(r[2], sigdigits=2)]
    return ticks, [@sprintf("%.1e", x) for x in ticks]
end

#* load mission
dir = "aus_ave"
mname = "111"
file_name = output_dir * "$dir/mission_$(mname)" * output_ext

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

for i in eachindex(beliefs)
    # GP maps
    axs, points = generateAxes(occ)
    pred_map, err_map = beliefs[i](tuple.(points, 1))

    # objective map
    sampleCost = mission.sampleCostType(
        mission.occupancy, samples[begin:i*num_quant], beliefs[i], quantities, mission.weights
    )
    obj_map = -sampleCost.(points)

    # blocked points
    pred_map[occ] .= NaN
    err_map[occ] .= NaN
    obj_map[occ] .= NaN

    pred_u_s = sort(unique(pred_map[.!occ]))
    pred_range = (pred_u_s[5], pred_u_s[end-5])
    pred_ticks = createColorbarTicks(pred_range)
    err_u_s = sort(unique(err_map[.!occ]))
    err_range = (err_u_s[5], err_u_s[end-5])
    err_ticks = createColorbarTicks(err_range)
    obj_u_s = sort(unique(obj_map[.!occ]))
    obj_range = (obj_u_s[5], obj_u_s[end-5])
    obj_ticks = createColorbarTicks(obj_range)

    p1 = heatmap(axs..., pred_map',
                 title="Predicted Values",
                 clim=(0,1),
                 colorbar_ticks=[0.0, 0.5, 1.0],
                 )
    scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
             label=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)

    p2 = heatmap(axs..., err_map',
                 title="Uncertainties",
                 clim=err_range,
                 colorbar=false,
                 )
    scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
             label=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)

    p3 = heatmap(axs..., mission.sampler[1]',
                 title="Ground Truth",
                 clim=(0,1),
                 colorbar_ticks=[0.0, 0.5, 1.0],
                 )

    p4 = heatmap(axs..., obj_map',
                 title="Sample Utility",
                 clim=obj_range,
                 colorbar=false,
                 )
    scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
             label=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)

    plot(p1, p2, p3, p4,
         layout=4,
         framestyle=:none,
         ticks=false,
         size=(1000, 800),
         titlefontsize=24,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         )

    # savefig(output_dir * "$dir/mission_$(mname)_frames/$(lpad(i, 2, '0')).png")
end

#* load missions
dir = "aus_ave"
mis_file_name = output_dir * "$dir/mission_000" * output_ext
mis_file_name_m = output_dir * "$dir/mission_111" * output_ext
met_file_name = output_dir * "$dir/metrics_000" * output_ext
met_file_name_m = output_dir * "$dir/metrics_111" * output_ext

mis_data = load(mis_file_name)
mission = mis_data["mission"]
beliefs = mis_data["beliefs"]
samples = mis_data["samples"]
met_data = load(met_file_name)
maes = met_data["metrics"].mae

mis_data_m = load(mis_file_name_m)
mission_m = mis_data_m["mission"]
beliefs_m = mis_data_m["beliefs"]
samples_m = mis_data_m["samples"]
met_data_m = load(met_file_name_m)
maes = [maes met_data_m["metrics"].mae]
cors = met_data_m["metrics"].cors

occ = mission.occupancy
quantities = eachindex(mission.sampler)
num_quant = length(mission.sampler)

xp = first.(getfield.(samples, :x))
x1 = getindex.(xp, 1)
x2 = getindex.(xp, 2)

xp_m = first.(getfield.(samples_m, :x))
x1_m = getindex.(xp_m, 1)
x2_m = getindex.(xp_m, 2)

axs, points = generateAxes(occ)

for i in eachindex(beliefs)
    # GP maps
    pred_map, _ = beliefs[i](tuple.(points, 1))
    pred_map_m, _ = beliefs_m[i](tuple.(points, 1))

    # blocked points
    pred_map[occ] .= NaN
    pred_map_m[occ] .= NaN

    p1 = heatmap(axs..., pred_map_m',
                 title="Predictions: All Priors",
                 framestyle=:none,
                 clim=(0,1),
                 colorbar_ticks=[0.0, 0.5, 1.0],
                 )
    scatter!(x1_m[begin:i*num_quant], x2_m[begin:i*num_quant];
             label=false,
             ticks=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)

    p2 = plot(
        hcat((c[2:end] for c in cors[1:i,:])...)',
        title="Correlation to Vegetation",
        labels=["Elevation" "Ground Temperature" "Rainfall"],
        seriescolors=[RGB(0.4, 0.2, 0.1) RGB(0.9, 0.4, 0.0) RGB(0.1,0.3,1)],
        # legend=:right,
        marker=true,
        xlim=(0,30),
        xticks=(5:5:30),
        ylim=(-1,1),
        markersize=8,
        linewidth=4,
        # size=(width, height)
    )

    p3 = heatmap(axs..., pred_map',
                 title="Predictions: No Priors",
                 framestyle=:none,
                 ticks=false,
                 clim=(0,1),
                 colorbar_ticks=[0.0, 0.5, 1.0],
                 )
    scatter!(x1[begin:i*num_quant], x2[begin:i*num_quant];
             label=false,
             color=:green,
             legend=(0.15, 0.87),
             markersize=8)

    p4 = plot(
        maes[1:i,:],
        title="Prediction Errors",
        xlabel="Sample Number",
        labels=["No Priors" "All Priors"],
        seriescolors=[:black RGB(0.1,0.7,0.2)],
        marker=true,
        xlim=(0,30),
        xticks=(5:5:30),
        ylim=(0,.4),
        markersize=8,
        linewidth=4,
        size=(width, height)
    )

    plot(p1, p2, p3, p4,
         size=(1000, 800),
         titlefontsize=24,
         tickfontsize=20,
         colorbar_tickfontsize=20,
         legendfontsize=14,
         labelfontsize=20,
         )

    # savefig(output_dir * "$dir/compare_frames/$(lpad(i, 2, '0')).png")
end
