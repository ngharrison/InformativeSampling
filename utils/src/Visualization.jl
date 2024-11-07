"""
A module with methods to visualize mission data. The form displayed is
determined by what data types are passed in.

Main public types and functions:
$(EXPORTS)
"""
module Visualization

using Plots: plot, heatmap, scatter!, @layout, mm, grid
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, EXPORTS

using InformativeSampling
using .Maps: Map, res, generateAxes, getBounds
using .MultiQuantityGPs: MQGP
using .SampleCosts: SampleCost

export visualize, vis

# placeholders to avoid recomputing
axes = nothing
points = nothing

# default properties for all plots
default_res = [0.01, 0.01]
default_size = (1200, 800)
default_markersize = 4
default_margin = 5mm
sample_color = :green
new_sample_color = :red

"""
$(TYPEDSIGNATURES)

Main method to visualize the current state of a search. Generates all the other
visuals through their respective methods and lays them out in a grid. Currently
shows the belief model, the ground truth, and the obstacles.

Arguments pass through to the sub-methods that need them. res is the grid
resolution when plotting continuous-valued functions and defaults to $(default_res).

If no ground truth is available, it is not plotted.
"""
function visualize(md, samples, beliefModel::MQGP, sampleCost, new_loc; quantity=1)
    a = visualize(beliefModel, samples, md.occupancy, new_loc; quantity)
    b = plot(legend=false, grid=false, foreground_color_subplot=:white) # blank plot
    # TODO this will need to be updated to test for actual types
    if hasmethod(getindex, Tuple{typeof(md.sampler), Integer})
        # we actually have ground truth
        b = visualize(md.sampler[quantity], "Quantity of Interest")
    end
    c = visualize(sampleCost, samples, md.occupancy, new_loc)
    l = @layout [a ; b c]
    plot(a, b, c, layout=l, size=default_size, margin=default_margin)
end

"""
$(TYPEDSIGNATURES)
"""
function visualize(md, samples, beliefModel::MQGP, new_loc; quantity=1)
    sampleCost = md.sampleCostType(
        md.occupancy, samples, beliefModel, eachindex(md.sampler), md.weights
    )
    a = visualize(beliefModel, samples, md.occupancy, new_loc; quantity)
    b = plot(legend=false, grid=false, foreground_color_subplot=:white) # blank plot
    # TODO this will need to be updated to test for actual types
    if hasmethod(getindex, Tuple{typeof(md.sampler), Integer})
        # we actually have ground truth
        b = visualize(md.sampler[quantity], "Quantity of Interest")
    end
    c = visualize(sampleCost, samples, md.occupancy, new_loc)
    l = @layout [a ; b c]
    plot(a, b, c, layout=l, size=default_size, margin=default_margin)
end

"""
$(TYPEDSIGNATURES)

Method to show a ground truth map and up to three other prior data maps.
Pass each map in as its own argument.
"""
function visualize(maps::Map...;
                   titles=["Quantity of Interest"; ["Prior $i" for i in 1:length(maps)-1]],
                   points=[])
    plot(
        visualize(maps[1], titles[1]),
        visualize.(maps[2:end], titles[2:end]; points)...,
        layout=grid(2,2),
        size=default_size,
        margin=default_margin
    )
end

## functions to visualize individual pieces

"""
$(TYPEDSIGNATURES)

Method to show any Map data.
"""
function visualize(map::Map, title="Map"; points=[], clim=nothing)
    axes = range.(getBounds(map)..., size(map))
    plt = heatmap(axes..., map';
                  xlabel="x1",
                  ylabel="x2",
                  title,
                  clim
                  )
    if !isempty(points)
        scatter!(first.(points), last.(points);
                 label="Samples",
                 color=sample_color,
                 markersize=default_markersize)
    end
    return plt
end

"""
$(TYPEDSIGNATURES)

Method to show ground truth data from a function.
"""
function visualize(sampler, map)
    axes, points = getAxes(map)
    data = sampler(points)
    heatmap(axes..., data';
            xlabel="x1",
            ylabel="x2",
            title="Ground Truth"
            )
end

"""
$(TYPEDSIGNATURES)

Method to show belief model values of mean and standard deviation and the sample
locations that they were generated from. Shows two plots side-by-side.
"""
function visualize(beliefModel::MQGP, samples, occupancy, new_loc=nothing; quantity=1)
    axes, points = getAxes(occupancy)
    pred_map, err_map = beliefModel(tuple.(points, quantity))

    # blocked points
    pred_map[occupancy] .= NaN
    err_map[occupancy] .= NaN

    xp = first.(getfield.(filter(s->s.x[2]==quantity, samples), :x))
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    p1 = heatmap(axes..., pred_map')
    scatter!(x1, x2;
             xlabel="x1",
             ylabel="x2",
             title="GP Mean",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    if new_loc != nothing
        scatter!([new_loc[1]], [new_loc[2]];
                 color=new_sample_color,
                 markersize=default_markersize)
    end


    p2 = heatmap(axes..., err_map')
    scatter!(x1, x2;
             xlabel="x1",
             ylabel="x2",
             title="GP Std",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    if new_loc != nothing
        scatter!([new_loc[1]], [new_loc[2]];
                 color=new_sample_color,
                 markersize=default_markersize)
    end

    plot(p1, p2)
end


"""
$(TYPEDSIGNATURES)

Method to show sample cost values.
"""
function visualize(sampleCost, samples, occupancy, new_loc=nothing)
    isnothing(sampleCost) && return plot()

    axes, points = getAxes(occupancy)
    data = -sampleCost.(points)

    # blocked points
    data[occupancy] .= NaN

    xp = first.(getfield.(samples, :x))
    x1 = getindex.(xp, 1)
    x2 = getindex.(xp, 2)

    plt = heatmap(axes..., data')
    scatter!(x1, x2;
             xlabel="x1",
             ylabel="x2",
             title="Obj Function",
             legend=nothing,
             color=sample_color,
             markersize=default_markersize)
    if new_loc != nothing
        scatter!([new_loc[1]], [new_loc[2]], color=new_sample_color, markersize=default_markersize)
    end

    return plt
end

"""
$(TYPEDSIGNATURES)

Method to get the x and y plotting axes. This (re)generates them only if needed
and saves them into global module variables for future use.
"""
function getAxes(map)
    global axes
    global points
    global default_res
    if res(map) !== default_res # recompute
        default_res = res(map)
        axes, points = generateAxes(map)
    elseif axes === nothing # recompute
        axes, points = generateAxes(map)
    end
    return axes, points
end

"""
$(TYPEDEF)

Simple convenience function that also displays the output of any visualize
function. See those.
"""
const vis = display ∘ visualize

end
