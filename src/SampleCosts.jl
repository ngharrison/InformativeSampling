module SampleCosts

using LinearAlgebra: norm
using Statistics: mean
using DocStringExtensions: SIGNATURES

using Environment: pointToCell, cellToPoint, res
using Paths: PathCost

abstract type SampleCost end

"""
Cost to take a new sample at a location. This is a fallback method that
calculates a simple linear combination of all the values of a SampleCost.
"""
function (sc::SampleCost)(loc)
    vals = values(sc, loc)
    return sum(w*v for (w,v) in zip(sc.weights, vals))
end

"""
A basic cost function used for choosing a new sample location.
"""
struct BasicSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    pathCost
end

"""
$SIGNATURES

A pathCost is constructed automatically from the other arguments.

This object can then be called to get the cost of sampling at a location:
sampleCost(x)
"""
function BasicSampleCost(md, samples, beliefModel, quantities)
    start = pointToCell(samples[end].x[1], md.occupancy) # just looking at location
    pathCost = PathCost(start, md.occupancy, res(md.occupancy))
    BasicSampleCost(md.occupancy, samples, beliefModel,
                    quantities, md.weights, pathCost)
end

"""
Combines belief mean and standard deviation, travel distance,
and sample proximity.

Has the form:
cost = - w1 μ - w2 σ + w3 τ + w4 D
"""
function values(sc::BasicSampleCost, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_ave, σ_ave = mean.(beliefs)

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    radius = minimum(sc.occupancy.ub .- sc.occupancy.lb)/4
    dists = norm.(sample.x[1] - loc for sample in sc.samples)
    P = sum((radius./dists).^3) # proximity to other points

    return [-μ_ave, -σ_ave, τ, P]
end

struct NormedSampleCost <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function NormedSampleCost(md, samples, beliefModel, quantities)
    start = pointToCell(samples[end].x[1], md.occupancy) # just looking at location
    pathCost = PathCost(start, md.occupancy, res(md.occupancy))

    # using the max values from the current belief
    locs = [cellToPoint(ci, md.occupancy) for ci in vec(CartesianIndices(md.occupancy))]
    μ = zeros(quantities); σ = zeros(quantities)
    for q in quantities
        μ[q], σ[q] = maximum.(beliefModel(tuple.(locs, q)))
    end
    belief_max = (; μ, σ)

    NormedSampleCost(md.occupancy, samples, beliefModel,
                     quantities, md.weights, belief_max, pathCost)
end

function values(sc::NormedSampleCost, loc)
    beliefs = sc.beliefModel([(loc, q) for q in sc.quantities]) # means and standard deviations
    μ_ave, σ_ave = mean.(beliefs ./ Tuple(sc.belief_max))

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location

    # this value is now ignored thanks to log(σ)
    # radius = minimum(sc.occupancy.ub .- sc.occupancy.lb)/4
    # dists = norm.(sample.x[1] - loc for sample in sc.samples)
    # P = sum((radius./dists).^3) # proximity to other points

    return (-μ_ave, -log(σ_ave), τ)
end

end