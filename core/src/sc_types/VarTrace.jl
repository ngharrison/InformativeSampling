
using LinearAlgebra: logdet

using ..Maps: generateAxes
using ..Samples: Sample
using ..BeliefModels: BeliefModel

struct VarTrace <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
    test_points
end

function VarTrace(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    # this is set to low dimensions because its very expensive
    _, test_points = generateAxes(occupancy.bounds, (20, 20))
    VarTrace(occupancy, samples, beliefModel,
             quantities, weights, belief_max, pathCost, test_points)
end

function values(sc::VarTrace, loc)
    # add new sample with filler y-value since mean isn't used
    samples_c = [sc.samples; Sample((loc, 1), 0.0)]

    belief_model_c = BeliefModel(samples_c, sc.beliefModel.θ, sc.beliefModel.N; sc.beliefModel.kernel)

    _, σ = belief_model_c(tuple.(sc.test_points, 1)) # get all variances

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    return (0.0, sum(s^2 for s in σ), τ, 0.0)
end