
using LinearAlgebra: logdet

using ..Maps: generateAxes
using ..Samples: Sample
using ..MultiQuantityGPs: MQGP, fullCov

@doc raw"""
Derived from the idea of information gain across the region. Returns the entropy
of a set of points (10x10 grid) given the new sample location. Minimizing this
entropy is equivalent to maximizing information gain since the entropy before
the sample is always the same.

This function is very computationally expensive, which is why the test grid is
set at 10x10.

It has the form:
```math
C(x) = \log |Σ|
```
"""
struct InfoGain <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
    test_points
end

function InfoGain(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    # this is set to low dimensions because its very expensive
    _, test_points = generateAxes(occupancy.bounds, (10, 10))
    InfoGain(occupancy, samples, beliefModel,
             quantities, weights, belief_max, pathCost, test_points)
end

function values(sc::InfoGain, loc)
    # add new sample with filler y-value since mean isn't used
    samples_c = [sc.samples; Sample((loc, 1), 0.0)]

    belief_model_c = MQGP(samples_c, sc.beliefModel.θ, sc.beliefModel.N; sc.beliefModel.kernel)

    Σ = fullCov(belief_model_c, tuple.(vec(sc.test_points), 1)) # get full covariance matrix

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    τ = isinf(τ) ? Inf : 0.0

    return (0.0, logdet(Σ), τ, 0.0)
end
