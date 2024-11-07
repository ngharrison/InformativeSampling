
using ..MultiQuantityGPs: meanDerivAndVar

@doc raw"""
Uses the norm of the derivative of the belief model mean and the belief model
variance, then scales it all by a normalized travel distance:
```math
C(x) = \frac{- w_1 \, {\left\lVert \frac{\partial μ}{\partial x}(x) \right\rVert}^2 - w_2 \, σ^2(x)}
       {1 + β \, \frac{τ(x)}{\left\lVert \boldsymbol{\ell}_d \right\rVert_1}}
```
where ``β`` is a parameter to delay the distance effect until a few samples have
been taken.
"""
struct DistScaledDerivVar <: SampleCost
    occupancy
    samples
    beliefModel
    quantities
    weights
    belief_max
    pathCost
end

function DistScaledDerivVar(occupancy, samples, beliefModel, quantities, weights)
    start = pointToCell(samples[end].x[1], occupancy) # just looking at location
    pathCost = PathCost(start, occupancy, res(occupancy))

    belief_max = nothing

    DistScaledDerivVar(occupancy, samples, beliefModel,
                     quantities, weights, belief_max, pathCost)
end

function values(sc::DistScaledDerivVar, loc)
    dμ, σ = meanDerivAndVar(sc.beliefModel, (loc, 1)) # mean and standard deviation

    τ = sc.pathCost(pointToCell(loc, sc.occupancy)) # distance to location
    bounds = getBounds(sc.occupancy)
    τ_norm = τ / mean(bounds.upper .- bounds.lower) # normalized

    d = (τ_norm == Inf ? Inf : 0.0)

    # gradually delay distance scaling
    n_scale = 2/(1 + exp(1 - length(sc.samples))) - 1
    d_scale = 1/(1 + n_scale*τ_norm^2)
    d_scale = isnan(d_scale) ? 1.0 : d_scale # prevent 0*Inf=NaN

    return (-dμ^2*d_scale, -σ^2*d_scale, d, 0.0)
end
