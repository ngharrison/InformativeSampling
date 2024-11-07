"""
This module contains functions for initializing mission data and the function
for running the entire search mission. The entry-point to the actual informative
sampling. This contains the main loop and most of the usage of Samples and
belief models.

Main public types and functions:
$(EXPORTS)
"""
module Missions

using Random: seed!
using Statistics: median
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, EXPORTS

using ..Maps: randomPoint, getBounds
using ..Samples: Sample, selectSampleLocation, takeSamples
using ..MultiQuantityGPs: MQGP, quantityCorMat
using ..Kernels: multiKernel
using ..SampleCosts: values, DistScaledEIGF

export Mission, replay

"""
Fields:
$(TYPEDFIELDS)

Defined as a keyword struct, so all arguments are passed in as keywords:
```julia
mission = Mission(; occupancy,
                  sampler,
                  num_samples,
                  sampleCostType,
                  weights,
                  start_locs,
                  prior_samples,
                  noise,
                  kernel)
```
"""
@kwdef struct Mission
    "an occupancy map, true in cells that are occupied"
    occupancy
    "a function that returns a measurement value for any input"
    sampler
    "the number of samples to collect in one run"
    num_samples
    "a constructor for the function that returns the (negated) value of taking a sample (default DistScaledEIGF)"
    sampleCostType = DistScaledEIGF
    "weights for picking the next sample location"
    weights
    "the locations that should be sampled first (default [])"
    start_locs = []
    "any samples taken previously (default empty)"
    prior_samples = Sample[]
    "the kernel to be used in the belief model (default multiKernel)"
    kernel = multiKernel
    "whether or not to use a non-zero mean for each quantity and to learn means (default (true, false))"
    means = (use=true, learned=false)
    "a named tuple of noise value(s) and if learned further (default (0.0, false))"
    noise = (value=0.0, learned=false)
    "whether or not to use the conditional distribution of the data to train the belief model (default false)"
    use_cond_pdf = false
    "whether or not to drop hypotheses and settings for it (default (false, 10, 5, 0.4))"
    hyp_drop = (dropout=false, start=10, num=5, threshold=0.4)
end

"""
The main function that runs the informative sampling routine. For each iteration, a
sample location is selected, a sample is collected, the belief model is updated,
and visuals are possibly shown. The run finishes when the designated number of
samples is collected.

Inputs:
- `func`: any function to be run at the end of the update loop, useful for
  visualization or saving data (default does nothing)
- `samples`: a vector of samples, this can be used to jump-start a mission
  or resume a previous mission (default empty)
- `beliefs`: a vector of beliefs, this pairs with the previous argument
  (default empty)
- `seed_val`: the seed for the random number generator, an integer (default 0)
- `sleep_time`: the amount of time to wait after each iteration, useful for
  visualizations (default 0)

Outputs:
- `samples`: a vector of new samples collected
- `beliefs`: a vector of probabilistic representations of the quantities being
  searched for, one for each sample collection

# Examples
```julia
using Missions: synMission

mission = synMission(num_samples=10) # create the specific mission
samples, beliefs = mission(visuals=true, sleep_time=0.5) # run the mission
```
"""
function (M::Mission)(func=Returns(nothing);
                      samples=Sample[], beliefs=MQGP[],
                      other_samples=Sample[], times=Float64[], cors=Vector{Float64}[],
                      seed_val=0, sleep_time=0)

    # initialize
    seed!(seed_val)
    new_loc = !isempty(M.start_locs) ? M.start_locs[1] : randomPoint(M.occupancy)

    bounds = getBounds(M.occupancy)
    quantities = 1:1 # only first quantity

    prior_samples = copy(M.prior_samples)
    num_q = length(quantities) # number of quantities being sampled
    N = maximum(s->s.x[2], prior_samples, init=num_q) # defaults to num_q
    prior_quantities = collect(range(num_q+1, N)) # will be empty if no prior_quantities

    println("Mission started")

    for i in 1:M.num_samples
        println()
        println("Sample number $i")

        println("Next sample location: $new_loc")

        # prevent invalid locations
        M.occupancy(new_loc) && error("sample location is within obstacle")

        # sample all quantities
        new_samples = takeSamples(new_loc, M.sampler)
        push!(samples, new_samples[1]) # the one we act on
        append!(other_samples, new_samples[2:end]) # the ones we don't
        println("Sample values: $(getfield.(new_samples, :y))")

        t = @elapsed begin # computation time
            # new belief
            beliefModel = MQGP([prior_samples; samples], bounds;
                                      N, M.kernel, M.means, M.noise, M.use_cond_pdf)
            push!(beliefs, beliefModel)

            # calculate correlations to first
            c = quantityCorMat(beliefModel)[:,1]
            push!(cors, c)

            # hypothesis dropout chosen by coefficient of determination
            if (M.hyp_drop.dropout
                && M.hyp_drop.start <= i < M.num_samples
                && !isempty(prior_quantities))

                # get coefficients of determination over past five samples
                recent_cors = cors[(i-M.hyp_drop.num+1):i] # vector of vectors
                recent_coeffs = [r.^2 for r in recent_cors]

                # transpose it around so it's by quantity
                recent_coeffs_q = [getindex.(recent_coeffs, q) for q in eachindex(recent_coeffs[1])]

                # choose worst quantity by most recent
                worst_q = argmin(q->recent_coeffs[end][q], prior_quantities)
                others_q = filter(!=(worst_q), prior_quantities)

                # uniformly below threshold or uniformly below another quantity
                if (all(recent_coeffs_q[worst_q] .< M.hyp_drop.threshold)
                    || any(all(recent_coeffs_q[worst_q] .< other_recent_coeffs)
                           for other_recent_coeffs in recent_coeffs_q[others_q]))
                    filter!(!=(worst_q), prior_quantities)
                    filter!(s -> s.x[2] != worst_q, prior_samples)
                end
            end

            # new sample location
            sampleCost = nothing
            new_loc = nothing
            if i < M.num_samples
                if i < length(M.start_locs)
                    new_loc = M.start_locs[i+1]
                else
                    sampleCost = M.sampleCostType(
                        M.occupancy, samples, beliefModel, quantities, M.weights
                    )
                    new_loc = selectSampleLocation(sampleCost, bounds)

                    @debug "cost function values: $(Tuple(values(sampleCost, new_loc)))"
                    @debug "cost function weights: $(Tuple(M.weights))"
                    @debug "cost function terms: $(Tuple(values(sampleCost, new_loc)) .* Tuple(M.weights))"
                    @debug "cost function value: $(sampleCost(new_loc))"
                end
            end
        end
        push!(times, t)

        # user-defined function (visualization, saving, etc.)
        func(M, [samples; other_samples], beliefModel, sampleCost, new_loc)
        @debug "belief model hyperparams: " beliefModel.θ
        @debug "output correlations: " c
        sleep(sleep_time)
    end

    println()
    println("Mission complete")

    return [samples; other_samples], beliefs, cors, times
end

"""
$(TYPEDSIGNATURES)

Replays a mission that has already taken place. Mainly for visualization
purposes.

Inputs:
- `func`: any function to be run at the end of the update loop, useful for
visualization or saving data (default does nothing)
- `full_samples`: a vector of samples
- `beliefs`: a vector of beliefs
- `sleep_time`: the amount of time to wait after each iteration, useful for
visualizations (default 0)
"""
function replay(func, M::Mission, full_samples, beliefs; sleep_time=0.0)
    quantities = eachindex(M.sampler) # all current available quantities

    println("Mission started")

    for i in eachindex(full_samples)
        println()
        println("Sample number $i")

        sample = full_samples[i]

        new_loc = sample.x[1]
        println("Next sample location: $new_loc")

        new_vals = sample.y
        println("Sample values: $(new_vals)")

        samples = full_samples[1:i]

        beliefModel = beliefs[i]
        sampleCost = M.sampleCostType(
            M.occupancy, samples, beliefModel, quantities, M.weights
        )

        new_loc = i < M.num_samples ? full_samples[i+1].x[1] : nothing
        func(M, samples, beliefModel, sampleCost, new_loc)
        @debug "output correlation matrix:" quantityCorMat(beliefModel)
        sleep(sleep_time)
    end

    println()
    println("Mission complete")
end

function replay(M::Mission, full_samples, beliefs; sleep_time=0.0)
    replay(Returns(nothing), M, full_samples, beliefs; sleep_time)
end

function replay(func, M::Mission, full_samples; sleep_time=0.0)
    quantities = 1:1 # only first quantity
    num_q = length(quantities) # number of quantities being sampled
    N = maximum(s->s.x[2], M.prior_samples, init=num_q) # defaults to num_q
    beliefs = map(1:length(full_samples)) do i
        MQGP([M.prior_samples; full_samples[1:i]], getBounds(M.occupancy);
                    N, M.kernel, M.means, M.noise, M.use_cond_pdf)
    end
    replay(func, M, full_samples, beliefs; sleep_time)
end

function replay(M::Mission, full_samples; sleep_time=0.0)
    replay(Returns(nothing), M, full_samples; sleep_time)
end

function medianAndAbsoluteDeviation(X::AbstractArray)
    medX = median(X)
    return medX, median(abs.(X .- medX))
end

function medianAndAbsoluteDeviation(Xs::AbstractArray{<:AbstractArray})
    medXs = median.(Xs)
    return medXs, median.(abs.(X .- medX) for (X, medX) in zip(Xs, medXs))
end

end
