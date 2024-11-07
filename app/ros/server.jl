#!/usr/bin/env julia

# this file runs the services connected with informative sampling

using Pkg
Pkg.activate(Base.source_dir() * "/..")

using RobotOS
@rosimport informative_sampling.srv: GenerateBeliefModel, GenerateBeliefMaps,
                                     GenerateBeliefMapsFromModel, NextSampleLocation,
                                     BeliefMapsAndNextSampleLocation
@rosimport informative_sampling.msg:BeliefModelParameters
@rosimport std_msgs.msg: Float64MultiArray, MultiArrayLayout, MultiArrayDimension
rostypegen()
using .informative_sampling.srv
using .informative_sampling.msg: BeliefModelParameters
using .std_msgs.msg: Float64MultiArray, MultiArrayLayout, MultiArrayDimension

using InformativeSampling
using .Samples, .MultiQuantityGPs, .Maps, .SampleCosts

using Random: seed!

# takes in a function and returns a new function that does the same thing, but
# wraps it in a try-catch block, so errors are caught and printed
function catchErrors(func)
    return (args...; kwargs...) -> begin
        try
            func(args...; kwargs...)
        catch err
            println("Error caught: $err")
        end
    end
end

# callback functions
function handleGenerateBeliefModel(req)
    println("Generating belief model")

    seed!(0)

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    # default values will be (0.0, false)
    noise = (; req.noise.value, req.noise.learned)

    beliefModel = MQGP(samples, bounds; noise)

    params = BeliefModelParameters(beliefModel.θ...)
    return GenerateBeliefModelResponse(params)
end

function handleGenerateBeliefMaps(req)
    println("Generating belief maps")

    seed!(0)

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    # default values will be (0.0, false)
    noise = (; req.noise.value, req.noise.learned)

    beliefModel = MQGP(samples, bounds; noise)

    dims = Tuple(req.dims)
    quantity = req.quantity_index

    _, points = generateAxes(bounds, dims)
    μ, σ = beliefModel(tuple.(vec(points), quantity))

    belief_map, uncertainty_map = (
        Float64MultiArray(
            MultiArrayLayout(MultiArrayDimension.("", collect(dims), 1), 0),
            data
        )
        for data in (μ, σ)
    )

    return GenerateBeliefMapsResponse(belief_map, uncertainty_map)
end

function handleGenerateBeliefMapsFromModel(req)
    println("Generating belief maps from model")

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    params = (
        σ  = req.params.amplitudes,
        ℓ  = req.params.length_scale,
        σn = req.params.noise
    )

    beliefModel = MQGP(samples, params)

    bounds = (; req.bounds.lower, req.bounds.upper)
    dims = Tuple(req.dims)
    quantity = req.quantity_index

    _, points = generateAxes(bounds, dims)
    μ, σ = beliefModel(tuple.(vec(points), quantity))

    belief_map, uncertainty_map = (
        Float64MultiArray(
            MultiArrayLayout(MultiArrayDimension.("", collect(dims), 1), 0),
            data
        )
        for data in (μ, σ)
    )

    return GenerateBeliefMapsFromModelResponse(belief_map, uncertainty_map)
end

function handleNextSampleLocation(req)
    println("Choosing next sample location")

    seed!(0)

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    # default values will be (0.0, false)
    noise = (; req.noise.value, req.noise.learned)

    beliefModel = MQGP(samples, bounds; noise)

    dims = Tuple(d.size for d in req.occupancy.layout.dim)
    occupancy = Map(reshape(Bool.(req.occupancy.data), dims), bounds)

    sampleCost = DistScaledEIGF(
        occupancy, samples, beliefModel, req.quantities, req.weights
    )
    new_loc = selectSampleLocation(sampleCost, bounds)

    return NextSampleLocationResponse(new_loc)
end

function handleBeliefMapsAndNextSampleLocation(req)
    println("Generating belief maps and next sample location")

    seed!(0)

    samples = map(req.samples) do s
        Sample((s.location, s.quantity_index), s.measurement)
    end
    bounds = (; req.bounds.lower, req.bounds.upper)
    # default values will be (0.0, false)
    noise = (; req.noise.value, req.noise.learned)

    beliefModel = MQGP(samples, bounds; noise)

    dims = Tuple(d.size for d in req.occupancy.layout.dim)
    quantity = req.quantity_index

    _, points = generateAxes(bounds, dims)
    μ, σ = beliefModel(tuple.(vec(points), quantity))

    belief_map, uncertainty_map = (
        Float64MultiArray(
            MultiArrayLayout(MultiArrayDimension.("", collect(dims), 1), 0),
            data
        )
        for data in (μ, σ)
    )

    occupancy = Map(reshape(Bool.(req.occupancy.data), dims), bounds)

    sampleCost = DistScaledEIGF(
        occupancy, samples, beliefModel, req.quantities, req.weights
    )
    new_loc = selectSampleLocation(sampleCost, bounds)

    return BeliefMapsAndNextSampleLocationResponse(belief_map, uncertainty_map, new_loc)
end

function main()
    init_node("generate_belief_model_server")

    # create services
    Service("generate_belief_model", GenerateBeliefModel,
            catchErrors(handleGenerateBeliefModel))
    Service("generate_belief_maps", GenerateBeliefMaps,
            catchErrors(handleGenerateBeliefMaps))
    Service("generate_belief_maps_from_model", GenerateBeliefMapsFromModel,
            catchErrors(handleGenerateBeliefMapsFromModel))
    Service("next_sample_location", NextSampleLocation,
            catchErrors(handleNextSampleLocation))
    Service("belief_maps_and_next_sample_location", BeliefMapsAndNextSampleLocation,
            catchErrors(handleBeliefMapsAndNextSampleLocation))

    # wait for requests
    println("Ready to serve")
    spin()
end

main()
