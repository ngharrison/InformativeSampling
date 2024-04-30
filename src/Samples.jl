module Samples

using Optim: optimize, ParticleSwarm
using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, TYPEDEF

using ..Maps: Map

export Sample, MapsSampler, selectSampleLocation, takeSamples

"""
$(TYPEDEF)
Location of sample
"""
const Location = Vector{Float64}

"""
$(TYPEDEF)
Sample input, the combination of: ([`Location`](@ref), sensor index)
"""
const SampleInput = Tuple{Location, Int}

"""
$(TYPEDEF)
Value of sample measurement, the measurement mean and standard deviation
"""
const SampleOutput = Tuple{Float64, Float64}

"""
Struct to hold the input and output of a sample.

Fields:
$(TYPEDFIELDS)
"""
struct Sample{T}
    "the sample input, usually a location and sensor id"
    x::SampleInput
    "the sample output or observation, a scalar"
    y::T
end

"""
Handles samples of the form (location, quantity) to give the value from the
right map. Internally a tuple of Maps.

Constructor can take in a tuple or vector of Maps or each Map as a separate
argument.

# Examples
```julia
ss = MapsSampler(Map(zeros(5, 5)), Map(ones(5, 5)))

loc = [.2, .75]
ss(loc) # result: [0, 1]
ss((loc, 2)) # result: 1
```
"""
struct MapsSampler{T1<:Real}
    maps::Tuple{Vararg{Map{T1}}}
end

MapsSampler(maps::Map...) = MapsSampler(maps)
MapsSampler(maps::AbstractVector{<:Map}) = MapsSampler(Tuple(maps))

(ss::MapsSampler)(loc::Location) = [map(loc) for map in ss]
(ss::MapsSampler)((q, loc)::SampleInput) = ss[q](loc)

# make it behave like a tuple
Base.keys(m::MapsSampler) = keys(m.maps)
Base.length(m::MapsSampler) = length(m.maps)
Base.iterate(m::MapsSampler) = iterate(m.maps)
Base.iterate(m::MapsSampler, i::Integer) = iterate(m.maps, i)
Base.Broadcast.broadcastable(m::MapsSampler) = Ref(m) # don't broadcast
Base.IndexStyle(::Type{<:MapsSampler}) = IndexLinear()
Base.getindex(m::MapsSampler, i::Integer) = m.maps[i]

# change display
function Base.show(io::IO, ss::MapsSampler{T1}) where T1
    print(io, "MapsSampler{$T1}:")
    for map in ss
        print("\n\t", map)
    end
end

"""
$(TYPEDSIGNATURES)

Pulls a ground truth value from a given location and constructs a Sample object
to hold them both.

Inputs:
- `loc`: the location to sample
- `sampler`: a function that returns ground truth values
- `quantities`: (optional) a vector of integers which represent which
  quantities to sample, defaults to all of them

Outputs a vector of Samples containing input x and measurement y
"""
function takeSamples(loc, sampler)
    Y = sampler(loc) # get sample values at location for all quantities
    return [Sample((loc, q), y) for (q, y) in enumerate(Y)]
end

function takeSamples(loc, sampler, quantities)
    return [Sample((loc, q), sampler((loc, q))) for q in quantities]
end

"""
$(TYPEDSIGNATURES)

The optimization of choosing a best single sample location.

Inputs:
- `sampleCost`: a function from sample location to cost (x->cost(x))
- `lb`: map lower bounds
- `ub`: map upper bounds

Returns the sample location, a vector
"""
function selectSampleLocation(sampleCost, lb, ub)
    # in future could optimize for measured quantity as well
    loc0 = @. (ub - lb)/2 + lb
    opt = optimize(
        sampleCost,
        loc0,
        ParticleSwarm(; lower=lb, upper=ub, n_particles=40)
    )
    @debug "sample optimizer:" opt
    return opt.minimizer
end

end
