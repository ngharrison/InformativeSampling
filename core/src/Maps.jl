"""
This module contains types for holding and handling real and synthetic data.
Mainly used by, but foundational for the other parts too.

Main public types and functions:
$(EXPORTS)
"""
module Maps

using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, TYPEDEF, EXPORTS

export Map, randomPoint, res, pointToCell, cellToPoint,
       generateAxes, Bounds, getBounds

"""
$(TYPEDEF)

The bounds of the region. Consists of the lower and upper bounds, each a list of
floating-point values.
"""
const Bounds = @NamedTuple begin
    lower::Vector{Float64}
    upper::Vector{Float64}
end

"""
A general type for holding multidimensional data (usually a matrix) along with
associated dimension bounds. It's main purpose is to handle the conversion
between world coordinates and grid indices internally. Converting between the
two representations treats rows as the first variable (x-axis), columns as
the second (y-axis), and so on.

Its typical use is to act as a 2D map of some value that can be sampled. A Map
will return the value of the grid cell that a given point falls within. In other
words, the map value is constant within each cell. One can also think of this as
a nearest-neighbor approximation.

Each cell index is treated as the center of its cell. Thus the map's lower bound
is at the center of the first cell and the map's upper bound is at the center of
the last cell.

Also made to function directly like a built-in N-dimensional array by sub-typing
and implementing the base methods.

Fields:
$(TYPEDFIELDS)
"""
struct Map{T1<:Any, N<:Any, A<:AbstractArray{T1, N}, T2<:Real} <: AbstractArray{T1, N}
    "N-dimensional array of data"
    data::A
    "vectors of lower and upper bounds, defaults to zeros and ones"
    bounds::Bounds

    function Map(data, bounds)
        length(bounds.lower) == length(bounds.upper) == ndims(data) ||
            throw(DimensionMismatch("lengths of bounds don't match data dimensions"))
        new{eltype(data), ndims(data), typeof(data), eltype(bounds.lower)}(data, bounds)
    end
end

function Map(data::AbstractArray{<:Any})
    bounds = (
        lower=zeros(ndims(data)),
        upper=ones(ndims(data))
    )
    return Map(data, bounds)
end

"""
Method accepts a single vector (the location), returns a scalar (the value at
that point).

# Examples
```julia
data = reshape(1:25, 5, 5)
bounds = (
    lower = [0.0, 0.0],
    upper = [1.0, 1.0]
)
m = Map(data, bounds)
m2 = Map(data) # bounds will be zero to one

x = [.2, .75]
val = m(x) # returns the value at a single 2D point
val2 = m[1,4] # can also use as if it's just the underlying matrix
```
"""
function (map::Map)(x)
    checkBounds(x, map)
    map[pointToCell(x, map)]
end

# make a map behave like an array
Base.size(m::Map) = size(m.data)
Base.IndexStyle(::Type{<:Map}) = IndexLinear()
Base.getindex(m::Map, i::Integer) = m.data[i]
Base.setindex!(m::Map, v, i::Integer) = (m.data[i] = v)

# change display
function Base.show(io::IO, map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.bounds.lower), $(map.bounds.upper)]")
end
function Base.show(io::IO, ::MIME"text/plain", map::Map{T1}) where T1
    print(io, "Map{$T1} [$(map.bounds.lower), $(map.bounds.upper)]:\n")
    show(io, "text/plain", map.data)
end

"""
$(TYPEDSIGNATURES)

Generates a random point in the map. Returns the location and its value.

# Examples
```julia
data = reshape(1:25, 5, 5)
map = Map(data)
rand(map)
```
"""
function Base.rand(map::Map)
    x = randomPoint(map)
    return x, map(x)
end

"""
$(TYPEDSIGNATURES)

Generates a random point in the map. Returns the location.
"""
function randomPoint(map::Map)
    dif = (map.bounds.upper .- map.bounds.lower)
    return map.bounds.lower .+ rand(ndims(map)) .* dif
end

"""
Get the lower and upper bounds of the map.
"""
getBounds(map::Map) = map.bounds

"""
$(TYPEDSIGNATURES)

Function emits error if location is outside of map bounds.

# Examples
```julia
x = [.2, .75]
data = reshape(1:25, 5, 5)
map = Map(data)
checkBounds(x, map) # no error thrown
```
"""
function checkBounds(x, map::Map)
    length(x) == ndims(map) ||
        throw(DomainError(x, "length doesn't match map dimensions: $(ndims(map))"))
    all(map.bounds.lower .<= x .<= map.bounds.upper) ||
        throw(DomainError(x, "out of map bounds: ($(map.bounds.lower), $(map.bounds.upper))"))
end

"""
$(TYPEDSIGNATURES)

Returns the resolution for each dimension of the given Map as a vector.
"""
function res(map)
    return (map.bounds.upper .- map.bounds.lower) ./ (size(map) .- 1)
end

"""
$(TYPEDSIGNATURES)

Takes in a point in world-coordinates and a Map and returns a CartesianIndex for
the underlying array.
"""
function pointToCell(x, map)
    dif = (x .- map.bounds.lower)
    return CartesianIndex(Tuple(round.(Int, dif ./ res(map)) .+ 1))
end

"""
$(TYPEDSIGNATURES)

Takes in a CartesianIndex and a Map and returns a point in world-coordinates.
"""
function cellToPoint(ci, map)
    return (collect(Tuple(ci)) .- 1) .* res(map) .+ map.bounds.lower
end

"""
$(TYPEDSIGNATURES)

Method to generate the x, y, etc. axes and points of a Map. Useful for plotting.
"""
generateAxes(map) = generateAxes(map.bounds, size(map))

function generateAxes(bounds, dims)
    axes = range.(bounds..., dims)
    points = collect.(Iterators.product(axes...))
    return axes, points
end

end
