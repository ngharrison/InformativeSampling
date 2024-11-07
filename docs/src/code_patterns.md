# Code Patterns

## Naming

The chosen naming conventions for the project:
- Types (data structures) are `UpperCamelCase`
- Functions (and callable types) are `lowerCamelCase`
- Simple constants are `CAPITAL_SNAKE_CASE`
- All other variables are `snake_case`

## Data structures and functions

A consistent pattern is followed for most data and functions in this repo using the Julia language feature of callable types. Defined data types can be given associated methods to make them callable. In many places in the code, you will find the following in order:
1. a struct definition -- what data is contained
2. zero to many constructor definitions -- how it is initialized
3. zero to many method definitions -- what it does when it is called (possibly with arguments)


As an example, here is a simplified version of the Map type:
```julia
# struct definition
struct Map
    data
    bounds
end

# constructor definition
Map(data) = Map(data, (lower=[0.0, 0.0], upper=[1.0, 1.0]))

# method definition, returns the value at that location
function (map::Map)(x::Location)
    checkBounds(x, map)
    map.data[pointToCell(x, map)]
end
```

It can then be used like:
```julia
data = reshape(1:25, 5, 5)
m = Map(data) # initialize through constructor

x = [.2, .75]
val = m(x) # call method to perform its function and get result
```

The type's methods are meant to be the primary purpose of that object. e.g. a `BeliefModel` returns the belief, a `SampleCost` returns the sample cost, etc. If an object's data are used in any secondary way, it will be its own separate function with its own name, and the object will be passed in directly. e.g. `quantityCorMat(beliefModel::BeliefModel)`.
