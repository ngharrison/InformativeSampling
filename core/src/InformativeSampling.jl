module InformativeSampling

# put all the code into the top namespace
include("Maps.jl")
include("Paths.jl")
include("Samples.jl")
include("Kernels.jl")
include("MultiQuantityGPs.jl")
include("SampleCosts.jl")
include("ROSInterface.jl")
include("Missions.jl")

export Maps, Paths, SampleCosts, Samples, MultiQuantityGPs, Kernels,
       ROSInterface, Missions

end
