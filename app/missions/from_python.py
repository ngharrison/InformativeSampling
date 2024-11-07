
# this script must be run from within the InformativeSampling directory for file
# paths to work

from julia import Pkg, Main

Pkg.activate('app')

from julia.InformativeSampling import Samples, MultiQuantityGPs, Maps, SampleCosts, Missions

# from julia.InformativeSamplingUtils import DataIO, Visualization

# from julia.LinearAlgebra import I, norm
# from julia.Statistics import mean, cor
# from julia.Random import seed_b

# test creating simple belief model
samples = [
    Samples.Sample(([.1, .8], 1), 1.4),
    Samples.Sample(([.5, .4], 1), .2),
    Samples.Sample(([.2, .2], 2), .8),
    Samples.Sample(([.9, .1], 2), .1),
]

bounds = Main.eval('(lower=[0.0, 0.0], upper=[1.0, 1.0])')

noise = Main.eval('(value=0.0, learned=false)')

beliefModel = MultiQuantityGPs.MQGP(samples, bounds, noise=noise)

print(beliefModel.θ)

#* test running full julia script

Main.include('app/missions/example.jl')

print(Main.samples)

print(Main.beliefs[-1].θ)
