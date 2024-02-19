# allows using modules defined in any file in project src directory
mod_dir = dirname(Base.active_project()) * "/src/modules"
if mod_dir ∉ LOAD_PATH
    push!(LOAD_PATH, mod_dir)
end

using Maps: Map
using Missions: Mission
using BeliefModels: BeliefModel
using Samples: Sample
using Outputs: output_dir, output_ext
using ROSInterface: ROSConnection
using Visualization
using Outputs: save, saveBeliefMapToPng

using Statistics: mean, std
using FileIO: load
using Plots
using Plots: mm

file_name = output_dir * "2024-02-15-12-02-46_mission.jld2"
file_name = output_dir * "2024-02-15-12-22-16_mission.jld2"
file_name = output_dir * "2024-02-15-12-56-06_mission.jld2"

data = load(file_name)
mission = data["mission"]
beliefs = data["beliefs"]
samples = data["samples"]
quantities = eachindex(mission.sampler)
sampleCost = mission.sampleCostType(mission, samples, beliefs[end], quantities)
display(visualize(mission, samples, beliefs[end], mission.occupancy.lb; quantity=1))


## concatenate samples
file_name1 = output_dir * "2024-02-15-17-26-06_mission.jld2"
file_name2 = output_dir * "2024-02-15-18-26-15_mission.jld2"

data1 = load(file_name1)
data2 = load(file_name2)
samples = [data1["samples"]; data2["samples"]]
mission = data1["mission"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub
beliefs = map(1:length(samples)) do i
    BeliefModel(samples[1:i], lb, ub)
end
save(mission, samples, beliefs)

## make png from proxy ground truth
name = "2024-02-19-13-27-06_mission"
file_name = output_dir * "$(name).jld2"
data = load(file_name)
mission = data["mission"]
samples = data["samples"]
lb, ub = mission.occupancy.lb, mission.occupancy.ub
beliefModel = BeliefModel(samples, lb, ub)

saveBeliefMapToPng(beliefModel, mission.occupancy, name)
