# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

# change working directory to where this file is
# this is important for file paths to be used right
cd(Base.source_dir())

using Initialization: simData, realData, conradData, rosData
using BeliefModels: outputCorMat
using Visualization: visualize
using Exploration: explore

## initialize data for mission
missionData = simData()

## run search alg
@time samples, beliefModel = explore(missionData; visuals=true, sleep_time=0.0);

println()
println("Output correlations:")
display(outputCorMat(beliefModel))
