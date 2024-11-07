
using InformativeSampling

using InformativeSamplingUtils
using .DataIO, .Visualization

using FileIO, JLD2

#* load data

dir = "pye_farm_trial2/"
file_names = readdir(output_dir * dir * "dense")

samples = []
for file_name in file_names
    data = load(output_dir * dir * "dense/" * file_name)
    append!(samples, data["metrics"])
end

#* save

jldsave(output_dir * dir * file_names[end]; samples)


#* package mission data

using OrderedCollections: OrderedDict
using YAML

# maintain printed form of arrays in yaml
struct ArrayWrapper{T<:AbstractArray}
    a::T
end
Base.show(io::IO, a::ArrayWrapper{T}) where T = print(io, "$(a.a)")

dir = "pye_farm_trial2/"
file_names = filter(n->!any(contains.(n, ("strange", "grid"))), readdir(output_dir * dir * "named"))

for file_name in file_names
    data = load(output_dir * dir * "named/" * file_name)
    m = data["mission"]

    n = OrderedDict(key => getfield(m, key) for key ∈ fieldnames(typeof(m)))
    n[:start_locs] = ArrayWrapper(n[:start_locs])

    base_name = splitext(file_name)[1]
    mkpath(output_dir * dir * "packaged/$(base_name)/")
    YAML.write_file(output_dir * dir * "packaged/$(base_name)/"
                    * "mission.yml", n)
end

#* save belief model maps and samples

using DelimitedFiles

using .MultiQuantityGPs: MQGP

dir = "pye_farm_trial2/"
file_names = filter(n->!any(contains.(n, ("strange", "grid"))),
                    readdir(output_dir * dir * "named"))

for file_name in file_names
    base_name = splitext(file_name)[1]

    data = load(output_dir * dir * "named/" * file_name)
    mission = data["mission"]
    samples = data["samples"]
    beliefs = data["beliefs"]

    new_samples = (s->((s.x...,), s.y)).(samples)
    writedlm(output_dir * dir * "packaged/" * base_name * "/samples.txt", [new_samples], "\n")

    bm_params = (bm->bm.θ).(beliefs)
    writedlm(output_dir * dir * "packaged/" * base_name * "/belief_params.txt", [bm_params], "\n")

    belief_maps = produceMaps(beliefs[end], mission.occupancy)
    m, s = mapToImg.(belief_maps)

    writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_belief.csv", m, ',')
    writedlm(output_dir * dir * "packaged/" * base_name * "/avg_height_uncertainty.csv", s, ',')

    ndvi_samples = filter(s->s.x[2]==2, samples)
    if !isempty(ndvi_samples)
        bm = MQGP(ndvi_samples, mission.occupancy.bounds)

        belief_maps = produceMaps(bm, mission.occupancy; quantity=2)
        m, s = mapToImg.(belief_maps)

        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_ndvi_belief.csv", m, ',')
        writedlm(output_dir * dir * "packaged/" * base_name * "/avg_ndvi_uncertainty.csv", s, ',')
    end
end

#* save dense grid maps

file_name = output_dir * dir * "named/50x50.jld2"
data = load(file_name)
mission = data["mission"]

file_name = output_dir * dir * "named/50x50_dense_grid.jld2"
data = load(file_name)
samples = data["samples"]

base_name = output_dir * dir * "packaged/50x50_dense_grid"
mkpath(base_name)

new_samples = (s->((s.x...,), s.y)).(samples)
writedlm(base_name * "/samples.txt", [new_samples], "\n")

height_samples = filter(s->s.x[2]==1, samples)
ndvi_samples = filter(s->s.x[2]==2, samples)

using .Maps

m_grid = Map(zeros(25, 25), mission.occupancy.bounds)
for s in height_samples
    m_grid[pointToCell(s.x[1], m_grid)] = s.y
end
writedlm(base_name * "/avg_height.csv", mapToImg(m_grid), ',')

m_grid = Map(zeros(25, 25), mission.occupancy.bounds)
for s in ndvi_samples
    m_grid[pointToCell(s.x[1], m_grid)] = s.y
end
writedlm(base_name * "/avg_ndvi.csv", mapToImg(m_grid), ',')

bm = MQGP(height_samples, mission.occupancy.bounds; mission.noise)
belief_maps = produceMaps(bm, mission.occupancy)
m, s = mapToImg.(belief_maps)

writedlm(base_name * "/avg_height_belief.csv", m, ',')
writedlm(base_name * "/avg_height_uncertainty.csv", s, ',')

bm = MQGP(ndvi_samples, mission.occupancy.bounds; mission.noise)
belief_maps = produceMaps(bm, mission.occupancy; quantity=2)
m, s = mapToImg.(belief_maps)

writedlm(base_name * "/avg_ndvi_belief.csv", m, ',')
writedlm(base_name * "/avg_ndvi_uncertainty.csv", s, ',')

#* look at correlation between height and ndvi

using Statistics

vals = map((height_samples, ndvi_samples)) do a
    getfield.(sort(a, by=s->s.x[1]), :y)
end

cor(vals...)
