module DataIO

using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
using Images: save as saveImg, colorview, RGBA
using Plots

using AdaptiveSampling
using .Maps: generateAxes, Map

include("Visualization.jl")
using .Visualization: visualize

export normalize, spatialAve, imgToMap, save, maps_dir,
       output_dir, output_ext, saveBeliefMapToPng

const maps_dir = dirname(Base.active_project()) * "/maps/"
const output_dir = dirname(Base.active_project()) * "/output/"
const output_ext = ".jld2"

"""
A helper method to normalize an array so its values are within the range [0, 1].
"""
function normalize(a)
    l, h = extrema(filter(!isnan, a))
    return (a .- l) ./ (h - l)
end

"""
A helper method to perform a spatial average on a matrix.
The extent of the average can be chosen with its default being 1.
"""
function spatialAve(M, extent=1)
    N = zero(M)
    for i in axes(M,1), j in axes(M,2)
        tot = 0
        count = 0
        for k in -extent:extent, l in -extent:extent
            m = i + k
            n = j + l
            if 1 <= m <= size(M,1) && 1 <= n <= size(M,2) && !isnan(M[m,n])
                tot += M[m,n]
                count += 1
            end
        end
        N[i,j] = tot/count
    end
    return N
end

"""
Takes a matrix in the format created from an image, re-formats it, and returns a
Map. Images view a matrix with its indexing top-down and left-right. Maps view a
matrix with its indexing left-right and bottom-up.

# Examples
```julia
using DelimitedFiles: readdlm

image = readdlm(file_name, ',')
lb = [0.0, 0.0]; ub = [1.0, 1.0]
map = imgToMap(image, lb, ub)
map = imgToMap(image) # or auto bounds
```
"""
imgToMap(img, args...) = Map(permutedims(reverse(img, dims=1), (2,1)), args...)


"""
Creates a string containing the date and time separated by dashes.
Can pass in a DateTime object, defaults to current time.
"""
function dateTimeString(dt=now())
    parts = (year, month, day, hour, minute, second)
    return join(lpad.(dt .|> parts, 2, '0'), "-")
end

function save(mission, samples, beliefs;
              animation=false,
              sub_dir_name="",
              file_name=dateTimeString() * "_mission")

    mkpath(output_dir * sub_dir_name * "/" * dirname(file_name))

    full_path_name = output_dir * sub_dir_name * "/" * file_name * output_ext

    jldsave(full_path_name; mission, samples, beliefs)

    if animation
        num_quant = length(mission.sampler)
        animation = @animate for i in eachindex(beliefs)
            visualize(beliefs[i],
                      samples[begin:i*num_quant],
                      mission.occupancy,
                      1)
        end
        mp4(animation, output_dir * file_name * ".mp4"; fps=1)
    end

end

function save(metrics; sub_dir_name="", file_name=dateTimeString() * "_metrics")
    mkpath(output_dir * dirname(file_name))

    full_path_name = output_dir * sub_dir_name * "/" * file_name * output_ext

    jldsave(full_path_name; metrics)
end

# This is really just to give something out to munch,
# so it needs to be an rgba png with the last channel as the amount
function saveBeliefMapToPng(beliefModel, occupancy,
                            file_name=dateTimeString() * "_belief_map")
    mkpath(output_dir)

    axs, points = generateAxes(occupancy)
    dims = Tuple(length.(axs))
    μ, _ = beliefModel(tuple.(vec(points), 1))
    pred_map = reshape(μ, dims)

    l, h = extrema(pred_map)
    amount = (pred_map .- l) ./ (h - l)

    # map to image
    amount = reverse(permutedims(amount, (2,1)), dims=1)

    map_img = stack((0.8*amount,
                     0.3*amount,
                     zeros(size(pred_map)),
                     amount), dims=1)

    saveImg("$(output_dir)$(file_name).png",
            colorview(RGBA, map_img))

end

end