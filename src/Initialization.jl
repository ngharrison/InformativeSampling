
module Initialization


using LinearAlgebra: I
using Images: load, imresize, Gray, gray
using DelimitedFiles: readdlm
using Statistics: cor

using Environment: Map, imgToMap, GaussGroundTruth, MultiMap, Peak, Region
using Samples: Sample
# using ROSInterface: ROSConnection
using Visualization: visualize

function simData()

    lb = [0, 0]; ub = [1, 1]

    # read in elevation
    elev_img = load("maps/arthursleigh_shed_small.tif")
    elevMap = imgToMap(gray.(elev_img), lb, ub)

    # read in obstacles
    obs_img = load("maps/obstacles_fieldsouth_220727.tif")
    obs_img_res = imresize(obs_img, size(elev_img))
    # the image we have has zeros for obstacles, need to flip
    occ_mat = Matrix{Bool}(Gray.(obs_img_res) .== 0')
    occupancy = imgToMap(occ_mat, lb, ub)

    ## initialize ground truth

    # simulated
    peaks = [Peak([0.3, 0.3], 0.03*I, 1.0),
             Peak([0.8, 0.7], 0.008*I, 0.4)]
    ggt = GaussGroundTruth(peaks)
    axs = range.(lb, ub, size(elev_img))
    points = collect.(Iterators.product(axs...))
    groundTruth = Map(ggt(points), lb, ub)

    ## Create prior prior_samples

    # none -- leave uncommented
    prior_maps = []

    # additive
    push!(prior_maps, Map(abs.(groundTruth .+ 0.1 .* randn(size(groundTruth))), lb, ub))

    # multiplicative
    push!(prior_maps, Map(abs.(groundTruth .* randn()), lb, ub))

    # # both
    # push!(prior_maps, Map(abs.(groundTruth .* randn() + 0.1 .* randn(size(groundTruth))), lb, ub))

    # # spatial shift
    # t = rand(1:7)
    # push!(prior_maps, [zeros(size(groundTruth,1),t) groundTruth[:,1:end-t]]) # shift

    # purely random
    num_peaks = 3
    peaks = [Peak(rand(2).*(ub-lb) .+ lb, 0.02*I, rand())
             for i in 1:num_peaks]
    tggt = GaussGroundTruth(peaks)
    # push!(prior_maps, Map(tggt(points), lb, ub))

    multiGroundTruth = MultiMap(groundTruth, Map(tggt(points), lb, ub))


    ## initialize alg values
    weights = [1, 6, 1, 1e-2] # mean, std, dist, prox
    start_loc = [0.5, 0.2] # starting location
    num_samples = 20


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(multiGroundTruth)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(groundTruth.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(multiGroundTruth.maps..., prior_maps...;
              titles=["Ground Truth 1", "Ground Truth 2", "Prior 1", "Prior 2"],
              samples=points_sp)

    region = Region(occupancy, multiGroundTruth)

    return region, start_loc, weights, num_samples, prior_samples

end


normalize(a) = a ./ maximum(filter(!isnan, a))


function realData()
    # have it run around australia

    lb = [0, 0]; ub = [1, 1]

    file_names = [
        "maps/vege_720x360.csv",
        "maps/topo_720x360.csv",
        "maps/temp_720x360.csv",
        "maps/rain_720x360.csv"
    ]

    images = readdlm.(file_names, ',')

    for img in images
        img[img .== 99999] .= NaN
    end

    australia = (202:258, 587:668)

    groundTruth = imgToMap(normalize(images[1][australia...]), lb, ub)
    multiGroundTruth = MultiMap(groundTruth)

    prior_maps = [imgToMap(normalize(img[australia...]), lb, ub) for img in images[2:end]]

    occupancy = imgToMap(Matrix{Bool}(isnan.(images[1][australia...])), lb, ub)


    ## initialize alg values
    weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
    start_loc = [0.8, 0.6] # starting location
    num_samples = 50


    # sample sparsely from the prior maps
    # currently all data have the same sample numbers and locations
    n = (5,5) # number of samples in each dimension
    axs_sp = range.(lb, ub, n)
    points_sp = vec(collect.(Iterators.product(axs_sp...)))
    prior_samples = [Sample((x, i+length(multiGroundTruth)), d(x))
                     for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]

    # Calculate correlation coefficients
    [cor(groundTruth.(points_sp), d.(points_sp)) for d in prior_maps]

    visualize(multiGroundTruth.maps..., prior_maps...;
              titles=["Vegetation", "Elevation", "Ground Temperature", "Rainfall"],
              samples=points_sp)

    region = Region(occupancy, multiGroundTruth)

    return region, start_loc, weights, num_samples, prior_samples

end

# function rosData()
#
#     lb = [0, 0]; ub = [1, 1]
#
#     # read in elevation
#     elev_img = load("maps/arthursleigh_shed_small.tif")
#     elevMap = imgToMap(gray.(elev_img), lb, ub)
#
#     # read in obstacles
#     obs_img = load("maps/obstacles_fieldsouth_220727.tif")
#     obs_img_res = imresize(obs_img, size(elev_img))
#     # the image we have has zeros for obstacles, need to flip
#     occ_mat = Matrix{Bool}(Gray.(obs_img_res) .== 0')
#     occupancy = imgToMap(occ_mat, lb, ub)
#
#     sub_nodes = [
#         "/GP_Data/avg_NDVI",
#         "/GP_Data/avg_crop_height",
#         "/GP_Data/cover_NDVI"
#     ]
#
#     multiGroundTruth = ROSConnection(sub_nodes)
#
#     region = Region(occupancy, multiGroundTruth)
#
#     ## initialize alg values
#     weights = [1, 6, 1, 3e-3] # mean, std, dist, prox
#     start_loc = [0.0, 0.0]
#     num_samples = 50
#
#     prior_samples = []
#
#     return region, start_loc, weights, num_samples, prior_samples
# end

end