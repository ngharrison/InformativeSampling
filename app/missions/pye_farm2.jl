
using Logging: global_logger, ConsoleLogger, Info, Debug
using FileIO: load
using Images: gray

using InformativeSampling
using .Maps: Map, generateAxes, getBounds
using .SampleCosts: EIGF, DistScaledEIGF
using .Samples: Sample, MapsSampler
using .Missions: Mission
using .MultiQuantityGPs: MQGP

# this requires a working rospy installation
using .ROSInterface: ROSSampler

using InformativeSamplingUtils
using .Visualization: vis
using .DataIO: maps_dir, imgToMap, output_dir, output_ext, produceMaps

function pyeFarmMission(; num_samples=4,
    sampleCostType=DistScaledEIGF,
    use_priors=false,
    sub_patch=false,
    start_locs=[])

    # the topics that will be listened to for measurements
    data_topics = [
        # Crop height avg in frame (excluding wheels)
        "/rss/gp/crop_height_avg"
        "/rss/silios/ndvi_avg"
    ]

    done_topic = "sortie_finished"
    pub_topic = "latest_sample"

    sampler = ROSSampler(data_topics, done_topic, pub_topic)

    if sub_patch
        # 15x15 meter sub-patch (alt3)
        lower = [284745.0, 6241345.0]
        upper = [284760.0, 6241360.0]
        bounds = (; lower, upper)

        elev_img = load(maps_dir * "dem_15x15.tif")
        elevMap = imgToMap(gray.(elev_img), bounds)
    else
        # 50x50 meter space (alt2)
        lower = [284725.0, 6241345.0]
        upper = [284775.0, 6241395.0]
        bounds = (; lower, upper)

        elev_img = load(maps_dir * "dem_50x50.tif")
        elevMap = imgToMap(gray.(elev_img), bounds)
    end

    occupancy = Map(zeros(Bool, 100, 100), bounds)

    # # to test
    # h, n = let
    #     name = "50x50_dense_grid"
    #     file_name = output_dir * "pye_farm_trial2/named/" * name * output_ext
    #     data = load(file_name)
    #     height_samples = filter(s->s.x[2]==1, data["samples"])
    #     ndvi_samples = filter(s->s.x[2]==2, data["samples"])
    #
    #     bm = MQGP(height_samples, bounds)
    #     h, _ = produceMaps(bm, occupancy)
    #     # vis(h)
    #     bm = MQGP(ndvi_samples, bounds)
    #     n, _ = produceMaps(bm, occupancy; quantity=2)
    #     # vis(n)
    #     h, n
    # end
    #
    # sampler = MapsSampler(h, n)

    prior_maps = [elevMap]

    prior_samples = Sample{Float64}[]

    if use_priors
        # sample sparsely from the prior maps
        # currently all data have the same sample numbers and locations
        n = (7, 7) # number of samples in each dimension
        axs_sp = range.(bounds..., n)
        points_sp = vec(collect.(Iterators.product(axs_sp...)))
        prior_samples = [Sample{Float64}((x, i + 1), d(x))
                         for (i, d) in enumerate(prior_maps)
                         for x in points_sp if !isnan(d(x))]
    end

    ## initialize alg values
    weights = (; μ=1, σ=5e1, τ=1, d=1) # mean, std, dist, prox

    noise = (value=0.1, learned=false)

    mission = Mission(;
        occupancy,
        sampler,
        num_samples,
        sampleCostType,
        weights,
        start_locs,
        prior_samples,
        noise
    )

    return mission, prior_maps
end

#* Run

# set the logging level: Info or Debug
global_logger(ConsoleLogger(stderr, Debug))

using .DataIO: save

## initialize data for mission
mission, prior_maps = pyeFarmMission(
    num_samples=30,
    sampleCostType=EIGF,
    use_priors=true,
    sub_patch=false,
    start_locs=[[284748.0, 6241355.0]]
)

vis(prior_maps[1]; points=first.(getfield.(mission.prior_samples, :x)))

## run search alg
@time samples, beliefs = mission(
    sleep_time=0.0
) do M, samples, beliefModel, sampleCost, new_loc
    vis(M, samples, beliefModel, sampleCost, new_loc)
    save(M, samples, beliefModel; sub_dir_name="pye_farm_trial2")
end;
save(mission, samples, beliefs; sub_dir_name="pye_farm_trial2")


#* Publish maps

using PyCall

using .DataIO: mapToImg

rospy = pyimport("rospy")
std_msg = pyimport("std_msgs.msg")
sen_msg = pyimport("sensor_msgs.msg")

axs, points = generateAxes(mission.occupancy)
dims = size(mission.occupancy)

quantities = eachindex(mission.sampler)

# sets up the publishers
array_publishers, image_publishers = map(("array", "image"),
    (std_msg.Float64MultiArray, sen_msg.Image)) do type_name, type
    map(quantities) do q
        map(("pred", "err")) do name
            rospy.Publisher("/informative_sampling/$(name)_$(type_name)_$(q)",
                type, queue_size=1, latch=true)
        end
    end
end

rospy.sleep(.1)

M = mission
other_bm = MQGP(filter(s->s.x[2]==2, samples), getBounds(M.occupancy);
                       M.noise, M.kernel)
bm_vals = map(enumerate((beliefs[end], other_bm))) do (i, bm)
    bm(tuple.(points, i))
end

for q in quantities
    for (i, data) in enumerate(bm_vals[q])

        # map to image
        data_flipped = vec(mapToImg(data))

        # publish as array
        array = std_msg.Float64MultiArray()
        array.layout.dim = std_msg.MultiArrayDimension.("", collect(dims), 1)
        array.layout.data_offset = 0
        array.data = data_flipped

        l, h = extrema(data_flipped)
        data_scaled = (data_flipped .- l) ./ (h - l + eps())

        # convert to byte values and repeat each 4 times for rgba format
        byte_data = round.(UInt8, 255 .* data_scaled)
        img_data = pybytes([x for x in byte_data for _ in 1:4])

        env_height, env_width = dims

        # publish as image
        image = sen_msg.Image()
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = "env_image"
        image.height = env_height
        image.width = env_width
        image.encoding = "rgba8"
        image.is_bigendian = 1
        image.step = env_width * 4
        image.data = img_data
        image_publishers[q][i].publish(image)
        rospy.sleep(.01) # allow the publisher time to publish
    end
end

rospy.sleep(.1)
