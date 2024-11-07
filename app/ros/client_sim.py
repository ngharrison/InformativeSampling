#!/usr/bin/env python

# this script is used to test server.jl

from __future__ import print_function

import sys
import rospy
from informative_sampling.srv import *
from informative_sampling.msg import *
from std_msgs.msg import *

samples = [
    Sample([.1, .8], 1, 1.4),
    Sample([.5, .4], 1, .2),
    Sample([.2, .2], 2, .8),
    Sample([.9, .1], 2, .1),
]

params = BeliefModelParameters(
    [0.5999898234956208, 0.2811307595186129, 0.20847681628154585],
    0.15899537159833718,
    0.0
)

bounds = Bounds([0.0, 0.0], [1.0, 1.0])

noise = Noise(0.0, False)

dims = [50, 50]

quantity_index = 1

occupancy = ByteMultiArray(
    MultiArrayLayout([MultiArrayDimension("", d, 1) for d in dims], 0),
    bytearray(dims[0]*dims[1])
)

weights = [1, 1e2, 1, 0]

quantities = [1]

def generateBeliefModelClient():
    print("Requesting Belief Model")
    rospy.wait_for_service('generate_belief_model')
    try:
        generateBeliefModel = rospy.ServiceProxy('generate_belief_model', GenerateBeliefModel)
        result = generateBeliefModel(samples, bounds, noise)
        print("Received %s" % result)
        return result.params
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def generateBeliefMapsClient():
    print("Requesting belief maps")
    rospy.wait_for_service('generate_belief_maps')
    try:
        proxy = rospy.ServiceProxy('generate_belief_maps', GenerateBeliefMaps)
        result = proxy(samples, bounds, noise, dims, quantity_index)
        print("Received %s" % result)
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def generateBeliefMapsFromModelClient():
    print("Requesting belief maps from model")
    rospy.wait_for_service('generate_belief_maps_from_model')
    try:
        proxy = rospy.ServiceProxy('generate_belief_maps_from_model', GenerateBeliefMapsFromModel)
        result = proxy(samples, params, bounds, dims, quantity_index)
        print("Received %s" % result)
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def nextSampleLocationClient():
    print("Requesting next sample location")
    service_name = 'next_sample_location'
    rospy.wait_for_service(service_name)
    try:
        proxy = rospy.ServiceProxy(service_name, NextSampleLocation)
        result = proxy(samples, bounds, noise, occupancy, weights, quantities)
        print("Received %s" % result)
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def beliefMapsAndNextSampleLocationClient():
    print("Requesting belief maps and next sample location")
    service_name = 'belief_maps_and_next_sample_location'
    rospy.wait_for_service(service_name)
    try:
        proxy = rospy.ServiceProxy(service_name, BeliefMapsAndNextSampleLocation)
        result = proxy(samples, bounds, noise, quantity_index, occupancy, weights, quantities)
        print("Received %s" % result)
        return result
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    generateBeliefModelClient()

    generateBeliefMapsClient()

    generateBeliefMapsFromModelClient()

    nextSampleLocationClient()

    beliefMapsAndNextSampleLocationClient()
