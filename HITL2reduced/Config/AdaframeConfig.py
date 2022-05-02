#!/usr/bin/env python3

import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms

WAYPOINT_UPDATE_TIME = 5.0
