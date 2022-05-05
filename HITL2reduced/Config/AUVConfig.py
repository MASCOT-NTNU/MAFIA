#!/usr/bin/env python3

import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms

# == Waypoint
WAYPOINT_UPDATE_TIME = 3.
# ==

# == YoYo
YOYO_LATERAL_DISTANCE = 60.
YOYO_VERTICAL_DISTANCE = 5.
# ==

# == Depth
DEPTH_TOP = .5
DEPTH_BOTTOM = 5.5
# ==
