#! /usr/bin/env python

import rospy
import math
#imprt the String message type
from std_msgs.msg import String
from std_srvs.srv import Empty
rospy.init_node('reset_world')

rospy.wait_for_service('/gazebo/reset_world')
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

#create a ros publisher to publish to a topic of string type
pub = rospy.Publisher('sim_reset', String, queue_size=10)

while True:
#wait for 10 seconds
    rospy.sleep(3)
    reset_world()
    reset_world()
    reset_world()
    #publish "reset" to the topic
    pub.publish("reset")