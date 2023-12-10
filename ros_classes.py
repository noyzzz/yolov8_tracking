
import queue

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
#import nav_msgs/Odometry
from nav_msgs.msg import Odometry
# from my_tracker.msg import ImageDetectionMessage
import numpy as np
from collections import deque
class image_converter:
    def __init__(self, is_track_publish_activated=0):
    #initialize a queue to store cv_image from callback
        DELAY_FRAMES = 1
        self.cv_image_queue = queue.Queue()
        self.sim_reset_queue = queue.Queue()
        self.depth_image_queue = queue.Queue()
        self.odom_queue = queue.Queue()
        self.gt_queue = queue.Queue()
        self.gt_internal_queue = deque(maxlen=DELAY_FRAMES)
        self.is_track_publish_activated = is_track_publish_activated
        if self.is_track_publish_activated == 1:
            from my_tracker.msg import ImageDetectionMessage
            self.track_publisher = rospy.Publisher("/image_tracks",ImageDetectionMessage)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed",CompressedImage,self.callback)
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw/compressed",CompressedImage,self.depth_image_callback)
        self.sim_reset_sub = rospy.Subscriber("/sim_reset_tracker",String,self.sim_reset_callback)
        self.odom_sub = rospy.Subscriber("/odometry/filtered",Odometry,self.odom_callback)
        self.sub_gt = rospy.Subscriber('/carla_tracks', Float32MultiArray, self.callback_gt)

    def process_gt(self, gt_raw):
        #gt_raw is a Float32MultiArray
        #get every five elements and convert them to a list of dictionary, each element is (id, min_x, min_y, max_x, max_y)
        gt_list = []
        for i in range(int(len(gt_raw.data)/5)):
            gt_list.append({'id': gt_raw.data[5*i], 'min_x': gt_raw.data[5*i+1], 'min_y': gt_raw.data[5*i+2], 'max_x': gt_raw.data[5*i+3], 'max_y': gt_raw.data[5*i+4]})
        return gt_list


    def callback_gt(self, data):
        self.gt_internal_queue.append(data)
        self.gt_queue.put(self.process_gt(self.gt_internal_queue[0]))

    def sim_reset_callback(self,data):
        print("sim_reset_callback")
        self.sim_reset_queue.put(data)
    def callback(self,data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "passthrough")
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image_queue.put(cv_image)
        except CvBridgeError as e:
            print(e)

    def depth_image_callback(self,data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "passthrough")
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #data is Float32MultiArray and has a layout, convert it to numpy array
            # cv_image = np.array(data.data).reshape(data.layout.dim[0].size,data.layout.dim[1].size)
            self.depth_image_queue.put(cv_image)
        except CvBridgeError as e:
            print(e)

    def odom_callback(self,data):
        self.odom_queue.put(data)



    def get_queue_size(self):
        return self.cv_image_queue.qsize()

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)