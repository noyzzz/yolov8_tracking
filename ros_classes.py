
import queue

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
#import nav_msgs/Odometry
from nav_msgs.msg import Odometry
# from my_tracker.msg import ImageDetectionMessage

class image_converter:
    def __init__(self, is_track_publish_activated=0):
    #initialize a queue to store cv_image from callback
        self.cv_image_queue = queue.Queue()
        self.sim_reset_queue = queue.Queue()
        self.depth_image_queue = queue.Queue()
        self.odom_queue = queue.Queue()
        self.is_track_publish_activated = is_track_publish_activated
        if self.is_track_publish_activated == 1:
            from my_tracker.msg import ImageDetectionMessage
            self.track_publisher = rospy.Publisher("/image_tracks",ImageDetectionMessage)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed",CompressedImage,self.callback)
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw/compressed",CompressedImage,self.depth_image_callback)
        self.sim_reset_sub = rospy.Subscriber("/sim_reset_tracker",String,self.sim_reset_callback)
        self.odom_sub = rospy.Subscriber("/odometry/filtered",Odometry,self.odom_callback)

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