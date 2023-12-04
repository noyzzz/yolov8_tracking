#!/usr/bin/env python3.8
# impelement a ros node that subscribes to /odometry/filtered and gets z orientation
import rospy
from nav_msgs.msg import Odometry
import numpy as np
import math
import time
import tf
# import compressed image type and openCV
from sensor_msgs.msg import CompressedImage, Image
from tf2_msgs.msg import TFMessage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys

#define a struct for the three f values, F_U_ROT, F_U_TRANS, F_V, a set of values for simulation and a set of values for real life and initialize them to 0 
# and generate getter function with the argument is_sim

class F_values:
    __F_U_ROT = 10.0
    __F_U_TRANS = 10.0
    __F_V = 10.0
    __F_U_ROT_SIM = 480.0
    __F_U_TRANS_SIM = 1600.0
    __F_V_SIM = 480.0
    def get_u_rot(is_sim):
        if is_sim:
            return F_values.__F_U_ROT_SIM
        else:
            return F_values.__F_U_ROT
    def get_u_trans(is_sim):
        if is_sim:
            return F_values.__F_U_TRANS_SIM
        else:
            return F_values.__F_U_TRANS
    def get_v(is_sim):
        if is_sim:
            return F_values.__F_V_SIM
        else:
            return F_values.__F_V
    def decrease_u_rot(is_sim):
        if is_sim:
            F_values.__F_U_ROT_SIM -= 10
        else:
            F_values.__F_U_ROT -= 10
    def increase_u_rot(is_sim):
        if is_sim:
            F_values.__F_U_ROT_SIM += 10
        else:
            F_values.__F_U_ROT += 10
    def decrease_u_trans(is_sim):
        if is_sim:
            F_values.__F_U_TRANS_SIM -= 10
        else:
            F_values.__F_U_TRANS -= 10
    def increase_u_trans(is_sim):
        if is_sim:
            F_values.__F_U_TRANS_SIM += 10
        else:
            F_values.__F_U_TRANS += 10
    def decrease_v(is_sim):
        if is_sim:
            F_values.__F_V_SIM -= 10
        else:
            F_values.__F_V -= 10

    def increase_v(is_sim):
        if is_sim:
            F_values.__F_V_SIM += 10
        else:
            F_values.__F_V += 10

class GetZOrientation:
    def __init__(self, is_sim = True):
        self.is_sim = is_sim
        # current_yaw is in radians
        self.current_yaw = 0
        self.current_position = None
        self.current_depth = None
        self.initial_yaw = 0
        self.initial_u = 0
        self.initial_v = 0
        self.initial_position = np.array([0, 0])
        self.initial_depth = 0.
        self.initial_rot_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.last_u_estimate = 0
        self.last_v_estimate = 0
        self.last_yaw = None
        self.last_x = None
        self.last_y = None
        # initialize the window to display the image
        # cv2.namedWindow("image1")
        # cv2.setMouseCallback("image", self.get_location_to_track)
        self.odom_sub = rospy.Subscriber(
            "/odometry_slam", Odometry, self.callback, queue_size=1)
        if self.is_sim:
            self.camera_sub = rospy.Subscriber(
                "/camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
            self.depth_image_sub = rospy.Subscriber(
                "/camera/depth/image_raw/compressed",CompressedImage, self.callback_depth)
        else:
            self.camera_sub = rospy.Subscriber(
                "/color_image", CompressedImage, self.image_callback, queue_size=1)
            self.depth_image_sub = rospy.Subscriber(
                "/depth_image",CompressedImage, self.callback_depth)

        self.last_image = None
        self.circle_location = None
        self.depth_image = None
        self.bridge = CvBridge()

    def get_location_to_track(self, event, x, y, flags, param):
        # if left mouse button is pressed, get the location of the mouse pointer and put it in self.circle_location
        if event == cv2.EVENT_LBUTTONDOWN:
            self.circle_location = np.array([x, y])
            # print the location of the mouse pointer
            print("x: ", x, "y: ", y)
    def callback(self, data):
        self.current_yaw = data.pose.pose.orientation.z
        #PRINT THE CURRENT YAW
        # print("current yaw: ", math.degrees(self.current_yaw))
        # update current position with data.pose.pose.position and convert to numpy array
        self.current_position = np.array(
            [data.pose.pose.position.x, data.pose.pose.position.y])
        # print current position
        # print("current position: ", self.current_position)   

        # print degrees
        # print("degrees: ", math.degrees(self.current_yaw))

        

    def callback_depth(self, data):
        try:
            if self.is_sim:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "passthrough")
            else:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)
            # check if cv_image has three dimensions
        if len(cv_image.shape) == 3:
            (rows, cols, channels) = cv_image.shape
        else:
            (rows, cols) = cv_image.shape

        cv_image = np.array(cv_image, dtype=np.float32)
        # change all of the zero values to nan
        cv_image[np.where(cv_image == 0)] = np.nan
        # divide by 10 because the published depth has been multiplied by 10
        self.depth_image = cv_image/10.0


    def estimate_rot_u2(self, alpha, u1, f):
        # calculate f*(tan(alpha) +(u1/f))/(1-(u1/f)*tan(alpha))
        # alpha is in radians
        # u1 is in pixels
        # f is in pixels
        gamma = math.atan(u1/f)

        return f*(math.tan(alpha+gamma))

    def estimate_rot_linear_u2(self, alpha, u1, f):
        # calculate u1+(u1^2/f^2)+1)*f*alpha
        return u1+(u1**2/f**2+1)*f*alpha

    def estimate_trans_u2(self, d1, D, u1, f):
        # calculate (d1*sin(atan(u1/f)))/(D-cos(atan(u1/f))))
        # d1 is in meters
        # D is in meters
        # u1 is in pixels
        # f is in pixels
        # calcualte all with numpy functions

        gamma = math.atan(u1/f)
        # numerator = d1*math.sin(gamma)
        # denominator = D-math.cos(gamma)
        # return numerator/denominator
        # B = ((d1+D)*tan(gamma/2))/(d1-D)
        B = (d1+D)*math.tan(gamma/2)/(d1-D)
        # gamma2_tan = (tan(gamma/2) + B)/(1-tan(gamma/2)*B)
        gamma2_tan = (math.tan(gamma/2) + B)/(1-math.tan(gamma/2)*B)
        return f*gamma2_tan

    def image_callback(self, data):
        # convert image to cv2 image
        np_arr = np.fromstring(data.data, np.uint8)
        if self.last_image is not None:
            cv2.imshow("image", self.last_image)
            cv2.setMouseCallback("image", self.get_location_to_track)

        self.last_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # cv2.waitKey(1)
        # cv2.imshow("image", self.last_image)
        if cv2.waitKey(5) == ord('a'):
            # if 'a' is pressed, print the current yaw
            print("A PRESSED")
            self.current_depth = self.depth_image[self.circle_location[1], self.circle_location[0]]
            print("current depth: ", self.current_depth)
            print("current yaw: ", math.degrees(get_z_orientation.current_yaw))
            self.initial_yaw = self.current_yaw
            #initial u and initial v should be with respect to the center of the image
            #get the center of the image
            center = np.array([self.last_image.shape[1]/2, self.last_image.shape[0]/2])
            self.initial_u = self.circle_location[0]-center[0]
            self.initial_v = self.circle_location[1]-center[1]
            self.initial_depth = self.current_depth
            self.initial_position = self.current_position
            # self.initial_rot_matrix = self.rot_matrix

        #if down arrow is pressed, decrease the f_u_rot by 10
        pressed_key = cv2.waitKey(5)
        if pressed_key == ord('s'):
            print("S PRESSED")
            F_values.decrease_v(self.is_sim)
            print("f_v: ", F_values.get_v(self.is_sim))
        #if up arrow is pressed, increase the f_u_rot by 10
        if pressed_key == ord('w'):
            print("W PRESSED")
            F_values.increase_v(self.is_sim)
            print("f_v: ", F_values.get_v(self.is_sim))

        if pressed_key == ord('d'):
            print("D PRESSED")
            F_values.decrease_u_rot(self.is_sim)
            print("u_rot: ", F_values.get_u_rot(self.is_sim))
        #if up arrow is pressed, increase the f_u_trans by 10
        if pressed_key == ord('e'):
            print("E PRESSED")
            F_values.increase_u_rot(self.is_sim)
            print("u_rot: ", F_values.get_u_rot(self.is_sim))

        if pressed_key == ord('f'):
            print("D PRESSED")
            F_values.decrease_u_trans(self.is_sim)
            print("u_trans: ", F_values.get_u_trans(self.is_sim))
        #if up arrow is pressed, increase the f_u_trans by 10
        if pressed_key == ord('r'):
            print("E PRESSED")
            F_values.increase_u_trans(self.is_sim)
            print("u_trans: ", F_values.get_u_trans(self.is_sim))

        #if p is pressed, print the current estimated u and v
        if pressed_key == ord('p'):
            print("P PRESSED")
            print("estimated u: ", self.last_u_estimate, "estimated v: ", self.last_v_estimate)

        
        # current_y = self.current_position[1]
        # current_x = self.current_position[0]
        # current_yaw = self.current_yaw

        # if self.last_yaw is None or self.last_x is None or self.last_y is None:
        #     self.last_yaw = current_yaw
        #     self.last_x = current_x
        #     self.last_y = current_y
        # diff_y = current_y-self.last_y
        # diff_x = current_x-self.last_x

        # self.delta_x += diff_x*math.cos(current_yaw)-diff_y*math.sin(current_yaw)
        # self.delta_y += diff_x*math.sin(current_yaw)+diff_y*math.cos(current_yaw)

        # self.last_yaw = current_yaw
        # self.last_x = current_x
        # self.last_y = current_y


        diff_x_odom = self.current_position[0]-self.initial_position[0]
        diff_y_odom = self.current_position[1]-self.initial_position[1]

        delta_yaw = self.current_yaw-self.initial_yaw
        # delta_x_car_frame = diff_x_odom*math.cos(self.current_yaw)+diff_y_odom*math.sin(self.current_yaw)
        # delta_y_car_frame = -diff_x_odom*math.sin(self.current_yaw)+diff_y_odom*math.cos(self.current_yaw)
        delta_x_car_frame = np.sqrt(diff_x_odom**2+diff_y_odom**2)
        #print the three values with 3 decimal points of precision, format the string with f-strings
        # print(f"delta_x: {delta_x_car_frame:.3f} delta_y: {delta_y_car_frame:.3f} delta_yaw: {delta_yaw:.3f}")
        

        # print("delta_x: ", delta_x_car_frame, "delta_y: ", delta_y_car_frame, "delta_yaw: ", self.initial_yaw)

        # robot_movement = self.current_position-self.initial_position
        # robot_movement_3x1 = np.array([robot_movement[1], robot_movement[0], 0])
        # translation_b1 = np.matmul(self.initial_rot_matrix, robot_movement_3x1)
        # print("estimated u2 with translational move", self.estimate_trans_u2(self.initial_depth, distance, self.initial_u, 250.0))


        #set theta as the angle between the current position and initial_position
        theta = math.atan((self.current_position[1]-self.initial_position[1])/(self.current_position[0]-self.initial_position[0]))
        #if theta is nan, set it to 0
        if math.isnan(theta):
            theta = 0
        # print("theta: ", theta, 'y:  ', self.current_position[1]-self.initial_position[1],'x: ', self.current_position[0]-self.initial_position[0],)
        first_rot_angle =  -(-self.initial_yaw + theta)
        trans_mov = np.sqrt((self.current_position[0]-self.initial_position[0])**2+(self.current_position[1]-self.initial_position[1])**2)
        second_rot_angle =  -(-theta + self.current_yaw)
        first_estimate = self.estimate_rot_u2(first_rot_angle, self.initial_u, F_values.get_u_rot(self.is_sim))
        second_estimate = self.estimate_trans_u2(self.initial_depth, delta_x_car_frame, first_estimate, F_values.get_u_trans(self.is_sim))
        third_estimate = self.estimate_rot_u2(second_rot_angle, second_estimate, F_values.get_u_rot(self.is_sim))
        print("x_diff", delta_x_car_frame, "Eucidean dis", trans_mov,\
            "first estimate: ", first_estimate, "second estimate: ", second_estimate, "third estimate: ", third_estimate)

        estimate_u_pure_rot = self.estimate_rot_u2(-(self.current_yaw-self.initial_yaw), self.initial_u, F_values.get_u_rot(self.is_sim))
        estimate_u_trans_post_rot = self.estimate_trans_u2(self.initial_depth, delta_x_car_frame, estimate_u_pure_rot, F_values.get_u_trans(self.is_sim))
        superpos_estimate_u = third_estimate
        # superpos_estimate_u = (self.estimate_rot_u2(-(self.current_yaw-self.initial_yaw), self.initial_u, F_values.get_u_rot(self.is_sim)) -
        #                      self.initial_u) + self.estimate_trans_u2(self.initial_depth, delta_x_car_frame, self.initial_u, F_values.get_u_trans(self.is_sim))
        
        # superpos_estimate_u = self.estimate_trans_u2(self.initial_depth, delta_x_car_frame, self.initial_u, F_values.get_u_trans(self.is_sim))
        # superpos_estimate_u = self.estimate_rot_u2(-(self.current_yaw-self.initial_yaw), self.initial_u, F_values.get_u_rot(self.is_sim))
        # superpos_estimate_u = self.initial_u
        superpos_estimate_v = self.estimate_trans_u2(self.initial_depth, delta_x_car_frame, self.initial_v, F_values.get_v(self.is_sim))

        self.last_u_estimate = superpos_estimate_u
        self.last_v_estimate =  superpos_estimate_v

        #draw a circle at last_u_estimate and last_v_estimate on self.last_image , the positions are wrt the center of the image so add the center of the image to the positions
        center = np.array([self.last_image.shape[1]/2, self.last_image.shape[0]/2])
        cv2.circle(self.last_image, (int(superpos_estimate_u+center[0]), int(superpos_estimate_v+center[1])), 10, (0, 0, 255), -1)

        #print image width and height
        # print("image width: ", self.last_image.shape[1], "image height: ", self.last_image.shape[0])

 
        
        #print estimated u and v
        # print("estimated u: ", superpos_estimate_u , "estimated v: ", superpos_estimate_v)
        # print("translation wrt B1: ", translation_b1)
        
        


if __name__ == '__main__':
    #get the argument is_sim from the command line and put it in the variable is_sim if there is no argument given in the command line, set is_sim to False
    is_sim = True
    if len(sys.argv) > 1:
        #parse the arguments with argparse.ArgumentParser()
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--is_sim', action='store_true')
        args = parser.parse_args()
        is_sim = args.is_sim
    print("is_sim: ", is_sim)
    
    rospy.init_node('get_z_orientation', anonymous=True)
    print("KIIER")
    get_z_orientation = GetZOrientation(is_sim = is_sim)
    rospy.spin()

