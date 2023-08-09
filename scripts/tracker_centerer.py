#!/usr/bin/env python3.8

#make a ros node class that subscribes to /image_tracks and prints the message
import rospy
from my_tracker.msg import ImageDetectionMessage
from std_msgs.msg import String
import cv2
import numpy as np
from geometry_msgs.msg import Twist
# from nn_modules import *
from tqdm import tqdm
from std_msgs.msg import String
from std_srvs.srv import Empty

import queue


class track_centerizer:
    def __init__(self):
        self.track_subscriber = rospy.Subscriber("/image_tracks",ImageDetectionMessage,self.track_callback)
        #define a publisher to publish to /cmd_vel geometry_msgs/Twist "linear
        self.robot_move_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.reset_sim_tracker_publisher = rospy.Publisher('sim_reset_tracker', String, queue_size=10)
        self.reset_object_mover_publisher = rospy.Publisher('object_mover_reset', String, queue_size=10)
        self.my_image = []
        self.my_tracks = []
        self.annotated_frames = []
        self.image_width = None
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        #define tracks_data as a queue with max size of 4
        self.tracks_data = []
    def reset_sim(self):
        self.reset_sim_tracker_publisher.publish("reset")
        self.reset_object_mover_publisher.publish("reset")
        self.reset_world()
        #delay for 0.5 seconds
        rospy.sleep(0.5) #wait for service to be finished
        # self.reset_world()
        # self.reset_world()

    #define a function for get_tracks_data that returns the tracks_data and clears the queue, it should wait until there are at least 4 elements in the queue
    def get_tracks_data(self):
        #wait until there are at least 4 elements in the queue
        while len(self.tracks_data) < 1:
            #sleep for 0.1 seconds
            rospy.sleep(0.01)
        #get the tracks data
        tracks_data = self.tracks_data
        #clear the queue
        self.tracks_data = []
        #return the tracks data
        return tracks_data
    
    def multi_track_data_parser(self, tracks_data):
        #get the tracks data as input which is a list of tracks data for n consecutive frames
        #each track data is a list of 5 elements (x1, y1, x2, y2, track_id), use the track_data_parser function to parse each track data and 
        #return a list of tracks for each frame
        output = []
        for track_data in tracks_data:
            output.append(self.track_data_parser(track_data))
        return output


    def track_data_parser(self, track_data):
    #every 5 consecutive elements in the list are a track in form of (x1, y1, x2, y2, track_id)
    #so we need to split the list into chunks of 5 elements
    #and then create a llist of tracks
        tracks = []
        for i in range(0, len(track_data), 5):
            track = track_data[i:i+5]
            #create a dict for the track
            track = {
                "x1": track[0],
                "y1": track[1],
                "x2": track[2],
                "y2": track[3],
                "track_id": track[4]
            }
            tracks.append(track)
        return tracks

    def find_track_distance_from_center(self, track, image_width):
        #find the center of the track
        track_center_x = (track["x1"] + track["x2"]) / 2
        track_center_y = (track["y1"] + track["y2"]) / 2
        #find the distance from the vertical center line of the image
        distance_from_center_x = abs(track_center_x - image_width / 2)
        #return the distance
        return distance_from_center_x
    
    def get_track_centering_penalty(self, tracks, image_width, num_of_tracks=3):
        if len(tracks) == 0:
            return 10000
        #get the distance from the center for each track
        track_distances = []
        for track in tracks:
            track_distances.append(self.find_track_distance_from_center(track, image_width))
        #find the average distance
        penalty = sum(track_distances) / len(track_distances)
        if len(track_distances) < num_of_tracks:
            penalty  += (num_of_tracks - len(track_distances)) * 400
        return penalty
    
    #define an  RL algorithm that will center the track in the image
    #the action space is the linear and angular speed of the robot
    #the observation space is the image and the tracks
    #the reward function is the distance of the track from the center of the image
    #the goal is to minimize the distance of the tracks from the center of the image
    #the algorithm is a deep q learning algorithm
    #the algorithm will be trained in a simulator

    def get_track_centering_reward(self, tracks, image_width):
        #use the get_track_centering_penalty function to calculate the reward
        return -self.get_track_centering_penalty(tracks, image_width)
     

    def track_callback(self,data):
        # print("track_callback")
        image = data.im_data
        image_width = data.im_width
        self.image_width = image_width
        image_height = data.im_height
        tracks_data = data.tracks_data   
        np_image = np.array(list(image)).reshape((image_height,image_width,3))
        np_image = np_image.astype(np.uint8)
        self.my_image = np_image
        self.tracks_data.append(tracks_data)
        self.annotated_frames.append(np_image)
        #if the length of the annotated_frames is more than 4, just keep the last 4 frames
        if len(self.annotated_frames) > 4:
            self.annotated_frames = self.annotated_frames[-4:]
        #if the length of the tracks_data is more than 4, remove the oldest tracks_data
        if len(self.tracks_data) > 4:
            self.tracks_data = self.tracks_data[-4:]

    def resize_frames(self, frames, width=84, height=84):
        #frames is a list of 4 images
        #resize each image to width and height and grayscale it
        resized_frames = []
        #make sure that the frames are 4
        assert len(frames) == 4
        #normalize the pixel values to be between 0 and 1
        for frame in frames:
            #resize the frame
            resized_frame = cv2.resize(frame, (width, height))
            #convert to grayscale
            # print(resized_frame.shape)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            #append the resized frame to the list
            resized_frames.append(resized_frame)
        frames = np.array(resized_frames) / 255.0
        return frames

    def command_message_generator(self,linear_speed, angular_speed):
        #create a new message of this type
        twist = Twist()
        #fill the message
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        #return the message
        return twist
    
    def move_robot(self, action):
        #action is a 4 dimensional vector left, down, right, up
        #convert the action to linear and angular speed
        # if left, linear speed = 0, angular speed = 0.1
        # if down, linear speed = -0.1, angular speed = 0
        # if right, linear speed = 0, angular speed = -0.1
        # if up, linear speed = 0.1, angular speed = 0
        # if no action, linear speed = 0, angular speed = 0
        # print("action is : ", action)
        if action == 0:
            linear_speed = 0
            angular_speed = 0.1
        elif action == 1:
            linear_speed = -0.1
            angular_speed = 0
        elif action == 2:
            linear_speed = 0
            angular_speed = -0.1
        elif action == 3:
            linear_speed = 0.1
            angular_speed = 0
        twist = self.command_message_generator(linear_speed, angular_speed)
        #publish the message
        self.robot_move_publisher.publish(twist)
        #print the message
        # rospy.loginfo("Moving robot: linear speed: " + str(linear_speed) + " angular speed: " + str(angular_speed))

    
    def print_last_frame(self, track_centerer):
        if len(track_centerizer.my_image) != 0:
            # print(len(track_centerizer.my_image))
            # print("*****************")
            # print("*****************")
            # print("*****************")
            # print("*****************")
            # print("*****************")
            # print("*****************")

            cv2.namedWindow("KIR", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow("KIR", track_centerizer.my_image.shape[1], track_centerizer.my_image.shape[0])
            cv2.imshow("KIR", track_centerizer.my_image)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()


if __name__ == "__main__":
    print("KIR")
    track_centerizer = track_centerizer()
    rospy.init_node('track_centerizer', anonymous=True)
    track_centerizer.run_rl(training_mode=True, pretrained=False, double_dqn=True, num_episodes=1, exploration_max = 1)
    while not rospy.is_shutdown():
        rospy.spin()
