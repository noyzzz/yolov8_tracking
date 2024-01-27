from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import copy


class MATracker (ABC):

    def __init__(self, use_depth, use_odometry):
        self.last_time_stamp = 0 #time in secons
        self.use_depth = use_depth
        self.use_odometry = use_odometry
        self.fps = None


    def update_time(self, odom, frame_count):
        if odom is None:
            self.fps = 25
            return
        current_time = odom.header.stamp.to_time()
        time_now = current_time
        if time_now - self.last_time_stamp == 0:
            self.fps = 25
            print(f"odom header time stamp at {frame_count} is the same as the lasts time stamp")
        else:
            self.fps =  (1.0/(time_now - self.last_time_stamp))*1
            # print(f'at frame {self.frame_id} fps is {self.fps}')

        self.last_time_stamp = time_now
        # print("fps: ", self.fps)
    
class MATrack (ABC):
    current_yaw = 0;
    current_yaw_dot = 0;
    current_yaw_dot_filtered = 0;
    current_depth_image = None
    yaw_dot_list = deque(maxlen=2)
    

    def get_d1(self):
        #calculate the depth of the object in the depth image
        #get the bounding box of the object in the depth image
        #get the median of the depth values in the bounding box excluding the zeros and the nans
        #return the depth value
        bounding_box = copy.deepcopy(self.get_tlwh())
        #get the depth of the bounding box in the depth image
        #clip the bounding box to the image size and remove the negative values
        bounding_box[bounding_box < 0] = 0
        bounding_box[np.isnan(bounding_box)] = 0
        #if any of the bounding values is inf then set it to zero
        # bounding_box[np.isinf(bounding_box)] = 0
        track_depth = copy.deepcopy(MATrack.current_depth_image)[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]), int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
        #get the median of the depth values in the bounding box excluding the zeros and the nans
        track_depth = track_depth[track_depth != 0]
        track_depth = track_depth[~np.isnan(track_depth)]
        if len(track_depth) == 0:
            return 0
        self.bb_depth = np.median(track_depth)
        return self.bb_depth

    def update_depth_image(depth_image):
        #convert depth image type to float32
        depth_image = depth_image.astype(np.float32)
        depth_image/=10
        MATrack.current_depth_image = depth_image

    def update_ego_motion(odom, fps_rot):
        quat = False
        if quat:
            quaternion = (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
            #get the yaw from the quaternion not using the tf library
            yaw = np.arctan2(2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
                            1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]))
        else:
            yaw = odom.pose.pose.orientation.z
        while  abs(yaw-MATrack.current_yaw) > np.pi :
            if yaw < MATrack.current_yaw :
                yaw += 2*np.pi
            else:
                yaw -= 2*np.pi
        twist = odom.twist.twist
        MATrack.current_yaw_dot = twist.angular.z / fps_rot # frames are being published at 20Hz in the simulator
        MATrack.yaw_dot_list.append(MATrack.current_yaw_dot)
        MATrack.current_yaw = yaw
        MATrack.current_yaw_dot_filtered = np.mean(MATrack.yaw_dot_list)
        raw_x_dot = twist.linear.x
        raw_y_dot = twist.linear.y
        x_dot = raw_x_dot*np.cos(MATrack.current_yaw)+raw_y_dot*np.sin(MATrack.current_yaw)
        y_dot = -raw_x_dot*np.sin(MATrack.current_yaw)+raw_y_dot*np.cos(MATrack.current_yaw)
        MATrack.current_D_dot = x_dot / fps_rot


    def __init__(self):
        self.bb_depth = None

    @abstractmethod
    def get_tlwh(self):
        pass

