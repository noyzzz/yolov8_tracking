from abc import ABC, abstractmethod
from collections import deque

class MATracker (ABC):

    def __init__(self, use_depth, use_odometry):
        self.last_time_stamp = 0 #time in secons
        self.use_depth = use_depth
        self.use_odometry = use_odometry
        self.fps = None

    @abstractmethod
    def update_time(self, odom):
        pass

    def update_time(self, odom):
        if odom is None:
            self.fps = 25
            return
        current_time = odom.header.stamp.to_time()
        time_now = current_time
        if time_now - self.last_time_stamp == 0:
            self.fps = 25
            print(f"odom header time stamp at {self.frame_count} is the same as the lasts time stamp")
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

    def __init__(self, yaw_dot_list):
        self.bb_depth = None

    @abstractmethod
    def get_d1(self):
        pass

    @abstractmethod
    def update_depth_image(depth_image):
        pass

    @abstractmethod
    def update_ego_motion(odom, fps_rot, fps_depth):
        pass
