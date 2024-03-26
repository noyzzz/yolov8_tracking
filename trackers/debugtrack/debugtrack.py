
from ..MATracker import MATracker, MATrack
import numpy as np


class Track (MATrack):
    last_id = 0
    def __init__(self, tlwh, score, rgb_image, depth_image, odom, frame_count, mask = None):
        super().__init__()
        self.track_id = Track.last_id
        Track.last_id += 1
        # self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        mean_pos = Track.tlwh_to_xyah(tlwh)
        mean_vel = np.zeros(4)
        self.mean = np.concatenate((mean_pos, mean_vel))
        self.image_width = rgb_image.shape[1]
        self.image_height = rgb_image.shape[0]
        self.focal_length = MATracker.FOCAL_LENGTH
        self.dt = 1
        self.control_mat = np.zeros((8, 1))


    def calculate_control_mat(self, mean):
        u1 = mean[0] - self.image_width/2
        robot_yaw_to_pixel_coeff = (u1**2/self.focal_length**2 + 1)*self.focal_length
        self.control_mat[0, 0] = robot_yaw_to_pixel_coeff*self.dt
        return self.control_mat
    
    def calculate_depth_control_mat(self, mean, control_signal):
        if control_signal[2] == 0:
            return np.zeros((8, 1))
        u1 = mean[0] - self.image_width/2
        v1 = mean[1] - self.image_height/2
        bottom_y = v1 + mean[3]/2 #bottom right corner y wrt image center
        u_coeff = u1*np.sqrt(u1**2 + self.focal_length**2)/(self.focal_length * control_signal[2])
        v_coeff = v1*np.sqrt(v1**2 + self.focal_length**2)/(self.focal_length * control_signal[2])
        h_coeff = (bottom_y*np.sqrt(bottom_y**2 + self.focal_length**2) - v1*np.sqrt(v1**2 + self.focal_length**2))\
                   /(self.focal_length * control_signal[2])
        depth_control_mat = np.zeros((8, 1))
        depth_control_mat[0, 0] = u_coeff
        depth_control_mat[1, 0] = v_coeff
        depth_control_mat[3, 0] = h_coeff
        return depth_control_mat
    
    def tlwh_to_xyah(tlwh):
        xyah = np.zeros(4)
        xyah[0] = tlwh[0] + tlwh[2]/2
        xyah[1] = tlwh[1] + tlwh[3]/2
        xyah[2] = tlwh[2]/tlwh[3]
        xyah[3] = tlwh[3]
        return xyah
    
    def xyxy_to_tlwh(xyxy):
        tlwh = np.zeros(4)
        tlwh[0] = xyxy[0]
        tlwh[1] = xyxy[1]
        tlwh[2] = xyxy[2] - xyxy[0]
        tlwh[3] = xyxy[3] - xyxy[1]
        return tlwh
    
    def xyxy_to_xyah(xyxy):
        xyah = np.zeros(4)
        xyah[0] = (xyxy[0] + xyxy[2])/2
        xyah[1] = (xyxy[1] + xyxy[3])/2
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        xyah[2] = w/h
        xyah[3] = h
        return xyah
    def get_tlwh(self):
        #convert self.mean to tlwh
        #mean is in the form [x, y, a, h]
        tlwh = np.zeros(4)
        tlwh[0] = self.mean[0] - self.mean[2]/2
        tlwh[1] = self.mean[1] - self.mean[3]/2
        tlwh[2] = self.mean[2] *self.mean[3]
        tlwh[3] = self.mean[3]
        return tlwh
        
    def get_xyxy(self):
        #convert self.mean to xyxy
        #mean is in the form [x, y, a, h]
        xyxy = np.zeros(4)
        w = self.mean[2] * self.mean[3]
        h = self.mean[3]
        xyxy[0] = self.mean[0] - w/2
        xyxy[1] = self.mean[1] - h/2
        xyxy[2] = self.mean[0] + w/2
        xyxy[3] = self.mean[1] + h/2
        return xyxy
    def predict(self):
        #with the current_D_dot and the current_yaw_dot_filtered we can predict the next position of the object
        current_yaw_dot = self.current_yaw_dot_filtered
        current_D_dot = self.current_D_dot
        current_d1 = self.get_d1()
        control = np.array([current_yaw_dot, current_D_dot, current_d1])
        self.control_signal = control
        self.control_mat = self.calculate_control_mat(self.mean)
        depth_control_mat = self.calculate_depth_control_mat(self.mean, self.control_signal)
        mean_rot_applied = np.dot(self.control_signal[0], self.control_mat.T)[0]
        mean_trans_applied = np.dot(self.control_signal[1], depth_control_mat.T)[0]
        self.mean = self.mean + mean_rot_applied + mean_trans_applied
        return self.mean




        
    
    

class DebugTracker (MATracker):
    def __init__(self, use_depth, use_odometry):
        super().__init__(use_depth, use_odometry)
        self.frame_count = 0
        self.tracks = []

    def update(self, dets, rgb_image, depth_image = None, odom = None, masks = None):
        self.frame_count += 1
        self.update_time(odom, self.frame_count)
        if odom is not None:
            Track.update_ego_motion(odom, self.fps)
            Track.update_depth_image(depth_image)
        for track in self.tracks:
            track.predict()
        #generate a new track for each detection
        #generate a random number between 0 and 1
        for det in dets:
            rand_num = np.random.rand()
            if rand_num >  0.01:
                continue
            track = Track(Track.xyxy_to_tlwh(det[0:4]), det[4], rgb_image, depth_image, odom, self.frame_count, masks)
            self.tracks.append(track)
        #keep the last 5 tracks
        self.tracks = self.tracks[-10:]
        outputs = []
        for track in self.tracks:
            output = []
            output.extend(track.get_xyxy())
            output.append(track.track_id)
            output.append(0)
            output.append(track.score)
            output.append(-1)
            outputs.append(output)
        return outputs


