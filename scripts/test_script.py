from kitti_loader import KittiLoader
import numpy as np
from collections import deque


import cv2
import numpy as np
import copy

image_width = 1242
image_height = 375
focal_length = 721.54
fps_rot = 9.6
yaw_dot_list = deque(maxlen=2)
def calculate_control_mat( mean):
    u1 = mean[0] - image_width/2
    robot_yaw_to_pixel_coeff = (u1**2/focal_length**2 + 1)*focal_length
    control_mat = np.zeros((4, 1))
    control_mat[0, 0] = robot_yaw_to_pixel_coeff*1
    return control_mat

def calculate_depth_control_mat(mean, control_signal):
    if control_signal[2] == 0:
        return np.zeros((4, 1))
    u1 = mean[0] - image_width/2
    v1 = mean[1] - image_height/2
    w = mean[2]
    h = mean[3]
    bottom_y = v1+h/2.# v1 + mean[3]/2 #bottom right corner y wrt image center
    right_x = u1+w/2.# u1 + mean[2]/2 #bottom right corner x wrt image center
    u_coeff = u1*np.sqrt(u1**2 + focal_length**2)/(focal_length * control_signal[2])
    v_coeff = v1*np.sqrt(v1**2 + focal_length**2)/(focal_length * control_signal[2])
    h_coeff = (bottom_y*np.sqrt(bottom_y**2 + focal_length**2) - v1*np.sqrt(v1**2 + focal_length**2))\
                /(focal_length * control_signal[2])
    w_coeff = (right_x*np.sqrt(right_x**2 + focal_length**2) - u1*np.sqrt(u1**2 + focal_length**2))\
                    /(focal_length * control_signal[2])
    
    
    depth_control_mat = np.zeros((4, 1))
    depth_control_mat[0, 0] = u_coeff
    depth_control_mat[1, 0] = v_coeff
    depth_control_mat[2, 0] = w_coeff
    depth_control_mat[3, 0] = h_coeff
    
    return depth_control_mat


def predict(mean, control_input):
    this_control_mat = calculate_control_mat(mean)
    mean_rot_applied = np.dot(control_input[0], this_control_mat.T)[0]
    depth_control_mat = calculate_depth_control_mat(mean, control_input)
    mean_trans_applied = np.dot(control_input[1], depth_control_mat.T)[0]
    # mean_trans_applied[2] = mean_trans_applied[2] * control_input[1]
    mean = mean +  mean_trans_applied[0:4] + mean_rot_applied[0:4]
    return mean


def get_d1(current_depth_image, bounding_box):

        #get the depth of the bounding box in the depth image
        #clip the bounding box to the image size and remove the negative values
        bounding_box[bounding_box < 0] = 0
        bounding_box[np.isnan(bounding_box)] = 0
        #if any of the bounding values is inf then set it to zero
        # bounding_box[np.isinf(bounding_box)] = 0
        track_depth = copy.deepcopy(current_depth_image)[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]), int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
        #get the median of the depth values in the bounding box excluding the zeros and the nans
        track_depth = track_depth[track_depth != 0]
        track_depth = track_depth[~np.isnan(track_depth)]
        if len(track_depth) == 0:
            return 0
        bb_depth = np.median(track_depth)
        if bb_depth < 0:
            raise Exception("bb_depth is negative")
        return bb_depth

def test_kalman_formula():
    source = "/home/rosen/mhmd/vslam_ws/data/kittiMOT/training"
    dataset = KittiLoader(
        source,
        sequence="0005",
        imgsz=[320, 320],
        stride=32,
        auto=True,
        transforms=None,
        depth_image=True,
    )
    rect_center = (222, 183)
    rect_width = 20
    rect_height = 20

    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s, extra_output = batch
        # show im0s[frame_idx] with opencv
        image = im0s[0]
        # show the image
        #define a red rectangle on the image with the center and the width and height
        cv2.rectangle(image, (int(rect_center[0]-rect_width/2), int(rect_center[1]-rect_height/2)), (int(rect_center[0]+rect_width/2), int(rect_center[1]+rect_height/2)), (0, 0, 255), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        odom = dict()
        odom["header"] = "kitti"
        odom["t_imu_cam"] = dataset.calib.T_imu_cam
        odom["oxts"] = dataset.extra_output["oxt"].packet
        depth_dict = dict()
        depth_dict["header"] = "kitti"
        depth_dict["depth_image"] = dataset.extra_output["depth_image"]
        print("odom: ", odom)
        t_imu_cam = odom["t_imu_cam"]
        oxt_packet = odom["oxts"]
        v_imu = np.array([[oxt_packet.vf], [oxt_packet.vl], [oxt_packet.vu], [1]])
        t_cam_imu = np.vstack((np.hstack((t_imu_cam[0:3,0:3].T, np.dot(-t_imu_cam[0:3,0:3].T, t_imu_cam[0:3,3].T).reshape(3,1))), [0,0,0,1]))
        v_cam = np.dot(t_cam_imu, v_imu)
        current_z_velocity_cam = v_cam[2, 0]
        current_yaw_dot = oxt_packet.wu / fps_rot
        current_yaw_dot = current_yaw_dot
        yaw_dot_list.append(current_yaw_dot)
        current_D_dot = current_z_velocity_cam / fps_rot
        this_depth_image = extra_output["depth_image"]
        current_depth_image = this_depth_image
        bb_rect = [rect_center[0], rect_center[1], rect_width, rect_height]
        #make bb_rect a numpy array
        bb_rect = np.array(bb_rect)
        control_input = np.array([current_yaw_dot, current_D_dot, get_d1(current_depth_image, bb_rect)])
        new_pos = predict(bb_rect, control_input)
        rect_center = (new_pos[0], new_pos[1])
        rect_width = new_pos[2]
        rect_height = new_pos[3]




if __name__ == "__main__":
    test_kalman_formula()
