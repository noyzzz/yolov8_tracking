import numpy as np
import cv2

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
def test_kitti_odom(im0, t_imu_cam, oxt_packet):
    # Extract linear velocity in IMU frame
    v_imu = np.array([[oxt_packet.vf], [oxt_packet.vl], [oxt_packet.vu], [1]])

    t_cam_imu = np.vstack((np.hstack((t_imu_cam[0:3,0:3].T, np.dot(-t_imu_cam[0:3,0:3].T, t_imu_cam[0:3,3].T).reshape(3,1))), [0,0,0,1]))
    
    # Transform linear velocity to camera frame
    v_cam = np.dot(t_cam_imu, v_imu)
    current_z_velocity_cam = v_cam[2, 0]
    current_yaw_dot = oxt_packet.wu

    
    
    # Annotate the image with velocity information
    cv2.putText(im0, f'x: {current_z_velocity_cam:.2f} m/s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f'rot: {current_yaw_dot:.2f} rad/s', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return im0


def test_kitti_depth(im0, depth_image):
    #convert depth image type to float32
    depth_image = depth_image.astype(np.float32)
    normalized_depth_map = depth_image / np.max(depth_image)
    cmap = plt.cm.gray_r
    depth_map_color = cmap(normalized_depth_map)
    #depth map values are between 0 and 100. 100 is the farthest distance. i want the max value to be a very bright color and the min value to be a very dark color
    #show the depth image instead of the RGB image
    im0 = depth_image
    
    return depth_map_color
