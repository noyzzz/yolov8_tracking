import numpy as np
from collections import namedtuple
import cv2
import sys
sys.path.append("../")
import time
from extern.build import depth_utils
from scipy.spatial.transform import Rotation as R


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                key, value = line.split(' ',1)
            else:
                key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def upsample_velodyne(velodata_cam, params):
    total_vbeams = params.get('total_vbeams', 128)
    total_hbeams = params.get('total_hbeams', 1500)
    vbeam_fov = params.get('vbeam_fov', 0.2)
    hbeam_fov = params.get('hbeam_fov', 0.08)
    phioffset = 10
    scale = params.get('upsample', 1.0)

    vscale = 1.0
    hscale = 1.0
    vbeams = int(total_vbeams * vscale)
    hbeams = int(total_hbeams * hscale)
    vf = vbeam_fov / vscale
    hf = hbeam_fov / hscale
    rmap = np.zeros((vbeams, hbeams), dtype=np.float32)

    # Cast to Angles
    rtp = np.zeros((velodata_cam.shape[0], 3))
    rtp[:, 0] = np.sqrt(np.sum(np.square(velodata_cam), axis=1))
    rtp[:, 1] = np.arctan2(velodata_cam[:, 0], velodata_cam[:, 2]) * (180 / np.pi)
    rtp[:, 2] = np.arcsin(velodata_cam[:, 1] / rtp[:, 0]) * (180 / np.pi) - phioffset

    # Bin Data
    for i in range(rtp.shape[0]):
        r, theta, phi = rtp[i]
        thetabin = int(((theta / hf) + hbeams / 2) + 0.5)
        phibin = int(((phi / vf) + vbeams / 2) + 0.5)
        if not (0 <= thetabin < hbeams and 0 <= phibin < vbeams):
            continue
        current_r = rmap[phibin, thetabin]
        if r < current_r or current_r == 0:
            rmap[phibin, thetabin] = r

    # Upsample
    vscale = vscale * scale
    hscale = hscale * scale
    vbeams = int(total_vbeams * vscale)
    hbeams = int(total_hbeams * hscale)
    vf = vbeam_fov / vscale
    hf = hbeam_fov / hscale
    rmap = cv2.resize(rmap, (0, 0), fx=hscale, fy=vscale, interpolation=cv2.INTER_NEAREST)

    # Regenerate
    xyz_new = np.ones((rmap.size, 4))
    for phibin in range(rmap.shape[0]):
        for thetabin in range(rmap.shape[1]):
            i = phibin * rmap.shape[1] + thetabin
            phi = ((phibin - (vbeams / 2)) * vf + phioffset) * (np.pi / 180)
            theta = ((thetabin - (hbeams / 2)) * hf) * (np.pi / 180)
            r = rmap[phibin, thetabin]
            xyz_new[i, 0] = r * np.cos(phi) * np.sin(theta)
            xyz_new[i, 1] = r * np.sin(phi)
            xyz_new[i, 2] = r * np.cos(phi) * np.cos(theta)

    return xyz_new

def generate_depth(velodata, intr_raw, M_velo2cam, width, height, params):
    upsample = params.get('upsample', 1.0)
    filtering = params.get('filtering', 1)

    # Transform to Camera Frame
    velodata_cam = (M_velo2cam @ velodata.T).T

    # Remove points behind camera
    valid_indices = velodata_cam[:, 2] >= 0.1
    velodata_cam = velodata_cam[valid_indices]

    # Upsample
    if upsample > 1:
        velodata_cam = upsample_velodyne(velodata_cam, params)

    # Project and Generate Pixels
    velodata_cam_proj = (intr_raw @ velodata_cam.T).T
    velodata_cam_proj[:, 0] /= velodata_cam_proj[:, 2]
    velodata_cam_proj[:, 1] /= velodata_cam_proj[:, 2]

    # Z Buffer assignment
    dmap_raw = np.zeros((height, width))
    for i in range(velodata_cam_proj.shape[0]):
        u, v = (int(velodata_cam_proj[i, 0] + 0.5), int(velodata_cam_proj[i, 1] + 0.5)) if not np.isnan(velodata_cam_proj[i, 0]) and not np.isnan(velodata_cam_proj[i, 1]) else (0, 0)
        if not (0 <= u < width and 0 <= v < height):
            continue
        z = velodata_cam_proj[i, 2]
        if z < dmap_raw[v, u] or dmap_raw[v, u] == 0:
            dmap_raw[v, u] = z

    # Filtering
    dmap_cleaned = np.zeros((height, width))
    offset = filtering
    for v in range(offset, height - offset - 1):
        for u in range(offset, width - offset - 1):
            z = dmap_raw[v, u]
            bad = False

            # Check neighbors
            for vv in range(v - offset, v + offset + 1):
                for uu in range(u - offset, u + offset + 1):
                    if vv == v and uu == u:
                        continue
                    zn = dmap_raw[vv, uu]
                    if zn == 0:
                        continue
                    if zn - z < -1:
                        bad = True
                        break

            if not bad:
                dmap_cleaned[v, u] = z

    return dmap_cleaned


def bilinear_interpolation(observed_values, height, width):
    # Create an empty array to store the recovered image
    recovered_image = np.zeros((height, width))

    # Iterate over each pixel in the observed values
    # for i in range(int(height/2)-50, height):
    for i in range(height):
        for j in range(width):
            # If the pixel is observed, copy its value to the recovered image
            if observed_values[i, j] != 0:
                recovered_image[i, j] = observed_values[i, j]
            else:
                # Perform bilinear interpolation for the missing pixels
                # Find the nearest observed pixels
                count = 0
                sum = 0
                kernel = 1
                while count == 0 and kernel < 5:
                    us, vs = np.meshgrid(range(i-kernel,i+1+kernel), range(j-kernel,j+1+kernel))
                    for u,v in zip(us.flatten(), vs.flatten()):
                        if u >= 0 and u < height and v >= 0 and v < width and observed_values[u, v] != 0 and (u, v) != (i, j):
                            # Use the value of the nearest pixel to fill in the missing pixel
                            sum += (observed_values[u, v]  / kernel)
                            count += 1
                    if count > 0:
                        recovered_image[i, j] = sum / count
                    else:
                        kernel += 1

    return recovered_image


def my_approx_depth(velodata, intr_raw, M_velo2cam, width, height, points_count_threshold=1, initial_kernel=3, max_kernel=2):
    total_weighted_value = 0
    total_weight = 0
    points_count = 0

    velodata_cam = (M_velo2cam @ velodata.T).T

    # Remove points behind camera
    valid_indices = velodata_cam[:, 2] >= 0.1
    velodata_cam = velodata_cam[valid_indices]

    # Project and Generate Pixels
    velodata_cam_proj = (intr_raw @ velodata_cam.T).T
    velodata_cam_proj[:, 0] /= velodata_cam_proj[:, 2]
    velodata_cam_proj[:, 1] /= velodata_cam_proj[:, 2]

    depth = np.sqrt(velodata_cam[:, 1]**2 + velodata_cam[:, 1]**2 + velodata_cam[:, 2]**2)
    max_depth = np.max(depth)
    depth = depth / np.max(depth)

    sampled_points = np.hstack((velodata_cam_proj[:, :2], depth.reshape(-1, 1))) # N * (x, y, depth)
    sampled_points = np.array(sampled_points, dtype=np.float32)
    # delete the points with x value less than 0 or greater than width+5

    sampled_points = sampled_points[(sampled_points[:, 0] >= -max_kernel) & (sampled_points[:, 0] < width+max_kernel)]
    # delete the points with y value less than 0 or greater than height+5
    sampled_points = sampled_points[(sampled_points[:, 1] >= -max_kernel) & (sampled_points[:, 1] < height+max_kernel)]
    #convert sample points to numpy.ndarray[numpy.float32]
    t1 = time.time()
    depth_map = depth_utils.depth_cloud2depth_map_wrapper(sampled_points, width, height, max_kernel);
    # print("Time taken: ", time.time() - t1)
    #i have a type example.Point which requires 3 float values as arguments
    #convert sampled_points to a list of example.Point

    # cpp_sampled_points = [depth_utils.Point(x, y, z) for x, y, z in sampled_points]

    # print("length of sampled_points: ", len(cpp_sampled_points))
    # depth_map_kiri = depth_utils.depth_cloud2depth_map(cpp_sampled_points, width, height, max_kernel)
    # depth_map_kiri = np.array(depth_map_kiri)

    #rotate the depth_map by 90 degrees
    depth_map = np.rot90(depth_map, 3)
    #compare depth_map and depth_map_kiri elementwise and tell me if they are equal
    #flip the depth_map horizontally
    depth_map = np.fliplr(depth_map)
    return np.array(depth_map)*max_depth

def approx_depth(velodata, intr_raw, M_velo2cam, width, height, points_count_threshold=1, initial_kernel=3, max_kernel=4):
    total_weighted_value = 0
    total_weight = 0
    points_count = 0

    velodata_cam = (M_velo2cam @ velodata.T).T

    # Remove points behind camera
    valid_indices = velodata_cam[:, 2] >= 0.1
    velodata_cam = velodata_cam[valid_indices]

    # Project and Generate Pixels
    velodata_cam_proj = (intr_raw @ velodata_cam.T).T
    velodata_cam_proj[:, 0] /= velodata_cam_proj[:, 2]
    velodata_cam_proj[:, 1] /= velodata_cam_proj[:, 2]

    depth = np.sqrt(velodata_cam[:, 1]**2 + velodata_cam[:, 1]**2 + velodata_cam[:, 2]**2)
    depth = depth / np.max(depth)
    sampled_points = np.hstack((velodata_cam_proj[:, :2], depth.reshape(-1, 1))) # N * (x, y, depth)
    # delete the points with x value less than 0 or greater than width+5
    sampled_points = sampled_points[(sampled_points[:, 0] >= -max_kernel) & (sampled_points[:, 0] < width+max_kernel)]
    # delete the points with y value less than 0 or greater than height+5
    sampled_points = sampled_points[(sampled_points[:, 1] >= -max_kernel) & (sampled_points[:, 1] < height+max_kernel)]
    
    reconstructed_image = np.zeros((height, width), dtype=np.float32)
    t1 = time.time()
    for y in range(int(height/2), height):
        for x in range(width):
            kernel = initial_kernel
            total_weighted_value = 0
            total_weight = 0
            points_count = 0
            while points_count < points_count_threshold and kernel < max_kernel:
                indices = np.where(np.linalg.norm(sampled_points[:, :2] - np.array([x, y]), axis=1) < kernel)
                neighbor_coords = sampled_points[indices][:, :2]
                neighbor_values = sampled_points[indices][:, 2]
                points_count = neighbor_coords.shape[0]
                kernel += 1

            if points_count == 0:
                pixel_value = 0

            else:
                neighbor_points = np.hstack((neighbor_coords, neighbor_values.reshape(-1, 1)))
                for (sx, sy, value) in neighbor_points:
                    distance = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                    # if distance < 1e-2:
                    #     pixel_value = value  # Exact match, no need for interpolation
                    #     raise RuntimeError
                    weight = 1 / distance
                    total_weighted_value += weight * value
                    total_weight += weight
                pixel_value = total_weighted_value / total_weight

            reconstructed_image[y, x] = pixel_value
    # print("Time taken: ", time.time() - t1)
    return reconstructed_image

# from github.com/castacks/DytanVO/blob/main/evaluator/transformation.py
def SEs2ses(data):
    '''
    data: N x 12
    ses: N x 6
    '''
    data_size = data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        ses[i,:] = SE2se(line2mat(data[i]))
    return ses

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result

def SO2so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()


def line2mat(line_data):
    '''
    12 -> 4 x 4
    '''
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

MotionPacket = namedtuple('MotionPacket',
                        'vf, vl, vu, ' +
                        'wf, wl, wu, ' +
                        'trsl_accuracy, rot_accuracy, ')

MotionData = namedtuple('MotionData', 'packet')

if __name__ == "__main__":


    # Load the image
    np.random.seed(0)
    im = cv2.imread("runs/sample.jpg", cv2.IMREAD_GRAYSCALE)
    # resizes the image to 128, 384
    im = cv2.resize(im, (300, 256))

    im = im.astype(np.float32)/255
    observed_fraction = 0.10
    height, width = im.shape
    sampled_mask = np.random.choice([0, 1], size=im.shape, p=[1 - observed_fraction, observed_fraction])
    # cpmpute the indices of the observed pixels
    sampled_points = np.argwhere(sampled_mask.T == 1)
    observed_values = im * sampled_mask
    reconstructed_image = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if sampled_mask[y, x] == 1:
                reconstructed_image[y, x] = observed_values[y, x]
            else:
                kernel = 2
                total_weighted_value = 0
                total_weight = 0
                points_count = 0
                while points_count < 3 and kernel < 4:
                    indices = np.where(np.linalg.norm(sampled_points - np.array([x, y]),axis=1) < kernel)
                    neighbor_coords = sampled_points[indices]
                    neighbor_values = np.array([im[j,i] for (i,j) in neighbor_coords])
                    points_count = len(indices[0])
                    kernel += 1

                if points_count == 0:
                    pixel_value = 0

                else:
                    neighbor_points = np.hstack((neighbor_coords, neighbor_values.reshape(-1, 1)))
                    for (sx, sy, value) in neighbor_points:
                        distance = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                        if distance < 1e-2:
                            pixel_value =  value # Exact match, no need for interpolation
                            break
                        weight = 1 / distance
                        total_weighted_value += weight * value
                        total_weight += weight
                    pixel_value = total_weighted_value / total_weight

                reconstructed_image[y, x] = pixel_value

    # Perform bilinear interpolation to recover the image
    # recovered_image = bilinear_interpolation(observed_values, width=image_size[0], height=image_size[1])

    # Display the original and recovered images using cv2.imshow
    cv2.imshow('Observed Pixels', observed_values)
    # cv2.imshow('Recovered Image', recovered_image.astype(np.uint8))
    cv2.imshow('Recovered Image', reconstructed_image)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()