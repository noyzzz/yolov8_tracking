# Ultralytics YOLO 🚀, GPL-3.0 license
from __future__ import print_function

import glob
import os
# from pathlib import Path
# from threading import Thread
import cv2
import numpy as np

import sys
sys.path.append("../")
from collections import namedtuple

from ultralytics.yolo.data.augment import LetterBox
import kitti_loader_utils as utils

class KittiLoader:
    """Load KittiMOT dataset
    Each frame is returned as a tuple of (base_path, image, image_raw, label, extra_output)
    extra_output is a dictionary of {"depth": depthmap: np.NDarray(H,W),
                                     "velodyne": velo: np.NDarray(N,4), 
                                     "oxt": oxt: namedtuple("OxtsData", ["packet", "T_w_imu"])
                                     "dets": dets, 
                                     "gt": None}
    each oxt packet is a namedtuple of OxtsPacket(lat, lon, alt, roll, pitch, yaw,
                                                     vn, ve, vf, vl, vu, 
                                                     ax, ay, az, af, al, au, 
                                                     wx, wy, wz, wf, wl, wu, 
                                                     pos_accuracy, vel_accuracy, 
                                                     navstat, numsats, posmode, velmode, orimode)
    lat:   latitude of the oxts-unit (deg)
    lon:   longitude of the oxts-unit (deg)
    alt:   altitude of the oxts-unit (m)
    roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    vn:    velocity towards north (m/s)
    ve:    velocity towards east (m/s)
    vf:    forward velocity, i.e. parallel to earth-surface (m/s)
    vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
    vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
    ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
    ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
    ay:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
    af:    forward acceleration (m/s^2)
    al:    leftward acceleration (m/s^2)
    au:    upward acceleration (m/s^2)
    wx:    angular rate around x (rad/s)
    wy:    angular rate around y (rad/s)
    wz:    angular rate around z (rad/s)
    wf:    angular rate around forward axis (rad/s)
    wl:    angular rate around leftward axis (rad/s)
    wu:    angular rate around upward axis (rad/s)
    pos_accuracy:  velocity accuracy (north/east in m)
    vel_accuracy:  velocity accuracy (north/east in m/s)
    navstat:       navigation status (see navstat_to_string)
    numsats:       number of satellites tracked by primary GPS receiver
    posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
    velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
    orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)


    Args:
        kitii_base_path: path to kittiMOT dataset
        sequence: sequence number
        imgsz: image size
        stride: stride for letterbox
        auto: auto letterbox
        transforms: image transformations

"""
    def __init__(self, kitii_base_path: str, sequence: str, imgsz=640, stride=32, auto=True, transforms=None, **kwargs):
        self.base_path = kitii_base_path
        self.sequence = sequence
        self.transforms = transforms
        self.index = 0 
        self.imtype = 'png'
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.kwargs = kwargs
        # Find all the data files
        self._get_file_lists()
        self._load_calib()
        self._load_oxts() # it loads all the oxts data for a sequence. not an effiecient way to load oxts data
        self._load_dets() # it loads all the det data for a sequence. not an effiecient way to load dets data


    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_02',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_03',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.base_path,
                        'velodyne',
                        self.sequence,
                         '*.bin')))
        
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.base_path,
                        'oxts',
                         f'{self.sequence}.txt')))
        
        self.dets_files = sorted(glob.glob(
            os.path.join(self.base_path,
                        'permatrack_kitti_test',
                        f'{self.sequence}.txt')))
        
    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.base_path,'calib', f'{self.sequence}.txt')
        filedata = utils.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        Tr_velo_cam = np.reshape(filedata['Tr_velo_cam'], (3, 4))
        data['Tr_velo_cam'] = np.vstack([Tr_velo_cam, [0, 0, 0, 1]])

        Tr_imu_velo = np.reshape(filedata['Tr_imu_velo'], (3, 4))
        data['Tr_imu_velo'] = np.vstack([Tr_imu_velo, [0, 0, 0, 1]])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        self._calib = namedtuple('CalibData', data.keys())(*data.values())
    
    @property
    def calib(self):
        """Return a namedtuple of calibration parameters."""
        return self._calib
    
    
    
    def _load_oxts(self):
        """Load OXTS data from file.

           Poses are given in an East-North-Up coordinate system 
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in self.oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = utils.OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = utils.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = utils.transform_from_rot_trans(R, t - origin)

                    oxts.append(utils.OxtsData(packet, T_w_imu))

        self.oxts = oxts

    def _load_dets(self):
        """Load detections from file.
            method is taken from
            https://github.com/noahcao/OC_SORT/blob/7a390a5f35dbbb45df6cd584588c216aea527248/tools/run_ocsort_public.py#L101

        """
        for seq_file in self.dets_files:
            cats = ['Pedestrian', 'Car', 'Cyclist', "Van", "Truck"]
            cat_ids = {cat: i for i, cat in enumerate(cats)}

            print("starting seq {}".format(self.sequence))
            seq_trks = np.empty((0, 18))
            seq_file = open(seq_file)
            lines = seq_file.readlines()
            line_count = 0 
            for line in lines:
                print("{}/{}".format(line_count,len(lines)))
                line_count+=1
                line = line.strip()
                tmps = line.strip().split()
                tmps[2] = cat_ids[tmps[2]]
                trk = np.array([float(d) for d in tmps])
                trk = np.expand_dims(trk, axis=0)
                seq_trks = np.concatenate([seq_trks, trk], axis=0)
        self.seq_trks = seq_trks
                
            
    def __getitem__(self, frame_index):
        """Return the data from a particular frame_index."""
        # Load the data from disk
        cam2_0 = cv2.imread(self.cam2_files[frame_index].strip())

        # cam2_0 = cv2.resize(cam2_0, (320, 192))
        velo = np.fromfile(self.velo_files[frame_index].strip(), dtype=np.float32).reshape((-1, 4))
        velo[:, 3] = 1.
        upsampled_params = {"filtering": 1, "upsample": 1}
        intr_raw = np.hstack((self.calib.K_cam2, np.zeros((3, 1))))
        if self.kwargs.get("depth_image", False):
            depthmap = utils.generate_depth(velodata=velo, M_velo2cam=self.calib.Tr_velo_cam, 
                                            width=cam2_0.shape[1], height=cam2_0.shape[0],
                                            intr_raw=intr_raw, params=upsampled_params)
        else:
            depthmap = None
            
        oxt = self.oxts[frame_index]
        _det_ind = list(range(6,10)) + [-2, -1]
        # Assuming that the frame index starts with 0 in detection file
        dets = self.seq_trks[np.where(self.seq_trks[:,0]==frame_index)][:,_det_ind]
        # Apply the data transformations
        if self.transforms is not None:
            cam2 = self.transforms(cam2_0)
        else:
            cam2 = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=cam2_0)
            cam2 = cam2.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            cam2 = np.ascontiguousarray(cam2)  # contiguous
        
        
        self.extra_output = {"depth_image": depthmap, "velodyne": velo, "oxt": oxt, "dets": dets, "gt": None} #TODO: add gt

        return self.base_path, cam2, [cam2_0], None, "", self.extra_output
    
    def __iter__(self):
        """Return the iterator object."""
        self.index = 0
        return self

    def __next__(self):
        """Return the next sequence."""
        # Get the data from the next index
        data = self.__getitem__(self.index)

        # Increment the index and loop if necessary
        self.index += 1
        if self.index >= len(self):
            raise StopIteration

        return data
    
    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.cam2_files)
    

if __name__ == "__main__":
    import time
    dataset = KittiLoader("/home/apera/mhmd/kittiMOT/data_kittiMOT/testing", "0001", transforms = None)
    print(len(dataset))
    print(dataset[-1])
    for data in dataset:
        print(data[0])
        cv2.imshow("image", data[1])
        print(data[1].shape)
        print(data[2].shape)
        print(data[3])
        print(data[4]["velodyne"].shape)
        print(data[4]["oxt"].packet)
        print(data[4]["oxt"].T_w_imu)
        print(data[4]["depth"].shape)
        print(data[4]["dets"])
        # time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break