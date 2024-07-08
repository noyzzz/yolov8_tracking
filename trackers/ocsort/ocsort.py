"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *
from yolov8.ultralytics.yolo.utils.ops import xywh2xyxy
from collections import deque
import copy
from ..MATracker import MATracker, MATrack
import time
import torch


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    if x[2] * x[3] < 0:
        w = np.sqrt(-x[2] * x[3]) #FIXME this is a hack to avoid nan
    else:
        w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(MATrack):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0


    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.

        """

        # define constant velocity model
        super().__init__()
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=9, dim_z=5)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0], 
                              [0, 0, 1, 0, 0, 0, 0, 1, 0], 
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 1], 
                              [0, 0, 0, 0, 0, 1, 0, 0, 0], 
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:5, 2:5] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # self.conf = bbox[-1]
        
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        
        if bbox is not None:
            # self.conf = bbox[-1]
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            _R = self.kf.R * self.get_d1_noise()
            self.kf.update(np.append(convert_bbox_to_z(bbox),self.get_d1()), R=_R)
        else:
            _R = self.kf.R * self.get_d1_noise()
            self.kf.update(np.append(bbox, self.get_d1()), R=_R)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        
        #if KalmanBoxTracker has attributes current_yaw_dot and current_D_dot then use them as control input
        if hasattr(KalmanBoxTracker, 'current_yaw_dot') and hasattr(KalmanBoxTracker, 'current_D_dot'):
            control_input = np.array([KalmanBoxTracker.current_yaw_dot, KalmanBoxTracker.current_D_dot, self.get_d1()])
        else:
            control_input = None
        self.kf.predict(control_input = control_input)
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    
    def get_tlwh(self):
        """
        Returns the current bounding box in tlwh format [top left x, top left y, width, height].
        """
        bbox = convert_x_to_bbox(self.kf.x)
        tlwh = np.array([bbox[0, 0], bbox[0, 1], bbox[0, 2]-bbox[0, 0], bbox[0, 3]-bbox[0, 1]])
        return tlwh


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(MATracker):
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, use_depth=False, use_odometry=False):
        """
        Sets key parameters for SORT
        """
        super().__init__(use_odometry=use_odometry, use_depth=use_depth)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0
    
    def update(self, dets, _, depth_image = None, odom = None, masks = None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.update_time(odom, self.frame_count)
        if odom is not None:
            KalmanBoxTracker.update_ego_motion(odom, self.fps)
            KalmanBoxTracker.update_depth_image(depth_image)
        # self.frame_count += 1
        cates = dets[:, 5]
        scores = dets[:, 4]
        dets = (dets[:, 0:4])

        # print("dets: ", dets)
        # print("cates: ", cates)
        # print("scores: ", scores)
        # time.sleep(2)
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)
        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = torch.Tensor(dets)[remain_inds]
        #convert dets to numpy
        dets = dets.numpy()

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            # time.sleep(1)
            return np.concatenate(ret)
        return np.empty((0, 7))












        # xyxys = dets[:, 0:4]
        # confs = dets[:, 4]
        # clss = dets[:, 5]
        
        # classes = clss.numpy()
        # xyxys = xyxys.numpy()
        # confs = confs.numpy()

        # output_results = np.column_stack((xyxys, confs, classes))
        
        # inds_low = confs > 0.1
        # inds_high = confs < self.det_thresh
        # inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        # dets_second = output_results[inds_second]  # detections for second matching
        # remain_inds = confs > self.det_thresh
        # dets = output_results[remain_inds]

        # # get predicted locations from existing trackers.
        # trks = np.zeros((len(self.trackers), 5))
        # to_del = []
        # ret = []
        # for t, trk in enumerate(trks):
        #     pos = self.trackers[t].predict()[0]
        #     trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        #     if np.any(np.isnan(pos)):
        #         to_del.append(t)
        # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # for t in reversed(to_del):
        #     self.trackers.pop(t)

        # velocities = np.array(
        #     [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        # last_boxes = np.array([trk.last_observation for trk in self.trackers])
        # k_observations = np.array(
        #     [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        # """
        #     First round of association
        # """
        # matched, unmatched_dets, unmatched_trks = associate(
        #     dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        # for m in matched:
        #     self.trackers[m[1]].update(dets[m[0], :5], dets[m[0], 5])

        # """
        #     Second round of associaton by OCR
        # """
        # # BYTE association
        # # if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
        # #     u_trks = trks[unmatched_trks]
        # #     iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
        # #     iou_left = np.array(iou_left)
        # #     if iou_left.max() > self.iou_threshold:
        # #         """
        # #             NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
        # #             get a higher performance especially on MOT17/MOT20 datasets. But we keep it
        # #             uniform here for simplicity
        # #         """
        # #         matched_indices = linear_assignment(-iou_left)
        # #         to_remove_trk_indices = []
        # #         for m in matched_indices:
        # #             det_ind, trk_ind = m[0], unmatched_trks[m[1]]
        # #             if iou_left[m[0], m[1]] < self.iou_threshold:
        # #                 continue
        # #             self.trackers[trk_ind].update(dets_second[det_ind, :5], dets_second[det_ind, 5])
        # #             to_remove_trk_indices.append(trk_ind)
        # #         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
        #     left_dets = dets[unmatched_dets]
        #     left_trks = last_boxes[unmatched_trks]
        #     iou_left = self.asso_func(left_dets, left_trks)
        #     iou_left = np.array(iou_left)
        #     if iou_left.max() > self.iou_threshold:
        #         """
        #             NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
        #             get a higher performance especially on MOT17/MOT20 datasets. But we keep it
        #             uniform here for simplicity
        #         """
        #         rematched_indices = linear_assignment(-iou_left)
        #         to_remove_det_indices = []
        #         to_remove_trk_indices = []
        #         for m in rematched_indices:
        #             det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
        #             if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
        #                 continue
        #             self.trackers[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5])
        #             to_remove_det_indices.append(det_ind)
        #             to_remove_trk_indices.append(trk_ind)
        #         unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
        #         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # for m in unmatched_trks:
        #     self.trackers[m].update(None, None)

        # # create and initialise new trackers for unmatched detections
        # for i in unmatched_dets:
        #     trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], delta_t=self.delta_t)
        #     self.trackers.append(trk)
        # i = len(self.trackers)
        # for trk in reversed(self.trackers):
        #     if trk.last_observation.sum() < 0:
        #         d = trk.get_state()[0]
        #     else:
        #         """
        #             this is optional to use the recent observation or the kalman filter prediction,
        #             we didn't notice significant difference here
        #         """
        #         d = trk.last_observation[:4]
        #     if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        #         # +1 as MOT benchmark requires positive
        #         ret.append(np.concatenate((d, [trk.id+1], [trk.cls], [trk.conf])).reshape(1, -1))
        #     # else:
        #     #     ret.append(np.concatenate((d, [trk.id+1], [-1], [trk.conf])).reshape(1, -1))
        #     i -= 1
        #     # remove dead tracklet
        #     if(trk.time_since_update > self.max_age):
        #         self.trackers.pop(i)
        # if(len(ret) > 0):
        #     return np.concatenate(ret)
        # return np.empty((0, 5))
