from typing import List
import cv2
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from filterpy.kalman import KalmanFilter
from yolov8.ultralytics.yolo.utils.ops import xywh2xyxy, xyxy2xywh

class Detection:
    def __init__(self, det):
        self.center_point = Detection.xyxy2xywh(det[0:4].numpy())
        self.conf = det[4]
        self.cls = det[5]
        pass

    def get_bounding_box(self):
        x, y, w, h = self.center_point
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
        return pts.reshape((-1,1,2))
    
    def get_tlwh(self):
        x, y, w, h = self.center_point
        return np.array([x - w/2, y - h/2, w, h])
    
    def get_xyxy2(self):
        x, y, w, h = self.center_point
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    
    def xyxy2xywh(x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y
    

class Track:
    last_track_id = 0
    dt = 1.0
    #constant velocity model
    F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
    q = 0.1
    G = np.array([[dt**2 / 2, 0],
              [0, dt**2 / 2],
              [dt, 0],
              [0, dt]])
    Q = G.dot(G.T) * q**2
    R = np.array([[1, 0],
              [0, 1]])
    

    def __init__(self, detection: Detection):
        self.track_id = Track.last_track_id
        self.cls = detection.cls
        Track.last_track_id += 1
        self.last_update_time = BYTETracker.current_time
        self.updated = False
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        x0, y0 = detection.center_point[0:2]
        self.kf.x = np.array([x0, y0, 0, 0]) #the third and fourth element are the velocity
        P0 = np.diag([100, 100, 10, 10]) #high uncertainty for pos and low uncertainty for vel
        self.kf.P = P0
        self.kf.F = Track.F
        self.kf.H = Track.H
        self.kf.Q = Track.Q
        self.kf.R = Track.R
        self.trace = deque(maxlen=30)

        pass
    
    def predict(self):
        self.kf.predict()
        pass

    def update(self, detection: Detection):
        self.last_update_time = BYTETracker.current_time
        self.updated = True
        self.kf.update(detection.center_point[0:2])
        pass

class BYTETracker(object):
    MAX_AGE = 10
    current_time = 0
    SHOW_DETECTIONS = False
    CONF_TH = 0.8
    def __init__(self, track_thresh=0.45, match_thresh=0.8, track_buffer=25, frame_rate=30):
        self.tracks : List[Track] = []
        pass

    def iou(self, box1, box2):
        #calculate the iou between two boxes
        #return a float number between 0 and 1
        #box1 and box2 are numpy arrays of size 4
        #box1 = [x1, y1, x2, y2]
        #box2 = [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1:
            return 0
        else:
            intersection = (x2 - x1) * (y2 - y1)
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
            return intersection / union
        pass

    def calculate_iou(self, track, detection):
        track : Track
        detection: Detection
        track_center = track.kf.x[0:2]
        box_width = detection.center_point[2]
        box_height = detection.center_point[3]
        det_center = detection.center_point[0:2]
        track_box = np.array([track_center[0] - box_width/2, track_center[1] - box_height/2, track_center[0] + box_width/2, track_center[1] + box_height/2])
        det_box = np.array([det_center[0] - box_width/2, det_center[1] - box_height/2, det_center[0] + box_width/2, det_center[1] + box_height/2])
        iou = self.iou(track_box, det_box)
        return iou


        

        pass
    def match(self, tracks, detections):
        if len(detections) == 0 or len(tracks) == 0:
            return []
        track_det_iou = np.zeros((len(tracks), len(detections)))
        matching_tuples = []
        for i, track in enumerate(tracks):
            track : Track
            for j, detection in enumerate(detections):
                detection : Detection
                track_det_iou[i, j] = self.calculate_iou(track, detection)
                #if detection class is not the same as track class, set iou to 0
                if track.cls != detection.cls:
                    track_det_iou[i, j] = 0
        
        while np.max(track_det_iou) > 0:
            max_iou = np.max(track_det_iou)
            max_iou_index = np.where(track_det_iou == max_iou)
            matching_tuples.append((tracks[max_iou_index[0][0]], detections[max_iou_index[1][0]]))
            track_det_iou[max_iou_index[0][0], :] = 0
            track_det_iou[:, max_iou_index[1][0]] = 0
        return matching_tuples


        
        pass
    def get_hlconf(det_list):
        h_list : List[Detection] = [] 
        l_list : List[Detection] = [] 
        for det in det_list :
            det: Detection
            if det.conf > BYTETracker.CONF_TH:
                h_list.append(det)
            else:
                l_list.append(det)
        return h_list, l_list
            

    def update(self, dets, frame_img):
        BYTETracker.current_time = BYTETracker.current_time + 1
        #set updated to False for all tracks
        for track in self.tracks:
            track : Track
            track.updated = False
        #seperate the detections based on conf
        curr_detections = []
        for det in dets:
            curr_detections.append(Detection(det))
            if BYTETracker.SHOW_DETECTIONS:
                cv2.polylines(frame_img,[curr_detections[-1].get_bounding_box()],True,(0,255,255),thickness=2)
                cv2.imshow('image',frame_img)
        d_high, d_low = BYTETracker.get_hlconf(curr_detections)
        all_matches  = []
        if len(self.tracks) == 0:
            #initialize tracks for every detection
            for detection in curr_detections:
                detection : Detection
                self.tracks.append(Track(detection))
                all_matches.append([(self.tracks[-1], detection)])
            
        else:

            matching_tuples = self.match(self.tracks,d_high)
            for track, detection in matching_tuples:
                track : Track
                detection : Detection
                track.update(detection)
                track.trace.append(detection.center_point)
            #add a copy of the matching tuples to all_matches 
            all_matches.append(matching_tuples.copy())
            
            #find tracks that have no match
            remain_tracks = []
            for track in self.tracks:
                track : Track
                if track not in [t for t, d in matching_tuples]:
                    remain_tracks.append(track)
            #find detections that have no match
            remain_detections_h = []
            for detection in d_high:
                detection : Detection
                if detection not in [d for t, d in matching_tuples]:
                    remain_detections_h.append(detection)
            

            #now match the remain tracks with d_low
            matching_tuples = self.match(remain_tracks, d_low)
            for track, detection in matching_tuples:
                track : Track
                detection : Detection
                track.update(detection)
                track.trace.append(detection.center_point)
            #add a copy of the matching tuples to all_matches
            all_matches.append(matching_tuples.copy())
            
            #find tracks that have no match by checking if they are updated
            re_remain_tracks = []
            for track in self.tracks:
                track : Track
                if not track.updated:
                    re_remain_tracks.append(track)
            #ignore the remain detections that are in d_low
            #for remain_detection in remain_detections_h init a new track
            for remain_detection in remain_detections_h:
                remain_detection : Detection
                self.tracks.append(Track(remain_detection))
            
            #for re_remain_tracks check if the last update is 30 frames ago, if so, delete the track
            for re_remain_track in re_remain_tracks:
                re_remain_track : Track
                if BYTETracker.current_time - re_remain_track.last_update_time > BYTETracker.MAX_AGE:
                    self.tracks.remove(re_remain_track)

        outputs = []
        #search for matches in all_matches and loop through each pair of track and detection
        for matching_tuples in all_matches:
            for track, detection in matching_tuples:
                track : Track
                detection : Detection
                output = []
                xyxy = detection.get_xyxy2()
                # xyxy = np.squeeze(xyxy, axis=0)
                output.extend(xyxy)
                output.append(track.track_id)
                output.append(track.cls)
                output.append(detection.conf)
                outputs.append(output)
        return outputs
    




