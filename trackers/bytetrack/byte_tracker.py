import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from yolov8.ultralytics.yolo.utils.ops import xywh2xyxy, xyxy2xywh


from trackers.bytetrack.kalman_filter import KalmanFilter
from trackers.bytetrack import matching
from trackers.bytetrack.basetrack import BaseTrack, TrackState
import time
from torch.utils.tensorboard import SummaryWriter
import datetime

IMG_WIDTH = 960
IMG_HEIGHT = 540
FOCAL_LENGTH = 480.0

class STrack(BaseTrack):
    current_yaw = 0;
    current_yaw_dot = 0;
    current_yaw_dot_filtered = 0;
    current_depth_image = None
    yaw_dot_list = deque(maxlen=2)
    shared_kalman = KalmanFilter(IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH) #TODO check the parameters
    def __init__(self, tlwh, score, cls):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        #define a mean_history list with max length 10
        self.mean_history = deque(maxlen=3)
        self.bb_depth = None


    def get_d1(self):
        #calculate the depth of the object in the depth image
        #get the bounding box of the object in the depth image
        #get the median of the depth values in the bounding box excluding the zeros and the nans
        #return the depth value
        if self.state != TrackState.Tracked and self.state != TrackState.New and self.bb_depth is not None:
            return self.bb_depth
        bounding_box = copy.deepcopy(self.tlwh)
        #get the depth of the bounding box in the depth image
        #clip the bounding box to the image size and remove the negative values
        bounding_box[bounding_box < 0] = 0
        bounding_box[np.isnan(bounding_box)] = 0
        #if any of the bounding values is inf then set it to zero
        # bounding_box[np.isinf(bounding_box)] = 0
        track_depth = copy.deepcopy(STrack.current_depth_image)[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]), int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
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
        depth_image/=10;
        STrack.current_depth_image = depth_image

    def update_ego_motion(odom, fps_rot, fps_depth):
        quat = False
        if quat:
            quaternion = (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
            #get the yaw from the quaternion not using the tf library
            yaw = np.arctan2(2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
                            1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]))
        else:
            yaw = odom.pose.pose.orientation.z
        while  abs(yaw-STrack.current_yaw) > np.pi :
            if yaw < STrack.current_yaw :
                yaw += 2*np.pi
            else:
                yaw -= 2*np.pi
        twist = odom.twist.twist
        STrack.current_yaw_dot = twist.angular.z / fps_rot # frames are being published at 20Hz in the simulator
        STrack.yaw_dot_list.append(STrack.current_yaw_dot)
        STrack.current_yaw = yaw
        STrack.current_yaw_dot_filtered = np.mean(STrack.yaw_dot_list)
        raw_x_dot = twist.linear.x
        raw_y_dot = twist.linear.y
        x_dot = raw_x_dot*np.cos(STrack.current_yaw)+raw_y_dot*np.sin(STrack.current_yaw)
        y_dot = -raw_x_dot*np.sin(STrack.current_yaw)+raw_y_dot*np.cos(STrack.current_yaw)
        STrack.current_D_dot = x_dot / fps_depth
        # print("current y_dot is ", y_dot, "   current_x_dot is ", x_dot)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, np.array([STrack.current_yaw_dot]))

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            # create a n*3 array of yaw_dot, current_D_dot and depth
            control_input = np.array([[STrack.current_yaw_dot, STrack.current_D_dot, st.get_d1() ]for st in stracks])
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, control_input)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False): #TODO kalamn update should be called with odom
        #initialize the measurement mask to all trues with size 5

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # self.cls = cls
        measurement = self.tlwh_to_xyah(new_track.tlwh)
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement)
        self.mean_history.append(self.mean)
        if new_track is not None:
            self.state = TrackState.Tracked
            self.is_activated = True

    def update_dummy_2(self):
        mean_history = np.asarray(self.mean_history)
        mean_history = mean_history
        this_mean = np.mean(mean_history, axis=0)
        self.mean[4:] = this_mean[4:]
        self.mean[4] = self.mean[4]/2
        self.mean[5] = self.mean[5]/3
    
    def update_dummy(self):
        self.mean, self.covariance = self.kalman_filter.update_dummy(
            self.mean, self.covariance, STrack.current_yaw_dot)
            

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
            mean[2] = w/h
            mean[3] = h

        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, track_thresh=0.45, match_thresh=0.8, track_buffer=25, frame_rate=30):
        self.last_time_stamp = 0 #time in seconds
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.write_log = False
        self.track_buffer=track_buffer
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size 
        self.kalman_filter = KalmanFilter(IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH) #TODO check the parameters
        self.use_depth = True
        self.use_odometry = True
        self.time_window_list = deque(maxlen=300)
        self.time_window_list.append(30)
        self.last_time = time.time()
        #initialize self.writer tensorboard writer
        self.writer = SummaryWriter(f"./runs/slow/{self.kalman_filter._q1:0.3f}_{self.kalman_filter._q4:0.5f}_{self.kalman_filter._r1:0.3f}\
                                    __{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}")

    def get_all_track_predictions(self):
        """
        Get the predictions of all the active tracks
        :return: list of bounding boxes of all the active tracks
        """
        bboxes = []
        for track in joint_stracks(self.tracked_stracks, self.lost_stracks):
            bbox = track.tlwh
            #append the track id to the bbox
            bbox = np.append(bbox, track.track_id)
            bboxes.append(bbox)
        return bboxes

    def update_time(self, odom):
        current_time = odom.header.stamp.secs + odom.header.stamp.nsecs*1e-9
        time_now = current_time
        self.fps =  (1.0/(time_now - self.last_time_stamp))*2.2
        self.fps_depth = (1.0/(time_now - self.last_time_stamp)) *2.5 #TODO This is the fps for translational motion (depth) only

        self.last_time_stamp = time_now
        # print("fps: ", self.fps)

    def update(self, dets, color_image, depth_image = None, odom = None, masks = None):
        self.update_time(odom)
        STrack.update_ego_motion(odom, self.fps, self.fps_depth)
        STrack.update_depth_image(depth_image)
        #get the current time and compare it with the last time update was called
        time_now = time.time()
        self.time_window_list.append(1.0/(time_now - self.last_time))
        #print the average and standard deviation of the time window
        # if self.frame_id % 100 == 0:
        #     print("Average fps: ", np.mean(self.time_window_list), "Standard deviation: ", np.std(self.time_window_list))
        # print((self.fps/2 - self.time_window_list[-1]))
        self.last_time = time_now

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        xyxys = dets[:, 0:4]
        xywh = xyxy2xywh(xyxys.numpy())
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()

        remain_inds = confs > self.track_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_second = xywh[inds_second]
        dets = xywh[remain_inds]
        
        scores_keep = confs[remain_inds]
        scores_second = confs[inds_second]
        
        clss_keep = classes[remain_inds]
        clss_second = classes[inds_second]
        

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(xyxy, s, c) for 
                (xyxy, s, c) in zip(dets, scores_keep, clss_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        if len(tracked_stracks) > 0:
            track_print = tracked_stracks[-1]
            # print(
            #     f"track id: {track_print.track_id:>2}, \
            #         x: {track_print.mean[0]:>5.2f}, x_dot: {track_print.mean[4]:>5.2f}, x_cov: {track_print.covariance[0,0]:>5.2f} \
            #         y: {track_print.mean[1]:>5.2f}, y_dot: {track_print.mean[5]:>5.2f}, y_cov: {track_print.covariance[1,1]:>5.2f}\
            #             , current_yaw_dot:   {STrack.current_yaw_dot:>5.2f}")
            # print (f"track id: , {track_print.track_id:>5},  x_dot:  , {100*track_print.mean[4]:>5.2f}, x_cov:  {track_print.covariance[0,0]:>5.2f} , current_yaw_dot:   {100*STrack.current_yaw_dot:>5.2f}", flush=True)

        if self.write_log:
            #take log from the track means and covariances and visualize them in tensorboard
            for track in joint_stracks(self.tracked_stracks, self.lost_stracks):
                track_id = track.track_id
                mean = track.mean
                covariance = track.covariance
                if self.writer:
                    self.writer.add_scalar(f"loc/Track {track_id} x", mean[0], self.frame_id)
                    self.writer.add_scalar(f"loc/Track {track_id} y", mean[1], self.frame_id)
                    self.writer.add_scalar(f"loc/Track {track_id} a", mean[2], self.frame_id)
                    self.writer.add_scalar(f"loc/Track {track_id} h", mean[3], self.frame_id)
                    self.writer.add_scalar(f"vel/Track {track_id} x_dot", mean[4], self.frame_id)
                    self.writer.add_scalar(f"vel/Track {track_id} y_dot", mean[5], self.frame_id)
                    self.writer.add_scalar(f"vel/Track {track_id} a_dot", mean[6], self.frame_id)
                    self.writer.add_scalar(f"vel/Track {track_id} h_dot", mean[7], self.frame_id)
                    self.writer.add_scalar(f"loc_cov/Track {track_id} x_cov", covariance[0,0], self.frame_id)
                    self.writer.add_scalar(f"loc_cov/Track {track_id} y_cov", covariance[1,1], self.frame_id)
                    self.writer.add_scalar(f"loc_cov/Track {track_id} a_cov", covariance[2,2], self.frame_id)
                    self.writer.add_scalar(f"loc_cov/Track {track_id} h_cov", covariance[3,3], self.frame_id)
                    self.writer.add_scalar(f"vel_cov/Track {track_id} x_dot_cov", covariance[4,4], self.frame_id)
                    self.writer.add_scalar(f"vel_cov/Track {track_id} y_dot_cov", covariance[5,5], self.frame_id)
                    self.writer.add_scalar(f"vel_cov/Track {track_id} a_dot_cov", covariance[6,6], self.frame_id)
                    self.writer.add_scalar(f"vel_cov/Track {track_id} h_dot_cov", covariance[7,7], self.frame_id)
                    # self.writer.add_scalar("vel/current_yaw_dot/raw", STrack.current_yaw_dot, self.frame_id)
                    # self.writer.add_scalars('current_yaw_dot', {'raw': STrack.current_yaw_dot,
                                                                # 'filtered': STrack.current_yaw_dot_filtered}, self.frame_id)
                    # self.writer.add_scalar(f"vel/current_yaw_dot_filtered", STrack.current_yaw_dot_filtered, self.frame_id)
                    #flush the writer
                    self.writer.flush()
                
                




        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        #if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(xywh, s, c) for (xywh, s, c) in zip(dets_second, scores_second, clss_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            # if len(track.mean_history) > 0:
            #     track.update_dummy_2()
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        #if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        #for tracks in lost_stracks, if the x and y are not in the size of the image times 1.5 then remove the track
        for track in self.lost_stracks:
            if track.mean[0] < 0 or track.mean[0] > 1.5*IMG_WIDTH or track.mean[1] < 0 or track.mean[1] > 1.5*IMG_HEIGHT:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        #copy self.lost_stracks and change the class of the tracks to -1
        local_lost_stracks = copy.deepcopy(self.lost_stracks)
        for track in local_lost_stracks:
            track.cls = -1
        test_all_stracks = joint_stracks(self.tracked_stracks, local_lost_stracks)
        for t in output_stracks:
            output= []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)
            output.append(t.mean[4])
            outputs.append(output)

        return outputs
#track_id, class_id, conf

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
