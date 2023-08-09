import math
import operator
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
import trackers.unsvstracker.kalman_filter as kf
from trackers.unsvstracker.matching import iou_distance, linear_assignment
from typing import List


class Detection:
    def __init__(self,*, det = None, yyxx = None, encoding = None, mask = None):
        if det is not None:
            self.center_point = Detection.xyxy2xywh(det[0:4].numpy())
            self.conf = det[4]
            self.cls = det[5]
            self.encoding = None
        else:
            assert yyxx is not None
            # initialize when encoding is available
            self.center_point = Detection.yyxx_to_xywh(yyxx)
            self.encoding = encoding
            self.mask = mask

    def calculate_depth(self, depth_img):
        #using the mask to calculate the depth by getting the mean of the depth values in the mask
        #mask is a binary image with 1s in the pixels that belong to the object and 0s in the pixels that don't
        #depth_img is the depth image of the frame
        #depth_img is a numpy array of shape (height, width)
        #mask is a numpy array of shape (height, width)
        #returns the mean depth of the object
        #resize the mask to the size of the depth image
        resized_mask = cv2.resize(self.mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        #change depth_img to float
        depth_img = depth_img.astype(np.float32)
        #change values of 0 in the depth_img to nan
        depth_img[depth_img == 0] = np.nan
        return np.nanmean(depth_img[resized_mask == 1])

    @property
    def tlbr(self):
        # generate tlbr from center point
        # center point is [x, y, width, height]
        # output is [x1, y1, x2, y2] which represent the top left and bottom right corners of the bounding box
        x, y, width, height = self.center_point
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        return np.array([x1, y1, x2, y2])

    def yyxx_to_xywh(yyxx):
        # input is [y1, y2, x1, x2]
        # output is [x, y, width, height] (x, y) is the center of the box
        return np.array(
            [
                (yyxx[2] + yyxx[3]) / 2,
                (yyxx[0] + yyxx[1]) / 2,
                yyxx[3] - yyxx[2],
                yyxx[1] - yyxx[0],
            ]
        )

    def yyxx_to_xyah(yyxx):
        # input is [y1, y2, x1, x2]
        # output is [x, y, aspect ratio, height]
        return np.array(
            [
                (yyxx[2] + yyxx[3]) / 2,
                (yyxx[0] + yyxx[1]) / 2,
                (yyxx[3] - yyxx[2]) / (yyxx[1] - yyxx[0]),
                yyxx[1] - yyxx[0],
            ]
        )

    def get_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        x, y, w, h = self.center_point
        return np.array([x, y, w / h, h])

    def get_bounding_box(self):
        x, y, w, h = self.center_point
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        return pts.reshape((-1, 1, 2))

    def get_tlwh(self):
        x, y, w, h = self.center_point
        return np.array([x - w / 2, y - h / 2, w, h])

    def get_yyxx(self):
        x, y, w, h = self.center_point
        return np.array([y - h / 2, y + h / 2, x - w / 2, x + w / 2])

    def yyxx_to_xyxy(yyxx):
        return np.array([yyxx[2], yyxx[0], yyxx[3], yyxx[1]])

    def yyxx_to_tlwh(yyxx):
        return np.array([yyxx[2], yyxx[0], yyxx[3] - yyxx[2], yyxx[1] - yyxx[0]])

    def get_xyxy2(self):
        x, y, w, h = self.center_point
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    def xyxy2xywh(x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    def get_class(self):
        return self.cls


vis_autoenc = tf.keras.models.load_model(
    "/home/rosen/my_self_tracker/autoencoder_model_16kernel"
)


import tensorflow as tf
import tensorflow_hub as hub

feature_vec_generator = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
            output_shape=[1280],
            trainable=False,
        )
    ]
)


# os.mkdir(this_folder_path)


class VisualThreshold:
    def __init__(self):
        self.dissim_list = deque(maxlen=1000)

    def get_threshold(self):
        mu, std = norm.fit(self.dissim_list)
        return mu - 1 * std

    def get_threshold_track2track(self):
        mu, std = norm.fit(self.dissim_list)
        return mu - 1 * std

    def add_frame_imgs(self, frame_img_encodings):
        for obj1_it in range(len(frame_img_encodings)):
            for obj2_it in range(len(frame_img_encodings)):
                if obj1_it == obj2_it:
                    continue
                self.dissim_list.append(
                    vis_comparator(
                        frame_img_encodings[obj1_it], frame_img_encodings[obj2_it]
                    )
                )


def get_bb(dataset, frame_id, frames_detections, image_id):
    frame = get_frame_visual(dataset, frame_id)
    coordinates = frames_detections[frame_id][image_id]
    visual = frame[coordinates[0] : coordinates[1], coordinates[2] : coordinates[3]]
    return visual


def get_vis_rep2(x):
    x = vis_autoenc.get_layer("input_1")(x)
    x = vis_autoenc.get_layer("conv2d")(x)
    x = vis_autoenc.get_layer("max_pooling2d")(x)
    x = vis_autoenc.get_layer("conv2d_1")(x)
    x = vis_autoenc.get_layer("max_pooling2d_1")(x)
    return x


def get_vis_rep(x):
    return feature_vec_generator.predict(x)


def resize_normalize_img(img):
    return np.asarray(
        cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype("float32")
        / 255.0
    )


def det_parser(det_path):
    frame_detections = {}
    frame_id_max = {}
    with open(det_path) as f:
        lines = f.readlines()
    for line in lines:
        line_splitted = line.split(",")
        frame_id, bb_left, bb_top, bb_width, bb_height = (
            int(float(line_splitted[0])),
            int(float(line_splitted[2])),
            int(float(line_splitted[3])),
            int(float(line_splitted[4])),
            int(float(line_splitted[5])),
        )
        confidence = float(line_splitted[6].split("\n")[0])
        if confidence < 0.5:
            continue
        if not frame_id in frame_id_max.keys():
            frame_id_max[frame_id] = 0
        this_id = frame_id_max[frame_id]
        frame_id_max[frame_id] = frame_id_max[frame_id] + 1
        if not frame_id in frame_detections.keys():
            frame_detections[frame_id] = {}
        frame_detections[frame_id][this_id] = [
            max(bb_top, 0),
            (bb_top) + (bb_height),
            max(bb_left, 0),
            (bb_left) + (bb_width),
        ]
    return frame_detections


def get_frame_visual(dataset, frame_id):
    frame = cv2.imread(
        dataset_root
        + dataset
        + images_relative_path
        + "{:06d}".format(frame_id)
        + ".jpg"
    )
    return frame


def get_bb(dataset, frame_id, frames_detections, image_id):
    frame = get_frame_visual(dataset, frame_id)
    coordinates = frames_detections[frame_id][image_id]
    visual = frame[coordinates[0] : coordinates[1], coordinates[2] : coordinates[3]]
    return visual


import pickle


def convert_image_to_ae_inp(visual_dict):
    """
    :param visual_dict:
    :return: images id list, visuals ready for autoencoder
    """
    image_id_list = []
    visuals_list = []
    # for image_id in visual_dict:
    #     image_id_list.append(image_id)
    #     visuals_list.append(cv2.resize(visual_dict[image_id], (224, 224), interpolation=cv2.INTER_LINEAR))
    # visuals_list = np.asarray([l.astype('float32') / 255. for l in visuals_list])

    filehandler = open("visual_dict_1", "wb")
    for image_id in visual_dict:
        image_id_list.append(image_id)
        visuals_list.append(resize_normalize_img(visual_dict[image_id]))
    visuals_list = np.array(visuals_list)
    return image_id_list, visuals_list


REPRESENTER_WINDOW_SIZE = 5


class Track:
    last_used_id = 0
    shared_kalman = kf.KalmanFilter()

    def __init__(
        self,
        frame_id,
        image_id,
        position,
        vis_encoding,
        object_class,
        kalman_filter: kf.KalmanFilter,
    ):
        self.track_id = Track.last_used_id
        Track.last_used_id = Track.last_used_id + 1
        self.last_seen = frame_id
        self.average_vis_encoding = [vis_encoding]
        self.last_median_position = position
        self.track_history = [(frame_id, image_id, position, vis_encoding)]
        self.cls = object_class
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(
            Detection.yyxx_to_xyah(position)
        )

    def get_encoding(self):
        return self.average_vis_encoding[-1]
    
    @property
    def tlbr(self):
        # get the tlbr from the mean
        # first four values of mean is [x, y, aspect ratio, height]
        # output is [x1, y1, x2, y2] which represent the top left and bottom right corners of the bounding box
        x, y, aspect_ratio, height = self.mean[:4]
        width = aspect_ratio * height
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        return np.array([x1, y1, x2, y2])

    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in tracks])
            multi_covariance = np.asarray([st.covariance for st in tracks])
            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    def __lt__(self, other):
        return self.last_seen > other.last_seen

    def is_merging_candidate(self, other: "Track", merging_visual_threshold):
        """
        check if the two tracks are mergeable. Will be called when a potential tracks is ready to be added to "open tracks" list.
        check1: if frame ids of the objects are not conflicting.
        check2: if the two encodings are similar
        :param other:
        :return:
        """
        # check2:
        median_list_vis_encoding = []
        for track_vis_encoding in self.average_vis_encoding:
            for other_vis_encoding in other.average_vis_encoding:
                this_visual_penalty = vis_comparator(
                    track_vis_encoding, other_vis_encoding
                )
                median_list_vis_encoding.append(this_visual_penalty)
        visual_penalty = np.median(median_list_vis_encoding)
        if visual_penalty > merging_visual_threshold:
            return False, visual_penalty
        # check1:
        track_object_frame_id_list = []
        other_object_frame_id_list = []
        for i in range(len(self.track_history)):
            this_image_id = self.track_history[i][0]
            track_object_frame_id_list.append(this_image_id)
        for i in range(len(other.track_history)):
            this_image_id = other.track_history[i][0]
            other_object_frame_id_list.append(this_image_id)
        intersection = list(
            set(track_object_frame_id_list) & set(other_object_frame_id_list)
        )
        if len(intersection) != 0:
            return False, visual_penalty
        return True, visual_penalty

    def add_detection(self, frame_id, image_id, image_position, image_encoding):
        if frame_id > self.last_seen:
            self.last_seen = frame_id
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, Detection.yyxx_to_xyah(image_position)
        )
        self.track_history.append((frame_id, image_id, image_position, image_encoding))
        self.track_history.sort(key=lambda tup: tup[0])
        # find median of 10 last positions:
        positions = []
        for j in range(max(0, len(self.track_history) - 10), len(self.track_history)):
            (positions.append(self.track_history[j][2]))
        self.last_median_position = [
            np.median(np.asarray(positions).T[0]),
            np.median(np.asarray(positions).T[1]),
            np.median(np.asarray(positions).T[2]),
            np.median(np.asarray(positions).T[3]),
        ]
        self.average_vis_encoding.append(image_encoding)
        self.average_vis_encoding = self.average_vis_encoding[-REPRESENTER_WINDOW_SIZE:]

    @classmethod
    def merge_tracks(cls, main_track: "Track", sec_track: "Track"):
        main_track_frame_ids = []
        for frame_id, image_id, position, vis_encoding in main_track.track_history:
            main_track_frame_ids.append(frame_id)
        for frame_id, image_id, position, vis_encoding in sec_track.track_history:
            if frame_id in main_track_frame_ids:
                continue
            main_track.add_detection(frame_id, image_id, position, vis_encoding)
        return main_track


MAX_FRAME_DISTANCE = 30

POTENTIAL_REMOVAL_THRESHOLD = 10

POTENTIAL_TO_OPEN_THRESHOLD = 10


def track_filter(
    open_tracks,
    closed_tracks,
    potential_tracks,
    frame_id,
    visual_threshold,
    removed_potential_tracks,
):
    tracks_to_remove = []
    for track in open_tracks:
        if abs(track.last_seen - frame_id) > MAX_FRAME_DISTANCE:
            closed_tracks.append(track)
            tracks_to_remove.append(track)
    for track_to_remove in tracks_to_remove:
        open_tracks.remove(track_to_remove)

    tracks_to_remove = []
    for track in potential_tracks:
        if abs(track.last_seen - frame_id) > POTENTIAL_REMOVAL_THRESHOLD:
            tracks_to_remove.append(track)
    for track_to_remove in tracks_to_remove:
        # second chance for ready to remove potential tracks
        open_track_merging_candidates = []
        for open_track_ind in range(len(open_tracks)):
            is_mergeable, visual_penalty = track_to_remove.is_merging_candidate(
                open_tracks[open_track_ind],
                visual_threshold.get_threshold_track2track(),
            )
            if is_mergeable:
                open_track_merging_candidates.append((open_track_ind, visual_penalty))
        if len(open_track_merging_candidates) != 0:
            open_track_merging_candidates.sort(key=operator.itemgetter(1))
            selected_open_track_ind = open_track_merging_candidates[0][0]
            merged_track = Track.merge_tracks(
                open_tracks[selected_open_track_ind], track_to_remove
            )
            del open_tracks[selected_open_track_ind]
            open_tracks.append(merged_track)
            print("******************merged*************************")
        else:
            removed_potential_tracks.append(track_to_remove)
        potential_tracks.remove(track_to_remove)

    tracks_to_add_to_open = []
    for track in potential_tracks:
        if len(track.track_history) > POTENTIAL_TO_OPEN_THRESHOLD:
            ### add track to "open track list" after this
            open_track_merging_candidates = []
            for open_track_ind in range(len(open_tracks)):
                is_mergeable, visual_penalty = track.is_merging_candidate(
                    open_tracks[open_track_ind], visual_threshold.get_threshold()
                )
                if is_mergeable:
                    open_track_merging_candidates.append(
                        (open_track_ind, visual_penalty)
                    )
            if len(open_track_merging_candidates) != 0:
                open_track_merging_candidates.sort(key=operator.itemgetter(1))
                selected_open_track_ind = open_track_merging_candidates[0][0]
                merged_track = Track.merge_tracks(
                    open_tracks[selected_open_track_ind], track
                )
                del open_tracks[selected_open_track_ind]
                open_tracks.append(merged_track)
                print("******************merged*************************")

            else:
                open_tracks.append(track)
            ### END OF add track to "open track list" after this
            tracks_to_add_to_open.append(track)
    for track_to_remove in tracks_to_add_to_open:
        potential_tracks.remove(track_to_remove)


def position_comparator(position1, position2):
    center_1 = (
        ((position1[1] + position1[0]) / 2.0),
        ((position1[3] + position1[2]) / 2.0),
    )
    center_2 = (
        ((position2[1] + position2[0]) / 2.0),
        ((position2[3] + position2[2]) / 2.0),
    )
    first_vec = np.asarray(center_2) - np.asarray(center_1)
    distance = np.linalg.norm(first_vec)
    return distance


def vis_comparator(vis_encoding1, vis_encoding2):
    return np.linalg.norm(vis_encoding1 - vis_encoding2)


def penalty_calculator(
    position1, position2, vis_encoding1, vis_encoding_list, frame_difference
):
    median_list_vis_encoding = []
    for track_vis_encoding in vis_encoding_list:
        this_visual_penalty = vis_comparator(vis_encoding1, track_vis_encoding)
        median_list_vis_encoding.append(this_visual_penalty)
    visual_penalty = np.median(median_list_vis_encoding)
    position_penalty = position_comparator(position1, position2)
    if position_penalty > 200:
        return position_penalty * 30
    return visual_penalty  # + position_penalty #+ frame_difference * 100


def matcher(position, encoding, available_tracks, frame_id, threshold):
    if len(available_tracks) == 0:
        return False, -1
    available_tracks.sort()
    available_tracks_penalty = []
    for track_it in range(len(available_tracks)):
        this_track = available_tracks[track_it]
        this_penalty = penalty_calculator(
            position,
            this_track.last_median_position,
            encoding,
            this_track.average_vis_encoding,
            (frame_id - this_track.last_seen),
        )
        available_tracks_penalty.append(this_penalty)
    index_min = min(
        range(len(available_tracks_penalty)), key=available_tracks_penalty.__getitem__
    )
    if available_tracks_penalty[index_min] > threshold:
        return False, -1
    else:
        # print(available_tracks_penalty[index_min])
        return True, available_tracks[index_min]


def find_min_penalty(position, encoding, open_tracks, frame_id):
    if len(open_tracks) == 0:
        return 0
    min_penalty = 10000000
    for track_it in range(len(open_tracks)):
        this_track = open_tracks[track_it]
        this_penalty = penalty_calculator(
            position,
            this_track.last_median_position,
            encoding,
            this_track.average_vis_encoding,
            (frame_id - this_track.last_seen),
        )
        if this_penalty < min_penalty:
            min_penalty = this_penalty
    return min_penalty


def generate_mot_challenge_output(track):
    text_out = ""
    for track_det in track.track_history:
        frame_id, image_id, position, _ = track_det
        track_id = track.track_id
        text_out = (
            text_out
            + str(frame_id)
            + ","
            + str(track_id)
            + ","
            + str(position[2])
            + ","
            + str(position[0])
            + ","
            + str(abs(position[3] - position[2]))
            + ","
            + str(abs(position[0] - position[1]))
            + ","
            + "-1,-1,-1,-1\n"
        )
    return text_out


from pathlib import Path
import shutil


def finished_track2file(finished_track, track_type, detection_coordinates, dataset):
    root_path = track_root_path
    if track_type == "closed":
        down_path = "tracks/"
    if track_type == "potential":
        down_path = "potential_tracks/"
    if track_type == "appended":
        down_path = "appended_tracks/"
    image_tracks_folder_path = (
        "/home/rosen/PycharmProjects/Self-Tracker/"
        + root_path
        + dataset
        + down_path
        + str(finished_track.track_id)
        + "/"
    )
    if (
        Path(image_tracks_folder_path).exists()
        and Path(image_tracks_folder_path).is_dir()
    ):
        shutil.rmtree(Path(image_tracks_folder_path))
    os.mkdir(image_tracks_folder_path)
    for tracked_detection in finished_track.track_history:
        frame_id, image_id, frame_positions, image_encoding = tracked_detection
        bb = get_bb(dataset, frame_id, detection_coordinates, image_id)
        im = Image.fromarray(bb)
        im.save(image_tracks_folder_path + str(frame_id) + ".jpeg")


def frame_neg_exp_gen(dataset_frame_positions, frame_number):
    frame_visuals = {}
    frame_positions = {}
    for bb in dataset_frame_positions[frame_ids[frame_number]]:
        frame_visuals[bb] = get_bb(
            dataset, frame_ids[frame_number], dataset_frame_positions, bb
        )
        frame_positions[bb] = dataset_frame_positions[frame_ids[frame_number]][bb]
    frame_img_ids, frame_img_visuals_ae = convert_image_to_ae_inp(frame_visuals)
    encodings = get_vis_rep(frame_img_visuals_ae)
    # Image.fromarray(np.uint8(rotate_image(frame_img_visuals_ae[1]*255,50))).convert('RGB').save("A.jpg")
    out1 = []
    out2 = []
    out = []
    for image1_id in range(len(frame_img_ids)):
        for image2_id in range(len(frame_img_ids)):
            if image2_id >= image1_id:
                continue
            this_out1 = np.concatenate(
                (
                    np.asarray(encodings[frame_img_ids[image1_id]]).flatten(),
                    np.asarray(frame_positions[frame_img_ids[image1_id]]),
                ),
                axis=0,
            )
            this_out2 = np.concatenate(
                (
                    np.asarray(encodings[frame_img_ids[image2_id]]).flatten(),
                    np.asarray(frame_positions[frame_img_ids[image2_id]]),
                ),
                axis=0,
            )
            out.append((this_out1, this_out2))
    return out, [0] * len(out)


def frame_pos_exp_gen(dataset_frame_positions, frame_number):
    frame_visuals = {}
    frame_positions = {}
    for bb in dataset_frame_positions[frame_ids[frame_number]]:
        frame_visuals[bb] = get_bb(
            dataset, frame_ids[frame_number], dataset_frame_positions, bb
        )
        frame_positions[bb] = dataset_frame_positions[frame_ids[frame_number]][bb]
    frame_img_ids, frame_img_visuals_ae = convert_image_to_ae_inp(frame_visuals)
    encodings = get_vis_rep(frame_img_visuals_ae)
    out = []
    for image1_id in range(len(frame_img_ids)):
        this_out1 = np.concatenate(
            (
                np.asarray(encodings[frame_img_ids[image1_id]]).flatten(),
                np.asarray(frame_positions[frame_img_ids[image1_id]]),
            ),
            axis=0,
        )
        this_out2 = np.concatenate(
            (
                np.asarray(encodings[frame_img_ids[image2_id]]).flatten(),
                np.asarray(frame_positions[frame_img_ids[image2_id]]),
            ),
            axis=0,
        )
        out.append((this_out1, this_out2))
    return out, [1] * len(out)


class UnsVSTrack(object):
    current_time = 0

    def __init__(self):
        self.use_odometry = True
        self.use_depth = True
        self.visual_threshold = VisualThreshold()
        Track.last_used_id = 0
        self.open_tracks = []
        self.closed_tracks = []
        self.potential_tracks = []
        self.removed_potential_tracks = []


    def calculate_visual_cost(self, tracks: List[Track], detections: List[Detection]):
        # tracks is a list of tracks
        # detections is a list of detections
        # each track has an encoding which is accessed by track.get_encoding()
        # each detection has an encoding which is accessed by detection.encoding
        #this function returns a cost matrix of size len(tracks) x len(detections)
        #cost_matrix[i][j] is the cost of assigning track i to detection j
        #the cost is the l2 distance between the two encodings
        #finally the values in the cost matrix are normalized to be between 0 and 1
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for track_ind, track in enumerate(tracks):
            for detection_ind, detection in enumerate(detections):
                cost_matrix[track_ind][detection_ind] = np.linalg.norm(track.get_encoding() - detection.encoding)
        if  len(cost_matrix) > 0 and len(cost_matrix[0]) > 0 and np.max(cost_matrix) != 0:
            cost_matrix = cost_matrix / np.max(cost_matrix)
        return cost_matrix


    def kalman_matcher(self, tracks, detections, visual_threshold, depth_image):
        # each track has already been predicted in the multi_predict function
        # each track has a mean and covariance
        # each detection has a position and encoding
        # mean is a 4x1 vector of the form [x, y, aspect ratio, height]
        # covariance is a 4x4 matrix
        # detection has a center point of the form [x, y, width, height]
        cost_matrix = iou_distance(tracks, detections)
        visual_cost_matrix = self.calculate_visual_cost(tracks, detections)
        cost_matrix = cost_matrix + visual_cost_matrix
        #normalize the cost matrix
        if len(cost_matrix) > 0 and len(cost_matrix[0]) > 0 and np.max(cost_matrix) != 0:
            cost_matrix = cost_matrix / np.max(cost_matrix)
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=0.8)
        # u_track and u_detection are the unassigned tracks and detections
        for track_ind, detection_ind in matches:
            track = tracks[track_ind]
            detection = detections[detection_ind]
            #print the depth of the detection
            print("                                     track id            ", track.track_id  ,  "depth:                 ", detection.calculate_depth(depth_image))
            track.add_detection(
                UnsVSTrack.current_time,
                detection_ind,
                detection.get_yyxx(),
                detection.encoding,
            )
        unmatched_tracks = []
        for track_ind in u_track:
            track = tracks[track_ind]
            unmatched_tracks.append(track)

        unmatched_detections = []
        for detection_ind in u_detection:
            detection = detections[detection_ind]
            unmatched_detections.append(detection)

        return matches, unmatched_tracks, unmatched_detections

    def update(self, dets, frame_img, depth_img, odom, masks):
        UnsVSTrack.current_time = UnsVSTrack.current_time + 1
        curr_detections = []
        for det in dets:
            curr_detections.append(Detection(det = det))
        print("Track last ID USED: ", Track.last_used_id)
        print("open tracks len  ", len(self.open_tracks))
        track_filter(
            self.open_tracks,
            self.closed_tracks,
            self.potential_tracks,
            UnsVSTrack.current_time,
            self.visual_threshold,
            self.removed_potential_tracks,
        )
        print("all tracks:----", len(self.open_tracks) + len(self.closed_tracks))
        frame_visuals = {}
        frame_positions = {}
        frame_img_classes = {}
        np_masks = {}
        for index, (this_detection, this_mask) in enumerate(zip(curr_detections, np.array(masks.cpu()))):
            this_detection: Detection
            frame_visuals[index] = frame_img[
                int(this_detection.get_xyxy2()[1]) : int(this_detection.get_xyxy2()[3]),
                int(this_detection.get_xyxy2()[0]) : int(this_detection.get_xyxy2()[2]),
            ]
            np_masks[index] = this_mask
            frame_positions[index] = this_detection.get_yyxx()
            frame_img_classes[index] = this_detection.get_class()

        frame_img_ids, frame_img_visuals_ae = convert_image_to_ae_inp(frame_visuals)
        encodings = get_vis_rep(frame_img_visuals_ae)
        # using frame_img_ids and encodings and frame_positions create list of type: Detection
        detections = []
        for (image_id, encoding, position, mask) in zip(
            frame_img_ids, encodings, list(frame_positions.values()), np_masks.values()
        ):
            new_detection = Detection(yyxx = position, encoding = encoding, mask = mask)
            detections.append(new_detection)

        vis_rep_dict = dict(zip(frame_img_ids, encodings))
        self.visual_threshold.add_frame_imgs(vis_rep_dict)
        print("VISUAL THRESHOLD:    ", self.visual_threshold.get_threshold())
        matched_tracks_ids = []
        unmatched_image_ids = frame_img_ids.copy()
        # create a list that combines two lists of tracks: open and potential
        Track.multi_predict(self.open_tracks + self.potential_tracks)

        matches, unmatched_tracks, unmatched_detections = self.kalman_matcher(
            self.open_tracks,
            detections,
            self.visual_threshold.get_threshold(), depth_img
        )

        #now check mathcing with potential tracks just for the unmatched detections
        matches, unmatched_tracks, unmatched_detections_after_pot = self.kalman_matcher(
            self.potential_tracks,
            unmatched_detections,
            self.visual_threshold.get_threshold(), depth_img
        )


        #create a new track for each unmatched detectionk, use enumerate
        for i , unmatched_detection in enumerate(unmatched_detections_after_pot):
            new_track = Track(
                UnsVSTrack.current_time,
                i,
                unmatched_detection.get_yyxx(),
                unmatched_detection.encoding,
                1,
                kf.KalmanFilter())
            

            if UnsVSTrack.current_time == 0:
                self.open_tracks.append(new_track)
            else:
                self.potential_tracks.append(new_track)

            

        # for image_id in frame_img_ids:
        #     available_tracks = []
        #     for track in self.open_tracks:
        #         if track.track_id not in matched_tracks_ids:
        #             available_tracks.append(track)
        #     matched, matched_track = matcher(
        #         frame_positions[image_id],
        #         encodings[image_id],
        #         available_tracks,
        #         UnsVSTrack.current_time,
        #         self.visual_threshold.get_threshold(),
        #     )
        #     if matched:
        #         matched_tracks_ids.append(matched_track.track_id)
        #         matched_track.add_detection(
        #             UnsVSTrack.current_time,
        #             image_id,
        #             frame_positions[image_id],
        #             encodings[image_id],
        #         )
        #         unmatched_image_ids.remove(image_id)
        # matched_tracks_ids = []
        # copied_unmatched_image_ids = unmatched_image_ids.copy()
        # for unmatched_image_id in copied_unmatched_image_ids:
        #     available_tracks = []
        #     for track in self.potential_tracks:
        #         if track.track_id not in matched_tracks_ids:
        #             available_tracks.append(track)
        #     matched, matched_track = matcher(
        #         frame_positions[unmatched_image_id],
        #         encodings[unmatched_image_id],
        #         available_tracks,
        #         UnsVSTrack.current_time,
        #         self.visual_threshold.get_threshold(),
        #     )
        #     if matched:
        #         matched_tracks_ids.append(matched_track.track_id)
        #         matched_track.add_detection(
        #             UnsVSTrack.current_time,
        #             unmatched_image_id,
        #             frame_positions[unmatched_image_id],
        #             encodings[unmatched_image_id],
        #         )
        #         unmatched_image_ids.remove(unmatched_image_id)

        # for unmatched_image_id in unmatched_image_ids:
        #     new_track = Track(
        #         UnsVSTrack.current_time,
        #         unmatched_image_id,
        #         frame_positions[unmatched_image_id],
        #         encodings[unmatched_image_id],
        #         frame_img_classes[unmatched_image_id],
        #     )
        #     if UnsVSTrack.current_time == 0:
        #         open_tracks.append(new_track)
        #     else:
        #         self.potential_tracks.append(new_track)
        outputs = []
        for track in self.open_tracks + self.potential_tracks:
            if track.last_seen == UnsVSTrack.current_time:
                output = []
                xyxy = track.last_median_position

                # xyxy = np.squeeze(xyxy, axis=0)
                output.extend(Detection.yyxx_to_xyxy(xyxy))
                output.append(track.track_id)
                output.append(track.cls)
                output.append(1)
                outputs.append(output)
        return outputs


# if __name__ == '__main__':
#     main()
