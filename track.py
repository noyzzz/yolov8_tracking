#!/usr/bin/env python3.8
import argparse
import copy
import queue
import threading
import time
import cv2
import os
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#add .. to path
# sys.path.append("/home/rosen/tracking_catkin_ws/src/my_tracker")
print(sys.path)
# from my_tracker.msg import ImageDetectionMessage
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import subprocess

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
np.random.seed(0)
torch.manual_seed(0)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker
from ros_classes import image_converter

def add_noise2tensor(tensor, image_shape, noise_scale=9/10):
    """
    tensor: [N, 6]
    """
    noise = torch.randn_like(tensor) * noise_scale
    # add (tensor[:, :2] - tensor[:, :0]) * noise to tensor[:, :2]
    tensor[:, 0] += (tensor[:, 2] - tensor[:, 0]) * noise[:,0]
    tensor[:, 1] += (tensor[:, 2] - tensor[:, 0]) * noise[:,1]
    tensor[:, 2] += (tensor[:, 3] - tensor[:, 1]) * noise[:,2]
    tensor[:, 3] += (tensor[:, 3] - tensor[:, 1]) * noise[:,3]
    
    # tensor[:, :4] += noise[:, :4]
    # clip the tensor[:, :4] to be in the range of image_shape
    tensor[:, 0] = torch.clamp(tensor[:, 0], 0, image_shape[1])
    tensor[:, 1] = torch.clamp(tensor[:, 1], 0, image_shape[0])
    tensor[:, 2] = torch.clamp(tensor[:, 2], 0, image_shape[1])
    tensor[:, 3] = torch.clamp(tensor[:, 3], 0, image_shape[0])
    #make sure that the width and height are positive not even 0
    tensor[:, 2] = torch.max(tensor[:, 2], tensor[:, 0] + 1)
    tensor[:, 3] = torch.max(tensor[:, 3], tensor[:, 1] + 1)
    return tensor

def play_bag(is_ros_bag):
    if is_ros_bag:
        player_state = subprocess.Popen(["rosbag", "play", "/home/rosen/tracking_catkin_ws/src/my_tracker/bags/carla_stationary_obj_slow.bag"])

def get_intersection(bbox, image_bbox):
    """
    bbox: [x1, y1, x2, y2]
    image_bbox: [x1, y1, x2, y2]
    """
    x1 = max(bbox[0], image_bbox[0])
    y1 = max(bbox[1], image_bbox[1])
    x2 = min(bbox[2], image_bbox[2])
    y2 = min(bbox[3], image_bbox[3])
    intersection = (x2 - x1) * (y2 - y1)
    return intersection

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        ros_package = 0,
        ros_bag = 1,
        op_mode = "eval"
):
    # OP_MODE = "EVAL" #YOLO or EVAL; EVAL uses the ground truth detections
    is_ros = isinstance(source, image_converter)
    #copy source to avoid changing the original
    source_copy = copy.copy(source)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    #check if source is instance of ros_classes.image_converter
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) or is_ros
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name+'_'+tracking_method+'_'+op_mode if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    # what is the name of last child folder of save_dir
    text_file_name = save_dir.name
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        if is_ros:
            dataset = LoadStreams(
            source_copy,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        else:
            dataset = LoadStreams(
                source,
                imgsz=imgsz,
                stride=stride,
                auto=pt,
                transforms=getattr(model.model, 'transforms', None),
                vid_stride=vid_stride
            )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        #if is ros node 
        if is_ros:
            path, im, im0s, vid_cap, s, extra_output = batch
        else:
            path, im, im0s, vid_cap, s = batch


        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            time1 = round(time.time() * 1000)
            # print(im.shape)
            # create a pytorch random tensor with size of 1,3,192,320
            my_tensor = torch.rand(1, 3, 192, 320,device=device)
            #change my_tensor to im
            preds = model(im, augment=augment, visualize=visualize)
            time2 = round(time.time() * 1000)
            # print("inference time", time2 - time1)
        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                nms_dets_list = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                nms_dets_list = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # mkdir save_path = str(save_dir / txt_file_name /
        # if not os.path.exists(str(project)):
        #     os.mkdir(str(project))
        # Process detections
        for i, det in enumerate(nms_dets_list):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                if not is_ros:
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    txt_file_name = "tracks_preds"
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0
            # if not is_ros:
            txt_path = str(project / text_file_name / text_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
            with dt[3]:
                # #if is ros node and reset_simulation_signal is true reset simulation
                # if is_ros and extra_output["reset_signal"]:
                #     print("reset simulation*********************")
                #     tracker_list = []
                #     for i in range(bs):
                #         tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
                #         tracker_list.append(tracker, )
                #         if hasattr(tracker_list[i], 'model'):
                #             if hasattr(tracker_list[i].model, 'warmup'):
                #                 tracker_list[i].model.warmup()
                #     outputs = [None] * bs
                depth_image = extra_output["depth_image"]
                if extra_output["odom"] is not None:
                    odom = extra_output["odom"]
                modified_gt_list = None 
                if extra_output["gt"] is not None:  
                    modified_gt_list = []
                    modified_annotation_gt_list = [] # for the generation of the ground truth
                    gt_list = extra_output["gt"]
                    for gt in gt_list:
                        gt_vals = list(gt.values())
                        ### TESTING ONLY WITH ID #1
                        # if gt_vals[0] != 70:
                        # #     print("*******************************removing ids 11 and 94*****************************************")
                        #     continue

                        this_xyxy = gt_vals[1:5]
                        #if xyxy is out of the image or if any of them is nan or negative, skip it
                        if this_xyxy[0] < 0 or this_xyxy[1] < 0 or this_xyxy[2] > im0.shape[1] or this_xyxy[3] > im0.shape[0] or np.isnan(this_xyxy).any():
                            continue
                        this_conf = 1.0
                        this_cls = 0
                        this_msg = [this_xyxy[0], this_xyxy[1], this_xyxy[2], this_xyxy[3], this_conf, this_cls]
                        this_annotated_msg = [gt_vals[0], this_xyxy[0], this_xyxy[1], this_xyxy[2], this_xyxy[3], this_conf, this_cls]
                        this_det = torch.tensor([this_msg])
                        modified_gt_list.append(this_det)
                        modified_annotation_gt_list.append(this_annotated_msg)
                        #convert modified_gt_list to torch tensor
                        #generate a random number with probability 0.7
                        output_random = np.random.rand()
                    if len(modified_gt_list) and output_random < 0.6:
                        modified_gt_list = torch.cat(modified_gt_list, dim=0)
                        modified_gt_list = add_noise2tensor(modified_gt_list, [im0.shape[0], im0.shape[1]], 0.10) 
                    else:
                        modified_gt_list = torch.empty((0,6))
                outputs[i] = None
                track_pred_tlwhs = None
                if hasattr(tracker_list[i], "use_depth") and hasattr(tracker_list[i], "use_odometry") and tracker_list[i].use_depth and tracker_list[i].use_odometry:
                    if op_mode == "yolo":
                        outputs[i] = tracker_list[i].update(det.cpu(), im0, depth_image, odom, None)
                    elif op_mode == "eval":
                        outputs[i] = tracker_list[i].update(modified_gt_list, im0, depth_image, odom, None)
                else:
                    if op_mode == "yolo":
                        outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    elif op_mode == "eval":
                        outputs[i] = tracker_list[i].update(modified_gt_list, im0)
            #     track_pred_tlwhs = tracker_list[i].get_all_track_predictions()

            #         #what is each det element? [x1, y1, x2, y2, conf, cls, cls_conf]
            #         # outputs[i] =  tracker_list[i].update(det.cpu(), im0)

            # for track_pred in track_pred_tlwhs:
            #     xyxy = ops.xywh2xyxy(track_pred[0:4])
            #     track_id = track_pred[4]
            #     #if xyxy is out of the image or if any of them is nan, do not draw it 
            #     if xyxy[0] < 0 or xyxy[1] < 0 or xyxy[2] > im0.shape[1] or xyxy[3] > im0.shape[0] or np.isnan(xyxy).any():
            #         continue
                # annotator.box_label(xyxy, str(track_id), color=(255, 0, 0))

            if det is not None and len(det):  
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    
            
            save_gt = True
            gt_path = str(save_dir / "gt")  # im.txt
            # if gt_path does not exist, create it
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            if save_gt and len(modified_annotation_gt_list) > 0:
                for gt in modified_annotation_gt_list:
                    this_frame_idx = frame_idx
                    this_id = gt[0]
                    this_x = gt[1]
                    this_y = gt[2]
                    this_w = gt[3] - gt[1]
                    this_h = gt[4] - gt[2]
                    this_msg = [this_frame_idx+1, this_id, this_x, this_y, this_w, this_h, 1, 1, 1]
                    with open(gt_path + '/gt.txt', 'a') as f:
                        f.write(('%g ' * 9 + '\n') % tuple(this_msg))


            for j, (output) in enumerate(outputs[i]):
                
                bbox = output[0:4]
                id = output[4]
                cls = output[5]
                conf = output[6]
                if len(output) > 7:
                    depth = output[7]
                    try:
                        depth = float(depth)
                    except:
                        depth = -1

                else:
                    depth = -1


                if save_txt:
                    # to MOT format
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    # Write MOT compliant results to file
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = None if hide_labels else "NO DET  " + str(id) if c == -1 else (f'{id} {names[c]}' if hide_conf else \
                        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                    label += f' {depth:.2f}' if depth is not None else ''
                    color = colors(c, True)

                    #check if bbox has any intersection with the image and give the intersection
                    intersection = get_intersection(bbox, [0, 0, im0.shape[1], im0.shape[0]])

                    #if bbox is out of the image or if any of them is nan, do not draw it
                        #clip the bbox to the image
                    bbox[0] = np.clip(bbox[0], 0, im0.shape[1])
                    bbox[1] = np.clip(bbox[1], 0, im0.shape[0])
                    bbox[2] = np.clip(bbox[2], 0, im0.shape[1])
                    bbox[3] = np.clip(bbox[3], 0, im0.shape[0])
                    #if nan, do not draw it
                    if not np.isnan(np.array(bbox)).any() and intersection > 0:
                        annotator.box_label(bbox, label, color=color)
                    
                    if save_trajectories and tracking_method == 'strongsort':
                        q = output[7]
                        tracker_list[i].trajectory(im0, q, color=color)
                    if save_crop:
                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                        if c == -1: c = 1 #if no detection just save with class 1 for visualization
                        save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{j}.jpg', BGR=True)
                            
                
            # Stream results
            im0 = annotator.result()
            #save im0 as jpg with name: frame_idx.jpg in subfolder of frames in save_dir
            #create  subfolder of frames in save_dir


            save_path = str(save_dir/ 'tracks'/ f'{frame_idx}.jpg')
            cv2.imwrite(save_path, im0)

            if ros_package == "1": #it means that the image_detection message type is being generated and published
                from my_tracker.msg import ImageDetectionMessage
                im0_flatten = im0.flatten().tolist()
                im0_height = im0.shape[0]
                im0_width = im0.shape[1]
                image_detection_message = ImageDetectionMessage()
                image_detection_message.im_width = im0_width
                image_detection_message.im_height = im0_height
                image_detection_message.im_data = im0_flatten
                if outputs[0] is not None and len(outputs[0]) > 0:
                    outputs[0] = np.array(outputs[0])
                    tracks_object_bbs = outputs[0][:,:5] #track_id_size x (x1, y1, x2, y2, track_id)
                    tracks_flatten = tracks_object_bbs.flatten().tolist()
                    tracks_height = tracks_object_bbs.shape[0]
                    tracks_width = tracks_object_bbs.shape[1]
                    image_detection_message.tracks_width = tracks_width
                    image_detection_message.tracks_height = tracks_height
                    image_detection_message.tracks_data = tracks_flatten
                else:
                    image_detection_message.tracks_width = 0
                    image_detection_message.tracks_height = 0
                    image_detection_message.tracks_data = []
                source_copy.track_publisher.publish(image_detection_message)
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_vid:
    #     s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    with open(save_dir / 'seqinfo.ini', 'w') as file:
        file.write('[Sequence]\n')
        file.write('name=' + text_file_name + '\n')
        file.write('imDir=img1\n')
        file.write('frameRate=30\n')
        file.write('seqLength=' + str(int(frame_idx*40) + 1) + '\n')
        file.write('imWidth=' + str(im0.shape[1]) + '\n')
        file.write('imHeight=' + str(im0.shape[0]) + '\n')
        file.write('imExt=.jpg\n')
    
    # /home/rosen/TrackEval/scripts/run_mot_challenge.py
    p = subprocess.Popen(
        args=[
            sys.executable, Path('scripts') / 'run_mot_challenge.py',
            "--GT_FOLDER", str(project),
            "--BENCHMARK", "",
            "--TRACKERS_FOLDER", save_dir,   # project/name
            "--TRACKERS_TO_EVAL", "mot",  # project/name/mot
            "--SPLIT_TO_EVAL", "train",
            "--METRICS", "HOTA", "CLEAR", "Identity",
            "--USE_PARALLEL", "False",
            "--TRACKER_SUB_FOLDER", "",
            "--NUM_PARALLEL_CORES", "4",
            "--SKIP_SPLIT_FOL", "True",
            "--SEQ_INFO", *[str(text_file_name)]
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = p.communicate()

    # Check the return code of the subprocess
    if p.returncode != 0:
        LOGGER.error(stderr)
        LOGGER.error(stdout)
        sys.exit(1)

    LOGGER.info(stdout)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'mot_eval', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--ros-package', type=str, default='0', help='is this running in a ros package')
    parser.add_argument('--ros-bag', type=str, default='0', help='run ros bag')
    parser.add_argument('--op-mode', type=str, default='eval', help='get detection from "yolo" or from "eval" ground truth')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def ros_init(is_ros_package=0):
    ic = image_converter(is_ros_package)
    rospy.init_node('image_converter', anonymous=True)
    return ic
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    # cv2.destroyAllWindows()

def main(opt):
    import warnings

    # Raise a warning as an exception
    warnings.filterwarnings("error", category=RuntimeWarning)
    play_bag(int(opt.ros_bag))
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    ic = ros_init(int(opt.ros_package))
    opt.source = ic
    run(**vars(opt))



# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
