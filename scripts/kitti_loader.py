# Ultralytics YOLO 🚀, GPL-3.0 license
from __future__ import print_function

import glob
import math
import os
import time
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append("../")
from ros_classes import image_converter

import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.yolo.utils.checks import check_requirements
import threading

class LoadStreams:
    # YOLOv8 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1, time_out = 2):
        is_ros = isinstance(sources, image_converter)
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.imgsz = imgsz
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        self.depth_image = None
        self.time_out = time_out
        self.lock = threading.Lock()
        self.lock_extra = threading.Lock()
        if not is_ros:
            sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        else:
            sources = [sources]
        n = len(sources)
        if not is_ros:
            self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        else:
            self.sources = sources
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if not is_ros:
                if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                    # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                    check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                    import pafy  # noqa
                    s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
                s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
                if s == 0:
                    assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                    assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
                # if "rtsp" in s:

                cap = cv2.VideoCapture(s)
                assert cap.isOpened(), f'{st}Failed to open {s}'
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
                self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
                self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

                _, self.imgs[i] = cap.read()  # guarantee first frame
            else:
                # while s.cv_image_queue.qsize() == 0: pass
                # if s.cv_image_queue.qsize() > 0:
                #     while(s.cv_image_queue.qsize() > 0):
                #         self.imgs[i] = s.cv_image_queue.get()
                self.imgs[i] = s.cv_image_queue.get()
                w = int(self.imgs[i].shape[0])
                h = int(self.imgs[i].shape[1])
            if not is_ros:
                self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            else:
                self.threads[i] = Thread(target=self.update, args=([i, None, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([LetterBox(imgsz, auto, stride=stride)(image=x).shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        
        is_ros = isinstance(stream, image_converter)
        if is_ros:
            while True:            
                my_im = stream.cv_image_queue.get()
                my_im = cv2.resize(my_im, (0,0), fx=1., fy=1.)
                im = my_im
                self.imgs[i] = im
                if self.lock_extra.locked():
                    self.lock_extra.release()
                self.lock.acquire()
                #check sim_reset_queue size
                
        else:
            n, f = 0, self.frames[i]  # frame number, frame array
            while cap.isOpened() and n < f:
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    print("In update loop of stream_loader  ", success)
                    if success:
                        self.imgs[i] = im
                    else:
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        self.imgs[i] = np.zeros_like(self.imgs[i])
                        cap.open(stream)  # re-open stream if signal was lost
        time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        #if it's a ros node the output will also include the simulation reset signal
        if isinstance(self.sources[0], image_converter):
            lock_success = self.lock_extra.acquire(timeout=self.time_out)
            if not lock_success:
                raise StopIteration
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        im0 = self.imgs.copy()



        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([LetterBox(self.imgsz, self.auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        #if ros node, return the reset signal as well
        if isinstance(self.sources[0], image_converter):

            #do the same for the depth image queue
            # self.depth_image = None
            # tick = time.time()
            # while self.sources[0].depth_image_queue.qsize() == 0 or self.sources[0].odom_queue.qsize() == 0:
            #     # if no data received for time_out seconds, return None
            #     if time.time() - tick > self.time_out:
            #         raise StopIteration
                
            # if self.sources[0].depth_image_queue.qsize() > 0:
            #     while(self.sources[0].depth_image_queue.qsize() > 0):
            depth_image = self.sources[0].depth_image_queue.get()
            self.depth_image = depth_image

            #do the same for the odom queue
            self.odom = None
            #sleep until the odom queue is not empty
            
            # while self.sources[0].odom_queue.qsize() == 0: pass #FIXME: this is a hack to make sure the odom queue is not empty (for logging)
            # if self.sources[0].odom_queue.qsize() > 0:
            #     while(self.sources[0].odom_queue.qsize() > 0):
            odom = self.sources[0].odom_queue.get()
            self.odom = odom
            self.gt = []
            # if self.sources[0].gt_queue.qsize() > 0:
            #     while(self.sources[0].gt_queue.qsize() > 0):
            gt = self.sources[0].gt_queue.get()
            self.gt = gt
            
            #make a dictionary that has the reset signal and the depth image
            self.extra_output = { "depth_image": self.depth_image, "odom": self.odom, "gt": self.gt}
            self.lock.release()
            return self.sources, im, im0, None,'', self.extra_output
        else:
            return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
