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
sys.path.append("../../../")
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


class LoadScreenshots:
    # YOLOv8 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, imgsz=640, stride=32, auto=True, transforms=None):
        # source = [screen_number left top width height] (pixels)
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640, stride=32, auto=True, transforms=None):
        if not isinstance(im0, list):
            im0 = [im0]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.mode = 'image'
        # generate fake paths
        self.paths = [f"image{i}.jpg" for i in range(len(self.im0))]

    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def _single_preprocess(self, im, auto):
        if self.transforms:
            im = self.transforms(im)  # transforms
        else:
            im = LetterBox(self.imgsz, auto=auto, stride=self.stride)(image=im)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        auto = all(x.shape == self.im0[0].shape for x in self.im0) and self.auto
        im = [self._single_preprocess(im, auto) for im in self.im0]
        im = np.stack(im, 0) if len(im) > 1 else im[0][None]
        self.count += 1
        return self.paths, im, self.im0, None, ''

    def __iter__(self):
        self.count = 0
        return self


if __name__ == "__main__":
    img = cv2.imread(str(ROOT / "assets/bus.jpg"))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
