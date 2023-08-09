#!/usr/bin/env python3.8
import sys

sys.path.append("/home/rosen/tracking_catkin_ws/src/my_tracker")
from track import parse_opt, main

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)