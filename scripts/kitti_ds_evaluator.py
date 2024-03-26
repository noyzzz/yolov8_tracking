def run_track_script():
    import sys
    sys.path.append("./")
    from track import parse_opt, main
    opt = parse_opt()
    for i in range(0, 21):
        for tracker_type in ['ocsort', 'deepocsort', 'emap', "bytetrack", 'botsort' ]:
            for j in range(0, 2):
                if j == 0:
                    use_depth = False
                    use_odometry = False
                else:
                    use_depth = True
                    use_odometry = True
                if tracker_type == 'emap' and use_depth == False:
                    continue
                if tracker_type == 'bytetrack' and use_depth == True:
                    continue
                opt.use_depth = use_depth
                opt.use_odometry = use_odometry
                opt.tracking_method = tracker_type
                kitti_seq = f"{i:04d}"
                opt.kitti_seq = kitti_seq
                opt.tracking_config = f"trackers/{tracker_type}/configs/{tracker_type}.yaml"
                #print configuration
                print(f"KITTI sequence {kitti_seq} with tracker {tracker_type} using depth {use_depth} and odometry {use_odometry}")
                eval_res = main(opt)
                if eval_res is None:
                    print(f"KITTI sequence {kitti_seq} evaluation results: None")
                    continue
                eval_str_list = eval_res.split("\n")
                lines_to_select = [63,64, 67, 68, 71, 72]
                for line in lines_to_select:
                    print(eval_str_list[line])
                # print(f"KITTI sequence {kitti_seq} evaluation results: {eval_res}")

if __name__ == "__main__":
    run_track_script()