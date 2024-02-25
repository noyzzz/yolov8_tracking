def generate_res_tables(res_folder_name):
    """
    for each folder in res_folder_name, the strucure is as follows:
    name of the folder is: "seq_number"_"tracker_type"_"depth_use"
    seq_number is the number of the sequence and is in four digits
    tracker_type is the type of the tracker and is one of the following: "ocsort", "deepocsort", "emap", "bytetrack", "botsort"
    depth_use is a boolean and is either "no_depth" or "emap"
    I want to generate a table for each tracker type and for each depth use and also an average table for each tracker type on all sequences
    """
    import os
    import pandas as pd
    import numpy as np
    import re
    #get the list of folders in res_folder_name
    folder_list = os.listdir(res_folder_name)
    tracker_type_list = ["ocsort", "deepocsort", "emap", "bytetrack", "botsort"]
    depth_use_list = ["no_depth", "emap"]
    #create a dictionary with a pair as key which is (tracker_type, depth_use) and the value is a list of the evaluation results
    eval_dict = {}
    for tracker_type in tracker_type_list:
        for depth_use in depth_use_list:
            if tracker_type == "emap":
                eval_dict[("bytetrack", depth_use)] = {}
            else:
                eval_dict[(tracker_type, depth_use)] = {}
    #for each folder in folder_list
    for folder in folder_list:
        #get the sequence number, tracker type and depth use
        try:
            seq_num, tracker_type, depth_use = re.split("_", folder)
        except:
            seq_num, tracker_type, depth_use, depth_use_2 = re.split("_", folder)
            depth_use = f"{depth_use}_{depth_use_2}"
        seq_num = int(seq_num)
        if tracker_type == "emap":
            assert depth_use == "emap"
            tracker_type = "bytetrack"
            #open the file: res_folder_name/folder/mot/pedestrian_summary.txt, it contains two rows, the first row is the header and the second row is the evaluation results
        with open(f"{res_folder_name}/{folder}/mot/pedestrian_summary.txt", "r") as f:
            lines = f.readlines()
            eval_res = lines[1]
            eval_res = eval_res.split(" ")
            eval_res = [float(x) for x in eval_res]
            metrics = lines[0]
            #split the metrics into a list
            metrics = metrics.split(" ")
            #get HOTA, MOTA, IDF1, IDSW, MT, ML
            #get index of each metric
            metrics_to_select = ["HOTA", "MOTA", "IDF1","IDFP", "IDFN", "IDSW","Frag","AssA", "AssRe"]
            selected_metric_values = []
            for metric in metrics_to_select:
                index = metrics.index(metric)
                selected_metric_values.append(eval_res[index])
            eval_dict[(tracker_type, depth_use)][seq_num] = selected_metric_values
    #in eval_dict sort the values of each key by the sequence number
    for key in eval_dict:
        eval_dict[key] = dict(sorted(eval_dict[key].items()))
    average_no_depth = {}
    for tracker_type in tracker_type_list:
        if tracker_type == "emap":
            continue
        average_no_depth[tracker_type] = np.zeros(len(selected_metric_values))
        for seq_num in (eval_dict[(tracker_type, "no_depth")].keys()):
            # if seq_num not in seq_nums_to_select:
            #     continue
            average_no_depth[tracker_type] += eval_dict[(tracker_type, "no_depth")][seq_num]
    #for each key in average_no_depth, divide the value by the number of sequences
    for key in average_no_depth:
        average_no_depth[key] = average_no_depth[key]/len(eval_dict[(key, "no_depth")])
    #do the same for the depth_use
    average_depth_use = {}
    for tracker_type in tracker_type_list:
        if tracker_type == "emap":
            continue
        average_depth_use[tracker_type] = np.zeros(len(selected_metric_values))
        for seq_num in ((eval_dict[(tracker_type, "emap")].keys())):
            average_depth_use[tracker_type] += eval_dict[(tracker_type, "emap")][seq_num]
    #for each key in average_no_depth, divide the value by the number of sequences
    for key in average_depth_use:
        average_depth_use[key] = average_depth_use[key]/len(eval_dict[(key, "emap")])

            
    #print the results for average_no_depth and average_depth_use in a table, each value should be rounded to 2 decimal places
    print("Average results for no depth")
    # print metrics_to_select from the list
    #print tracker_type in 7 characters, use formatting
    print(f"{'tracker':^15}", end=", ")
    for metric in metrics_to_select:
        # print metric in 7 characters, use formatting
        print(f"{metric:^7}", end=", ")
    print()
    for key in average_no_depth:
        print(f"{key:^15}", end=", ")
        for value in average_no_depth[key]:
            val = round(value, 2)
            #print value in 7 characters, use formatting
            print(f"{(val):^7}", end=", ")
        print()
    print("Average results for depth use")
    #do the same for average_depth_use
    # print metrics_to_select from the list
    #print tracker_type in 7 characters, use formatting
    print(f"{'tracker':^15}", end=", ")
    for metric in metrics_to_select:
        # print metric in 7 characters, use formatting
        print(f"{metric:^7}", end=", ")
    print()
    for key in average_depth_use:
        print(f"{key:^15}", end=", ")
        for value in average_depth_use[key]:
            val = round(value, 2)
            #print value in 7 characters, use formatting
            print(f"{(val):^7}", end=", ")
        print()
        

    print("KIR")

        

        #get the evaluation



if __name__ == "__main__":
    generate_res_tables("runs/mot_eval_yolo/mot_eval")