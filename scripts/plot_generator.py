def generate_plots(res_folder_name):
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
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

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
    df_tracker_list = []
    df_used_emap = []
    df_seq_list = []
    df_hota_list = []
    df_mota_list = []
    df_idf1_list = []
    df_idsw_list = []
    for key in eval_dict:
        eval_dict[key] = dict(sorted(eval_dict[key].items()))
        # convert eval_dict to sns data format
        for _seq, _results in eval_dict[key].items():
            _tracker_name = f"{key[0]}_{key[1]}"
            df_tracker_list.append(key[0])
            _emap = "w/ EMAP" if key[1]=="emap" else "w/o EMAP"
            df_used_emap.append(_emap)
            df_seq_list.append(_seq)
            df_hota_list.append(_results[metrics_to_select.index("HOTA")])
            df_idsw_list.append(_results[metrics_to_select.index("IDSW")])
            df_mota_list.append(_results[metrics_to_select.index("MOTA")])
            df_idf1_list.append(_results[metrics_to_select.index("IDF1")])

    df = dict(tracker=df_tracker_list, emap=df_used_emap, sequence=df_seq_list, hota=df_hota_list,
              idsw=df_idsw_list, idf1=df_idf1_list, mota=df_mota_list)
    hrdf = pd.DataFrame(data=df)

    custom_pallete = {"w/ EMAP": "#66c2a5", "w/o EMAP": "#fc8d62"}
    sns.set_style("whitegrid")
    order_trk = ["ocsort", "deepocsort", "bytetrack", "botsort"]
    
    
    #----------------------------START of bar/box plot for idsw and hota--------------------------------
    # labels = ['OC-SORT', 'Deep OC-SORT', 'ByteTrack', 'BoT-SORT']

    
    # plt.figure()
    # plt.grid(True, linestyle='--', zorder=0)
    # order_trk = ["ocsort", "deepocsort", "bytetrack", "botsort"]
    # idsw_plt = sns.barplot(data=hrdf, x="tracker", y="idsw", hue="emap", order=order_trk,
    #                   palette=custom_pallete, edgecolor=".3", capsize=0.1, err_kws={"linewidth": 1})#,whis=(0, 100))
    # # bar = sns.violinplot(data=hrdf, x="tracker", y="idsw", hue="emap", order=sorted(set(df["tracker"])), cut=0, inner="quart")
    # idsw_plt.set_xlabel("Tracker", fontsize=12)
    # idsw_plt.set_ylabel("IDSW", fontsize=12)
    # idsw_plt.legend(title='', fontsize=11)
    # # change the name on the x axis to new names
    # idsw_plt.set_xticklabels(labels, fontsize=11)
    # plt.tight_layout() 
    # plt.savefig("test_results/idsw.svg", dpi=300, format="svg")

    # plt.figure()
    # sns.set_style("whitegrid")
    # plt.grid(True, linestyle='--', zorder=0)
    # hota_plt = sns.boxplot(data=hrdf, x="tracker", y="hota", hue="emap", order=order_trk, 
    #                        palette=custom_pallete,whis=(0, 100), showmeans=True,
    #                        meanprops={'marker':'o',
    #                                 'markerfacecolor':'white', 
    #                                 'markeredgecolor':'#636363',
    #                                 'markersize':'6'})
    # # hota_plt = sns.violinplot(data=hrdf, x="tracker", y="hota", hue="emap", order=sorted(set(df["tracker"])),
    # #                          palette=custom_pallete, cut=1, native_scale=True)
    # # inner="quart", cut=0,
    # hota_plt.set_xlabel("Tracker", fontsize=12)
    # hota_plt.set_ylabel("HOTA", fontsize=12)
    # hota_plt.legend(title='', fontsize=11)
    # # plt.xticks(rotation=45)
    # hota_plt.set_xticklabels(labels, fontsize=11)
    # plt.tight_layout()
    # plt.savefig("test_results/hota.svg", dpi=300, format="svg")
    
    
    #----------------------------END of bar/box plot for idsw and hota--------------------------------
    

    # dis_plot = sns.displot(data=hrdf, y="hota", x="idsw", hue="emap", col="tracker", kind="kde", thresh=0.12, levels=2, fill=True,  alpha=.6, cut=1, palette=custom_pallete)

    
    #----------------------------START of distribution plot--------------------------------
    # _mean_hota = []
    # _std_hota = []
    # _mean_idsw = []
    # _std_idsw = []
    # _tracker = []
    # _emap = []
    # order_trk = ["ocsort", "deepocsort", "bytetrack", "botsort"]
    # for tracker in set(df["tracker"]):
    #     for emap in set(df["emap"]):
    #         _df = hrdf[(hrdf["tracker"]==tracker) & (hrdf["emap"]==emap)]
    #         _emap.append(emap)
    #         _tracker.append(tracker)
    #         _mean_hota.append(_df["hota"].mean())
    #         _mean_idsw.append(_df["idsw"].mean())
    #         # calc std
    #         _std_hota.append(_df["hota"].std())
    #         _std_idsw.append(_df["idsw"].std())
    # _df = dict(tracker=_tracker, emap=_emap, mean_hota=_mean_hota, mean_idsw=_mean_idsw, std_hota=_std_hota, std_idsw=_std_idsw)
    # stat_hrdf = pd.DataFrame(data=_df)
    # sns.set_style("whitegrid")
    # # custom_pallete_trk = {"ocsort": "#7570b3", "deepocsort": "#66a61e", "bytetrack": "#e6ab02", "botsort": "#e7298a"}
    # color_indes = [0,1,2,3]
    # custom_color = [sns.color_palette("Dark2")[ind] for ind in color_indes]
    # scatterplt = sns.scatterplot(data=stat_hrdf, x="mean_idsw", y="mean_hota", style="emap", hue="tracker", palette=custom_color, zorder=2, hue_order=order_trk, 
    #                              style_order=['w/ EMAP', 'w/o EMAP'], s=80)
    # # change the legend values to new values
    # handles, _ = scatterplt.get_legend_handles_labels()
    # labels = ['OC-SORT', 'Deep OC-SORT', 'ByteTrack','BoT-SORT','----------------', 'w/ EMAP', 'w/o EMAP']
    # scatterplt.legend(handles=handles[1:], labels=labels, fontsize=11, loc='lower right', bbox_to_anchor=(1.012, -0.015))
    # scatterplt.tick_params(axis='both', which='major', labelsize=11)
    # # move legend to tight bottom right
    # # scatterplt.set_xlabel(r"IDSW$\downarrow$", fontsize=12)
    # # scatterplt.set_ylabel(r"HOTA$\uparrow$", fontsize=12)
    # scatterplt.set_xlabel("IDSW", fontsize=12)
    # scatterplt.set_ylabel("HOTA", fontsize=12)
    # # add an arrow under the xlabel
    # # scatterplt.annotate("", xy=(0.2, -0.07), xytext=(0.8,-0.07), xycoords='axes fraction', textcoords='axes fraction',
    # #                     arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
    # # scatterplt.annotate("", xy=(-0.06, 0.8), xytext=(-0.06, 0.2), xycoords='axes fraction', textcoords='axes fraction',
    # #                     arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
    
    # # scatterplt.annotate("", xy=(0.45, -0.14), xytext=(0.55, -0.14), xycoords='axes fraction', textcoords='axes fraction',
    # #                     arrowprops=dict(arrowstyle="->", color='black', lw=1))
    # # scatterplt.annotate("", xy=(-0.11, 0.56), xytext=(-0.11, 0.44), xycoords='axes fraction', textcoords='axes fraction',
    # #                     arrowprops=dict(arrowstyle="->", color='black', lw=1))
    # # plot oval for each tracker with radius of 1 std of hota and idsw
    # color_set = [["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"][ind] for ind in color_indes]  #Set2
    # color_dark = [["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"][ind] for ind in color_indes]#Dark2
    # color_pastel = [["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9", "#fff2ae"][ind] for ind in color_indes] #Pastel2

    # color_dict_set = dict(zip(order_trk, color_set))
    # color_dict_dark = dict(zip(order_trk, color_dark))
    # color_dict_pastel = dict(zip(order_trk, color_pastel))
    # color_dict_edges = {"w/ EMAP": color_dict_dark, "w/o EMAP": color_dict_set}
    # color_dict_faces = {"w/ EMAP": color_dict_set, "w/o EMAP": color_dict_pastel}
    
    # for tracker in set(df["tracker"]):
    #     for emap in set(df["emap"]):
    #         _df = stat_hrdf[(stat_hrdf["tracker"]==tracker) & (stat_hrdf["emap"]==emap)]
    #         _mean_hota = _df["mean_hota"].values[0]
    #         _mean_idsw = _df["mean_idsw"].values[0]
    #         _std_hota = _df["std_hota"].values[0]
    #         _std_idsw = _df["std_idsw"].values[0]
    #         # plot oval
    #         ellipse = Ellipse((_mean_idsw, _mean_hota), width=_std_idsw, height=_std_hota, edgecolor=color_dict_edges[emap][tracker], facecolor=color_dict_faces[emap][tracker], alpha=0.5)
    #         plt.gca().add_patch(ellipse)

    # plt.xlim(0, 180)
    # plt.ylim(45, 80)
    # # plt.xlabel("IDSW")
    # # plt.ylabel("HOTA")
    # # save the plot to a file in high resolution in svg format
    # plt.tight_layout()
    # plt.savefig("test_results/hota_idsw_dist.svg", dpi=300, format="svg")
    # ----------------------------END of distribution plot--------------------------------
    
    
    # ----------------------------START of box plot for mota and idf1--------------------------------
    plt.figure()
    sns.set_style("whitegrid")
    plt.grid(True, linestyle='--', zorder=0)
    mota_plt = sns.boxplot(data=hrdf, x="tracker", y="mota", hue="emap" ,order=order_trk,
                           palette=custom_pallete,whis=(0, 100), showmeans=True,
                           meanprops={'marker':'o',
                                    'markerfacecolor':'white', 
                                    'markeredgecolor':'#636363',
                                    'markersize':'6'})
    mota_plt.set_xlabel("Tracker")
    mota_plt.set_ylabel("MOTA")
    mota_plt.legend(title='')
    plt.tight_layout()
    labels = ['OC-SORT', 'Deep OC-SORT', 'ByteTrack', 'BoT-SORT',]
    mota_plt.set_xticklabels(labels)


    plt.figure()
    sns.set_style("whitegrid")
    plt.grid(True, linestyle='--', zorder=0)
    idf1_plt = sns.boxplot(data=hrdf, x="tracker", y="idf1", hue="emap" ,order=order_trk,
                           palette=custom_pallete,whis=(0, 100), showmeans=True,
                           meanprops={'marker':'o',
                                    'markerfacecolor':'white', 
                                    'markeredgecolor':'#636363',
                                    'markersize':'6'})
    idf1_plt.set_xlabel("Tracker")
    idf1_plt.set_ylabel("IDF1")
    idf1_plt.legend(title='')
    idf1_plt.set_xticklabels(labels)
    plt.tight_layout()
    # ----------------------------END of box plot for mota and idf1--------------------------------
    
    plt.show()

if __name__ == "__main__":

    generate_plots("runs/mot_perma")