import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

class ResultIllustrator:
    def __init__(self, exp_path):
        self.exp_path = Path(exp_path)
        self.folders = [f.stem for f in self.exp_path.iterdir() if f.is_dir()]
        self.env_type = ['_'.join(self.folders[i].split("_")[:-3]) for i in range(len(self.folders))]
        self.exp_scene = [self.folders[i].split("_")[-3] for i in range(len(self.folders))]
        self.tracker_type = [self.folders[i].split("_")[-2] for i in range(len(self.folders))]
        self.det_source = [self.folders[i].split("_")[-1][:4] for i in range(len(self.folders))]
        # remove any digits in the det_source
        self.env_id = set(self.env_type)
        self.exp_scene_id = set(self.exp_scene)
        print("env_type: ", self.env_id)
        print("exp_scene: ", self.exp_scene_id)
        pass

    def plot_results_per_exp(self, env_type, exp_scene, det_src, metrics):

        print('-'*100)
        print("env_type: ", env_type, ", exp_scene: ", exp_scene)
        print(f"det_source: {det_src}")
        #print metrics with 10 spaces between them and 10 space from the left margin
        # print(f'{" "*19}{str(metrics)}')
        print('-'*100)
        file_path = self.exp_path / self.folders[0] / 'mot' / 'pedestrian_summary.txt'  # Replace with the actual path to your file
        method_results = dict()
        tracker_type_2_metrics = dict()
        for indx, id in enumerate(self.folders):
            if env_type in self.env_type[indx] and det_src == self.det_source[indx]:
                if  exp_scene == self.exp_scene[indx] or exp_scene == 'all':
                    file_path = self.exp_path / id / 'mot' / 'pedestrian_summary.txt'
                else:
                    continue
        # Open the file and read its contents
            with open(file_path, 'r') as file:
                # Read each line in the file
                keys = file.readline().strip().split()
                values = file.readline().strip().split()
                data_dict = dict(zip(keys, values))
                data_dict = {key: float(data_dict[key]) for key in data_dict.keys() if key in metrics}
                method_results[self.tracker_type[indx]] = data_dict
                formatted_values = '   '.join([f"{float(value):0.2f}" for value in data_dict.values()])
                print(f"tracker: {self.tracker_type[indx]:<10}, {formatted_values}")
                #append to  tracker_type_2_metrics for each tracker type with pair of metric values with their names as a tuple and also the name of the experiment 
                #if tracker_type_2_metrics does not have the key self.tracker_type[indx], 
                # then initialize it with a list that has  (data_dict, id) as it's item, otherwise append to the list
                if self.tracker_type[indx] not in tracker_type_2_metrics.keys():
                    tracker_type_2_metrics[self.tracker_type[indx]] = {id.split("_")[2]: data_dict}
                else:
                    tracker_type_2_metrics[self.tracker_type[indx]][id.split("_")[2]] = data_dict
        print('')
        print(f'{" "*19}{list(data_dict.keys())}')

        # print()
        # print()
        # for metric in metrics:
        #     print(f"       {metric:<5}", end='')
        # print()
        print('')

        print('')
        print('')
        print('')

        print(f'{" "*19}{list(metrics)}')

        #for each of the trackers iterate over tracker_type_2_metrics and generate the average of the metrics for each tracker type
        for tracker_type in tracker_type_2_metrics.keys():
            #get the average of the metrics for each tracker type
            avg_metrics = {metric: np.mean([tracker_type_2_metrics[tracker_type][id][metric] for id in tracker_type_2_metrics[tracker_type].keys()]) for metric in metrics}
            formatted_values = '   '.join([f"{float(value):0.2f}" for value in avg_metrics.values()])

            #print the name of the metrics formatted in one line

            print(f"tracker: {tracker_type:<10}, {formatted_values}")
            print('-'*100)




    def plot_results_per_method(results, labels, title, exp_path):
        pass



if __name__ == "__main__":
    exp_path = "/home/rosen/tracking_catkin_ws/src/my_tracker/runs/ablation"
    result_illustrator = ResultIllustrator(exp_path)
    result_illustrator.plot_results_per_exp('day', 'sc4', 'eval', ['MOTA', 'HOTA', 'AssA', 'IDF1', 'AssRe', 'IDSW', "Frag", "IDFP", "IDFN", "MODA"])
    # result_illustrator.plot_results()