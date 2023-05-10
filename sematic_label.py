import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import numpy as np
import re
import glob
from utils_ml3d import parse_config
from tqdm import tqdm
import build.lib.ml3d.datasets as datasets
# Class colors, RGB values as ints for easy reading
COLOR_MAP = {
    0: (0, 0, 0),
    1: (245, 150, 100),
    2: (245, 230, 100),
    3: (150, 60, 30),
    4: (180, 30, 80),
    5: (255, 0., 0),
    6: (30, 30, 255),
    7: (200, 40, 255),
    8: (90, 30, 150),
    9: (255, 0, 255),
    10: (255, 150, 255),
    11: (75, 0, 75),
    12: (75, 0., 175),
    13: (0, 200, 255),
    14: (50, 120, 255),
    15: (0, 175, 0),
    16: (0, 60, 135),
    17: (80, 240, 150),
    18: (150, 240, 255),
    19: (0, 0, 255),
}


# ------ for custom self.data -------
kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}

# s3dis_labels = {
#             0: 'ceiling',
#             1: 'floor',
#             2: 'wall',
#             3: 'beam',
#             4: 'column',
#             5: 'window',
#             6: 'door',
#             7: 'table',
#             8: 'chair',
#             9: 'sofa',
#             10: 'bookcase',
#             11: 'board',
#             12: 'clutter'
#         }


class custom_point_cloud_visualizer:
    def __init__(self,config=None):
        self.config = parse_config("src/config.json")
        self.cfg_file = "ml3d/configs/randlanet_s3dis.yml"
        self.custom_pcd_dataset_path = self.config["visualize"]['custom_pcd_dataset_path']
        # self.cfg_file = "/home/carlos/Open3D/build/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
        self.cfg = _ml3d.utils.Config.load_from_file(self.cfg_file)

        # Load the RandLANet self.model
        self.model = ml3d.models.RandLANet(**self.cfg.model)
        self.cfg.dataset['dataset_path'] = 'input/SemanticKitti'
        self.dataset_path = self.cfg.dataset['dataset_path']
        self.cfg.dataset['custom_dataset_path'] = 'open3d_experiments/pcds'
        self.custom_dataset_path = self.cfg.dataset['custom_dataset_path']
        config = parse_config(self.cfg_file)
        # print("self.dataset_path: ", self.dataset_path)
        # print("self.custom_dataset_path: ", self.custom_dataset_path)
        # exit()
        self.pcd_name = 'Goodman.pcd'
        # Add path to the SemanticKitti self.dataset and your own custom dataset
        # self.cfg.dataset['self.pcd_path'] = '/media/carlos/SeagateExpansionDrive/kitti/SemanticKitti/'

        # self.cfg.dataset['custom_pcd_path'] = './pcds'

        # Load the datasets
        # self.dataset = ml3d.datasets.SemanticKITTI(self.cfg.dataset.pop('self.pcd_path', None), **self.cfg.dataset)
        self.dataset = ml3d.datasets.S3DIS(self.cfg.dataset.pop(
            'dataset_path', None), **self.cfg.dataset)
        # self.custom_dataset = self.load_custom_dataset(
        #     self.cfg.dataset.pop('custom_dataset_path', None))
        self.custom_dataset = self.load_custom_dataset()

        # Create the ML pipeline
        self.pipeline = ml3d.pipelines.SemanticSegmentation(
            self.model, dataset=self.dataset, device="gpu", **self.cfg.pipeline)

        # Download the weights.
        self.ckpt_folder = "./logs/"
        self.ckpt_folder = "open3d_experiments/logs"
        os.makedirs(self.ckpt_folder, exist_ok=True)
        # logs/RandLANet_Semantic3D_tf/checkpoint/ckpt-61.index
        # load the self.model from the checkpoint
        # self.ckpt_path = "ml3d/configs/randlanet_s3dis.yml"
        self.ckpt_path = self.ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
        self.randlanet_url = "https://storage.googleapis.com/open3d-releases/self.model-zoo/randlanet_semantickitti_202201071330utc.pth"
        # Load an ML configuration file
        self.cfg_file = "ml3d/configs/randlanet_semantickitti.yml"
        # self.pcd_path = ""
        pass


    def pts_to_pcd_npy(self):
    # load txt file
        self.inputFile = open("input/Dr_Mark/Goodman.pts", "r")  # Input-file
        self.outputFile = "open3d_experiments/pcds/Goodman.pcd"

        # create a list to store points and colors
        points = []
        colors = []
        self.length = int(self.inputFile.readline())  # First line is the length
        with tqdm(total=self.length, desc='Reading point cloud', unit='points') as input_bar:
            for line in self.inputFile:
                currentLine = line.rstrip().split(" ")
                x = float(currentLine[0])
                y = float(currentLine[1])
                z = float(currentLine[2])
                r = int(float(currentLine[4]))
                g = int(float(currentLine[5]))
                b = int(float(currentLine[6]))
                rgb = (r << 16) + (g << 8) + b
                points.append([x, y, z])
                colors.append([r, g, b])
                input_bar.update(1)

        self.inputFile.close()

        # convert the lists of points and colors to numpy arrays
        points = np.asarray(points)
        colors = np.asarray(colors)

        # create an Open3D point cloud object from the numpy arrays
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # write the point cloud object to a file
        o3d.io.write_point_cloud(self.outputFile, pcd)

        print("All done")
        # print output file directory
        print("Output file: " + self.outputFile)

        # Load saved point cloud and visualize it
        pcd_load = o3d.io.read_point_cloud(self.outputFile, 
                                           format='pcd',
                                           remove_nan_points=True, 
                                           remove_infinite_points=True, print_progress=True)
        o3d.visualization.draw_geometries([pcd_load])

        # convert Open3D.o3d.geometry.PointCloud to numpy array
        xyz_load = np.asarray(pcd_load.points)
        print('xyz_load')
        print(xyz_load)
        # output xyz_load to npy file
        npy_file = 'open3d_experiments/pcds/Goodman.npy'
        np.save(npy_file, xyz_load)
        print('npy_file directory')
        print(npy_file)
    
        return npy_file
    def pts_to_pcd(self):
        # # load txt file

        # inputFile = open("open3d_experiments/pcds/Xinyafull-rv.pts", "r") # Input-file
        self.inputFile = open("input/Dr_Mark/Goodman.pts", "r")  # Input-file
        # outputFile = open("open3d_experiments/pcds/Xinyafull.pcd", "w") # Output-file
        self.outputFile = open("open3d_experiments/pcds/Goodman.pcd", "w")
        length = int(self.inputFile.readline())  # First line is the length

        self.outputFile.write("VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nself.data ascii\n".format(
            length, length))  # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php

        currentLinePosition = 0

        for line in self.inputFile:
            currentLinePosition = currentLinePosition + 1

            if (currentLinePosition % 100000 == 0):
                print("Current file position: " + str(currentLinePosition))

            currentLine = line.rstrip().split(" ")

            self.outputFile.write(" ".join([
                currentLine[0],  # x-value
                currentLine[1],  # y-value
                currentLine[2],  # z-value
                "{:e}".format((int(float(currentLine[4])) << 16) +
                              (int(float(currentLine[5])) << 8) +
                              int(float(currentLine[6])))]) + "\n")  # rgb value renderd in scientific format

        self.inputFile.close()
        self.outputFile.close()
        
        print("All done")
        # Load saved point cloud and visualize it
        pcd_load = o3d.io.read_point_cloud(self.outputFile)
        o3d.visualization.draw_geometries([pcd_load])

        # convert Open3D.o3d.geometry.PointCloud to numpy array
        xyz_load = np.asarray(pcd_load.points)
        print('xyz_load')
        print(xyz_load)
    def custom_draw_geometry(self, pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        # save the pcd
        # o3d.io.write_point_cloud("open3d_experiments/pcds/Goodman_origin.pcd", pcd)

    def get_label_to_names(self):

        data_labels = dict()
        if self.dataset == "s3dis":
            data_labels = datasets.s3dis.get_label_to_names()
            return data_labels
        elif self.dataset == "scannet":
            data_labels = datasets.scannet.get_label_to_names()
            return data_labels
        elif self.dataset == "semantic-kitti":
            data_labels = datasets.semantickitti.get_label_to_names()
            return kitti_labels
        elif self.dataset == "custom":
            data_labels = datasets.semantic3d.get_label_to_names()
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))

    def load_custom_dataset(self):
        print("Loading custom dataset")
        # load self.cfg.dataset['custom_dataset_path'] to print
        print("PCD files directory: ", self.custom_dataset_path)
        # load all pcd files
        # if os.path.exists("open3d_experiments/pcds/Xinyafull.pcd"):
        # if os.path.exists(self.cfg.dataset['custom_dataset_path'] + self.pcd_path):
        if os.path.exists(self.custom_dataset_path + "/"+ self.pcd_name):  
            # pcd_paths = glob.glob(self.pcd_path+"/Goodma*.pcd")
            # take path from custom dataset
            self.pcd_paths = glob.glob(self.custom_dataset_path + "/"+ self.pcd_name)
            # pcd_paths = glob.glob(self.pcd_path+"/Xinya*.pcd")
        else:
            # pcd_paths = glob.glob(self.pcd_path+"/*.pcd")
            # take path from stabdard dataset
            # self.pcd_paths = self.dataset_path +"/*/"+ self.pcd_path)
            self.pcd_paths = glob.glob(self.custom_dataset_path+"/*.pcd")
            print("PCD files directory: ", self.pcd_paths)
        # pcd_paths = glob.glob(self.pcd_path+"/*.pcd")
        self.pcds = []
        for pcd_path in self.pcd_paths:
            self.pcds.append(o3d.io.read_point_cloud(pcd_path))
            # print(pcd_path)
        print("Loaded {} point clouds".format(len(self.pcds)))

    def prepare_point_cloud_for_inference(self):
        # Remove NaNs and infinity values
        self.pcd.remove_non_finite_points()
        print("Loading point cloud from {}".format(self.pcd_path))
        # Extract the xyz points
        xyz = np.asarray(self.pcd.points)
        # Set the points to the correct format for inference
        self.data = {"point": xyz, 'feat': None,
                     'label': np.zeros((len(xyz),), dtype=np.int32)}

    def load_point_cloud_for_inference(self):
        # self.pcd_path = self.pcd_path + "/" + self.file_path
        # Load the file
        self.pcd_path = self.custom_dataset_path + "/"+ self.pcd_name
        print("Loading point cloud from {}".format(self.pcd_path))

        if os.path.exists(self.pcd_path):
            self.pcd = o3d.io.read_point_cloud(self.pcd_path)
        print(self.pcd)
        # Remove NaNs and infinity values
        self.pcd.remove_non_finite_points()
        # Extract the xyz points
        xyz = np.asarray(self.pcd.points)
        # Set the points to the correct format for inference
        self.data = {"point": xyz, 'feat': None,
                     'label': np.zeros((len(xyz),), dtype=np.int32)}
        print("Loaded point cloud with {} points".format(len(xyz)))


    def main(self):
        # load all above functions
        self.load_custom_dataset()
        self.load_point_cloud_for_inference()
        # self.prepare_point_cloud_for_inference(self.pcd)

        for label in COLOR_MAP:
            COLOR_MAP[label] = tuple(val/255 for val in COLOR_MAP[label])

        if not os.path.exists(self.ckpt_path):
            cmd = "wget {} -O {}".format(self.randlanet_url, self.ckpt_path)
            os.system(cmd)
        # Example string
        num_points_clouds = str(self.pcds[:])
        print(num_points_clouds)
        # Define regex pattern
        pattern = r'with (\d+) points'
        # Find matches in the string using the regex pattern
        match = re.search(pattern, num_points_clouds)
        # Extract the integer value from the match object
        num_points = int(match.group(1))
        # print(num_points)  # Output: 33892041
        # Get one test point cloud from the custom dataset
        pc_idx = 0  # change the index to get a different point cloud
        # self.data, self.pcd = self.prepare_point_cloud_for_inference(
        #     self.pcds[pc_idx])
        # print(self.custom_dataset)
        print (self.pcds[pc_idx])
        self.pcd = self.pcds[pc_idx]
        print(self.pcds[pc_idx])
        exit()
        # self.data, self.pcd = self.prepare_point_cloud_for_inference(
        # self.pcds[pc_idx])
        self.data, self.pcd = self.prepare_point_cloud_for_inference()
        exit()
        # Run inference
        result = self.pipeline.run_inference(self.data)
        # Colorize the point cloud with predicted labels
        colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])]
        # show the labels and classes
        print("labels: ", result['predict_labels'])
        # print("classes: ", [get_label_to_names[clr] for clr in list(result['predict_labels'])])
        print("classes: ", [kitti_labels[clr]
              for clr in list(result['predict_labels'])])

        # classes_path = "open3d_experiments/pcds/classes.txt"
        #  save classes to a file
        # with open( classes_path, 'w') as f:
        #     for s in kitti_labels:
        #         f.write(str(s) + '\n')

        # print("classes: ", kitti_labels)

        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create visualization for original model
        vis_original_pcd = self.custom_draw_geometry(self.pcd)
        # save the point cloud visualization
        o3d.io.write_point_cloud(
            "open3d_experiments/pcds/Goodman.pcd", self.pcd)

        # show legend in the visualization
        # custom_draw_geometry_with_legend(pcd)
        # evaluate performance on the test set; this will write logs to './logs'.
        # pipeline.run_test()


# pts_to_pcd(None)
# count line of the file
# with open("open3d_experiments/pcds/Xinyafull.pts") as f:
#     for i, l in enumerate(f):
#         pass
# print("Number of points: ", i+1)
# exit()
# Convert class colors to doubles from 0 to 1, as expected by the visualizer

    # # Load the parameters of the self.model.
    # pipeline.load_ckpt(self.ckpt_path=self.ckpt_path)

    # # Get one test point cloud from the SemanticKitti dataset
    # pc_idx = 256 # change the index to eget a different point cloud
    # test_split = dataset.get_split("test")
    # self.data = test_split.get_data(pc_idx)

    # # run inference on a single example.
    # # returns dict with 'predict_labels' and 'predict_scores'.
    # result = pipeline.run_inference(data)

    # Create a pcd to be visualized
    # pcd = o3d.geometry.PointCloud()
    # xyz = data["point"] # Get the points
    # pcd.points = o3d.utility.Vector3dVector(xyz)

    # colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])] # Get the color associated to each predicted label
    # pcd.colors = o3d.utility.Vector3dVector(colors) # Add color self.data to the point cloud

    # # Create visualization
    # custom_draw_geometry(pcd)

    # tqdm for preapre_point_cloud_for_inference
    # Iterate over the custom self.dataset and show progress using tqdm


if __name__ == "__main__":
    point_cloud_visualizer = custom_point_cloud_visualizer()
    # point_cloud_visualizer.main()
    point_cloud_visualizer.pts_to_pcd_npy()
