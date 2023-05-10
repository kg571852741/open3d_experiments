import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import numpy as np

import glob
from utils_ml3d import parse_config
from tqdm import tqdm
import build.lib.ml3d.datasets as datasets

def pts_to_pcd(pts_path):
    # # load txt file 

    # inputFile = open("open3d_experiments/pcds/Xinyafull-rv.pts", "r") # Input-file
    inputFile = open("input/Dr_Mark/Goodman.pts", "r") # Input-file
    # outputFile = open("open3d_experiments/pcds/Xinyafull.pcd", "w") # Output-file
    outputFile = open("open3d_experiments/pcds/Goodman.pcd", "w")
    length = int(inputFile.readline()) # First line is the length

    outputFile.write("VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(length, length)) # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php

    currentLinePosition = 0

    for line in inputFile:
        currentLinePosition = currentLinePosition + 1
        
        if(currentLinePosition % 100000 == 0):
            print ("Current file position: " + str(currentLinePosition))
        
        currentLine = line.rstrip().split(" ")
        
        outputFile.write(" ".join([
            currentLine[0], # x-value
            currentLine[1], # y-value
            currentLine[2], # z-value
            "{:e}".format((int(float(currentLine[4]))<<16) + (int(float(currentLine[5]))<<8) + int(float(currentLine[6])))

 # rgb value renderd in scientific format
        ]) + "\n")
            
    inputFile.close()
    outputFile.close()

    print ("All done")
    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("../../TestData/sync.ply")
    o3d.visualization.draw_geometries([pcd_load])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)
    exit()
def custom_draw_geometry(pcd):
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.get_render_option().point_size = 2.0
	vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()

def load_custom_dataset(dataset_path):
    print("Loading custom dataset")
    print("PCD files directory: ", dataset_path)
    # load all pcd files    
    # if os.path.exists("open3d_experiments/pcds/Xinyafull.pcd"):
    if os.path.exists("open3d_experiments/pcds/Goodman.pcd"):
        pcd_paths = glob.glob(dataset_path+"/Goodma*.pcd")
        # self.pcd_paths = dataset_path+"/Goodman.pcd"
        # pcd_paths = glob.glob(dataset_path+"/Xinya*.pcd")
    else:
        pcd_paths = glob.glob(dataset_path+"/*.pcd")
        # self.pcd_paths = dataset_path+"/*.pcd"
    # pcd_paths = glob.glob(dataset_path+"/*.pcd")
    
    pcds = []
    for pcd_path in pcd_paths:
        pcds.append(o3d.io.read_point_cloud(pcd_path))
    return pcds


def prepare_point_cloud_for_inference(pcd):
	# Remove NaNs and infinity values
	pcd.remove_non_finite_points()
	# Extract the xyz points
	xyz = np.asarray(pcd.points)
	# Set the points to the correct format for inference
	data = {"point":xyz, 'feat': None, 'label':np.zeros((len(xyz),), dtype=np.int32)}

	return data, pcd

def load_point_cloud_for_inference(file_path, dataset_path):
	pcd_path = dataset_path + "/" + file_path
	# Load the file
	pcd = o3d.io.read_point_cloud(pcd_path)
	# Remove NaNs and infinity values
	pcd.remove_non_finite_points()
	# Extract the xyz points
	xyz = np.asarray(pcd.points)
	# Set the points to the correct format for inference
	data = {"point":xyz, 'feat': None, 'label':np.zeros((len(xyz),), dtype=np.int32)}

	return data, pcd

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

def get_label_to_names(dataset):
    
    data_labels = dict()
    if dataset == "s3dis":
        data_labels = datasets.s3dis.get_label_to_names()
        return data_labels
    elif dataset == "scannet":
        data_labels = datasets.scannet.get_label_to_names()
        return data_labels
    elif dataset == "semantic-kitti":
        data_labels = datasets.semantickitti.get_label_to_names()
        return kitti_labels
    elif dataset == "custom":
        data_labels = datasets.semantic3d.get_label_to_names()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
# ------ for custom data -------
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
# pts_to_pcd(None)
# count line of the file
# with open("open3d_experiments/pcds/Xinyafull.pts") as f:
#     for i, l in enumerate(f):
#         pass
# print("Number of points: ", i+1)    
# exit()
# Convert class colors to doubles from 0 to 1, as expected by the visualizer
for label in COLOR_MAP:
	COLOR_MAP[label] = tuple(val/255 for val in COLOR_MAP[label])

# Load an ML configuration file

cfg_file = "ml3d/configs/randlanet_semantickitti.yml"
# cfg_file = "ml3d/configs/randlanet_s3dis.yml"

# cfg_file = "/home/carlos/Open3D/build/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# Load the RandLANet model
model = ml3d.models.RandLANet(**cfg.model)

# Add path to the SemanticKitti dataset and your own custom dataset
# cfg.dataset['dataset_path'] = '/media/carlos/SeagateExpansionDrive/kitti/SemanticKitti/'
cfg.dataset['dataset_path'] = 'input/SemanticKitti'
cfg.dataset['custom_dataset_path'] = 'open3d_experiments/pcds'
# cfg.dataset['custom_dataset_path'] = './pcds'

# Load the datasets
# dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
custom_dataset = load_custom_dataset(cfg.dataset.pop('custom_dataset_path', None))

# Create the ML pipeline
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# Download the weights.
ckpt_folder = "./logs/"
ckpt_folder = "open3d_experiments/logs"
os.makedirs(ckpt_folder, exist_ok=True)
# logs/RandLANet_Semantic3D_tf/checkpoint/ckpt-61.index 
# load the model from the checkpoint
# ckpt_path = "ml3d/configs/randlanet_s3dis.yml"
ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
    os.system(cmd)

# # Load the parameters of the model.
# pipeline.load_ckpt(ckpt_path=ckpt_path)

# # Get one test point cloud from the SemanticKitti dataset
# pc_idx = 256 # change the index to eget a different point cloud
# test_split = dataset.get_split("test")
# data = test_split.get_data(pc_idx)

# # run inference on a single example.
# # returns dict with 'predict_labels' and 'predict_scores'.
# result = pipeline.run_inference(data)

# Create a pcd to be visualized 
# pcd = o3d.geometry.PointCloud()
# xyz = data["point"] # Get the points
# pcd.points = o3d.utility.Vector3dVector(xyz)

# colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])] # Get the color associated to each predicted label
# pcd.colors = o3d.utility.Vector3dVector(colors) # Add color data to the point cloud

# # Create visualization
# custom_draw_geometry(pcd)

# tqdm for preapre_point_cloud_for_inference
# Iterate over the custom dataset and show progress using tqdm

import re

# Example string
num_points_clouds= str(custom_dataset[:])

# Define regex pattern
pattern = r'with (\d+) points'

# Find matches in the string using the regex pattern
match = re.search(pattern, num_points_clouds)

# Extract the integer value from the match object
num_points = int(match.group(1))
#  int to iterate
print(num_points)  # Output: 33892041
# exit()

# Get one test point cloud from the custom dataset
pc_idx = 0 # change the index to get a different point cloud
data, pcd = prepare_point_cloud_for_inference(custom_dataset[pc_idx])


# Run inference

result = pipeline.run_inference(data)

# Colorize the point cloud with predicted labels
colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])]
# show the labels and classes
print("labels: ", result['predict_labels'])
# print("classes: ", [get_label_to_names[clr] for clr in list(result['predict_labels'])])
# 
print("classes: ", [kitti_labels[clr] for clr in list(result['predict_labels'])])
# # show all classes

# print("classes: ", kitti_labels)

pcd.colors = o3d.utility.Vector3dVector(colors)

# Create visualization
custom_draw_geometry(pcd)
# show legend in the visualization
# custom_draw_geometry_with_legend(pcd)
# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()


def main():

if __name__ == main():
    main()