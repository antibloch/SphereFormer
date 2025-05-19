import numpy as np
import os
import sys
import shutil
import open3d as o3d
learning_map = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
learning_map_inv = {  # inverse of previous map
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81     # "traffic-sign"
}
colors = {
    0: [0, 0, 0],       # unlabeled - black
    1: [0, 0, 1],       # car - blue
    2: [1, 0, 0],       # bicycle - red
    3: [1, 0, 1],       # motorcycle - magenta
    4: [0, 1, 1],       # truck - cyan
    5: [0.5, 0.5, 0],   # other-vehicle - olive
    6: [1, 0.5, 0],     # person - orange
    7: [1, 1, 0],       # bicyclist - yellow
    8: [1, 0, 0.5],     # motorcyclist - pink
    9: [0.5, 0.5, 0.5], # road - gray
    10: [0.5, 0, 0],    # parking - dark red
    11: [0, 0.5, 0],    # sidewalk - dark green
    12: [0, 0, 0.5],    # other-ground - dark blue
    13: [0, 0.5, 0.5],  # building - teal
    14: [0.5, 0, 0.5],  # fence - purple
    15: [0, 1, 0],      # vegetation - green
    16: [0.7, 0.7, 0.7],# trunk - light gray
    17: [0.7, 0, 0.7],  # terrain - light purple
    18: [0, 0.7, 0.7],  # pole - light cyan
    19: [0.7, 0.7, 0]   # traffic-sign - light yellow
}
velodyne_pth = 'kitti_ds/dataset/sequences/08/velodyne'
lab_pth = 'kitti_ds/dataset/sequences/08/labels'
target_pth = 'target_data'
if os.path.exists(target_pth):
    shutil.rmtree(target_pth)
os.makedirs(target_pth, exist_ok=True)
os.makedirs(os.path.join(target_pth, 'pc'), exist_ok=True)
os.makedirs(os.path.join(target_pth, 'label'), exist_ok=True)
velo_files = os.listdir(velodyne_pth)
lab_files = os.listdir(lab_pth)
velo_files = sorted(velo_files, key=lambda x: int(x.split('.')[0]))
lab_files = sorted(lab_files, key=lambda x: int(x.split('.')[0]))
for i in range(len(velo_files)):
    if i <1:
        file_name = velo_files[i].split('.')[0]
        velo_file = os.path.join(velodyne_pth, velo_files[i])
        lab_file = os.path.join(lab_pth, lab_files[i])
        target_file = os.path.join(target_pth, lab_files[i])
        # Read the velodyne file
        velo_data = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        # Read the label file
        lab_data = np.fromfile(lab_file, dtype=np.int32)
        label_data = lab_data.reshape((-1))
        semantic_labels = lab_data  & 0xFFFF  # bitwise AND with 0xFFFF
        # inst_labels = lab_data >> 16
        # assert ((semantic_labels + (inst_labels << 16) == label_data).all())
        remap_dict = learning_map
        # max_key = max(remap_dict.keys())
        # remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        # remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        # label = remap_lut[semantic_labels]
        u_s_l = np.unique(semantic_labels)
        label = np.zeros((semantic_labels.shape[0]), dtype=np.int32)
        for i in range(len(u_s_l)):
            label[semantic_labels == u_s_l[i]] = remap_dict[u_s_l[i]]
        label = (label - 1) % 19
        unique_labels = np.unique(label)
        colors_labs = np.zeros((len(label), 3)).astype(np.float32)
        for j, lab in enumerate(unique_labels):
            # colors_labs[j] = colors[lab]
            mask = (label == lab)
            colors_labs[mask] = colors[lab]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(velo_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors_labs)
        o3d.visualization.draw_geometries([pcd])
        np.save(os.path.join(target_pth, 'pc', file_name + '.npy'), velo_data)
        np.save(os.path.join(target_pth, 'label', file_name + '.npy'), label)