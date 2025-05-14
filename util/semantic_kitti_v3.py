import os
import random
import glob
import numpy as np
import torch
import yaml
import pickle
from util.data_util import data_prepare
import scipy

#Elastic distortion
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0)//gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag


class SemanticKITTI_Pathetic(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        voxel_size=[0.1, 0.1, 0.1], 
        return_ref=True, 
        label_mapping="util/semantic-kitti.yaml", 
        rotate_aug=True, 
        flip_aug=True, 
        scale_aug=True, 
        scale_params=[0.95, 1.05], 
        transform_aug=True, 
        trans_std=[0.1, 0.1, 0.1],
        elastic_aug=False, 
        elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        ignore_label=255, 
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
        test_path = '',
        test_path_label = '',
    ):
        super().__init__()
        self.num_classes = 19
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.return_ref = return_ref
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.elastic_aug = elastic_aug
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]
        self.use_tta = use_tta
        self.vote_num = vote_num
        self.test_path = test_path
        self.test_path_label = test_path_label


        splits = semkittiyaml['split']['valid']

        self.files = []
        point_names = os.listdir(self.test_path)
        point_names = sorted(point_names, key=lambda x: int(x.split('.')[0]))
        for fil in point_names:
            # self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'velodyne', "*.bin")))
            full_file_path = os.path.join(self.test_path, fil)
            self.files.append(full_file_path)

        self.lab_files = []
        lab_names = os.listdir(self.test_path_label)
        lab_names = sorted(lab_names, key=lambda x: int(x.split('.')[0]))
        for fil in lab_names:
            # self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'velodyne', "*.bin")))
            full_file_path = os.path.join(self.test_path_label, fil)
            self.lab_files.append(full_file_path)


        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)


        self.voxel_size = voxel_size

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.files)

    def __getitem__(self, index):

        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):

        file_path = self.files[index]
        label_path = self.lab_files[index]

        # raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        raw_data = np.load(file_path)

        points = raw_data[:, :4]

        # labels_in = np.zeros(points.shape[0]).astype(np.uint8)
        labels_in = np.load(label_path)
        labels_in = labels_in.astype(np.uint8)  

        feats = points

        xyz = points[:, :3]

        ref_points = xyz.copy()


        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])


        coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, 'val', self.voxel_size, self.voxel_max, None, self.xyz_norm)

        fil_name = os.path.basename(file_path)
        fil_name = fil_name.split('.')[0]

        return coords, xyz, feats, labels, inds_reconstruct, ref_points, fil_name
