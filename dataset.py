from os.path import join
import numpy as np
import torch
from tifffile import tifffile


class get_dataset:
    def __init__(self, path, txtName, imgSize, feature_levels, cfg):
        self.imgSize = imgSize
        self.imgPath = join(path, cfg['raw_img_dir_name'])
        self.maskPath = join(path, cfg['label_dir_name'])
        feature_name = cfg['feature_dir_name']
        self.featurePath = [join(path, f'{feature_name}/level_{i}') for i in range(feature_levels)]
        self.feature_levels = feature_levels
        with open(join(path, txtName), 'r') as f:
            self.names = f.read().strip().split('\n')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # 读取主图像
        img = tifffile.imread(join(self.imgPath, self.names[index]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # 读取层级特征图
        feature_maps = []
        for level_path in self.featurePath:
            feature_map = tifffile.imread(join(level_path, self.names[index]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
            feature_maps.append(np.expand_dims(feature_map, axis=0))
        all_features = np.concatenate([img] + feature_maps, axis=0)  # 按通道维度堆叠
        #all_features = np.concatenate(feature_maps, axis=0)  #
        all_features = torch.from_numpy(all_features)

        mask = tifffile.imread(join(self.maskPath, self.names[index]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]] / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return all_features, mask, self.names[index]

