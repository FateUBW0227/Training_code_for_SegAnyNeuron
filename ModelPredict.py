'''预测类'''

import torch
from os.path import join
import numpy as np
# from skimage import exposure
import os
from models.model import LoadModel


class ModelPredictClass:
    def __init__(self, modelPath, device=torch.device('cpu'), two_encoders=False, feature_levels=0):
        self.device = device
        # 加载网络
        modelCfg = {
            'name': 'UNet3D',
            # number of input channels to the model
            'in_channels': 1 + feature_levels,
            # number of output channels
            'out_channels': 1,
            # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
            'layer_order': 'gcr',
            # number of features at each level of the U-Net
            'f_maps': [16, 32, 64, 128, 256],
            # 'f_maps': [16, 32, 64, 128],
            # 'f_maps_1': [8, 16],
            # 'f_maps_2': [16, 32, 64, 128],
            # 'addMapsId': 1,
            # 'f_maps': [32, 64, 128, 256, 512],
            # number of groups in the groupnorm
            'num_groups': 8,
            # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
            # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
            # to normalize with Sigmoid or Softmax
            'final_sigmoid': True,
            # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
            'is_segmentation': True
        }
        if feature_levels == 15:
            self.model = LoadModel(modelCfg, modelPath, -1)
        else:
            self.model = LoadModel(modelCfg, modelPath, feature_levels)
        self.model.to(device)
        self.model.eval()


    def __call__(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            seg = self.model(img)
            # seg[seg > 0.5] = 255
            seg = seg * 255
            seg = seg.to(torch.uint8).cpu().numpy()[0, 0].astype(np.uint8)
            return seg

