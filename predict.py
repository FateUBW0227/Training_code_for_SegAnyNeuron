import os
from os.path import join

import numpy
import numpy as np
import torch
from tifffile import tifffile, imwrite
from ModelPredict import ModelPredictClass
# from models.unet_3D import UNet3D
#from models.UnetModel4 import UNet3D
from tensorboardX import SummaryWriter

from loss import BCEDiceLoss, EvalScore, LSDLoss

# 定义数据加载函数
def load_test_data(path, names, imgSize, random, feature_levels = 3, key = None):
    erase_raw = False
    if feature_levels == 6:
        erase_raw = True
        feature_levels += 1

    imgPath = join(path, "raw")  # 原始图像路径
    featurePath = [join(path, f"feature_img/level_{i}") for i in range(feature_levels)]  # 各层级的特征图路径

    test_data = []
    for name in names:
        if key != None:
            temp_key = name.split('_')[0]
            if temp_key != key:
                continue

        img = tifffile.imread(join(imgPath, name))[:imgSize[2], :imgSize[1], :imgSize[0]].astype(np.float32)
        img = np.expand_dims(img, axis=0) # 扩展维度，添加通道维度

        feature_maps = []
        id = 1
        for level_path in featurePath:
            if random == 0:
                feature_map = tifffile.imread(join(level_path, name))[:imgSize[2], :imgSize[1], :imgSize[0]].astype(np.float32)
            elif random == 1:
                feature_map = (np.random.rand(128, 128, 128) * 256).astype(np.float32)
            elif random == 2:
                feature_map = (np.zeros((128, 128, 128))).astype(np.float32)
            feature_maps.append(np.expand_dims(feature_map, axis=0))  # 同样扩展维度
            # if id < -1:
            #     feature_map = (np.ones((128, 128, 128)) * 255 * 0 / 9.0).astype(np.float32)
            #     id = id + 1
            #     feature_maps.append(np.expand_dims(feature_map, axis=0))  # 同样扩展维度
            # else:
            #     feature_map = (np.ones((128, 128, 128)) * 255 * (id) / 9.0).astype(np.float32)
            #     id = id + 1
            #     feature_maps.append(np.expand_dims(feature_map, axis=0))  # 同样扩展维度
        if erase_raw == True:
            all_features = np.concatenate(feature_maps, axis=0)
        else:
            all_features = np.concatenate([img] + feature_maps, axis=0)  # 将原图像和所有特征图沿通道维度拼接
        test_data.append(all_features)

    test_data = torch.from_numpy(np.stack(test_data))  # 转换为Tensor格式
    return test_data, names


def load_test_data2(path, name, imgSize, random, feature_levels = 3):
    if feature_levels == 15:
        feature_levels = 0
    imgPath = join(path, "raw")  # 原始图像路径
    featurePath = [join(path, f"feature_img/level_{i}") for i in range(feature_levels)]  # 各层级的特征图路径

    test_data = []

    img = tifffile.imread(join(imgPath, name))[:imgSize[2], :imgSize[1], :imgSize[0]].astype(np.float32)
    img = np.expand_dims(img, axis=0) # 扩展维度，添加通道维度

    feature_maps = []
    for level_path in featurePath:
        if random == 0:
            feature_map = tifffile.imread(join(level_path, name))[:imgSize[2], :imgSize[1], :imgSize[0]].astype(np.float32)
        elif random == 1:
            feature_map = (np.random.rand(128, 128, 128) * 256).astype(np.float32)
        elif random == 2:
            feature_map = (np.zeros((128, 128, 128))).astype(np.float32)
        feature_maps.append(np.expand_dims(feature_map, axis=0))  # 同样扩展维度
    all_features = np.concatenate([img] + feature_maps, axis=0)  # 将原图像和所有特征图沿通道维度拼接
    test_data.append(all_features)

    test_data = torch.from_numpy(np.stack(test_data))  # 转换为Tensor格式
    return test_data, name

def obtain_list(names, group):
    for ls in names:
        prefix = ls.split('_')[0]
        if prefix not in group:
            group[prefix] = []
        group[prefix].append(ls)

def obtain_range(names):
    maxx = 0
    maxy = 0
    maxz = 0
    for name in names:
        x = int(name.split('_')[1])
        y = int(name.split('_')[2])
        z = int(name.split('_')[3].split('.')[0])
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
        if z > maxz:
            maxz = z
    return maxx, maxy, maxz


if __name__ == "__main__":
    test_path = r'.\fMost_dataset\fMost_dataset1'
    feature_levels = 0
    two_encoders = False
    random = 0

    imgSize = (192, 192, 192)
    batch_size = 1
    if feature_levels == 0:
        model_path = r'./pretrained_models/Without_feature/model1.pth'
        output_dir = os.path.join(test_path, 'without_feature')
        print(model_path)
    elif feature_levels == 7:
        model_path = r'./pretrained_models/With_features/model1.pth'
        output_dir = os.path.join(test_path, 'with_feature')
        print(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # 模型配置
    modelCfg = {
        "name": "UNet3D",
        "in_channels": 1+feature_levels,  # 主图像 + 7 个特征层级
        "out_channels": 1,
        "layer_order": "gcr",
        "f_maps": [16, 32, 64, 128, 256],
        "num_groups": 8,
        'final_sigmoid': True,
        'is_segmentation': True
    }

    # 检查是否有可用的 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 如果有 GPU 则使用 GPU，否则使用 CPU
    print(f"Using device: {device}")

    model = ModelPredictClass(modelPath=model_path, device=device, two_encoders=two_encoders, feature_levels=feature_levels)  # 预测类
    for ls in os.listdir(os.path.join(test_path, 'raw')):
        if os.path.isfile(os.path.join(os.path.join(test_path, 'raw'), ls)) and ls.split('.')[-1] == 'tif':
            test_names = [ls]
            print(test_names)
            print(f"Found {len(test_names)} test images.")

            # 加载测试数据
            test_features, test_names = load_test_data(test_path, test_names, imgSize, random, feature_levels)
            print(f"Loaded test data: {test_features.shape}")

            # 批处理预测
            with torch.no_grad():
                for i in range(0, len(test_features), batch_size):
                    name = test_names[i]
                    batch = test_features[i:i + batch_size]

                    predictions = model(batch)  # (batch_size, out_channels, depth, height, width)

                    for j in range(batch.size(0)):
                        name = test_names[i + j]  # 获取当前批次对应的文件名
                        output_path = ''
                        if feature_levels == 7:
                            output_path = os.path.join(output_dir, name.split('.')[0] + "with.tif")
                        elif feature_levels == 0:
                            output_path = os.path.join(output_dir, name.split('.')[0] + "without.tif")
                        predictions[predictions < 103] = 0
                        imwrite(output_path, predictions, compression='lzw')  # 假设是单通道输出
                        print(f"Saved prediction: {output_path}")

