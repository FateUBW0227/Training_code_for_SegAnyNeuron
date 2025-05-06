import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import random  # 替换 np.random.choice


def hybrid_crop_blocks(image_folder, label_folder, output_folder, crop_size=128, step=128, fiber_threshold=128,
                       high_fiber_ratio=0.8, num_crops=400):
    """
    结合高纤维和低纤维区域的裁剪策略。

    参数：
        image_folder: 原图文件夹路径
        label_folder: 标签文件夹路径
        output_folder: 输出文件夹路径
        crop_size: 裁剪块的大小 (默认 128)
        step: 滑窗步长 (默认 128)
        fiber_threshold: 判断纤维体素的阈值 (默认 128)
        high_fiber_ratio: 高纤维区域的采样比例 (默认 70%)
        num_crops: 每张图像最终采样的块数量 (默认 400)
    """
    # 创建输出文件夹
    cropped_image_folder = os.path.join(output_folder, "cropped_images")
    cropped_label_folder = os.path.join(output_folder, "cropped_labels")
    os.makedirs(cropped_image_folder, exist_ok=True)
    os.makedirs(cropped_label_folder, exist_ok=True)

    # 获取原图和标签文件列表
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))

    assert len(image_files) == len(label_files), "原图和标签数量不一致！"

    for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        # 加载原图和标签
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        image = tiff.imread(image_path)  # 读取原图
        label = tiff.imread(label_path)  # 读取标签

        assert image.shape == label.shape, "图像和标签形状不一致！"

        D, H, W = label.shape
        high_fiber_blocks = []
        low_fiber_blocks = []

        for z in range(0, D - crop_size + 1, step):
            for y in range(0, H - crop_size + 1, step):
                for x in range(0, W - crop_size + 1, step):
                    # 裁剪出小块
                    image_block = image[z:z + crop_size, y:y + crop_size, x:x + crop_size]
                    label_block = label[z:z + crop_size, y:y + crop_size, x:x + crop_size]

                    # 计算纤维体素占比
                    fiber_ratio = np.mean(label_block > fiber_threshold)

                    if fiber_ratio > 0.04:  # 高纤维区域
                        high_fiber_blocks.append((image_block, label_block))
                    elif 0 < fiber_ratio <= 0.01:  # 低纤维区域
                        low_fiber_blocks.append((image_block, label_block))

        # 计算采样数量
        num_high_fiber = int(num_crops * high_fiber_ratio)
        num_low_fiber = num_crops - num_high_fiber

        # 随机采样
        high_fiber_samples = random.sample(high_fiber_blocks, min(len(high_fiber_blocks), num_high_fiber))
        low_fiber_samples = random.sample(low_fiber_blocks, min(len(low_fiber_blocks), num_low_fiber))

        # 合并采样结果
        all_samples = high_fiber_samples + low_fiber_samples

        # 保存小块
        for i, (image_block, label_block) in enumerate(all_samples):
            cropped_image_path = os.path.join(cropped_image_folder, f"{os.path.splitext(image_file)[0]}_block_{i}.tif")
            cropped_label_path = os.path.join(cropped_label_folder, f"{os.path.splitext(label_file)[0]}_block_{i}.tif")

            tiff.imwrite(cropped_image_path, image_block.astype(np.uint16))  # 保存原图小块
            tiff.imwrite(cropped_label_path, label_block.astype(np.uint8))  # 保存标签小块

    print(f"裁剪完成，结果已保存至 {cropped_image_folder} 和 {cropped_label_folder}")


# 示例调用
image_folder = r"D:\DQW\Data\3D_train\raw_image"  # 替换为你的原图文件夹路径
label_folder = r"D:\DQW\Data\3D_train\mask"  # 替换为你的标签文件夹路径
output_folder = r"D:\DQW\Data\3D_train\select_block"  # 替换为存储路径

hybrid_crop_blocks(image_folder, label_folder, output_folder, crop_size=128, step=128, fiber_threshold=128,
                   high_fiber_ratio=0.7, num_crops=400)


