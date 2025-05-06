import os
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import tifffile as tiff
from tqdm import tqdm


def augment_and_save(image_folder, label_folder, output_folder, selected_augmentation="gaussian_noise"):
    """
    对原图进行单一数据增强并保存，仅生成一个增强样本，标签保持不变。

    参数：
        image_folder: 原图文件夹路径
        label_folder: 标签文件夹路径
        output_folder: 输出文件夹路径
        selected_augmentation: 选择的数据增强方法 ("gaussian_noise", "brightness", "blur")
    """
    # 创建输出文件夹
    augmented_image_folder = os.path.join(output_folder, "augmented_images")
    augmented_label_folder = os.path.join(output_folder, "augmented_labels")
    os.makedirs(augmented_image_folder, exist_ok=True)
    os.makedirs(augmented_label_folder, exist_ok=True)

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

        assert image.shape == label.shape, f"图像和标签形状不一致: {image_file}, {label_file}"

        # 保存原始图像和标签
        base_name = os.path.splitext(image_file)[0]
        tiff.imwrite(os.path.join(augmented_image_folder, f"{base_name}_original.tif"), image.astype(np.uint16))
        tiff.imwrite(os.path.join(augmented_label_folder, f"{base_name}_original.tif"), label.astype(np.uint8))

        # 生成一个增强样本
        augmented_image = augment_image(image, selected_augmentation)

        augmented_image_path = os.path.join(augmented_image_folder, f"{base_name}_aug.tif")
        augmented_label_path = os.path.join(augmented_label_folder, f"{base_name}_aug.tif")

        tiff.imwrite(augmented_image_path, augmented_image.astype(np.uint16))
        tiff.imwrite(augmented_label_path, label.astype(np.uint8))  # 标签保持不变


def augment_image(image, method):
    """
    对图像进行单一增强。

    参数：
        image: 原图 (numpy array)
        method: 增强方法 ("gaussian_noise", "brightness", "blur")
    返回：
        augmented_image: 增强后的图像
    """
    augmented_image = image.copy()

    if method == "gaussian_noise":
        # 添加高斯噪声
        noise = np.random.normal(0, random.uniform(5, 10), size=augmented_image.shape)
        augmented_image = augmented_image + noise
        augmented_image = np.clip(augmented_image, 0, 4095)

    elif method == "brightness":
        # 调整亮度
        factor = random.uniform(0.8, 1.2)
        augmented_image = augmented_image * factor
        augmented_image = np.clip(augmented_image, 0, 4095)

    elif method == "blur":
        # 随机模糊
        sigma = random.uniform(0.5, 1.5)
        augmented_image = gaussian_filter(augmented_image, sigma=sigma)

    return augmented_image


# 示例调用
image_folder = r"D:\DQW\Data\3D_train\select_block\cropped_images"  # 替换为你的原图文件夹路径
label_folder = r"D:\DQW\Data\3D_train\select_block\cropped_labels"  # 替换为你的标签文件夹路径
output_folder = r"D:\DQW\Data\3D_train\select_block_aug"  # 替换为存储路径

# 选择数据增强方法: "gaussian_noise", "brightness", or "blur"
selected_augmentation = "gaussian_noise"

augment_and_save(image_folder, label_folder, output_folder, selected_augmentation=selected_augmentation)
