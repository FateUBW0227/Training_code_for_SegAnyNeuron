# """
# #512*512*512 -> 64*64*64
# import os
# import numpy as np
# from tifffile import tifffile
#
# def split_large_block(data, block_size, overlap, save_dir=None):
#
#     # 将大块数据切割成小块并保存。
#     # Args:
#     #     data: 输入的大块数据 (D, H, W)
#     #     block_size: 每个小块的大小 (depth, height, width)
#     #     overlap: 块之间的重叠大小 (depth, height, width)
#     #     save_dir: 保存切割块的目录 (如果为 None，仅返回切割后的数据列表)
#     # Returns:
#     #     blocks: 切割后的数据块列表 [(z_start, y_start, x_start, block)]
#
#     d, h, w = data.shape
#     bd, bh, bw = block_size
#     od, oh, ow = overlap
#
#     blocks = []
#     count = 0
#
#     # 创建保存目录
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#
#     # 遍历整个大块数据
#     for z in range(0, d - bd + 1, bd - od):
#         for y in range(0, h - bh + 1, bh - oh):
#             for x in range(0, w - bw + 1, bw - ow):
#                 # 提取小块
#                 block = data[z:z + bd, y:y + bh, x:x + bw]
#                 blocks.append((z, y, x, block))
#
#                 # 保存小块
#                 if save_dir:
#                     filename = os.path.join(save_dir, f"block_{count:05d}.tif")
#                     tifffile.imwrite(filename, block)
#                     print(f"Saved: {filename}")
#                 count += 1
#
#     print(f"Total blocks generated: {count}")
#     return blocks
#
#
# # 示例使用
# if __name__ == "__main__":
#     # 加载原始 512x512x512 数据
#     large_data = tifffile.imread("data\limo.tif")
#
#     # 设置分块参数
#     block_size = (64, 64, 64)  # 每块大小
#     overlap = (0, 0, 0)  # 块之间的重叠
#
#     # 分块并保存到目录
#     save_directory = r"path\limo_block"
#     blocks = split_large_block(large_data, block_size, overlap, save_dir=save_directory)
# """
#
# # import os
# # import numpy as np
# # from tifffile import tifffile
# #
# #
# # def split_large_block(data, block_size, save_dir):
# #     """
# #     将大块数据切割成小块并保存为 TIFF 文件。
# #
# #     Args:
# #         data: 输入的大块数据 (D, H, W)
# #         block_size: 每个小块的大小 (depth, height, width)
# #         save_dir: 保存切割块的目录
# #
# #     Returns:
# #         None
# #     """
# #     d, h, w = data.shape
# #     bd, bh, bw = block_size
# #
# #     # 创建保存目录
# #     os.makedirs(save_dir, exist_ok=True)
# #
# #     count = 0
# #     # 遍历整个大块数据并切割成小块
# #     for z in range(0, d, bd):
# #         for y in range(0, h, bh):
# #             for x in range(0, w, bw):
# #                 # 切割小块，不需要处理边界，直接取该位置的区域
# #                 block = data[z:z + bd, y:y + bh, x:x + bw]
# #
# #                 # 保存小块为 TIFF 文件
# #                 filename = os.path.join(save_dir, f"block_{count:05d}.tif")
# #                 tifffile.imwrite(filename, block)
# #                 print(f"Saved: {filename}")
# #
# #                 count += 1
# #
# #     print(f"Total blocks generated: {count}")
# #
# #
# # # 示例使用
# # if __name__ == "__main__":
# #     # 加载原始 512x512x512 数据
# #     large_data = tifffile.imread("data/qh1.tif")
# #
# #     # 设置分块参数
# #     block_size = (64, 64, 64)  # 每块大小
# #
# #     # 保存切割块的目录
# #     save_directory = r"path2/qh1_block"
# #
# #     # 切割并保存小块
# #     split_large_block(large_data, block_size, save_dir=save_directory)
#
#
# import os
# import numpy as np
# from tifffile import tifffile
#
#
# def merge_blocks(blocks_dir, block_size, original_shape):
#     """
#     将切割的小块拼接回原始的大块数据。
#
#     Args:
#         blocks_dir: 保存小块的目录路径
#         block_size: 每个小块的大小 (depth, height, width)
#         original_shape: 原始数据的形状 (depth, height, width)
#
#     Returns:
#         merged_data: 拼接后的大块数据
#     """
#     bd, bh, bw = block_size
#     d, h, w = original_shape
#
#     # 计算每个维度需要多少个块
#     num_blocks_z = d // bd
#     num_blocks_y = h // bh
#     num_blocks_x = w // bw
#
#     # 初始化空的大块数据
#     merged_data = np.zeros(original_shape, dtype=np.uint16)
#
#     # 获取所有块的文件名并按顺序排序
#     block_files = [f for f in os.listdir(blocks_dir) if f.endswith('.tif')]
#     block_files.sort()  # 假设文件名按顺序命名
#
#     count = 0
#     for block_file in block_files:
#         # 加载小块
#         block = tifffile.imread(os.path.join(blocks_dir, block_file))
#
#         # 获取小块的块编号，假设文件名格式为 "block_XXXXX.tif"
#         block_index = int(block_file.split('_')[1].split('.')[0])  # 提取块的编号
#
#         # 计算块的坐标
#         z = (block_index // (num_blocks_y * num_blocks_x)) * bd
#         y = ((block_index % (num_blocks_y * num_blocks_x)) // num_blocks_x) * bh
#         x = (block_index % num_blocks_x) * bw
#
#         # 将小块放回原始数据的对应位置
#         merged_data[z:z + bd, y:y + bh, x:x + bw] = block
#         count += 1
#         print(f"Merged block {block_index} into position ({z}, {y}, {x})")
#
#     print(f"Total blocks merged: {count}")
#     return merged_data
#
#
# # 示例使用
# if __name__ == "__main__":
#     # 设置参数
#     block_size = (64, 64, 64)  # 小块大小
#     original_shape = (512, 512, 512)  # 原始数据形状
#
#     # 小块保存目录
#     blocks_dir = r"limo_4_pre"
#
#     # 拼接小块成大块
#     merged_data = merge_blocks(blocks_dir, block_size, original_shape)
#
#     # 可以保存合并后的大块数据
#     tifffile.imwrite(r"limo_4.tif", merged_data)
#     print("Merged data saved to 'merged_limo.tif'")
#
#
#
