import os
import cv2
import torch
from tqdm import tqdm
import numpy as np

from annotator.uniformer import UniformerDetector
from annotator.util import resize_image, HWC3

# 初始化分割模型
uniformer = UniformerDetector()

# 输入输出路径
input_dir = '/content/drive/MyDrive/VisDrone2019-DET/VisDrone2019-DET-train/images'
output_dir = '/content/drive/MyDrive/VisDrone2019-DET/VisDrone2019-DET-train/segmaps'
os.makedirs(output_dir, exist_ok=True)

# 遍历图像并生成语义图
for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith(('.jpg', '.png')):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

    # 读取图像并预处理
    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to read {input_path}")
        continue

    image = HWC3(image)
    resized_image = resize_image(image, 512)

    # 推理语义图（默认ADE20k）
    with torch.no_grad():
        seg_map = uniformer(resized_image)

    # 恢复原图尺寸
    seg_map = cv2.resize(seg_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 保存语义图（单通道图）
    cv2.imwrite(output_path, seg_map)

print("✅ 批量语义分割完成！结果已保存至：", output_dir)