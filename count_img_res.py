import os
from PIL import Image
from collections import defaultdict


def count_image_resolutions(directory):
    # 用于存储每种分辨率的图片数量
    resolution_count = defaultdict(int)

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理图片文件，扩展名为 .jpg、.jpeg、.png 等
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                file_path = os.path.join(root, file)

                try:
                    # 打开图片并获取分辨率
                    with Image.open(file_path) as img:
                        resolution = img.size  # (width, height)
                        resolution_count[resolution] += 1
                except Exception as e:
                    print(f"无法处理图片 {file_path}: {e}")

    # 打印每种分辨率及其数量
    for resolution, count in resolution_count.items():
        print(f"分辨率 {resolution[0]}x{resolution[1]} 的图片数量: {count}")


if __name__ == "__main__":
    # 替换成你要遍历的文件夹路径
    folder_path = "/home/flyingbucket/DataStore/scsr_results/coco/hr"
    print(f"folder_path:{folder_path}")
    count_image_resolutions(folder_path)
