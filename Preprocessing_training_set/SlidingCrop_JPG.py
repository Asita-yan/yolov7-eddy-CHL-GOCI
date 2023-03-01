import glob
import os
from PIL import Image
import numpy as np
# 输入大图片路径和输出小图片路径
gts_all_path = glob.glob('E:\\photo\\??\\*.jpg')
for file in gts_all_path:
    input_path = file
    file_name=file.split('\\')[-1][:-4]
    output_path = "E:\\photo\\clip\\"+file.strip().split('\\')[2]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 每个小图片的大小
    patch_size = (640, 640)

    # 重叠度
    eddy_radius=100  #km
    overlap = eddy_radius*1000/500/640  # 重叠度(最大半径/分辨率/patch大小)

    # 打开大图片
    with Image.open(input_path) as img:
        width, height = img.size
        # print(f"Image size: {width} x {height}")

        # 计算水平和垂直方向的重叠大小
        overlap_size = (int(patch_size[0] * overlap), int(patch_size[1] * overlap))

        # 计算小图片的行数和列数
        num_cols = int((width - patch_size[0]) / (patch_size[0] - overlap_size[0])) + 1
        num_rows = int((height - patch_size[1]) / (patch_size[1] - overlap_size[1])) + 1
        print(f"file name: {file_name}")

        # 裁剪小图片
        for row in range(num_rows):
            for col in range(num_cols):
                patch_name = f"{file_name}_{row}_{col}.jpg"
                patch_path = os.path.join(output_path, patch_name)
                if os.path.exists(patch_path):
                    continue
                # 计算裁剪区域的左上角和右下角坐标
                left = col * (patch_size[0] - overlap_size[0])
                upper = row * (patch_size[1] - overlap_size[1])
                right = left + patch_size[0]
                lower = upper + patch_size[1]

                # 如果裁剪区域超出了原图范围，则将其调整为合法范围
                if right > width:
                    left = width - patch_size[0]
                    right = width
                if lower > height:
                    upper = height - patch_size[1]
                    lower = height


                # 裁剪小图片并保存

                patch = img.crop((left, upper, right, lower))
                vaild=np.array(patch)
                if (vaild==0).sum()/(512*512)>0.95:
                    continue
                else:
                    patch.save(patch_path)

