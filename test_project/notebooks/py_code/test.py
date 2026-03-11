import glob
import os
import numpy as np


# 使用示例：



if __name__ == "__main__":
    from toolbox.readData import jp2astiff
    r10m_dir = (
        "/mnt/e/S2B_MSIL2A_20230611T030529_N0509_R075_T49RFL_20230611T053545/"
        "S2B_MSIL2A_20230611T030529_N0509_R075_T49RFL_20230611T053545.SAFE/"
        "GRANULE/L2A_T49RFL_A032705_20230611T031522/IMG_DATA/R10m"
    )
    if not os.path.isdir(r10m_dir):
        print(f"错误: 目录不存在: {r10m_dir}")
        raise SystemExit(1)

    jp2_files = sorted(glob.glob(os.path.join(r10m_dir, "*.jp2")))
    if not jp2_files:
        print(f"错误: 未找到 JP2 文件: {r10m_dir}")
        raise SystemExit(1)

    jp2astiff(
        jp2_files,
        "/mnt/d/pyLearn/pyLearn/test_project/data/tif/"
        "S2B_MSIL2A_20230611T030529_N0509_R075_T49RFL_20230611T053545_10m_7bands.tif",
    )

# if __name__ == "__main__":

#         from matplotlib import pyplot as plt
#         from osgeo import gdal
#         from matplotlib import pyplot as plt
#         gdal.UseExceptions()
#         data_path=("/mnt/d/pyLearn/pyLearn/test_project/data/tif/"
#                 "S2B_MSIL2A_20230611T030529_N0509_R075_T49RFL_20230611T053545_10m_7bands.tif")
#         with gdal.Open(data_path) as src:
#              band=src.ReadAsArray()
        
#         rgb=band[[2,1,0],:,:] # 选择需要的波段，注意波段索引可能需要调整
#         img=rgb.transpose(1,2,0) # 将波段数据转换为图像格式，注意波段顺序可能需要调整
#         # 以rgb进行显示
#         # 对uint16类型的图像进行归一化处理
#         img = img.astype('float32') / band.max()  # 使用实际的最大值进行归一化
#         # 显示图像
#         plt.imshow(img)
#         plt.show()
#         rgb = band[[2, 1, 0], :, :]  # B4,B3,B2
#         img = rgb.transpose(1, 2, 0).astype("float32")
        
#         # 2-98% 拉伸（逐通道）
#         for i in range(3):
#             p2, p98 = np.percentile(img[:, :, i], (2, 98))
#             img[:, :, i] = np.clip((img[:, :, i] - p2) / (p98 - p2), 0, 1)
#         gamma = 0.80  # 可以调整gamma值
#         img = np.power(img, gamma)  # 应用gamma校正
#         plt.imshow(img)
#         plt.show()
