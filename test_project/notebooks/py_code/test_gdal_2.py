# 在当前文件中测试空间分析与坐标变换的功能
# 尝试使用 gdal实现对栅格数据的裁剪操作，仿照采用 rasterio 的方式进行裁剪

from osgeo import gdal
import geopandas as gpd
import os

# 1. 设置输入输出路径
data_path_file = r'D:\\pyLearn\\test_project\\data\tif\\v_mean_15_25.tif'  # 替换为你要裁剪的数据的文件夹路径
shp_path = r'D:\\pyLearn\\test_project\\data\\shp\\result_no_Sea.shp'  # 替换为shp文件路径
clipped_data_file = r'D:\\pyLearn\\test_project\\data\\tif\\clipped.tif'  # 替换为裁剪后数据的输出文件路径

options=gdal.WarpOptions(
    cutlineDSName=shp_path, 
    cropToCutline=True, 
    dstSRS='EPSG:4326',  # 输出坐标系为WGS84
    dstNodata=0,
    resampleAlg=gdal.GRA_Bilinear,  # 双线性插值
    format='GTiff'
)

# 裁剪
gdal.Warp(clipped_data_file, data_path_file, options=options)