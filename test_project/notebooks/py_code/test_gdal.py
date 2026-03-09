### GDAL官网教程的示例代码 ###

from osgeo import gdal
from error_deal import catch_error 

filename ="D:/pyLearn/test_project/data/tif/resultData_withoutSea.tif"

gdal.UseExceptions()
try:
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
except Exception as e:
    print(f"Error opening dataset.Error message: {e} !")

### 1. 错误处理示例 ###
# gdal.UseExceptions()
# try:
#     dataset = gdal.Open("non_existent_file.tif", gdal.GA_ReadOnly)
# except Exception as e:
#     print(f"Error opening dataset.Error message: {e} !")

# gdal.PushErrorHandler(catch_error.gdal_error_handler)
# gdal.Open("non_existent_file.tif", gdal.GA_ReadOnly)
# gdal.PopErrorHandler()

# print("Finished error handling test.")


### 2. 读取数据元数据测试 ###
gdal.PushErrorHandler(catch_error.gdal_error_handler)
adfGeoTransform = dataset.GetGeoTransform()
print("GeoTransform: ", adfGeoTransform) # 输出地理变换参数
"""
GeoTransform 对应关系如下：
adfGeoTransform[0] = top left x (左上角x坐标)
adfGeoTransform[1] = w-e pixel resolution (像素宽度)
adfGeoTransform[2] = rotation, 0 if image is "north up" (旋转，0表示图像是“北向上”)
adfGeoTransform[3] = top left y (左上角y坐标)
adfGeoTransform[4] = rotation, 0 if image is "north up" (旋转，0表示图像是“北向上”)
adfGeoTransform[5] = n-s pixel resolution (像素高度，通常为负值)

"""
gdal.PopErrorHandler()


### 3. 读取数据波段信息测试 ###
band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType))) 

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True) # 如果没有读取到，则计算最小值和最大值，参数True表示使用近似算法以加快计算速度
print("Min={:.3f}, Max={:.3f}".format(min,max))

# 还可以检查波段是否有概述（overviews）或颜色表（color table）
if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

### 4. 读取波段对象数据值测试 ###
# 读取结果为一个字节字符串，长度为XSize*YSize*4（每个像素值占4字节，因为是32位浮点数），需要使用struct模块将其解包成一个浮点数元组，每个元素对应一个像素值
scanline = band.ReadRaster(xoff=0, yoff=0,
                        xsize=band.XSize, ysize=band.YSize,
                        buf_xsize=band.XSize, buf_ysize=band.YSize,
                        buf_type=gdal.GDT_Float32)

import struct
# 解包后为一个包含所有像素值的浮点数元组，长度为XSize*YSize，之后若需要展示为二维数组，可以使用numpy进行重塑
tuple_of_floats = struct.unpack('f' * band.XSize * band.YSize, scanline) # 将字节数据解包成浮点数元组，'f'表示每个元素是一个32位浮点数，乘以XSize*YSize表示有多少个元素
import numpy as np
numpy_array = np.array(tuple_of_floats).reshape((band.YSize, band.XSize)) # 将一维的像素值元组重塑为二维数组，行数为YSize，列数为XSize
#展示读取到的band影像
import matplotlib.pyplot as pyplot
pyplot.plot(tuple_of_floats)
pyplot.show()
pyplot.imshow(numpy_array, cmap='gray')

### 5. 关闭数据集测试 ###
dataset = None # 关闭数据集，释放资源,python没有dataset.GDALClose()方法，直接将dataset置为None即可关闭数据集并释放资源
#建议上述全过程采用 with 语句管理资源，例如：
# with gdal.Open(filename, gdal.GA_ReadOnly) as dataset:
#     # 在这里使用dataset进行操作，离开with块后dataset会自动关闭

### 6. 拷贝数据文件 ###
# 使用GDAL驱动创建一个新的数据集，并将原数据集的内容复制到新数据集中，参数strict=0表示在复制过程中允许某些不兼容的选项被忽略，
# options参数可以指定一些创建选项，例如TILED=YES表示创建一个分块的tif文件，COMPRESS=PACKBITS表示使用PackBits压缩算法来压缩数据以节省存储空间
src_filename = "D:/pyLearn/test_project/data/tif/resultData_withoutSea.tif"
dst_filename = "D:/pyLearn/test_project/data/tif/resultData_withoutSea_copy.tif"
driver = gdal.GetDriverByName('GTiff') # 获取GTiff驱动对象，GTiff是GDAL支持的一个常见的栅格数据格式，代表GeoTIFF格式，可以处理地理空间信息的TIFF文件
src_ds = gdal.Open(src_filename)
dst_ds = driver.CreateCopy(dst_filename, src_ds, strict=0,
                        options=["TILED=YES", "COMPRESS=PACKBITS"])
# Once we're done, close properly the dataset
dst_ds = None
src_ds = None 

### 7. 创建指定类型空白文件 ###
dst_ds = driver.Create(dst_filename, xsize=512, ysize=512,
                    bands=1, eType=gdal.GDT_Byte)
from osgeo import osr
import numpy
dst_ds.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
srs = osr.SpatialReference() # 创建一个空间参考对象
srs.SetUTM(11, 1)
srs.SetWellKnownGeogCS("NAD27")
dst_ds.SetProjection(srs.ExportToWkt()) # 设置投影信息，使用WKT格式，ExportToWkt()方法将空间参考对象转换为WKT字符串
raster = numpy.zeros((512, 512), dtype=numpy.uint8)
dst_ds.GetRasterBand(1).WriteArray(raster) # 将numpy数组写入波段对象，WriteArray()方法将数组数据写入到波段中，参数raster是一个512x512的二维数组，数据类型为uint8（无符号8位整数），表示每个像素值的范围是0-255
# Once we're done, close properly the dataset
dst_ds = None