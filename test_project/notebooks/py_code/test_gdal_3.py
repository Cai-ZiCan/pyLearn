# -*- coding: utf-8 -*-
"""
GDAL 实用工具方法集
================================================================================
本脚本整理了使用 GDAL 处理遥感影像时的常用便捷方法。

主要功能：
1. find_files: 快速搜索指定目录下的影像文件（支持递归）。
2. stack_bands: 利用 VRT 虚拟数据集技术，高效合并多波段数据。

使用方法：
可以直接运行本脚本查看示例输出，或导入特定函数到其他脚本中使用。
"""

import os
import glob
from osgeo import gdal

# ------------------------------------------------------------------------------
# 1. 快速文件读取模块
# ------------------------------------------------------------------------------

def find_files(input_dir, pattern="*.tif", recursive=False):
    """
    在指定目录中搜索符合模式的文件。

    Args:
        input_dir (str): 搜索的根目录路径。
        pattern (str): 文件匹配模式，例如 "*.tif" 或 "*.jp2"。
                       - Sentinel-2 原始数据通常是 .jp2
                       - Landsat 通常是 .tif
        recursive (bool): 是否递归搜索子目录。默认为 False。

    Returns:
        list: 匹配到的文件完整路径列表。
    """
    if recursive:
        # 使用 ** 通配符进行递归搜索
        search_path = os.path.join(input_dir, "**", pattern)
        files = glob.glob(search_path, recursive=True)
    else:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
    
    if not files:
        print(f"提示: 在 {input_dir} 中未找到匹配 {pattern} 的文件。")
    else:
        print(f"在 {input_dir} 中发现 {len(files)} 个匹配文件。")
        
    return files

# ------------------------------------------------------------------------------
# 2. 波段合成模块
# ------------------------------------------------------------------------------

def stack_bands(input_files, output_file):
    """
    将多个单波段文件合并为一个多波段 GeoTIFF 文件。
    
    原理：
    通过构建 VRT (Virtual Dataset) 中间文件来实现。这种方法非常高效，
    因为它不需要在内存中实际复制像素数据，而是创建一个 XML 描述文件引用原始数据，
    最后再一次性转录为目标文件。

    Args:
        input_files (list): 包含单波段文件路径的列表。列表顺序决定输出文件的波段顺序。
        output_file (str): 输出的多波段 GeoTIFF 文件路径。

    Returns:
        bool: 成功返回 True，失败返回 False。
    """
    if not input_files:
        print("错误: 输入文件列表为空。")
        return False

    # 1. 创建 VRT 虚拟文件
    # separate=True 是关键参数，表示将每个输入文件作为一个独立的波段（stacked），
    # 而不是进行空间上的拼接（mosaic）。
    vrt_options = gdal.BuildVRTOptions(separate=True)
    temp_vrt = "temp_stack.vrt"
    
    try:
        # 创建虚拟数据集
        vrt_ds = gdal.BuildVRT(temp_vrt, input_files, options=vrt_options)
        
        if vrt_ds is None:
            print("错误: 创建 VRT 失败，请检查输入文件路径是否正确。")
            return False

        print(f"正在转换 VRT 到 GeoTIFF: {output_file} ...")

        # 2. 将 VRT 转换为真正的 GeoTIFF
        # CreateCopy 会执行实际的像素读取和写入操作
        # 推荐参数: COMPRESS=LZW (无损压缩), TILED=YES (分块存储，优化读取速度)
        driver = gdal.GetDriverByName("GTiff")
        output_ds = driver.CreateCopy(output_file, vrt_ds, strict=0, 
                                     options=["COMPRESS=LZW", "TILED=YES"])
        
        # 3. 释放资源，确保数据写入磁盘
        output_ds = None
        vrt_ds = None # 关闭 VRT 数据集

        print(f"成功: 已合并 {len(input_files)} 个波段。")
        return True

    except Exception as e:
        print(f"发生异常: {str(e)}")
        return False
        
    finally:
        # 4. 清理临时文件
        if os.path.exists(temp_vrt):
            try:
                os.remove(temp_vrt)
            except OSError:
                pass

# ------------------------------------------------------------------------------
# 主程序执行入口
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # 示例 1: 搜索文件
    # 假设有一个数据目录 (请根据实际情况修改)
    sample_dir = "./data/tif" 
    
    # 确保存储目录存在，避免报错
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        print(f"创建测试目录: {sample_dir}")

    tif_files = find_files(sample_dir, "*.tif")
    
    # 示例 2: 波段合成
    # 假设我们找到了一些文件，或者手动指定
    # 这里为了演示，构造一些虚拟的文件名
    landsat_bands = [
        os.path.join(sample_dir, "LC08_B1.tif"), # Coastal
        os.path.join(sample_dir, "LC08_B2.tif"), # Blue
        os.path.join(sample_dir, "LC08_B3.tif"), # Green
        os.path.join(sample_dir, "LC08_B4.tif")  # Red
    ]

    # 注意：如果这些文件实际上不存在，GDAL 会报错。
    # 在实际运行前，请确保文件真实存在。
    # stack_bands(landsat_bands, "result/Landsat8_Stack.tif")
    
    print("\n脚本执行完毕")