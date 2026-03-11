# ------------------------------------------------------------------------------
# 1. 波段合成模块(VRT方法)
# ------------------------------------------------------------------------------
from osgeo import gdal
import math
import os


def _nodata_values_equal(left_value, right_value):
    if left_value is None or right_value is None:
        return left_value is None and right_value is None

    try:
        if math.isnan(left_value) and math.isnan(right_value):
            return True
    except TypeError:
        pass

    return left_value == right_value


def _resolve_output_nodata(input_files, fallback_nodata_value):
    input_nodata_values = []

    for input_file in input_files:
        input_ds = gdal.Open(input_file)
        if input_ds is None:
            raise RuntimeError(f"无法打开输入文件: {input_file}")

        try:
            if input_ds.RasterCount < 1:
                raise RuntimeError(f"输入文件不包含波段: {input_file}")

            input_band = input_ds.GetRasterBand(1)
            input_nodata_values.append(input_band.GetNoDataValue())
        finally:
            input_ds = None

    resolved_nodata = input_nodata_values[0]
    if all(_nodata_values_equal(resolved_nodata, value) for value in input_nodata_values[1:]):
        if resolved_nodata is not None:
           return resolved_nodata
        else:
            print("提示: 所有输入波段的 NoData 都是 None，将输出 NoData 设置为 0。")
            return 0  # 如果所有输入波段的 NoData 都是 None，则回退到 0

    print(
        "提示: 输入波段的 NoData 值不一致，"
        f"将输出 NoData 设置为 {fallback_nodata_value}。"
    )
    return fallback_nodata_value


def stack_bands(input_files, output_file, nodata_value=0):
    """
    将多个单波段tiff文件合并为一个多波段 GeoTIFF 文件。
    
    原理：
    通过构建 VRT (Virtual Dataset) 中间文件来实现。这种方法非常高效，
    因为它不需要在内存中实际复制像素数据，而是创建一个 XML 描述文件引用原始数据，
    最后再一次性转录为目标文件。

    Args:
        input_files (list): 包含单波段文件路径的列表。列表顺序决定输出文件的波段顺序。
        output_file (str): 输出的多波段 GeoTIFF 文件路径。
        nodata_value (int | float, optional): 当输入波段的 NoData 值不一致时，输出多波段数据采用的回退 NoData 值。默认值为 0。

    Returns:
        bool: 成功返回 True，失败返回 False。
    """
    if not input_files:
        print("错误: 输入文件列表为空。")
        return False

    resolved_nodata = _resolve_output_nodata(input_files, nodata_value)

    # 1. 创建 VRT 虚拟文件
    # separate=True 是关键参数，表示将每个输入文件作为一个独立的波段（stacked），
    # 而不是进行空间上的拼接（mosaic）。
    vrt_options_kwargs = {"separate": True}
    if resolved_nodata is not None:
        vrt_options_kwargs["VRTNodata"] = resolved_nodata
    vrt_options = gdal.BuildVRTOptions(**vrt_options_kwargs)
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

        if output_ds is None:
            print("错误: 创建输出 GeoTIFF 失败。")
            return False

        for band_index in range(1, output_ds.RasterCount + 1):
            output_band = output_ds.GetRasterBand(band_index)
            if resolved_nodata is not None:
                output_band.SetNoDataValue(resolved_nodata)
        print(f"Nodata 值已设置为 {resolved_nodata}。")
        output_ds.FlushCache()  # 确保所有数据写入磁盘
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
