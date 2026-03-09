from osgeo import gdal

def get_dataset_info(dataset: gdal.Dataset) -> dict:
    """
    读取并解析 GDAL 数据集的元数据信息。

    该函数提取遥感或地理数据的关键属性，包括驱动信息、几何范围及投影系统。
    常用于 SBAS-InSAR 流程中对输入干涉图堆栈的预检查。

    Args:
        dataset (gdal.Dataset): 已打开的 GDAL 数据集对象。

    Returns:
        dict: 包含以下字段的元数据字典:
            - Driver (str): 驱动简称与全称。
            - Size (tuple): 图像维度 (xsize, ysize, bands)。
            - Projection (str): WKT 格式的投影字符串。
            - Origin (tuple/None): 仿射变换原点坐标 (x, y)。
            - Pixel Size (tuple/None): 像元分辨率 (dx, dy)。
    """
    if not dataset:
        raise ValueError("Provided dataset is None or invalid.")

    # 提取驱动与基本属性
    driver_info = f"{dataset.GetDriver().ShortName}/{dataset.GetDriver().LongName}"
    size = (dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount)
    proj = dataset.GetProjection()
    
    # 提取地理变换信息
    gt = dataset.GetGeoTransform()
    origin = (gt[0], gt[3]) if gt else None
    pixel_size = (gt[1], gt[5]) if gt else None

    # 控制台输出 (用于调试)
    # print(f"Driver: {driver_info}")
    # print(f"Size: {size[0]} x {size[1]} x {size[2]}")
    # print(f"Projection: {proj}")
    # print(f"Pixel Size: {pixel_size}")
    # print(f"Origin: {origin}")

    return {
        'Driver': driver_info,
        'Size': size,
        'Projection': proj,
        'Origin': origin,
        'Pixel Size': pixel_size
    }
