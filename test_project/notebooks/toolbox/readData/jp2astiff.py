# 用于实现将jp2格式的哨兵2数据转换为tiff格式的功能
import rasterio
from toolbox.raster import stack_bands

def jp2astiff(jp2_paths, tiff_path):
    """
    将多个JP2格式的哨兵2数据转换为一个多波段的GeoTIFF文件。
    
    Args:
        jp2_paths (list): 包含JP2文件路径的列表。列表顺序决定输出文件的波段顺序。
        tiff_path (str): 输出的多波段GeoTIFF文件路径。
    Returns:
        bool: 成功返回True，失败返回False。
    """
    try:
        # 1. 使用 rasterio 读取 JP2 文件并获取其路径列表
        # 这里假设 jp2_paths 已经是一个包含所有需要转换的 JP2 文件路径的列表
        input_files = jp2_paths
        # 2. 调用 stack_bands 函数将 JP2 文件合并为一个多波段 GeoTIFF 文件
        success = stack_bands(input_files, tiff_path)
        
        if success:
            print(f"成功将 {len(input_files)} 个 JP2 文件转换为 {tiff_path}")
            return True
        else:
            print("转换失败，请检查输入文件和输出路径。")
            return False
            
    except Exception as e:
        print(f"发生异常: {str(e)}")
        return False


    
    