import os
import glob

def search_files(input_dir, pattern="*.tif", recursive=False):
    """
    在指定目录中搜索符合模式的文件。
    (原 find_files，重命名为 search_files 以更贴切)

    Args:
        input_dir (str): 搜索的根目录路径。
        pattern (str): 文件匹配模式，例如 "*.tif" 或 "*.jp2"。
        recursive (bool): 是否递归搜索子目录。默认为 False。

    Returns:
        list: 匹配到的文件完整路径列表。
    """
    if recursive:
        search_path = os.path.join(input_dir, "**", pattern)
        files = glob.glob(search_path, recursive=True)
    else:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
    
    if not files:
        print(f"提示: 在 {input_dir} 中未找到匹配 {pattern} 的文件。")
    
    return files
