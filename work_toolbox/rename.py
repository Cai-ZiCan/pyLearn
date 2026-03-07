# 用于实现对一些文件的重命名功能，逻辑：根据原始文件中的学号，进行匹配，找到对应的姓名和班级信息，然后按照指定的格式进行重命名。
# input:
# 1. 班级人员名单excel，包括班级、姓名、学号，
# 2. 需要重命名的文件所在的目录
# 3. 重命名后，输出文件所在的目录
# 4. 学号的长度，默认是10位，通过匹配10位数字来提取学号
# 5. 指定的后缀名字
# 6. 指定的重命名后 各名字组分 的排列顺序，reorder_list = [班级，姓名，学号,后缀名字](Default)
# output:
# 1. 将文件重命名为：指定的名字组分的排列顺序，班级-姓名-学号-后缀名字
import glob
import os
import shutil
import pandas as pd

def extract_student_id(filename, id_length=10):
    """从文件名中提取学号，默认学号长度为10位
    Args:
        filename: 文件名字符串
        id_length: 学号的长度，默认是10位
    
    Returns:
        学号字符串，如果未找到则返回None
    """
    import re
    pattern = r'\d{' + str(id_length) + r'}'
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    return None

def generate_new_filename(class_name, student_name, student_id, suffix_name, reorder_list):
    """生成新的文件名，按照指定的名字组分的排列顺序
    Args:
        class_name: 班级名称
        student_name: 学生姓名
        student_id: 学号
        suffix_name: 指定的后缀名字
        reorder_list: 指定的重命名后 各名字组分 的排列顺序
    
    Returns:
        新的文件名字符串
    """
    name_components = {
        '班级': class_name,
        '姓名': student_name,
        '学号': student_id,
        '后缀名字': suffix_name
    }
    new_filename = '-'.join([name_components[component] for component in reorder_list])
    return new_filename

def rename_files(excel_path, input_dir, output_dir, suffix_name, reorder_list=['班级', '姓名', '学号', '后缀名字'], match_keywords=None,pattern="*.docx"):
    """docstring for rename_files
    Args:
        excel_path: 班级人员名单excel路径
        input_dir: 需要重命名的文件所在的目录
        output_dir: 重命名后，输出文件所在的目录
        suffix_name: 指定的后缀名字
        reorder_list: 指定的重命名后 各名字组分 的排列顺序，Default = [班级，姓名，学号,后缀名字]
        match_keywords: 额外匹配的字符串列表，默认已包含学号匹配。如需匹配特定文件（如“报告”），可在此添加。
        pattern: 文件匹配模式，默认为"*.docx"，可修改为其他文件类型，例如"*.pdf"或"*.*"（表示所有文件）。
    
    Returns:
        None
    """
    # 1. 读取班级人员名单excel
    df = pd.read_excel(excel_path)
    # 强制将学号列转换为字符串类型，并去除首尾空格，解决因数据类型(int vs str)不一致导致的匹配失败问题
    if '学号' in df.columns:
        df['学号'] = df['学号'].astype(str).str.strip()
    
    # 2. 遍历需要重命名的文件所在的目录，递归查找所有文件
    search_path = os.path.join(input_dir, "**", pattern) # 查找doc后缀的文件，若需要查找其他后缀的文件，可以修改这里的'*.docx'为相应的模式，例如'*.pdf'或'*.*'（表示所有文件） 
    files = glob.glob(search_path, recursive=True)
    for filename in files:
        # 2.1 检查是否满足额外的匹配关键字
        if match_keywords:
            if isinstance(match_keywords, str):
                keywords = [match_keywords]
            else:
                keywords = match_keywords
            
            # 如果文件名不包含所有指定的关键字，则跳过
            if not all(keyword in filename for keyword in keywords):
                continue

        # 3. 提取文件名中的学号
        student_id = extract_student_id(filename)
        if student_id is None:
            print(f"无法从文件名 {filename} 中提取学号，跳过该文件。")
            continue
        # 4. 根据学号在班级人员名单中找到对应的姓名和班级信息
        student_info = df[df['学号'] == student_id]
        if student_info.empty:
            print(f"在班级人员名单中未找到学号 {student_id}，跳过该文件。")
            continue
        class_name = student_info['班级'].values[0]
        student_name = student_info['姓名'].values[0]
        # 5. 按照指定的格式进行重命名
        new_filename = generate_new_filename(class_name, student_name, student_id, suffix_name, reorder_list)
        
        # 获取原始文件的扩展名
        _, ext = os.path.splitext(filename)
        new_filename_with_ext = new_filename + ext
        
        # 6. 将文件重命名并保存到输出目录,不从源文件修改，而是复制到输出目录并重命名，保持源文件不变
        new_filepath = os.path.join(output_dir, new_filename_with_ext)
        shutil.copy2(filename, new_filepath)  # 使用shutil.copy2复制文件并保留元数据
        print(f"已将文件 {filename} 复制并重命名为 {new_filepath}。")

# 示例用法
# excel文件内部格式如下：
# 班级    姓名    学号
# 1班    张三    1234567890

if __name__ == "__main__":
    excel_path = r"D:\\pyLearn\\work_toolbox\\test_data\\测绘2301班级名单.xlsx"  # 班级人员名单excel路径
    input_dir = r"D:\\GIS课程设计"  # 需要重命名的文件所在的目录
    output_dir = r"test_result\\output_files"  # 重命名后，输出文件所在的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果输出目录不存在，则创建该目录
    suffix_name = "作业1"  # 指定的后缀名字
    reorder_list = ['班级', '姓名', '学号', '后缀名字']  # 指定的重命名后 各名字组分 的排列顺序
    
    # 可选：指定匹配关键字，例如只处理包含“课程设计报告”的文件
    match_keywords = "课设报告" 
    pattern = "*.docx"  # 只处理docx后缀的文件，若需要处理其他后缀的文件，可以修改这里的模式，例如"*.pdf"或"*.*"（表示所有文件）
    # match_keywords = ["课程设计报告"] 

    rename_files(excel_path, input_dir, output_dir, suffix_name, reorder_list, match_keywords,pattern)