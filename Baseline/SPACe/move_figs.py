import os
import shutil

def move_files():
    # 获取当前目录
    current_dir = '/data/boom/cpg0001/2020_08_11_Stain3_Yokogawa/images/BR00115125'
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    # 定义目标文件夹路径
    target_dir = os.path.join(parent_dir, "others")
    
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历当前目录中的所有文件
    for filename in os.listdir(current_dir):
        # 获取文件的完整路径
        file_path = os.path.join(current_dir, filename)
        
        # 检查是否是文件且不以 "BR00115125" 开头
        if os.path.isfile(file_path) and not filename.startswith("BR00115125"):
            # 构造目标路径
            target_path = os.path.join(target_dir, filename)
            # 移动文件
            shutil.move(file_path, target_path)
            print(f"Moved: {file_path} -> {target_path}")

if __name__ == "__main__":
    move_files()