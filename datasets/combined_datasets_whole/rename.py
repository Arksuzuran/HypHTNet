import os

# # 设置目录路径
# dir_path = './'
#
# # 遍历目录下的所有文件
# for filename in os.listdir(dir_path):
#     # 检查文件名是否符合要求
#     if filename.endswith(' .jpg'):
#         # 构造新的文件名
#         new_filename = filename.replace(' .jpg', '.jpg')
#         # 构造完整的文件路径
#         old_file_path = os.path.join(dir_path, filename)
#         new_file_path = os.path.join(dir_path, new_filename)
#         # 重命名文件
#         os.rename(old_file_path, new_file_path)
#         print(f'文件{filename}已重命名为{new_filename}')

# 设置目录路径
dir_path = './'

for i in range(1, 10):
    # 遍历目录下的所有文件
    for filename in os.listdir(dir_path):
        # 检查文件名是否符合要求
        if filename.startswith(f's0{i}_s') and filename.endswith('.jpg'):
            # 构造新的文件名，在第二个s后面加上0
            parts = filename.split('_')
            if len(parts[1]) == 2:  # 判断第二个部分是否为单个数字
                new_filename = f'{parts[0]}_s0{parts[1][1]}_{parts[2]}_{parts[3]}'
                # 构造完整的文件路径
                old_file_path = os.path.join(dir_path, filename)
                new_file_path = os.path.join(dir_path, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'文件{filename}已重命名为{new_filename}')