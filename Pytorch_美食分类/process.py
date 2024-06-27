import os
import re
import shutil

##########################################################################
# 创造文件夹
# for i in range(0,11):
#     os.mkdir(f"./data/{i}")

##########################################################################
# # 训练文件移动
# folder_path = './food-11/training'
#
# # 初始化类别数量字典
# category_count = {}
#
# # 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     # 使用正则表达式提取类别信息
#     match = re.match(r'(\d+)_(\d+)\.jpg', filename)
#
#     if match:
#         # 提取到类别和编号信息
#         category = match.group(1)
#         number = match.group(2)
#
#         # 更新类别数量字典
#         category_count[category] = category_count.get(category, 0) + 1
#
#         # 移动文件
#         source_path = os.path.join("./food-11/training", filename)
#         target_path = os.path.join(f"./data/{category}", filename)
#
#         # 使用shutil.move移动文件
#         shutil.move(source_path, target_path)
#
# # 打印类别数量
# for category, count in category_count.items():
#     print(f'类别 {category}: {count} 个文件')

##########################################################################
# # 验证文件移动
# folder_path = './food-11/validation'
#
# # 初始化类别数量字典
# category_count = {}
#
# # 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     # 使用正则表达式提取类别信息
#     match = re.match(r'(\d+)_(\d+)\.jpg', filename)
#
#     if match:
#         # 提取到类别和编号信息
#         category = match.group(1)
#         number = match.group(2)
#
#         # 更新类别数量字典
#         category_count[category] = category_count.get(category, 0) + 1
#
#         # 移动文件
#         source_path = os.path.join("./food-11/validation", filename)
#         target_path = os.path.join(f"./data/{category}", f"{category}_{int(number) + 2000}.jpg")
#
#         # 使用shutil.move移动文件
#         shutil.move(source_path, target_path)
#
# # 打印类别数量
# for category, count in category_count.items():
#     print(f'类别 {category}: {count} 个文件')

##########################################################################
# 修改文件夹名
# data_folder_path = './data'
#
# # 映射数字到文字的字典
# category_mapping = {
#     '0': 'Bread',
#     '1': 'Dairy product',
#     '2': 'Dessert',
#     '3': 'Egg',
#     '4': 'Fried food',
#     '5': 'Meat',
#     '6': 'Noodles',
#     '7': 'Rice',
#     '8': 'Seafood',
#     '9': 'Soup',
#     '10': 'Vegetable'
# }
#
# for category_old, category_new in category_mapping.items():
#     print(f'旧类别名称: {category_old}, 新类别名称: {category_new}')
#
#     folder_path = os.path.join("./data", category_old)
#     new_folder_path = os.path.join("./data", category_new)
#     os.rename(folder_path, new_folder_path)

##########################################################################
# 统计数据
data_folder_path = './data'

# 遍历data文件夹下的所有子文件夹
for folder_name in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder_name)

    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 统计文件夹中的文件个数
        file_count = len(os.listdir(folder_path))

        print(f'文件夹 {folder_name}: {file_count} 个文件')