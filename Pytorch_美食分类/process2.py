# 将数据进行划分8:2
import os
import random
import shutil

# 数据文件夹路径
data_folder = './data'

# 训练集和验证集比例
train_ratio = 0.8
validation_ratio = 0.2

# 遍历数据文件夹下的每个子文件夹
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    # 如果是文件夹
    if os.path.isdir(folder_path):
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)

        # 随机打乱文件顺序
        random.shuffle(files)

        # 计算训练集和验证集的分界索引
        split_index = int(len(files) * train_ratio)

        # 分割数据
        train_files = files[:split_index]
        validation_files = files[split_index:]

        # 创建训练集和验证集文件夹
        train_folder = os.path.join(data_folder, 'train', folder_name)
        validation_folder = os.path.join(data_folder, 'validation', folder_name)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(validation_folder, exist_ok=True)

        # 将文件复制到相应的文件夹中
        for file in train_files:
            shutil.copy(os.path.join(folder_path, file), os.path.join(train_folder, file))

        for file in validation_files:
            shutil.copy(os.path.join(folder_path, file), os.path.join(validation_folder, file))

print("数据划分完成！")
