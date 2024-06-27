import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import ImageRNNCNN
import time
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 数据预处理:数据转换和加载数据集
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# 设置批次大小和数据路径
batch_size = 32
data_folder = './data/train'
# 数据加载,使用ImageFolder加载训练集和验证集，并通过DataLoader生成批量数据
train_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

data_folder = './data/validation'
val_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ImageRNNCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #随机梯度下降

# 训练循环模型
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
start_time = time.time()

# 初始化最佳准确率和最佳模型状态字典
epoch_acc_best = 0
for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs) # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
    # 计算训练阶段的平均损失和准确率
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions / len(train_dataset)

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    print(f'Epoch Train {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    # 验证
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = correct_predictions / len(val_dataset)

        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        print(f'Epoch Val {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        # 保存表现最好的模型
        if epoch_acc > epoch_acc_best:
            epoch_acc_best = epoch_acc
            model_state_dict_best = model.state_dict()
            print(f"更新了模型，{epoch_acc_best:.4f}")

print(f"训练时间：{time.time() - start_time}")

# 保存模型
torch.save(model_state_dict_best, './weights/RNN.pth')

# 绘制acc和loss曲线
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("./images/RNN.png", dpi=300)
plt.show()
