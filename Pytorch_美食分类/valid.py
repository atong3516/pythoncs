import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import ImageRNNCNN  # 确保你的 ImageRNNCNN 模型文件路径正确
from tqdm import tqdm
import numpy as np

# 数据转换和加载测试数据集
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_folder = './data/validation'
test_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 模型加载与配置
model = ImageRNNCNN()

# 加载预训练模型权重
model_weights_path = './weights/RNN.pth'
model.load_state_dict(torch.load(model_weights_path,map_location='cpu'))
model.eval()

# 损失函数
criterion = nn.CrossEntropyLoss()

# 在测试集上进行预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 结果收集与评估
all_labels = []
all_predictions = []
all_losses = []
error_indices = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Validation', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        loss = criterion(outputs, labels)
        all_losses.append(loss.item())

        # 记录错误分类的索引
        for i in range(len(labels)):
            if labels[i] != predicted[i]:
                error_indices.append((inputs[i].cpu(), labels[i].cpu().item(), predicted[i].cpu().item()))

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("./images/RNN_Confusion.png", dpi=300)
plt.show()

# 输出分类报告
class_report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes, digits=4)
print("Classification Report:\n", class_report)

# 计算并打印各项指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 平均损失
average_loss = np.mean(all_losses)
print("Average Loss:", average_loss)

# 输出部分错误分类的数据
print(f"Number of misclassified samples: {len(error_indices)}")
print("Sample Misclassifications (Image, True Label, Predicted Label):")
for i, (img, true, pred) in enumerate(error_indices[:10]):  # 输出前10个错误分类的数据
    img = img.permute(1, 2, 0)  # 转换为 HWC 格式
    plt.figure()
    plt.imshow(img)
    plt.title(f'True: {test_dataset.classes[true]}, Predicted: {test_dataset.classes[pred]}')
    plt.axis('off')
    plt.savefig(f"./images/misclassified_{i}.png", dpi=300)
    plt.show()

# 可选：将错误分类数据保存到文件
with open("error_classifications.txt", "w") as file:
    for img, true, pred in error_indices:
        file.write(f"True: {test_dataset.classes[true]}, Predicted: {test_dataset.classes[pred]}\n")
