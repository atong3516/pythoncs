# import torch
# import torch.nn as nn
# import torchvision.models as models
#
#
# class ImageRNNCNN(nn.Module):
#     def __init__(self, input_dim=50176, hidden_dim=100, layer_dim=2, output_dim=11):
#         """
#         初始化ImageRNNCNN模型，结合ResNet50特征提取与RNN分类器
#         """
#         super(ImageRNNCNN, self).__init__()
#         # ResNet50 提取特征
#         resnet50 = models.resnet50(pretrained=True)  # 修改为加载ResNet50预训练模型
#         self.features = nn.Sequential(*list(resnet50.children())[:-2])  # 移除ResNet最后的全连接层及平均池化层
#
#         # 以下是RNN的属性
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#
#         self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
#         # 注意：由于ResNet50输出特征维度可能与ResNet18不同，需要根据实际情况调整input_dim
#         # 假设ResNet50最后一层卷积输出特征维度为2048（具体数值需根据实际模型输出确定）
#         self.input_dim = 2048  # 请根据ResNet50的实际输出调整此值
#
#         self.fc1 = nn.Linear(hidden_dim, output_dim)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def forward(self, x):
#         # ResNet50 提取特征
#         x = self.features(x)
#         # 注意调整ResNet50输出的展平方式，假设最后一层输出尺寸为7x7x2048（具体根据ResNet50的输出调整）
#         x = x.view(x.size(0), 1, -1)
#
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
#         self.rnn.flatten_parameters()
#
#         # 分类隐藏状态，避免梯度爆炸
#         out, hn = self.rnn(x, h0.detach())
#         out = self.fc1(out[:, -1, :])
#
#         return out

import torch
import torch.nn as nn
import torchvision.models as models


class ImageRNNCNN(nn.Module):
    def __init__(self, input_dim=50176, hidden_dim=100, layer_dim=2, output_dim=11):
        """
        初始化ImageRNNCNN模型，结合ResNet18特征提取与RNN分类器
        input_dim = 44944  # 输入维度(输入的节点数量)
        hidden_dim = 100  # 隐藏层的维度(每个隐藏层的节点数)
        layer_dim = 2  # 2层RNN(隐藏层的数量 2层)
        out_dim = 2  # 输出维度
        """
        super(ImageRNNCNN, self).__init__()
        # resnet18 提取特征
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-3])  # 移除ResNet最后的全连接层

        # 以下是RNN的属性
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        """
            batch_first：当 batch_first设置为True时，输入的参数顺序变为：
            x：[batch, seq_len, input_size]，
            h0：[batch, num_layers, hidden_size]。
        """
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # restnet18 提取特征
        x = self.features(x)
        x = x.view(x.size(0), 1, -1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        self.rnn.flatten_parameters()

        # 分类隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc1(out[:, -1, :])  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，

        return out
