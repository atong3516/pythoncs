import os  # 导入os模块
import sys

import torch
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QLabel, QFileDialog, \
    QGridLayout, QVBoxLayout, QHBoxLayout
from torchvision import transforms
# 自定义的深度学习模型，用于图像识别
from model import ImageRNNCNN


class HandwrittenDigitRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model = ImageRNNCNN()
        checkpoint = torch.load('./weights/RNN.pth',map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(device)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(280, 280)

        self.history_textedit = QTextEdit()
        self.history_textedit.setReadOnly(True)

        self.history_list = []  # 用于保存历史记录的列表

        self.init_ui()

    def init_ui(self):
        # 使用垂直布局管理主窗口
        main_layout = QVBoxLayout(self)

        # 设置窗口背景颜色
        self.setStyleSheet("background-color: #f4f4f9;")

        # 图片显示区域
        self.image_label.setFixedSize(400, 400)  # 调整图片框大小
        self.image_label.setStyleSheet("border: 1px solid #ddd; border-radius: 10px;")  # 边框和圆角

        # 历史记录文本框
        self.history_textedit.setStyleSheet(
            "font-size: 16px; color: #333; border-radius: 10px; background-color: #ffffff;")

        # 设置按钮样式
        button_style = """
        QPushButton {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                             stop: 0 #6dd5ed, stop: 1 #0081e6);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #0081e6;
        }
        """

        # 上传按钮
        upload_button = QPushButton('打开文件')
        upload_button.setIcon(QIcon.fromTheme('document-open'))
        upload_button.setStyleSheet(button_style)
        upload_button.clicked.connect(self.upload_image)

        # 识别按钮
        recognize_button = QPushButton('识别美食')
        recognize_button.setIcon(QIcon.fromTheme('edit-select'))
        recognize_button.setStyleSheet(button_style)
        recognize_button.clicked.connect(self.recognize_digit)

        # 上部布局：左侧为图片显示区域，右侧为历史记录文本框
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.image_label, stretch=2)  # 图片区域占2份空间
        upper_layout.addWidget(self.history_textedit, stretch=1)  # 历史记录占1份空间

        # 底部布局：放置按钮
        button_layout = QHBoxLayout()
        button_layout.addWidget(upload_button)
        button_layout.addWidget(recognize_button)

        # 将布局添加到主布局
        main_layout.addLayout(upper_layout)
        main_layout.addLayout(button_layout)

        # 设置窗口标题
        self.setWindowTitle('美食分类应用')

    def upload_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '打开图片', '', '图像文件 (*.png *.jpg *.bmp)')

        if image_path:
            pixmap = QPixmap(image_path)

            # 获取self.image_label的大小
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            # 缩放pixmap以适应self.image_label
            scaled_pixmap = pixmap.scaled(label_width, label_height, aspectRatioMode=Qt.KeepAspectRatio)

            # 设置缩放后的pixmap
            self.image_label.setPixmap(scaled_pixmap)

            # 保存图片路径以便后续识别时使用
            self.current_image_path = image_path

    def recognize_digit(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path)
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
            image = transform(image).unsqueeze(0).to(device)  # 添加批次维度

            # 使用加载的模型进行预测
            with torch.no_grad():
                output = self.model(image)

            # 定义类别映射字典
            class_mapping = {
                0: "面包 (Bread)",
                1: "乳制品 (Dairy product)",
                2: "甜点 (Dessert)",
                3: "鸡蛋 (Egg)",
                4: "油炸食品 (Fried food)",
                5: "肉类 (Meat)",
                6: "面条 (Noodles)",
                7: "米饭 (Rice)",
                8: "海鲜 (Seafood)",
                9: "汤 (Soup)",
                10: "蔬菜/水果 (Vegetable/Fruit)"
            }

            # 模型输出的预测索引
            predicted_digit = torch.argmax(output).item()

            # 从映射中获取类别
            predicted_class = class_mapping[predicted_digit]

            # 获取文件名部分而不是整个路径
            file_name = os.path.basename(self.current_image_path)

            # 更新历史记录列表
            history_item = f"文件名：{file_name}\n美食分类：{predicted_class}\n"
            self.history_list.append(history_item)

            # 更新历史记录文本框
            self.history_textedit.append(history_item)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = QApplication(sys.argv)
    main_window = HandwrittenDigitRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())
