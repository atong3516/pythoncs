import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog, \
    QDialog, QDialogButtonBox, QVBoxLayout, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image, ImageQt
import torch
from torchvision import transforms
from model import ImageRNNCNN
import os  # 导入os模块


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
        layout = QVBoxLayout()

        upload_button = QPushButton('上传')
        upload_button.clicked.connect(self.upload_image)

        recognize_button = QPushButton('识别')
        recognize_button.clicked.connect(self.recognize_digit)

        layout.addWidget(self.image_label)
        layout.addWidget(upload_button)
        layout.addWidget(recognize_button)
        layout.addWidget(self.history_textedit)

        self.setLayout(layout)
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
                6: "面条/意大利面 (Noodles/Pasta)",
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
            history_item = f"图片：{file_name}\n属于：{predicted_class}\n"
            self.history_list.append(history_item)

            # 更新历史记录文本框
            self.history_textedit.append(history_item)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = QApplication(sys.argv)
    main_window = HandwrittenDigitRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())
