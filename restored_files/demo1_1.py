# 早停机制版本
# pip install torch torchvision
# pip uninstall torch torchvision -y
# pip uninstall torch torchvision  -y
# pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu128
# pip install --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/cu128

# import sys
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# =========================
# 0️⃣ 设置工作目录
# =========================
# import os

# 切换到脚本所在目录，解决相对路径问题
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
# print(f"当前工作目录: {os.getcwd()}")

# =========================
# 1️⃣ 导入库
# =========================
import torch  # PyTorch核心库

print(torch.__version__)

print("CUDA可用:", torch.cuda.is_available())
import torch.nn as nn  # 神经网络模块（层、模型）
import torch.optim as optim  # 优化器模块（如Adam）

from torchvision import datasets, transforms  # 数据集 & 图像预处理
from torch.utils.data import DataLoader  # 数据加载器（批量读取数据）

# =========================
# 2️⃣ 数据预处理
# =========================
transform = transforms.Compose(
    [
        # 还不是以0为中心  能训练 但是不稳不够快
        transforms.ToTensor(),  # 将图片转换为Tensor， 会把图片从 [0, 255] 的像素值转换为 [0, 1] 的浮点数
        # 新值 = (原值 - 0.5) / 0.5
        transforms.Normalize((0.5,), (0.5,)),  # 标准化到[-1,1]，提升训练稳定性
    ]
)

# =========================
# 3️⃣ 加载数据集（MNIST手写数字）
# =========================
train_dataset = datasets.MNIST(
    root="./data",  # 数据存储路径
    train=True,  # 是否为训练集
    download=True,  # 如果没有就自动下载
    transform=transform,  # 应用预处理
)

test_dataset = datasets.MNIST(
    root="./data",  # 数据存储路径
    train=False,  # 测试集
    download=True,
    transform=transform,
)

# =========================
# 4️⃣ 数据加载器（按批读取）
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # 每批64张图片
    shuffle=True,  # 打乱数据（防止模型记忆顺序）
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,  # 测试时一次读更多，加快速度
    shuffle=False,  # 测试不需要打乱
)


# =========================
# 5️⃣ 定义卷积神经网络
# =========================
class SimpleCNN(nn.Module):  # 继承nn.Module
    def __init__(self):
        super(SimpleCNN, self).__init__()  # 初始化父类
        # ===== 卷积层 =====
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入通道（灰度图=1） 彩色图片（生活中常见）其实是三层叠在一起的：红（R）绿（G） 蓝（B）  in_channels = 3
            out_channels=16,  # 输出通道（16个特征图）  从一张图片中提取出 16种不同的特征表示
            kernel_size=3,  # 卷积核大小3x3
            padding=1,  # 填充，保持尺寸不变  不做膨胀卷积
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        # ===== 池化层 =====
        # 在保留重要特征的同时，把图片变小（降维）
        # 池化层做的事情就是：“缩小图片，但尽量保留最重要的信息”
        # stride（步长）= 卷积核 / 池化窗口 每次移动的“步子大小”
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2  # 2x2池化  # 步长2（尺寸减半）
        )
        # ===== 全连接层 =====
        # 7 * 7 是知道要经过两次池化 得到的内容的尺寸
        # 32 * 7 * 7 ===>相当于 32个 7*7的方块
        # 把 1568 个特征“压缩并组合”成 128 个更有代表性的特征
        self.fc1 = nn.Linear(
            32 * 7 * 7, 128  # 输入：卷积后展开的特征  # 输出：128个神经元
        )
        self.fc2 = nn.Linear(128, 10)  # 输入  # 输出类别数（0~9）
        # ===== 激活函数 =====
        self.relu = nn.ReLU()  # 非线性激活

    # =========================
    # 前向传播（数据流动路径）
    # =========================
    def forward(self, x):
        # 第1层：卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 输入：[B,1,28,28]
        # 输出：[B,16,14,14]

        # 第2层：卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 输出：[B,32,7,7]
        # 展平（Flatten）
        # -1 的意思是：让 PyTorch 自动帮你推断这一维的大小。
        x = x.view(-1, 32 * 7 * 7)  # 把每个样本的转成一维向量
        # 全连接层1
        x = self.relu(self.fc1(x))
        # 输出层（分类结果）
        x = self.fc2(x)
        return x


# =========================
# 6️⃣ 初始化模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU
model = SimpleCNN().to(device)  # 模型放到设备上（CPU/GPU）
# =========================
# 7️⃣ 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 要优化的参数  # 学习率


# =========================
# 8️⃣ 训练函数
# =========================
import copy


def train():
    count = 0  # 记录连续没有超过最小损失的论述
    # 损失初始化是 正无穷大
    min_loss = float("inf")
    epoch = 0  # 记录训练的轮次
    model.train()  # 切换到训练模式
    # 初始化一个模型到内存
    best_model_wts = copy.deepcopy(model.state_dict())
    while count < 5:
        epoch += 1  # 轮次累加
        total_loss = 0  # 记录每一轮的总损失
        for data, target in train_loader:
            # 数据搬到设备（GPU/CPU）
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 梯度清零（非常重要）
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            # 梯度就是损失函数对模型参数的偏导数，用来指导参数往让误差变小的方向更新。
            loss.backward()  # 反向传播（计算梯度）
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
        # min_loss 第一轮的时候 是无穷大
        if total_loss >= min_loss:
            count += 1
        else:  # 总损失小于 最小损失  则说明模型还可以进步
            count = 0
            min_loss = total_loss  # 跟新最小损失
            # ⭐ 保存到内存（不是文件）
            best_model_wts = copy.deepcopy(model.state_dict())
            print("✅ 更新最优模型（内存中）")

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    print(f"在第  {epoch-5}, 轮，触发早停: {min_loss:.4f}")
    torch.save(best_model_wts, "mnist_cnn_v001.pth")


# =========================
# 9️⃣ 测试函数  新增函数
# =========================
def test():
    model = SimpleCNN().to(device)
    # 🔥 加载训练好的模型
    model.load_state_dict(torch.load("mnist_cnn_v001.pth", map_location=device))
    model.eval()  # 切换到评估模式
    correct = 0  # 记录预测正确数量
    with torch.no_grad():  # 关闭梯度（节省内存 & 提速）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 前向传播
            pred = output.argmax(dim=1)  # 取概率最大类别
            correct += pred.eq(target).sum().item()  # 统计正确数
    print(f"Accuracy: {correct / len(test_dataset):.4f}")


from PIL import Image  # 用来读取图片


# =========================
# 🔥 新增：单张图片预测函数
# =========================
def predict_image(image_path):
    model = SimpleCNN().to(device)
    # 🔥 加载训练好的模型
    model.load_state_dict(torch.load("mnist_cnn_v001.pth", map_location=device))
    model.eval()  # 推理模式
    # 1️⃣ 读取图片（灰度图）
    img = Image.open(image_path).convert("L")
    # 2️⃣ 调整尺寸（必须28x28！）
    img = img.resize((28, 28))
    # 3️⃣ 应用和训练时一样的transform
    img = transform(img)

    print(f"img", img.shape)

    # 4️⃣ 增加batch维度 [1,1,28,28]
    img = img.unsqueeze(0).to(device)
    # 5️⃣ 推理
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    print(f"预测结果是：{pred}")


# =========================
# 🔟 主函数入口
# =========================
if __name__ == "__main__":
    # train()  # 训练模型
    # test()  # 测试模型
    predict_image("ai_study/images/8.jpg")
    # num = 0
    # for data, target in train_loader:
    #     num += 1
    #     print(data)
    #     print(data.shape)
    #     if num == 1:
    #         break
