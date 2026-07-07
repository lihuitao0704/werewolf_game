#torch.tensor  创建张量

import torch
import numpy as np

def test01():
    #手动生成
    ref_tensor = torch.tensor([[1,23,4],[2,42,5]])
    zeros_like = torch.zeros_like(ref_tensor)
    print(zeros_like)


    #随机生成
    rand_tensor = torch.rand(2, 3)
    print(rand_tensor)


    tensor = torch.tensor()
    # 数据类型转换：转为浮点型
    float_tensor = tensor.to(torch.float32)  # 或使用 tensor.float()
    print(float_tensor)


def test02():

    # 1. 创建需要求导的张量
    x = torch.tensor(3.0, requires_grad=True)

    # 2. 定义计算过程（前向传播）: y = x^2
    y = x ** 2 

    # 3. 触发反向传播（计算 dy/dx）
    y.backward()  

    # 4. 查看梯度
    print(x.grad)  # 输出: tensor(6.) 
    # 数学验证: dy/dx = 2x, 当 x=3 时，2*3 = 6



def test03():
    #模拟真实的模型训练循环（手动更新参数）
    # 1. 初始化权重和偏置，并开启自动求导
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    learning_rate = 0.1

    print("=== 1. 初始参数 ===")
    print(f"w = {w.item()}, b = {b.item()}")

    # 2. 前向传播：计算预测值
    y_pred = w * 3.0 + b      
    print("\n=== 2. 前向传播 ===")
    print(f"预测值 y_pred = {y_pred.item()}")  # 2.0 * 3.0 + 1.0 = 7.0

    # 3. 计算损失：假设真实目标值是 8.0
    loss = (y_pred - 8.0)**2  
    print("\n=== 3. 计算损失 ===")
    print(f"Loss = {loss.item()}")  # (7.0 - 8.0)^2 = 1.0

    # 4. 反向传播：计算梯度
    loss.backward()           
    print("\n=== 4. 反向传播计算梯度 ===")
    print(f"w 的梯度 (dw) = {w.grad.item()}")  # 2 * (7.0 - 8.0) * 3.0 = -6.0
    print(f"b 的梯度 (db) = {b.grad.item()}")  # 2 * (7.0 - 8.0) * 1.0 = -2.0

    # 5. 更新参数（必须用 torch.no_grad() 包裹）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print("\n=== 5. 参数更新 ===")
    print(f"更新后的 w = {w.item()}")  # 2.0 - 0.1 * (-6.0) = 2.6
    print(f"更新后的 b = {b.item()}")  # 1.0 - 0.1 * (-2.0) = 1.2

    # 6. 清空梯度（极其重要！）
    w.grad.zero_()
    b.grad.zero_()

    print("\n=== 6. 梯度清零 ===")
    print(f"清零后 w 的梯度 = {w.grad}")  # 输出: None 或 tensor(0.)
    print(f"清零后 b 的梯度 = {b.grad}")

if __name__ == '__main__':
    test03()