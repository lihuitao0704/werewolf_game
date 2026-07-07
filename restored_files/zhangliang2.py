#torch.tensor  创建张量

import torch
import numpy as np

def test01():

    data = torch.arange(0,10,2) #创建张量 
    print(data)

    data1 =torch.linspace(0,11,10)
    print(data1)



def test02():

    data = torch.randn(2,3)   #2行3列的二维张量，float形式,随机张量
    print(data)

    torch.manual_seed(100)
    data = torch.randn(2,3)   #2行3列的二维张量，float形式,随机张量
    print(data)

    torch.manual_seed(100)
    data = torch.randn(2,3)   #2行3列的二维张量，float形式,随机张量
    print(data)

    #initial_seed() ： 获取默认的随机种子
    torch.manual_seed(20)
    print("随机种子：",torch.random.initial_seed())



def test03():
    #创建0和1 的张量
    data = torch.zero(2,4)
    print(data)
    

if __name__ == '__main__':
    test02()