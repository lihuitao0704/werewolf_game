#torch.tensor  创建张量

import torch
import numpy as np

def test01():
    data = torch.tensor(10) #创建张量 
    print(data)

    print('-'*20)

    data1 = np.random.randn(2,3)
    print(data1)
    data1 = torch.tensor(data1)
    print(data1)

    data = [[10.0,20.0,30.0],[40.0,50.0,60.0]]
    data = torch.tensor(data)
    print(data)



def test02():
    data = torch.Tensor(2,3)   #2行3列的二维张量，float形式
    print(data)
    data = torch.Tensor([10])
    print(data)
    data = torch.Tensor([10,20])
    print(data)

def test03():
    data = torch.IntTensor(2,3)
    print(data)
    data = torch.IntTensor([2.5,3.3])
    print(data)
    data = torch.ShortTensor()

    print(data)



if __name__ == '__main__':
    test03()