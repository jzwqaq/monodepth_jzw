# -*-coding:utf-8-*-
import torch

a = torch.randint(0, 1, (3, 4))
print(a.shape)
print(a)
b = a.reshape(1, 1, 3, 4)
print(b.shape)
print(b)
print(a)