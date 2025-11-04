from data_loader import RCV1DataLoader
from model import DNNClassifier, MultiLabelClassifier
from evaluator import MultiLabelEvaluator
import numpy as np
import warnings
import torch

data_loader = RCV1DataLoader(data_dir='data')

x, y = data_loader.load_data('train')
print(x.shape)
print(y.shape)

print("\nFirst sample features:")
print(x[1])


print("\nFirst sample labels:")
print(y[1])