import pandas as pd
import torch
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

data = data.apply(pd.to_numeric, errors = 'coerce') #any value not a number gets turned into NaN value

#debug statemnt for progress
print(data)

#converts pd array into pytorch array, ready to be used for machine learning
x = torch.Tensor(data.values[:,1]).unsqueeze(dim=1)
y = torch.Tensor(data.values[:,1]).unsqueeze(dim=1)

#debug statement for progress
print(x, y)