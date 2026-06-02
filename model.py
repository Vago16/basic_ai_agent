import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn

data = pd.read_csv('Position_Salaries.csv')

data = data.apply(pd.to_numeric, errors = 'coerce') #any value not a number gets turned into NaN value

#debug statemnt for progress
print(data)


#converts pd array into pytorch array, ready to be used for machine learning
x = torch.Tensor(data.values[:,1]).unsqueeze(dim=1)
y = torch.Tensor(data.values[:,2]).unsqueeze(dim=1)

#debug statement for progress
print(x, y)

#Plot for Position level vs Salary
plt.plot(x,y)
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Position Level vs Salary')
plt.show()

#preprocessing, first by standardizing and removing mean and then scaling
scaler_x= StandardScaler()
scaler_y =StandardScaler()

x_scaled = torch.Tensor(scaler_x.fit_transform(x))
y_scaled = torch.Tensor(scaler_y.fit_transform(y))

#split and train using x_scaled and y_scaled, (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2)

#debug statement, print out length of splits
print(len(x_train), len(x_test), len(y_train), len(y_test))

#simple neural network has 2 hidden layers in total
model = nn.Sequential(
    nn.Linear(1, 64),   #input layer to hidden layer
    nn.ReLU(),
    nn.Linear(64,64),   # hidden layer to hidden layer
    nn.ReLU(),
    nn.Linear(64,1)     #hidden layer to output layer
)

#debug statement for model
print(model)

#define loss function and optimizer
loss_fn = nn.MSELoss()  #Mean Absolute Error
optimizer = torch.optim.Adam(model.parameters(), lr =0.01)

#define number of epochs
epochs = 100

#Training loop
for epoch in range(epochs):
    pass