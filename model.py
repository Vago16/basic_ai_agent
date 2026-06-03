import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn

data = pd.read_csv('Position_Salaries.csv')

data = data.apply(pd.to_numeric, errors = 'coerce') #any value not a number gets turned into NaN value

#debug statemnt for progress
#print(data)


#converts pd array into pytorch array, ready to be used for machine learning
x = torch.Tensor(data.values[:,1]).unsqueeze(dim=1)
y = torch.Tensor(data.values[:,2]).unsqueeze(dim=1)

#debug statement for progress
#print(x, y)

#Plot for Position level vs Salary
#plt.plot(x,y)
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.title('Position Level vs Salary')
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300

for epoch in range(epochs):
    model.train()

    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}")

model.eval()
with torch.inference_mode():
    y_pred_scaled = model(x)

    y_preds = scaler_y.inverse_transform(
        y_pred_scaled.detach().cpu().numpy()
    )

x_plot = scaler_x.transform(x.numpy())
x_plot = torch.tensor(x_plot, dtype=torch.float32)

with torch.inference_mode():
    plt.plot(x, y, label="Actual Data")
    plt.scatter(x, scaler_y.inverse_transform(model(x_plot).detach().numpy()), label="Model Prediction")

plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Model Predictions vs Actual Data')
plt.legend()
plt.show()