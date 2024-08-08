#A Basic Neural Network that has been made to predict the unknown record by learning previous record
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Load data
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Open Price": [150.0, 152.3, 153.0, 155.4, 156.2, 157.8, 158.5, 160.0, 161.3, 162.5],
    "High": [151.0, 153.5, 154.7, 156.9, 157.3, 158.9, 159.8, 161.2, 162.6, 163.8],
    "Close Price": [150.8, 153.2, 154.0, 156.0, 156.8, 158.4, 159.2, 160.5, 161.9, 163.0]
}
df = pd.DataFrame(data)

# Prepare the data
X = df[["Open Price", "High"]].values
y = df["Close Price"].values

X_train = torch.tensor(X[:8], dtype=torch.float32) #Only 9 day data has given to train 
y_train = torch.tensor(y[:8], dtype=torch.float32).view(-1, 1)#Only 9 day data has given to train

# Define the neural network model
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = StockPredictor()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
epochs = 200000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Predicting the 6th day close price
o=float(input("Opening:"))
h=float(input("high:"))
while o!=0 and h!=0:
    model.eval()
    with torch.no_grad():
        
        X_test = torch.tensor([[o,h]], dtype=torch.float32) 
        prediction = model(X_test)
        print(f"Predicted closing price for day 6: {prediction.item():.2f}")
    o=float(input("Opening:"))
    h=float(input("high:"))
