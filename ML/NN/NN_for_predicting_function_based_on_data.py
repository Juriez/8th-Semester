import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples=1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
    y = np.sin(x) + np.cos(x)
    x_norm = (x - np.mean(x)) / np.std(x)
    return torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1), \
           torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class CustomizedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def train_model(model, x_train, y_train, epochs=4000, lr=0.002):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    mse = torch.mean((predictions - y_test)**2).item()
    ss_total = torch.sum((y_test - torch.mean(y_test))**2)
    ss_res = torch.sum((y_test - predictions)**2)
    r2 = 1 - ss_res / ss_total
    match = max(0, min(100, r2.item() * 100))

    print(f"\nMSE: {mse:.6f}")
    #print(f"R² Score: {r2.item():.4f}")
    print(f"Function Matching: {match:.2f}%")

    plt.figure(figsize=(8,4))
    plt.scatter(x_test.numpy(), y_test.numpy(), color='blue', s=10, label='True y')
    plt.scatter(x_test.numpy(), predictions.numpy(), color='red', s=10, label='Predicted')
    plt.legend()
    plt.title(f'Function Approximation (Match: {match:.2f}%)')
    plt.show()

# Generate and split dataset
x, y = generate_data()
indices = torch.randperm(len(x))
train_size = int(0.8 * len(x))
train_idx, test_idx = indices[:train_size], indices[train_size:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

num_hidden_layers = int(input("Enter number of hidden layers: "))
model = CustomizedNN(1, 64, num_hidden_layers, 1)
trained = train_model(model, x_train, y_train)
evaluate_model(trained, x_test, y_test)
