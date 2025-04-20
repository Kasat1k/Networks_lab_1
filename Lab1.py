import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

#1. Двовимірні гауссівські точки 
np.random.seed(42)
n_samples = 200

mean1 = [2, 2]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [5, 5]
cov2 = [[1, -0.3], [-0.3, 1]]

X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)

y1 = np.zeros(n_samples)
y2 = np.ones(n_samples)

X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Клас 0')
plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Клас 1')
plt.legend()
plt.title("Гауссівські точки (2D)")
plt.grid(True)
plt.show()

#2. Побудова МНК-регресії
log_reg = LogisticRegression()
log_reg.fit(X, y)

xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yy = -(log_reg.coef_[0][0] * xx + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Клас 0')
plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Клас 1')
plt.plot(xx, yy, 'k--', label='МНК межа')
plt.title("Лінійна регресія (МНК)")
plt.legend()
plt.grid(True)
plt.show()

#3. Перцептрон з різними активаціями (PyTorch)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

class Perceptron(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.fc(x))

def train_model(activation_fn, name):
    model = Perceptron(activation_fn)
    criterion = nn.BCELoss() if isinstance(activation_fn, nn.Sigmoid) else nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    losses = []

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_labels = (y_pred > 0.5).float()
        acc = (y_pred_labels == y_test_tensor).float().mean().item()
        print(f"{name} — точність: {acc:.2f}")
        print(f"Ваги: {model.fc.weight.data.numpy()}, Зсув: {model.fc.bias.data.numpy()}")

    plt.plot(losses, label=name)

plt.figure()
train_model(nn.Sigmoid(), "Sigmoid")
train_model(nn.ReLU(), "ReLU")
train_model(nn.Tanh(), "Tanh")
plt.title("Втрати при навчанні (активації)")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

#4. Багатовимірний випадок (3 класи)
X_multi, y_multi = make_classification(
    n_samples=300, n_features=2, n_classes=3, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

plt.figure()
plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_multi, cmap=plt.cm.Set1)
plt.title("3 класи (2D)")
plt.grid(True)
plt.show()

#5. Тривимірний випадок (3D)
mean3D_1 = [2, 2, 2]
cov3D_1 = np.eye(3)
mean3D_2 = [5, 5, 5]
cov3D_2 = np.eye(3)

X3D_1 = np.random.multivariate_normal(mean3D_1, cov3D_1, 100)
X3D_2 = np.random.multivariate_normal(mean3D_2, cov3D_2, 100)
X3D = np.vstack((X3D_1, X3D_2))
y3D = np.hstack((np.zeros(100), np.ones(100)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3D_1[:, 0], X3D_1[:, 1], X3D_1[:, 2], color='blue', label='Клас 0')
ax.scatter(X3D_2[:, 0], X3D_2[:, 1], X3D_2[:, 2], color='red', label='Клас 1')
ax.set_title("3D Гауссівські точки")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
