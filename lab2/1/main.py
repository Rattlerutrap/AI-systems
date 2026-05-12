import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

class SingleNeuron(nn.Module):
    def __init__(self, input_dim=2, activation='sigmoid'):
        super(SingleNeuron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
            
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

def prepare_data(data):
    X = []
    y = []
    for row in data:
        X.append([float(row[0]), float(row[1])])
        y.append(float(row[2]))
    return np.array(X), np.array(y)    

def trainSingleNeuron(x0_train_t, y0_train_t, x0_test_t, y0_test_t, activation, optimizer_name, epochs):


    model = SingleNeuron(2, activation)

    criterion = nn.BCEWithLogitsLoss()  # Бинарная кросс-энтропия

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())

    epoch_good = -1
    accuracies = np.array([])
    losses = np.array([])
    for epoch in range(epochs):
        # Прямой проход
        outputs = model(x0_train_t)
        loss = criterion(outputs, y0_train_t)
        
        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            test_outputs = model(x0_test_t)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y0_test_t).float().mean()
            accuracies = np.append(accuracies, accuracy)
            losses = np.append(losses, loss.item())
            if accuracy == 1 and epoch_good == -1:
                epoch_good = epoch
            # print(f'Эпоха {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}')
    # print(f'Epoch with accuracy == 1: {epoch_good}')
    return accuracies, losses, epoch_good, np.arange(epochs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
set_seed(122)

def plot_classes(X, y, title="Классы точек"):
    """
    Визуализация точек с цветовой маркировкой по классам
    
    Args:
        X: np.array, признаки (N, 2)
        y: np.array, метки классов (N,), значения -1 или 1
        title: str, заголовок графика
    """
    plt.figure(figsize=(8, 6))
    
    # Разделяем точки по классам
    class_minus1 = X[y == -1]
    class_plus1 = X[y == 1]
    
    # Рисуем точки
    plt.scatter(class_minus1[:, 0], class_minus1[:, 1], 
                color='red', label='Класс -1', alpha=0.7, s=30)
    plt.scatter(class_plus1[:, 0], class_plus1[:, 1], 
                color='green', label='Класс 1', alpha=0.7, s=30)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.show()

nn_0 = utility.importCSV('nn_0.csv', 1, delimiter=',')
nn_1 = utility.importCSV('nn_1.csv', 1, delimiter=',')

random.shuffle(nn_0)
random.shuffle(nn_1)

x0, y0 = prepare_data(nn_0)
x1, y1 = prepare_data(nn_1)

plot_classes(x0, y0, 'nn_0')
plot_classes(x1, y1, 'nn_1')

y0 = (y0 + 1) / 2
y1 = (y1 + 1) / 2

scaler = StandardScaler()
x0_scaled = scaler.fit_transform(x0)
x1_scaled = scaler.fit_transform(x1)

x0_train, x0_test, y0_train, y0_test = train_test_split(
    x0_scaled, y0, test_size=0.2, random_state=42
)
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1_scaled, y1, test_size=0.2, random_state=42
)

x0_train_t = torch.FloatTensor(x0_train)
y0_train_t = torch.FloatTensor(y0_train).reshape(-1, 1)
x0_test_t = torch.FloatTensor(x0_test)
y0_test_t = torch.FloatTensor(y0_test).reshape(-1, 1)

x1_train_t = torch.FloatTensor(x1_train)
y1_train_t = torch.FloatTensor(y1_train).reshape(-1, 1)
x1_test_t = torch.FloatTensor(x1_test)
y1_test_t = torch.FloatTensor(y1_test).reshape(-1, 1)

opts = ['adam', 'sgd', 'rmsprop']
acts = ['sigmoid', 'tanh', 'relu']
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'olive']
plt.figure(figsize=(12, 6))
color = 0
for i in acts:
    for j in opts:
        accs, losss, good, epochs = trainSingleNeuron(x0_train_t, y0_train_t, x0_test_t, y0_test_t, i, j, 1000)

        print(f'{i} + {j} epoch, when accuracy becames 1.0: {good}')

        plt.subplot(1, 2, 1)
        plt.plot(epochs, accs, colors[color], linewidth=2, label=f'{i} + {j}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, losss, colors[color], linewidth=2, label=f'{i} + {j}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.legend()
        color += 1

        
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('nn_0')
plt.savefig(f'nn_0')


plt.figure(figsize=(12, 6))
color = 0
for i in acts:
    for j in opts:
        accs, losss, good, epochs = trainSingleNeuron(x1_train_t, y1_train_t, x1_test_t, y1_test_t, i, j, 1000)

        print(f'{i} + {j} epoch, when accuracy becames 1.0: {good}')

        plt.subplot(1, 2, 1)
        plt.plot(epochs, accs, colors[color], linewidth=2, label=f'{i} + {j}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, losss, colors[color], linewidth=2, label=f'{i} + {j}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.legend()
        color += 1

        
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('nn_1')
plt.savefig(f'nn_1')


