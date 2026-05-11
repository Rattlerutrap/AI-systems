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

class MultiLayerNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, activation='relu'):
        super(MultiLayerNet, self).__init__()
        
        # Скрытый слой (2 → 4 нейрона)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Выходной слой (4 → 1 нейрон)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden(x)      # 2 входа → 4 нейрона
        x = self.activation(x)  # функция активации
        x = self.output(x)      # 4 нейрона → 1 выход
        x = self.sigmoid(x)     # для вероятности 0-1
        return x

def prepare_data(data):
    X = []
    y = []
    for row in data:
        X.append([float(row[0]), float(row[1])])
        y.append(float(row[2]))
    return np.array(X), np.array(y)    

def trainSingleNeuron(x0_train_t, y0_train_t, x0_test_t, y0_test_t, activation, optimizer_name, epochs):


    model = MultiLayerNet(2, activation=activation)

    criterion = nn.BCEWithLogitsLoss()

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

nn_1 = utility.importCSV('nn_1.csv', 1, delimiter=',')

random.shuffle(nn_1)

x1, y1 = prepare_data(nn_1)

y1 = (y1 + 1) / 2

scaler = StandardScaler()
x1_scaled = scaler.fit_transform(x1)

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1_scaled, y1, test_size=0.2, random_state=42
)


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
        accs, losss, good, epochs = trainSingleNeuron(x1_train_t, y1_train_t, x1_test_t, y1_test_t, i, j, 500)

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


