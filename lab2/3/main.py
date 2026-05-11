import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Загрузка MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Conv1: 1 канал → 32 канала
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28→14
        
        # Conv2: 32 канала → 64 канала
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14→7
        
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(epochs=5):
    for epoch in range(epochs):
        model.train()
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
        acc = 100 * correct / len(train_dataset)
        print(f'Epoch {epoch+1}: Train Acc = {acc:.2f}%')


def visualize_filters(model, layer_name='conv1', num_filters=32):
    """
    Визуализация фильтров сверточного слоя
    
    Args:
        model: обученная модель CNN
        layer_name: имя слоя ('conv1' или 'conv2')
        num_filters: количество фильтров для отображения
    """
    # Получаем веса фильтров из указанного слоя
    if layer_name == 'conv1':
        filters = model.conv1.weight.data.cpu().numpy()
    elif layer_name == 'conv2':
        filters = model.conv2.weight.data.cpu().numpy()
    else:
        print(f"Слой {layer_name} не найден")
        return
    
    # filters.shape = (out_channels, in_channels, height, width)
    # conv1: (32, 1, 3, 3)
    # conv2: (64, 32, 3, 3)
    
    # Определяем размер сетки для отображения
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    plt.figure(figsize=(12, 12))
    
    for i in range(min(num_filters, filters.shape[0])):
        # Для conv1: один входной канал → берем filters[i, 0]
        # Для conv2: 32 входных канала → усредняем их
        if layer_name == 'conv1':
            filter_weights = filters[i, 0]  # (3, 3)
        else:  # conv2
            filter_weights = filters[i].mean(axis=0)  # усредняем 32 канала → (3, 3)
        
        # Нормализуем значения в диапазон [0, 1] для отображения
        min_val = filter_weights.min()
        max_val = filter_weights.max()
        if max_val - min_val > 1e-8:  # избегаем деления на ноль
            filter_weights = (filter_weights - min_val) / (max_val - min_val)
        else:
            filter_weights = np.zeros_like(filter_weights)
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(filter_weights, cmap='gray', interpolation='nearest')
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    
    plt.suptitle(f'Визуализация фильтров слоя {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.show()


train()

visualize_filters(model, 'conv1', 32)
visualize_filters(model, 'conv2', 64)