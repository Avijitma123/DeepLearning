# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Created LSTM
class LSTM_23(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_23, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.0001
batch_size = 64
number_epoch = 3

# Load data

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = LSTM_23(input_size, hidden_size, num_layers, num_classes).to(device)

# loss function & optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(number_epoch):
    total_loss = 0
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        target = target.to(device=device)

        predicted = model(data)
        loss = criterion(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch =", epoch + 1, "/", number_epoch, " Loss =", total_loss / (batch_index + 1))


def Check_accuracy(loader, model):
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct = (prediction == y).sum()
            num_sample = prediction.size(0)
        print("The accuracy: ", (num_correct.item() / num_sample) * 100)


print("Training accuracy:")
Check_accuracy(train_loader, model)
print("Test accuracy:")
Check_accuracy(test_loader, model)




