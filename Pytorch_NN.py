
#Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Created fully connected network

class NN(nn.Module):
    def __init__(self,input_size,number_class):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,50)
        self.fc3 = nn.Linear(50,number_class)
    def forward(self,x):
        x= F.relu(self.fc1(x))
        h_1= F.relu(self.fc2(x))
        out = self.fc3(h_1)
        return out
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.0001
batch_size = 64
number_epoch = 20

#Load data

train_dataset=datasets.MNIST(root="dataset/",train = True, transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root="dataset/",train = False, transform=transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


model = NN(input_size=input_size,number_class=num_classes).to(device)

#loss function & optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(number_epoch):
    total_loss = 0
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        # Get the correct shape
        data = data.reshape(data.shape[0], -1)

        predicted = model(data)
        loss = criterion(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch =", epoch + 1, "/", number_epoch, " Loss =", total_loss /(batch_index+1))


def Check_accuracy(loader,model):
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,prediction = scores.max(1)
            num_correct = (prediction==y).sum()
            num_sample = prediction.size(0)
        print("The accuracy: ",(num_correct.item()/num_sample)*100)

print("Training accuracy:")
Check_accuracy(train_loader,model)
print("Test accuracy:")
Check_accuracy(test_loader,model)




