import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Transformer import Transformer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess the images

"""
1) transforms.Resize((32, 32)): This step resizes the input images to a fixed size of 32x32 pixels.
Resizing is necessary because deep learning models often require a consistent input size. 
In this case, the CIFAR-10 dataset is being used, which consists of 32x32 pixel images.

2) transforms.ToTensor(): This step converts the resized image from a PIL Image object to a PyTorch Tensor. 
Tensors are the primary data structure used in PyTorch for storing and manipulating data, 
and they are essential for performing operations in deep learning models.

3) transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)): This step normalizes the pixel values of the 
image tensor. The provided values (0.5, 0.5, 0.5) represent the mean values for the three color channels (RGB), 
and (0.5, 0.5, 0.5) represent the standard deviation values for the three color channels. By subtracting the 
mean and dividing by the standard deviation, the pixel values are centered around zero and scaled to
a range that is more suitable for training deep learning models. 
Normalization helps in stabilizing the learning process and improving model performance.
"""
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Define data splits
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define data loaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model parameters
d_model = 300
n_heads = 50
n_layers = 12
num_classes = 10
dropout = 0.1

# Create the transformer model
model = Transformer(d_model, n_heads, n_layers, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    total_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training dataset
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_loss += loss.item()

    # Calculate training accuracy and loss
    train_accuracy = correct / total
    train_loss = total_loss / len(train_loader)

    model.eval()  # Set the model in evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Iterate over the validation dataset
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    # Calculate validation accuracy and loss
    val_accuracy = val_correct / val_total
    val_loss /= len(val_loader)

    # Print epoch statistics
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f} | Validation_Acc: {val_accuracy:.4f}")
    print("-------------------------------------")

# Save the trained model
torch.save(model.state_dict(), "transformer_cifar10.pth")