import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
# Define your loss function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(262144, 1024)  # Input size: 32 * 52 * 44
        self.fc2 = nn.Linear(1024, 4)
        self.fc3 = nn.Linear(128, 4)  # Output layer with 4 classes


    def forward(self, x):
        # Apply first Convolutional layer
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)  # Max pooling with kernel size 2

        # Reshape the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply first fully connected layer
        x = self.fc1(x)
        x = nn.ReLU()(x)

        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)

        return x

device = torch.device("cuda")


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

full_dataset = torchvision.datasets.ImageFolder(r"/content/Data", transform=transform)
split_ratio = 0.8
dataset_size = len(full_dataset)
train_size = int(split_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = CNN().to(device)

num_epochs = 30

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    # Training
    model.train()  # Set model to training mode
    train_loss = 0.0  # zero the training loss and accuracy for new training epoch
    train_correct = 0  # zero the training accuracy

    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # send inputs and labels to the GPU
        labels = labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Update training metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item() #tensor.item() strips all dimensions and

    train_loss /= len(train_loader.dataset)  # Average training loss per sample
    train_accuracy = 100.0 * train_correct / len(train_loader.dataset)  # Training accuracy (%)


    # Testing
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0  # zero test loss and correct count for new testing epoch
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update testing metrics
            test_loss += loss.item() #
            _, predicted = torch.max(outputs.data, 1) # get the model's prediction
            test_correct += (predicted == labels).sum().item() # count the amount of correct results

    test_loss /= len(test_loader.dataset)  # Average testing loss per sample
    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)  # Testing accuracy (%), correct results / dataset size


    # Print epoch statistics
    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")



torch.save(model.state_dict(), "tumor_cnn_with_softmax.pt")



