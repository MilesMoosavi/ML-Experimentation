import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimpleNN(nn.Module):
  def __init__(self):
    # Initialize parent nn.Module for PyTorch tracking
    super(SimpleNN, self).__init__()
    # Fully connected: 784 pixels -> 128 features
    self.fc1 = nn.Linear(28*28, 128)
    # Compress to 64 features
    self.fc2 = nn.Linear(128, 64)
    # Output 10 classes (digits 0-9)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    # Flatten batch of [batch,1,28,28] to [batch,784]
    x = x.view(-1, 28 * 28)
    # Linear transform + ReLU: add non-linearity
    x = torch.relu(self.fc1(x))
    # Another layer for hierarchical features
    x = torch.relu(self.fc2(x))
    # Final linear: logits for classification
    x = self.fc3(x)
    return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 60k training images
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 10k test images
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Batched training data, shuffled for randomness
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Batched test data, no shuffle for consistent evaluation
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
model = SimpleNN()
# Loss for multi-class classification
criterion = nn.CrossEntropyLoss()
# Optimizer with adaptive learning rates
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configure GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'Using device: {device}')

# Train for 5 epochs
for epoch in tqdm(range(5)):
    total_loss = 0
    # Loop over batches in training data
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        # Clear gradients from previous batch
        optimizer.zero_grad()
        # Forward pass: get predictions
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backpropagate: compute gradients
        loss.backward()
        # Update model weights
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

# Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        # Get predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update total samples
        total += labels.size(0)
        # Update correct predictions
        correct += (predicted == labels).sum().item()
# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')