# File system and general utilities
import os
import copy

# Numerical computations
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PyTorch core
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# PyTorch vision utilities
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


path = "alpai/alpai-dataset"

print("Path to dataset files:", path)

os.listdir(path)

task_files = os.path.join(path, 'Task_1_scene_level')

os.listdir(os.path.join(task_files))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path_labeled = 'Task_1_scene_level/X_labeled.npy'
path_unlabeled = 'Task_1_scene_level/X_unlabeled.npy'
path_y_labeled = 'Task_1_scene_level/y_labeled.npy
X_labeled = np.load(os.path.join(path, path_labeled), mmap_mode='r')
y_labeled = np.load(os.path.join(path, path_y_labeled), mmap_mode='r')
X_unlabeled = np.load(os.path.join(path, path_unlabeled), mmap_mode='r')

len(X_labeled)

len(X_unlabeled)

new_size_labeled = 4000
new_size_unlabeled = 4000 #8000
X_labeled = X_labeled[:new_size_labeled]
y_labeled = y_labeled[:new_size_labeled]
X_unlabeled = X_unlabeled[:new_size_unlabeled]

print(X_labeled.shape)
print(y_labeled.shape)
print(X_unlabeled.shape)

# Function to display images with their labels
# def display_images(X, y=None, num_images=5):
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#     for i in range(num_images):
#         axes[i].imshow(X[i].astype(np.uint8)) # Assuming images are stored as uint8
#         axes[i].axis('off')
#         if y is not None:
#             axes[i].set_title(f"Label: {y[i]}")
#     plt.show()

# # Display labeled images
# print("Labeled Images:")
# display_images(X_labeled, y_labeled)


# # Display unlabeled images (no labels available)
# print("Unlabeled Images:")
# display_images(X_unlabeled)

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_classes = 3
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

X_train, X_temp, y_train, y_temp = train_test_split(X_labeled, y_labeled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# change numpy array
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert the image to a PIL Image before applying transformations
        image = image.cpu().numpy()  # Move to CPU and convert to NumPy array
        image = transforms.ToPILImage()(image)  # Convert to PIL Image

        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


# Create data loaders
train_dataset = CustomDataset(X_train, y_train, transform=transform)
val_dataset = CustomDataset(X_val, y_val, transform=transform)
test_dataset = CustomDataset(X_test, transform=transform)  # No labels for the test set

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
train_losses = []
val_losses = []

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Convert labels to class indices before comparison
            labels = torch.argmax(labels, dim=1)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    acc = 100 * correct / total

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.2f}%')
    if acc > best_acc:
        best_acc = acc
        best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:.2f}%')
model.load_state_dict(best_model_wts)

# Plot the losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_plot_4000_0.png')
# Save the best model
torch.save(model.state_dict(), 'best_model.pth')

# Create custom datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        image = transforms.ToPILImage()(image)  # Convert to PIL Image

        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

# Create data loaders
train_dataset = CustomDataset(X_train, y_train, transform=transform)
val_dataset = CustomDataset(X_val, y_val, transform=transform)
test_dataset = CustomDataset(X_test, transform=transform)  # No labels for the test set

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate the model on the test set
y_pred = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())

# Convert y_test to multiclass format (assuming it's multilabel-indicator)
y_test_multiclass = np.argmax(y_test, axis=1) # Convert to class indices

accuracy = accuracy_score(y_test_multiclass, y_pred)  # Use y_test_multiclass
print(f'Test Accuracy (Labeled): {accuracy * 100:.2f}%')

unlabeled_dataset = CustomDataset(X_unlabeled, transform=transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Make predictions on the unlabeled data
y_unlabeled_pred = []
with torch.no_grad():
    for inputs in unlabeled_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_unlabeled_pred.extend(predicted.cpu().numpy())

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and move it to CPU before converting to NumPy
        image = self.images[idx].cpu().numpy()
        # Convert the image to a PIL Image before applying transformations
        image = transforms.ToPILImage()(image)  # Convert to PIL Image

        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

import copy
from itertools import cycle

# Hyperparameters
T1 = 5
T2 = 10
alpha_f = 3.0
num_epochs = 10
batch_size = 8

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to device
model = model.to(device)

# For tracking best model
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
train_losses = []
val_losses = []

# Make sure y_unlabeled_pred is a tensor
y_unlabeled_pred = torch.tensor(y_unlabeled_pred, dtype=torch.long, device=device)

for epoch in range(num_epochs):
    t = epoch + 1

    # Calculate alpha
    if t < T1:
        alpha = 0
    elif t < T2:
        alpha = ((t - T1) / (T2 - T1)) * alpha_f
    else:
        alpha = alpha_f

    print(f"Epoch {epoch+1}/{num_epochs}, Alpha: {alpha:.4f}")

    model.train()
    running_loss = 0.0
    # Sample unlabeled data ONCE per epoch
    unlabeled_subset_size = min(len(X_train), len(X_unlabeled)) # Ensure subset size is not larger than X_unlabeled
    unlabeled_indices = np.random.choice(len(X_unlabeled), size=unlabeled_subset_size, replace=False)
    X_unlabeled_subset = torch.from_numpy(X_unlabeled[unlabeled_indices]).to(device)
    y_unlabeled_subset = y_unlabeled_pred[unlabeled_indices]
    # Create unlabeled dataset and loader
    unlabeled_dataset = CustomDataset(X_unlabeled_subset, labels=y_unlabeled_subset, transform=transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    # Make iterator for unlabeled data
    unlabeled_iter = cycle(unlabeled_loader)

    for inputs_labeled, labels_labeled in train_loader:
        inputs_labeled = inputs_labeled.to(device)
        labels_labeled = labels_labeled.to(device)

        inputs_unlabeled, labels_unlabeled = next(unlabeled_iter)
        inputs_unlabeled = inputs_unlabeled.to(device)
        labels_unlabeled = labels_unlabeled.to(device)

        optimizer.zero_grad()

        outputs_labeled = model(inputs_labeled)
        loss_labeled = criterion(outputs_labeled, labels_labeled)

        outputs_unlabeled = model(inputs_unlabeled)
        loss_unlabeled = criterion(outputs_unlabeled, labels_unlabeled)

        # Total loss
        loss = loss_labeled + alpha * loss_unlabeled

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Average train loss
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

# After all epochs
print(f"Best Validation Accuracy: {best_acc:.2f}%")
model.load_state_dict(best_model_wts)

# Save the final best model
torch.save(model.state_dict(), 'best_model_semi_supervised.pth')

# Plot the losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_plot_4000_4000.png')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Already a NumPy array
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

test_dataset = CustomDataset(X_test, y_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load best model from semi-supervised training
model.load_state_dict(torch.load('best_model_semi_supervised.pth'))
model.eval()

# Test evaluation
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(torch.argmax(labels, dim=1).cpu().numpy())  # Convert one-hot to class index

# Compute accuracy
test_acc = accuracy_score(y_true, y_pred)
print(f'Test Accuracy (Semi-supervised model): {test_acc * 100:.2f}%')

if __name__ == "__main__":
    print("Running SSL (Scene Level).")