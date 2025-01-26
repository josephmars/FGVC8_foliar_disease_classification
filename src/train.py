import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from src.model import CNN_model, FocalLoss
from src.data import CustomTextLabelDataset, transform
from src.config import device, train_labels_path, train_dir

# Create dataset and data loaders
full_dataset = CustomTextLabelDataset(csv_path=train_labels_path, image_dir=train_dir, transform=transform)
train_size = int(0.08 * len(full_dataset))
val_size = int(0.01 * len(full_dataset))
test_size = int(0.01 * len(full_dataset))
not_used_size = len(full_dataset) - train_size - val_size - test_size
train_dataset, val_dataset, test_dataset, not_used = random_split(full_dataset, [train_size, val_size, test_size, not_used_size])

batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
num_classes = 12
model = CNN_model(num_classes).to(device)
criterion = FocalLoss(gamma=2, alpha=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 4
label_to_index = {label: index for index, label in enumerate(full_dataset.data[0]['label'])}

for epoch in range(epochs):
    model.train()
    for inputs, text_labels in train_loader:
        inputs, text_labels = inputs.to(device), text_labels
        numerical_labels = [label_to_index[label] for label in text_labels]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.tensor(numerical_labels).to(device))
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, text_labels in val_loader:
            inputs, text_labels = inputs.to(device), text_labels
            numerical_labels = [label_to_index[label] for label in text_labels]
            numerical_labels = torch.tensor(numerical_labels).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += len(numerical_labels)
            correct += (predicted == numerical_labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{epochs}, Train Set Loss: {loss.item():.4f}, Validation Set Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), './kaggle/cnn_model.pth')
