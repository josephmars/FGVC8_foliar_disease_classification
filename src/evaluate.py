import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import CNN_model
from src.config import device
from src.train import test_dataset, batch_size, label_to_index

# Load the pre-trained model
num_classes = 12
model = CNN_model(num_classes).to(device)
model.load_state_dict(torch.load('./kaggle/cnn_model.pth'))
model.eval()

# Define a function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, text_labels in data_loader:
            inputs, text_labels = inputs.to(device), text_labels
            numerical_labels = [label_to_index[label] for label in text_labels]
            numerical_labels = torch.tensor(numerical_labels).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(numerical_labels.cpu().numpy())

    return all_predictions, all_labels

# Evaluate the model
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
predictions, labels = evaluate_model(model, test_loader)

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
