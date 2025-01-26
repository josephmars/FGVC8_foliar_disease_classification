import torch

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
train_labels_path = './kaggle/train.csv'
train_dir = './kaggle/train_images'
test_dir = './kaggle/test_images'