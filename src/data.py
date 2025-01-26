import os
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, ColorJitter
from torch.utils.data import Dataset

# Define data transformations
resizing = Resize([64, 64], interpolation=InterpolationMode.BICUBIC)
increase_saturation = ColorJitter(saturation=4)
transform = Compose([resizing, increase_saturation, ToTensor()])

class CustomTextLabelDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = csv_path
        self.images_root = image_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        data = [{'filename': row['image'], 'label': row['labels']} for _, row in df.iterrows()]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        image_path = os.path.join(self.images_root, entry['filename'])
        text_label = entry['label']
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, text_label
