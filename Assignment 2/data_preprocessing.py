import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def create_dataset(text_file, images_folder):
    pairs = []
    with open(text_file, 'r') as file:
        number_of_records = int(file.readline().strip())
        for i, line in enumerate(file):
            parts = line.strip().split()
            if i < number_of_records:
                identity, image_number1, image_number2 = parts[0], int(parts[1]), int(parts[2])
                image_path1 = os.path.join(images_folder, f"{identity}/{identity}_{str(image_number1).zfill(4)}.jpg")
                image_path2 = os.path.join(images_folder, f"{identity}/{identity}_{str(image_number2).zfill(4)}.jpg")
                pairs.append((image_path1, image_path2, 0))
            else:
                identity1, image_number1, identity2, image_number2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                image_path1 = os.path.join(images_folder, f"{identity1}/{identity1}_{str(image_number1).zfill(4)}.jpg")
                image_path2 = os.path.join(images_folder, f"{identity2}/{identity2}_{str(image_number2).zfill(4)}.jpg")
                pairs.append((image_path1, image_path2, 1))
    return pairs

class SiameseNetworkDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __getitem__(self, index):
        img_path1, img_path2, label = self.pairs[index]
        image1 = Image.open(img_path1).convert('L')
        image2 = Image.open(img_path2).convert('L')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2), label

    def __len__(self):
        return len(self.pairs)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
    ])

def get_dataloader(text_file, images_folder, batch_size=32, shuffle=True):
    pairs = create_dataset(text_file, images_folder)
    dataset = SiameseNetworkDataset(pairs, transform=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

