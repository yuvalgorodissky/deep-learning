import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random


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


def get_transforms(use_augmentation):
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((250, 250)),  # Resize the image to 250x250
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=90),  # Random rotation up to 90 degrees
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomResizedCrop(250, scale=(0.6, 1.4), ratio=(1.0, 1.0)),  # Random zoom-in and zoom-out
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

def get_dataloader(text_file, images_folder, batch_size=32, shuffle=True,use_augmentation=0,splits=0.9):
    pairs = create_dataset(text_file, images_folder)
    random.shuffle(pairs)
    train_pairs = []
    if  use_augmentation>0:
        base_train_pairs , val_pairs = pairs[:int(len(pairs)*splits)], pairs[int(len(pairs)*splits):]
        for i in range(use_augmentation):
            train_pairs.extend(base_train_pairs)
        train_dataset = SiameseNetworkDataset(train_pairs, transform=get_transforms(use_augmentation))
        val_dataset = SiameseNetworkDataset(val_pairs, transform=get_transforms(use_augmentation))
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    dataset = SiameseNetworkDataset(pairs, transform=get_transforms(use_augmentation))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




