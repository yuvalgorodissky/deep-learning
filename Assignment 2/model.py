import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseBaseNetwork(nn.Module):
    """
    Base network for the Siamese neural network architecture.
    This network defines the convolutional layers that process each input image,
    now including batch normalization and dropout for improved regularization.
    """

    def __init__(self, dropout=0.1, batch_norm=True):
        super(SiameseBaseNetwork, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(self.dropout)  # Dropout after pooling

        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout(self.dropout)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.dropout3 = nn.Dropout(self.dropout)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.dropout4 = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(24 * 24 * 256, 4096)  # Assuming the output size matches the flattened dimensions
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.dropout5 = nn.Dropout(self.dropout)

    def forward(self, x):
        if self.batch_norm:
            x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
            x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
            x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
            x = self.flatten(x)
            x = self.dropout5(F.relu(self.bn_fc1(self.fc1(x))))
        else:
            x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
            x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
            x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
            x = self.dropout4(F.relu(self.conv4(x)))
            x = self.flatten(x)
            x = self.dropout5(F.relu(self.fc1(x)))

        return x


class SiameseNetwork(nn.Module):
    """
    Complete Siamese network combining two base networks with a distance measure at the end.
    """

    def __init__(self, dropout=0.1, batch_norm=True):
        super(SiameseNetwork, self).__init__()
        self.base_network = SiameseBaseNetwork(dropout=dropout, batch_norm=batch_norm)
        self.fc2 = nn.Linear(4096, 1)  # Single output for similarity score

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        x = torch.abs(output1 - output2)
        x = self.fc2(torch.sigmoid(x))
        return torch.sigmoid(x)

    def predict(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        x = torch.abs(output1 - output2)
        x = self.fc2(torch.sigmoid(x))
        predictions = torch.sigmoid(x)  # Sigmoid to get predictions between 0 and 1
        return (predictions > 0.5).float()

    def get_pred_labels(self, dataloader, device):
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(dataloader):
            img1, img2, labels = images[0].to(device), images[1].to(device), labels.to(device)
            predictions = self.predict(img1, img2)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
        # Convert to tensors
        all_predictions = torch.stack(all_predictions)
        all_labels = torch.stack(all_labels)
        return all_labels, all_predictions


