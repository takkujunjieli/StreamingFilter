"""
  Junjie Li
  Spring 2025

  This script loads the MNIST dataset and plots the first 6 digits from the test set.

"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_mnist_data():
    return torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())



def plot_mnist_digits(test_set):
    """
    Loads the MNIST test dataset and plots the first 6 digits with their labels.
    """

    # Plot the first 6 digits from the test set
    plt.figure(figsize=(10, 2))
    for i in range(6):
        image, label = test_set[i]  # get the image and label
        image = image.squeeze(0)    # remove channel dimension

        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(label))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer with a dropout rate of 35%
        self.dropout = nn.Dropout2d(p=0.35)
        self.fc1 = nn.Linear(self._get_flattened_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def _get_flattened_size(self):
        """
        Dynamically calculates the flattened size of the feature map after
        the convolution and pooling layers.
        """
        dummy_input = torch.zeros(1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(dummy_input), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return x.numel()


if __name__ == "__main__":
    test_set = get_mnist_data()
    plot_mnist_digits(test_set)
