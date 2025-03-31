"""
  Junjie Li
  Spring 2025

  This script using the MNIST Fashion data set and try to optimize the network performance.

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools

from torchvision import datasets, transforms


class CustomizedNetwork(nn.Module):
    def __init__(self, num_filters1=10, num_filters2=20, dropout_rate=0.35, kernel_size=5, hidden_units=50):
        super(CustomizedNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(self._get_flattened_size(num_filters1, num_filters2, kernel_size), hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def _get_flattened_size(self, num_filters1, num_filters2, kernel_size):
        dummy_input = torch.zeros(1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(dummy_input), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return x.numel()


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, n_epochs, log_interval):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx * len(data)) + ((epoch - 1) * len(self.train_loader.dataset))
                )
                torch.save(self.model.state_dict(), './model/model.pth')
                torch.save(self.optimizer.state_dict(), './model/optimizer.pth')

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        self.test_losses.append(test_loss)
        print(
            '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

    def plot_losses(self):
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='green')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Negative Log Likelihood Loss')
        plt.title('Training and Testing Loss')
        plt.grid()
        plt.show()


def hyperparameter_search():
    # Define hyperparameter ranges
    num_filters1_range = [16, 32, 64]
    num_filters2_range = [32, 64, 128]
    dropout_rate_range = [0.2, 0.5]
    kernel_size_range = [3, 5]
    hidden_units_range = [32, 64, 128]

    # Coordinate descent strategy: iterate over one hyperparameter at a time
    best_accuracy = 0
    best_params = None

    # Iterate over combinations of hyperparameters
    for num_filters1, num_filters2, dropout_rate, kernel_size, hidden_units in itertools.product(
        num_filters1_range, num_filters2_range, dropout_rate_range, kernel_size_range, hidden_units_range
    ):
        print(f"Evaluating: num_filters1={num_filters1}, num_filters2={num_filters2}, "
              f"dropout_rate={dropout_rate}, kernel_size={kernel_size}, hidden_units={hidden_units}")

        # Initialize the model with current hyperparameters
        model = CustomizedNetwork(
            num_filters1=num_filters1,
            num_filters2=num_filters2,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            hidden_units=hidden_units
        )

        # Initialize the loss function and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4)
        log_interval = 10
        n_epochs = 5
        # Create a Trainer instance
        trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, n_epochs, log_interval)

        # Train and evaluate the model
        trainer.test()  # Initial test before training
        for epoch in range(1, n_epochs+1):  # Use fewer epochs for faster evaluation
            trainer.train(epoch)
            trainer.test()

        # Calculate the final test accuracy
        correct = 0
        total = len(test_loader.dataset)
        for data, target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        final_accuracy = 100.0 * correct / total

        print(f"Test Accuracy: {final_accuracy:.2f}%")

        # Update the best parameters if current accuracy is better
        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_params = (num_filters1, num_filters2, dropout_rate, kernel_size, hidden_units)

    # Print the best hyperparameters and accuracy
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: num_filters1={best_params[0]}, num_filters2={best_params[1]}, "
          f"dropout_rate={best_params[2]}, kernel_size={best_params[3]}, hidden_units={best_params[4]}")


def main():
    # Define hyperparameters
    global train_loader, test_loader
    
    batch_size_train = 64
    batch_size_test = 1000
    

    # Set random seed for reproducibility
    random_seed = 43
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    # Perform hyperparameter search
    hyperparameter_search()


if __name__ == "__main__":
    main()