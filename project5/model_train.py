import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from digits_model import MyNetwork


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


def main():
    # Define hyperparameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    weight_decay=1e-4

    # Set random seed for reproducibility
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MyNetwork()
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Create a Trainer instance
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, n_epochs, log_interval)

    # Run the training and testing process
    trainer.test()  # Initial test before training
    for epoch in range(1, n_epochs + 1):
        trainer.train(epoch)
        trainer.test()

    # Plot the training and testing error
    trainer.plot_losses()


if __name__ == "__main__":
    main()