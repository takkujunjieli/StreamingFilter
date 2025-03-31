"""
  Junjie Li
  Spring 2025

  This script loads the MNIST model and processes the Greek dataset.
"""

import os
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model_exam import print_model_structure

class Greeketwork(nn.Module):
    def __init__(self):
        super(Greeketwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
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
        dummy_input = torch.zeros(1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(dummy_input), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return x.numel()


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.resize(x, (128, 128))
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def load_model():
    model = Greeketwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, 3)
    print_model_structure(model)
    return model


def evaluate_model(model, greek_train):
    criterion = nn.NLLLoss()
    model.eval()
    training_errors = []
    examples_seen = []
    cumulative_examples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(greek_train):
            output = model(data)
            loss = criterion(output, target)
            training_errors.append(loss.item())
            cumulative_examples += len(data)
            examples_seen.append(cumulative_examples)

    plt.plot(examples_seen, training_errors, color='green')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Training and Testing Loss')
    plt.grid()
    plt.show()

def preprocess_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        GreekTransform(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def process_images(model, image_folder):
    predictions = []
    model.eval()  # Set model to evaluation mode

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image_tensor = preprocess_image(image_path)

        # Move tensor to the same device as the model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        output = model(image_tensor)
        output_values = output.detach().cpu().numpy().flatten()
        predicted_index = output.argmax(dim=1).item()

        print(f"Image {image_name}:")
        print(f"Output values: {[f'{value:.2f}' for value in output_values]}")
        print(f"Predicted index: {predicted_index}\n")

        predictions.append((image_tensor.squeeze(0).cpu(), predicted_index))

    plt.figure(figsize=(10, 3))
    for i in range(min(3, len(predictions))):
        image, prediction = predictions[i]
        plt.subplot(1, 3, i + 1)
        plt.imshow(image.squeeze(0), cmap='gray')
        plt.title(f"Pred: {prediction}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    model = load_model()
    training_set_path = "./data/greek_train"
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    evaluate_model(model, greek_train)
    handwriting_set_path = "./images/greek"
    process_images(model, handwriting_set_path)
    


if __name__ == "__main__":
    main()

