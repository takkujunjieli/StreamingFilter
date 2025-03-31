import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import cv2


from digits_model import MyNetwork

def print_model_structure(model):
    """
    Print the structure of the model.
    """
    print("Model Structure:")
    print(model)

    # Print the name and structure of each layer
    print("\nLayer Details:")
    for name, layer in model.named_children():
        print(f"Layer Name: {name}, Layer Structure: {layer}")

def analyze_layer(layer):
    """
    Analyze the weights of the layer.
    """
    # Get the weights of the layer
    weights = layer.weight.data

    print("Filter Weights:")
    print(weights)

    print("Filter Weights Shape:", weights.shape)
    # Visualize the first 10 filters
    num_filters = min(10, weights.shape[0])
    plt.figure(figsize=(8, 6))
    for i in range(num_filters):
        plt.subplot(3, 4, i + 1)  # Create a 3x4 grid
        plt.imshow(weights[i, 0].cpu().numpy(), cmap='plasma')  # Visualize the first channel of each filter
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.title(f"Filter {i+1}")
    plt.tight_layout()
    plt.show()

def apply_filters_to_image(model, image_tensor):
    """
    Apply the first convolutional layer's filters to the input image using OpenCV's filter2D.
    Visualize both the filter weights and their effects on the digit in a 5x4 grid.
    """
    # Get the weights of the first convolutional layer
    with torch.no_grad():
        filters = model.conv1.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, kernel_height, kernel_width)

    # Extract the first channel of the input image
    image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (28, 28)

    # Apply each filter to the image using OpenCV's filter2D
    filtered_images = []
    for i in range(filters.shape[0]):  # Iterate over the filters
        kernel = filters[i, 0]  # Get the filter for the first channel
        filtered_image = cv2.filter2D(image, -1, kernel)  # Apply the filter
        filtered_images.append(filtered_image)

    # Plot the filter weights and their effects side by side
    plt.figure(figsize=(12, 10))
    for i in range(len(filtered_images)):
        # Visualize the filter weights
        plt.subplot(5, 4, 2 * i + 1)  # Odd positions for filter weights
        plt.imshow(filters[i, 0], cmap='gray')  # Visualize the first channel of each filter
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.title(f"Weight {i+1}")

        # Visualize the effect of the filter on the digit
        plt.subplot(5, 4, 2 * i + 2)  # Even positions for filtered images
        plt.imshow(filtered_images[i], cmap='gray')  # Use a colorful colormap
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.title(f"Effect {i+1}")

    plt.tight_layout()
    plt.show()



def main():
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()

    print_model_structure(model)
    print("Analyzing conv1:")
    analyze_layer(model.conv1)

    # Load the first training example from the MNIST dataset
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    first_image, _ = train_set[0]  # Get the first training example
    first_image_tensor = first_image.unsqueeze(0)  # Add a batch dimension

    # Apply filters to the first training example
    print("Applying filters to the first training example...")
    apply_filters_to_image(model, first_image_tensor)


if __name__ == "__main__":
    main()