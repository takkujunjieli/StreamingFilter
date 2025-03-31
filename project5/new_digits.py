import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from digits_model import MyNetwork

def preprocess_image(image_path):
    """
    Preprocess an image: convert to grayscale, resize to 28x28, and invert intensities.
    """
    # Open the image
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Invert intensities (white digits on black background)
    image = Image.eval(image, lambda x: 255 - x)

    transform = transforms.ToTensor()
    image_tensor = transform(image)

    # Add a batch dimension (1, 1, 28, 28)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def main():
    # Folder containing the images
    image_folder = "./images"

    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Process and evaluate each image in the folder
    predictions = []
    for idx, image_name in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)

        # Preprocess the image
        image_tensor = preprocess_image(image_path)

        # Run the image through the model
        output = model(image_tensor)
        output_values = output.detach().numpy().flatten()  # Convert to numpy array
        predicted_index = output.argmax(dim=1).item()  # Get the index of the max output value

        # Print the results
        print(f"Image {image_name}:")
        print(f"Output values: {[f'{value:.2f}' for value in output_values]}")
        print(f"Predicted index: {predicted_index}\n")

        # Store the prediction for plotting
        predictions.append((image_tensor.squeeze(0), predicted_index))

    # Plot all images in a 2*5 grid with predictions
    plt.figure(figsize=(10, 4))
    for i in range(10):
        image, prediction = predictions[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(0), cmap='gray')
        plt.title(f"Pred: {prediction}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()