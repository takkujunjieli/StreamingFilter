import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from digits_model import MyNetwork

def main():
    # Load the MNIST test dataset
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()  

    # Evaluate the first 10 examples
    predictions = []
    for idx, (data, target) in enumerate(test_loader):
        if idx >= 10: 
            break

        # Run the data through the model
        output = model(data)
        output_values = output.detach().numpy().flatten() 
        predicted_index = output.argmax(dim=1).item() 
        correct_label = target.item()

        # Print the results
        print(f"Example {idx + 1}:")
        print(f"Output values: {[f'{value:.2f}' for value in output_values]}")
        print(f"Predicted index (the index of the max output value): {predicted_index}, Correct label: {correct_label}\n")

        # Store the prediction for plotting
        predictions.append((data.squeeze(0), predicted_index))

    # Plot the first 9 digits in a 3x3 grid with predictions
    plt.figure(figsize=(8, 8))
    for i in range(9):
        image, prediction = predictions[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.squeeze(0), cmap='gray')
        plt.title(f"Pred: {prediction}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()