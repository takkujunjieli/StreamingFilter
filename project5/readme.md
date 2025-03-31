# Project 5: MNIST and Greek Character Recognition

### Name: Junjie Li

### I am using one travel day

This project contains various scripts for training, evaluating, and experimenting with neural networks for digit and character recognition tasks. Below is an explanation of each file and its purpose.

## File Descriptions

### 1. `digits_model.py`

- **Purpose**: Defines the `MyNetwork` class, a convolutional neural network (CNN) for recognizing MNIST digits.
- **Key Features**:
  - Loads the MNIST dataset.
  - Plots the first 6 digits from the test set.
  - Implements the forward pass of the CNN.
- **Usage**: Used as the base model for MNIST digit recognition.

---

### 2. `model_train.py`

- **Purpose**: Trains the `MyNetwork` model on the MNIST dataset.
- **Key Features**:
  - Implements a `Trainer` class for managing the training and testing process.
  - Saves the trained model and optimizer states.
  - Plots training and testing losses.
- **Usage**: Run this script to train the MNIST digit recognition model.

---

### 3. `mode_eval.py`

- **Purpose**: Evaluates the trained `MyNetwork` model on the MNIST test dataset.
- **Key Features**:
  - Loads the trained model.
  - Evaluates the first 10 examples from the test set.
  - Plots the first 9 digits in a 3x3 grid with their predictions.
- **Usage**: Run this script to visualize the model's predictions on the MNIST test set.

---

### 4. `new_digits.py`

- **Purpose**: Processes and evaluates custom digit images using the trained `MyNetwork` model.
- **Key Features**:
  - Preprocesses custom images (grayscale, resize, invert).
  - Runs the images through the trained model.
  - Displays predictions for up to 10 images in a 2x5 grid.
- **Usage**: Place custom digit images in the `./images` folder and run this script to see predictions.

---

### 5. `customized_digits.py`

- **Purpose**: Performs hyperparameter optimization for a CNN trained on the Fashion MNIST dataset.
- **Key Features**:
  - Defines the `CustomizedNetwork` class with configurable hyperparameters.
  - Implements a hyperparameter search using a coordinate descent strategy.
  - Trains and evaluates the model for each hyperparameter combination.
- **Usage**: Run this script to find the best hyperparameters for the Fashion MNIST dataset.

---

### 6. `greek_model.py`

- **Purpose**: Adapts the MNIST model for recognizing Greek characters.
- **Key Features**:
  - Defines the `Greeketwork` class, a modified version of `MyNetwork`.
  - Implements a custom `GreekTransform` for preprocessing Greek character images.
  - Evaluates the model on a Greek character dataset.
  - Processes handwritten Greek character images for predictions.
- **Usage**: Run this script to train and evaluate the model on Greek characters.

---

### 7. `model_exam.py`

- **Purpose**: Analyzes the structure and weights of the `MyNetwork` model.
- **Key Features**:
  - Prints the structure and details of the model layers.
  - Visualizes the weights of the first convolutional layer.
  - Applies the filters to an MNIST digit and visualizes the results.
- **Usage**: Run this script to inspect and analyze the trained model.

---

### Link

- This is link to my handwriting Greek letter
- https://drive.google.com/file/d/1wK4eKe5Q4fgjRl-R4lXb4WoP5qeu7xXP/view?usp=sharing

---

## Folder Structure
