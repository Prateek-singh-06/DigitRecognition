# DigitRecognition - Text Recognition using Neural Networks in C++
## Introduction

The DigitRecognition project is aimed at developing a neural network-based model for handwritten digit recognition using the C++ programming language, without relying on any external libraries. The focus is on implementing a neural network architecture with two layers to recognize digits from the MNIST dataset.
## Implementation Details

### Neural Network Architecture

The neural network comprises a hidden layer with 16 neurons and an output layer with 10 neurons. The Leaky ReLU activation function is employed for the hidden layer, and the softmax activation function is used for multiclass classification in the output layer. The neural network weights are initialized with random values.
### Training Process

The training process involves reading the MNIST training dataset, applying the backpropagation algorithm to iteratively update the neural network weights. Leaky ReLU is used for the hidden layer, and softmax for the output layer to calculate the loss. Stochastic Gradient Descent (SGD) optimization is employed for weight updates.

### Forward Propagation

1. **Input Pass:** Feed input data into the neural network through the input layer.
2. **Weighted Sum and Activation:** Multiply input values by weights, sum the results, and pass through activation functions for non-linearity.
3. **Output Prediction:** Compute values through subsequent layers until the final layer produces predictions or classifications.
4. **Loss Computation:** Compare predictions with actual target values; calculate loss using the -summation of Ai*LogOi loss function.

### Backpropagation

1. **Gradient Descent Initialization:** Initialize weights and biases with random values.
2. **Backward Pass (Backpropagation):** Calculate gradients of loss with respect to weights and biases using the chain rule of calculus.
3. **Weight Update:** Adjust weights and biases opposite to the gradient direction using an optimization algorithm like gradient descent.
4. **Iterative Process:** Repeat steps 2 and 3 iteratively over the training dataset until the model converges.
### Output
![output](https://github.com/Prateek-singh-06/DigitRecognition/assets/113665258/85fd57ce-c315-4e23-a3ef-05963657da98)



## Usage

To run the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd DigitRecognition`
3. Compile the C++ code: `g++ -o DigitRecognition DigitRecognition.cpp`
4. Run the executable: `./DigitRecognition`
