import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

# Create a results directory for saving output GIFs
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define the Multi-Layer Perceptron (MLP) class for forward and backward propagation
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)  # Set a fixed seed for reproducibility
        self.lr = lr
        self.activation_fn = activation

        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.bias_output = np.zeros((1, output_dim))

    def activation(self, x):
        # Activation functions: tanh, relu, and sigmoid
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Derivatives of activation functions used for backpropagation
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)

    def forward(self, X):
        # Perform a forward pass to compute hidden layer activations and output
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward(self, X, y):
        # Perform backpropagation to update the weights and biases
        loss_gradient = 2 * (self.a2 - y) / y.size
        
        # Compute gradients for output to hidden weights
        d_weights_hidden_output = np.dot(self.a1.T, loss_gradient * self.activation_derivative(self.z2))
        d_bias_output = np.sum(loss_gradient * self.activation_derivative(self.z2), axis=0, keepdims=True)
        
        # Compute gradients for hidden to input weights
        d_hidden = np.dot(loss_gradient * self.activation_derivative(self.z2), self.weights_hidden_output.T) * self.activation_derivative(self.z1)
        d_weights_input_hidden = np.dot(X.T, d_hidden)
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output -= self.lr * d_weights_hidden_output
        self.bias_output -= self.lr * d_bias_output
        self.weights_input_hidden -= self.lr * d_weights_input_hidden
        self.bias_hidden -= self.lr * d_bias_hidden

# Function to generate synthetic data
def generate_data(n_samples=500):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular decision boundary
    y = y.reshape(-1, 1)
    
    # Normalize data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y

# Function to update the visualization at each step
def update(frame, mlp, ax_hidden, ax_input, ax_gradient, X, y):
    # Clear the axes to refresh the plot at each step
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform forward and backward pass for training the network
    mlp.forward(X)
    mlp.backward(X, y)

    # Plot Hidden Space visualization
    hidden_features = mlp.a1
    num_hidden_neurons = hidden_features.shape[1]

    if num_hidden_neurons == 1:
        # Use green and purple for the color map in hidden layer scatter plot
        ax_hidden.scatter(hidden_features[:, 0], np.zeros_like(hidden_features[:, 0]), c=y.ravel(), cmap='winter', alpha=0.7)
    elif num_hidden_neurons == 2:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='winter', alpha=0.7)
    elif num_hidden_neurons >= 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='winter', alpha=0.7)

    # Compute the decision hyperplane in the hidden space
    if num_hidden_neurons == 2:
        x1_vals = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 100)
        x2_vals = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 100)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
        grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        preds = mlp.forward(grid_points)
        preds = preds.reshape(x1_grid.shape)
        
        # Plot decision boundary where the output is zero
        ax_hidden.contour(x1_grid, x2_grid, preds, levels=[0], linewidths=2, colors='black')

    ax_hidden.set_title(f"Hidden Space at Step {frame}")

    # Plot Input Space visualization (Decision boundary)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Use MLP to predict for grid points and plot decision boundary
    preds = mlp.forward(grid)
    preds = (preds > 0).astype(int).reshape(xx.shape)

    ax_input.contourf(xx, yy, preds, alpha=0.3, cmap='winter')  # Decision boundary in winter colormap
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='winter', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame}")

    # Plot Gradient Visualization (showing weights as edges)
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.set_title(f"Gradients at Step {frame}")

    input_labels = ['x1', 'x2']
    hidden_labels = [f'h{i+1}' for i in range(mlp.weights_input_hidden.shape[1])]
    output_label = 'y'

    # Positioning of nodes for the gradient visualization
    input_x, hidden_x, output_x = 0.1, 0.5, 0.9
    input_y_step = 1 / (len(input_labels) + 1)
    hidden_y_step = 1 / (len(hidden_labels) + 1)

    # Plot input layer nodes and label them
    for i, input_label in enumerate(input_labels):
        y_pos = 1 - (i + 1) * input_y_step
        ax_gradient.scatter(input_x, y_pos, s=400, color='green')  # Input node in green
        ax_gradient.text(input_x - 0.05, y_pos, input_label, fontsize=12, color='green', ha='right', va='center')

    # Plot hidden layer nodes, connections, and label them
    for j, hidden_label in enumerate(hidden_labels):
        hidden_y = 1 - (j + 1) * hidden_y_step
        ax_gradient.scatter(hidden_x, hidden_y, s=400, color='green')  # Hidden nodes in green
        ax_gradient.text(hidden_x - 0.05, hidden_y, hidden_label, fontsize=12, color='green', ha='right', va='center')
        
        # Visualize the weights as edges (connections)
        for i in range(len(input_labels)):
            input_y = 1 - (i + 1) * input_y_step
            weight = mlp.weights_input_hidden[i, j]
            ax_gradient.plot([input_x, hidden_x], [input_y, hidden_y],
                             color='orange', lw=np.abs(weight) * 5, alpha=0.5)  # Orange color for connections

    # Plot output node, connections from hidden layer, and label them
    ax_gradient.scatter(output_x, 0.5, s=400, color='green')  # Output node in green
    ax_gradient.text(output_x + 0.05, 0.5, output_label, fontsize=12, color='green', ha='left', va='center')
    
    for j in range(len(hidden_labels)):
        hidden_y = 1 - (j + 1) * hidden_y_step
        weight = mlp.weights_hidden_output[j, 0]
        ax_gradient.plot([hidden_x, output_x], [hidden_y, 0.5],
                         color='#FF4500', lw=np.abs(weight) * 5, alpha=0.5)  # Darker orange for output connections

# Function to visualize the neural network training progress
def visualize(activation, lr, hidden_dim, step_num):
    X, y = generate_data()  # Generate synthetic data
    mlp = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1, lr=lr, activation=activation)

    # Create a figure and subplots for visualization
    matplotlib.use('agg')  # Use agg backend for non-GUI environment
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create the animation for visualization
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=30)
    plt.close()

# Main block to run the visualization function
if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    hidden_dim = 3
    step_num = 1000
    visualize(activation, lr, hidden_dim, step_num)
