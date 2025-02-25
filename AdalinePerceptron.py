import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self, input_size:int) -> None:
        self.W = np.random.uniform(0, 1, input_size)  # input_size means the number of weights
        self.b = np.random.uniform(0, 1)  

    def predict(self, x: np.array):
        return 1 if np.dot(self.W, x) + self.b >= 0.5 else 0
    
    def train_perceptron(self, training_data: list, stop: float, learning_rate: float, n: int, plot: bool = False):
        errors = [stop + 1]  # Start with a large error to ensure the loop runs
        epoch = 0
        error_history = [] # for plotting
        while errors[-1] > stop and epoch < n: # Checks if either the last error is below the stop value or the number of epochs is higher than n
            Error = []
            for x, target in training_data:
                y = np.dot(self.W, x) + self.b  # No activation function for Adaline
                error = target - y
                self.W += learning_rate * error * x
                self.b += learning_rate * error  # Bias update includes learning rate
                Error.append(error ** 2)
            errors.append(sum(Error))
            error_history.append(sum(Error)) # for plotting
            epoch += 1


        # Plot error history
        if plot:
            plt.plot(range(1, len(error_history) + 1), error_history, marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Sum-squared-error')
            plt.title('Training Error over Epochs')
            plt.grid()
            plt.show()
        return self.W, self.b
    
    def test_perceptron(self, gate_name: str, epochs: int):  # Test function for binary logic gates
        
        gate_data = {
            'AND': [
                (np.array([-1, -1]), -1),
                (np.array([-1, 1]), -1),
                (np.array([1, -1]), -1),
                (np.array([1, 1]), 1),
            ],
            'OR': [
                (np.array([-1, -1]), -1),
                (np.array([-1, 1]), 1),
                (np.array([1, -1]), 1),
                (np.array([1, 1]), 1),
            ],
            'NOT': [
                (np.array([-1]), 1),
                (np.array([1]), -1),
            ]
        }

        training_data = gate_data[gate_name]
        input_size = len(training_data[0][0])
        correct_predictions = 0
    
        for _ in range(epochs):
            perceptron = Adaline(input_size)
            perceptron.train_perceptron(training_data)
            for x, expected_output in training_data:
                prediction = perceptron.predict(x)
                if (prediction >= 0 and expected_output == 1) or (prediction < 0 and expected_output == -1): # Threshold function
                    correct_predictions += 1

        accuracy = (correct_predictions / (epochs * len(training_data))) * 100
        return f"{gate_name} Gate Accuracy: {accuracy:.2f}%"

# Example usage:
#input_size = len(training_data[0][0])
#network = Adaline(input_size)
#print(network.test_perceptron('AND', 1))