from AdalinePerceptron import Adaline
import csv
import numpy as np
import matplotlib.pyplot as plt

# I will use the Pima Indians Diabetes dataset for testing
# a dataset with biometrical inputs like bmi,age, ... to determine whether a person has diabetes or not
# all data is from woman over 21 and of pima indian heritage
# The dataset has an possible max accuracy of around 80%

def testAdaline(X_train, Y_train, X_test, Y_test):
    # Prepare the training data as a list of tuples
    input_size = X_train.shape[1]
    training_data = [(x_train[i], Y_train[i]) for i in range(len(Y_train))]
    correct_predictions = 0
    network = Adaline(input_size)
    weights, bias = network.train_perceptron(training_data, 0.1, 0.05, 50)# stop, learning rate, number of epochs
    for x, expected_output in zip(X_test, Y_test):
                prediction = network.predict(x)
                if (prediction >= 0.5 and expected_output == 1) or (prediction < 0.5 and expected_output == 0):
                    correct_predictions += 1
    accuracy = (correct_predictions / len(X_test))* 100
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


def load_dataset(file_path):
    dataset = []
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the first row
        for row in csv_reader:
            dataset.append([float(value) for value in row]) # Read the csv data
    return np.array(dataset)

def preprocess_data(dataset):
    # Split the data into input(X) and output(Y)
    x = dataset[:, :-1]  # All columns except the last one
    y = dataset[:, -1]   # The last column
    #normalize the values for stability
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x, y

def split_data(x, y, train_ratio=0.8):
    # Split into training and test data
    splittingIndex = int(len(x)*train_ratio)
    x_train = x[:splittingIndex]#Split 80% of x into x_train
    x_test = x[splittingIndex:]# Split 20% of x into x_test
    y_train = y[:splittingIndex]#Split 80% of y into y_train
    y_test = y[splittingIndex:]# Split 20% of y into y_test
    return x_train, y_train, x_test, y_test


# Main code
dataset = load_dataset('diabetes.csv')
x, y = preprocess_data(dataset)
x_train, y_train, x_test, y_test = split_data(x, y)


testAdaline(x_train, y_train, x_test, y_test)