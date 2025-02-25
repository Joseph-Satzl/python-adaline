from flask import Flask, render_template, request, jsonify
from AdalinePerceptron import Adaline
import numpy as np
import csv

app = Flask(__name__)
adaline = None
learning_rate = 0.1
epochs = 50


def load_dataset(filepath):
    dataset = []
    with open(filepath) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        dataset = [list(map(float, row)) for row in csv_reader]
    return np.array(dataset)

def preprocess_data(dataset):
    x = dataset[:, :-1]
    y = dataset[:, -1]
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x, y

def split_data(x, y, train_ratio=0.8):
    splitting_index = int(len(x) * train_ratio)
    return x[:splitting_index], x[splitting_index:], y[:splitting_index], y[splitting_index:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global adaline, learning_rate, epochs
    data = request.json
    print(data)
    learning_rate = float(data['learningRate'])
    epochs = int(data['epochs'])

    dataset = load_dataset('diabetes.csv')
    x, y = preprocess_data(dataset)
    x_train, x_test, y_train, y_test = split_data(x, y)
    adaline = Adaline(x_train.shape[1])
    adaline.train_perceptron(list(zip(x_train, y_train)), 0.1, learning_rate, epochs)

    return jsonify({'message': 'Training completed successfully!'})
                    
@app.route('/test', methods=['GET'])
def test():
    global adaline, learning_rate, epochs
    dataset = load_dataset('diabetes.csv')
    x, y = preprocess_data(dataset)
    x_train, x_test, y_train, y_test = split_data(x, y)
    adaline = Adaline(x_train.shape[1])
    w, bias = adaline.train_perceptron(list(zip(x_train, y_train)), 0.1, learning_rate, epochs)
    weights = w.tolist()
    correct_predictions = sum(adaline.predict(x) == y for x, y in zip(x_test, y_test))
    accuracy = (correct_predictions / len(y_test)) * 100
    return jsonify({'accuracy': accuracy, 'weights' : weights , 'bias': bias})

if __name__ == '__main__':
    app.run(debug=True)
