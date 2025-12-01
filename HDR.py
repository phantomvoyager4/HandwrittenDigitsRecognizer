import numpy as np
import idx2numpy


def data_handling(pathimages, pathlabels):
    images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
    images = images / 255.0
    labels = idx2numpy.convert_from_file(pathlabels)
    # map = np.identity(10)
    # labels_encoded = map[labels] 
    print(images.shape, labels.shape) #(60000, 784) (60000, 10)
    return images, labels



pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
data_handling(pathimagess, pathlabelss)

#NEURONS LOGIC
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
    
    def fpropagation(self, input):
        self.output = np.dot(input, self.weights) + self.biases  
        return self.output

#ACTIVATE NEURONS USING RELU
class Activation:
    def forward(self, input):
        self.output= np.maximum(0, input)
        return self.output

class Softmax:
    def forward (self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        sum = np.sum(exp, axis=1, keepdims=True)
        self.output = exp/sum
        return self.output
    
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 0.0000001, 0.99999999)
        predicted = y_pred[range(y_pred.shape[0]), y_true]
        loss = -np.log(predicted)
        mean_loss = np.mean(loss)
        return mean_loss
