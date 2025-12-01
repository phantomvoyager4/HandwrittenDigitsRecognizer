import numpy as np
import idx2numpy


def data_handling(pathimages, pathlabels):
    images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
    images = images / 255.0
    labels = idx2numpy.convert_from_file(pathlabels)
    return images, labels

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
    
    def fpropagation(self, input):
        self.output = np.dot(input, self.weights) + self.biases  
        return self.output

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



pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
images, labels = data_handling(pathimagess, pathlabelss)

hidden_layer1 = Layer(n_inputs=784, n_neurons=128)
activation = Activation()
hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
output_layer = Layer(n_inputs=64, n_neurons=10)
output_layer_by_probability = Softmax()
losscalc = Loss_CategoricalCrossentropy()


# Layer1 -> ReLU -> Layer2 -> ReLU -> OutputLayer -> Softmax
first_layer = hidden_layer1.fpropagation(input=images)
first_activate = activation.forward(input=first_layer)
second_layer = hidden_layer2.fpropagation(input=first_activate)
second_activation = activation.forward(input=second_layer)
output_layer_output = output_layer.fpropagation(input=second_activation)
results =  output_layer_by_probability.forward(input=output_layer_output)
loss = losscalc.forward(y_pred=results, y_true=labels)
print("Loss:", loss) # ~-ln(1/10)
