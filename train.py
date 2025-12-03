import numpy as np
from HDR import data_handling, Layer, Activation, Optimizer, Backpropagation

pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
images, labels = data_handling(pathimagess, pathlabelss)


hidden_layer1 = Layer(n_inputs=784, n_neurons=128)
activation1 = Activation()
activation2 = Activation()
hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
output_layer = Layer(n_inputs=64, n_neurons=10)
loss_activation = Backpropagation()
optimization = Optimizer(0.5)

for epoch in range (400):
    first_layer = hidden_layer1.fpropagation(images)
    first__layer_activation = activation1.forward(first_layer)
    second_layer = hidden_layer2.fpropagation(first__layer_activation)
    second_layer_activation = activation2.forward(second_layer)
    output_layer_output = output_layer.fpropagation(second_layer_activation)

    loss = loss_activation.forward(output_layer_output, labels)

    highestchangebyrow = np.argmax(output_layer_output, axis=1)
    comparison = np.equal(highestchangebyrow, labels)
    accuracy = np.mean(comparison)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

    backwards = loss_activation.backward(labels=labels)
    olpass = output_layer.backward(backwards)
    aslpass = activation2.backward(olpass)
    slpass = hidden_layer2.backward(aslpass)
    aflpass = activation1.backward(slpass)
    flpass = hidden_layer1.backward(aflpass)

    optimization.adjust_parameters(hidden_layer1)
    optimization.adjust_parameters(hidden_layer2)
    optimization.adjust_parameters(output_layer)
