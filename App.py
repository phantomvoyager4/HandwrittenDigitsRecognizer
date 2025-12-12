from HDR import Layer, Activation, Softmax
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np

class App:
    def __init__(self):
        #network structure transfer
        self.model = np.load('models_data_storage/model_1/model_1_95.28_0.5.npz')
        self.layer1 = Layer(n_inputs=784, n_neurons=128)
        self.layer2 = Layer(n_inputs=128, n_neurons=64)
        self.output_layer = Layer(n_inputs=64, n_neurons=10)
        self.layer1.weights = self.model['w1']
        self.layer1.biases = self.model['b1']
        self.layer2.weights = self.model['w2']
        self.layer2.biases = self.model['b2']
        self.output_layer.weights = self.model['w3']
        self.output_layer.biases = self.model['b3']
        

        app_window = Tk()
        app_window.geometry('500x500')
        app_window.title('Digit recognizer')

        self.C = Canvas(app_window, height=300, width=300, bg='white')
        self.C.pack()

        self.image = Image.new(mode='RGB', size=(300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last.y = None

        self.predict = Button(app_window, text='Predict')
        self.predict.pack()

        self.clear = Button(app_window, text='Clear')
        self.clear.pack()

        self.textlabel = Label(app_window, text='Draw your digit')

        #run
        app_window.mainloop()

        
        