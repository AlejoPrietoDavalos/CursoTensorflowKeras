import time, os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input, Dense, Flatten

os.system("clear")
np.random.seed(14)


class ConvClassifier_Keras():
    def __init__(self, InputShape:tuple, N_Filters:list, DimFilters:list, ConvActivations:list, NeuronPerLayers:list, Activations:list, optimizer:object, loss_function:str, metricas:list):
        # Variables de la red convolucional.
        self.InputShape = InputShape
        self.N_Filters = N_Filters
        self.DimFilters = DimFilters                    # Es una lista que tiene dentro tuplas con el tamaño de la matriz del filtro.
        self.ConvActivations = ConvActivations
        
        # Variables de la red clasificadora.
        self.NeuronPerLayers = NeuronPerLayers
        self.Activations = Activations
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metricas = metricas

        self.model = self.CreateModel()
        self.history = None                             # Acá se va a guardar el progreso de la red al ser entrenada.


    def CreateModel(self):
        Layers = [Input(shape=self.InputShape)]         # Creamos un listado donde ir guardando las capas de la red neuronal.
        
        #----------Agregamos las capas convolucionales----------
        for filters, dim_filters, f_act in zip(self.N_Filters, self.DimFilters, self.ConvActivations):
            Layers.append(Conv2D(filters, dim_filters, activation=f_act) (Layers[-1]))
        Layers.append(Flatten() (Layers[-1]))           # Agregamos una capa de Flatten al final para conectar con la red neuronal clasificadora.
        
        #----------Agregamos las capas de la red neuronal----------
        for n_neur, f_act in zip(self.NeuronPerLayers, self.Activations):
            Layers.append(Dense(n_neur, activation=f_act) (Layers[-1]))


        #----------Creamos el modelo----------
        model = Model(Layers[0], Layers[-1])            # Creamos el modelo con esta capa de input y output.
        
        #----------Agregamos el optimizador al modelo y la loss function----------
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metricas)
        model.summary()                                 # Imprimimos en consola la arquitectura de la red.
        return model
    

    def TrainModel(self, x_train, y_train, x_test, y_test, epochs, batch_size, ShufflePerBatch=True, verbose=1):
        t_i = time.time()
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test), shuffle=ShufflePerBatch, verbose=verbose)
        t_f = time.time()
        print(f"\n---> El entrenamiento duró {round(t_f-t_i, 1)}s\n")


    def Evaluate(self, x):
        return self.model.predict(x).argmax()


    def Plot_LOSS(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train LOSS: {self.history.history['loss'][-1]}.\nLast Val LOSS: {self.history.history['val_loss'][-1]}")
        plt.plot(self.history.history['loss'], label='Train Loss', color=ColorTrain)
        plt.plot(self.history.history['val_loss'], label='Val Loss', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()

    def Plot_MSE(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train MSE: {self.history.history['mse'][-1]}.\nLast Val MSE: {self.history.history['val_mse'][-1]}")
        plt.plot(self.history.history['mse'], label='Train MSE', color=ColorTrain)
        plt.plot(self.history.history['val_mse'], label='Val MSE', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.show()
    
    def Plot_ACC(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train Accuracy: {self.history.history['acc'][-1]}.\nLast Val Accuracy: {self.history.history['val_acc'][-1]}")
        plt.plot(self.history.history['acc'], label='Train Accuracy', color=ColorTrain)
        plt.plot(self.history.history['val_acc'], label='Val Accuracy', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()


























#--------------------Importamos los datos--------------------
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_classes = 10

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


from Herramientas import UtilsPlot
util = UtilsPlot()



