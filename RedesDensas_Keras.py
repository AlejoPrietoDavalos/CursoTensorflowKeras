import time, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.system("clear")          # Por que me está molestando el cartel, ARREGLAR<<<<
np.random.seed(14)          # Para que la generación de números aleatorios sea siempre la misma, y poder probar diferentes modelos con resultados comparables.


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten


class MLP_Dense():
    def __init__(self, InputShape:int, NeuronPerLayers:list, Activations:list, optimizer:object, loss_function:str, metricas:list):
        '''
        InputShape me permite pasar una tupla con las dimenciones de la imagen, esto esta bueno por que no necesito trasformar la imagen
        en un vector de números. Puedo trabajar con la imágen en forma de matriz.
        '''
        self.InputShape = InputShape
        self.NeuronPerLayers = NeuronPerLayers
        self.Activations = Activations
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metricas = metricas

        self.model = self.CreateModel()
        self.history = None         # Acá se va a guardar el progreso de la red al ser entrenada.

    def CreateModel(self):
        Layers = [Input(shape=self.InputShape)]         # Creamos un listado donde ir guardando las capas de la red neuronal.
        
        for n_neur, f_act in zip(self.NeuronPerLayers, self.Activations):
            #NewLayer = Dense(n_neur, activation=f_act) (Layers[-1])
            #Layers.append(NewLayer)
            Layers.append(Dense(n_neur, activation=f_act) (Layers[-1]))
        
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



#from keras.metrics import MSE
#from keras.losses import categorical_crossentropy

#--------------------Importando los datos.
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_input = int(np.prod(x_train.shape[1:]))
n_classes = 10                                          # np.max(np.unique(y_train)) + 1


#--------------------Preparando los datos--------------------
from keras.utils import to_categorical
x_train = x_train.astype('float32') / 255.0             # Convertimos los valores en float32 y mapeamos al intervalo [0,1] cada pixel.
x_test = x_test.astype('float32') / 255.0               # Convertimos los valores en float32 y mapeamos al intervalo [0,1] cada pixel.

y_train = to_categorical(y_train, n_classes)            # One-Hot encoding
y_test = to_categorical(y_test, n_classes)              # One-Hot encoding

x_train.shape = (x_train.shape[0], n_input)             # Aplastamos la matriz de pixeles a un vector.
x_test.shape = (x_test.shape[0], n_input)               # Aplastamos la matriz de pixeles a un vector.



#--------------------Creo el optimizador para el modelo--------------------
from keras.optimizers import Adadelta
lr = 1
rho = 0.95
optimizer = Adadelta(learning_rate=lr, rho=rho)


#--------------------Creamos el modelo--------------------
MLP = MLP_Dense(n_input, [1024, 1024, n_classes], ['relu', 'relu', 'softmax'], optimizer, 'categorical_crossentropy', ['acc', 'mse'])


#--------------------Entrenamos el modelo--------------------
epochs = 10
batch_size = 256

MLP.TrainModel(x_train, y_train, x_test, y_test, epochs, batch_size)
MLP.Plot_LOSS()
MLP.Plot_ACC()
MLP.Plot_MSE()


#import cv2
#
#acierto = 0
#for n in range(10):
#    for i in range(10):
#        imagen = cv2.imread(f"Numeros/{n}_{i}.png", 0)
#        imagen = imagen.flatten().astype("float32") / 255.0




