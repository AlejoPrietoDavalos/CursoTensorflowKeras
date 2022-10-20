import numpy as np
from RedesConvolucionales_Keras import ConvClassifier_Keras
from keras.utils import to_categorical
from keras.optimizers import Adadelta

np.random.seed(14)


#--------------------Herramientas de Ploteo--------------------
from Herramientas import UtilsPlot
util = UtilsPlot()


#--------------------Importamos el Dataset--------------------
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_classes = 10


#--------------------Preparamos los Datos--------------------
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


#--------------------Creo el Optimizador--------------------
lr = 0.5
rho = 0.95
optimizer = Adadelta(learning_rate=lr, rho=rho)


#--------------------Creo el Modelo--------------------
model = ConvClassifier_Keras(
    x_train.shape[1:],
    [32, 64],
    [(3,3), (3,3)],
    ['relu', 'relu'],
    False,
    [64, 64, n_classes],
    ['relu', 'relu', 'softmax'],
    optimizer,
    'categorical_crossentropy',
    ['acc', 'mse']
)


#--------------------Entreno el Modelo--------------------
epochs = 5
batch_size = 32
model.TrainModel(x_train, y_train, x_test, y_test, epochs, batch_size)


#--------------------Hago un Plot de las Métricas--------------------
model.Plot_ACC()
model.Plot_LOSS()
model.Plot_MSE()






