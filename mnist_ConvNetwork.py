import numpy as np
from RedesConvolucionales_Keras import ConvClassifier_Keras
from keras.utils import to_categorical
from keras.optimizers import Adadelta

np.random.seed(14)


#--------------------Herramientas de Ploteo--------------------
from Herramientas import UtilsPlot
util = UtilsPlot()

#--------------------Importando los datos.
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_classes = 10                                          # np.max(np.unique(y_train)) + 1



#--------------------Preparando los datos--------------------
from keras.utils import to_categorical
x_train = x_train.astype('float32') / 255.0             # Convertimos los valores en float32 y mapeamos al intervalo [0,1] cada pixel.
x_test = x_test.astype('float32') / 255.0               # Convertimos los valores en float32 y mapeamos al intervalo [0,1] cada pixel.


x_train.shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test.shape = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


y_train = to_categorical(y_train, n_classes)            # One-Hot encoding
y_test = to_categorical(y_test, n_classes)              # One-Hot encoding


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
    [1024, 1024, n_classes],
    ['relu', 'relu', 'softmax'],
    optimizer,
    'categorical_crossentropy',
    ['acc', 'mse']
)


#--------------------Entreno el Modelo--------------------
epochs = 10
batch_size = 256
model.TrainModel(x_train, y_train, x_test, y_test, epochs, batch_size)


#--------------------Hago un Plot de las Métricas--------------------
model.Plot_ACC()
model.Plot_LOSS()
model.Plot_MSE()






import cv2

list_aciertos = [0]*10
for n in range(10):
    for i in range(10):
        imagen = cv2.imread(f"Numeros/{n}_{i}.png", 0)
        imagen = imagen.astype("float32") / 255.0
        imagen.shape = (1, 28, 28, 1)
        prediccion = model.Evaluate(imagen)
        if prediccion == n:
            list_aciertos[n] += 1
print(f"Acertamos {sum(list_aciertos)} de 100 números.")
print("Cantidad de aciertos por número.")
for n in range(10):
    print(f"{n}: Tuvo {list_aciertos[n]} aciertos.")



















