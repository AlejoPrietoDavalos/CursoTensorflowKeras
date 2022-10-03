import numpy as np
from RedesDensas_Keras import MLP_Dense

np.random.seed(14)          # Para que la generación de números aleatorios sea siempre la misma, y poder probar diferentes modelos con resultados comparables.


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
lr = 0.5
rho = 0.95
optimizer = Adadelta(learning_rate=lr, rho=rho)



#--------------------Creamos el modelo--------------------
MLP = MLP_Dense(n_input, [1024, 1024, 1024, 1024, n_classes], ['relu', 'relu', 'relu', 'relu', 'softmax'], optimizer, 'categorical_crossentropy', ['acc', 'mse'])



#--------------------Entrenamos el modelo--------------------
epochs = 10
batch_size = 256

MLP.TrainModel(x_train, y_train, x_test, y_test, epochs, batch_size)
MLP.Plot_ACC()
MLP.Plot_LOSS()
MLP.Plot_MSE()





import cv2

list_aciertos = [0]*10
for n in range(10):
    for i in range(10):
        imagen = cv2.imread(f"Numeros/{n}_{i}.png", 0)
        imagen = imagen.flatten().astype("float32") / 255.0
        imagen.shape = (1, imagen.shape[0])
        prediccion = MLP.Evaluate(imagen)
        if prediccion == n:
            list_aciertos[n] += 1
print(f"Acertamos {sum(list_aciertos)} de 100 números.")
print("Cantidad de aciertos por número.")
for n in range(10):
    print(f"{n}: Tuvo {list_aciertos[n]} aciertos.")



