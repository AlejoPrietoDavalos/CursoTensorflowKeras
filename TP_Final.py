import time, os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from keras.utils import to_categorical
from RedesConvolucionales_Keras import ConvClassifier_Keras
np.random.seed(14)

# Utilizaremos solo imagenes de 7 personas con mas de 70 imagenes disponibles.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)

#----------Cargamos los datos----------
X = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names      # El nombre de cada uno de los tarjets

os.system("clear")

#----------Pequeña exploración sobre el numero de datos----------
n_datos_por_clase = [0]*7
for i in range(len(y)):
    n_datos_por_clase[y[i]] += 1            # Esto devuelve [77, 236, 121, 530, 109, 71, 144], vemos una cantidad despareja de datos en algunas categorias.
for j in range(10):
    print("----->Número de datos por clase:")
    for i in range(len(target_names)):
        print(f"{target_names[i]}: {n_datos_por_clase[i]}")
    print("\nVemos una cantidad despareja de datos entre categorías.")
    print(f"El script reanudará en {10-j} segundos")
    time.sleep(1)
    os.system("clear")
    


'''IMPORTANTE a): Como hay un número tan desparejo de datos, no estaria bien (para mi) separar el 80% de los datos arbitrariamente.
Creo mas conveniente tomar el 80% de cada categoría y juntarlas para que ese 80% sea representativo del conjunto.
A continuación vamos a hacer eso:
1) Buscamos los indices que corresponden a cada una de las categorías.
2) Contamos cuantos datos hay para esa categoría y nos quedamos con el 80% de train y el resto de test.


IMPORTANTE b): Luego de hacer pruebas con lo que dije arriba, veo que la red neuronal predice muy bien aquellas
categorías con mayor cantidad de datos, y peor al resto. Por lo cual me voy a quedar con 71 datos de cada categoría (El menor número
de datos de una de las categorías), para que sea parejo.'''

#----------Acomodamos los Datos----------
x_train, y_train, x_test, y_test = [], [], [], []
for n_cat, n_datos in enumerate([77, 144, 121, 144, 109, 71, 144]):         # Estos son los datos que me voy a quedar por categoría.
    ind = np.where(y==n_cat)[0][0:n_datos]         # Busco los indices en los cuales aparece esa categoria y me quedo con cierta cantidad de datos.
    num_train_data = int(len(ind)*0.8)  # Me quedo con el 80% que corresponde a los datos de entrenamiento
    
    ind_train = ind[0:num_train_data]
    ind_test = ind[num_train_data:len(ind)]

    x_train += list(X[ind_train])
    y_train += [n_cat for _ in range(num_train_data)]
    x_test += list(X[ind_test])
    y_test += [n_cat for _ in range(len(ind) - num_train_data)]

#----------Convertimos todo a array de numpy----------
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


#----------Acomodamos los datos----------
x_train = x_train.astype("float32")         # Ya venian normalizados entre 0 y 1.
x_test = x_test.astype("float32")           # Ya venian normalizados entre 0 y 1.

x_train.shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test.shape = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#----------Instanciamos la clase del modelo----------
Conv = ConvClassifier_Keras(
    InputShape = x_train.shape[1:],
    N_Filters = [32, 32, 32],
    DimFilters = [(3, 3), (3, 3), (3, 3)],
    BatchNormalization=True,
    ConvActivations = ['relu', 'relu', 'relu'],
    NeuronPerLayers = [512, 7],
    Activations = ['relu', 'softmax'],
    Dropout=True,
    loss_function = 'categorical_crossentropy',
    metricas = ['acc', 'mse'],
    TargetNames=target_names
)

#----------Generamos el optimizador del modelo----------
lr = 0.7
rho = 0.95
Conv.SetOptimizer_Adadelta(lr, rho)

#----------Creamos el Modelo----------
Conv.CreateModel()

#----------Entrenamos el Modelo----------
epochs = 50
batch_size = 32
Conv.TrainModel(x_train, y_train, x_test, y_test, epochs, batch_size, True)

Conv.SaveModel("TPFinal_AlejoPrieto")
