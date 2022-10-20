import time, os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

os.system("clear")
np.random.seed(14)


class ConvClassifier_Keras():
    def __init__(self, InputShape:tuple, N_Filters:list, DimFilters:list, ConvActivations:list, BatchNormalization:bool, NeuronPerLayers:list, Activations:list, Dropout:bool, loss_function:str, metricas:list, TargetNames:list):
        ''''Notas: Poner % de droppout variable por capa (leer el paper del flaco).'''
        # Variables de la red convolucional.
        self.InputShape = InputShape
        self.N_Filters = N_Filters
        self.DimFilters = DimFilters                    # Es una lista que tiene dentro tuplas con el tamaño de la matriz del filtro.
        self.ConvActivations = ConvActivations
        self.BatchNormalization = BatchNormalization
        
        # Variables de la red clasificadora.
        self.NeuronPerLayers = NeuronPerLayers
        self.Activations = Activations
        self.Dropout = Dropout
        
        # Optimizadores y métricas.
        self.optimizer = None                           # Vamos a guardar el tipo de optimizador.
        self.loss_function = loss_function
        self.metricas = metricas
        self.TargetNames = TargetNames

        self.DirOutputs = f"Model: {str(datetime.now()).split('.')[0]}"

        # Creamos el modelo y el histórico de su entrenamiento.
        self.model = None
        self.history = None                             # Acá se va a guardar el progreso de la red al ser entrenada.

    def CreateModel(self):
        Layers = [Input(shape=self.InputShape)]         # Creamos un listado donde ir guardando las capas de la red neuronal.
        
        #----------Agregamos las capas convolucionales----------
        for filters, dim_filters, f_act in zip(self.N_Filters, self.DimFilters, self.ConvActivations):
            if self.BatchNormalization:         # Si quisieramos agregar capas de BatchNormalization.
                Layers.append(BatchNormalization() (Layers[-1]))
            Layers.append(Conv2D(filters, dim_filters, activation=f_act) (Layers[-1]))
        Layers.append(Flatten() (Layers[-1]))           # Agregamos una capa de Flatten al final para conectar con la red neuronal clasificadora.
        
        #----------Agregamos las capas de la red neuronal----------
        for i, (n_neur, f_act) in enumerate(zip(self.NeuronPerLayers, self.Activations)):
            Layers.append(Dense(n_neur, activation=f_act) (Layers[-1]))
            if self.Dropout and i+1<len(self.Activations):
                Layers.append(Dropout(0.25) (Layers[-1]))


        #----------Creamos el modelo----------
        model = Model(Layers[0], Layers[-1])            # Creamos el modelo con esta capa de input y output.
        
        #----------Agregamos el optimizador al modelo y la loss function----------
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metricas)
        
        #----------Guardamos el summary----------
        os.mkdir(self.DirOutputs)                       # En esta carpeta vamos a guardar todos los outputs de la red.
        model.summary()                                 # Imprimimos en consola la arquitectura de la red.
        self.SaveSummary(model)
        
        self.model = model

    def TrainModel(self, x_train, y_train, x_test, y_test, epochs, batch_size, data_augmentation=False, ShufflePerBatch=True, verbose=1):
        t_i = time.time()
        if not(data_augmentation):
            self.history = self.model.fit(
                x=x_train,
                y=y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test,y_test),
                shuffle=ShufflePerBatch,
                verbose=verbose)
        else:
            datagen = self.DataAugmentation()
            datagen.fit(x_train)
            steps_per_epoch = x_train.shape[0] // batch_size
            x_train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

            self.history = self.model.fit(
                x_train_generator,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test,y_test),
                workers=5,
                shuffle=ShufflePerBatch,
                verbose=verbose)
        t_f = time.time()
        print(f"\n---> El entrenamiento duró {round(t_f-t_i, 1)}s\n")
        
        self.SavePlotsMetrics(x_test, y_test)
        self.SaveParameters(len(x_train), len(x_test), epochs, batch_size, data_augmentation)

    def SaveModel(self, ModelName:str):
        """ Recibe el nombre del modelo (sin extensión) y lo guarda en la carpeta de outputs.
        ModelName {str}: Nombre del modelo sin extensión."""
        self.model.save(self.DirOutputs+f"/{ModelName}.h5")

    def Evaluate(self, x):
        return self.model.predict(x).argmax()

    def DataAugmentation(self):
        '''Esto es un setteo medio genérico, ver si se puede optimizar o si dependiendo del problema a solucionar
        hay parámetros que se puedan tocar.'''
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        
        return datagen

    def SetOptimizer_Adadelta(self, lr, rho=0.95):
        self.optimizer = Adadelta(learning_rate=lr, rho=rho)

    def SavePlotsMetrics(self, x_test, y_test):
        self.Plot_LOSS()
        self.Plot_MSE()
        self.Plot_ACC()
        self.Plot_HotMap(x_test, y_test)
    
    def Plot_LOSS(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train LOSS: {self.history.history['loss'][-1]}.\nLast Val LOSS: {self.history.history['val_loss'][-1]}.")
        plt.plot(self.history.history['loss'], label='Train Loss', color=ColorTrain)
        plt.plot(self.history.history['val_loss'], label='Val Loss', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.DirOutputs+"/Plot - LOSS.png")

    def Plot_MSE(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train MSE: {self.history.history['mse'][-1]}.\nLast Val MSE: {self.history.history['val_mse'][-1]}.")
        plt.plot(self.history.history['mse'], label='Train MSE', color=ColorTrain)
        plt.plot(self.history.history['val_mse'], label='Val MSE', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.savefig(self.DirOutputs+"/Plot - MSE.png")

    def Plot_ACC(self, ColorTrain="steelblue", ColorVal="Crimson"):
        plt.figure(dpi=200)
        plt.title(f"Last Train Accuracy: {self.history.history['acc'][-1]}.\nLast Val Accuracy: {self.history.history['val_acc'][-1]}.")
        plt.plot(self.history.history['acc'], label='Train Accuracy', color=ColorTrain)
        plt.plot(self.history.history['val_acc'], label='Val Accuracy', color=ColorVal)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig(self.DirOutputs+"/Plot - ACC.png")

    def Plot_HotMap(self, x_test, y_test):
        """ Mapa de calor, en el eje vertical está el input de la red, y en el horizontal su tarjet de llegada."""
        y_pred = self.model.predict(x_test, verbose=1)
        y_true = np.argmax(y_test, axis=1)
        y_model = np.argmax(y_pred, axis=1)
        CM = confusion_matrix(y_true, y_model)

        plt.figure(dpi=200)
        plt.title(f"Matriz de Confución: Con {len(self.TargetNames)} categorías.")
        plt.imshow(CM, cmap='jet')
        plt.xlabel("Input de la Red")
        plt.ylabel("Frecuencia Output de la Red")
        plt.xticks(np.arange(len(self.TargetNames)), self.TargetNames, rotation=90)       # Set text labels and properties.
        plt.yticks(np.arange(len(self.TargetNames)), self.TargetNames, rotation=0)        # Set text labels and properties.
        plt.colorbar()
        plt.savefig(self.DirOutputs+"/Plot - HotMap.png")

    def SaveSummary(self, model):
        """ Guardamos la estructura de la red en un txt."""
        with open(f"{self.DirOutputs}/ModelSummary.txt", 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def SaveParameters(self, NumTrainData, NumTestData, epochs, batch_size, data_augmentation):
        with open(f"{self.DirOutputs}/ModelParameters.txt", 'w') as f:
            f.writelines(f"----->Parámetros de la Red<-----\n")
            f.writelines(f"Cant. Train Data: {NumTrainData} [{round((100*NumTrainData)/(NumTrainData+NumTestData), 1)}%]\n")
            f.writelines(f"Cant. Test Data: {NumTestData} [{round((100*NumTestData)/(NumTrainData+NumTestData), 1)}%]\n")
            f.writelines(f"Epochs: {epochs}\n")
            f.writelines(f"Batch Size: {batch_size}\n")
            f.writelines(f"Data Augmentation: {data_augmentation}\n")
            f.writelines(f"Learning Rate: {self.optimizer.get_config()['learning_rate']}\n\n")
            f.writelines(f"Activations ConvNetwork: {self.ConvActivations}\n")
            f.writelines(f"Activations NeuralNetwork: {self.Activations}\n")
            f.writelines(f"Loss Function: {self.loss_function}\n")





