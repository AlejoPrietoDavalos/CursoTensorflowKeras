import numpy as np
import time

from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, UpSampling2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.datasets import cifar100
np.random.seed(14)



#----------Preparo los datos----------
(x_train, y_train), (x_test, y_test) = cifar100.load_data()         # Cargamos el data set que vamos a resolver.
n_classes = 100

x_train = preprocess_input(x_train)             # Esta es una utilidad de ResNet50 para hacer el pre-procesamiento de los datos.
x_test = preprocess_input(x_test)               # Esta es una utilidad de ResNet50 para hacer el pre-procesamiento de los datos.


y_train = to_categorical(y_train, n_classes)    # One-Hot encoding.
y_test = to_categorical(y_test, n_classes)      # One-Hot encoding.







#----------Aumentación de datos----------
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

datagen.fit(x_train)



#----------Creamos el modelo----------
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

#----------Ponemos en modo entrenamiento las capas de batch normalization----------
for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False


model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.25))
model.add(BatchNormalization())
model.add(Dense(n_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc', 'mse'])
model.build()
model.summary()


#----------Entrenamos el modelo----------
lr = 1.0
epochs = 5
batch_size = 32
steps_per_epoch = int(round(x_train.shape[0]/batch_size))


start_time = time.time()
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

history = model.fit(x=x_train,
                    y=y_train,
#                     train_generator,
#                     steps_per_epoch=steps_per_epoch,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=5,
                    shuffle=True,
                    verbose=1)

end_time = time.time()

print('\nElapsed Convolutional Model training time: {:.5f} seconds'.format(end_time-start_time))


model.save('./src/ResNet50_cifar100.h5')

history.history.keys()


