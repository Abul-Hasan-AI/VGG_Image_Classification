from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input
from keras.optimizers import SGD
import cv2, numpy as np
from keras import backend as K
from keras.datasets import mnist,cifar10
from keras.preprocessing.image import img_to_array, array_to_img, ImageDataGenerator
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt

dataset = 'mnist'

if dataset == 'mnist':
	data_augmentation = False

	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


	train_images = np.reshape(train_images,(train_images.shape[0], 28, 28,1))
	test_images = np.reshape(test_images,(test_images.shape[0],28, 28,1))

elif dataset == 'cifar10':
	data_augmentation = True
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()



train_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in train_images])

test_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in test_images])


train_images = train_images/ 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

input = Input(shape=(224, 224,train_images.shape[-1]))

x = ZeroPadding2D((1,1))(input)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = ZeroPadding2D((1,1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = ZeroPadding2D((1,1))(x)
x = Convolution2D(256, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(256, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(256, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)


x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

BatchSize = 32
Epochs = 50

VGG = Model(inputs=input, outputs=output)
VGG.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
VGG.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if not data_augmentation:
	history = VGG.fit(train_images, train_labels, batch_size=BatchSize,epochs=Epochs, validation_data=(test_images,test_labels))
	
else:
	print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
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

	    # Compute quantities required for feature-wise normalization
	    # (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(train_images)

    # Fit the model on the batches generated by datagen.flow().
	history = VGG.fit_generator(datagen.flow(train_images, train_labels,
	                                     batch_size= BatchSize),steps_per_epoch=len(train_images)/BatchSize,
	                        epochs=Epochs,
	                        validation_data=(test_images, test_labels),
	                        workers=4)


Results = VGG.evaluate(test_images,  test_labels, verbose=1)

print('\nTest Loss :', Results[0],'\nTest accuracy:', Results[1]*100)

# results save folder
if not os.path.isdir('VGG_results'):
    os.mkdir('VGG_results')


# plot the loss and accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


path = 'VGG_results/'
plt.title('Training and validation accuracy for ' + dataset)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')


plt.legend()
plt.savefig(path+dataset+ 'AccuracyPlot.png')


plt.figure()
plt.title('Training and validation loss for ' + dataset)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')


plt.legend()
plt.savefig(path+dataset+'LossPlot.png')

plt.show()

