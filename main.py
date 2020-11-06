# -*- coding: utf-8 -*-
"""Tesis Clasificacion v2

### Importación de librerías
"""

import itertools
from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
# %matplotlib inline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, AveragePooling2D, Flatten, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, plot_confusion_matrix

datasetFolderName='/ProyectoGrado/imgTodas'

train_path=datasetFolderName+'/train/'
validation_path=datasetFolderName+'/validation/'
test_path=datasetFolderName+'/test/'
img_rows, img_cols, numOfChannels =  224, 224, 3

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20

def load_data():
  train_datagen = ImageDataGenerator(
                  rescale=1./255,                
                  fill_mode="nearest"
                  )
  validation_datagen = ImageDataGenerator(
                  rescale=1./255,                
                  fill_mode="nearest"
                  )
  test_datagen = ImageDataGenerator(
                  rescale=1./255
                )

  train_generator = train_datagen.flow_from_directory(
          train_path,
          target_size=(img_rows, img_cols),
          batch_size=BATCH_SIZE,
          class_mode='categorical',
          color_mode="rgb")

  validation_generator = train_datagen.flow_from_directory(
          validation_path,
          target_size=(img_rows, img_cols),
          batch_size=BATCH_SIZE,
          class_mode='categorical',
          color_mode="rgb")

  test_generator = test_datagen.flow_from_directory(
          test_path,
          target_size=(img_rows, img_cols),
          batch_size=BATCH_SIZE,
          color_mode="rgb",
          class_mode=None,  # only data, no labels
          shuffle=False)
  
  return train_generator, validation_generator, test_generator

train_generator, validation_generator, test_generator = load_data()

#  x,y = train_generator.next()
#  print(x[0].shape)
#  plt.imshow(x[0])

"""### Construcción del modelo"""

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
base_model.summary()

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(LR))(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
  layer.trainable = False

model.summary()

opt = Adam(lr=LR, decay=LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint = ModelCheckpoint(monitor='loss',
                             verbose=2, 
                             save_best_only= True, 
                             mode='auto') 

early_stopping = EarlyStopping(monitor='loss',
                               patience=5,
                               verbose=2,
                               mode='auto')

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=EPOCHS,
                              callbacks=[checkpoint, early_stopping])

try:
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
except KeyError:
  print('Error')

plt.title('Accuracy vs Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

try:
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
except KeyError:
  print('Error')

plt.title('Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

def showResults(test, pred):
    target_names = ['positive', 'negative']
    # print(classification_report(test, pred, target_names=target_names))
    accuracy = accuracy_score(test, pred)
    precision=precision_score(test, pred, average='weighted')
    f1Score=f1_score(test, pred, average='weighted') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    return confusion_matrix(test, pred)

# Testing/Prediction phase
predictions = model.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
# Display the performance of the model on test data
cm = showResults(true_classes[:len(yPredictions)], yPredictions)

target_names = ['Positivo', 'Negativo']

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de confusión')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j,i, cm[i,j], horizontalalignment='center', 
    color ='white' if (cm[i,j] > thresh) else 'black')