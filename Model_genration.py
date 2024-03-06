import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import pdb
import preprocessing as pr
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()
folders=['Sirih', 'Nangka', 'Belimbing Wuluh', 'Kemangi', 'Jeruk Nipis', 'Jambu Biji', 'Seledri', 'Lidah Buaya', 'Pandan', 'Pepaya']
def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy')
    plt.legend([ 'train','test','train_loss','test_loss'], loc='upper left')
    plt.savefig("model_performance.png")

def model_Sq():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(250,250,3)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(70,activation='relu'))
    model.add(tf.keras.layers.Dense(40,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer ='adam' , loss = 'categorical_crossentropy', metrics = ['acc'])
    model.build(input_shape=(250,250))
    return model
pr.AUG()
pr.Train_test_split('new')
datagen = ImageDataGenerator(rescale=1./255)
training_set = datagen.flow_from_directory('train',target_size=(250,250),classes=folders)
validation_set = datagen.flow_from_directory('test',target_size=(250,250),classes=folders)
model=model_Sq()
pdb.set_trace()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
history=model.fit(training_set, validation_data = validation_set, epochs = 20 ,callbacks=[callback])
print('The final accuracy of the model against validtion set: ',history.history['val_acc'][-1]*100,'%')
plot(history)
for layer in model.layers:
    layer.trainable = False
model.save(f'Model.keras')
