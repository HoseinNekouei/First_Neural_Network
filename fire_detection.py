import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers

EPOCHS = 25
BATCH_SIZE = 32

def preprocess_data(dataPath):
    data_list = []
    labels=[]
    le = LabelEncoder()

    for i,item in enumerate(glob.glob(dataPath)):
        img = cv2.imread(item)
        # preprocessing on image
        #resize image
        img = cv2.resize(img,(32,32))
        # min max normalization
        img = img/255
        # img = img.flatten()

        data_list.append(img)

        label = item.split('\\')[-1].split('.')[0]
        # print(label)
        labels.append(label)

        if i % 100 ==0:
            print('[INFO]: {}/{} processed'.format(i,1000))

    data_list = np.array(data_list)

    x_train, x_test , y_train, y_test = train_test_split(data_list, labels, test_size=0.2, random_state=42)


    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test , y_train, y_test


def neural_network():
    # make sequential network 
    net = models.Sequential([
                            #Primary layers
                            layers.Flatten(input_shape= (32,32,3)),
                            layers.Dense(20,activation='relu', input_dim = 8),
                            layers.Dense(8,activation='relu'),
                            #Last Layer
                            layers.Dense(2,activation='softmax')
                            ])

    net.summary()

    # Determination of parameters
    net.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

    history = net.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data=(x_test, y_test)) 

    loss, acc = net.evaluate(x_test, y_test)
    print("loss: {:.2f} , accuracy: {:.2f}".format(loss,acc))

    net.save('neural_network/mlp.h5')

    return history


def show_results():
    plt.style.use('ggplot')
    plt.plot(np.arange(EPOCHS),H.history['loss'],label ='Train loss')
    plt.plot(np.arange(EPOCHS),H.history['val_loss'], label='Test loss')
    plt.plot(np.arange(EPOCHS),H.history['accuracy'],label='Train accuracy')
    plt.plot(np.arange(EPOCHS),H.history['val_accuracy'], label='Test accuracy')
    plt.legend()
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss & Accuracy')
    plt.title('Training on Fire Dataset')
    plt.show()


x_train, x_test , y_train, y_test = preprocess_data("fire_detection\\fire_dataset\\*\\*")
H = neural_network()
show_results()

