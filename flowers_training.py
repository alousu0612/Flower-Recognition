import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from numba import cuda
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


img_size = 128
num_channels = 1
input_shape = (128, 128, 1)
classes = ['tulip', 'rose', 'dandelion', 'daisy', 'sunflower']

train_data = np.load('./data/flower_train_data.npy')
test_data = np.load('./data/flower_test_data.npy')
x_train = np.array([img[0] for img in train_data])
y_train = np.array([img[1] for img in train_data])
x_test = np.array([img[0] for img in test_data])
id = np.array([img[1] for img in test_data])

batch_sizes = 32  # batch 的大小，如果出現 OOM error，請降低這個值
num_classes = len(classes)  # 類別的數量，Cifar 10 共有 10 個類別
epochs = int(math.ceil(len(x_train) / batch_sizes))  # 訓練的 epochs 數量

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(2823, 128, 128, 1)
x_test = x_test.reshape(2000, 128, 128, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)


def cifar_generator(img_array, lbl_array, batch_size):
    while True:
        for index in range(0, len(img_array), batch_size):
            images = img_array[index: index + batch_size]
            labels = lbl_array[index: index + batch_size]
            yield images, labels


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit_generator(cifar_generator(x_train, y_train, batch_sizes),
                              steps_per_epoch=int(len(x_train) / batch_sizes),
                              epochs=epochs,
                              verbose=1)


submission = pd.DataFrame(model.predict_classes(x_test, verbose=0))
submission.columns = ['flower_class']
submission['id'] = y_id
submission = submission[['id', 'flower_class']]
submission.to_csv('./submission.csv', index=False)


cuda.select_device(0)
cuda.close()
