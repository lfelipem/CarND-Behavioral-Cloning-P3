import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

samples = []
car_images = []
steering_angles = []

with open('C:\\Users\Luiz Felipe\Desktop\windows_sim\driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = batch_sample[0]
                center_image = cv2.imread(path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # FLIP image!
                flipped = cv2.flip(center_image, 1)
                images.append(flipped)
                angles.append(-center_angle)

                path = batch_sample[1]
                left_image = cv2.imread(path)
                left_angle = float(batch_sample[3]) + 0.2
                images.append(left_image)
                angles.append(left_angle)
                # FLIP image!
                flipped = cv2.flip(left_image, 1)
                images.append(flipped)
                angles.append(-left_angle)

                path = batch_sample[2]
                right_image = cv2.imread(path)
                right_angle = float(batch_sample[3]) - 0.2
                images.append(right_image)
                angles.append(right_angle)
                # FLIP image!
                flipped = cv2.flip(right_image, 1)
                images.append(flipped)
                angles.append(-right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



##############
batch_size = 64
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)
##############

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)/batch_size, nb_epoch=5, verbose=1)
#print(model.summary())
model.save('model.h5', overwrite=True)
print("Model Saved!")

############################################
# Visualization Loss
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
############################################


