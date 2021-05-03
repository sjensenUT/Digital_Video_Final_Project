#!/usr/bin/python3

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib as mpl
mpl.use ( 'Agg') # must be written both in import intermediate
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input, MaxPool2D, UpSampling2D, BatchNormalization
import os
import argparse

parser = argparse.ArgumentParser(description='Image De-Noising')
parser.add_argument('--inference', action="store_true", default=False)
parser.add_argument('--checkpoint_path', type=str, default='./model/2.ckpt')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--image_dir', type=str, default='./images_2')
#parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

inference = args.inference
checkpoint_path = args.checkpoint_path
num_epochs = args.num_epochs
batch_size = args.batch_size
image_dir = args.image_dir
#learning_rate=args.learning_rate


def img_generator(folderGT, folderNoisy, batch_size):
    file_list = os.listdir(folderGT) 
    i = 0
    while 1:
        #print("starting new batch")
        img_batch = []
        img_batch_noisy = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            filename = file_list[i]
            #print(filename)
            #print('img_batch size = ' + str(len(img_batch)))
            i += 1
            imgGT = Image.open(os.path.join(folderGT,filename))
            imgNoisy = Image.open(os.path.join(folderNoisy,filename))
            imgDataGT = np.asarray(imgGT, np.float32)
            imgDataNoisy = np.asarray(imgNoisy, np.float32)
            #imgTensorGT = tf.convert_to_tensor(imgDataGT, dtype=tf.float32)
            #imgTensorNoisy = tf.convert_to_tensor(imgDataNoisy, dtype=tf.float32)
            img_batch.append(imgDataGT/255.0)
            img_batch_noisy.append(imgDataNoisy/255.0)
            imgGT.close()
            imgNoisy.close()
        #print('yielding')
        yield np.stack(img_batch_noisy), np.stack(img_batch)


class NoiseReducer2(tf.keras.Model):
  def __init__(self):

    super(NoiseReducer2, self).__init__()

    self.encoder = tf.keras.Sequential([
      Input(shape=(384, 512, 3)),
      Conv2D(64, (3,3), activation='relu', padding='same'),
      MaxPool2D((2,2), padding = 'same'),
      BatchNormalization(),
      Conv2D(32, (3,3), activation='relu', padding='same'),
      MaxPool2D((2,2), padding = 'same'),
      BatchNormalization(),
      Conv2D(16, (3,3), activation='relu', padding='same'),
      MaxPool2D((2,2), padding = 'same')])

    self.decoder = tf.keras.Sequential([
      Conv2D(64, (3,3), activation='relu', padding='same'),
      UpSampling2D((2,2)),
      Conv2D(32, (3,3), activation='relu', padding='same'),
      UpSampling2D((2,2)),
      Conv2D(16, (3,3), activation='relu', padding='same'),
      UpSampling2D((2,2)),
      Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = NoiseReducer2()
autoencoder.compile(optimizer='adam', loss='mse')

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


folderGT = os.path.join(image_dir, '512x384')
folderNoisy = os.path.join(image_dir, '512x384_noisy')
steps_per_epoch = len(os.listdir(folderGT))/batch_size

if inference:
    autoencoder.load_weights(checkpoint_path)
else:
    with tf.device('/gpu:0'):
        autoencoder.fit(img_generator(folderGT, folderNoisy, batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[cp_callback])


#encoded_imgs=autoencoder.encoder(x_test_noisy).numpy()
#decoded_imgs=autoencoder.decoder(encoded_imgs)

#decoded_imgs=autoencoder.call(x_test_noisy)

