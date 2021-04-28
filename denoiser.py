#!/usr/bin/python3

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
import cv2
import os
import argparse
import random

parser = argparse.ArgumentParser(description='Image De-Noising')
parser.add_argument('--inference', action="store_true", default=False)
parser.add_argument('--checkpoint_path', type=str, default='./model/1.ckpt')
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args()

inference = args.inference
checkpoint_path = args.checkpoint_path
num_epochs = args.num_epochs

def load_images_from_folder(folderGT, folderNoisy):
    imgTensorsGT = []
    imgTensorsNoisy = []
    for filename in os.listdir(folderGT):
        print('loading ' + str(filename))
        imgGT = Image.open(os.path.join(folderGT,filename))
        imgNoisy = Image.open(os.path.join(folderNoisy,filename))
        imgDataGT = np.asarray(imgGT)
        imgDataNoisy = np.asarray(imgNoisy)
        imgTensorGT = tf.convert_to_tensor(imgDataGT, dtype=tf.float32)
        imgTensorNoisy = tf.convert_to_tensor(imgDataNoisy, dtype=tf.float32)
        imgTensorsGT.append(imgTensorGT)
        imgTensorsNoisy.append(imgTensorNoisy)
    return [tf.stack(imgTensorsGT)/255.0, tf.stack(imgTensorsNoisy)/255.0] 
        

print('----------loading images----------')
[ground_truth_images, noisy_images] = load_images_from_folder('./images/ground_truths', './images/noisy')
print('')
print('ground truth shape = ' + str(ground_truth_images.shape))
print('noisy shape = ' + str(noisy_images.shape))

rand = random.randint(0,1000000)
ground_truth_images = tf.random.shuffle(ground_truth_images, seed=rand)
noisy_images = tf.random.shuffle(noisy_images, seed=rand)



[x_train, x_test] = tf.split(ground_truth_images, [90, 10], axis=0)
[x_train_noisy, x_test_noisy] = tf.split(noisy_images, [90, 10], axis=0)




print('train gt shape = ' + str(x_train.shape))
print('test gt shape = ' + str(x_test.shape))
print('train noisy shape = ' + str(x_train_noisy.shape))
print('test noisy shape = ' + str(x_test_noisy.shape))

#n = 5
#plt.figure(figsize=(20, 8))
#plt.gray()
#for i in range(n):
#    ax = plt.subplot(2, n, i + 1) 
#    plt.title("original", size=10) 
#    plt.imshow(tf.squeeze(x_test[i])) 
#    plt.gray() 
#    bx = plt.subplot(2, n, n+ i + 1) 
#    plt.title("original + noise", size=10) 
#    plt.imshow(tf.squeeze(x_test_noisy[i])) 
#plt.show()

class NoiseReducer(tf.keras.Model):
  def __init__(self):

    super(NoiseReducer, self).__init__()

    self.encoder = tf.keras.Sequential([
      Input(shape=(512, 512, 3)),
      Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
      Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = NoiseReducer()
autoencoder.compile(optimizer='adam', loss='mse')

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

if inference:
    autoencoder.load_weights(checkpoint_path)
else:
    autoencoder.fit(x_train_noisy,
            x_train,
            epochs=num_epochs,
            shuffle=True,
            validation_data=(x_test_noisy, x_test),
            callbacks=[cp_callback])


encoded_imgs=autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs)

n = 10 
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
    # display original + noise 
    bx = plt.subplot(3, n, i + 1) 
    plt.title("noisy", size=10) 
    plt.imshow(tf.squeeze(x_test_noisy[i])) 
    bx.get_xaxis().set_visible(False) 
    bx.get_yaxis().set_visible(False) 
  
    # display reconstruction 
    cx = plt.subplot(3, n, i + n + 1) 
    plt.title("reconstructed", size=10) 
    plt.imshow(tf.squeeze(decoded_imgs[i])) 
    cx.get_xaxis().set_visible(False) 
    cx.get_yaxis().set_visible(False) 
  
    # display original 
    ax = plt.subplot(3, n, i + 2*n + 1) 
    plt.title("original", size=10) 
    plt.imshow(tf.squeeze(x_test[i])) 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 

plt.savefig('final.png')
