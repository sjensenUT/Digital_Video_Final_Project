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
parser.add_argument('--realdata', action="store_true", default=False)
parser.add_argument('--checkpoint_path', type=str, default='./model/2.ckpt')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--image_dir', type=str, default='./real-data')
#parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

inference = args.inference
realdata = args.realdata
checkpoint_path = args.checkpoint_path
num_epochs = args.num_epochs
batch_size = args.batch_size
image_dir = args.image_dir
#learning_rate=args.learning_rate

def load_images_from_folder(folderGT, folderNoisy):
    imgTensorsGT = []
    imgTensorsNoisy = []
    count = 0
    for filename in os.listdir(folderGT):
        print('loading ' + str(filename))
        count+=1
        imgGT = Image.open(os.path.join(folderGT,filename))
        imgNoisy = Image.open(os.path.join(folderNoisy,filename))
        imgDataGT = np.asarray(imgGT)
        imgDataNoisy = np.asarray(imgNoisy)
        if inference and realdata:
            imgDataGT = imgDataGT[0:384]
            imgDataNoisy = imgDataNoisy[0:384]
        imgTensorGT = tf.convert_to_tensor(imgDataGT, dtype=tf.float32)
        imgTensorNoisy = tf.convert_to_tensor(imgDataNoisy, dtype=tf.float32)
        imgTensorsGT.append(imgTensorGT)
        imgTensorsNoisy.append(imgTensorNoisy)
        if(count==5186): break
    return [tf.stack(imgTensorsGT)/255.0, tf.stack(imgTensorsNoisy)/255.0] 
        
class NoiseReducer(tf.keras.Model):
  def __init__(self):

    super(NoiseReducer, self).__init__()

    self.encoder = tf.keras.Sequential([
      Input(shape=(384, 512, 3)),
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


if inference:
    print('----------loading images for inference----------')
    #[x_test, x_test_noisy] = load_images_from_folder(os.path.join(image_dir, '512x384_validation'), os.path.join(image_dir, '512x384_noisy_validation'))
    image_dir = os.path.join(image_dir, 'validation')
    [x_test, x_test_noisy] = load_images_from_folder(os.path.join(image_dir, 'ground_truths'), os.path.join(image_dir, 'noisy'))
    print('')
    print('ground truth shape = ' + str(x_test.shape))
    print('noisy shape = ' + str(x_test_noisy.shape))
else:
    print('----------loading images for training----------')
    [x_train, x_train_noisy] = load_images_from_folder(os.path.join(image_dir, '512x384'), os.path.join(image_dir, '512x384_noisy'))
    print('')
    print('ground truth shape = ' + str(x_train.shape))
    print('noisy shape = ' + str(x_train_noisy.shape))
    
    train_size = int(x_train.shape[0]*0.90)
    test_size = x_train.shape[0] - train_size
    
    print('train_size = ' + str(train_size))
    print('test_size = ' + str(test_size))
    
    [x_train, x_test] = tf.split(x_train, [train_size, test_size], axis=0)
    [x_train_noisy, x_test_noisy] = tf.split(x_train_noisy, [train_size, test_size], axis=0)
    
    print('train gt shape = ' + str(x_train.shape))
    print('test gt shape = ' + str(x_test.shape))
    print('train noisy shape = ' + str(x_train_noisy.shape))
    print('test noisy shape = ' + str(x_test_noisy.shape))


autoencoder = NoiseReducer2()
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
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test_noisy, x_test),
            callbacks=[cp_callback])


decoded_imgs=autoencoder.predict(x_test_noisy, batch_size=batch_size)
real_psnr = tf.image.psnr(x_test_noisy, x_test, max_val=1.0)
model_psnr1 = tf.image.psnr(decoded_imgs, x_test, max_val=1.0)
avg_orig_psnr = tf.math.reduce_mean(real_psnr)
avg_recon_psnr = tf.math.reduce_mean(model_psnr1)
avg_orig_psnr = avg_orig_psnr.numpy() 
avg_recon_psnr = avg_recon_psnr.numpy()
print('Original Average PSNR ' + str(avg_orig_psnr))
print('Reconstructed Average PSNR '+ str(avg_recon_psnr))

"""
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

plt.savefig('real-final.png')
"""


