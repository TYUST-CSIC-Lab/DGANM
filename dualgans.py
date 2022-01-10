from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import to_categorical
import keras.backend as K
from glob import glob
import os
from frechet_kernel_Inception_distance import *
from inception_score import *
import matplotlib.pyplot as plt

import sys

import numpy as np
import numpy as np
import glob
import os
from skimage import io, transform,color

class DUALGAN():
    def __init__(self,a,b,c,d,e):
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 3
        self.img_dim = self.img_rows * self.img_cols


        optimizer = Adam(0.0002, 0.5)
        # optimizer = SGD(0.0002)

        # Build and compile the discriminators
        self.D_A = self.build_discriminator()
        self.D_A.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D_B = self.build_discriminator()
        self.D_B.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G_AB = self.build_generator(a,b,c,d,e)
        self.G_BA = self.build_generator(a,b,c,d,e)

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False

        # The generator takes images from their respective domains as inputs
        imgs_A = Input(shape=(self.img_dim,))
        imgs_B = Input(shape=(self.img_dim,))

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                            optimizer=optimizer,
                            loss_weights=[1, 1, 100, 100])

    def build_generator(self,a,b,c,d,e):

        X = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(a, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=d))
        model.add(Dropout(e))
        model.add(Dense(b))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=d))
        model.add(Dropout(e))
        model.add(Dense(c))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=d))
        model.add(Dropout(e))
        model.add(Dense(self.img_dim, activation='tanh'))

        X_translated = model(X)
        return Model(X, X_translated)

    def build_discriminator(self):

        img = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(512, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))

        validity = model(img)

        return Model(img, validity)

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, x_train,epochs,batch_size=32):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = x_train

        # Domain A and B (rotated)
        X_A = X_train[:int(X_train.shape[0]/2)]
        X_B = scipy.ndimage.interpolation.rotate(X_train[int(X_train.shape[0]/2):], 90, axes=(1, 2))

        X_A = X_A.reshape(X_A.shape[0], self.img_dim)
        X_B = X_B.reshape(X_B.shape[0], self.img_dim)

        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            # Train the discriminator for n_critic iterations
            for _ in range(n_critic):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Sample generator inputs
                imgs_A = self.sample_generator_input(X_A, batch_size)
                imgs_B = self.sample_generator_input(X_B, batch_size)

                # Translate images to their opposite domain
                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                # Train the discriminators
                D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)

                # Clip discriminator weights
                for d in [self.D_A, self.D_B]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            # Plot the progress
            print ("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                % (epoch, D_A_loss[0], D_B_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch == epochs-1:
                # fff = open('C:/Users/HPC/Desktop/Fedlearning/Fed1/loss.txt', 'a+')
                fff = open('C:/Users/HPC/Desktop/LY/morematlab/loss.txt', 'a+')
                loss = g_loss[0]
                F = []
                F.append(loss)
                for num in F:
                    fff.write(str(num) + '\t')
                fff.write('\n')
                return F
            if epoch >=epochs-500:
                self.save_imgs(epoch, X_A, X_B)

    def save_imgs(self, epoch, X_A, X_B):
        r, c = 1, 1
        # Sample generator inputs
        imgs_A = self.sample_generator_input(X_A, c)
        imgs_B = self.sample_generator_input(X_B, c)

        # Images translated to their opposite domain
        fake_B = self.G_AB.predict(imgs_A)
        fake_A = self.G_BA.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])
        gen_imgs = gen_imgs.reshape((1, 4, self.img_rows, self.img_cols, 1))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(1, 1)

        axs.imshow(gen_imgs[0, 0,:,:,0], cmap='gray')
        axs.axis('off')

        fig.savefig("C:/Users/HPC/Desktop/Fedlearning/Fed1/images/total1/" + str(epoch) + ".jpg")
        plt.close()



def read_img(path,value):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    print('reading pictures...')
    for idx, folder in enumerate(cate):
        if idx==value:
            for im in glob.glob(folder + '/*.jpg'):
                print('reading the images:%s' % (im))
                img = io.imread(im)
                img = color.rgb2gray(img)
                # img = Image.open(im)
                # gray = img.convert('L')
                # img=img.convert('RGB')
                img = transform.resize(img, (100, 100))
                imgs.append(img)
                labels.append(idx)
    return np.asarray(imgs, np.float32),np.asarray(labels, np.float32)



import numpy as pd
import csv
import re
import shutil
def function_all(aa):
    os.remove('C:/Users/HPC/Desktop/LY/morematlab/loss.txt')
    os.remove('C:/Users/HPC/Desktop/LY/morematlab/SOIandDOI.txt')
    os.remove('C:/Users/HPC/Desktop/LY/morematlab/FID.txt')

    shutil.rmtree('C:/Users/HPC/Desktop/Fedlearning/Fed1/images/total1')
    os.mkdir('C:/Users/HPC/Desktop/Fedlearning/Fed1/images/total1')
    RAW_DATA = 'C:/Users/HPC/Desktop/Fedlearning/Fed1/training/'
    x_train, y_train = read_img(RAW_DATA, 0)
    x_train = x_train.reshape(x_train.shape[0], 100, 100)

    lan = open('C:/Users/HPC/Desktop/LY/morematlab/N.txt', 'r').readlines()
    T = int(lan[-1])
    for t in range(T):
        # os.remove(r'C:/Users/HPC/Desktop/LY/morematlab/result.txt')
        line = open(aa, 'r').readlines()
        lines = line[t].strip()
        lines = re.split(',', lines)
        a = round(float(lines[0]))
        b = round(float(lines[1]))
        c = round(float(lines[2]))
        d = float(lines[3])
        e = float(lines[4])
        gan = DUALGAN(a, b, c, d, e)
        gan.train(x_train,epochs=2000, batch_size=32)

        # gan.build_generator()
        os.system("python C:/Users/HPC/Desktop/Fedlearning/Fed1/GAN_Metrics-Tensorflow-master/GAN_Metrics-Tensorflow-master/main.py")

    f = open("C:/Users/HPC/Desktop/LY/morematlab/loss.txt", 'r+')
    ff = f.readlines()
    f.seek(0)
    f.truncate()
    for i in range(T):
        f.write(ff[i].replace('nan', '21.386726'))
    f.close()



if __name__ == '__main__':

    # print(sys.argv)
    # if len(sys.argv) < 2:
    #     print('No file specified.')
    #     sys.exit()
    # else:
    #     f2 = function_all(sys.argv[1])
    #     # print(f2)a
    #

    aa ='C:/Users/HPC/Desktop/LY/morematlab/parameter3.txt'
    f2 = function_all(aa)
    print(f2)