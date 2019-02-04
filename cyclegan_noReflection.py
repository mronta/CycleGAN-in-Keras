# MIT License
#
# Copyright (c) 2017 Erik Linder-Norén
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified code based on: https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py

from __future__ import print_function, division

from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dropout, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader


class CycleGAN:
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'young2old_combined'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

        self.current_epoch = 0

    def build_generator(self):
        def conv2d(layer_input, filters, f_size=4, stride=2):
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
            d = InstanceNormalization()(d)
            d = Activation('relu')(d)
            return d

        def res_block(x, filters=256, use_dropout=False):
            y = conv2d(x, filters, 3, 1)
            if use_dropout:
                y = Dropout(0.5)(y)
            y = conv2d(y, filters, 3, 1)
            return Add()([y, x])

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Activation('relu')(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        # c7s1-64
        d1 = conv2d(d0, self.gf, 7, 1)
        # d128     Reflection padding was used to reduce artifacts?????
        d2 = conv2d(d1, self.gf * 2, 3)
        # d256     Reflection padding was used to reduce artifacts?????
        d3 = conv2d(d2, self.gf * 4, 3)
        # R256,R256,R256,R256,R256,R256
        x = res_block(d3)
        x = res_block(x)
        x = res_block(x)
        x = res_block(x)
        x = res_block(x)
        x = res_block(x)
        # u128
        u1 = deconv2d(x, self.gf * 2, 3)
        # u64
        u2 = deconv2d(u1, self.gf, 3)
        # c7s1-3
        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh')(u2)

        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.img_shape)
        # C64-C128-C256-C512
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def load_network(self, n_epoch_start):
        # returns a compiled model identical to the previous one
        self.current_epoch = n_epoch_start+1
        self.combined.load_weights("models/{}_adapted/combined_ep{}.h5".format(self.dataset_name, n_epoch_start))
        #self.combined.load_weights("models/{}_adapted/combined.h5".format(self.dataset_name))

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, aug=True)):
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1],
                       g_loss[0],
                       np.mean(g_loss[1:3]),
                       np.mean(g_loss[3:5]),
                       np.mean(g_loss[5:6]),
                       elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch+self.current_epoch, batch_i)

            if epoch % 10 == 0 and (epoch != 0 or self.current_epoch != 0):
                self.combined.save("models/{}_adapted/combined_ep{}.h5".format(
                    self.dataset_name,
                    epoch+self.current_epoch)
                )

    def sample_images(self, epoch, batch_i):
        r, c = 2, 3
        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)
        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("images/{}_adapted/{}_{}.png".format(
            self.dataset_name, epoch, batch_i))
        plt.close()

    def test_k_images(self, image_number=25):
        r, c = 2, 3

        imgs_A = self.data_loader.load_k_data(domain="A", image_number=image_number, is_testing=True)
        imgs_B = self.data_loader.load_k_data(domain="B", image_number=image_number, is_testing=True)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        for image_i in range(len(imgs_A)):
            gen_imgs = np.concatenate([
                np.expand_dims(imgs_A[image_i], axis=0),
                np.expand_dims(fake_B[image_i], axis=0),
                np.expand_dims(reconstr_A[image_i], axis=0),
                np.expand_dims(imgs_B[image_i], axis=0),
                np.expand_dims(fake_A[image_i], axis=0),
                np.expand_dims(reconstr_B[image_i], axis=0)
            ])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            titles = ['Original', 'Translated', 'Reconstructed']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("test_images/{}_adapted_v2/test_image_{}.png".format(self.dataset_name, image_i))
            plt.close()

    def test_k_images_epoch(self, image_number=25):
        r, c = 2, 3

        imgs_A = self.data_loader.load_k_data(domain="A", image_number=image_number, is_testing=True)
        imgs_B = self.data_loader.load_k_data(domain="B", image_number=image_number, is_testing=True)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        for image_i in range(len(imgs_A)):
            gen_imgs = np.concatenate([
                np.expand_dims(imgs_A[image_i], axis=0),
                np.expand_dims(fake_B[image_i], axis=0),
                np.expand_dims(reconstr_A[image_i], axis=0),
                np.expand_dims(imgs_B[image_i], axis=0),
                np.expand_dims(fake_A[image_i], axis=0),
                np.expand_dims(reconstr_B[image_i], axis=0)
            ])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            titles = ['Original', 'Translated', 'Reconstructed']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("test_images/{}_adapted_EPOCHES/test_image_{}_epo{}.png".format(self.dataset_name, image_i, self.current_epoch-1))
            plt.close()


if __name__ == '__main__':
    print("CycleGAN without Reflection Padding")
    cyclegan = CycleGAN()
    #cyclegan.load_network(100)
    cyclegan.train(epochs=150, batch_size=1, sample_interval=200)
    #cyclegan.test_k_images(image_number=100)

