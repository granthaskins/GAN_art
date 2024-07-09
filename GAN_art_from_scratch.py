import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout, ReLU,
                                     Conv2D, Conv2DTranspose, BatchNormalization, 
                                     LeakyReLU, UpSampling2D, Add, Reshape, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class GANArtScratch(object):
    
    def __init__(self, painting_dir, dest_dir, example_dir, gen_model_path=None, disc_model_path=None):
        self.painting_dir = painting_dir
        self.dest_dir = dest_dir
        self.example_dir = example_dir

        self.lr = 1e-4
        self.img_size = [256, 256, 3]

        self.paintings = self.get_paintings()
        self.partitioned_data_dict = self.partition_data()

        self.optimizer = Adam(self.lr)
        
        if gen_model_path:
            self.generator = load_model(gen_model_path)
        else:
            self.generator = self.build_generator()
        
        if disc_model_path:
            self.discriminator = load_model(disc_model_path)
        else:
            self.discriminator = self.build_discriminator()

        self.discriminator.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        
        self.discriminator.trainable = False

        gan_inputs = Input(shape=self.img_size)
        generator_output = self.generator(gan_inputs)
        pred = self.discriminator(generator_output)
        
        self.gan = Model(inputs=gan_inputs, outputs=pred)
        self.gan.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

    def get_paintings(self):
        paintings = []
        for file in os.listdir(self.painting_dir):
            if file.endswith('.jpg'):
                image = Image.open(os.path.join(self.painting_dir, file))
                image = image.resize((self.img_size[0], self.img_size[1]))
                paintings.append(np.array(image))
        return np.array(paintings)

    def partition_data(self):
        # Partition the data into training and validation sets
        idx = np.random.permutation(len(self.paintings))
        split_idx = int(0.8 * len(self.paintings))
        train_idx, val_idx = idx[:split_idx], idx[split_idx:]
        return {
            'train': self.paintings[train_idx],
            'val': self.paintings[val_idx]
        }

    def build_generator(self):
        inputs = Input(shape=self.img_size)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        for _ in range(2):
            x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D(size=2)(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        outputs = Conv2D(self.img_size[2], kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        
        return Model(inputs, outputs)

    def build_discriminator(self):
        inputs = Input(shape=self.img_size)

        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        for _ in range(2):
            x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)

        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs)

    def train_gan(self, epochs, batch_size):
        train_gen_losses, val_gen_losses = [], []
        train_dis_losses, val_dis_losses = [], []

        for epoch in range(epochs):
            for step in tqdm(range(len(self.partitioned_data_dict['train']) // batch_size)):
                real_imgs = self.partitioned_data_dict['train'][step*batch_size : (step+1)*batch_size]
                noise = np.random.normal(0, 1, (batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
                generated_imgs = self.generator.predict(noise)

                dis_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
                dis_loss_fake = self.discriminator.train_on_batch(generated_imgs, np.zeros((batch_size, 1)))
                dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)

                noise = np.random.normal(0, 1, (batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
                gen_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

                train_gen_losses.append(gen_loss[0])
                train_dis_losses.append(dis_loss[0])

            val_noise = np.random.normal(0, 1, (batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
            val_generated_imgs = self.generator.predict(val_noise)
            val_gen_loss = self.gan.evaluate(val_generated_imgs, np.ones((batch_size, 1)), verbose=0)
            val_dis_loss = self.discriminator.evaluate(self.partitioned_data_dict['val'], np.ones((batch_size, 1)), verbose=0)

            val_gen_losses.append(val_gen_loss[0])
            val_dis_losses.append(val_dis_loss[0])

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Generator loss: {gen_loss[0]}, Discriminator loss: {dis_loss[0]}")
            print(f"Validation Generator loss: {val_gen_loss[0]}, Validation Discriminator loss: {val_dis_loss[0]}")

            if epoch % 5 == 0:
                self.generator.save(os.path.join(self.dest_dir, f'generator_epoch_{epoch}.h5'))
                self.discriminator.save(os.path.join(self.dest_dir, f'discriminator_epoch_{epoch}.h5'))
                self.save_example_images(epoch)

        self.plot_losses(train_gen_losses, val_gen_losses, train_dis_losses, val_dis_losses)

    def save_example_images(self, epoch):
        noise = np.random.normal(0, 1, (1, self.img_size[0], self.img_size[1], self.img_size[2]))
        generated_img = self.generator.predict(noise)[0]
        generated_img = (generated_img * 127.5 + 127.5).astype(np.uint8)

        img = Image.fromarray(generated_img)
        img.save(os.path.join(self.example_dir, f'example_image_epoch_{epoch}.png'))

    def plot_losses(self, train_gen_losses, val_gen_losses, train_dis_losses, val_dis_losses):
        plt.figure(figsize=(8, 6))
        plt.title('Training and validation generator loss')
        plt.plot(train_gen_losses, label='Training')
        plt.plot(val_gen_losses, label='Validation')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.dest_dir, 'generator_loss.png'))
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.title('Training and validation discriminator loss')
        plt.plot(train_dis_losses, label='Training')
        plt.plot(val_dis_losses, label='Validation')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.dest_dir, 'discriminator_loss.png'))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN to generate art from scratch.')
    parser.add_argument('painting_dir', type=str, help='Directory containing paintings')
    parser.add_argument('dest_dir', type=str, help='Directory to save models and loss plots')
    parser.add_argument('example_dir', type=str, help='Directory to save example images')
    parser.add_argument('--gen_model_path', type=str, default=None, help='Path to pre-trained generator model')
    parser.add_argument('--disc_model_path', type=str, default=None, help='Path to pre-trained discriminator model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')

    args = parser.parse_args()

    gan_art = GANArtScratch(args.painting_dir, args.dest_dir, args.example_dir, args.gen_model_path, args.disc_model_path)

    gan_art.train_gan(args.epochs, args.batch_size)
