import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class EpochCheckpoint(Callback):
    def __init__(self, output_dir, every=5, start_at=0, ckpt_obj=None):
        super(EpochCheckpoint, self).__init__()

        self.checkpoint_dir = output_dir
        self.every = every
        self.int_epoch = start_at
        self.checkpoint = ckpt_obj

    # Using on_batch_end callback as we are running a single epochs with multiple steps
    def on_batch_end(self, batch, logs=None):
        if (self.int_epoch + 1) % self.every == 0:
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint.save(file_prefix=checkpoint_prefix)

        self.int_epoch += 1


class GANMonitor(Callback):
    def __init__(self, output_dir, images_per_class=10, latent_dim=128, start_at=0, every=5):
        self.images_per_class = images_per_class
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.start_at = start_at
        self.every = every

    # Using on_batch_end callback as we are running a single epochs with multiple steps
    def on_batch_end(self, batch, logs=None):
        if (self.start_at + 1) % self.every == 0:
            images_per_class = self.images_per_class
            z = tf.random.normal((images_per_class * self.model.n_class, self.latent_dim))

            labels = []
            for i in range(self.model.n_class):
                labels += [i] * images_per_class

            labels = np.asarray(labels, dtype=np.int32)

            images = self.model.generator.predict([z, labels])
            images = images * 0.5 + 0.5
            grid_row = self.model.n_class
            grid_col = images_per_class

            scale = 2
            f, axarr = plt.subplots(grid_row, grid_col, 
                                    figsize=(grid_col*scale, grid_row*scale))
            
            for row in range(grid_row):
                ax = axarr if grid_row==1 else axarr[row]
                for col in range(grid_col):
                    ax[col].imshow(images[row*grid_col + col])
                    ax[col].axis('off')

            figpath = os.path.join(self.output_dir, "epoch_{:03d}.png".format(self.start_at + 1))
            plt.savefig(figpath)

        self.start_at += 1