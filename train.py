# TODO: Training script for self attention GAN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)

from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from models.sagan import SAGAN
from models.sagan import hinge_loss_d, hinge_loss_g, build_discriminator, build_generator
from models.callbacks import EpochCheckpoint, GANMonitor
import config.sagan as config

@tf.function
def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5

    label = tf.cast(label, tf.int32)
    return image, label


if __name__ == "__main__":
    print("[INFO] Loading CIFAR-10 dataset...")
    
    # We use TFDS cifar10 as it returns tuples of (img, label)
    ds_train, ds_info = tfds.load("cifar10", split="train", as_supervised=True, shuffle_files=True, with_info=True)

    # .repeat means to generate an infinite data stream...
    train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(config.BUFFER_SIZE)
    train_ds = train_ds.batch(config.BATCH).repeat()

    d_opt = Adam(config.D_LR, 0.0, 0.9)
    g_opt = Adam(config.G_LR, 0.0, 0.9)

    discriminator = build_discriminator(config.N_CLASS, config.IMAGE_SHAPE)
    generator = build_generator(config.Z_DIM, config.N_CLASS)

    plot_model(discriminator, to_file="discriminator.png", show_shapes=True, show_layer_names=True)

    plot_model(generator, to_file="generator.png", show_shapes=True, show_layer_names=True)

    sagan = SAGAN(discriminator, generator, config.IMAGE_SHAPE, n_class=config.N_CLASS, batch_size=config.BATCH)

    sagan.compile(
        d_opt=d_opt,
        g_opt=g_opt,
        d_loss_fn=hinge_loss_d,
        g_loss_fn=hinge_loss_g
    )

    ckpt_dir = config.MODEL_CKPT
    start_at = 0

    # define the objects we want to persist
    ckpt_obj = tf.train.Checkpoint(
        d_opt=d_opt,
        g_opt=g_opt,
        generator=generator,
        discriminator=discriminator
    )

    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

    if latest_ckpt is not None:
        print("[INFO] Resuming from ckpt: {}".format(latest_ckpt))
        ckpt_obj.restore(latest_ckpt).assert_existing_objects_matched().expect_partial()

        latest_ckpt_idx = latest_ckpt.split(os.path.sep)[-1].split("-")[-1]
        start_at = int(latest_ckpt_idx)
        print(f"[INFO] Resuming ckpt at {start_at}")

    ckpt_callback = EpochCheckpoint(ckpt_dir, every=2500, start_at=start_at, ckpt_obj=ckpt_obj)

    gan_monitor = GANMonitor(config.PLOT_ARTIFACTS, images_per_class=10, latent_dim=config.Z_DIM, start_at=start_at, every=2500)

    # Example notebook states 50,000 steps with callbacks at every 2500 step
    sagan.fit(
        train_ds,
        steps_per_epoch=50000,
        epochs=1,
        callbacks=[ckpt_callback, gan_monitor]
    )

    sagan.generator.save(config.MODEL_ARTIFACTS)