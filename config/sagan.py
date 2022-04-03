import os

Z_DIM = 128
N_CLASS = 10 # num of target classes / labels
HEIGHT = 32
WIDTH = 32
IMAGE_SHAPE=(HEIGHT, WIDTH, 3)
BATCH = 32
BUFFER_SIZE = 200
LATENT_DIM = 64
EPOCHS = 32
# discriminator LR
D_LR = 4e-4

# Generator LR
G_LR = 1e-4


BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
MODEL_ARTIFACTS = os.path.join(BASE_OUTPUT_DIR, "model")
MODEL_CKPT = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
PLOT_ARTIFACTS = os.path.join(BASE_OUTPUT_DIR, "plots")