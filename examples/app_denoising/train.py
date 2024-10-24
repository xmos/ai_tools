from dataset import read_tfrecords
import tensorflow as tf
from model import get_trunet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 100
BATCH_SIZE = 4
MODEL_NAME = "models/model_64f_114k.h5"


def weighted_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    weights = tf.where(y_pred < y_true, 4.0, 1.0)
    weighted_squared_difference = weights * squared_difference
    wmse = tf.reduce_mean(weighted_squared_difference)
    return wmse


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=2e-5,
    min_delta=0.01,
)


model_checkpoint = ModelCheckpoint(
    filepath=MODEL_NAME,
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=False,
    save_freq='epoch'
)


def train():
    train_ds, test_ds = read_tfrecords("data/records_64", bs=BATCH_SIZE)
    model = get_trunet(64)
    optimizer = tf.keras.optimizers.Adam(4e-4)
    model.compile(optimizer=optimizer, loss=weighted_mse)
    model.fit(train_ds,
              epochs=EPOCHS, validation_data=test_ds,
              callbacks=[reduce_lr, early_stopping, model_checkpoint])
    model.save(MODEL_NAME)


if __name__ == "__main__":
    train()
