from dataset import read_tfrecords
from model import get_trunet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

EPOCHS = 100
BATCH_SIZE = 6


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


model_checkpoint = ModelCheckpoint(
    filepath="model.h5",
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=False,
    save_freq='epoch'
)


def train():
    train_ds, test_ds = read_tfrecords("data/records", batch_size=BATCH_SIZE)
    model = get_trunet()
    model.fit(train_ds,
              epochs=EPOCHS, validation_data=test_ds,
              callbacks=[reduce_lr, early_stopping, model_checkpoint])
    model.save("model.h5")


if __name__ == "__main__":
    train()
