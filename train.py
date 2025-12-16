import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from metrics import dice_coefficient, intersection_over_union, binary_crossentropy_dice_loss
from model import BetterNet
from data import load_data, tf_dataset

# --- CONFIGURATION (Colab Friendly) ---
# مسیر دیتاست خود را اینجا وارد کنید:
dataset_path = "/content/drive/MyDrive/PolypDataset/" 
batch_size = 8
learning_rate = 1e-4
num_epochs = 50
model_path = "BetterNet_Model.keras"
csv_path = "training_log.csv"

# --- Main Execution ---
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load Data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    # Build Model
    model = BetterNet(input_shape=(256, 256, 3))
    # model.summary() # Uncomment to see architecture

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, intersection_over_union]
    )

    # Callbacks (Original Paper Settings)
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    # Train
    print("Starting Training...")
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    print("Training Finished.")
