# --- START OF FILE train.py ---
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from metrics import dice_coefficient, intersection_over_union, binary_crossentropy_dice_loss, weighted_f_score, s_score, e_score
from model import BetterNet
from utils import create_directory
from data import load_data, tf_dataset

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs='+', required=True, help="List of dataset paths (e.g. Dataset/Kvasir Dataset/CVC)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    create_directory(args.save_dir)
    model_path = os.path.join(args.save_dir, "model.keras")
    csv_path = os.path.join(args.save_dir, "training_log.csv")

    # 1. Load Combined Data
    print(f"🔄 Loading data from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(args.dataset_paths, split=0.1)
    
    print(f"Training on {len(train_x)} images, Validating on {len(valid_x)} images.")

    # 2. Create Pipelines
    train_dataset = tf_dataset(train_x, train_y, batch_size=args.batch_size, augment=True) # Augmentation روشن برای آموزش
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=args.batch_size, augment=False)

    # 3. Build Model (Input size 224x224 as per paper)
    model = BetterNet(input_shape=(224, 224, 3), num_classes=1, dropout_rate=0.5)
    
    # 4. Compile with Paper's Metrics
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score]
    )

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1), # صبر بیشتر طبق مقاله
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # صبر 30 اپاک
    ]

    # 6. Train
    print("🚀 Starting Combined Training...")
    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    print("✅ Training Finished.")
