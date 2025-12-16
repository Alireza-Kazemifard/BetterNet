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
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Arguments
    parser = argparse.ArgumentParser(description='Train BetterNet')
    parser.add_argument("--dataset_paths", nargs='+', required=True, help="List of paths to datasets (e.g. Dataset/Kvasir Dataset/CVC)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model and logs")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Setup directories
    create_directory(args.save_dir)
    model_path = os.path.join(args.save_dir, "model.keras")
    csv_path = os.path.join(args.save_dir, "training_log.csv")

    # 1. Load Combined Data
    # The load_data function now handles multiple paths and shuffles them together
    print(f"🔄 Loading data from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(args.dataset_paths, split=0.1)
    
    print(f"Stats: Training on {len(train_x)} images, Validating on {len(valid_x)} images.")

    # 2. Create Data Pipelines
    # shuffle=True is ENABLED for training to ensure batches are random every epoch
    train_dataset = tf_dataset(train_x, train_y, batch_size=args.batch_size, augment=True, shuffle=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=args.batch_size, augment=False, shuffle=False)

    # 3. Build Model
    # Input shape matches the paper (224, 224, 3)
    model = BetterNet(input_shape=(224, 224, 3), num_classes=1, dropout_rate=0.5)
    
    # 4. Compile Model
    # Using the custom binary_crossentropy_dice_loss and all metrics from the paper
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score]
    )

    # model.summary() # Optional: Un-comment to see model structure

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    ]

    # 6. Start Training
    print("🚀 Starting Combined Training Process...")
    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    print("✅ Training Finished Successfully.")
