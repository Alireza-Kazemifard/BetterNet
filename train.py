
import os
# --- 1. Suppress TensorFlow Logs (Must be before importing TF) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=All, 1=Filter INFO, 2=Filter WARNING, 3=Filter ERROR
os.environ["TF_gpu_allocator"] = "cuda_malloc_async" # Helps with memory fragmentation

import argparse
import tensorflow as tf

# --- 2. Fix for "Factory already registered" warnings ---
try:
    tf.get_logger().setLevel('ERROR')
except:
    pass

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import BetterNet
from metrics import bce_dice_loss, dice_coefficient
from data import load_data, tf_dataset
from utils import create_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset_paths", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    # Paths
    save_dir = args.save_dir
    create_directory(save_dir)
    
    # Load Data
    print(f"📂 Loading Dataset from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(args.dataset_paths)
    print(f"   Train Images: {len(train_x)} | Val Images: {len(valid_x)}")
    
    train_dataset = tf_dataset(train_x, train_y, batch=args.batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=args.batch_size)

    # Model
    print("🧠 Building Model...")
    model = BetterNet(input_shape=(256, 256, 3))
    model.compile(loss=bce_dice_loss, optimizer=Adam(args.learning_rate), metrics=[dice_coefficient])

    # Callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(save_dir, "model.keras"), verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(save_dir, "training_log.csv")),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    print(f"🚀 Starting Training... Saving to: {save_dir}")
    print("⏳ Please wait 1-2 minutes for the first Epoch to initialize (GPU Warmup)...")
    
    model.fit(train_dataset, validation_data=valid_dataset, epochs=args.num_epochs, callbacks=callbacks)
    print("🎉 Training Finished!")
