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
    parser.add_argument("--dataset_paths", nargs='+', required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    create_directory(args.save_dir)
    model_path = os.path.join(args.save_dir, "model.keras")
    csv_path = os.path.join(args.save_dir, "training_log.csv")

    print(f"🔄 Loading data from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(args.dataset_paths, split=0.1)
    
    print(f"Stats: Training on {len(train_x)} images, Validating on {len(valid_x)} images.")

    # دیتاست
    train_dataset = tf_dataset(train_x, train_y, batch_size=args.batch_size, augment=True, shuffle=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=args.batch_size, augment=False, shuffle=False)

    # مدل - طبق مقاله وزن‌های انکودر فریز می‌شوند
    # اگر همگرایی ضعیف بود، freeze_encoder=False را تست کنید
    model = BetterNet(input_shape=(224, 224, 3), num_classes=1, dropout_rate=0.5, freeze_encoder=False) 
    # نکته: من اینجا False گذاشتم چون معمولا در عمل با داده کم، Fine-tuning (با نرخ یادگیری پایین) بهتر جواب می‌دهد.
    # اگر میخواهید دقیقا جمله مقاله را تست کنید، True کنید.

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score]
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    ]

    print("🚀 Starting Training Process...")
    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
