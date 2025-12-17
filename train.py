import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam

from metrics import (
    dice_coefficient,
    intersection_over_union,
    binary_crossentropy_dice_loss,
    weighted_f_score,
    s_score,
    e_score,
)
from model import BetterNet
from utils import create_directory
from data import load_data, tf_dataset


def build_lr_schedule(initial_lr, end_lr, steps_per_epoch, num_epochs):
    decay_steps = int(steps_per_epoch * num_epochs)
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=1.0
    )


def set_encoder_trainable(model: tf.keras.Model, trainable: bool, freeze_bn: bool):
    """
    EfficientNetB3 layers in your graph have names like:
    rescaling_*, normalization_*, stem_*, block*, top_*
    We toggle ONLY those as encoder.
    """
    encoder_prefixes = ("rescaling", "normalization", "stem_", "block", "top_")
    for layer in model.layers:
        is_encoder = layer.name.startswith(encoder_prefixes)
        if not is_encoder:
            continue

        if freeze_bn and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = trainable


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--end_learning_rate", type=float, default=1e-7)

    parser.add_argument("--use_polynomial_lr", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")

    # NEW:
    parser.add_argument("--resume_model_path", type=str, default="")
    parser.add_argument("--freeze_bn", action="store_true")  # recommended during fine-tuning

    parser.add_argument("--early_stopping_patience", type=int, default=0)
    args = parser.parse_args()

    create_directory(args.save_dir)
    model_path = os.path.join(args.save_dir, "model.keras")
    csv_path = os.path.join(args.save_dir, "training_log.csv")

    print(f"🔄 Loading data from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(args.dataset_paths, split=0.1, seed=42)
    print(f"Stats: Training on {len(train_x)} images, Validating on {len(valid_x)} images.")

    train_dataset = tf_dataset(train_x, train_y, batch_size=args.batch_size,
                               augment=args.use_augmentation, shuffle=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=args.batch_size,
                               augment=False, shuffle=False)

    steps_per_epoch = max(1, int(np.ceil(len(train_x) / args.batch_size)))

    # --------- Build / Resume model ----------
    if args.resume_model_path:
        print(f"📌 Resuming from model: {args.resume_model_path}")
        model = tf.keras.models.load_model(args.resume_model_path, compile=False)
    else:
        model = BetterNet(input_shape=(224, 224, 3), num_classes=1,
                          dropout_rate=0.5, freeze_encoder=args.freeze_encoder)

    # Apply (freeze/unfreeze) policy AFTER loading/building
    if args.freeze_encoder:
        set_encoder_trainable(model, trainable=False, freeze_bn=False)
        print("🧊 Encoder is frozen.")
    else:
        set_encoder_trainable(model, trainable=True, freeze_bn=args.freeze_bn)
        print(f"🔥 Encoder is unfrozen. freeze_bn={args.freeze_bn}")

    # --------- Optimizer ----------
    if args.use_polynomial_lr:
        lr = build_lr_schedule(args.learning_rate, args.end_learning_rate, steps_per_epoch, args.num_epochs)
        optimizer = Adam(learning_rate=lr)
        print("✅ Using Polynomial LR schedule.")
    else:
        optimizer = Adam(learning_rate=args.learning_rate)
        print("ℹ️ Using constant LR.")

    # --------- Compile ----------
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score]
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor="val_loss"),
        CSVLogger(csv_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='./logs'),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    ]

    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True)
        )

    print("🚀 Starting Training...")
    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    print(f"✅ Saved best model to: {model_path}")
