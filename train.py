import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam

from metrics import (
    dice_coefficient,
    intersection_over_union,
    binary_crossentropy_dice_loss,
    weighted_f_score,
    s_score,
    e_score,
    mean_absolute_error
)
from model import BetterNet
from utils import create_directory
from data import load_data, tf_dataset


def set_encoder_trainable(model: tf.keras.Model, trainable: bool, freeze_bn: bool):
    """Toggle EfficientNet encoder layers"""
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

    # ==================== Argument Parsing ====================
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs="+", required=True,
                       help="List of dataset paths")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save model and logs")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--end_learning_rate", type=float, default=1e-7)
    
    # Training options
    parser.add_argument("--use_polynomial_lr", action="store_true",
                       help="Use polynomial learning rate decay")
    parser.add_argument("--freeze_encoder", action="store_true",
                       help="Freeze EfficientNet encoder")
    parser.add_argument("--use_augmentation", action="store_true",
                       help="Use data augmentation")
    parser.add_argument("--freeze_bn", action="store_true",
                       help="Freeze BatchNorm layers even when encoder unfrozen")
    
    # Resume training
    parser.add_argument("--resume_model_path", type=str, default="",
                       help="Path to resume training from checkpoint")
    
    # Early stopping / LR reduction
    parser.add_argument("--patience", type=int, default=20,
                       help="Patience for ReduceLROnPlateau and EarlyStopping")
    
    args = parser.parse_args()

    # ==================== Setup ====================
    create_directory(args.save_dir)
    model_path = os.path.join(args.save_dir, "model.keras")
    csv_path = os.path.join(args.save_dir, "training_log.csv")

    print(f"🔄 Loading data from: {args.dataset_paths}")
    (train_x, train_y), (valid_x, valid_y) = load_data(
        args.dataset_paths, split=0.1, seed=42
    )
    print(f"📊 Stats: Training on {len(train_x)} images, Validating on {len(valid_x)} images.")

    # ==================== Create TF Datasets ====================
    train_dataset = tf_dataset(
        train_x, train_y, 
        batch_size=args.batch_size,
        augment=args.use_augmentation, 
        shuffle=True
    )
    valid_dataset = tf_dataset(
        valid_x, valid_y, 
        batch_size=args.batch_size,
        augment=False, 
        shuffle=False
    )

    steps_per_epoch = max(1, int(np.ceil(len(train_x) / args.batch_size)))

    # ==================== Build or Resume Model ====================
    if args.resume_model_path and os.path.exists(args.resume_model_path):
        print(f"📌 Resuming from checkpoint: {args.resume_model_path}")
        # ✅ Fix Lambda layer deserialization error
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(args.resume_model_path, compile=False)
    else:
        print("🏗️ Building new BetterNet model...")
        model = BetterNet(
            input_shape=(224, 224, 3), 
            num_classes=1,
            dropout_rate=0.5, 
            freeze_encoder=args.freeze_encoder
        )

    # ==================== Freeze/Unfreeze Encoder ====================
    if args.freeze_encoder:
        set_encoder_trainable(model, trainable=False, freeze_bn=False)
        print("🧊 Encoder is frozen.")
    else:
        set_encoder_trainable(model, trainable=True, freeze_bn=args.freeze_bn)
        print(f"🔥 Encoder is unfrozen. freeze_bn={args.freeze_bn}")

    # ==================== Optimizer (Float LR, NO Schedule Object) ====================
    optimizer = Adam(learning_rate=args.learning_rate)

    # ==================== Compile Model ====================
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: binary_crossentropy_dice_loss(
            y_true, y_pred, alpha=0.5
        ),
        metrics=[
            dice_coefficient, 
            intersection_over_union, 
            weighted_f_score,
            s_score, 
            e_score, 
            mean_absolute_error
        ]
    )

    # ==================== Callbacks ====================
    callbacks = [
        ModelCheckpoint(
            model_path, 
            monitor="val_loss", 
            save_best_only=True, 
            verbose=1
        ),
        CSVLogger(csv_path),
        TensorBoard(log_dir=os.path.join(args.save_dir, "logs"))
    ]

    # ✅ Polynomial LR Scheduler (as callback, NOT optimizer schedule)
    if args.use_polynomial_lr:
        def polynomial_schedule(epoch):
            """Polynomial decay: LR = initial_lr * (1 - epoch/total)^0.9 + end_lr"""
            progress = epoch / args.num_epochs
            decayed_lr = args.learning_rate * ((1.0 - progress) ** 0.9)
            return max(decayed_lr, args.end_learning_rate)
        
        callbacks.append(
            LearningRateScheduler(polynomial_schedule, verbose=1)
        )
        print("✅ Using Polynomial LR schedule.")

    # Optional: ReduceLROnPlateau (if patience > 0)
    if args.patience > 0:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss', 
                patience=args.patience, 
                factor=0.5, 
                min_lr=args.end_learning_rate, 
                verbose=1
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor='val_loss', 
                patience=args.patience * 2, 
                restore_best_weights=True, 
                verbose=1
            )
        )
        print(f"✅ ReduceLROnPlateau & EarlyStopping enabled (patience={args.patience}).")

    # ==================== Train ====================
    print(f"\n🚀 Starting Training for {args.num_epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Initial LR: {args.learning_rate}")
    print(f"   End LR: {args.end_learning_rate}")
    print(f"   Freeze encoder: {args.freeze_encoder}")
    print(f"   Augmentation: {args.use_augmentation}\n")

    model.fit(
        train_dataset,
        epochs=args.num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✅ Training complete! Best model saved to: {model_path}")
    print(f"📈 Training log saved to: {csv_path}")
