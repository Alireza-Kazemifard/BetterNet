import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm
from utils import load_trained_model, create_directory

# Fan toolbox metrics (local implementation)
try:
    import sys
    sys.path.insert(0, '/content/BetterNet')
    from eval.fan_metrics import MAE, Emeasure, Fmeasure, Smeasure, maxEmeasure
    FAN_AVAILABLE = True
    print("✅ Using Fan-style evaluation metrics")
except ImportError:
    FAN_AVAILABLE = False
    print("⚠️ Fan metrics not found. Using TensorFlow approximations.")

from data import load_test_data, IMAGE_HEIGHT, IMAGE_WIDTH
from metrics import dice_coefficient, intersection_over_union, mean_absolute_error

# CRF
try:
    import pydensecrf.densecrf as dcrf
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


def apply_morphology(mask01: np.ndarray):
    """Morphological operations (erosion + dilation)"""
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.erode(m, kernel, iterations=1)
    return m.astype(np.float32)


def apply_crf(rgb_uint8: np.ndarray, prob01: np.ndarray):
    """Dense CRF post-processing"""
    if not CRF_AVAILABLE:
        return (prob01 > 0.5).astype(np.float32)
    
    H, W = prob01.shape
    d = dcrf.DenseCRF2D(W, H, 2)
    
    # Unary potential
    p = np.clip(prob01, 1e-7, 1 - 1e-7)
    U = np.stack([1 - p, p], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)
    d.setUnaryEnergy(U)
    
    # Pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb_uint8, compat=10)
    
    Q = d.inference(10)
    pred = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
    return pred


def compute_metrics_fan_toolbox(y_true_uint8: np.ndarray, y_pred_uint8: np.ndarray):
    """
    Compute metrics using Fan-style evaluation
    Args:
        y_true_uint8: Ground truth (HxW, 0-255)
        y_pred_uint8: Prediction (HxW, 0-255)
    Returns:
        (iou, dice, fwb, s, e, maxe, mae)
    """
    # Normalize to [0,1]
    y_true_norm = y_true_uint8.astype(np.float32) / 255.0
    y_pred_norm = y_pred_uint8.astype(np.float32) / 255.0
    
    # Binary Dice & IoU (from TensorFlow metrics for consistency)
    yt_bin = (y_true_norm > 0.5).astype(np.float32).reshape(-1)
    yp_bin = (y_pred_norm > 0.5).astype(np.float32).reshape(-1)
    
    iou = float(intersection_over_union(yt_bin, yp_bin).numpy())
    dice = float(dice_coefficient(yt_bin, yp_bin).numpy())
    
    # Fan toolbox metrics
    if FAN_AVAILABLE:
        mae_score = MAE(y_pred_norm, y_true_norm)
        e_score = Emeasure(y_pred_norm, y_true_norm)
        max_e = maxEmeasure(y_pred_norm, y_true_norm, num_thresholds=100)
        f_score = Fmeasure(y_pred_norm, y_true_norm, beta=0.3)
        s_score = Smeasure(y_pred_norm, y_true_norm)
    else:
        # Fallback
        mae_score = float(mean_absolute_error(yt_bin, yp_bin).numpy())
        e_score = iou
        max_e = iou
        f_score = 0.0
        s_score = dice
    
    return iou, dice, f_score, s_score, e_score, max_e, mae_score


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split_mode", type=str, default="test", choices=["test", "full"])
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--use_morphology", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    create_directory(args.save_dir)
    print(f"⏳ Loading Model: {args.model_path}")
    
    # Enable unsafe deserialization for Lambda layers
    tf.keras.config.enable_unsafe_deserialization()
    model = load_trained_model(args.model_path)
    
    test_x, test_y = load_test_data(args.dataset_path, split_mode=args.split_mode, split=0.1, seed=42)
    print(f"📦 Testing on {len(test_x)} images (split_mode={args.split_mode})")
    
    if args.use_crf and not CRF_AVAILABLE:
        print("⚠️ CRF requested but pydensecrf not installed. Skipping CRF.")
    
    rows = []
    total_time = 0.0
    vis_dir = os.path.join(args.save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    for img_path, msk_path in tqdm(list(zip(test_x, test_y))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Read images
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        m = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        
        if bgr is None or m is None:
            continue
        
        # Resize
        bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x_in = rgb.astype(np.float32)  # EfficientNet expects 0-255
        
        # Ground truth
        gt = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        gt_uint8 = gt  # 0-255
        
        # Predict
        x = np.expand_dims(x_in, axis=0)
        t0 = time.time()
        pred = model.predict(x, verbose=0)[0, :, :, 0]  # [0,1]
        total_time += (time.time() - t0)
        
        pred01 = pred
        
        # CRF
        if args.use_crf and CRF_AVAILABLE:
            pred01 = apply_crf(rgb.astype(np.uint8), pred01)
        
        # Threshold
        pred01 = (pred01 > args.threshold).astype(np.float32)
        
        # Morphology
        if args.use_morphology:
            pred01 = apply_morphology(pred01)
        
        # Convert to uint8 for Fan metrics
        pred_uint8 = (pred01 * 255).astype(np.uint8)
        
        # Compute metrics
        iou, dice, fwb, s, e, maxe, mae = compute_metrics_fan_toolbox(gt_uint8, pred_uint8)
        rows.append([name, iou, dice, fwb, s, e, maxe, mae])
        
        # Save visualization
        overlay = bgr.copy()
        overlay[pred01 > 0.5] = (0.3 * overlay[pred01 > 0.5] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, f"{name}_overlay.png"), overlay)
    
    # Save results
    df = pd.DataFrame(rows, columns=["name", "IoU", "Dice", "Fwb", "S", "E", "maxE", "MAE"])
    df.to_csv(os.path.join(args.save_dir, "metrics.csv"), index=False)
    
    print("-" * 40)
    print(f"📊 Results for {os.path.basename(args.dataset_path)} (split_mode={args.split_mode})")
    print(df.mean(numeric_only=True))
    print(f"⏱ Avg inference time: {total_time / max(1, len(rows)):.4f} sec/image")
    print("-" * 40)
    print("✅ Done!")
