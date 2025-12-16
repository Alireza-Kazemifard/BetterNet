import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

from utils import load_trained_model, create_directory
from metrics import (
    intersection_over_union, dice_coefficient, weighted_f_score,
    s_score, e_score, max_e_score, mean_absolute_error
)
from data import load_test_data, IMAGE_HEIGHT, IMAGE_WIDTH


# CRF
try:
    import pydensecrf.densecrf as dcrf
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


def apply_morphology(mask01: np.ndarray):
    # mask01: 0/1 uint8 or float
    m = (mask01 > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.erode(m, kernel, iterations=1)
    return m.astype(np.float32)


def apply_crf(rgb_uint8: np.ndarray, prob01: np.ndarray):
    """
    rgb_uint8: HxWx3 uint8 (RGB)
    prob01: HxW float in [0,1] (foreground prob)
    returns: HxW float (0/1)
    """
    if not CRF_AVAILABLE:
        return (prob01 > 0.5).astype(np.float32)

    H, W = prob01.shape
    d = dcrf.DenseCRF2D(W, H, 2)

    # Unary
    p = np.clip(prob01, 1e-7, 1 - 1e-7)
    U = np.stack([1 - p, p], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)
    d.setUnaryEnergy(U)

    # Pairwise
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb_uint8, compat=10)

    Q = d.inference(10)
    pred = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
    return pred


def compute_metrics_np(y_true01: np.ndarray, y_pred01: np.ndarray):
    # flatten for tf-metrics (keeps consistent with saved funcs)
    yt = y_true01.reshape(-1).astype(np.float32)
    yp = y_pred01.reshape(-1).astype(np.float32)

    iou = float(intersection_over_union(yt, yp).numpy())
    dice = float(dice_coefficient(yt, yp).numpy())
    f = float(weighted_f_score(yt, yp).numpy())
    s = float(s_score(yt, yp).numpy())
    e = float(e_score(yt, yp).numpy())
    me = float(max_e_score(yt, yp).numpy())
    mae = float(mean_absolute_error(yt, yp).numpy())
    return iou, dice, f, s, e, me, mae


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
    model = load_trained_model(args.model_path)

    test_x, test_y = load_test_data(args.dataset_path, split_mode=args.split_mode, split=0.1, seed=42)
    print(f"📦 Testing on {len(test_x)} images. split_mode={args.split_mode}")
    if args.use_crf and not CRF_AVAILABLE:
        print("⚠️ CRF requested but pydensecrf is not installed. Skipping CRF.")

    rows = []
    total_time = 0.0

    for img_path, msk_path in tqdm(list(zip(test_x, test_y))):
        name = os.path.splitext(os.path.basename(img_path))[0]

        # read + preprocess (MATCH train)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        m = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or m is None:
            continue

        bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb01 = rgb.astype(np.float32) / 255.0

        m = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        m01 = (m.astype(np.float32) / 255.0) > 0.5
        m01 = m01.astype(np.float32)

        x = np.expand_dims(rgb01, axis=0)

        t0 = time.time()
        pred = model.predict(x, verbose=0)[0, :, :, 0]  # prob
        total_time += (time.time() - t0)

        pred01 = pred

        if args.use_crf and CRF_AVAILABLE:
            pred01 = apply_crf(rgb.astype(np.uint8), pred01)

        # threshold
        pred01 = (pred01 > args.threshold).astype(np.float32)

        if args.use_morphology:
            pred01 = apply_morphology(pred01)

        # metrics
        iou, dice, f, s, e, me, mae = compute_metrics_np(m01, pred01)
        rows.append([name, iou, dice, f, s, e, me, mae])

        # save visualization
        vis_dir = os.path.join(args.save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        overlay = bgr.copy()
        overlay[pred01 > 0.5] = (0.3 * overlay[pred01 > 0.5] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, f"{name}_overlay.png"), overlay)

    df = pd.DataFrame(rows, columns=["name", "IoU", "Dice", "Fwb", "S", "E", "maxE", "MAE"])
    df.to_csv(os.path.join(args.save_dir, "metrics.csv"), index=False)

    print(df.mean(numeric_only=True))
    print(f"⏱ Avg inference time: {total_time / max(1, len(rows)):.4f} sec/image")
    print("✅ Done.")
