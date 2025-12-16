import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm
from data import load_data
from utils import load_model, create_directory
from metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    create_directory(args.save_dir)
    print(f"Loading Model from {args.model_path}...")
    model = load_model(args.model_path)

    # Load images directly (for testing entire dataset usually, or split if preferred)
    from glob import glob
    test_x = sorted(glob(os.path.join(args.dataset_path, "images", "*")))
    test_y = sorted(glob(os.path.join(args.dataset_path, "masks", "*")))
    print(f"Testing on {len(test_x)} images.")

    metrics_score = [0.0] * 3 # IoU, Dice, MAE
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.split(x)[1].split(".")[0]
        
        # 1. Read & Preprocess Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x_img = cv2.resize(image, (256, 256)) / 255.0
        x_img = np.expand_dims(x_img, axis=0).astype(np.float32)

        # 2. Read Ground Truth Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256)) / 255.0
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)

        # 3. Predict
        start = time.perf_counter()
        y_pred = model.predict(x_img, verbose=0)[0]
        time_taken.append(time.perf_counter() - start)

        # 4. BINARIZE for Scientific Metrics (Dice/IoU requires binary)
        # We apply threshold 0.5 to both GT and Pred for fair comparison
        mask_binary = (mask > 0.5).astype(np.float32)
        y_pred_binary = (y_pred > 0.5).astype(np.float32)

        # 5. Save Visualization (Smooth for better visual appeal)
        y_pred_uint8 = (y_pred * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        final_img = np.concatenate([
            cv2.resize(image, (256, 256)), 
            np.ones((256, 10, 3))*255,
            cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR),
            np.ones((256, 10, 3))*255,
            cv2.cvtColor(y_pred_uint8, cv2.COLOR_GRAY2BGR)
        ], axis=1)
        cv2.imwrite(f"{args.save_dir}/{name}.png", final_img)

        # 6. Calculate Metrics (Strictly on Binary Masks)
        metrics_score[0] += intersection_over_union(mask_binary, y_pred_binary)
        metrics_score[1] += dice_coefficient(mask_binary, y_pred_binary)
        metrics_score[2] += mean_absolute_error(mask, y_pred) # MAE on probability map

    mean_fps = 1.0 / np.mean(time_taken)
    print(f"mDice: {metrics_score[1]/len(test_x):.4f} | FPS: {mean_fps:.2f}")

    pd.DataFrame([{
        "mIoU": metrics_score[0]/len(test_x),
        "mDice": metrics_score[1]/len(test_x),
        "MAE": metrics_score[2]/len(test_x),
        "FPS": mean_fps
    }]).to_csv(f"{args.save_dir}/results.csv", index=False)
