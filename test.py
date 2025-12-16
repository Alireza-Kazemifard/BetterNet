import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from glob import glob
from utils import load_trained_model, create_directory
from metrics import *

# Check for CRF
try:
    import pydensecrf.densecrf as dcrf
    CRF_AVAILABLE = True
except ImportError:
    print("Warning: pydensecrf not installed. Skipping CRF.")
    CRF_AVAILABLE = False

def apply_crf(image, probability_mask):
    if not CRF_AVAILABLE: return probability_mask
    image = (image * 255).astype(np.uint8)
    prob = probability_mask.squeeze()
    
    # Setup CRF
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    
    # Unary energy
    U = np.stack([1 - prob, prob], axis=0)
    U = -np.log(U + 1e-10)
    U = U.reshape((2, -1)).astype(np.float32)
    d.setUnaryEnergy(U)
    
    # Pairwise energy
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    # Inference
    Q = d.inference(10)
    map = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return map.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .keras model file")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--split_mode", type=str, default="full", choices=["test", "full"], 
                        help="test: Use 10% split (unseen during train), full: Use entire dataset")
    args = parser.parse_args()

    create_directory(args.save_dir)
    print(f"⏳ Loading Model from {args.model_path}...")
    model = load_trained_model(args.model_path)

    # 1. Load All Files First
    all_images = sorted(glob(os.path.join(args.dataset_path, "images", "*")))
    all_masks = sorted(glob(os.path.join(args.dataset_path, "masks", "*")))
    
    # Validation checks
    all_images = [x for x in all_images if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    all_masks = [x for x in all_masks if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if len(all_images) == 0:
        print("❌ No images found in the specified path.")
        exit()

    # 2. Handle Split Logic
    if args.split_mode == "test":
        print("✂️ Mode: TEST SPLIT (Using 10% unseen data)")
        # Must use same random_state as training to ensure we pick the correct 'unseen' part
        # We discard the training part (_) and keep the test part
        _, test_x, _, test_y = train_test_split(all_images, all_masks, test_size=0.1, random_state=42)
    else:
        print("📂 Mode: FULL DATASET (Generalization Test)")
        test_x, test_y = all_images, all_masks

    print(f"🎯 Testing on {len(test_x)} images.")
    
    metrics_score = [0.0] * 5  # IoU, Dice, F, S, E
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.split(x)[1].split(".")[0]
        
        # Read & Preprocess
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        if image is None: continue
        
        # Resize using Lanczos4 (Best quality)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        x_img = image / 255.0
        x_img_input = np.expand_dims(x_img, axis=0).astype(np.float32)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        mask = mask / 255.0
        mask_binary = (mask > 0.5).astype(np.float32)
        
        # Inference
        start_t = time.perf_counter()
        y_pred = model.predict(x_img_input, verbose=0)[0]
        end_t = time.perf_counter()
        time_taken.append(end_t - start_t)
        
        # Post-process (CRF)
        if CRF_AVAILABLE:
            y_pred_crf = apply_crf(x_img, y_pred)
            y_pred_binary = np.expand_dims(y_pred_crf, axis=-1)
        else:
            y_pred_binary = (y_pred > 0.5).astype(np.float32)

        # Metrics Calculation
        metrics_score[0] += intersection_over_union(mask_binary, y_pred_binary).numpy()
        metrics_score[1] += dice_coefficient(mask_binary, y_pred_binary).numpy()
        metrics_score[2] += weighted_f_score(mask_binary, y_pred_binary).numpy()
        metrics_score[3] += s_score(mask_binary, y_pred_binary).numpy()
        metrics_score[4] += e_score(mask_binary, y_pred_binary).numpy()
        
        # Save Sample Visuals (First 20 images)
        if i < 20: 
            final_img = np.concatenate([
                image, 
                np.ones((224, 10, 3))*255,
                cv2.cvtColor((mask_binary*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                np.ones((224, 10, 3))*255,
                cv2.cvtColor((y_pred_binary*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            ], axis=1)
            cv2.imwrite(f"{args.save_dir}/{name}.png", final_img)

    # Calculate Averages & Save
    if len(test_x) > 0:
        results = {
            "mIoU": metrics_score[0]/len(test_x),
            "mDice": metrics_score[1]/len(test_x),
            "F_score": metrics_score[2]/len(test_x),
            "S_score": metrics_score[3]/len(test_x),
            "E_score": metrics_score[4]/len(test_x),
            "FPS": 1.0 / np.mean(time_taken) if time_taken else 0
        }
        
        print("-" * 30)
        print(f"📊 Results for {os.path.basename(args.dataset_path)}:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print("-" * 30)

        pd.DataFrame([results]).to_csv(f"{args.save_dir}/results.csv", index=False)
    else:
        print("⚠️ No images found to test.")
