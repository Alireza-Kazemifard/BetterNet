# --- START OF FILE test.py ---
import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm
from utils import load_trained_model, create_directory
from metrics import *


try:
    import pydensecrf.densecrf as dcrf
    CRF_AVAILABLE = True
except ImportError:
    print("Warning: pydensecrf not installed. Skipping CRF post-processing.")
    CRF_AVAILABLE = False

def apply_crf(image, probability_mask):
    if not CRF_AVAILABLE:
        return probability_mask
        
    
    image = (image * 255).astype(np.uint8)
    prob = probability_mask.squeeze()
    
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2) # 2 classes
    
    # Unary energy
    U = np.stack([1 - prob, prob], axis=0) # [2, H, W]
    U = -np.log(U + 1e-10) # Log probability
    U = U.reshape((2, -1)).astype(np.float32)
    d.setUnaryEnergy(U)

    # Pairwise
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(10)
    map = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return map.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    create_directory(args.save_dir)
    model = load_trained_model(args.model_path)

    from glob import glob
    test_x = sorted(glob(os.path.join(args.dataset_path, "images", "*")))
    test_y = sorted(glob(os.path.join(args.dataset_path, "masks", "*")))
    
    metrics_score = [0.0] * 5 # IoU, Dice, F-score, S-score, E-score

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.split(x)[1].split(".")[0]
        
        # Read
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        x_img = image / 255.0
        x_img_input = np.expand_dims(x_img, axis=0).astype(np.float32)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)
        mask_binary = (mask > 0.5).astype(np.float32)

        # Predict
        y_pred = model.predict(x_img_input, verbose=0)[0]
        
        # Apply CRF (Optional but recommended by paper)
        if CRF_AVAILABLE:
            y_pred_crf = apply_crf(x_img, y_pred)
            y_pred_binary = np.expand_dims(y_pred_crf, axis=-1)
        else:
            y_pred_binary = (y_pred > 0.5).astype(np.float32)

        # Metrics (Numpy based for accuracy in testing)
        # توجه: برای سادگی اینجا از توابع tf استفاده میکنیم و با .numpy() مقدار میگیریم
        # در پیاده سازی دقیق تر باید نسخه numpy توابع metrics را نوشت
        metrics_score[0] += intersection_over_union(mask_binary, y_pred_binary).numpy()
        metrics_score[1] += dice_coefficient(mask_binary, y_pred_binary).numpy()
        metrics_score[2] += weighted_f_score(mask_binary, y_pred_binary).numpy()
        metrics_score[3] += s_score(mask_binary, y_pred_binary).numpy()
        metrics_score[4] += e_score(mask_binary, y_pred_binary).numpy()

        # Visualization
        final_img = np.concatenate([
            image, 
            np.ones((224, 10, 3))*255,
            cv2.cvtColor((mask_binary*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            np.ones((224, 10, 3))*255,
            cv2.cvtColor((y_pred_binary*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ], axis=1)
        cv2.imwrite(f"{args.save_dir}/{name}.png", final_img)

    results = {
        "mIoU": metrics_score[0]/len(test_x),
        "mDice": metrics_score[1]/len(test_x),
        "F_score": metrics_score[2]/len(test_x),
        "S_score": metrics_score[3]/len(test_x),
        "E_score": metrics_score[4]/len(test_x),
    }
    
    print(results)
    pd.DataFrame([results]).to_csv(f"{args.save_dir}/results.csv", index=False)
