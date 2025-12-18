"""
Fan-style Evaluation Metrics (PraNet/MICCAI 2020)
Simplified implementation compatible with test.py
"""
import numpy as np


def MAE(pred, gt):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - gt))


def Emeasure(pred, gt):
    """
    Enhanced-alignment measure
    Reference: Fan et al. "Enhanced-alignment Measure" (IJCAI 2018)
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Threshold
    threshold = 2 * pred.mean()
    if threshold > 1:
        threshold = 1
    
    binary_pred = (pred >= threshold).astype(np.float32)
    binary_gt = (gt >= 0.5).astype(np.float32)
    
    if binary_gt.sum() == 0:  # No foreground
        enhanced_matrix = 1.0 - binary_pred
    elif binary_pred.sum() == 0:  # No prediction
        enhanced_matrix = 1.0 - binary_gt
    else:
        align_matrix = 2 * (binary_pred * binary_gt) / (binary_pred + binary_gt + 1e-8)
        enhanced = align_matrix
        enhanced_matrix = enhanced
    
    return enhanced_matrix.mean()


def Fmeasure(pred, gt, beta=0.3):
    """
    Weighted F-measure (F_beta)
    beta=0.3 emphasizes precision over recall
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    
    if gt_bin.sum() == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
    
    TP = np.sum(pred_bin * gt_bin)
    FP = np.sum(pred_bin * (1 - gt_bin))
    FN = np.sum((1 - pred_bin) * gt_bin)
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    beta_sq = beta * beta
    fmeasure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)
    
    return fmeasure


def Smeasure(pred, gt):
    """
    Structure-measure
    Reference: Fan et al. "Structure-measure" (ICCV 2017)
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    
    # Object-aware structural similarity
    fg_mask = gt > 0.5
    bg_mask = gt <= 0.5
    
    if fg_mask.sum() > 0:
        pred_fg_mean = pred[fg_mask].mean()
        gt_fg_mean = gt[fg_mask].mean()
        object_score = 1 - np.abs(pred_fg_mean - gt_fg_mean)
    else:
        object_score = 1.0
    
    # Region-aware structural similarity
    if bg_mask.sum() > 0:
        pred_bg_mean = pred[bg_mask].mean()
        region_score = 1 - pred_bg_mean
    else:
        region_score = 1.0
    
    # Combine (equal weight)
    alpha = 0.5
    S = alpha * object_score + (1 - alpha) * region_score
    
    return np.clip(S, 0, 1)


def maxEmeasure(pred, gt, num_thresholds=255):
    """
    Max E-measure across different thresholds
    """
    max_e = 0.0
    thresholds = np.linspace(0, 1, num_thresholds)
    
    for thresh in thresholds:
        pred_thresh = (pred >= thresh).astype(np.float32)
        e = Emeasure(pred_thresh, gt)
        if e > max_e:
            max_e = e
    
    return max_e
