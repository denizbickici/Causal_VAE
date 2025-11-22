#!/usr/bin/env python3
"""
Simple evaluation script for extracted features
Calculates accuracy metrics from pre-extracted features and outputs
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report
import os

def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """Calculate top-k accuracy"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_features(feature_file, args):
    """Evaluate extracted features"""
    print(f"Loading features from: {feature_file}")
    
    # Load extracted features
    data = torch.load(feature_file, map_location='cpu')
    
    print("Available keys in feature file:", list(data.keys()))
    
    # Extract data
    features = data['feats']  # Shape: [N, num_crops, ...]
    cls_features = data['cls_feats']  # Classification features
    outputs = data['outputs']  # Model predictions
    targets = data['targets']  # Ground truth labels
    
    print(f"Features shape: {features.shape}")
    print(f"Classification features shape: {cls_features.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Number of classes: {outputs.shape[-1]}")
    print(f"Number of samples: {len(targets)}")
    
    # Handle multiple crops - average predictions across crops
    if len(outputs.shape) == 3:  # [N, num_crops, num_classes]
        print(f"Averaging across {outputs.shape[1]} crops")
        averaged_outputs = outputs.mean(dim=1)  # Average across crops
    else:
        averaged_outputs = outputs
    
    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(averaged_outputs, dim=1)
    
    # Calculate accuracy metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Top-1 and Top-5 accuracy
    acc1, acc5 = calculate_accuracy(averaged_outputs, targets, topk=(1, 5))
    print(f"Top-1 Accuracy: {acc1.item():.2f}%")
    print(f"Top-5 Accuracy: {acc5.item():.2f}%")
    
    # Using sklearn for additional metrics
    pred_labels = torch.argmax(averaged_outputs, dim=1).numpy()
    true_labels = targets.numpy()
    
    # Overall accuracy
    sklearn_acc = accuracy_score(true_labels, pred_labels) * 100
    print(f"Sklearn Top-1 Accuracy: {sklearn_acc:.2f}%")
    
    # Top-5 accuracy using sklearn
    if averaged_outputs.shape[1] >= 5:
        try:
            top5_sklearn = top_k_accuracy_score(true_labels, probs.numpy(), k=5) * 100
            print(f"Sklearn Top-5 Accuracy: {top5_sklearn:.2f}%")
        except ValueError as e:
            print(f"Sklearn Top-5 Accuracy: Error - {e}")
            print("Using PyTorch Top-5 accuracy instead")
    
    # Per-class accuracy (if requested)
    if args.detailed:
        print("\n" + "-"*30)
        print("DETAILED METRICS")
        print("-"*30)
        
        # Confusion matrix stats
        unique_labels = np.unique(true_labels)
        print(f"Number of unique classes in data: {len(unique_labels)}")
        print(f"Class distribution:")
        for label in unique_labels[:10]:  # Show first 10 classes
            count = np.sum(true_labels == label)
            print(f"  Class {label}: {count} samples")
        if len(unique_labels) > 10:
            print(f"  ... and {len(unique_labels) - 10} more classes")
        
        # Classification report (top 10 classes only for readability)
        if len(unique_labels) <= 20:
            print(f"\nClassification Report:")
            print(classification_report(true_labels, pred_labels, zero_division=0))
        else:
            print(f"\nSkipping detailed classification report (too many classes: {len(unique_labels)})")
    
    # Confidence statistics
    print("\n" + "-"*30)
    print("CONFIDENCE STATISTICS")
    print("-"*30)
    
    max_probs = torch.max(probs, dim=1)[0]
    print(f"Mean confidence: {max_probs.mean():.4f}")
    print(f"Std confidence: {max_probs.std():.4f}")
    print(f"Min confidence: {max_probs.min():.4f}")
    print(f"Max confidence: {max_probs.max():.4f}")
    
    # Correct vs incorrect predictions confidence
    correct_mask = (pred_labels == true_labels)
    if correct_mask.sum() > 0:
        correct_conf = max_probs[correct_mask].mean()
        print(f"Mean confidence (correct predictions): {correct_conf:.4f}")
    
    if (~correct_mask).sum() > 0:
        incorrect_conf = max_probs[~correct_mask].mean()
        print(f"Mean confidence (incorrect predictions): {incorrect_conf:.4f}")
    
    return {
        'top1_accuracy': acc1.item(),
        'top5_accuracy': acc5.item(),
        'num_samples': len(targets),
        'num_classes': outputs.shape[-1],
        'mean_confidence': max_probs.mean().item()
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate extracted features')
    parser.add_argument('--train-features', type=str, default='/home/dz/Projects/multi-modal_AR/LaViLa_adapted/egtea_train_feat.pt',
                        help='Path to training features file')
    parser.add_argument('--test-features', type=str, default='/home/dz/Projects/multi-modal_AR/LaViLa/output/mvit_temp_verb5/egtea_test_feat.pt',  
                        help='Path to test features file')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-class metrics')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Compare train vs test performance')
    
    args = parser.parse_args()
    
    # Evaluate test set
    if os.path.exists(args.test_features):
        print("EVALUATING TEST SET")
        print("="*60)
        test_results = evaluate_features(args.test_features, args)
    else:
        print(f"Test features file not found: {args.test_features}")
        test_results = None
    
    # Evaluate training set (if requested)
    if args.compare and os.path.exists(args.train_features):
        print("\n\nEVALUATING TRAINING SET")  
        print("="*60)
        train_results = evaluate_features(args.train_features, args)
        
        # Compare results
        if test_results:
            print("\n\nCOMPARISON")
            print("="*60)
            print(f"Training Top-1 Accuracy: {train_results['top1_accuracy']:.2f}%")
            print(f"Test Top-1 Accuracy: {test_results['top1_accuracy']:.2f}%")
            print(f"Accuracy Drop: {train_results['top1_accuracy'] - test_results['top1_accuracy']:.2f}%")
            
            print(f"Training Top-5 Accuracy: {train_results['top5_accuracy']:.2f}%")
            print(f"Test Top-5 Accuracy: {test_results['top5_accuracy']:.2f}%")
            print(f"Top-5 Accuracy Drop: {train_results['top5_accuracy'] - test_results['top5_accuracy']:.2f}%")
    
    elif args.compare:
        print(f"Training features file not found: {args.train_features}")

if __name__ == '__main__':
    main()