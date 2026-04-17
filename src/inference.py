"""
Inference script for CSIRO Pasture Biomass Challenge
Loads 3 separate models (Dead, Clover, Green) and computes all 5 targets
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BiomassDatasetMulti, get_val_transforms, collate_fn
from models.dual_crop_vit import BiomassModelSingle


class Config:
    """Configuration for inference"""
    DATA_DIR = Path("./data")
    MODEL_DIR = Path("./v2_trained")
    OUTPUT_DIR = Path("./submissions")

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
    IMG_SIZE = 512
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    N_FOLDS = 4

    USE_LOG_TRANSFORM = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_fold(test_df, fold, config, device):
    """
    Make predictions using models from a single fold.
    
    Args:
        test_df: Test DataFrame
        fold: Fold number
        config: Configuration object
        device: Device to run on
        
    Returns:
        predictions: Dict with predictions for each target
    """
    print(f"Loading models for fold {fold}...")
    
    # Create test dataset
    test_dataset = BiomassDatasetMulti(
        test_df,
        get_val_transforms(config.IMG_SIZE),
        config.DATA_DIR / "test_images",
        config.USE_LOG_TRANSFORM
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Load the 3 models for this fold
    models = {}
    for target_name in ["green", "dead", "clover"]:
        model = BiomassModelSingle(config.MODEL_NAME, pretrained=False).to(device)
        checkpoint_path = config.MODEL_DIR / f"fold{fold}_{target_name}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models[target_name] = model
    
    # Collect predictions
    all_preds = {"green": [], "dead": [], "clover": []}
    
    with torch.no_grad():
        for (imgs1, imgs2), _ in tqdm(test_loader, desc=f"Fold {fold}", leave=False):
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            
            # Predict each target
            pred_green = models["green"]((imgs1, imgs2)).cpu().numpy()
            pred_dead = models["dead"]((imgs1, imgs2)).cpu().numpy()
            pred_clover = models["clover"]((imgs1, imgs2)).cpu().numpy()
            
            # Inverse transform if needed
            if config.USE_LOG_TRANSFORM:
                pred_green = np.expm1(np.clip(pred_green, -20, 20))
                pred_dead = np.expm1(np.clip(pred_dead, -20, 20))
                pred_clover = np.expm1(np.clip(pred_clover, -20, 20))
            
            all_preds["green"].append(pred_green)
            all_preds["dead"].append(pred_dead)
            all_preds["clover"].append(pred_clover)
    
    # Concatenate predictions
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k])
    
    # Clean up models
    for model in models.values():
        del model
    torch.cuda.empty_cache()
    
    return all_preds


def main():
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_csv_path = config.DATA_DIR / "test.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)
    
    # Need to pivot test data to match BiomassDatasetMulti expectations
    # For test, we don't have targets, so create dummy ones
    test_df['Dry_Green_g'] = 0.0
    test_df['Dry_Dead_g'] = 0.0
    test_df['Dry_Clover_g'] = 0.0
    
    test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    
    print(f"Test images: {len(test_wide)}")
    
    # Collect predictions from all folds
    fold_predictions = []
    
    for fold in range(config.N_FOLDS):
        preds = predict_fold(test_wide, fold, config, device)
        fold_predictions.append(preds)
    
    # Average predictions across folds
    print("\nAveraging predictions across folds...")
    avg_preds = {
        "green": np.mean([p["green"] for p in fold_predictions], axis=0),
        "dead": np.mean([p["dead"] for p in fold_predictions], axis=0),
        "clover": np.mean([p["clover"] for p in fold_predictions], axis=0)
    }
    
    # Compute derived targets
    pred_gdm = avg_preds["green"] + avg_preds["clover"]
    pred_total = avg_preds["green"] + avg_preds["dead"] + avg_preds["clover"]
    
    # Create submission file
    print("Creating submission file...")
    
    target_names = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    predictions = [
        avg_preds["green"],
        avg_preds["dead"],
        avg_preds["clover"],
        pred_gdm,
        pred_total
    ]
    
    submission_data = []
    for i, row in test_wide.iterrows():
        sample_id = row['sample_id'] if 'sample_id' in row else row['image_path']
        
        for target_name, pred in zip(target_names, predictions):
            submission_data.append({
                'sample_id': sample_id,
                'target_name': target_name,
                'target': pred[i]
            })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission
    output_path = config.OUTPUT_DIR / 'submission.csv'
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission_df.shape}")
    print(f"\nSample predictions:")
    print(submission_df.head(10))
    
    # Print summary statistics
    print(f"\nPrediction Statistics:")
    for target_name, pred in zip(target_names, predictions):
        print(f"  {target_name:15s} - Mean: {pred.mean():8.2f}, Std: {pred.std():8.2f}, "
              f"Min: {pred.min():8.2f}, Max: {pred.max():8.2f}")


if __name__ == '__main__':
    main()
