#!/usr/bin/env python3
"""
Feature Inspector
Print the first 10 rows of image_feat.npy and text_feat.npy files in dataset directory.
"""

import os
import numpy as np
import argparse


def inspect_features(data_name='Games'):
    """
    Inspect and print the first 10 rows of feature files
    
    Args:
        data_name (str): Dataset name, default is 'Games'
    """
    # Get the dataset path
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, '..', 'data', data_name)
    
    print(f"Dataset path: {dataset_path}")
    print("=" * 80)
    
    # Check image features
    image_feat_path = os.path.join(dataset_path, 'image_feat.npy')
    text_feat_path = os.path.join(dataset_path, 'text_feat.npy')
    
    # Inspect image features
    if os.path.exists(image_feat_path):
        print(f"\n IMAGE FEATURES ({image_feat_path})")
        print("-" * 60)
        try:
            image_feat = np.load(image_feat_path, allow_pickle=True)
            print(f"Shape: {image_feat.shape}")
            print(f"Data type: {image_feat.dtype}")
            print("\nFirst 10 rows:")
            for i in range(min(10, len(image_feat))):
                print(f"Row {i:2d}: {image_feat[i][:10]}... (showing first 10 dimensions)")
            
            # Additional statistics
            print(f"\nStatistics:")
            print(f"Min value: {np.min(image_feat):.6f}")
            print(f"Max value: {np.max(image_feat):.6f}")
            print(f"Mean value: {np.mean(image_feat):.6f}")
            print(f"Std value: {np.std(image_feat):.6f}")
            if image_feat.size > 0:
                sparsity = (image_feat == 0).sum() / image_feat.size
                print(f"Sparsity: {sparsity:.2%}")
            
        except Exception as e:
            print(f"Error loading image features: {e}")
    else:
        print(f"\n Image features file not found: {image_feat_path}")
    
    # Inspect text features
    if os.path.exists(text_feat_path):
        print(f"\n📝 TEXT FEATURES ({text_feat_path})")
        print("-" * 60)
        try:
            text_feat = np.load(text_feat_path, allow_pickle=True)
            print(f"Shape: {text_feat.shape}")
            print(f"Data type: {text_feat.dtype}")
            print("\nFirst 10 rows:")
            for i in range(min(10, len(text_feat))):
                print(f"Row {i:2d}: {text_feat[i][:10]}... (showing first 10 dimensions)")
            
            # Additional statistics
            print(f"\nStatistics:")
            print(f"Min value: {np.min(text_feat):.6f}")
            print(f"Max value: {np.max(text_feat):.6f}")
            print(f"Mean value: {np.mean(text_feat):.6f}")
            print(f"Std value: {np.std(text_feat):.6f}")
            if text_feat.size > 0:
                sparsity = (text_feat == 0).sum() / text_feat.size
                print(f"Sparsity: {sparsity:.2%}")
            
        except Exception as e:
            print(f"Error loading text features: {e}")
    else:
        print(f"\n Text features file not found: {text_feat_path}")
    
    print("\n" + "=" * 80)


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Inspect image and text feature files')
    parser.add_argument('--dataset', '-d', type=str, default='Games', 
                       help='Dataset name (default: Games)')
    
    args = parser.parse_args()
    
    print(f"Inspecting features for dataset: {args.dataset}")
    inspect_features(args.dataset)


if __name__ == '__main__':
    main()
