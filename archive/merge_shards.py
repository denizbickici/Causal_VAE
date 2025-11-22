#!/usr/bin/env python3
"""
Merge validation feature shards into a single file with memory-efficient incremental processing.
Usage: python merge_shards.py --input-dir output/ [--output egtea_test_feat.pt]
"""

import argparse
import os
import glob
import torch
import gc
import psutil

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def print_memory_status(stage):
    """Print memory usage at different stages"""
    mem_gb = get_memory_usage()
    total_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    available_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
    print(f"[{stage}] Process memory: {mem_gb:.2f}GB | Available: {available_gb:.2f}GB / {total_gb:.2f}GB")

def main():
    parser = argparse.ArgumentParser(description='Merge validation feature shards')
    parser.add_argument('--input-dir', required=True, help='Directory containing shard files', default='output/')
    parser.add_argument('--output', default='egtea_test_feat.pt', help='Output filename')
    args = parser.parse_args()
    
    # Find all shard files
    shard_pattern = os.path.join(args.input_dir, 'egtea_test_feat_shard_*.pt')
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        print(f"No shard files found in {args.input_dir}")
        return
    
    print(f"Found {len(shard_files)} shard files")
    print_memory_status("Initial")
    
    # Initialize with the first shard
    print(f"\nLoading initial shard 1/{len(shard_files)}: {os.path.basename(shard_files[0])}")
    merged_data = torch.load(shard_files[0], map_location='cpu')
    print(f"  Initial shapes - feats: {merged_data['feats'].shape}, cls_feats: {merged_data['cls_feats'].shape}")
    print_memory_status("After loading first shard")
    
    # Incrementally merge remaining shards
    for i, shard_file in enumerate(shard_files[1:], start=2):
        print(f"\nProcessing shard {i}/{len(shard_files)}: {os.path.basename(shard_file)}")
        
        # Load next shard
        shard_data = torch.load(shard_file, map_location='cpu')
        print(f"  Shard shapes - feats: {shard_data['feats'].shape}, cls_feats: {shard_data['cls_feats'].shape}")
        
        # Concatenate with existing data
        print("  Concatenating with existing data...")
        merged_data['feats'] = torch.cat([merged_data['feats'], shard_data['feats']], dim=0)
        merged_data['cls_feats'] = torch.cat([merged_data['cls_feats'], shard_data['cls_feats']], dim=0)
        merged_data['outputs'] = torch.cat([merged_data['outputs'], shard_data['outputs']], dim=0)
        merged_data['targets'] = torch.cat([merged_data['targets'], shard_data['targets']], dim=0)
        
        # Clear the shard data from memory
        del shard_data
        gc.collect()
        
        print(f"  Cumulative shapes - feats: {merged_data['feats'].shape}, cls_feats: {merged_data['cls_feats'].shape}")
        print_memory_status(f"After merging shard {i}")
    
    # Save merged data
    output_path = os.path.join(args.input_dir, args.output)
    print(f"\nSaving merged data to {output_path}")
    torch.save(merged_data, output_path)
    
    print(f"\nMerging complete!")
    print(f"Total samples: {merged_data['feats'].shape[0]}")
    print(f"Feature shape: {merged_data['feats'].shape}")
    print(f"CLS feature shape: {merged_data['cls_feats'].shape}")
    print(f"Outputs shape: {merged_data['outputs'].shape}")
    print(f"Targets shape: {merged_data['targets'].shape}")
    print_memory_status("Final")

if __name__ == '__main__':
    main()
