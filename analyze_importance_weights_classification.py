#!/usr/bin/env python3
"""
Script to analyze importance weights files and classify them based on min/max values.

This script:
1. Reads all files matching pattern "importance_weights_before_icepop*.bin" from log_data_importance_weights directory
2. For each file, filters out leading and trailing zero elements
3. Classifies files into two categories:
   - Category 1: max > 3 OR min < 0.3
   - Category 2: all values >= 0.3 AND <= 3
4. For each file, calculates ratio of elements > 1.2 OR < 0.8 (after filtering zeros)
5. Reports statistics for each category: file count and average ratio
"""

import numpy as np
import os
import glob
import argparse
from pathlib import Path


def read_and_filter_single_file(filename):
    """
    Read importance weights from a single binary file and filter leading/trailing zeros.
    
    Args:
        filename (str): Path to the binary file
        
    Returns:
        tuple: (filtered_array, file_stats) where file_stats contains processing info
    """
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found!")
        return None, None
    
    try:
        # Read the binary data
        with open(filename, "rb") as f:
            data_bytes = f.read()
        
        # Assume float32 (most common for PyTorch tensors)
        dtype = np.float32
        element_size = np.dtype(dtype).itemsize
        
        if len(data_bytes) % element_size != 0:
            print(f"Warning: File '{filename}' size not divisible by {element_size} bytes (float32 size)")
            return None, None
        
        # Reconstruct the array
        data_array = np.frombuffer(data_bytes, dtype=dtype)
        
        # Find first and last non-zero indices
        non_zero_indices = np.nonzero(data_array)[0]
        
        if len(non_zero_indices) == 0:
            print(f"Warning: All elements are zero in file '{filename}'!")
            return None, None
        
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        
        # Filter out leading and trailing zeros
        filtered_array = data_array[first_non_zero:last_non_zero + 1]
        
        # Create file statistics
        file_stats = {
            'filename': os.path.basename(filename),
            'full_path': filename,
            'original_length': len(data_array),
            'filtered_length': len(filtered_array),
            'leading_zeros_removed': first_non_zero,
            'trailing_zeros_removed': len(data_array) - last_non_zero - 1,
            'min_value': filtered_array.min(),
            'max_value': filtered_array.max(),
            'mean_value': filtered_array.mean(),
            'std_value': filtered_array.std()
        }
        
        return filtered_array, file_stats
        
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None, None


def find_importance_weight_files(directory):
    """
    Find all files matching the pattern "importance_weights_before_icepop*.bin" in the directory.
    
    Args:
        directory (str): Directory path to search
        
    Returns:
        list: List of matching file paths
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found!")
        return []
    
    # Create the search pattern - note the typo "before" as specified in requirements
    pattern = os.path.join(directory, "importance_weights_before_icepop*.bin")
    
    # Find all matching files
    matching_files = glob.glob(pattern)
    
    # Sort files for consistent processing order
    matching_files.sort()
    
    return matching_files


def classify_file(filtered_data):
    """
    Classify a file based on its min/max values.
    
    Args:
        filtered_data (np.ndarray): Array with leading/trailing zeros removed
        
    Returns:
        str: 'category1' if max > 3 OR min < 0.3, 'category2' otherwise
    """
    min_val = filtered_data.min()
    max_val = filtered_data.max()
    
    if max_val > 3 or min_val < 0.3:
        return 'category1'
    else:
        return 'category2'


def calculate_outlier_ratio(filtered_data):
    """
    Calculate the ratio of elements that are > 1.2 OR < 0.8.
    
    Args:
        filtered_data (np.ndarray): Array with leading/trailing zeros removed
        
    Returns:
        float: Ratio of outlier elements (between 0 and 1)
    """
    outlier_mask = (filtered_data > 1.2) | (filtered_data < 0.8)
    outlier_count = np.sum(outlier_mask)
    total_count = len(filtered_data)
    
    return outlier_count / total_count if total_count > 0 else 0.0


def analyze_files(directory):
    """
    Analyze all importance weight files in the directory.
    
    Args:
        directory (str): Directory containing the .bin files
        
    Returns:
        dict: Analysis results containing statistics for each category
    """
    # Find all matching files
    files = find_importance_weight_files(directory)
    
    if not files:
        print(f"No files matching pattern 'importance_weights_before_icepop*.bin' found in '{directory}'")
        return None
    
    print(f"Found {len(files)} matching files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    print("\n" + "=" * 80)
    print("Processing files...")
    
    # Initialize data structures for analysis
    category1_files = []  # max > 3 OR min < 0.3
    category2_files = []  # all values >= 0.3 AND <= 3
    failed_files = []
    
    # Process each file
    for i, filename in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {os.path.basename(filename)}")
        
        filtered_data, file_stats = read_and_filter_single_file(filename)
        
        if filtered_data is not None and file_stats is not None:
            # Classify the file
            category = classify_file(filtered_data)
            
            # Calculate outlier ratio
            outlier_ratio = calculate_outlier_ratio(filtered_data)
            
            # Create analysis record
            analysis_record = {
                'filename': file_stats['filename'],
                'full_path': filename,
                'category': category,
                'min_value': file_stats['min_value'],
                'max_value': file_stats['max_value'],
                'filtered_length': file_stats['filtered_length'],
                'outlier_ratio': outlier_ratio,
                'outlier_percentage': outlier_ratio * 100
            }
            
            # Add to appropriate category
            if category == 'category1':
                category1_files.append(analysis_record)
            else:
                category2_files.append(analysis_record)
            
            print(f"  Category: {category}")
            print(f"  Min/Max: [{file_stats['min_value']:.6f}, {file_stats['max_value']:.6f}]")
            print(f"  Filtered length: {file_stats['filtered_length']}")
            print(f"  Outlier ratio (>1.2 or <0.8): {outlier_ratio:.4f} ({outlier_ratio*100:.2f}%)")
            
        else:
            failed_files.append(filename)
            print(f"  Failed to process file")
    
    # Prepare results
    results = {
        'total_files': len(files),
        'successful_files': len(category1_files) + len(category2_files),
        'failed_files': len(failed_files),
        'category1': {
            'description': 'Files with max > 3 OR min < 0.3',
            'count': len(category1_files),
            'files': category1_files,
            'average_outlier_ratio': np.mean([f['outlier_ratio'] for f in category1_files]) if category1_files else 0.0
        },
        'category2': {
            'description': 'Files with all values >= 0.3 AND <= 3',
            'count': len(category2_files),
            'files': category2_files,
            'average_outlier_ratio': np.mean([f['outlier_ratio'] for f in category2_files]) if category2_files else 0.0
        },
        'failed_file_list': failed_files
    }
    
    return results


def print_detailed_results(results):
    """
    Print detailed analysis results.
    
    Args:
        results (dict): Analysis results from analyze_files()
    """
    print(f"\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"Total files found: {results['total_files']}")
    print(f"Successfully processed: {results['successful_files']}")
    print(f"Failed to process: {results['failed_files']}")
    
    if results['failed_files'] > 0:
        print(f"\nFailed files:")
        for failed_file in results['failed_file_list']:
            print(f"  - {os.path.basename(failed_file)}")
    
    print(f"\n" + "-" * 60)
    print("CATEGORY 1: Files with max > 3 OR min < 0.3")
    print("-" * 60)
    print(f"File count: {results['category1']['count']}")
    print(f"Average outlier ratio: {results['category1']['average_outlier_ratio']:.4f} ({results['category1']['average_outlier_ratio']*100:.2f}%)")
    
    if results['category1']['files']:
        print(f"\nDetailed file list:")
        for i, file_info in enumerate(results['category1']['files'], 1):
            print(f"  {i:2d}. {file_info['filename']}")
            print(f"      Range: [{file_info['min_value']:.6f}, {file_info['max_value']:.6f}]")
            print(f"      Outlier ratio: {file_info['outlier_ratio']:.4f} ({file_info['outlier_percentage']:.2f}%)")
    
    print(f"\n" + "-" * 60)
    print("CATEGORY 2: Files with all values >= 0.3 AND <= 3")
    print("-" * 60)
    print(f"File count: {results['category2']['count']}")
    print(f"Average outlier ratio: {results['category2']['average_outlier_ratio']:.4f} ({results['category2']['average_outlier_ratio']*100:.2f}%)")
    
    if results['category2']['files']:
        print(f"\nDetailed file list:")
        for i, file_info in enumerate(results['category2']['files'], 1):
            print(f"  {i:2d}. {file_info['filename']}")
            print(f"      Range: [{file_info['min_value']:.6f}, {file_info['max_value']:.6f}]")
            print(f"      Outlier ratio: {file_info['outlier_ratio']:.4f} ({file_info['outlier_percentage']:.2f}%)")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Category 1 (max>3 OR min<0.3): {results['category1']['count']} files, avg outlier(>1.2 or <0.8) ratio: {results['category1']['average_outlier_ratio']*100:.2f}%")
    print(f"Category 2 (0.3<=values<=3):   {results['category2']['count']} files, avg outlier(>1.2 or <0.8) ratio: {results['category2']['average_outlier_ratio']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze and classify importance weight .bin files')
    parser.add_argument('--directory', type=str, default='log_data_importance_weights', 
                       help='Directory containing the .bin files (default: log_data_importance_weights)')
    
    args = parser.parse_args()
    
    print(f"Processing directory: {args.directory}")
    print(f"Looking for files matching: importance_weights_before_icepop*.bin")
    print("=" * 80)
    
    # Analyze all files in the directory
    results = analyze_files(args.directory)
    
    if results is not None:
        # Print detailed results
        print_detailed_results(results)
        
        print(f"\n=== Processing Complete ===")
        
    else:
        print("Failed to process any files.")


if __name__ == "__main__":
    # If run without arguments, show help
    import sys
    if len(sys.argv) == 1:
        print("Usage examples:")
        print(f"  python {sys.argv[0]}")
        print(f"  python {sys.argv[0]} --directory ./my_data_folder")
        print(f"\nRun 'python {sys.argv[0]} --help' for full help")
    else:
        main()
