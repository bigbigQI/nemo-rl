#!/usr/bin/env python3
"""
Script to read multiple importance weights .bin files from a folder, filter leading/trailing zeros,
combine all data, and create a histogram with custom binning: 0-2 in 0.2 intervals, >2 as one bin.

This script:
1. Finds all files matching pattern "importance_weights_before_icepop*.bin" in a directory
2. Reads each binary file (assumes float32 format)
3. Filters out leading and trailing zero elements from each file
4. Combines all filtered data into a single array
5. Creates a histogram with custom binning and percentage labels on each bar
"""

import numpy as np
import matplotlib.pyplot as plt
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
            'original_length': len(data_array),
            'filtered_length': len(filtered_array),
            'leading_zeros_removed': first_non_zero,
            'trailing_zeros_removed': len(data_array) - last_non_zero - 1,
            'min_value': filtered_array.min(),
            'max_value': filtered_array.max(),
            'mean_value': filtered_array.mean(),
            'std_value': filtered_array.std(),
            'zero_count': np.sum(filtered_array == 0)
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
    
    # Create the search pattern
    pattern = os.path.join(directory, "importance_weights_before_icepop*.bin")
    
    # Find all matching files
    matching_files = glob.glob(pattern)
    
    # Sort files for consistent processing order
    matching_files.sort()
    
    return matching_files


def process_multiple_files(directory):
    """
    Process all importance weight files in the directory.
    
    Args:
        directory (str): Directory containing the .bin files
        
    Returns:
        tuple: (combined_data, all_file_stats) where combined_data is the merged array
    """
    # Find all matching files
    files = find_importance_weight_files(directory)
    
    if not files:
        print(f"No files matching pattern 'importance_weights_before_icepop*.bin' found in '{directory}'")
        return None, []
    
    print(f"Found {len(files)} matching files:")
    
    print("\n" + "=" * 60)
    print("Processing files...")
    
    all_filtered_data = []
    all_file_stats = []
    
    # Process each file
    for i, filename in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {os.path.basename(filename)}")
        
        filtered_data, file_stats = read_and_filter_single_file(filename)
        
        if filtered_data is not None and file_stats is not None:
            all_filtered_data.append(filtered_data)
            all_file_stats.append(file_stats)
        else:
            print(f"  Failed to process file")
    
    if not all_filtered_data:
        print("No files were successfully processed!")
        return None, []
    
    # Combine all filtered data
    print(f"\n" + "=" * 60)
    print("Combining data from all files...")
    combined_data = np.concatenate(all_filtered_data)
    
    print(f"Successfully combined data from {len(all_filtered_data)} files")
    print(f"Total combined elements: {len(combined_data)}")
    
    return combined_data, all_file_stats


def create_custom_bins(data):
    """
    Create custom bins: 0-2 in 0.2 intervals, >2 as one bin.
    
    Args:
        data (np.ndarray): Input data array
        
    Returns:
        tuple: (bin_edges, bin_labels, bin_counts, bin_percentages)
    """
    # Create bin edges: [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, inf]
    bin_edges = np.arange(0, 2.2, 0.2)  # 0, 0.2, 0.4, ..., 2.0
    bin_edges = np.append(bin_edges, np.inf)  # Add infinity for >2
    
    # Create bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:  # Last bin for >2
            bin_labels.append(">2.0")
        else:
            bin_labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")
    
    # Count data in each bin
    bin_counts, _ = np.histogram(data, bins=bin_edges)
    
    # Calculate percentages
    total_count = len(data)
    bin_percentages = (bin_counts / total_count) * 100
    
    return bin_edges, bin_labels, bin_counts, bin_percentages


def create_custom_histogram(data, file_stats, title="Custom Binned Importance Weights Distribution", save_path=None):
    """
    Create and display/save histogram with custom binning and percentage labels.
    
    Args:
        data (np.ndarray): Combined array of values to plot
        file_stats (list): List of file statistics
        title (str): Plot title
        save_path (str): Optional path to save the plot
    """
    # Store original data statistics
    original_min = data.min()
    original_max = data.max()
    original_mean = data.mean()
    original_std = data.std()
    
    # Create custom bins
    bin_edges, bin_labels, bin_counts, bin_percentages = create_custom_bins(data)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Create x positions for bars (using indices)
    x_positions = np.arange(len(bin_labels))
    
    # Create bars
    bars = plt.bar(x_positions, bin_percentages, alpha=0.7, edgecolor='black', 
                   color='steelblue', width=0.8)
    
    # Add percentage labels on top of each bar
    for i, (bar, percentage, count) in enumerate(zip(bars, bin_percentages, bin_counts)):
        if percentage > 0.01:  # Only show label if percentage > 0.01%
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{percentage:.2f}%\n({count})', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Importance Weight Range', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    plt.xticks(x_positions, bin_labels, rotation=45, ha='right')
    
    # # Add statistics text
    # stats_text = f'Total elements: {len(data)}\n'
    # stats_text += f'Min: {original_min:.4f}\n'
    # stats_text += f'Max: {original_max:.4f}\n'
    # stats_text += f'Mean: {original_mean:.4f}\n'
    # stats_text += f'Std: {original_std:.4f}\n'
    # stats_text += f'Files processed: {len(file_stats)}'

    # plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #         fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Custom binned histogram saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== Custom Binned Data Statistics ===")
    print(f"Total files processed: {len(file_stats)}")
    print(f"Total elements: {len(data)}")
    print(f"Combined min value: {original_min:.6f}")
    print(f"Combined max value: {original_max:.6f}")
    print(f"Combined mean value: {original_mean:.6f}")
    print(f"Combined std deviation: {original_std:.6f}")
    
    # Print bin information
    print(f"\n=== Bin Distribution ===")
    print(f"{'Bin Range':<15} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    for i, (label, count, percentage) in enumerate(zip(bin_labels, bin_counts, bin_percentages)):
        print(f"{label:<15} {count:<10} {percentage:<12.4f}%")
    
    # Print percentiles using original data
    print(f"\n=== Percentiles ===")
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"{p:5.1f}%: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Create custom binned histogram from multiple importance weight .bin files')
    parser.add_argument('directory', help='Directory containing the .bin files')
    parser.add_argument('--save', type=str, help='Path to save the histogram plot')
    parser.add_argument('--title', type=str, default='Custom Binned Importance Weights Distribution', 
                       help='Title for the histogram plot')
    
    args = parser.parse_args()
    
    print(f"Processing directory: {args.directory}")
    print(f"Looking for files matching: importance_weights_before_icepop*.bin")
    print(f"Using custom binning: 0-2 in 0.2 intervals, >2 as one bin")
    print("=" * 60)
    
    # Process all files in the directory
    combined_data, file_stats = process_multiple_files(args.directory)
    
    if combined_data is not None:
        print("=" * 60)
        print("Creating custom binned histogram...")
        
        # Create histogram
        create_custom_histogram(combined_data, file_stats, 
                               title=args.title, save_path=args.save)
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed {len(file_stats)} files")
        print(f"Combined {len(combined_data)} elements total")
        
    else:
        print("Failed to process any files.")


if __name__ == "__main__":
    # If run without arguments, show help
    import sys
    if len(sys.argv) == 1:
        print("Usage examples:")
        print(f"  python {sys.argv[0]} /path/to/data/directory")
        print(f"  python {sys.argv[0]} ./log_data_importance_weights --save custom_histogram.png")
        print(f"  python {sys.argv[0]} . --title 'My Custom Binned Distribution'")
        print(f"\nRun 'python {sys.argv[0]} --help' for full help")
    else:
        main()

