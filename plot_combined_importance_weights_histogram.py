#!/usr/bin/env python3
"""
Script to read multiple importance weights .bin files from a folder, filter leading/trailing zeros,
combine all data, and create a single histogram to visualize the overall distribution.

This script:
1. Finds all files matching pattern "importance_weights_before_icepop*.bin" in a directory
2. Reads each binary file (assumes float32 format)
3. Filters out leading and trailing zero elements from each file
4. Combines all filtered data into a single array
5. Creates a histogram showing the distribution of combined values
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
    # for file in files:
    #     print(f"  - {os.path.basename(file)}")
    
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
            
            # print(f"  Original length: {file_stats['original_length']}")
            # print(f"  Filtered length: {file_stats['filtered_length']}")
            # print(f"  Leading zeros removed: {file_stats['leading_zeros_removed']}")
            # print(f"  Trailing zeros removed: {file_stats['trailing_zeros_removed']}")
            # print(f"  Value range: [{file_stats['min_value']:.6f}, {file_stats['max_value']:.6f}]")
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


def create_combined_histogram(data, file_stats, bins=100, title="Combined Importance Weights Distribution", save_path=None):
    """
    Create and display/save histogram of the combined data.
    Values greater than 10 are capped at 10 for plotting and binning,
    but original statistics are preserved.
    
    Args:
        data (np.ndarray): Combined array of values to plot
        file_stats (list): List of file statistics
        bins (int): Number of histogram bins
        title (str): Plot title
        save_path (str): Optional path to save the plot
    """
    # Store original data statistics before capping
    original_min = data.min()
    original_max = data.max()
    original_mean = data.mean()
    original_std = data.std()
    
    # Cap values at 10 for plotting
    capped_data = np.minimum(data, 10.0)
    
    plt.figure(figsize=(14, 10))
    
    # Create histogram with capped data and convert to percentages
    counts, bin_edges, patches = plt.hist(capped_data, bins=bins, alpha=0.7, edgecolor='black')
    
    # Convert counts to percentages
    total_count = len(data)
    percentages = (counts / total_count) * 100
    
    # Clear the plot and redraw with percentages
    plt.clf()
    plt.figure(figsize=(14, 10))
    plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', align='edge')
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('importance weight (capped at 10)', fontsize=14)
    plt.ylabel('%', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)

    # Count how many values were capped
    capped_count = np.sum(data > 10)
    
    # Add statistics text (using original data for accurate statistics)
    stats_text = f'min: {original_min:.4f}\n'
    stats_text += f'0.1%: {np.percentile(data, 0.1):.4f}\n'
    stats_text += f'1%: {np.percentile(data, 1):.4f}\n'
    stats_text += f'5%: {np.percentile(data, 5):.4f}\n'
    stats_text += f'95%: {np.percentile(data, 95):.4f}\n'
    stats_text += f'99%: {np.percentile(data, 99):.4f}\n'
    stats_text += f'99.9%: {np.percentile(data, 99.9):.4f}\n'
    stats_text += f'max: {original_max:.4f}\n'  
    stats_text += f'mean: {original_mean:.4f}\n'
    stats_text += f'std: {original_std:.4f}\n'
    if capped_count > 0:
        stats_text += f'capped (>10): {capped_count} ({capped_count/len(data)*100:.2f}%)'  

    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=18)
    
    # # Add file information text
    # if len(file_stats) <= 10:  # Only show file details if not too many files
    #     file_info = "Processed Files:\n"
    #     for i, stats in enumerate(file_stats[:10], 1):
    #         file_info += f"{i}. {stats['filename']} ({stats['filtered_length']} elements)\n"
    #     if len(file_stats) > 10:
    #         file_info += f"... and {len(file_stats) - 10} more files"
        
    #     plt.text(0.98, 0.98, file_info, transform=plt.gca().transAxes, 
    #              verticalalignment='top', horizontalalignment='right',
    #              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #              fontsize=8)
    # else:
    #     file_info = f"Files Processed: {len(file_stats)} files\n"
    #     total_original = sum(stats['original_length'] for stats in file_stats)
    #     total_filtered = sum(stats['filtered_length'] for stats in file_stats)
    #     file_info += f"Original total: {total_original}\n"
    #     file_info += f"Filtered total: {total_filtered}"
        
    #     plt.text(0.98, 0.98, file_info, transform=plt.gca().transAxes, 
    #              verticalalignment='top', horizontalalignment='right',
    #              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #              fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined histogram saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print detailed statistics (using original data)
    print(f"\n=== Combined Data Statistics ===")
    print(f"Total files processed: {len(file_stats)}")
    print(f"Total elements: {len(data)}")
    print(f"Combined min value: {original_min:.6f}")
    print(f"Combined max value: {original_max:.6f}")
    print(f"Combined mean value: {original_mean:.6f}")
    print(f"Combined std deviation: {original_std:.6f}")
    if capped_count > 0:
        print(f"Values capped at 10: {capped_count} ({capped_count/len(data)*100:.2f}%)")
    
    # Print bin information
    print(f"\n=== Histogram Bin Statistics ===")
    print(f"Number of bins: {len(percentages)}")
    print(f"Max bin percentage: {percentages.max():.4f}%")
    print(f"Average bin percentage: {percentages.mean():.4f}%")
    print(f"Total data count: {total_count}")
    
    # Show top 5 bins with highest percentage
    sorted_indices = np.argsort(percentages)[::-1][:5]
    print(f"\nTop 5 bins with highest percentage:")
    for i, idx in enumerate(sorted_indices):
        bin_start = bin_edges[idx]
        bin_end = bin_edges[idx + 1]
        bin_percentage = percentages[idx]
        bin_count = int(counts[idx])
        print(f"  {i+1}. Range [{bin_start:.6f}, {bin_end:.6f}): {bin_percentage:.4f}% ({bin_count} elements)")
    
    # # Print file-by-file summary
    # print(f"\n=== File-by-File Summary ===")
    # for i, stats in enumerate(file_stats, 1):
    #     print(f"{i:2d}. {stats['filename']}")
    #     print(f"     Original: {stats['original_length']:8d} -> Filtered: {stats['filtered_length']:8d}")
    #     print(f"     Range: [{stats['min_value']:8.6f}, {stats['max_value']:8.6f}]")


def main():
    parser = argparse.ArgumentParser(description='Create combined histogram from multiple importance weight .bin files')
    parser.add_argument('directory', help='Directory containing the .bin files')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins (default: 100)')
    parser.add_argument('--save', type=str, help='Path to save the histogram plot')
    parser.add_argument('--title', type=str, default='Combined Importance Weights Distribution', 
                       help='Title for the histogram plot')
    
    args = parser.parse_args()
    
    print(f"Processing directory: {args.directory}")
    print(f"Looking for files matching: importance_weights_before_icepop*.bin")
    print("=" * 60)
    
    # Process all files in the directory
    combined_data, file_stats = process_multiple_files(args.directory)
    
    if combined_data is not None:
        print("=" * 60)
        print("Creating combined histogram...")
        
        # Create histogram
        create_combined_histogram(combined_data, file_stats, bins=args.bins, 
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
        print(f"  python {sys.argv[0]} ./log_data_importance_weights --bins 200 --save combined_histogram.png")
        print(f"  python {sys.argv[0]} . --title 'My Combined Data Distribution'")
        print(f"\nRun 'python {sys.argv[0]} --help' for full help")
    else:
        main()
