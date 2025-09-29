#!/usr/bin/env python3
"""
Script to read importance weights from .bin file, filter leading/trailing zeros,
and create a histogram to visualize the distribution of values.

Based on read_importance_weights.py, this script:
1. Reads binary data from .bin file (assumes float32 format)
2. Filters out leading and trailing zero elements
3. Creates a histogram showing the distribution of remaining values
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def read_and_filter_importance_weights(filename):
    """
    Read importance weights from binary file and filter leading/trailing zeros.
    
    Args:
        filename (str): Path to the binary file
        
    Returns:
        np.ndarray or None: Filtered array with leading/trailing zeros removed
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return None
    
    try:
        # Read the binary data
        with open(filename, "rb") as f:
            data_bytes = f.read()
        
        print(f"File size: {len(data_bytes)} bytes")
        
        # Assume float32 (most common for PyTorch tensors)
        dtype = np.float32
        element_size = np.dtype(dtype).itemsize
        
        if len(data_bytes) % element_size != 0:
            print(f"Warning: File size not divisible by {element_size} bytes (float32 size)")
            return None
        
        # Reconstruct the array
        data_array = np.frombuffer(data_bytes, dtype=dtype)
        
        print(f"Original array shape: {data_array.shape}")
        print(f"Original array dtype: {data_array.dtype}")
        print(f"Total elements: {len(data_array)}")
        
        # Find first and last non-zero indices
        non_zero_indices = np.nonzero(data_array)[0]
        
        if len(non_zero_indices) == 0:
            print("Warning: All elements are zero!")
            return None
        
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        
        # Filter out leading and trailing zeros
        filtered_array = data_array[first_non_zero:last_non_zero + 1]
        
        print(f"\nFiltering results:")
        print(f"Leading zeros removed: {first_non_zero}")
        print(f"Trailing zeros removed: {len(data_array) - last_non_zero - 1}")
        print(f"Filtered array length: {len(filtered_array)}")
        
        # Statistics for filtered array
        zero_count = np.sum(filtered_array == 0)
        non_zero_count = len(filtered_array) - zero_count
        zero_percentage = (zero_count / len(filtered_array)) * 100
        
        print(f"\nFiltered array statistics:")
        print(f"Min value: {filtered_array.min()}")
        print(f"Max value: {filtered_array.max()}")
        print(f"Mean value: {filtered_array.mean()}")
        print(f"Standard deviation: {filtered_array.std()}")
        print(f"Zero elements (internal): {zero_count}")
        print(f"Non-zero elements: {non_zero_count}")
        print(f"Internal zero percentage: {zero_percentage:.2f}%")
        
        return filtered_array
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def create_histogram(data, bins=50, title="Importance Weights Distribution", save_path=None):
    """
    Create and display/save histogram of the data.
    
    Args:
        data (np.ndarray): Array of values to plot
        bins (int): Number of histogram bins
        title (str): Plot title
        save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create histogram and convert to percentages
    counts, bin_edges, patches = plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    
    # Convert counts to percentages
    total_count = len(data)
    percentages = (counts / total_count) * 100
    
    # Clear the plot and redraw with percentages
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', align='edge')
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('importance weight', fontsize=14)
    plt.ylabel('%', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'min: {data.min():.4f}\n'
    stats_text += f'0.1%: {np.percentile(data, 0.1):.4f}\n'
    stats_text += f'1%: {np.percentile(data, 1):.4f}\n'
    stats_text += f'5%: {np.percentile(data, 9):.4f}\n'
    stats_text += f'95%: {np.percentile(data, 95):.4f}\n'
    stats_text += f'99%: {np.percentile(data, 99):.4f}\n'
    stats_text += f'99.9%: {np.percentile(data, 99.9):.4f}\n'
    stats_text += f'max: {data.max():.4f}\n'  
    stats_text += f'mean: {data.mean():.4f}\n'
    stats_text += f'std: {data.std():.4f}\n'  

    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print bin information
    print(f"\n=== 分桶统计信息 (Bin Statistics) ===")
    print(f"分桶数量: {len(percentages)}")
    print(f"最大桶占比: {percentages.max():.4f}%")
    print(f"平均桶占比: {percentages.mean():.4f}%")
    print(f"总数据量: {total_count}")
    
    # Show top 5 bins with highest percentage
    sorted_indices = np.argsort(percentages)[::-1][:5]
    print(f"\n前5个占比最高的桶:")
    for i, idx in enumerate(sorted_indices):
        bin_start = bin_edges[idx]
        bin_end = bin_edges[idx + 1]
        bin_percentage = percentages[idx]
        bin_count = int(counts[idx])
        print(f"  {i+1}. 区间 [{bin_start:.6f}, {bin_end:.6f}): {bin_percentage:.4f}% ({bin_count} 个元素)")


def main():
    parser = argparse.ArgumentParser(description='Create histogram of importance weights from .bin file')
    parser.add_argument('filename', help='Path to the .bin file')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins (default: 50)')
    parser.add_argument('--save', type=str, help='Path to save the histogram plot')
    parser.add_argument('--title', type=str, default='Importance Weights Distribution', 
                       help='Title for the histogram plot')
    
    args = parser.parse_args()
    
    print(f"Reading and processing file: {args.filename}")
    print("=" * 60)
    
    # Read and filter the data
    filtered_data = read_and_filter_importance_weights(args.filename)
    
    if filtered_data is not None:
        print("=" * 60)
        print("Creating histogram...")
        
        # Create histogram
        create_histogram(filtered_data, bins=args.bins, title=args.title, save_path=args.save)
        
        print(f"\n=== 处理完成 (Processing Complete) ===")
        print(f"成功处理了 {len(filtered_data)} 个元素")
        
    else:
        print("Failed to read or process the file.")


if __name__ == "__main__":
    # If run without arguments, show help
    import sys
    if len(sys.argv) == 1:
        print("用法示例 (Usage examples):")
        print(f"  python {sys.argv[0]} importance_weights_before_icepop.bin")
        print(f"  python {sys.argv[0]} data.bin --bins 100 --save histogram.png")
        print(f"  python {sys.argv[0]} data.bin --title '我的数据分布' --bins 30")
        print("\n运行 python {} --help 查看完整帮助".format(sys.argv[0]))
    else:
        main()
