#!/usr/bin/env python3
"""
Script to read and print values from importance_weights_before_icepop.bin file.

This binary file contains numpy array data written with .tobytes() method.
The original data is a PyTorch tensor converted to numpy array with .cpu().numpy().
"""

import numpy as np
import os


def read_importance_weights(filename="importance_weights_before_icepop.bin"):
    """
    Read importance weights from binary file and print all values.
    
    Args:
        filename (str): Path to the binary file
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return
    
    try:
        # Read the binary data
        with open(filename, "rb") as f:
            data_bytes = f.read()
        
        # Since the original data was written as .tobytes() from a numpy array,
        # we need to reconstruct the array. We'll try float32 first (most common for PyTorch)
        
        print(f"File size: {len(data_bytes)} bytes")
        
        # Try different data types and see what makes sense
        for dtype in [np.float32, np.float64, np.float16]:
            try:
                # Calculate how many elements we have for this dtype
                element_size = np.dtype(dtype).itemsize
                if len(data_bytes) % element_size == 0:
                    num_elements = len(data_bytes) // element_size
                    
                    # Reconstruct the array
                    data_array = np.frombuffer(data_bytes, dtype=dtype)
                    
                    print(f"\n--- Trying dtype: {dtype} ---")
                    print(f"Number of elements: {num_elements}")
                    print(f"Array shape: {data_array.shape}")
                    print(f"Array dtype: {data_array.dtype}")
                    print(f"Min value: {data_array.min()}")
                    print(f"Max value: {data_array.max()}")
                    print(f"Mean value: {data_array.mean()}")
                    print(f"Standard deviation: {data_array.std()}")
                    
                    # Count zero elements
                    zero_count = np.sum(data_array == 0)
                    non_zero_count = len(data_array) - zero_count
                    zero_percentage = (zero_count / len(data_array)) * 100
                    
                    print(f"Zero elements: {zero_count}")
                    print(f"Non-zero elements: {non_zero_count}")
                    print(f"Zero percentage: {zero_percentage:.2f}%")
                    
                    print(f"\nFirst 20 values:")
                    print(data_array[:20])
                    
                    if len(data_array) > 20:
                        print(f"\nLast 20 values:")
                        print(data_array[-20:])
                    
                    print(f"\nAll values:")
                    print(data_array[-1000:])
                    
                    # float32 is most likely the correct type for PyTorch tensors
                    if dtype == np.float32:
                        print(f"\n=== MOST LIKELY CORRECT INTERPRETATION (float32) ===")
                        return data_array
                        
            except Exception as e:
                print(f"Failed to read as {dtype}: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


if __name__ == "__main__":
    print("Reading importance_weights_before_icepop.bin file...\n")
    data = read_importance_weights("log_data_importance_weights/importance_weights_before_icepop_957710.bin")
    
    if data is not None:
        print(f"\n=== SUMMARY ===")
        print(f"Successfully read {len(data)} values from the file")
        print(f"Data type: {data.dtype}")
        print(f"Shape: {data.shape}")
        
        # Final zero count summary
        zero_count = np.sum(data == 0)
        non_zero_count = len(data) - zero_count
        zero_percentage = (zero_count / len(data)) * 100
        
        print(f"\n=== ZERO ELEMENT STATISTICS ===")
        print(f"Total elements: {len(data)}")
        print(f"Zero elements: {zero_count}")
        print(f"Non-zero elements: {non_zero_count}")
        print(f"Zero percentage: {zero_percentage:.2f}%")
