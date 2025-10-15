#!/usr/bin/env python3
"""
Script to create sample datasets for the Streaking Tinder app.
Run this to generate sample .npy files that users can choose from.
"""

import numpy as np
import os

def create_sample_datasets():
    """Create sample datasets with different patterns"""
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Sample 1: Clear sinusoidal patterns (streaked)
    print("Creating Sample Dataset 1: Clear Sinusoidal Patterns...")
    samples1 = []
    for i in range(50):
        # Generate sinusoidal data
        x = np.linspace(0, 4*np.pi, 100)
        y = 4.7 + 0.1 * np.sin(x + i*0.1) + np.random.normal(0, 0.01, 100)
        samples1.append(np.column_stack([x, y]))
    
    np.save("datasets/sample1.npy", samples1)
    print(f"✓ Created datasets/sample1.npy with {len(samples1)} samples")
    
    # Sample 2: Mixed patterns (some streaked, some not)
    print("Creating Sample Dataset 2: Mixed Patterns...")
    samples2 = []
    for i in range(50):
        x = np.linspace(0, 4*np.pi, 100)
        if i % 3 == 0:  # Every 3rd sample is sinusoidal
            y = 4.7 + 0.1 * np.sin(x + i*0.2) + np.random.normal(0, 0.01, 100)
        else:  # Random noise
            y = 4.7 + np.random.normal(0, 0.05, 100)
        samples2.append(np.column_stack([x, y]))
    
    np.save("datasets/sample2.npy", samples2)
    print(f"✓ Created datasets/sample2.npy with {len(samples2)} samples")
    
    # Sample 3: Mostly random with occasional patterns
    print("Creating Sample Dataset 3: Mostly Random...")
    samples3 = []
    for i in range(50):
        x = np.linspace(0, 4*np.pi, 100)
        if i % 5 == 0:  # Every 5th sample has some pattern
            y = 4.7 + 0.05 * np.sin(x + i*0.3) + np.random.normal(0, 0.02, 100)
        else:  # Random noise
            y = 4.7 + np.random.normal(0, 0.08, 100)
        samples3.append(np.column_stack([x, y]))
    
    np.save("datasets/sample3.npy", samples3)
    print(f"✓ Created datasets/sample3.npy with {len(samples3)} samples")
    
    # Sample 4: Complex patterns (multiple frequencies)
    print("Creating Sample Dataset 4: Complex Patterns...")
    samples4 = []
    for i in range(50):
        x = np.linspace(0, 4*np.pi, 100)
        if i % 2 == 0:  # Every other sample
            # Complex sinusoidal pattern
            y = 4.7 + 0.08 * np.sin(x + i*0.1) + 0.03 * np.sin(2*x + i*0.2) + np.random.normal(0, 0.01, 100)
        else:  # Random
            y = 4.7 + np.random.normal(0, 0.06, 100)
        samples4.append(np.column_stack([x, y]))
    
    np.save("datasets/sample4.npy", samples4)
    print(f"✓ Created datasets/sample4.npy with {len(samples4)} samples")
    
    print("\n🎉 All sample datasets created successfully!")
    print("You can now update the preloaded_options in streamlit_app.py to include these datasets.")

if __name__ == "__main__":
    create_sample_datasets()
