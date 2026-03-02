#!/usr/bin/env python3
"""
Google Colab Setup for CIFAR-10 Training & Verification
Run this in a Colab cell to set everything up automatically
"""

# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================

import torch
import subprocess
import os

print("=" * 60)
print("GPU CHECK")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "=" * 60)
print("INSTALLING DEPENDENCIES")
print("=" * 60)

# Clone repo if not already cloned
if not os.path.exists('convexity_is_all_you_need'):
    !git clone https://github.com/YOUR_GITHUB_USERNAME/convexity_is_all_you_need.git
    
os.chdir('convexity_is_all_you_need')

# Install requirements
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning tensorboard scikit-learn matplotlib foolbox click dacite numpy

print("✓ Dependencies installed!")

# ============================================================================
# CELL 2: Train CIFAR-10 Models
# ============================================================================

import time
start_time = time.time()

print("=" * 60)
print("STARTING TRAINING: CIFAR-10 CATS/DOGS")
print("=" * 60)

!python convexrobust/main/main.py --data=cifar10_catsdogs --train

elapsed = (time.time() - start_time) / 3600
print(f"\n✓ Training complete! Took {elapsed:.1f} hours")

# ============================================================================
# CELL 3: Download Results
# ============================================================================

from google.colab import files
import shutil
import os

print("=" * 60)
print("PREPARING RESULTS FOR DOWNLOAD")
print("=" * 60)

results_path = 'out/cifar10_catsdogs-standard'

if os.path.exists(results_path):
    # Create zip archive
    shutil.make_archive('cifar10_results', 'zip', results_path)
    
    # Download
    files.download('cifar10_results.zip')
    print(f"✓ Downloaded: cifar10_results.zip ({os.path.getsize('cifar10_results.zip') / 1e6:.1f} MB)")
else:
    print(f"✗ Results not found at {results_path}")

# ============================================================================
# CELL 4: Run ab-CROWN Verification (Optional)
# ============================================================================

print("=" * 60)
print("STARTING AB-CROWN VERIFICATION")
print("=" * 60)

start_time = time.time()

# Run the ab-CROWN script
!bash scripts/abcrown/cifar_abcrown.sh

elapsed = (time.time() - start_time) / 3600
print(f"\n✓ Verification complete! Took {elapsed:.1f} hours")

# ============================================================================
# CELL 5: Download Complete Results with Plots
# ============================================================================

print("=" * 60)
print("GENERATING PLOTS AND DOWNLOADING FINAL RESULTS")
print("=" * 60)

# Generate plots
!python convexrobust/main/plot.py --data=cifar10_catsdogs --labels=cifar10_catsdogs_paper

# Create final archive
shutil.rmtree('cifar10_results')  # Remove old one
shutil.make_archive('cifar10_final_results', 'zip', 'out/cifar10_catsdogs-standard')

files.download('cifar10_final_results.zip')
print(f"✓ Downloaded: cifar10_final_results.zip")

# Also zip the figures
shutil.make_archive('cifar10_plots', 'zip', 'figs/cifar10_catsdogs-standard')
files.download('cifar10_plots.zip')
print(f"✓ Downloaded: cifar10_plots.zip")

print("\n" + "=" * 60)
print("ALL TASKS COMPLETE!")
print("=" * 60)
print("\nYou now have:")
print("  - Trained models (convex_reg & abcrown)")
print("  - ab-CROWN verification results")
print("  - Publication-ready plots (L1, L2, L∞)")
print("\nTotal estimated time: 4-5 hours on T4 GPU")
