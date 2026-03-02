# Google Colab Setup Guide for CIFAR-10 Training & Verification

## Step 1: Open Google Colab
Go to https://colab.research.google.com and create a new notebook

## Step 2: Enable GPU
- Click **Runtime** > **Change runtime type**
- Select **GPU** (T4)
- Click **Save**

## Step 3: Clone Repository
```python
!git clone https://github.com/your-repo/convexity_is_all_you_need.git
%cd convexity_is_all_you_need
```

## Step 4: Install Dependencies
```python
!pip install -q -r requirements.txt
```

## Step 5: Train Models (CIFAR-10 Cats/Dogs)
```python
!python convexrobust/main/main.py --data=cifar10_catsdogs --train
```

**Expected Time on T4:**
- convex_reg: ~1-2 hours
- abcrown: ~1-2 hours
- Total: ~2-3 hours

## Step 6: Download Results
```python
from google.colab import files
import shutil

# Create zip of results
shutil.make_archive('cifar10_results', 'zip', 'out/cifar10_catsdogs-standard')
files.download('cifar10_results.zip')
```

## Step 7: Run ab-CROWN Verification (Optional, requires ab-CROWN config)
```bash
bash scripts/abcrown/cifar_abcrown.sh
```

## Step 8: Download Plots
```python
# After verification, download all results
shutil.make_archive('cifar10_complete', 'zip', 'out/cifar10_catsdogs-standard')
files.download('cifar10_complete.zip')
```

---

## Key Differences from Your Local Setup
- Colab already has PyTorch with CUDA support **fully configured**
- No compatibility issues with your Blackwell GPU
- Automatic GPU detection, no config needed
- Free & reliable for your use case

## Alternative: If You Want to Continue Locally
You can upload the trained models back to your machine and just use Colab for ab-CROWN verification phase (which is the bottleneck).

---

## Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'convexrobust'"**
```python
import sys
sys.path.insert(0, '/content/convexity_is_all_you_need')
```

**Issue: Out of memory**
- T4 has 15GB VRAM, should be plenty for CIFAR-10
- If needed, reduce batch size in datamodule config

**Issue: Weights/checkpoints not loading**
Make sure the checkpoint paths are correct after git clone
