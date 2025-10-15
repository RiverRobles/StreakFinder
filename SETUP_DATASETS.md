# Setting Up Sample Datasets

## Quick Setup

1. **Generate sample datasets:**
   ```bash
   python create_sample_datasets.py
   ```

2. **Deploy your app** - the datasets will be included automatically!

## What Gets Created

The script creates 4 sample datasets in the `datasets/` folder:

- **`sample1.npy`** - Clear sinusoidal patterns (mostly streaked)
- **`sample2.npy`** - Mixed patterns (some streaked, some not)  
- **`sample3.npy`** - Mostly random data (mostly not streaked)
- **`sample4.npy`** - Complex patterns (multiple frequencies)

## Adding Your Own Datasets

### Method 1: Replace Sample Files
1. Put your `.npy` files in the `datasets/` folder
2. Update the `preloaded_options` dictionary in `streamlit_app.py`
3. Redeploy

### Method 2: Add More Datasets
1. Add your `.npy` files to `datasets/`
2. Add entries to `preloaded_options` in `streamlit_app.py`:
   ```python
   preloaded_options = {
       "Clear Sinusoidal Patterns": "datasets/sample1.npy",
       "Your Dataset Name": "datasets/your_file.npy",
       # ... more datasets
   }
   ```

## Dataset Format

Your `.npy` files should contain a list of 2D arrays, where each array has shape `(N, 2)` representing `[x, y]` coordinates:

```python
import numpy as np

# Example: Create a dataset with 10 samples
samples = []
for i in range(10):
    x = np.linspace(0, 10, 100)
    y = 4.7 + 0.1 * np.sin(x + i*0.5) + np.random.normal(0, 0.01, 100)
    samples.append(np.column_stack([x, y]))

np.save("datasets/my_dataset.npy", samples)
```

## Deployment

When you deploy to Streamlit Cloud, Heroku, etc., make sure to include the `datasets/` folder in your repository. The app will automatically find and load these files.
