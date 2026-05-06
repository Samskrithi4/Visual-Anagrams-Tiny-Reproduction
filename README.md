# Visual Anagrams – Track 1 Tiny Reproduction
**ECE 570 - Spring 2026**

Tiny reproduction of:
    *Visual Anagrams: Generating Multi-View Optical Illusions* (CVPR 2024):
    https://dangeng.github.io/visual_anagrams/

---

## What This Project Does

The original paper generates images that change appearance under geometric transformations (e.g., rotating 180 degrees reveals a different scene) using GPU-heavy diffusion models specifically DeepFloyd IF, a pixel-based diffusion model.

This project **simplifies the core mathematical claim**. It averages noise estimates from orthogonally-transformed views and forces it into a single pixel tensor to simultaneously resolve into two distinct targets. This acts as our parallel gradient-descent optimizer while only being on a CPU. Therefore, no GPU or generative model is required.

The key algorithm is compared against a **naive pixel-average baseline** with quantitative metrics (per-view MSE and SSIM) along with visualizations. 

---

## Code Authorship
Below are the main functions in my python file. None of my code uses code from the original paper as the original paper uses diffusion models while mine dont. I did have to look at how to code the mathematical formulas especially in get_laplacian_pyramid() and get_sobel_edges() to make sure the math I coded was similar to the theoretical math.

| File | Status | Notes |
|------|--------|-------|
| `final.py` | Written by me | All code with some inspirations from Stack Overflow, Python documentation, and examples of mathematical formulas |
| `get_laplacian_pyramid()` | Written mostly by me | Needed to look up the formulas and how to code it in Python, had inspiration |
| `get_sobel_edges()` | Written mostly by me | Needed to look up the formulas and how to code it in Python, had inspiration |
| `compute_loss()` | Written by me | Compute loss to compare our models |
| `naive_baseline()` | Written by me | Compare naive with the tiny reproduction |
| `evaluate()` | Written by me | Evaluates and visualizes our metrics |
| `run_all_experiments()` | Written by me | Runs the experiments |

---

## Dependencies

```bash
pip install torch torchvision Pillow matplotlib numpy scikit-image
```

My python version was 3.12.4.  No GPU required as it is tested on CPU.

---

## Project Structure

```
.
|--- final.py            # Main script
|--- img1.jpg                           # Target image A (pair #1)        
|--- img2.jpg                           # Target image B (pair #2)
|--- giraffe.jpg                        # Target image A  (pair #2)
|--- penguin.jpg                        # Target image B  (pair #2)
|--- README.md
|--- results/                           # Created automatically on first run
    |--- giraffe_penguin/               # One image pair. Includes all 3 transformations, loss curves, and comparisons
        |--- hflip_comparison.png 
        |--- hflip_loss_curves.png 
        |--- hflip_normal.png 
        |--- hflip_transformed.png 
        |--- rot180_comparison.png 
        |--- rot180_loss_curves.png 
        |--- rot180_normal.png
        |--- rot180_transformed.png 
        |--- vflip_comparison.png 
        |--- vflip_loss_curves.png 
        |--- flip_normal.png 
        |--- vflip_transformed.png
    |--- img1_img2                      # Second image pair. Includes all 3 transformations, loss curves, and comparisons
        |--- hflip_comparison.png 
        |--- hflip_loss_curves.png 
        |--- hflip_normal.png 
        |--- hflip_transformed.png 
        |--- rot180_comparison.png 
        |--- rot180_loss_curves.png 
        |--- rot180_normal.png 
        |--- rot180_transformed.png 
        |--- vflip_comparison.png 
        |--- vflip_loss_curves.png 
        |--- vflip_normal.png 
        |--- vflip_transformed.png
    |--- all_data.json                  # Data from all the pairs in a json format
    |--- summary_metrics.png            # Table of all the metrics
```

---

## How to Run
To get the same results I did, all you have to do is make sure you have all dependencies, and run final.py. The default arguments will automatically generate 3 geometrical transformations for 2 image-pairs.

### Default run
```bash
python final.py
```

You can also change the number of steps
```bash
python final.py --steps 500
```

You can also change the views, for example if you only want the 180 degree rotation and no flips, it would look like
```bash
python final.py --views rot180
```

If you want to change the output directory to results_new
```bash
python final.py --out_dir results_new
```



### Full Arguments Breakdown for running code

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `1000` | Optimisation steps |
| `--views` | `rot180` | One or more of: `rot180 vflip hflip identity` |
| `--out_dir` | `results` | Output directory |

---

## Datasets / Models

No external dataset or pretrained model is needed.  You could add more image-pairs. I have included 2 image-pairs in my submission.
---

## Outputs Explained

| File | Purpose |
|------|---------|
| `*_comparison.png` | Side-by-side: original images / tiny reproduction / naive baseline |
| `*_loss_curves.png` | Total loss + per-view MSE convergence |
| `*_normal.png` | Optimised anagram in its upright orientation |
| `*_transformed.png` | Same anagram under the chosen transform |
| `metrics_table.png` | Summary table |
| `metrics.json` | Raw numbers for future work |

---
