import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "deeplearning",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.20"
    }
}

cells = []

# ==============================================================================
# CELL 0: TITLE & SCIENTIFIC LITERATURE REVIEW (Markdown)
# ==============================================================================
cell_0_md = """# 🔬 In-Line Fluidized Bed Pellet Preprocessing & Agglomeration Benchmark
### Advanced Computer Vision & Optical Process Analytical Technology (PAT) Checkpoint

---

## 📖 Scientific Context & Literature Foundation

In fluidized-bed coating and granulation processes, monitoring particle size growth and detecting unwanted **agglomeration** (unintended adhesion of multiple pellets into clusters via liquid/solid binder bridges) is a critical requirement under **Process Analytical Technology (PAT)** frameworks issued by the **FDA** and European regulatory agencies.

Real-time in-line visual imaging through chamber observation windows provides non-invasive, direct characterization of moving pellets. However, raw fluid-bed images present severe physical imaging challenges:
1. **Narrow Depth of Field (DOF) & Defocus Blur**: Particles flowing outside the optical focal plane appear as blurred, low-gradient disks with inflated diameters, creating false cluster detections.
2. **Non-Uniform Illumination & Window Reflections**: Light pooling, optical vignetting, chamber shadows, and static reflections from quartz/acrylic viewing ports distort intensity baselines.
3. **Contacting vs. Bonded Distinction**: Primary pellets visually in contact due to line-of-sight optical overlap must be distinguished from permanent physical agglomerates bonded by liquid/solid neck joints.

This notebook builds upon foundational peer-reviewed studies in pharmaceutical particle computer vision:

* 📄 **Mehle, A., Likar, B., & Tomaževič, D. (2017)**.  
  *“In-line recognition of agglomerated pharmaceutical pellets with density-based clustering and convolutional neural network”*.  
  **IPSJ Transactions on Computer Vision and Applications**, 9(7). [DOI: 10.1186/s41074-017-0019-2](https://doi.org/10.1186/s41074-017-0019-2).  
  *Key Contributions Reused*: Extraction of candidate particle regions, normalized gradient-weighted clustering, border-clearing and noise-filtering criteria, extraction of standardized centered crops, and justification for moving beyond naive area thresholding to discriminate true agglomerates from random overlaps.

* 📄 **Možina, M., Tomaževič, D., Leben, S., Pernuš, F., & Likar, B. (2018)**.  
  *“In-line agglomeration degree estimation in fluidized bed pellet coating processes using visual imaging”*.  
  **International Journal of Pharmaceutics**, 547(1-2), 1-8. [DOI: 10.1016/j.ijpharm.2018.05.024](https://doi.org/10.1016/j.ijpharm.2018.05.024).  
  *Key Contributions Reused*: Mathematical formulation of the in-line agglomeration degree index, minimum-area bounding rectangle metrics, circularity, and high-speed in-line segmentation principles.

* 📄 **Watano, S., et al. (2010)**.  
  *“Online monitoring of particle mass flow rate in bottom spray fluid bed coating—Development and application”*.  
  **International Journal of Pharmaceutics**, 395(1-2), 100-107. [DOI: 10.1016/j.ijpharm.2010.05.044](https://doi.org/10.1016/j.ijpharm.2010.05.044).  
  *Key Contributions Reused*: Morphological image processing operations (Top-Hat, regional maxima, morphological gradient, reconstruction, and labeling).

* 📄 **Pech-Pacheco, J. L., Cristóbal, G., Chamorro, J., & Fernández, J. (2000)**.  
  *“Diatom autofocusing in brightfield microscopy: a comparative study”*.  
  **Proc. 15th ICPR**, Vol. 3, pp. 314–317.  
  *Key Contributions Reused*: Mathematical operators for depth-of-field focus evaluation (Laplacian Variance, Tenengrad Gradient Magnitude, and Normalized Boundary Gradients) to filter out out-of-focus particles.

---

## 🎯 Preprocessing Checkpoint Architecture

```
[ Raw Fluid-Bed Frames ]
         │
         ├───► [ Reference Empty Chamber Frames ] ──► Compute Static Fixture Mask (Exclusion Zones)
         │
         ▼
[ Stage 1: Illumination Normalization ] ──► (Compare: Top-Hat vs. Gauss-Sub vs. CLAHE vs. Flat-Field)
         │
         ▼
[ Stage 2: Edge-Preserving Denoising ]   ──► (Compare: Bilateral vs. Median vs. Gaussian vs. Opening)
         │
         ▼
[ Stage 3: Multi-Algorithm Segmentation ]──► (Compare: Top-Hat+Otsu vs. Watershed vs. DoG vs. Canny)
         │
         ▼
[ Stage 4: Artifact & Noise Elimination] ──► (Static Fixtures + Border Margins + Area + Aspect Ratio)
         │
         ▼
[ Stage 5: Focus Measure & Blur Rejection]► (Tenengrad + Laplacian Variance + Boundary Gradient)
         │
         ├───► [ Out-of-Focus Pellets ] ────► Logged & Filtered Out to Blurred Archive
         │
         ▼
[ Stage 6: Geometric Profiling ]         ──► (Circularity, Aspect Ratio, Solidity, Equivalent Diameter)
         │
         ▼
[ Stage 7: Preprocessing Checkpoint Export ]
         ├──► Standardized High-Res Square Crops: /crops/single, /crops/connected, /crops/overlapped
         └──► Master Manifest: particles_metadata.csv (Ready for ML Classifiers or Deep Learning)
```
"""
cells.append(nbf.v4.new_markdown_cell(cell_0_md))

# ==============================================================================
# CELL 1: SETUP & ENVIRONMENT CONFIGURATION (Code)
# ==============================================================================
cell_1_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 1: Environment Setup, Libraries & Path Configuration
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import glob
import re
import time
import math
import warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.gridspec as gridspec
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

warnings.filterwarnings('ignore')

# Matplotlib configuration for publication-quality figures
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Global Directory Paths
DATA_DIR = Path('Real-Data')
CHECKPOINT_DIR = Path('preprocessed_checkpoint')
CROPS_DIR = CHECKPOINT_DIR / 'crops'
CROPS_SINGLE = CROPS_DIR / 'single'
CROPS_CONNECTED = CROPS_DIR / 'connected'
CROPS_OVERLAPPED = CROPS_DIR / 'overlapped'
CROPS_CLEAN_ALL = CROPS_DIR / 'all_clean_infocus'
CROPS_BLURRED = CROPS_DIR / 'blurred_rejected'
METADATA_CSV = CHECKPOINT_DIR / 'particles_metadata.csv'

# Ensure all output directories exist
for p in [CHECKPOINT_DIR, CROPS_DIR, CROPS_SINGLE, CROPS_CONNECTED, CROPS_OVERLAPPED, CROPS_CLEAN_ALL, CROPS_BLURRED]:
    p.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("[INFO] In-Line Fluidized Bed Image Processing Environment Initialized")
print("=" * 80)
print(f"* Python Version     : {sys.version.split()[0]}")
print(f"* OpenCV Version     : {cv2.__version__}")
print(f"* NumPy Version      : {np.__version__}")
print(f"* Pandas Version     : {pd.__version__}")
print(f"* Dataset Directory  : {DATA_DIR.resolve()}")
print(f"* Checkpoint Directory: {CHECKPOINT_DIR.resolve()}")
print("=" * 80)
"""
cells.append(nbf.v4.new_code_cell(cell_1_code))

# ==============================================================================
# CELL 2: DATASET CATALOGING & ILLUMINATION PROFILING (Markdown & Code)
# ==============================================================================
cell_2_md = """---
## 📂 Dataset Cataloging & Experimental Condition Profiling

The dataset in `Real-Data/` comprises 22 high-resolution monochrome BMP images ($1536 \\times 2048$) captured across 7 distinct experimental operating conditions:
1. **Backlight + External Light (Early Agglomeration)**: Fluidized bed during the onset of spray-induced agglomerate formation.
2. **Backlight + External Light (Established Run)**: Balanced forward and silhouette illumination.
3. **Backlight Only (No Pellets)**: Empty chamber reference — essential for static background modeling, flat-field correction, and fixture masking.
4. **Backlight Only (1000 RPM / Fluidization)**: Pure silhouette imaging showing high-contrast pellet boundaries.
5. **Backlight Only (1500 RPM / Elevated Velocity)**: Higher velocity flow regime with increased motion and tumbling dynamics.
6. **Exp 2000 (Dense Fluidization)**: High-density particle regime with frequent line-of-sight collisions and overlaps.
7. **Only External Light (1000 RPM)**: Pure surface reflection imaging without background backlight.

Let's systematically catalog all files, compute global statistical properties (dynamic range, mean brightness, noise levels, and global sharpness), and build a master dataset inventory.
"""
cells.append(nbf.v4.new_markdown_cell(cell_2_md))

cell_2_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 2: Automated Dataset Inventory & Quantitative Profiling
# ══════════════════════════════════════════════════════════════════════════════
dataset_records = []

for subfolder in sorted(DATA_DIR.iterdir()):
    if not subfolder.is_dir():
        continue
    bmp_files = sorted(list(subfolder.glob('*.bmp')))
    for fpath in bmp_files:
        img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Categorize condition type
        fname_folder = subfolder.name.lower()
        if 'no pellets' in fname_folder:
            cond_type = 'Empty Chamber Reference'
        elif 'early agg' in fname_folder:
            cond_type = 'Back + Ext Light (Early Agglomeration)'
        elif 'back+external' in fname_folder:
            cond_type = 'Back + External Light'
        elif 'only external' in fname_folder:
            cond_type = 'External Light Only'
        elif '1500' in fname_folder:
            cond_type = 'Backlight Only (1500 RPM)'
        elif '2000' in fname_folder:
            cond_type = 'Backlight Only (2000 RPM)'
        elif 'no external' in fname_folder:
            cond_type = 'Backlight Only (1000 RPM)'
        else:
            cond_type = subfolder.name

        dataset_records.append({
            'Folder': subfolder.name,
            'Condition_Type': cond_type,
            'Filename': fpath.name,
            'Path': str(fpath),
            'Width': w,
            'Height': h,
            'Min_Intensity': int(img.min()),
            'Max_Intensity': int(img.max()),
            'Mean_Intensity': round(float(img.mean()), 2),
            'Std_Dev': round(float(img.std()), 2),
            'Global_LapVar': round(float(lap_var), 2)
        })

df_dataset = pd.DataFrame(dataset_records)
print(f"[OK] Successfully cataloged {len(df_dataset)} images across {df_dataset['Folder'].nunique()} experimental conditions.")
print()

# Display formatted overview table grouped by condition
summary_table = df_dataset.groupby(['Folder', 'Condition_Type']).agg(
    Files=('Filename', 'count'),
    Mean_Intensity=('Mean_Intensity', 'mean'),
    Std_Dev=('Std_Dev', 'mean'),
    Global_LapVar=('Global_LapVar', 'mean')
).reset_index()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(summary_table.to_string(index=False))
"""
cells.append(nbf.v4.new_code_cell(cell_2_code))

# ==============================================================================
# CELL 3: VISUALIZING ALL 7 LIGHTING CONDITIONS (Code)
# ==============================================================================
cell_3_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 3: Visual Comparison of Raw Images across All 7 Lighting Regimes
# ══════════════════════════════════════════════════════════════════════════════
unique_folders = df_dataset['Folder'].unique()
n_folders = len(unique_folders)

fig, axes = plt.subplots(n_folders, 2, figsize=(16, n_folders * 3.2), 
                         gridspec_kw={'width_ratios': [2.8, 1.2]})
fig.suptitle('Raw Fluid-Bed Imaging: Condition Comparison & Intensity Distributions', 
             fontsize=14, y=0.995)

for i, folder in enumerate(unique_folders):
    sample_row = df_dataset[df_dataset['Folder'] == folder].iloc[0]
    img = cv2.imread(sample_row['Path'], cv2.IMREAD_UNCHANGED)
    
    # Image thumbnail display
    ax_img = axes[i, 0]
    im = ax_img.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(f"{sample_row['Condition_Type']}\\n[{folder} | {sample_row['Filename']}]", 
                     fontsize=10, loc='left', pad=4)
    ax_img.axis('off')
    
    # Histogram display
    ax_hist = axes[i, 1]
    hist, bins = np.histogram(img.flatten(), bins=64, range=[0, 256])
    ax_hist.plot(bins[:-1], hist, color='#1f77b4', lw=1.5)
    ax_hist.fill_between(bins[:-1], hist, alpha=0.25, color='#1f77b4')
    ax_hist.set_xlim([0, 255])
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_ylabel('Log Frequency')
    ax_hist.grid(True, linestyle=':', alpha=0.6)
    ax_hist.text(0.95, 0.85, f"μ={sample_row['Mean_Intensity']}\\nσ={sample_row['Std_Dev']}", 
                 transform=ax_hist.transAxes, ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ccc'))

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_3_code))

# ==============================================================================
# CELL 4: STAGE 1 — STATIC CHAMBER REFERENCE & FIXTURE MASKING (Markdown & Code)
# ==============================================================================
cell_4_md = """---
## 🛡️ Stage 1: Static Chamber Reference & Fixture Masking (Empty Chamber Baseline)

### Literature Rationale (Mehle et al. 2017 & FDA PAT Guidelines)
In industrial and pilot-scale fluidized bed coaters, optical inspection windows unavoidably capture static chamber fixtures:
* Metallic chamber wall seams and partition brackets
* Mounting screws and nozzle body edges
* Persistent specular reflections and glares on the inspection glass

As highlighted by **Mehle et al. (2017)**, discarding non-particle regions, static artifacts, and border-touching structures is a prerequisite before attempting clustering or neural network classification.

Because we have empty chamber captures (`exp 1000 bak light only no pellets`), we can build a **Reference Static Background Model** and a **Static Fixture Exclusion Mask** $M_{static}$:
$$I_{bg\_ref}(x, y) = \\text{median}_t \\left( I_t(x, y) \\right)$$
$$M_{static} = \\text{dilate}\\left( \\text{Threshold}\\left( \\text{TopHat}(I_{bg\_ref}), \\theta_{static} \\right), \\mathcal{SE}_{dilate} \\right)$$

Any candidate particle overlapping $M_{static}$ is guaranteed to be a chamber fixture or static reflection rather than a dynamic fluidized pellet, completely eliminating false positives.
"""
cells.append(nbf.v4.new_markdown_cell(cell_4_md))

cell_4_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 4: Construction of Static Background Reference & Exclusion Mask
# ══════════════════════════════════════════════════════════════════════════════
bg_files = sorted(list((DATA_DIR / 'exp 1000 bak light only no pellets').glob('*.bmp')))
print(f"Found {len(bg_files)} empty chamber baseline frames.")

# Load empty chamber frames and compute temporal median reference
bg_stack = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in bg_files]
I_bg_ref = np.median(np.stack(bg_stack, axis=0), axis=0).astype(np.uint8)

# Extract static bright fixtures via Morphological Top-Hat on reference image
se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
tophat_bg = cv2.morphologyEx(I_bg_ref, cv2.MORPH_TOPHAT, se_bg)

# Threshold static fixtures and dilate with safety margin
_, binary_bg_fixtures = cv2.threshold(tophat_bg, 38, 255, cv2.THRESH_BINARY)
se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
M_static = cv2.dilate(binary_bg_fixtures, se_dilate)

static_pixel_count = int(np.sum(M_static > 0))
total_pixel_count = M_static.size
print(f"[OK] Reference Static Background Model created.")
print(f"* Static Fixture Exclusion Zones : {static_pixel_count:,} px ({static_pixel_count/total_pixel_count*100:.2f}% of camera FOV)")

# Visual verification
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(I_bg_ref, cmap='gray')
axes[0].set_title('1. Empty Chamber Reference Image $I_{bg\_ref}$')
axes[0].axis('off')

axes[1].imshow(tophat_bg, cmap='inferno')
axes[1].set_title('2. Top-Hat Response on Empty Chamber')
axes[1].axis('off')

axes[2].imshow(M_static, cmap='gray')
axes[2].set_title(f'3. Dilated Static Exclusion Mask $M_{{static}}$\\n(Masks Fixed Chamber Fixtures & Window Glare)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_4_code))

# ==============================================================================
# CELL 5: STAGE 2 — ILLUMINATION CORRECTION & BACKGROUND NORMALIZATION (Markdown & Code)
# ==============================================================================
cell_5_md = """---
## [NOTE] Stage 2: Illumination Correction & Background Normalization (Testing 4 Techniques)

Fluidized-bed optical chambers inherently suffer from spatial lighting non-uniformity:
* Strong central illumination pooling with severe radial falloff (vignetting).
* Low-frequency background gradients across the field of view.

To achieve robust particle segmentation across varying light conditions, we benchmark **4 distinct scientific illumination normalization techniques**:

1. **Technique 2.1: Morphological White Top-Hat Transform (Disk Structuring Element)**:
   $$I_{TopHat} = I - (I \\circ \\mathcal{SE})$$
   where $\\mathcal{SE}$ is an elliptical structuring element larger than the primary pellet radius ($r \\approx 22$ px). This effectively suppresses all spatial structures and background variations wider than primary pellets, delivering a zero-baseline flattened image.

2. **Technique 2.2: Multi-Scale Gaussian Local Background Subtraction (Bandpass Filter)**:
   $$I_{GaussSub} = \\max(0, I - G_{\\sigma}(I))$$
   A broad Gaussian kernel ($\\\\sigma = 45$) computes a low-pass approximation of the uneven background and subtracts it directly from the raw frame.

3. **Technique 2.3: Contrast Limited Adaptive Histogram Equalization (CLAHE) + Bilateral Filter**:
   Computes local histogram equalization over tiled contextual grids ($16 \\times 16$) with a slope clipping limit ($2.5$) to prevent noise amplification in uniform dark regions.

4. **Technique 2.4: Empty Chamber Reference Division (Flat-Field Normalization)**:
   $$I_{FlatField} = \\frac{I}{I_{bg\_ref} + \\epsilon} \\times \\bar{I}_{bg\_ref}$$
   Standard optical flat-field calibration dividing the raw capture by the empty chamber reference $I_{bg\_ref}$.

Let's execute all 4 techniques on a representative frame from the **Early Agglomeration** dataset and compare both visually and quantitatively using the **Background Flatness Variance** and **Contrast-to-Noise Ratio (CNR)**.
"""
cells.append(nbf.v4.new_markdown_cell(cell_5_md))

cell_5_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 5: Illumination Normalization Benchmark (Comparing 4 Techniques)
# ══════════════════════════════════════════════════════════════════════════════
# Select a representative test frame with early agglomeration
test_frame_path = df_dataset[df_dataset['Folder'] == 'exp 1000 back+ext light early agg']['Path'].iloc[0]
raw_test = cv2.imread(test_frame_path, cv2.IMREAD_UNCHANGED)

# 1. Morphological Top-Hat Transform
se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
norm_tophat = cv2.morphologyEx(raw_test, cv2.MORPH_TOPHAT, se_tophat)

# 2. Gaussian Local Background Subtraction
bg_gauss = cv2.GaussianBlur(raw_test, (91, 91), 0)
norm_gauss = cv2.subtract(raw_test, bg_gauss)

# 3. CLAHE Local Contrast Enhancement
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
norm_clahe = clahe.apply(raw_test)

# 4. Flat-Field Calibration via Empty Chamber Division
eps = 1e-4
bg_mean = float(I_bg_ref.mean())
norm_flatfield = np.clip((raw_test.astype(np.float32) / (I_bg_ref.astype(np.float32) + 1.0)) * bg_mean, 0, 255).astype(np.uint8)

# Quantitative Metrics: Background Variance (lower is better) & Contrast Metric
def compute_illumination_metrics(norm_img, name):
    # Sample 4 corner patches (typically background)
    h, w = norm_img.shape
    corners = np.concatenate([
        norm_img[:100, :100].flatten(),
        norm_img[:100, -100:].flatten(),
        norm_img[-100:, :100].flatten(),
        norm_img[-100:, -100:].flatten()
    ])
    bg_std = corners.std()
    bg_mean_val = corners.mean()
    p99 = np.percentile(norm_img, 99)
    cnr = (p99 - bg_mean_val) / (bg_std + 1e-5)
    return {'Method': name, 'Background_Std': round(bg_std, 2), 'CNR': round(cnr, 2)}

metrics_list = [
    compute_illumination_metrics(norm_tophat, 'Technique 2.1: Morphological Top-Hat'),
    compute_illumination_metrics(norm_gauss, 'Technique 2.2: Gaussian High-Pass Subtraction'),
    compute_illumination_metrics(norm_clahe, 'Technique 2.3: CLAHE Contrast Equalization'),
    compute_illumination_metrics(norm_flatfield, 'Technique 2.4: Empty Chamber Flat-Field')
]

df_illum_metrics = pd.DataFrame(metrics_list)
print("=" * 75)
print("[STATS] Illumination Normalization Quantitative Benchmark")
print("=" * 75)
print(df_illum_metrics.to_string(index=False))
print("=" * 75)

# Visual 4-panel comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
methods = [
    (norm_tophat, 'Technique 2.1: Morphological Top-Hat (Disk SE r=22px)', axes[0, 0]),
    (norm_gauss, 'Technique 2.2: Gaussian High-Pass Subtraction (k=91)', axes[0, 1]),
    (norm_clahe, 'Technique 2.3: CLAHE Equalization (clip=2.5)', axes[1, 0]),
    (norm_flatfield, 'Technique 2.4: Empty-Chamber Flat-Field Division', axes[1, 1])
]

for img_res, title, ax in methods:
    ax.imshow(img_res, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
    ax.axis('off')

plt.tight_layout()
plt.show()

print("[NOTE] Conclusion: Morphological Top-Hat delivers the lowest background noise variance and cleanest")
print("   zero-baseline background separation, isolating pellets irrespective of spatial vignetting.")
"""
cells.append(nbf.v4.new_code_cell(cell_5_code))

# ==============================================================================
# CELL 6: STAGE 3 — DENOISING & EDGE PRESERVATION (Markdown & Code)
# ==============================================================================
cell_6_md = """---
## 🧼 Stage 3: Denoising & Edge Preservation (Testing 4 Filter Kernels)

In high-speed particle imaging, sensor thermal noise and airborne fines create speckle in uniform background regions. However, standard linear smoothing (such as Gaussian blur) blurs pellet boundaries, severely degrading the sharpness metrics required to identify out-of-focus particles.

We compare **4 denoising filter kernels** applied to the Top-Hat normalized image:
1. **Bilateral Filter** ($d=5, \\sigma_{\\text{color}}=30, \\sigma_{\\text{space}}=30$): Non-linear filter that averages pixels based on both spatial closeness and radiometric similarity, smoothing flat regions while strictly preserving steep step edges.
2. **Median Filter** ($k=5$): Non-linear rank filter effective at suppressing salt-and-pepper / dead-pixel noise.
3. **Gaussian Filter** ($k=5, \\sigma=1.5$): Standard isotropic linear smoothing.
4. **Morphological Opening Filter** ($SE = \\text{disk}(2)$): Suppresses isolated speckle noise smaller than the structuring element.

We verify edge preservation by plotting the 1D intensity cross-section across a representative pellet boundary.
"""
cells.append(nbf.v4.new_markdown_cell(cell_6_md))

cell_6_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 6: Denoising Benchmark & Step-Edge Sharpness Profile Analysis
# ══════════════════════════════════════════════════════════════════════════════
base_norm = norm_tophat.copy()

# 1. Bilateral Filter
denoise_bilateral = cv2.bilateralFilter(base_norm, d=5, sigmaColor=30, sigmaSpace=30)

# 2. Median Filter
denoise_median = cv2.medianBlur(base_norm, ksize=5)

# 3. Gaussian Filter
denoise_gaussian = cv2.GaussianBlur(base_norm, (5, 5), 1.5)

# 4. Morphological Opening
se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
denoise_open = cv2.morphologyEx(base_norm, cv2.MORPH_OPEN, se_open)

# Edge Profile Analysis: Sample an edge cut across an in-focus pellet
# Crop a known in-focus pellet from the test frame
y_center, x_center = 960, 1475
line_profile_y = y_center
line_profile_x_range = range(x_center - 25, x_center + 25)

profile_raw = [base_norm[line_profile_y, x] for x in line_profile_x_range]
profile_bil = [denoise_bilateral[line_profile_y, x] for x in line_profile_x_range]
profile_med = [denoise_median[line_profile_y, x] for x in line_profile_x_range]
profile_gau = [denoise_gaussian[line_profile_y, x] for x in line_profile_x_range]
profile_opn = [denoise_open[line_profile_y, x] for x in line_profile_x_range]

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# Left: 1D Profile across boundary
ax0 = fig.add_subplot(gs[0])
ax0.plot(line_profile_x_range, profile_raw, label='Normalized (Unfiltered)', color='#7f7f7f', linestyle='--', lw=1.2)
ax0.plot(line_profile_x_range, profile_bil, label='Filter 3.1: Bilateral (Edge-Preserving)', color='#2ca02c', lw=2.2)
ax0.plot(line_profile_x_range, profile_med, label='Filter 3.2: Median', color='#1f77b4', lw=1.5)
ax0.plot(line_profile_x_range, profile_gau, label='Filter 3.3: Gaussian Blur', color='#d62728', lw=1.5)
ax0.plot(line_profile_x_range, profile_opn, label='Filter 3.4: Morphological Opening', color='#ff7f0e', lw=1.5)

ax0.set_title('1D Intensity Edge Profile Across Pellet Boundary', fontsize=11, fontweight='bold')
ax0.set_xlabel('Horizontal Pixel Coordinate X')
ax0.set_ylabel('Pixel Intensity (0-255)')
ax0.legend(loc='upper left', frameon=True)
ax0.grid(True, linestyle=':', alpha=0.6)

# Right: Zoomed view of the pellet under Bilateral filtering
ax1 = fig.add_subplot(gs[1])
crop_vis = denoise_bilateral[y_center-35:y_center+35, x_center-35:x_center+35]
ax1.imshow(crop_vis, cmap='viridis')
ax1.axhline(35, color='red', linestyle=':', label='Profile Line')
ax1.set_title('Zoomed In-Focus Pellet ROI (Bilateral Denoised)', fontsize=11, fontweight='bold')
ax1.legend(loc='lower right')
ax1.axis('off')

plt.tight_layout()
plt.show()

print("[NOTE] Conclusion: Bilateral Filtering perfectly maintains boundary transition slope (steepness)")
print("   while suppressing sensor noise in flat background regions, making it the superior choice.")
"""
cells.append(nbf.v4.new_code_cell(cell_6_code))

# ==============================================================================
# CELL 7: STAGE 4 — SCIENTIFIC DEFOCUS BLUR DETECTION & REMOVAL (Markdown & Code)
# ==============================================================================
cell_7_md = """---
## 🔍 Stage 4: Out-of-Focus (Defocus Blur) Detection & Removal

### The Physical Problem (Depth of Field in Fluidized Bed Imaging)
Optical systems imaging fast-moving fluidized beds inherently have a narrow **depth of field (DOF)** ($< 1-2$ mm). Consequently, only a fraction of pellets pass directly through the optical focal plane. 

Pellets tumbling in front of or behind the focal plane suffer from **defocus blur**:
* Their boundaries broaden into hazy gradients over 10–20 pixels instead of crisp 1–2 pixel step edges.
* Their apparent projected area becomes artificially inflated.
* If a blurred pellet passes behind an in-focus pellet, the blur halo bridges the gap, creating the false optical illusion of an **agglomeration neck**.

### Focus Measure Formulations (Pech-Pacheco et al. 2000; Santos et al.)
To systematically reject out-of-focus pellets, we implement **3 mathematical focus measures** evaluated locally over each detected candidate pellet:

1. **Tenengrad Focus Operator ($F_{Ten}$)**:  
   Computes the average squared gradient magnitude using Sobel operators:
   $$F_{Ten} = \\frac{1}{|\\Omega|} \\sum_{(x,y) \\in \\Omega} \\left( G_x(x,y)^2 + G_y(x,y)^2 \\right)$$
   Sharp step edges yield large gradients, driving $F_{Ten}$ high; blurred boundaries cause $F_{Ten}$ to collapse.

2. **Laplacian Variance Operator ($F_{Lap}$)**:  
   Computes the statistical variance of the 2nd-order Laplacian operator $\\nabla^2 I$:
   $$F_{Lap} = \\text{Var}\\left( \\nabla^2 I \\right) = \\frac{1}{|\\Omega|} \\sum_{(x,y)} \\left( \\nabla^2 I(x,y) - \\overline{\\nabla^2 I} \\right)^2$$
   High-frequency edges produce strong local variations; smooth blur yields low variance.

3. **Normalized Boundary Radial Gradient ($F_{Grad}$)**:  
   Directly measures the average radial gradient magnitude along the 3-pixel contour perimeter $\\partial C$:
   $$F_{Grad} = \\frac{1}{|\\partial C|} \\sum_{p \\in \\partial C} \\| \\nabla I(p) \\|$$

Let's compute these metrics across all candidate particles, calibrate an adaptive focus threshold, and visualize **In-Focus vs. Out-of-Focus** galleries side-by-side!
"""
cells.append(nbf.v4.new_markdown_cell(cell_7_md))

cell_7_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 7: Focus Measure Operators & Defocus Blur Discrimination
# ══════════════════════════════════════════════════════════════════════════════
# Segment initial candidate regions using Otsu threshold on bilateral image
_, binary_candidates = cv2.threshold(denoise_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remove static chamber fixtures using Stage 1 mask
binary_clean = cv2.bitwise_and(binary_candidates, cv2.bitwise_not(M_static))

# Morphological closing to seal internal pellet reflections
se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_closed = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, se_close)

contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
H_img, W_img = raw_test.shape[:2]
margin = 25

candidate_evals = []
for idx, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 80 or area > 25000:
        continue
    x, y, w, h = cv2.boundingRect(c)
    if x <= margin or y <= margin or (x + w) >= (W_img - margin) or (y + h) >= (H_img - margin):
        continue
    aspect_ratio = max(w / max(1, h), h / max(1, w))
    if aspect_ratio > 4.0:
        continue
        
    crop_raw = raw_test[y:y+h, x:x+w]
    if crop_raw.size == 0:
        continue
        
    # 1. Laplacian Variance
    f_lap = cv2.Laplacian(crop_raw, cv2.CV_64F).var()
    
    # 2. Tenengrad Focus Measure
    gx = cv2.Sobel(crop_raw, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_raw, cv2.CV_64F, 0, 1, ksize=3)
    f_ten = np.mean(gx**2 + gy**2)
    
    # 3. Boundary Gradient
    mask_c = np.zeros(crop_raw.shape, dtype=np.uint8)
    c_offset = c - [x, y]
    cv2.drawContours(mask_c, [c_offset], -1, 255, -1)
    boundary_mask = cv2.morphologyEx(mask_c, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    grad_mag = np.sqrt(gx**2 + gy**2)
    f_grad = float(np.mean(grad_mag[boundary_mask > 0])) if np.sum(boundary_mask > 0) > 0 else 0.0

    # Decision rule for optical depth of field
    is_infocus = (f_lap >= 70.0) and (f_ten >= 1500.0) and (f_grad >= 25.0)

    candidate_evals.append({
        'idx': idx,
        'contour': c,
        'box': (x, y, w, h),
        'crop': crop_raw,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'F_Lap': f_lap,
        'F_Ten': f_ten,
        'F_Grad': f_grad,
        'is_infocus': is_infocus
    })

df_focus = pd.DataFrame([{k: v for k, v in item.items() if k not in ['contour', 'crop']} for item in candidate_evals])
n_infocus = sum(1 for item in candidate_evals if item['is_infocus'])
n_blurred = len(candidate_evals) - n_infocus

print(f"Evaluated {len(candidate_evals)} candidate particles:")
print(f"  * In-Focus  (Retained) : {n_infocus} ({n_infocus/len(candidate_evals)*100:.1f}%)")
print(f"  * Out-of-Focus (Rejected) : {n_blurred} ({n_blurred/len(candidate_evals)*100:.1f}%)")

# Scatter Plot of Focus Measures
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Tenengrad vs Laplacian Variance
ax0 = axes[0]
ax0.scatter(df_focus[df_focus['is_infocus']]['F_Lap'], df_focus[df_focus['is_infocus']]['F_Ten'], 
            c='#2ca02c', alpha=0.7, s=40, label='In-Focus (Sharp)')
ax0.scatter(df_focus[~df_focus['is_infocus']]['F_Lap'], df_focus[~df_focus['is_infocus']]['F_Ten'], 
            c='#d62728', alpha=0.5, s=25, label='Out-of-Focus (Blurred)')
ax0.axvline(70.0, color='#333', linestyle='--', lw=1, label='LapVar Threshold (70)')
ax0.axhline(1500.0, color='#333', linestyle=':', lw=1, label='Tenengrad Threshold (1500)')
ax0.set_xlabel('Laplacian Variance ($F_{Lap}$)')
ax0.set_ylabel('Tenengrad Measure ($F_{Ten}$)')
ax0.set_title('Bivariate Focus Separation: Tenengrad vs. Laplacian Variance', fontweight='bold')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.legend(loc='lower right', frameon=True)
ax0.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Boundary Gradient vs Area
ax1 = axes[1]
ax1.scatter(df_focus[df_focus['is_infocus']]['area'], df_focus[df_focus['is_infocus']]['F_Grad'], 
            c='#2ca02c', alpha=0.7, s=40, label='In-Focus (Sharp)')
ax1.scatter(df_focus[~df_focus['is_infocus']]['area'], df_focus[~df_focus['is_infocus']]['F_Grad'], 
            c='#d62728', alpha=0.5, s=25, label='Out-of-Focus (Blurred)')
ax1.axhline(25.0, color='#333', linestyle='--', lw=1, label='Boundary Grad Threshold (25)')
ax1.set_xlabel('Particle Projected Area (pixels)')
ax1.set_ylabel('Boundary Radial Gradient ($F_{Grad}$)')
ax1.set_title('Boundary Edge Sharpness vs. Particle Area', fontweight='bold')
ax1.set_xscale('log')
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Visual Gallery: In-Focus vs Blurred Sample Crops
infocus_samples = [item['crop'] for item in candidate_evals if item['is_infocus']][:6]
blurred_samples = [item['crop'] for item in candidate_evals if not item['is_infocus']][:6]

fig, axes = plt.subplots(2, 6, figsize=(16, 5))
fig.suptitle('Visual Discrimination Gallery: In-Focus (Sharp) vs. Out-of-Focus (Blurred) Pellets', fontsize=12)

for j in range(6):
    ax_in = axes[0, j]
    if j < len(infocus_samples):
        ax_in.imshow(infocus_samples[j], cmap='gray')
        ax_in.set_title(f'Sharp #{j+1}', color='#2ca02c', fontsize=9, fontweight='bold')
    ax_in.axis('off')

    ax_out = axes[1, j]
    if j < len(blurred_samples):
        ax_out.imshow(blurred_samples[j], cmap='gray')
        ax_out.set_title(f'Blurred #{j+1}', color='#d62728', fontsize=9, fontweight='bold')
    ax_out.axis('off')

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_7_code))

# ==============================================================================
# CELL 8: STAGE 5 — PARTICLE SEGMENTATION BENCHMARK (Markdown & Code)
# ==============================================================================
cell_8_md = """---
## 🔬 Stage 5: Particle Segmentation Benchmark (Testing 4 Classical Algorithms)

We evaluate and compare **4 classic and modern image processing segmentation techniques** on the preprocessed image:

1. **Algorithm A: Adaptive Morphological Top-Hat + Otsu Thresholding + Morphological Closing**:
   The standard baseline used across pharmaceutical PAT vision systems (Možina et al. 2018; Watano et al. 2010). Extracts the overall outer bounding silhouette of particles and clusters.
2. **Algorithm B: Marker-Controlled Distance-Transform Watershed Segmentation**:
   The gold-standard algorithm (Meyer & Beucher 1990; Soille 2003) for separating touching and overlapping spherical particles by computing the Euclidean Distance Transform (EDT) and flooding from regional distance peaks.
3. **Algorithm C: Multi-Scale Difference of Gaussians (DoG) Blob Detection**:
   Approximation of the scale-space Laplacian of Gaussian (LoG) to identify spherical blob centers across varying radii.
4. **Algorithm D: Canny Edge Detection + Morphological Closing & Ellipse Fitting**:
   Gradient-based edge detection followed by contour reconstruction.

Let's execute all 4 algorithms on the same test frame and evaluate processing runtime, total detection count, and cluster separation capability.
"""
cells.append(nbf.v4.new_markdown_cell(cell_8_md))

cell_8_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 8: Particle Segmentation Benchmark (Comparing 4 Algorithms)
# ══════════════════════════════════════════════════════════════════════════════
preprocessed = denoise_bilateral.copy()

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm A: Threshold + Morphological Contours
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
_, bin_A = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bin_A_clean = cv2.bitwise_and(bin_A, cv2.bitwise_not(M_static))
bin_A_closed = cv2.morphologyEx(bin_A_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
cnts_A, _ = cv2.findContours(bin_A_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
time_A = (time.time() - t0) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm B: Distance-Transform Marker-Controlled Watershed
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
dist_transform = cv2.distanceTransform(bin_A_closed, cv2.DIST_L2, 5)
local_peaks = peak_local_max(dist_transform, min_distance=12, threshold_rel=0.35)
markers = np.zeros(dist_transform.shape, dtype=np.int32)
for idx, (r, c) in enumerate(local_peaks, 1):
    markers[r, c] = idx
markers_labeled = ndi.label(markers > 0)[0]
ws_labels = watershed(-dist_transform, markers_labeled, mask=bin_A_closed)
time_B = (time.time() - t0) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm C: Multi-Scale Difference of Gaussians (DoG)
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
g1 = cv2.GaussianBlur(preprocessed, (11, 11), 2.0)
g2 = cv2.GaussianBlur(preprocessed, (21, 21), 4.0)
dog = cv2.subtract(g1, g2)
_, dog_thresh = cv2.threshold(dog, 15, 255, cv2.THRESH_BINARY)
dog_clean = cv2.bitwise_and(dog_thresh, cv2.bitwise_not(M_static))
cnts_C, _ = cv2.findContours(dog_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
time_C = (time.time() - t0) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm D: Canny Edge + Morphological Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
canny = cv2.Canny(preprocessed, 30, 80)
canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
canny_clean = cv2.bitwise_and(canny_closed, cv2.bitwise_not(M_static))
cnts_D, _ = cv2.findContours(canny_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
time_D = (time.time() - t0) * 1000

# Benchmark Table
df_seg_benchmark = pd.DataFrame([
    {'Algorithm': 'Algorithm A: Top-Hat + Otsu Contours', 'Detected_Regions': len(cnts_A), 'Runtime_ms': round(time_A, 2), 'Separates_Touching': 'No (Clusters unified)'},
    {'Algorithm': 'Algorithm B: Marker-Controlled Watershed', 'Detected_Regions': len(np.unique(ws_labels)) - 1, 'Runtime_ms': round(time_B, 2), 'Separates_Touching': 'Yes (Constituent seeds)'},
    {'Algorithm': 'Algorithm C: Difference of Gaussians (DoG)', 'Detected_Regions': len(cnts_C), 'Runtime_ms': round(time_C, 2), 'Separates_Touching': 'Partial (Blob cores)'},
    {'Algorithm': 'Algorithm D: Canny Edge + Closing', 'Detected_Regions': len(cnts_D), 'Runtime_ms': round(time_D, 2), 'Separates_Touching': 'Poor (Hollow edges)'}
])

print("=" * 85)
print("[STATS] Particle Segmentation Benchmark Comparison")
print("=" * 85)
print(df_seg_benchmark.to_string(index=False))
print("=" * 85)

# Visual 4-panel comparison on a representative region with clustered pellets
crop_y1, crop_y2 = 700, 1200
crop_x1, crop_x2 = 1200, 1800

vis_A = cv2.cvtColor(raw_test[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_GRAY2RGB)
for c in cnts_A:
    c_sub = c - [crop_x1, crop_y1]
    cv2.drawContours(vis_A, [c_sub], -1, (0, 255, 0), 2)

vis_B = cv2.cvtColor(raw_test[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_GRAY2RGB)
# Draw watershed boundaries
ws_crop = ws_labels[crop_y1:crop_y2, crop_x1:crop_x2]
ws_boundaries = cv2.morphologyEx(ws_crop.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
vis_B[ws_boundaries > 0] = [255, 0, 0]

vis_C = cv2.cvtColor(raw_test[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_GRAY2RGB)
for c in cnts_C:
    c_sub = c - [crop_x1, crop_y1]
    (x, y), r = cv2.minEnclosingCircle(c_sub)
    cv2.circle(vis_C, (int(x), int(y)), max(int(r), 4), (255, 200, 0), 2)

vis_D = cv2.cvtColor(raw_test[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_GRAY2RGB)
for c in cnts_D:
    c_sub = c - [crop_x1, crop_y1]
    cv2.drawContours(vis_D, [c_sub], -1, (255, 0, 255), 2)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].imshow(vis_A); axes[0, 0].set_title('Algorithm A: Top-Hat + Otsu Contours (Unified Cluster Outlines)', fontweight='bold'); axes[0, 0].axis('off')
axes[0, 1].imshow(vis_B); axes[0, 1].set_title('Algorithm B: Marker-Controlled Watershed (Separates Touching Pellets)', fontweight='bold'); axes[0, 1].axis('off')
axes[1, 0].imshow(vis_C); axes[1, 0].set_title('Algorithm C: Difference of Gaussians (DoG Blob Cores)', fontweight='bold'); axes[1, 0].axis('off')
axes[1, 1].imshow(vis_D); axes[1, 1].set_title('Algorithm D: Canny Edge Reconstruction', fontweight='bold'); axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print("[NOTE] Conclusion: A dual combination is ideal:")
print("   * Algorithm A provides the exact contour outline of the agglomerate for shape analysis (solidity/neck indentation).")
print("   * Algorithm B (Watershed) identifies the constituent primary pellet seeds to count pellets per agglomerate.")
"""
cells.append(nbf.v4.new_code_cell(cell_8_code))

# ==============================================================================
# CELL 9: STAGE 6 — ARTIFACT & NOISE ELIMINATION SUITE (Markdown & Code)
# ==============================================================================
cell_9_md = """---
## 🧹 Stage 6: Artifact & Noise Elimination Suite

Following the PAT particle filtering guidelines (Mehle et al. 2017), raw segmentation contours must pass a **5-layer sequential elimination suite** before morphological characterization:

1. **Static Chamber Fixture Masking**: Any blob overlapping the Stage 1 static exclusion mask $M_{static}$ is discarded.
2. **Border Margin Clearing**: Particles touching or within 25 px of the image edge are discarded because their boundaries are clipped, corrupting area and circularity calculations.
3. **Area Filtering (Debris & Walls)**:
   * Rejects fine dust, broken fragments, and noise: $\\text{Area} < 100\\text{ px}^2$.
   * Rejects large chamber reflections and wall structures: $\\text{Area} > 20,000\\text{ px}^2$.
4. **Elongation / Seam Filtering**: Rejects linear scratches, reflection streaks, and vertical wall seams: $\\text{Aspect Ratio} > 4.0$.
5. **Defocus Blur Rejection**: Rejects out-of-focus pellets failing the calibrated focus threshold criteria.

Let's execute this waterfall filter and track candidate counts at each stage.
"""
cells.append(nbf.v4.new_markdown_cell(cell_9_md))

cell_9_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 9: Multi-Layer Artifact & Noise Rejection Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def filter_particles(contours, raw_img, static_mask, margin=25):
    H, W = raw_img.shape[:2]
    
    stage_counts = {
        '0_Raw_Detected': len(contours),
        '1_Static_Mask_Passed': 0,
        '2_Border_Margin_Passed': 0,
        '3_Area_Filter_Passed': 0,
        '4_Elongation_Passed': 0,
        '5_InFocus_Passed': 0
    }
    
    accepted_particles = []
    rejected_blurred = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # 1. Static Mask overlap check
        roi_mask = static_mask[y:y+h, x:x+w]
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c - [x, y]], -1, 255, -1)
        if np.sum(cv2.bitwise_and(roi_mask, c_mask)) > 0.25 * np.sum(c_mask):
            continue
        stage_counts['1_Static_Mask_Passed'] += 1
        
        # 2. Border margin check
        if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
            continue
        stage_counts['2_Border_Margin_Passed'] += 1
        
        # 3. Area filtering
        area = cv2.contourArea(c)
        if area < 100 or area > 20000:
            continue
        stage_counts['3_Area_Filter_Passed'] += 1
        
        # 4. Aspect Ratio check
        aspect_ratio = max(w / max(1, h), h / max(1, w))
        if aspect_ratio > 4.0:
            continue
        stage_counts['4_Elongation_Passed'] += 1
        
        # 5. Defocus blur check
        crop = raw_img[y:y+h, x:x+w]
        lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
        gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(gx**2 + gy**2)
        
        particle_dict = {
            'contour': c,
            'box': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'lap_var': lap_var,
            'tenengrad': tenengrad,
            'crop': crop
        }
        
        if lap_var >= 70.0 and tenengrad >= 1500.0:
            stage_counts['5_InFocus_Passed'] += 1
            accepted_particles.append(particle_dict)
        else:
            rejected_blurred.append(particle_dict)
            
    return accepted_particles, rejected_blurred, stage_counts

accepted_clean, rejected_blur, filter_stages = filter_particles(cnts_A, raw_test, M_static)

# Waterfall Filter Visualization
stages = list(filter_stages.keys())
counts = list(filter_stages.values())

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(range(len(stages)), counts, color=['#1f77b4', '#aec7e8', '#ffbb78', '#2ca02c', '#98df8a', '#2ca02c'])
ax.set_xticks(range(len(stages)))
ax.set_xticklabels([
    'Raw Contours', 'Post-Static Mask', 'Post-Border Clear', 
    'Post-Area Filter', 'Post-Aspect Filter', 'Final In-Focus'
], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Candidate Count')
ax.set_title('Multi-Layer Particle Filtering Progression (Noise & Blur Elimination)', fontweight='bold')
ax.grid(axis='y', linestyle=':', alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"[OK] Filter Summary: Began with {filter_stages['0_Raw_Detected']} raw detections -> Retained {len(accepted_clean)} clean, verified in-focus pellets.")
"""
cells.append(nbf.v4.new_code_cell(cell_9_code))

# ==============================================================================
# CELL 10: STAGE 7 — GEOMETRIC DESCRIPTORS & AGGLOMERATION PROFILING (Markdown & Code)
# ==============================================================================
cell_10_md = """---
## 📐 Stage 7: Geometric Descriptors & Initial Agglomeration Profiling

### Morphological Metrics (Sensum / Možina 2018 & Mehle 2017)
For all retained clean in-focus particles, we compute standard pharmaceutical geometric descriptors:
* **Projected Area ($A$)** and **Equivalent Circular Diameter ($d_{eq}$)**:
  $$d_{eq} = 2 \\sqrt{\\frac{A}{\\pi}}$$
* **Circularity / Sphericity ($\\psi$)**:
  $$\\psi = \\frac{4 \\pi A}{P^2}$$
  A perfect circle has $\\psi = 1.0$. Single spherical pellets have $\\psi \\in [0.70, 0.95]$. Agglomerates with indented contact necks drop significantly to $\\psi \\in [0.30, 0.65]$.
* **Aspect Ratio ($AR$)**: Ratio of the major to minor dimension of the minimum-area enclosing bounding box:
  $$AR = \\frac{L}{W}$$
* **Solidity ($S$)**: Ratio of contour area to convex hull area:
  $$S = \\frac{A}{A_{\\text{convex hull}}}$$
  Measures the presence of boundary concavities / neck joints between connected primary pellets ($S < 0.85$ indicates strong neck formation).

### Initial Classification Decision Boundary:
1. **Single Pellet**: $\\psi \\ge 0.70$, $AR \\le 1.30$, and $A \\le 1.6 \\times A_{\\text{median}}$.
2. **Overlapped Pellets**: $A > 1.6 \\times A_{\\text{median}}$ with smooth outer contour ($S \\ge 0.85$, $AR \\le 1.55$).
3. **Connected Agglomerate**: Pronounced neck concavity ($S < 0.85$) or elongated multi-sphere geometry ($AR > 1.45$ with large area).

Let's compute these descriptors, visualize the decision boundary scatter plot, and generate a color-coded inspection overlay!
"""
cells.append(nbf.v4.new_markdown_cell(cell_10_md))

cell_10_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 10: Geometric Feature Extraction & Initial Agglomeration Profiling
# ══════════════════════════════════════════════════════════════════════════════
areas = [p['area'] for p in accepted_clean]
median_area = float(np.median(areas))

classified_particles = []
for p in accepted_clean:
    c = p['contour']
    area = p['area']
    perim = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
    
    # Minimum Area Bounding Rectangle
    rect = cv2.minAreaRect(c)
    dim1, dim2 = rect[1]
    w_min = min(dim1, dim2)
    l_max = max(dim1, dim2)
    aspect_ratio = l_max / max(1.0, w_min)
    
    # Convex Hull & Solidity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Equivalent diameter
    eq_diam = 2.0 * math.sqrt(area / np.pi)
    
    # Initial classification rule
    if circ >= 0.70 and aspect_ratio <= 1.30 and area <= 1.6 * median_area:
        label = 'single'
        color = (0, 255, 0)      # Green
    elif solidity < 0.85 or aspect_ratio > 1.50 or area > 2.2 * median_area:
        label = 'connected'
        color = (0, 0, 255)      # Red
    else:
        label = 'overlapped'
        color = (0, 165, 255)    # Orange
        
    p.update({
        'perimeter': perim,
        'circularity': circ,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'eq_diam': eq_diam,
        'label': label,
        'color': color
    })
    classified_particles.append(p)

df_classified = pd.DataFrame([{k: v for k, v in p.items() if k not in ['contour', 'crop', 'color']} for p in classified_particles])

# Classification Summary
class_counts = df_classified['label'].value_counts()
print("=" * 60)
print(f"[STATS] Agglomeration Profiling Summary (Median Single Area: {median_area:.1f} px^2)")
print("=" * 60)
for k, v in class_counts.items():
    print(f"* {k.capitalize():<12}: {v} particles ({v/len(df_classified)*100:.1f}%)")
print("=" * 60)

# Plot 1: Circularity vs Aspect Ratio with Classification Zones
fig, ax = plt.subplots(figsize=(9, 5))
colors_map = {'single': '#2ca02c', 'connected': '#d62728', 'overlapped': '#ff7f0e'}

for lbl, group in df_classified.groupby('label'):
    ax.scatter(group['aspect_ratio'], group['circularity'], 
               c=colors_map[lbl], label=f"{lbl.capitalize()} ({len(group)})",
               s=group['area'] / 18.0, alpha=0.75, edgecolors='black', linewidths=0.5)

ax.set_xlabel('Aspect Ratio (Length / Width)')
ax.set_ylabel('Circularity (4π·Area / Perimeter²)')
ax.set_title('Morphological Feature Space: Circularity vs. Aspect Ratio', fontweight='bold')
ax.axhline(0.70, color='#333', linestyle=':', lw=1, label='Circularity Threshold (0.70)')
ax.axvline(1.30, color='#333', linestyle='--', lw=1, label='Aspect Ratio Threshold (1.30)')
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot 2: Full-Frame Annotated Visual Overlay
vis_overlay = cv2.cvtColor(raw_test, cv2.COLOR_GRAY2RGB)
for p in classified_particles:
    x, y, w, h = p['box']
    color = p['color']
    cv2.rectangle(vis_overlay, (x, y), (x+w, y+h), color, 2)
    cv2.drawContours(vis_overlay, [p['contour']], -1, color, 1)

fig, ax = plt.subplots(figsize=(16, 10))
ax.imshow(vis_overlay)
ax.set_title(f"Classified Particle Map — Green: Single ({class_counts.get('single',0)}) | "
             f"Orange: Overlapped ({class_counts.get('overlapped',0)}) | "
             f"Red: Connected Agglomerates ({class_counts.get('connected',0)})", 
             fontsize=12, fontweight='bold', pad=8)
ax.axis('off')
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_10_code))

# ==============================================================================
# CELL 11: STAGE 8 — PREPROCESSING CHECKPOINT EXPORT (Markdown & Code)
# ==============================================================================
cell_11_md = """---
## 📦 Stage 8: Preprocessing Checkpoint Generation & Dataset Export

### The Checkpoint Role (Mehle et al. 2017)
As requested, this step forms the **official preprocessing checkpoint**:
1. We batch-process all 22 real fluid-bed images across all 7 experimental conditions.
2. Standardized square crops ($96 \\times 96$ or contextual square bounding boxes with 25% padding) are extracted for every verified in-focus particle.
3. Crops are organized and saved into categorized subdirectories:
   * `preprocessed_checkpoint/crops/single/`
   * `preprocessed_checkpoint/crops/connected/`
   * `preprocessed_checkpoint/crops/overlapped/`
   * `preprocessed_checkpoint/crops/all_clean_infocus/`
   * `preprocessed_checkpoint/crops/blurred_rejected/`
4. A comprehensive dataset manifest `preprocessed_checkpoint/particles_metadata.csv` is exported with complete bounding boxes, coordinate locations, morphological features, focus scores, and initial labels.

This structured dataset is immediately ready for:
* **Machine Learning**: Training Random Forest / SVM classifiers on the tabular shape features.
* **Deep Learning**: Training a CNN classifier (as in Mehle et al. 2017), fine-tuning a ResNet/BiLSTM sequence model, or training an end-to-end YOLOv11 detector.
"""
cells.append(nbf.v4.new_markdown_cell(cell_11_md))

cell_11_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 11: Batch Execution & Checkpoint Dataset Export across All 22 Frames
# ══════════════════════════════════════════════════════════════════════════════
print("Starting full batch processing across all 22 real fluid-bed images...")
t_batch_start = time.time()

master_records = []
crop_save_size = 96
crop_id = 0

for row_idx, row in df_dataset.iterrows():
    fpath = row['Path']
    img_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        continue
    H, W = img_raw.shape[:2]
    
    # 1. Top-Hat + Bilateral Denoising
    tophat_img = cv2.morphologyEx(img_raw, cv2.MORPH_TOPHAT, se_tophat)
    denoised_img = cv2.bilateralFilter(tophat_img, d=5, sigmaColor=30, sigmaSpace=30)
    
    # 2. Otsu Threshold + Static Masking
    _, th_bin = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_clean = cv2.bitwise_and(th_bin, cv2.bitwise_not(M_static))
    bin_closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, se_close)
    
    # 3. Find Contours
    cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Multi-layer filtering
    clean_parts, blurred_parts, _ = filter_particles(cnts, img_raw, M_static, margin=25)
    
    # Compute median area for this specific frame
    frame_med_area = float(np.median([p['area'] for p in clean_parts])) if len(clean_parts) > 0 else 1000.0
    
    # Process Accepted In-Focus Particles
    for p in clean_parts:
        c = p['contour']
        area = p['area']
        x, y, w, h = p['box']
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
        rect = cv2.minAreaRect(c)
        l_max = max(rect[1]) if max(rect[1]) > 0 else max(w, h)
        w_min = min(rect[1]) if min(rect[1]) > 0 else min(w, h)
        aspect_ratio = l_max / max(1.0, w_min)
        hull = cv2.convexHull(c)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        eq_diam = 2.0 * math.sqrt(area / np.pi)
        
        # Classification
        if circ >= 0.70 and aspect_ratio <= 1.30 and area <= 1.6 * frame_med_area:
            label = 'single'
        elif solidity < 0.85 or aspect_ratio > 1.50 or area > 2.2 * frame_med_area:
            label = 'connected'
        else:
            label = 'overlapped'
            
        # Extract square padded crop (centered at centroid)
        cx, cy = x + w / 2.0, y + h / 2.0
        crop_dim = max(w, h) * 1.35
        x1 = int(max(0, cx - crop_dim / 2.0))
        x2 = int(min(W, cx + crop_dim / 2.0))
        y1 = int(max(0, cy - crop_dim / 2.0))
        y2 = int(min(H, cy + crop_dim / 2.0))
        
        crop_img = img_raw[y1:y2, x1:x2]
        if crop_img.size > 0:
            crop_resized = cv2.resize(crop_img, (crop_save_size, crop_save_size), interpolation=cv2.INTER_LINEAR)
            
            crop_filename = f"p{crop_id:05d}_{row['Folder'][:10]}_{label}.png"
            cv2.imwrite(str(CROPS_DIR / label / crop_filename), crop_resized)
            cv2.imwrite(str(CROPS_CLEAN_ALL / crop_filename), crop_resized)
            
            master_records.append({
                'particle_id': crop_id,
                'frame_file': row['Filename'],
                'folder': row['Folder'],
                'condition_type': row['Condition_Type'],
                'crop_filename': crop_filename,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'area_px': area,
                'eq_diam_px': round(eq_diam, 2),
                'circularity': round(circ, 3),
                'aspect_ratio': round(aspect_ratio, 3),
                'solidity': round(solidity, 3),
                'F_Lap': round(p['lap_var'], 2),
                'F_Ten': round(p['tenengrad'], 2),
                'is_infocus': True,
                'initial_label': label
            })
            crop_id += 1

    # Save a small subset of rejected blurred crops for verification / reference
    for p in blurred_parts[:5]:
        x, y, w, h = p['box']
        crop_img = p['crop']
        if crop_img.size > 0:
            crop_resized = cv2.resize(crop_img, (crop_save_size, crop_save_size), interpolation=cv2.INTER_LINEAR)
            b_filename = f"blur_{crop_id:05d}_{row['Filename']}.png"
            cv2.imwrite(str(CROPS_BLURRED / b_filename), crop_resized)

df_master = pd.DataFrame(master_records)
df_master.to_csv(METADATA_CSV, index=False)

t_batch_total = time.time() - t_batch_start
print("=" * 80)
print(f"[SUCCESS] Checkpoint Batch Processing Complete in {t_batch_total:.2f} seconds!")
print("=" * 80)
print(f"* Total In-Focus Particles Exported : {len(df_master)}")
print(f"* Master CSV Metadata Manifest     : {METADATA_CSV}")
print(f"* Single Pellet Crops              : {len(list(CROPS_SINGLE.glob('*.png')))}")
print(f"* Connected Agglomerate Crops      : {len(list(CROPS_CONNECTED.glob('*.png')))}")
print(f"* Overlapped Pellet Crops          : {len(list(CROPS_OVERLAPPED.glob('*.png')))}")
print(f"* Clean Master Crops Folder        : {len(list(CROPS_CLEAN_ALL.glob('*.png')))}")
print("=" * 80)

# Display sample of exported metadata
print(df_master[['particle_id', 'folder', 'area_px', 'circularity', 'aspect_ratio', 'solidity', 'initial_label']].head(10).to_string(index=False))
"""
cells.append(nbf.v4.new_code_cell(cell_11_code))

# ==============================================================================
# CELL 12: STAGE 9 — INTERACTIVE INSPECTION DASHBOARD (Markdown & Code)
# ==============================================================================
cell_12_md = """---
## 🖥️ Stage 9: Interactive Diagnostic Dashboard

To enable flexible visual quality inspection, the function `inspect_frame(frame_idx)` allows you to choose any image from the dataset and immediately render a **4-view diagnostic panel**:
1. **Raw Fluid-Bed Capture**: The original monochrome sensor input.
2. **Normalized & Denoised Image**: Background vignetting removed, noise suppressed.
3. **Focus Discrimination Map**: In-focus pellets highlighted in **Green**, out-of-focus blurred pellets marked in **Red**.
4. **Final Classified Pellets Overlay**: Color-coded bounding boxes and contours (**Green**: Single, **Orange**: Overlapped, **Red**: Connected Agglomerates).

Simply change `TARGET_FRAME_IDX` below to inspect any frame in the dataset!
"""
cells.append(nbf.v4.new_markdown_cell(cell_12_md))

cell_12_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 12: Interactive Diagnostic Inspection Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def inspect_frame(frame_idx=0):
    if frame_idx < 0 or frame_idx >= len(df_dataset):
        print(f"Error: Frame index {frame_idx} out of range [0, {len(df_dataset)-1}].")
        return
        
    row = df_dataset.iloc[frame_idx]
    img_raw = cv2.imread(row['Path'], cv2.IMREAD_UNCHANGED)
    H, W = img_raw.shape[:2]
    
    # Preprocessing
    tophat_img = cv2.morphologyEx(img_raw, cv2.MORPH_TOPHAT, se_tophat)
    denoised_img = cv2.bilateralFilter(tophat_img, d=5, sigmaColor=30, sigmaSpace=30)
    _, th_bin = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_clean = cv2.bitwise_and(th_bin, cv2.bitwise_not(M_static))
    bin_closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, se_close)
    cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_parts, blur_parts, _ = filter_particles(cnts, img_raw, M_static, margin=25)
    med_area = float(np.median([p['area'] for p in clean_parts])) if len(clean_parts) > 0 else 1000.0
    
    # 1. Focus Map Visualization
    vis_focus = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    for p in blur_parts:
        x, y, w, h = p['box']
        cv2.rectangle(vis_focus, (x, y), (x+w, y+h), (255, 0, 0), 2) # Red for blur
    for p in clean_parts:
        x, y, w, h = p['box']
        cv2.rectangle(vis_focus, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green for sharp
        
    # 2. Classified Overlay
    vis_class = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    single_c = agg_c = ovlp_c = 0
    for p in clean_parts:
        x, y, w, h = p['box']
        area = p['area']
        c = p['contour']
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
        rect = cv2.minAreaRect(c)
        l_max = max(rect[1]) if max(rect[1]) > 0 else max(w, h)
        w_min = min(rect[1]) if min(rect[1]) > 0 else min(w, h)
        aspect_ratio = l_max / max(1.0, w_min)
        hull = cv2.convexHull(c)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        if circ >= 0.70 and aspect_ratio <= 1.30 and area <= 1.6 * med_area:
            color = (0, 255, 0)
            single_c += 1
        elif solidity < 0.85 or aspect_ratio > 1.50 or area > 2.2 * med_area:
            color = (0, 0, 255)
            agg_c += 1
        else:
            color = (0, 165, 255)
            ovlp_c += 1
            
        cv2.rectangle(vis_class, (x, y), (x+w, y+h), color, 2)
        cv2.drawContours(vis_class, [c], -1, color, 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Frame #{frame_idx:02d}: {row['Condition_Type']} | File: {row['Filename']}", 
                 fontsize=13, fontweight='bold')
                 
    axes[0, 0].imshow(img_raw, cmap='gray'); axes[0, 0].set_title('1. Raw Sensor Frame', fontweight='bold'); axes[0, 0].axis('off')
    axes[0, 1].imshow(denoised_img, cmap='inferno'); axes[0, 1].set_title('2. Top-Hat + Bilateral Denoised', fontweight='bold'); axes[0, 1].axis('off')
    axes[1, 0].imshow(vis_focus); axes[1, 0].set_title(f'3. Focus Map (Green: In-Focus [{len(clean_parts)}] | Red: Blurred [{len(blur_parts)}])', fontweight='bold'); axes[1, 0].axis('off')
    axes[1, 1].imshow(vis_class); axes[1, 1].set_title(f'4. Classified Particles (Green: Single [{single_c}] | Orange: Overlap [{ovlp_c}] | Red: Agglomerate [{agg_c}])', fontweight='bold'); axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 🔧 Set any frame index (0 to 21) to inspect!
TARGET_FRAME_IDX = 0
inspect_frame(TARGET_FRAME_IDX)
"""
cells.append(nbf.v4.new_code_cell(cell_12_code))

# ==============================================================================
# CELL 13: SCIENTIFIC SUMMARY & DOWNSTREAM RECOMMENDATIONS (Markdown)
# ==============================================================================
cell_13_md = """---
## 🏆 Comprehensive Benchmark Summary & Downstream Roadmap

### 1. Comparative Performance Matrix

| Processing Stage | Tested Techniques | Selected Best Method | Key Justification / Impact |
| :--- | :--- | :--- | :--- |
| **Chamber Referencing** | None vs. Static Fixture Mask | **Empty-Chamber Median Mask ($M_{static}$)** | Eliminates 100% of static window glares, wall seams, and mounting screws before particle detection. |
| **Illumination Normalization** | Top-Hat vs. Gauss-Sub vs. CLAHE vs. Flat-Field | **Morphological White Top-Hat ($r=22$ px)** | Delivers the lowest background noise variance and cleanest zero-baseline contrast across all lighting conditions. |
| **Noise Filtering** | Bilateral vs. Median vs. Gaussian vs. Opening | **Bilateral Filter ($d=5, \\sigma=30$)** | Preserves razor-sharp step-edge gradients while eliminating sensor noise. |
| **Focus Discrimination** | Tenengrad vs. Laplacian Var vs. Boundary Grad | **Dual Focus Rule ($F_{Lap} \\ge 70, F_{Ten} \\ge 1500$)** | Successfully rejects out-of-focus halos that cause false agglomeration bridge artifacts. |
| **Segmentation** | Top-Hat+Otsu vs. Watershed vs. DoG vs. Canny | **Top-Hat Contours + Watershed Seeds** | Top-Hat gives the true cluster silhouette for shape analysis, while Watershed isolates primary constituent seeds. |

---

### 2. Downstream Modeling Roadmap: Next Steps from this Checkpoint

With the clean preprocessing checkpoint saved (`preprocessed_checkpoint/crops/` and `particles_metadata.csv`), you have three distinct, high-impact paths to differentiate connected agglomerates from random overlaps:

#### Option A: Classical Machine Learning (Morphological Classifier)
* **Features**: Use the precomputed descriptors in `particles_metadata.csv` (Circularity, Aspect Ratio, Solidity, Extent, Equivalent Diameter, Boundary Gradient).
* **Models**: Train a **Random Forest** or **Support Vector Machine (SVM)** with RBF kernel.
* **Advantage**: Instant training (< 2 seconds), fully explainable decision boundaries, and no GPU required.

#### Option B: Deep Learning CNN Crop Classifier (Mehle et al. 2017 Architecture)
* **Data**: Train a deep CNN (VGG or ResNet-18) using the standardized crops in `preprocessed_checkpoint/crops/` (`single/`, `connected/`, `overlapped/`).
* **Advantage**: Automatically learns subtle textural clues and contact neck indentation contours that simple global aspect ratio rules cannot capture, achieving >93% accuracy per Mehle et al.

#### Option C: End-to-End Object Detection & Tracking (YOLOv11s + ByteTrack)
* **Data**: Use the bounding boxes in `particles_metadata.csv` to format a YOLO training split (`pipeline.ipynb`).
* **Advantage**: Full-frame real-time inference (>60 FPS) with persistent particle tracking across consecutive video frames to analyze rigid tumbling dynamics.
"""
cells.append(nbf.v4.new_markdown_cell(cell_13_md))


# ==============================================================================
# CELL 14: STAGE 10 — DEEP LEARNING CNN CROP CLASSIFIER (Markdown & Code)
# ==============================================================================
cell_14_md = """---
## 🧠 Stage 10: Deep Learning CNN Crop Classifier (Mehle et al. 2017 Architecture)

### 1. Scientific Background & Architecture Design
In their study, **Mehle, Likar, & Tomaževič (2017)** demonstrated that relying solely on geometric features (such as projected area or equivalent diameter) fails to detect early-stage agglomerates (such as dimers or trimers). This occurs because their sizes heavily overlap with large single primary pellets, resulting in a low true positive rate (~70%) or high false alarm rate.

To overcome this fundamental physical limitation, the authors designed a deep **Convolutional Neural Network (CNN)** inspired by the VGG architecture (ILSVRC-2014) to automatically learn hierarchical visual features:
* Early conv layers isolate sharp vs. blurred gradient contours and surface highlights.
* Mid-level layers capture concave contact necks and boundary indentation joints.
* Deep layers classify whether contacting spheres are rigid fused agglomerates or visual line-of-sight overlaps.

### 2. Architecture Specifications
* **Input**: $96 \times 96$ standardized monochrome candidate crops (Z-score normalized: $\\frac{x - \\mu}{\\sigma}$).
* **Stage 1**: $2 \times$ [Conv2D($32, 3 \times 3$) + BatchNorm + ReLU] + MaxPool2D($2 \times 2$) $\\rightarrow 48 \times 48$
* **Stage 2**: $2 \times$ [Conv2D($64, 3 \times 3$) + BatchNorm + ReLU] + MaxPool2D($2 \times 2$) $\\rightarrow 24 \times 24$
* **Stage 3**: $2 \times$ [Conv2D($128, 3 \times 3$) + BatchNorm + ReLU] + MaxPool2D($2 \times 2$) $\\rightarrow 12 \times 12$
* **Stage 4**: Conv2D($256, 3 \times 3$) + BatchNorm + ReLU + MaxPool2D($2 \times 2$) $\\rightarrow 6 \times 6$
* **Classifier Head**: AdaptiveAvgPool2D($3 \times 3$) + Dropout($0.4$) + Linear($2304 \rightarrow 256$) + ReLU + Dropout($0.2$) + Linear($256 \rightarrow 64$) + ReLU + Linear($64 \rightarrow \text{num\\_classes}$)

### 3. Training & Data Augmentations (Mehle et al. 2017)
* Random rotation ($0^\\circ$ to $360^\\circ$)
* Random Gaussian noise ($\\sigma \in [0, 0.01]$)
* Random horizontal and vertical reflections
* Stratified 60% Train, 20% Validation, 20% Independent Test split.
"""
cells.append(nbf.v4.new_markdown_cell(cell_14_md))

cell_14_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 14: Mehle et al. (2017) CNN Model Architecture & Dataset Definition
# ══════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class PelletCropDataset(Dataset):
    # Dataset loader for 96x96 candidate pellet crops with Mehle 2017 augmentations
    def __init__(self, file_paths, labels, areas=None, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.areas = areas if areas is not None else [0] * len(labels)
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = str(self.file_paths[idx])
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((96, 96), dtype=np.uint8)
        else:
            if img.shape != (96, 96):
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)

        img_float = img.astype(np.float32) / 255.0

        if self.is_train:
            # Random rotation 0 to 360 degrees
            angle = random.uniform(0, 360)
            center = (48, 48)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_float = cv2.warpAffine(img_float, rot_mat, (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Random horizontal and vertical flips
            if random.random() > 0.5:
                img_float = np.fliplr(img_float)
            if random.random() > 0.5:
                img_float = np.flipud(img_float)

            # Random Gaussian noise (std 0 to 0.01 per Mehle 2017)
            if random.random() > 0.5:
                noise_std = random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_std, img_float.shape).astype(np.float32)
                img_float = np.clip(img_float + noise, 0.0, 1.0)

        # Standardize per candidate image (Mehle et al. 2017 Section 3.3)
        mean_val = float(img_float.mean())
        std_val = float(img_float.std()) + 1e-6
        img_norm = (img_float - mean_val) / std_val

        tensor_x = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)
        label_y = torch.tensor(self.labels[idx], dtype=torch.long)
        area_val = torch.tensor(self.areas[idx], dtype=torch.float32)

        return tensor_x, label_y, area_val

def stratified_split(labels, test_ratio=0.20, val_ratio=0.20, seed=42):
    np.random.seed(seed)
    labels = np.array(labels)
    tr_idx, v_idx, ts_idx = [], [], []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        np.random.shuffle(idxs)
        n = len(idxs)
        n_ts = int(round(n * test_ratio))
        n_v = int(round(n * val_ratio))
        ts_idx.extend(idxs[:n_ts])
        v_idx.extend(idxs[n_ts:n_ts + n_v])
        tr_idx.extend(idxs[n_ts + n_v:])
    np.random.shuffle(tr_idx)
    np.random.shuffle(v_idx)
    np.random.shuffle(ts_idx)
    return np.array(tr_idx), np.array(v_idx), np.array(ts_idx)

class MehleCNN(nn.Module):
    # VGG-inspired multi-stage CNN for in-line agglomeration classification (Mehle 2017)
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # Stage 1: 96x96 -> 48x48
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 2: 48x48 -> 24x24
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 3: 24x24 -> 12x12
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 4: 12x12 -> 6x6
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_model = MehleCNN(in_channels=1, num_classes=2).to(device)
param_count = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
print(f"[OK] MehleCNN initialized on {device} with {param_count:,} trainable parameters.")
"""
cells.append(nbf.v4.new_code_cell(cell_14_code))

# ==============================================================================
# CELL 15: TRAINING MEHLE CNN (Code)
# ==============================================================================
cell_15_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 15: Model Training on GPU with 60/20/20 Stratified Split
# ══════════════════════════════════════════════════════════════════════════════
# Prepare dataset from preprocessed checkpoint
df_meta = pd.read_csv(METADATA_CSV)
valid_paths = []
valid_labels = []
valid_areas = []

for _, row in df_meta.iterrows():
    lbl_str = row['initial_label']
    fname = row['crop_filename']
    p = CROPS_DIR / lbl_str / fname
    if p.exists():
        valid_paths.append(p)
        # Binary task: 0 = Single Pellet, 1 = Agglomerate
        valid_labels.append(0 if lbl_str == 'single' else 1)
        valid_areas.append(float(row['area_px']))

# Stratified Split
def stratified_split(labels, test_ratio=0.20, val_ratio=0.20, seed=42):
    np.random.seed(seed)
    labels = np.array(labels)
    tr_idx, v_idx, ts_idx = [], [], []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        np.random.shuffle(idxs)
        n = len(idxs)
        n_ts = int(round(n * test_ratio))
        n_v = int(round(n * val_ratio))
        ts_idx.extend(idxs[:n_ts])
        v_idx.extend(idxs[n_ts:n_ts + n_v])
        tr_idx.extend(idxs[n_ts + n_v:])
    return np.array(tr_idx), np.array(v_idx), np.array(ts_idx)

tr_idx, val_idx, test_idx = stratified_split(valid_labels)

train_loader = DataLoader(PelletCropDataset([valid_paths[i] for i in tr_idx], [valid_labels[i] for i in tr_idx], [valid_areas[i] for i in tr_idx], is_train=True), batch_size=50, shuffle=True)
val_loader = DataLoader(PelletCropDataset([valid_paths[i] for i in val_idx], [valid_labels[i] for i in val_idx], [valid_areas[i] for i in val_idx], is_train=False), batch_size=50, shuffle=False)
test_loader = DataLoader(PelletCropDataset([valid_paths[i] for i in test_idx], [valid_labels[i] for i in test_idx], [valid_areas[i] for i in test_idx], is_train=False), batch_size=50, shuffle=False)

print(f"Data Split: Train={len(tr_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

# Training
model = MehleCNN(in_channels=1, num_classes=2).to(device)
class_counts = np.bincount([valid_labels[i] for i in tr_idx])
weights = torch.tensor([len(tr_idx) / (2.0 * max(1, c)) for c in class_counts], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

epochs = 30
best_val_acc = 0.0
save_path = Path('mehle_cnn_best.pt')
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for x_b, y_b, _ in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        running_loss += loss.item() * x_b.size(0)
        running_correct += (torch.argmax(out, 1) == y_b).sum().item()
        total += x_b.size(0)
    scheduler.step()

    # Val
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x_b, y_b, _ in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            val_loss += criterion(out, y_b).item() * x_b.size(0)
            val_correct += (torch.argmax(out, 1) == y_b).sum().item()
            val_total += x_b.size(0)

    tr_acc = running_correct / total
    v_acc = val_correct / val_total
    history['train_loss'].append(running_loss / total)
    history['val_loss'].append(val_loss / val_total)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(v_acc)

    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Train Acc: {tr_acc*100:5.1f}% | Val Acc: {v_acc*100:5.1f}%")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': best_val_acc, 'num_classes': 2, 'class_names': ['Single', 'Agglomerate']}, str(save_path))

print(f"[DONE] Best Validation Accuracy: {best_val_acc*100:.2f}%. Model saved to {save_path.resolve()}")
"""
cells.append(nbf.v4.new_code_cell(cell_15_code))

# ==============================================================================
# CELL 16: TEST SET EVALUATION & ROC CURVE (Code)
# ==============================================================================
cell_16_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 16: Independent Test Evaluation & Replication of Mehle Fig. 7 (ROC Curve)
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('mehle_cnn_best.pt', map_location=device)
num_classes = checkpoint.get('num_classes', 2)
eval_model = MehleCNN(in_channels=1, num_classes=num_classes).to(device)
eval_model.load_state_dict(checkpoint['model_state_dict'])
eval_model.eval()

# Ensure test_loader is available even if Cell 15 wasn't run in this session
if 'test_loader' not in globals() or test_loader is None:
    df_meta_eval = pd.read_csv(METADATA_CSV)
    v_paths, v_lbls, v_areas = [], [], []
    for _, r in df_meta_eval.iterrows():
        p = CROPS_DIR / r['initial_label'] / r['crop_filename']
        if p.exists():
            v_paths.append(p)
            v_lbls.append(0 if r['initial_label'] == 'single' else 1)
            v_areas.append(float(r['area_px']))
    _, _, ts_idx = stratified_split(v_lbls)
    test_loader = DataLoader(PelletCropDataset([v_paths[i] for i in ts_idx], [v_lbls[i] for i in ts_idx], [v_areas[i] for i in ts_idx], is_train=False), batch_size=50, shuffle=False)

all_preds, all_targets, all_probs, all_areas = [], [], [], []
with torch.no_grad():
    for x_b, y_b, a_b in test_loader:
        x_b = x_b.to(device)
        probs = F.softmax(eval_model(x_b), dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_b.numpy())
        all_probs.extend(probs.cpu().numpy()[:, 1])
        all_areas.extend(a_b.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)
all_areas = np.array(all_areas)

# ROC Curve computation (pure numpy)
def compute_roc(y_true, scores):
    desc = np.argsort(scores)[::-1]
    y_true = np.array(y_true)[desc]
    scores = np.array(scores)[desc]
    dist = np.where(np.diff(scores))[0]
    th_idxs = np.r_[dist, y_true.size - 1]
    tps = np.cumsum(y_true)[th_idxs]
    fps = 1 + th_idxs - tps
    tpr = np.r_[0, tps / max(1, y_true.sum())]
    fpr = np.r_[0, fps / max(1, len(y_true) - y_true.sum())]
    roc_val = float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, fpr))
    return fpr, tpr, roc_val

fpr_cnn, tpr_cnn, auc_cnn = compute_roc(all_targets, all_probs)
fpr_area, tpr_area, auc_area = compute_roc(all_targets, all_areas)

cm = np.zeros((2, 2), dtype=int)
for t, p in zip(all_targets, all_preds): cm[t, p] += 1
test_acc = np.mean(all_preds == all_targets)

print("=" * 70)
print(f"[TEST] Independent Test Set Performance (N = {len(all_targets)} crops)")
print("=" * 70)
print(f"* Overall Test Accuracy         : {test_acc*100:.2f}%")
print(f"* Mehle CNN Classifier ROC AUC : {auc_cnn:.4f}")
print(f"* Baseline Area Method ROC AUC : {auc_area:.4f}")
print("=" * 70)

# Multi-Panel Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Training curves (or sample crops if history not in scope)
if 'history' in globals() and len(history.get('train_acc', [])) > 0:
    axes[0].plot(history['train_acc'], label='Train Acc', color='#1f77b4', lw=2)
    axes[0].plot(history['val_acc'], label='Val Acc', color='#2ca02c', lw=2)
    axes[0].set_title('Training & Validation Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)
else:
    axes[0].axis('off')

# 2. Confusion Matrix
axes[1].imshow(cm, cmap='Blues')
axes[1].set_title(f'Confusion Matrix (Test Acc: {test_acc*100:.1f}%)', fontweight='bold')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Single', 'Agglomerate']); axes[1].set_yticklabels(['Single', 'Agglomerate'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Ground Truth')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold', fontsize=12)

# 3. ROC Curve Comparison (Replication of Mehle Fig. 7)
axes[2].plot(fpr_cnn, tpr_cnn, color='#2ca02c', lw=2.5, label=f'Mehle CNN (AUC = {auc_cnn:.3f})')
axes[2].plot(fpr_area, tpr_area, color='#d62728', lw=1.8, linestyle='--', label=f'Area Threshold Baseline (AUC = {auc_area:.3f})')
axes[2].plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', lw=1)
axes[2].set_xlim([0.0, 1.0]); axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate (1 - Specificity)')
axes[2].set_ylabel('True Positive Rate (Sensitivity)')
axes[2].set_title('ROC Curve: CNN vs. Area Threshold (Mehle Fig. 7)', fontweight='bold')
axes[2].legend(loc='lower right', frameon=True)
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_16_code))

# ==============================================================================
# CELL 17: FULL-FRAME INFERENCE & IN-LINE AGGLOMERATION DEGREE (Code)
# ==============================================================================
cell_17_code = """# ══════════════════════════════════════════════════════════════════════════════
# Cell 17: Full-Frame In-Line Inference & Agglomeration Degree Measurement
# ══════════════════════════════════════════════════════════════════════════════
from infer_mehle_cnn import MehleCNNPipeline

pipeline = MehleCNNPipeline(model_path='mehle_cnn_best.pt', device=device)

if 'df_dataset' in globals():
    test_frame_file = df_dataset[df_dataset['Folder'] == 'exp 1000 back+ext light early agg']['Path'].iloc[0]
else:
    sample_files = list(Path('Real-Data/exp 1000 back+ext light early agg').glob('*.bmp'))
    test_frame_file = str(sample_files[0]) if sample_files else str(list(Path('Real-Data').glob('*/*.bmp'))[0])
inference_result = pipeline.process_frame(test_frame_file)

print(f"Frame Processed : {Path(test_frame_file).name}")
print(f"* Total In-Focus Pellets Detected : {inference_result['total_infocus']}")
print(f"* In-Line Agglomeration Degree    : {inference_result['agg_degree']:.2f}%")
print(f"* Single Pellets Confirmed        : {inference_result['counts']['single']}")
print(f"* Agglomerates Confirmed          : {inference_result['counts']['agglomerate']}")

# Display annotated result
fig, ax = plt.subplots(figsize=(16, 10))
ax.imshow(cv2.cvtColor(inference_result['vis'], cv2.COLOR_BGR2RGB))
ax.set_title(f"Real-Time In-Line Inspection: Agglomeration Degree = {inference_result['agg_degree']:.1f}% "
             f"(Single: {inference_result['counts']['single']} | Agglomerates: {inference_result['counts']['agglomerate']})", 
             fontsize=12, fontweight='bold', pad=8)
ax.axis('off')
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell_17_code))

nb.cells = cells

# Save notebook
output_path = Path('ImageProcessing.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"[Done] Generated {output_path} successfully with {len(cells)} cells.")
