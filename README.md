# Disrupted Developmental Trajectories of Cortical Circuitry, Structural Organization, and Spine Morphology in 7q11.23 Copy Number Variant Syndromes

## Overview
This repository contains the two-photon calcium imaging analysis pipeline used to characterize network-level neural activity in GCaMP6m-expressing cortical organoids derived from individuals with 7q11.23 copy number variant (CNV) syndromes (Williams syndrome deletion and duplication). The pipeline computes pairwise cross-correlation coefficients (CC) among neurons, detects population-level synchrony events, and quantifies changes in network dynamics following pharmacological dissection of inhibitory and excitatory circuits.

## Authors

- **Inyoung Hwang** — Project lead, data collection, dataset curation
  Department of Molecular, Cellular, and Developmental Biology, Neuroscience Research Institute, University of California, Santa Barbara, CA, USA
  inyounghwang@ucsb.edu

- **Jasmine S. Yeo** — Data analysis and visualization
  Department of Molecular, Cellular, and Developmental Biology, University of California Santa Barbara, Santa Barbara, CA, USA 93106
  jasmineyeo@ucsb.edu

## Dataset Description
Two-photon calcium imaging was performed on cortical organoids expressing GCaMP6m. Imaging was carried out on a PrairieView microscope at ~10–15 Hz frame rate. Cell segmentation and fluorescence extraction were performed with Suite2p prior to this analysis.

### Data Format
| File Type | Description |
|-----------|-------------|
| `suite2p/plane0/F.npy` | Raw fluorescence traces (cells × frames) |
| `suite2p/plane0/Fneu.npy` | Neuropil fluorescence |
| `suite2p/plane0/iscell.npy` | Cell classifier output from Suite2p |
| `suite2p/plane0/stat.npy` | ROI statistics |
| `suite2p/plane0/ops.npy` | Acquisition options |
| `<recording>.xml` | PrairieView XML metadata (frame timestamps, frame rate, zoom) |
| `<recording>_processed_POPULATION_SYNC.h5` | Pipeline output: correlation matrices, synchrony events, filtering stats |

## Analysis Pipeline

Scripts are organized by execution order. Each script is self-documented with its purpose, inputs, outputs, and dependencies in its header.

| Order | Script | Description |
|-------|--------|-------------|
| 1 | `2P-organoid/CalculateCC.py` | Main pipeline: dF/F → spike deconvolution → two-stage ROI filtering → cross-correlation → population synchrony detection |
| 2 | `2P-organoid/GabazineComparison.py` | Compare network synchrony before and after Gabazine (GABA-A block) application |
| 3 | `2P-organoid/MultiDrug_Comparison.py` | Track network synchrony across sequential pharmacological dissection (Gabazine → AP5 → CNQX → TTX) |

### Helper Modules (`helper/`)

| Module | Description |
|--------|-------------|
| `twop.py` | Loads Suite2p outputs and computes neuropil-corrected dF/F |
| `Process_Spike_GC6m.py` | OASIS spike deconvolution tuned for GCaMP6m (τ = 245 ms) |
| `files.py` | Reads PrairieView XML metadata; reads/writes nested dicts to HDF5 |
| `time.py` | Datetime conversion utilities for HDF5 storage |

## Requirements

- Python >= 3.9
- See `environment.yml` for full dependency list

## Installation & Setup

```bash
cd Organoid_GCaMP_CC_Analysis
conda env create -f environment.yml
conda activate organoid-cc
```

## Usage

### 1. Baseline cross-correlation analysis (all recordings in a folder)

Edit `folder_path` in `CalculateCC.py` to point to your batch recording directory, then run:

```bash
python 2P-organoid/CalculateCC.py
```

Each subfolder is expected to contain a `suite2p/plane0/` directory. Output `.h5` files and figures are saved within each recording folder.

### 2. Gabazine comparison

Edit `base_folder` in `GabazineComparison.py` and run:

```bash
python 2P-organoid/GabazineComparison.py
```

### 3. Multi-drug dissection

Edit `base_folder` in `MultiDrug_Comparison.py` and run:

```bash
python 2P-organoid/MultiDrug_Comparison.py
```

### 4. Video generation (optional)

Edit `suite2p_path` in `helper/CreateVideos.py` and run:

```bash
python helper/CreateVideos.py
```

## Project Structure

```
Organoid_GCaMP_CC_Analysis/
├── README.md
├── environment.yml
├── .gitignore
├── 2P-organoid/
│   ├── CalculateCC.py            # Main CC analysis pipeline
│   ├── GabazineComparison.py     # Pre/post Gabazine comparison
│   └── MultiDrug_Comparison.py   # Sequential multi-drug analysis
└── helper/
    ├── __init__.py               # Package initialization
    ├── twop.py                   # Suite2p data loader + dF/F calculator
    ├── Process_Spike_GC6m.py     # OASIS spike deconvolution (GCaMP6m)
    ├── files.py                  # XML + HDF5 I/O utilities
    ├── time.py                   # Datetime conversion utilities
    └── CreateVideos.py           # Video generation from Suite2p data
```
