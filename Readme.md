
## 🚀 Note: The complete source code and pretrained weights will be released soon.


# SYNAPSE-Net: A Unified Framework with Lesion-Aware Hierarchical Gating for Robust Segmentation of Heterogeneous Brain Lesions

This repository provides the model description for **SYNAPSE-Net**, a single, unified framework for robustly segmenting heterogeneous brain lesions from multi-modal MRI scans, as presented in the paper:

> **SYNAPSE-Net: A Unified Framework with Lesion-Aware Hierarchical Gating for Robust Segmentation of Heterogeneous Brain Lesions**  
> *Md. Mehedi Hassan, Shafqat Alam, Shahriar Ahmed Seam, and Maruf Ahmed*

## Overview

SYNAPSE-Net is designed to address two critical challenges in clinical neuroimaging: the lack of **generalization** across different pathologies and the high **performance variance** of specialized deep learning models.

Instead of creating separate 'point solutions' for each task, SYNAPSE-Net provides a single, powerful, and adaptable architecture that achieves state-of-the-art performance across a variety of brain lesion segmentation tasks, including white matter hyperintensities (WMH), ischemic stroke, and glioblastoma.

## Key Features

- **Unified & Generalizable Architecture:** A single, fixed architecture that adapts to a variable number of input MRI modalities without modification, successfully handling diverse clinical tasks.
- **Advanced Hybrid Architecture:** Integrates the strengths of CNNs for local feature extraction and Transformers for global context modeling.
- **Adaptive Modality Fusion:** Employs a novel Cross-Modal Attention Fusion (CMAF) mechanism that intelligently integrates information from different MRI sequences.
- **High-Fidelity Reconstruction:** A hierarchical gated decoder focuses on lesion-rich areas to reconstruct highly accurate and geometrically precise segmentation masks.
- **Robust Training Strategy:** A variance-reduction training paradigm combines difficulty-aware sampling and a composite loss function to improve reliability and reduce performance inconsistency, especially on small or subtle lesions.

## Architecture

The SYNAPSE-Net architecture is composed of three key stages: a multi-stream encoder, a hybrid bottleneck for feature fusion, and a hierarchical gated decoder for mask reconstruction.

![Graphical Abstract](Graphical%20Abstract.png)
*Fig 1. The Graphical Abstract*

#### 1. Unified Multi-Stream Encoder
To preserve crucial modality-specific pathological information, each input MRI sequence (e.g., T1, T2-FLAIR, DWI) is processed by an independent, parallel CNN encoder. This "late-fusion" approach prevents the early loss of subtle indicators. Skip connections from each stream are later refined and fused using a Convolutional Block Attention Module (CBAM) before being passed to the decoder.

#### 2. Hybrid Bottleneck
This stage creates a unified, semantically-rich feature representation from the deepest features of each encoder stream.
- **Intra-Modal Refinement:** A **Swin Transformer** layer is applied independently to each stream's feature map to enrich it with global spatial context, overcoming the limited receptive field of standard CNNs.
- **Adaptive Cross-Modal Attention Fusion (CMAF):** The refined feature maps are then fused using a bi-directional multi-head cross-attention mechanism. This allows the different modality streams to query and enrich one another, creating a powerful, unified multi-modal representation.

#### 3. Hierarchical Gated Decoder
The decoder is responsible for precise mask reconstruction and is built upon a **UNet++** backbone for its dense skip pathways.
- **Hierarchical Gating:** Before entering the decoder, skip connections are modulated by a series of 'Lesion Gate' modules. These gates use feature maps from deeper, more abstract layers to create a spatial attention mask, forcing the network to focus on informative pathological areas and improve boundary definition.
- **Deep Supervision:** The decoder generates multiple outputs at different depths, ensuring a strong gradient flow throughout the network for stable and effective training.

## Performance Highlights

SYNAPSE-Net was evaluated on three challenging public datasets, demonstrating a superior and more balanced performance profile than specialized, state-of-the-art methods. Pretrained Weights can be found in this [link](https://drive.google.com/drive/folders/1dTaGdnppu6p6WQDrL0tHeQCsbtnj50xG?usp=sharing).

*   **MICCAI 2017 WMH Challenge (Chronic Small Vessel Disease):**
    *   Achieved the highest **DSC of 0.831** and the best lesion **F1-Score of 0.816**.
    *   Set a new state-of-the-art in boundary accuracy with an **HD95 score of 3.03**, outperforming previous methods by over 50% in some cases.

*   **ISLES 2022 Challenge (Acute Ischemic Stroke):**
    *   Excelled in boundary precision, achieving a state-of-the-art **HD95 score of 9.69**.
    *   Attained a top-tier **DSC of 0.7532**, demonstrating robust volumetric accuracy.

*   **BraTS 2020 Challenge (Glioblastoma):**
    *   Achieved the highest **DSC of 0.865** for the challenging Tumor Core (TC) region.
    *   Delivered the most favorable **HD95 scores for both Tumor Core (4.34) and Enhancing Tumor (10.94)**, reflecting its enhanced capacity for delineating complex tumor boundaries.

## Citation

If you use SYNAPSE-Net in your research, please cite the original paper.

```bibtex
@article{Hassan_SYNAPSE-Net_2025,
  author    = {Md. Mehedi Hassan and Shafqat Alam and Shahriar Ahmed Seam and Dr. Maruf Ahmed},
  title     = {SYNAPSE-Net: A Lesion-aware Hierarchical Gating Framework for Robust Multimodal Brain Lesion Segmentation},
  journal   = {arXiv preprint},
  year      = {2025}
}
```

# SYNAPSE-Net Model

This repository contains the implementation of the SYNAPSE-Net model for brain lesion segmentation from multimodal MRI datasets.

## Project Structure(Example case was provided for WMH scripts)

```
.
├── src/                                      # Main source code for the project
│   ├── data_loaders/                         # Handles all data loading and preprocessing
│   │   ├── __init__.py                       # Makes 'data_loaders' a Python package
│   │   └── dataset_wmh.py                    # Defines the PyTorch Dataset for loading and augmenting WMH brain scan data
│   ├── models/                               # Contains the neural network architecture definitions
│   │   ├── __init__.py                       # Makes 'models' a Python package
│   │   ├── blocks.py                         # Contains reusable neural network components like convolutional and residual blocks
│   │   └── SYNAPSE-Net_N_mod.py              # Defines the main neural network architecture for the segmentation task
│   └── utils/                                # Utility scripts and helper functions
│       ├── helpers/                          # More specific helper modules
│       │   ├── __init__.py                   # Makes 'helpers' a Python package
│       │   └── wmh_helpers.py                # Provides specialized helper functions for WMH data processing or evaluation
│       ├── __init__.py                       # Makes 'utils' a Python package
│       └── utils.py                          # Contains general utility functions like loss functions, metrics, and training helpers
├── scripts/                                  # Executable scripts for running different stages of the pipeline
│   └── wmh/                                  # Scripts specifically for the WMH (White Matter Hyperintensities) task
│       ├── evaluate_test_set.py              # Calculates and reports final performance metrics on the test set predictions
│       ├── inference_sliding_window.py       # Generates full-resolution probability maps on a dataset using sliding window inference
│       ├── test_inference_final.py           # Runs final inference on the test set and saves the resulting binary segmentation masks
│       ├── train.py                          # Main script to execute the model training and validation pipeline
│       └── tune_postprocessing.py            # Finds optimal post-processing parameters (threshold, min size) using validation set results
├── requirements.txt                          # Lists project dependencies to be installed via pip
└── README.md                                 # Provides an overview of the project, setup instructions, and how to run it
```

## Requirements
## Data Format

The repository expects datasets mounted under `./data/` (default) and outputs/checkpoints under `./work/`. Three dataset layouts are supported:

### WMH Dataset Layout
```
data/wmh_split_data/          # Default WMH data root
├── train/
│   ├── training/            
│   │   ├── flair/           # {subject}_flair.nii.gz
│   │   ├── t1/              # {subject}_t1.nii.gz
│   │   └── ground_truth/    # {subject}_wmh.nii.gz
│   └── validation/
│       ├── flair/
│       ├── t1/
│       └── ground_truth/
└── test/
    └── train/
        ├── flair/
        ├── t1/
        └── ground_truth/
```

PowerShell command examples (WMH):
```powershell
# Train model (uses env vars or CLI args)
$env:DATA_ROOT = "./data/wmh_split_data"; $env:WORK_DIR = "./work"; python scripts/wmh/train.py

# Generate validation probability maps
python scripts/wmh/inference_sliding_window.py --model ./work/models/best_model.pth --data_root ./data/wmh_split_data/train/validation --output_dir ./work/validation_probabilities --roi_size 208,208

# Run test-set inference
python scripts/wmh/test_inference_final.py --model ./work/models/best_model.pth --test_data_root ./data/wmh_split_data/test --output_dir ./work/predictions

# Tune post-processing parameters
python scripts/wmh/tune_postprocessing.py --pred_dir ./work/validation_probabilities --gt_root ./data/wmh_split_data/train/validation

# Evaluate final predictions
python scripts/wmh/evaluate_test_set.py --prediction_dir ./work/predictions --test_gt_root ./data/wmh_split_data/test
```

### ISLES Dataset Layout
```
data/isles/                   # Default ISLES data root
├── sub-XXXX/                # One folder per subject
│   └── ses-0001/
│       └── dwi/
│           ├── sub-XXXX_ses-0001_dwi.nii.gz
│           └── sub-XXXX_ses-0001_adc.nii.gz
└── derivatives/
    └── sub-XXXX/
        └── ses-0001/
            └── sub-XXXX_ses-0001_msk.nii.gz
```

PowerShell command examples (ISLES):
```powershell
# Train model
$env:DATA_ROOT = "./data/isles"; $env:WORK_DIR = "./work"; python scripts/isles/train.py

# Generate probability maps
python scripts/isles/inference_sliding_window.py --model ./work/models/best_model.pth --data_root ./data/isles --output_dir ./work/isles_probs --roi_size 208,208

# Run final inference
python scripts/isles/test_inference_final.py --model ./work/models/best_model.pth --data_root ./data/isles --output_dir ./work/isles_predictions --threshold 0.45 --min_lesion_size 15

# Tune post-processing
python scripts/isles/tune_postprocessing.py --pred_dir ./work/isles_probs --gt_root ./data/isles/derivatives

# Evaluate predictions
python scripts/isles/evaluate_test_set.py --prediction_dir ./work/isles_predictions --test_gt_root ./data/isles/derivatives
```

### BraTS Dataset Layout
```
data/brats/                   # Default BraTS data root
├── subject_001/             # One folder per subject
│   ├── flair.nii.gz         # FLAIR modality
│   ├── t1.nii.gz           # T1 modality
│   ├── t1ce.nii.gz         # T1CE modality
│   └── t2.nii.gz           # T2 modality
└── labels/ (optional)
    └── subject_001_label.nii.gz
```

PowerShell command examples (BraTS):
```powershell
# Train model
$env:DATA_ROOT = "./data/brats"; $env:WORK_DIR = "./work"; python scripts/brats/train.py

# Generate probability maps
python scripts/brats/inference_sliding_window.py --model ./work/models/best_model.pth --data_root ./data/brats --output_dir ./work/brats_probs --roi_size 208,208

# Run final inference
python scripts/brats/test_inference_final.py --model ./work/models/best_model.pth --data_root ./data/brats --output_dir ./work/brats_predictions

# Tune post-processing
python scripts/brats/tune_postprocessing.py --pred_dir ./work/brats_predictions --val_root ./data/brats --out ./work/brats_tuning_results.csv

# Evaluate predictions
python scripts/brats/evaluate_test_set.py --data_root ./data/brats --prediction_dir ./work/brats_predictions --results_dir ./work/brats_results
```

Python packages required for this project are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

The model is based on the SYNAPSE-Net architecture with the following key components:

- Dual-encoder pathway for FLAIR and T1 MRI processing
- Swin Transformer blocks for feature enhancement
- Cross-modal fusion bottleneck
- UNet++-style decoder with lesion-aware gating
- Multi-scale supervision with auxiliary outputs
