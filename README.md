# CountARFactuals

This is the code repository for my Master's Thesis: **"Using Counterfactual Explanations to Find Gut Bacteria Related to Colorectal Cancer"**.

---

## Repository Structure

```text
├── CF_methods/
│   └── LIME/
│       ├── lime_counterfactual.py
│       └── __init__.py
│
├── CountARFactuals/
│   ├── arf.py
│   ├── CountARFactuals.py
│   ├── transforms.py
│   ├── utils.py
│   ├── zi_beta.py
│   └── __init__.py
│
├── data/
│   ├── data_diet_filtered.zip
│   └── winequality-red.csv
│
└── experiments/
    ├── augmentation.py
    ├── augmentation_ablation_arf.py
    ├── augmentation_ablation_discr.py
    ├── augmentation_ablation_rfdiscr.py
    ├── counterfactuals.py
    ├── experiment_utils.py
    └── metrics.py
```
---

## File Description

### CountARFactuals
- **`arf.py`**: Extension to `arfpy`, the official Python implementation of Adversarial Random Forests ([arfpy GitHub](https://github.com/bips-hb/arfpy)).  
  - **My main contributions**:
    - Implementation of immutable features.
    - Calculate a leaf posterior based on evidence (e.g., immutable features, desired class).
  
- **`CountARFactuals.py`**: Python-based implementation of the original R package ([countARFactuals GitHub](https://github.com/bips-hb/countARFactuals)).

### CF_methods
- Contains the LIME-C method ([LIME-C GitHub](https://github.com/yramon/LimeCounterfactual)), modified to support DataFrames used in this project.  
- DiCE variants are unmodified and loaded from the `dice_ml` package.

### Experiments
- **`counterfactuals.py`**: Runs benchmarking experiments on counterfactual explanations.
- **`augmentation.py`**: Runs benchmarking experiments on data augmentation methods.
- **Ablation Studies**:
  - **`augmentation_ablation_arf.py`**: Compares different hyperparameters for Adversarial Random Forest (ARF).
  - **`augmentation_ablation_discr.py`**: Compares discriminator models (XGB, MLP, LinReg).
  - **`augmentation_ablation_rfdiscr.py`**: Compares hyperparameters for Random Forest discriminator models.

---

## Data
- **`data_diet_filtered.zip`**: Contains the .csv file for a microbiome relative abundance dataset, called "CRC" in the thesis.
- **`winequality-red.csv`**: Public dataset for wine quality analysis.
- Note: The California Housing dataset was loaded directly from sklearn. The PDAC dataset is privately used by the Uniklinikum and not published in this repository.
