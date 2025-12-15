# Project Structure

This document describes the organized structure of the OS Fingerprinting project.

## Directory Layout

```
.
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python package dependencies
├── README.md                       # Main project documentation
├── .gitignore                      # Git ignore rules (excludes large data files)
├── PROJECT_STRUCTURE.md            # This file
│
├── data/                           # Dataset and data files
│   ├── flows_ml_ready.csv         # Main ML-ready dataset (52MB)
│   ├── flows_ground_truth_merged_anonymized.csv  # Ground truth data (92MB)
│   ├── flows_ground_truth_merged_anonymized.zip # Compressed version
│   ├── test.csv                   # Test dataset
│   ├── advanced-models.zip         # Model artifacts
│   └── catboost_info/             # CatBoost training logs
│
├── notebooks/                     # Jupyter notebooks
│   ├── Cleaning&EDA.ipynb         # Data cleaning and exploratory analysis
│   ├── BaseModelsWithExplainability.ipynb  # Basic models with explainability
│   ├── AdvancedModelsWithExplainability.ipynb  # Advanced models with explainability
│   └── SEE_OSFingerprinting.ipynb # Additional analysis
│
├── images/                         # Visualization images
│   ├── permutation_importance_all_models.png
│   ├── shap_summary_RandomForest.png
│   ├── shap_beeswarm_RandomForest.png
│   ├── lime_RandomForest_sample_*.png
│   ├── lime_LogisticRegression_sample_*.png
│   ├── logistic_coefficients_*.png
│   ├── feature_importance_heatmap_all_models.png
│   └── model_agreement_correlation.png
│
└── docs/                          # Documentation files
    ├── git_push_commands.md       # Git commands guide
    ├── git_abort_commands.md      # Git abort commands
    └── merge_master_to_main.md    # Branch merging guide
```

## File Organization Rationale

### Root Level
- **app.py**: Main application entry point
- **requirements.txt**: Python dependencies for easy installation
- **README.md**: Primary project documentation
- **.gitignore**: Excludes large files and unnecessary artifacts

### data/
Contains all dataset files and data artifacts:
- Large CSV files (excluded from Git via .gitignore)
- Compressed archives
- Model training artifacts (catboost_info)

### notebooks/
All Jupyter notebooks organized by purpose:
- Data cleaning and EDA
- Model training and evaluation
- Explainability analysis

### images/
All visualization outputs:
- Model explainability plots
- Feature importance visualizations
- Performance metrics charts

### docs/
Supporting documentation:
- Git workflow guides
- Command references

## Git Configuration

### Files Excluded from Git (.gitignore)
- `data/*.csv` - Large dataset files
- `data/*.zip` - Compressed archives
- `data/catboost_info/` - Training logs
- Python cache files
- IDE configuration files
- OS-specific files

### Recommended: Use Git LFS for Large Files
If you need to version control large files:
```bash
git lfs install
git lfs track "data/*.csv"
git lfs track "data/*.zip"
```

## Running the Application

1. Ensure data file exists: `data/flows_ml_ready.csv`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Streamlit app: `streamlit run app.py`

## Code Updates Made

- Updated `app.py` to load data from `data/flows_ml_ready.csv`
- All image paths updated to reference `images/` folder
- README.md updated with new project structure
- Created comprehensive .gitignore file

