# OS Fingerprinting Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing OS fingerprinting models, dataset characteristics, and explainability results.

## Features

- 📈 **Data Analytics Dashboard**: Dataset overview, modifications, and distributions
- 🤖 **Model Results (Basic)**: Performance metrics from basic models (before SMOTE)
- ⚖️ **SMOTE Results**: Advanced model performance after SMOTE balancing
- 🔀 **Comparison & Analysis**: Before/after SMOTE comparison with detailed analysis
- 🧪 **XGBoost Experiments**: Ablation studies, feature sets, and robustness analysis
- 🔍 **Explainability**: SHAP, LIME, permutation importance, and coefficient analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Download Data from https://zenodo.org/records/7635138
2. Modify 'Cleaning&EDA.ipynb' accordingly to generate 'flows_ml_ready.csv'
1. Make sure you have the `data/flows_ml_ready.csv` file in the `data/` directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. The dashboard will open in your default web browser

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── Cleaning&EDA.ipynb
├── BaseModelsWithExplainability.ipynb
├── AdvancedModelsWithExplainability.ipynb
├── SEE_OSFingerprinting.ipynb
├── data/                 # Dataset files(Must be created)
│   ├── flows_ml_ready.csv
│   └── ...
├── images/               # Visualization images
    ├── permutation_importance_all_models.png
    ├── shap_*.png
    ├── lime_*.png
    └── ...

```

## Dashboard Pages

### 📈 Data Analytics
- Dataset statistics and overview
- Data preprocessing steps and modifications
- OS family distribution visualizations
- Feature category breakdown
- Data quality metrics

### 🤖 Model Results (Basic)
- Performance metrics from basic models (LogisticRegression, RandomForest, ExtraTrees)
- Results before SMOTE resampling
- Train vs test accuracy comparisons
- Key observations and insights

### ⚖️ SMOTE Results
- Advanced model performance (XGBoost, LightGBM, CatBoost, MLP)
- Results after SMOTE balancing
- Detailed classification reports by class
- Macro F1 scores and class-specific performance

### 🔀 Comparison & Analysis
- Side-by-side comparison of before/after SMOTE results
- Detailed analysis of what changed and why
- Trade-offs and recommendations
- Performance impact analysis

### 🧪 XGBoost Experiments
- **Resampling Strategies**: Class weights vs SMOTE vs RUS+SMOTE
- **Feature Set Ablation**: TCP-only vs TCP+TLS vs Full feature set
- **Robustness Analysis**: Performance across different train/test splits
- Implications and recommendations

### 🔍 Explainability
- **Permutation Importance**: Global feature importance across models
- **SHAP Values**: Game-theory based feature contributions
- **LIME Explanations**: Local interpretable explanations for specific instances
- **Logistic Coefficients**: Direct interpretation of linear model weights
- Summary and implications

## Supported OS Families

- Android
- Linux
- Other
- Windows
- iOS

## Models Analyzed

### Basic Models
- LogisticRegression
- RandomForest
- ExtraTrees

### Advanced Models
- XGBoost
- LightGBM
- CatBoost
- MLP (Multi-layer Perceptron)

## Data Format

The dashboard expects:
- `data/flows_ml_ready.csv`: Main dataset with semicolon delimiter (place in `data/` folder)
- Optional: Explainability visualization images in `images/` folder (SHAP, LIME, coefficient plots)

## Key Insights

1. **TCP/IP features are highly informative**: 17 TCP features alone achieve 98% accuracy
2. **TLS adds incremental value**: TCP+TLS reaches highest accuracy (98.39%)
3. **SMOTE improves class balance**: Better macro F1 scores, especially for minority classes
4. **Model robustness**: XGBoost shows stable performance across different data splits
5. **Explainability confirms domain knowledge**: TCP SYN Size consistently most important feature

## Notes

- The dashboard loads data from `data/flows_ml_ready.csv` on startup (cached for performance)
- Visualization images in `images/` folder are optional - dashboard works without them
- All metrics and results are extracted from the analysis notebooks in `notebooks/`
- Large data files (CSV, ZIP) are excluded from Git via `.gitignore` - use Git LFS if needed
- Performance may vary based on system resources
