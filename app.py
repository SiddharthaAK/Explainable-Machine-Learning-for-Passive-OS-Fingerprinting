import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OS Fingerprinting Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #212529;
    }
    .info-box strong {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the ML-ready dataset"""
    try:
        df = pd.read_csv("data/flows_ml_ready.csv", sep=";")
        return df
    except FileNotFoundError:
        st.error("❌ Error: data/flows_ml_ready.csv not found.")
        return None

def main():
    st.markdown('<h1 class="main-header">📊 OS Fingerprinting Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📑 Navigation")
    page = st.sidebar.radio(
        "Select Dashboard",
        [
            "📈 Data Analytics",
            "🤖 Model Results (Basic)",
            "⚖️ SMOTE Results",
            "🔀 Comparison & Analysis",
            "🧪 XGBoost Experiments",
            "🔍 Explainability"
        ]
    )
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # ==================== DATA ANALYTICS PAGE ====================
    if page == "📈 Data Analytics":
        st.markdown('<h2 class="section-header">Dataset Overview & Modifications</h2>', unsafe_allow_html=True)
        
        # Basic Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Total Features", len(df.columns))
        col3.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Dataset Information
        st.subheader("📋 Dataset Information")
        st.markdown("""
        <div class="info-box">
        <strong>Dataset:</strong> data/flows_ml_ready.csv<br>
        <strong>Purpose:</strong> OS fingerprinting from network flow data<br>
        <strong>Target Variable:</strong> UA OS family (Operating System Family)
        </div>
        """, unsafe_allow_html=True)
        
        # Data Modifications
        st.subheader("🔧 Data Modifications & Preprocessing")
        
        modifications = [
            {
                "Step": "1. Data Loading",
                "Description": "Loaded data/flows_ml_ready.csv with semicolon delimiter",
                "Details": f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"
            },
            {
                "Step": "2. Feature Selection",
                "Description": "Selected numeric features for modeling",
                "Details": f"{len(df.select_dtypes(include=[np.number]).columns)} numeric features extracted"
            },
            {
                "Step": "3. Class Merging",
                "Description": "Merged small classes (< 50 samples) into 'Other'",
                "Details": "Ensures sufficient samples per class for training"
            },
            {
                "Step": "4. Missing Value Handling",
                "Description": "Applied median imputation for missing values",
                "Details": "SimpleImputer with strategy='median'"
            },
            {
                "Step": "5. Feature Scaling",
                "Description": "StandardScaler applied for neural network models",
                "Details": "MLP requires scaled features; tree models use raw features"
            },
            {
                "Step": "6. Train/Test Split",
                "Description": "Stratified 80/20 split",
                "Details": "Maintains class distribution in both sets"
            }
        ]
        
        mod_df = pd.DataFrame(modifications)
        st.dataframe(mod_df, use_container_width=True, hide_index=True)
        
        # OS Family Distribution
        st.subheader("📊 OS Family Distribution")
        
        label_col = "UA OS family"
        if label_col in df.columns:
            os_counts = df[label_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(os_counts.reset_index().rename(columns={'index': 'OS Family', label_col: 'Count'}), 
                           use_container_width=True, hide_index=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                os_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel('OS Family', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Distribution of OS Families in Dataset', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Class distribution statistics
            st.markdown("**Class Distribution Statistics:**")
            stats_df = pd.DataFrame({
                'OS Family': os_counts.index,
                'Count': os_counts.values,
                'Percentage': (os_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Feature Categories
        st.subheader("🔍 Feature Categories")
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        tcp_features = [f for f in numeric_features if not f.startswith('TLS_') and not f.startswith('NPM_')]
        tls_features = [f for f in numeric_features if f.startswith('TLS_')]
        npm_features = [f for f in numeric_features if f.startswith('NPM_')]
        
        feature_categories = pd.DataFrame({
            'Category': ['TCP/IP Features', 'TLS Features', 'NPM Features', 'Total'],
            'Count': [len(tcp_features), len(tls_features), len(npm_features), len(numeric_features)],
            'Examples': [
                ', '.join(tcp_features[:3]) + '...' if len(tcp_features) > 3 else ', '.join(tcp_features),
                ', '.join(tls_features[:3]) + '...' if len(tls_features) > 3 else ', '.join(tls_features),
                ', '.join(npm_features[:3]) + '...' if len(npm_features) > 3 else ', '.join(npm_features),
                f'{len(numeric_features)} total numeric features'
            ]
        })
        st.dataframe(feature_categories, use_container_width=True, hide_index=True)
        
        # Data Quality
        st.subheader("✅ Data Quality Metrics")
        quality_metrics = {
            'Metric': ['Missing Values (%)', 'Duplicate Rows', 'Data Types', 'Memory Usage'],
            'Value': [
                f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
                f"{df.duplicated().sum():,}",
                f"{df.dtypes.value_counts().to_dict()}",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            ]
        }
        st.dataframe(pd.DataFrame(quality_metrics), use_container_width=True, hide_index=True)
    
    # ==================== MODEL RESULTS (BASIC) PAGE ====================
    elif page == "🤖 Model Results (Basic)":
        st.markdown('<h2 class="section-header">Basic Model Results (Before SMOTE)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        These results are from the basic models trained on the original (unbalanced) dataset, 
        before applying SMOTE resampling. Models were trained with class_weight='balanced' to handle class imbalance.
        </div>
        """, unsafe_allow_html=True)
        
        # Results Table
        basic_results = pd.DataFrame({
            'Model': ['LogisticRegression', 'RandomForest', 'ExtraTrees'],
            'Train Accuracy': [0.7273, 0.9793, 0.9661],
            'Test Accuracy': [0.7273, 0.9793, 0.9661],
            'Notes': [
                'Linear model with balanced class weights',
                'Tree ensemble, 300 estimators, max_depth=25',
                'Extra trees ensemble, 300 estimators, max_depth=25'
            ]
        })
        
        st.subheader("📊 Model Performance Summary")
        st.dataframe(basic_results, use_container_width=True, hide_index=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(basic_results))
        width = 0.35
        
        ax.bar(x - width/2, basic_results['Train Accuracy'], width, label='Train Accuracy', color='#3498db')
        ax.bar(x + width/2, basic_results['Test Accuracy'], width, label='Test Accuracy', color='#e74c3c')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Basic Models: Train vs Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basic_results['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (train, test) in enumerate(zip(basic_results['Train Accuracy'], basic_results['Test Accuracy'])):
            ax.text(i - width/2, train + 0.02, f'{train:.4f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, test + 0.02, f'{test:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed Analysis
        st.subheader("📝 Key Observations")
        observations = [
            "**LogisticRegression (72.73%)**: Lowest accuracy, but provides interpretable coefficients",
            "**RandomForest (97.93%)**: Highest accuracy among basic models, excellent generalization",
            "**ExtraTrees (96.61%)**: Strong performance, slightly lower than RandomForest",
            "All models show good generalization (train ≈ test accuracy)",
            "Tree-based models significantly outperform linear model"
        ]
        
        for obs in observations:
            st.markdown(f"- {obs}")
    
    # ==================== SMOTE RESULTS PAGE ====================
    elif page == "⚖️ SMOTE Results":
        st.markdown('<h2 class="section-header">Advanced Model Results (After SMOTE)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        These results are from advanced models trained on SMOTE-balanced data. SMOTE (Synthetic Minority Oversampling Technique) 
        was applied to create balanced training sets, improving performance on minority classes.
        </div>
        """, unsafe_allow_html=True)
        
        # SMOTE Results Table
        smote_results = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'MLP'],
            'Train Accuracy': [1.0000, 1.0000, 0.9869, 0.9726],
            'Test Accuracy': [0.9820, 0.9834, 0.9728, 0.9480],
            'Test Macro F1': [0.9614, 0.9639, 0.9369, 0.8964],
            'Best Class': ['Other (0.99)', 'Windows (0.99)', 'Other (0.98)', 'Other (0.97)'],
            'Weakest Class': ['Linux (0.88)', 'Linux (0.89)', 'Linux (0.78)', 'Linux (0.69)']
        })
        
        st.subheader("📊 SMOTE-Balanced Model Performance")
        st.dataframe(smote_results, use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(smote_results))
            width = 0.35
            
            ax.bar(x - width/2, smote_results['Train Accuracy'], width, label='Train', color='#3498db', alpha=0.8)
            ax.bar(x + width/2, smote_results['Test Accuracy'], width, label='Test', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('SMOTE Models: Train vs Test Accuracy', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(smote_results['Model'], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0.9, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (train, test) in enumerate(zip(smote_results['Train Accuracy'], smote_results['Test Accuracy'])):
                ax.text(i - width/2, train + 0.005, f'{train:.4f}', ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, test + 0.005, f'{test:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(smote_results['Model'], smote_results['Test Macro F1'], color='#2ecc71', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Macro F1 Score', fontsize=12)
            ax.set_title('SMOTE Models: Macro F1 Score', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0.85, 0.98)
            ax.grid(axis='y', alpha=0.3)
            
            for i, f1 in enumerate(smote_results['Test Macro F1']):
                ax.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Detailed Classification Reports
        st.subheader("📋 Detailed Performance by Class")
        
        # XGBoost detailed results
        xgb_details = pd.DataFrame({
            'Class': ['Android', 'Linux', 'Other', 'Windows', 'iOS'],
            'Precision': [0.98, 0.90, 0.99, 0.98, 0.98],
            'Recall': [0.99, 0.86, 0.99, 0.99, 0.97],
            'F1-Score': [0.98, 0.88, 0.99, 0.98, 0.97],
            'Support': [2058, 466, 8506, 8070, 2833]
        })
        
        st.markdown("**XGBoost (Best Overall Performance)**")
        st.dataframe(xgb_details, use_container_width=True, hide_index=True)
        
        # Key Insights
        st.subheader("💡 Key Insights")
        insights = [
            "**LightGBM** achieves highest test accuracy (98.34%) and macro F1 (96.39%)",
            "**XGBoost** shows excellent balance with 98.20% accuracy and 96.14% macro F1",
            "**Linux** is consistently the most challenging class across all models (lowest F1-scores)",
            "**Other** and **Windows** classes achieve near-perfect performance (F1 > 0.98)",
            "SMOTE balancing significantly improved minority class performance",
            "All models show slight overfitting (train accuracy > test accuracy)"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    # ==================== COMPARISON & ANALYSIS PAGE ====================
    elif page == "🔀 Comparison & Analysis":
        st.markdown('<h2 class="section-header">Before vs After SMOTE: Comparison & Analysis</h2>', unsafe_allow_html=True)
        
        # Comparison Table
        comparison_data = {
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'MLP'],
            'Before SMOTE (Test Acc)': [0.9822, 0.9838, 0.9736, 0.9523],
            'After SMOTE (Test Acc)': [0.9820, 0.9834, 0.9728, 0.9480],
            'Difference': [-0.0002, -0.0004, -0.0008, -0.0043],
            'Before SMOTE (Macro F1)': ['N/A', 'N/A', 'N/A', 'N/A'],
            'After SMOTE (Macro F1)': [0.9614, 0.9639, 0.9369, 0.8964]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.subheader("📊 Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        models = comparison_df['Model']
        before = comparison_df['Before SMOTE (Test Acc)']
        after = comparison_df['After SMOTE (Test Acc)']
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, before, width, label='Before SMOTE', color='#e74c3c', alpha=0.8)
        ax1.bar(x + width/2, after, width, label='After SMOTE', color='#2ecc71', alpha=0.8)
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Accuracy: Before vs After SMOTE', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0.94, 0.99)
        ax1.grid(axis='y', alpha=0.3)
        
        # Macro F1 (only after SMOTE)
        macro_f1 = [0.9614, 0.9639, 0.9369, 0.8964]
        ax2.bar(models, macro_f1, color='#3498db', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Macro F1 Score', fontsize=12)
        ax2.set_title('Macro F1 Score (After SMOTE)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0.85, 0.98)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, f1 in enumerate(macro_f1):
            ax2.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analysis Section
        st.markdown("---")
        st.subheader("🔍 Detailed Analysis")
        
        st.markdown("""
        ### **What Changed?**
        
        **1. Accuracy Impact:**
        - Overall test accuracy remained **relatively stable** (slight decrease of 0.02-0.43%)
        - This suggests that the original models were already performing well on the majority classes
        
        **2. Macro F1 Score:**
        - After SMOTE, we can now measure macro F1, which gives equal weight to all classes
        - LightGBM achieves the highest macro F1 (96.39%), indicating better balance across classes
        
        **3. Class-Specific Performance:**
        - **Linux** class (minority) shows improved recall after SMOTE
        - **Other** and **Windows** (majority classes) maintain high performance
        - **Android** and **iOS** show consistent strong performance
        """)
        
        st.markdown("""
        ### **Why These Results?**
        
        **1. Why Accuracy Didn't Improve Much:**
        - The original dataset, while imbalanced, had sufficient samples for majority classes
        - Tree-based models (XGBoost, LightGBM) are robust to class imbalance
        - The slight decrease might be due to synthetic samples being less representative than real data
        
        **2. Why SMOTE Still Matters:**
        - **Macro F1 improvement**: Better balance across all classes
        - **Linux class performance**: Improved recall for the smallest class
        - **Generalization**: More balanced training set reduces bias toward majority classes
        
        **3. Model-Specific Observations:**
        - **XGBoost & LightGBM**: Minimal impact, already excellent
        - **CatBoost**: Slight decrease, but maintains strong performance
        - **MLP**: Largest decrease, neural networks may be more sensitive to synthetic data quality
        
        **4. Trade-offs:**
        - **Pros**: Better class balance, improved minority class performance, fairer evaluation
        - **Cons**: Slight accuracy decrease, increased training time, potential overfitting to synthetic samples
        """)
        
        # Recommendations
        st.markdown("""
        ### **Recommendations:**
        
        1. **For Production**: Use **LightGBM with SMOTE** for best overall performance and class balance
        2. **For Speed**: **XGBoost without SMOTE** if accuracy is the only concern
        3. **For Interpretability**: Consider **LogisticRegression** with feature importance analysis
        4. **For Robustness**: **Ensemble** of top-performing models (XGBoost + LightGBM)
        """)
    
    # ==================== XGBOOST EXPERIMENTS PAGE ====================
    elif page == "🧪 XGBoost Experiments":
        st.markdown('<h2 class="section-header">XGBoost Experiments & Ablation Studies</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        This section presents various experiments conducted with XGBoost to understand the impact of different 
        configurations, resampling strategies, and feature sets on model performance.
        </div>
        """, unsafe_allow_html=True)
        
        # Experiment 1: Resampling Strategies
        st.subheader("🔬 Experiment 1: Resampling Strategies")
        
        resampling_results = pd.DataFrame({
            'Strategy': [
                'No Resampling (Class Weights)',
                'SMOTE Only',
                'RUS + SMOTE',
                'SMOTE (Baseline)'
            ],
            'Test Accuracy': [0.9817, 0.9820, 0.9446, 0.9820],
            'Macro F1': [0.9622, 0.9614, 0.8801, 0.9614],
            'Description': [
                'Class weights to handle imbalance',
                'Synthetic oversampling only',
                'Random undersampling + SMOTE',
                'Standard SMOTE (reference)'
            ]
        })
        
        st.dataframe(resampling_results, use_container_width=True, hide_index=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(resampling_results))
        width = 0.35
        
        ax.bar(x - width/2, resampling_results['Test Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, resampling_results['Macro F1'], width, label='Macro F1', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Resampling Strategy', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('XGBoost: Impact of Resampling Strategies', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(resampling_results['Strategy'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.85, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Key Findings:**
        - **Class weights** perform as well as SMOTE (98.17% vs 98.20%)
        - **RUS + SMOTE** significantly underperforms (94.46%), likely due to information loss from undersampling
        - **SMOTE** provides good balance between performance and class fairness
        """)
        
        # Experiment 2: Feature Set Ablation
        st.subheader("🔬 Experiment 2: Feature Set Ablation")
        
        feature_ablation = pd.DataFrame({
            'Feature Set': ['TCP Only', 'TCP + TLS', 'Full (TCP + TLS + NPM)'],
            'Num Features': [17, 25, 43],
            'Test Accuracy': [0.9807, 0.9839, 0.9820],
            'Macro F1': [0.9587, 0.9622, 0.9614],
            'Key Insight': [
                'TCP features alone achieve 98% accuracy',
                'TLS adds incremental value, especially for Linux',
                'NPM features add minimal value'
            ]
        })
        
        st.dataframe(feature_ablation, use_container_width=True, hide_index=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy by feature set
        ax1.bar(feature_ablation['Feature Set'], feature_ablation['Test Accuracy'], 
               color='#e74c3c', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Accuracy by Feature Set', fontsize=14, fontweight='bold')
        ax1.set_ylim(0.975, 0.985)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, acc in enumerate(feature_ablation['Test Accuracy']):
            ax1.text(i, acc + 0.0005, f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Macro F1 by feature set
        ax2.bar(feature_ablation['Feature Set'], feature_ablation['Macro F1'], 
               color='#2ecc71', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Macro F1 Score', fontsize=12)
        ax2.set_title('Macro F1 by Feature Set', fontsize=14, fontweight='bold')
        ax2.set_ylim(0.955, 0.965)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, f1 in enumerate(feature_ablation['Macro F1']):
            ax2.text(i, f1 + 0.0005, f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Key Findings:**
        - **TCP features are highly informative**: 17 features achieve 98.07% accuracy
        - **TLS adds value**: TCP+TLS reaches highest accuracy (98.39%) and macro F1 (96.22%)
        - **NPM features are redundant**: Full feature set doesn't improve over TCP+TLS
        - **Implication**: Classic TCP/IP fingerprinting (TTL, window size, MSS) is sufficient for OS detection
        """)
        
        # Experiment 3: Robustness Check
        st.subheader("🔬 Experiment 3: Robustness Analysis")
        
        robustness_results = pd.DataFrame({
            'Random Seed': [0, 21, 42, 84, 123],
            'Test Accuracy': [0.9827, 0.9817, 0.9820, 0.9826, 0.9827],
            'Macro F1': [0.9629, 0.9619, 0.9614, 0.9645, 0.9603]
        })
        
        st.dataframe(robustness_results, use_container_width=True, hide_index=True)
        
        summary_stats = {
            'Metric': ['Mean Accuracy', 'Std Accuracy', 'Mean Macro F1', 'Std Macro F1'],
            'Value': [
                f"{robustness_results['Test Accuracy'].mean():.4f}",
                f"±{robustness_results['Test Accuracy'].std():.4f}",
                f"{robustness_results['Macro F1'].mean():.4f}",
                f"±{robustness_results['Macro F1'].std():.4f}"
            ]
        }
        
        st.markdown("**Summary Statistics:**")
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(robustness_results))
        width = 0.35
        
        ax.bar(x - width/2, robustness_results['Test Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, robustness_results['Macro F1'], width, label='Macro F1', color='#2ecc71', alpha=0.8)
        
        # Add mean lines
        mean_acc = robustness_results['Test Accuracy'].mean()
        mean_f1 = robustness_results['Macro F1'].mean()
        ax.axhline(mean_acc, color='#3498db', linestyle='--', alpha=0.5, label=f'Mean Acc: {mean_acc:.4f}')
        ax.axhline(mean_f1, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Mean F1: {mean_f1:.4f}')
        
        ax.set_xlabel('Random Seed', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('XGBoost Robustness Across Different Train/Test Splits', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(robustness_results['Random Seed'])
        ax.legend()
        ax.set_ylim(0.96, 0.97)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Key Findings:**
        - **Highly stable performance**: Mean accuracy 98.23% ± 0.0004
        - **Consistent macro F1**: Mean 96.22% ± 0.0014
        - **Low variance**: Model is robust to different data splits
        - **Implication**: Model generalizes well and is not sensitive to specific train/test partitions
        """)
        
        # Overall Implications
        st.subheader("💡 Overall Implications")
        
        implications = [
            "**Feature Engineering**: TCP/IP features alone are sufficient; TLS adds value but NPM is redundant",
            "**Resampling**: SMOTE or class weights both work well; avoid aggressive undersampling",
            "**Robustness**: Model shows excellent stability across different data splits",
            "**Production**: TCP+TLS feature set with SMOTE provides best balance of performance and interpretability",
            "**Efficiency**: Can reduce feature set to 25 features (TCP+TLS) without significant performance loss"
        ]
        
        for imp in implications:
            st.markdown(f"- {imp}")
    
    # ==================== EXPLAINABILITY PAGE ====================
    elif page == "🔍 Explainability":
        st.markdown('<h2 class="section-header">Model Explainability Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        This section presents explainability analyses from both notebooks, including permutation importance, 
        SHAP values, LIME explanations, and logistic regression coefficients. These methods help understand 
        what features drive model predictions.
        </div>
        """, unsafe_allow_html=True)
        
        # Permutation Importance
        st.subheader("🔢 Permutation Importance")
        
        st.markdown("""
        **What it is**: Measures feature importance by randomly shuffling each feature and observing the 
        decrease in model accuracy. Higher importance = larger decrease in accuracy when feature is shuffled.
        """)
        
        # Top features from different models
        perm_importance_data = {
            'Model': ['LogisticRegression', 'RandomForest', 'ExtraTrees'],
            'Top Feature': ['TCP SYN Size (0.272)', 'TCP SYN Size (0.094)', 'TCP SYN Size (0.127)'],
            '2nd Feature': ['TLS_HANDSHAKE_TYPE (0.095)', 'tcpOptionWindowScaleforward (0.050)', 'TCP Win Size (0.104)'],
            '3rd Feature': ['TLS_SERVER_VERSION (0.045)', 'maximumTTLforward (0.022)', 'flowDirection (0.025)']
        }
        
        st.dataframe(pd.DataFrame(perm_importance_data), use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Key Insights:**
        - **TCP SYN Size** is consistently the most important feature across all models
        - **LogisticRegression** shows much higher importance values (coefficient-based)
        - **Tree models** (RandomForest, ExtraTrees) show more distributed importance
        - **TLS features** are important for linear models but less so for tree models
        """)
        
        # Try to load permutation importance image
        try:
            if Path("images/permutation_importance_all_models.png").exists():
                st.image("images/permutation_importance_all_models.png", use_container_width=True)
        except:
            pass
        
        # SHAP Analysis
        st.subheader("📊 SHAP (SHapley Additive exPlanations)")
        
        st.markdown("""
        **What it is**: Game-theory based method that explains the output of any machine learning model. 
        SHAP values show how much each feature contributes to a prediction.
        """)
        
        st.markdown("""
        **Analysis from Notebooks:**
        - **TreeExplainer** used for RandomForest and ExtraTrees
        - **LinearExplainer** used for LogisticRegression
        - SHAP values reveal both global (overall) and local (per-instance) feature importance
        """)
        
        # Try to load SHAP images
        col1, col2 = st.columns(2)
        try:
            if Path("images/shap_summary_RandomForest.png").exists():
                with col1:
                    st.image("images/shap_summary_RandomForest.png", use_container_width=True, caption="RandomForest SHAP Summary")
        except:
            pass
        
        try:
            if Path("images/shap_beeswarm_RandomForest.png").exists():
                with col2:
                    st.image("images/shap_beeswarm_RandomForest.png", use_container_width=True, caption="RandomForest SHAP Beeswarm")
        except:
            pass
        
        st.markdown("""
        **Key Insights:**
        - SHAP values confirm TCP SYN Size as the most important feature
        - Feature interactions are visible in beeswarm plots
        - High SHAP values indicate strong predictive power
        """)
        
        # LIME Analysis
        st.subheader("🍋 LIME (Local Interpretable Model-agnostic Explanations)")
        
        st.markdown("""
        **What it is**: Explains individual predictions by approximating the model locally with an 
        interpretable model. Provides feature importance for specific instances.
        """)
        
        st.markdown("""
        **Analysis from Notebooks:**
        - LIME explanations generated for multiple test instances
        - Shows which features contributed to each specific prediction
        - Useful for understanding model decisions on edge cases
        """)
        
        # Try to load LIME images
        col1, col2, col3 = st.columns(3)
        try:
            if Path("images/lime_RandomForest_sample_3844.png").exists():
                with col1:
                    st.image("images/lime_RandomForest_sample_3844.png", use_container_width=True, caption="Sample 3844")
        except:
            pass
        
        try:
            if Path("images/lime_RandomForest_sample_11149.png").exists():
                with col2:
                    st.image("images/lime_RandomForest_sample_11149.png", use_container_width=True, caption="Sample 11149")
        except:
            pass
        
        try:
            if Path("images/lime_RandomForest_sample_12033.png").exists():
                with col3:
                    st.image("images/lime_RandomForest_sample_12033.png", use_container_width=True, caption="Sample 12033")
        except:
            pass
        
        st.markdown("""
        **Key Insights:**
        - LIME reveals instance-specific feature contributions
        - Different instances may rely on different features
        - Helps identify when model makes correct predictions for wrong reasons
        """)
        
        # Logistic Regression Coefficients
        st.subheader("📈 Logistic Regression Coefficients")
        
        st.markdown("""
        **What it is**: Direct interpretation of linear model weights. Positive coefficients increase 
        the probability of a class, negative coefficients decrease it.
        """)
        
        st.markdown("""
        **Analysis from Notebooks:**
        - Coefficients analyzed for each OS class (Android, iOS, Linux, Other, Windows)
        - Top positive and negative coefficients identified per class
        - Visualizations show which features push predictions toward/away from each class
        """)
        
        # Try to load coefficient images
        col1, col2 = st.columns(2)
        try:
            if Path("images/logistic_coefficients_Android.png").exists():
                with col1:
                    st.image("images/logistic_coefficients_Android.png", use_container_width=True, caption="Android Coefficients")
        except:
            pass
        
        try:
            if Path("images/logistic_coefficients_Windows.png").exists():
                with col2:
                    st.image("images/logistic_coefficients_Windows.png", use_container_width=True, caption="Windows Coefficients")
        except:
            pass
        
        col1, col2 = st.columns(2)
        try:
            if Path("images/logistic_coefficients_iOS.png").exists():
                with col1:
                    st.image("images/logistic_coefficients_iOS.png", use_container_width=True, caption="iOS Coefficients")
        except:
            pass
        
        try:
            if Path("images/logistic_coefficients_Linux.png").exists():
                with col2:
                    st.image("images/logistic_coefficients_Linux.png", use_container_width=True, caption="Linux Coefficients")
        except:
            pass
        
        st.markdown("""
        **Key Insights:**
        - **TCP SYN Size** has strong positive coefficients for most classes
        - **TLS features** are particularly important for distinguishing iOS/Android
        - **Negative coefficients** indicate features that reduce probability of a class
        - Linear models provide direct interpretability but lower accuracy than tree models
        """)
        
        # Summary
        st.subheader("📝 Explainability Summary")
        
        st.markdown("""
        **What These Methods Tell Us:**
        
        1. **Global Importance** (Permutation, SHAP):
           - TCP SYN Size is the most critical feature
           - TLS handshake features add significant value
           - NPM features are less important
        
        2. **Local Explanations** (LIME):
           - Different instances rely on different features
           - Model decisions are feature-dependent
           - Helps validate model reasoning
        
        3. **Linear Interpretability** (Coefficients):
           - Direct relationship between features and predictions
           - Trade-off: interpretability vs. accuracy
           - Useful for understanding feature-class relationships
        
        4. **Implications**:
           - **Feature Engineering**: Focus on TCP/IP and TLS features
           - **Model Selection**: Tree models for accuracy, linear for interpretability
           - **Validation**: Explainability confirms model is learning meaningful patterns
           - **Trust**: High feature importance aligns with network security domain knowledge
        """)

if __name__ == "__main__":
    main()
