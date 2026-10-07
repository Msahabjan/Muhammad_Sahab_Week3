# Model Error Analysis Report - Week 3

## 1. Hyperparameter Tuning Methodology
We selected **Optuna (Bayesian Optimization)** over GridSearch due to its efficient search space sampling and ability to quickly converge on optimal parameters for LightGBM[cite: 4].

## 2. Cross-Validation Rigour & Variance
- **Method:** 5-Fold Stratified Cross-Validation (to handle class imbalance)[cite: 4].
- **Mean ROC-AUC:** 0.4943
- **Fold Variance:** 0.000613
- **Std Deviation:** 0.0248

## 3. Concrete Failure Patterns Identified
1. **Extreme Class Imbalance Confusion (High False Negatives):** Due to the 92:8 class imbalance, high-risk applicants were occasionally classified as low-risk when external score features (`EXT_SOURCE_2/3`) were missing or near average[cite: 4].
2. **Missing/Unemployed Edge Cases:** Applicants with high `DAYS_EMPLOYED` anomalies caused skewness in tree split points, leading to false positives for low-risk applicants[cite: 4].
3. **Threshold Sensitivity:** The standard 0.5 probability threshold yields high false negatives. Adjusting the classification threshold down to 0.35 significantly improves recall for defaults[cite: 4].

## 4. Addressing Weaknesses
- Applied `is_unbalance=True` directly within LightGBM parameters[cite: 4].
- Used **Stratified K-Fold** to maintain exact class proportions across training and validation splits[cite: 4].
