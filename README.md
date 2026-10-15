# Industrial Machine Condition Classification

## 1. Project Overview

This repository contains the complete, production-grade machine learning solution for **Industrial Machine Condition Classification**. The objective is to accurately predict the operational health condition of industrial machinery based on multivariate physical and electrical sensor telemetry.

Predicting machine degradation prior to catastrophic failure enables condition-based preventive maintenance, substantially reducing unplanned industrial downtime, hazard risks, and machinery destruction.

The task is formulated as a **3-class multiclass classification** problem:

- **`0 — Normal`**: Machine operating within standard nominal tolerances (healthy baseline).
- **`1 — Thermal Failure`**: Failure caused by excessive process temperature, heat buildup, and coolant deficiency.
- **`2 — Mechanical Failure`**: Failure driven by bearing fatigue, component wear, high torque overload, and dynamic vibration.

---

## 2. Dataset Overview

- **Training Set (`student_dataset.csv`)**: 16,000 labeled machine observations.
- **Test Set (`test.csv`)**: 4,000 unseen machine observations for leaderboard evaluation.
- **Submission Format (`sample_submission.csv`)**: Prescribed evaluation format (`unit_id,condition`).

### Class Distribution (16,000 Training Records)

- **Class 0 (Normal)**: 8,000 samples (**50.00%**)
- **Class 1 (Thermal Failure)**: 4,000 samples (**25.00%**)
- **Class 2 (Mechanical Failure)**: 4,000 samples (**25.00%**)

The dataset features an exact 2:1:1 balanced failure structure where healthy baseline units constitute 50% of the operational history, and thermal and mechanical degradation modes each comprise 25%.

### Input Features (12 Raw Telemetry Attributes)

- **Identifier**: `unit_id` (Unique machine serial tag, excluded from training).
- **Categorical Feature**: `unit_type` (Machine structural category: `S` = Small [45.1%], `M` = Medium [34.6%], `H` = Heavy-Duty [20.1%]).
- **Thermal Sensors**: `ambient_temp` (°C), `process_temp` (°C), `coolant_flow` (L/min), `thermal_stress`.
- **Kinematic & Mechanical Sensors**: `shaft_speed` (RPM), `drive_torque` (Nm), `power_load` (kW), `component_wear` (min), `mechanical_load`, `vibration_index` (mm/s).

---

## 3. Quick Start: Run on Google Colab

The pipeline is packaged into a clean, self-contained Jupyter Notebook: **`Machine_Condition_Classification.ipynb`**.

### How to Run in Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `Machine_Condition_Classification.ipynb`.
3. In the Colab file manager (left sidebar), upload:
   - `train.csv`
   - `test.csv`
4. Run all cells (`Runtime` &rarr; `Run all` or `Ctrl + F9`).
5. The notebook will automatically:
   - Install necessary gradient boosting libraries (`lightgbm`, `xgboost`, `catboost`).
   - Perform EDA and physical feature engineering.
   - Run 5-fold cross-validation across all supervised models and baselines.
   - Optimize blending weights and calibration multipliers.
   - Generate and verify Kaggle `submission.csv`.
   - Output complete classification reports, confusion matrices, and metrics.

---

## 4. Quick Start: Run Locally

You can run the notebook locally using Jupyter Notebook, Jupyter Lab, or VS Code:

```bash
pip install lightgbm xgboost catboost scikit-learn matplotlib seaborn
jupyter notebook Machine_Condition_Classification.ipynb
```

---

## 5. Pipeline Architecture

1. **Leak-Free Modular Preprocessing (`CleanMachinePreprocessor`)**:
   - Zero-leakage median imputation for continuous sensors and mode imputation for `unit_type`, computed strictly per training fold.
   - Fold-wise `StandardScaler` for linear and generative Gaussian models (Logistic Regression, Bayes, QDA, Naive Bayes).
   - Numerical $\epsilon = 10^{-4}$ safeguard against division-by-zero, plus infinity sanitization.
2. **Physics & Domain Feature Engineering (35 Engineered Features)**:
   - **Thermodynamic Balance**: Temperature gradient ($\Delta T = \text{process\_temp} - \text{ambient\_temp}$) and cooling dissipation ratio ($\frac{\text{power\_load}}{\text{coolant\_flow} + \epsilon}$).
   - **Non-Linear Boundary Interactions**: $\text{shaft\_speed} \times \text{vibration\_index}$, $\text{drive\_torque} \times \text{component\_wear}$, and orthogonal discriminator $\text{thermal\_stress} - \text{vibration\_index}$.
   - **Rotational Dynamics & 3-Way Tensors**: Centrifugal force, rotational kinetic energy, friction heat flux, and shock wave energy.
3. **Simulation Noise Down-Weighting**:
   - Identified 647 synthetic simulation noise samples where out-of-fold predictions strongly contradicted the noisy label ($P(\text{incorrect}) > 0.80$).
   - Down-weighted samples during training (`sample_weight = 0.25`), preventing corrupted gradient steps.
4. **Validation Protocol**:
   - 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

---

## 6. Model Benchmark & Results

All models were evaluated across 5 stratified folds (16,000 total out-of-fold observations):

### Summary Table

| Model                                                     | Best Score Achieved  |
| --------------------------------------------------------- | -------------------- |
| **Calibrated Multi-Model Ensemble + TTA**                 | **89.28% (0.89275)** |
| **XGBoost Classifier**                                    | **89.20% (0.89200)** |
| **Top-3 Equal Ensemble (LGBM + XGB + CatBoost)**          | **89.18% (0.89181)** |
| **CatBoost Classifier**                                   | **89.16% (0.89156)** |
| **LightGBM Classifier**                                   | **88.99% (0.88994)** |
| **Random Forest Classifier**                              | **88.69% (0.88688)** |
| **Logistic Regression**                                   | **85.79% (0.85788)** |
| **Bayes Classifier (Linear Discriminant Analysis - LDA)** | **84.50% (0.84500)** |
| **Quadratic Bayes Classifier (QDA)**                      | **83.88% (0.83881)** |
| **Naive Bayes Classifier (Gaussian NB)**                  | **79.44% (0.79438)** |

### Comprehensive Metrics Breakdown

| Model                         | Accuracy   | Balanced Acc | Macro F1   | Weighted F1 | Log Loss   | Cohen Kappa | MCC        |
| ----------------------------- | ---------- | ------------ | ---------- | ----------- | ---------- | ----------- | ---------- |
| **Calibrated Ensemble + TTA** | **89.28%** | **87.40%**   | **88.32%** | **89.19%**  | **0.3112** | **0.8260**  | **0.8273** |
| **XGBoost**                   | 89.20%     | 87.34%       | 88.24%     | 89.11%      | 0.3125     | 0.8249      | 0.8261     |
| **Top-3 Equal Ensemble**      | 89.18%     | 87.31%       | 88.21%     | 89.09%      | 0.3105     | 0.8246      | 0.8258     |
| **CatBoost**                  | 89.16%     | 87.30%       | 88.17%     | 89.07%      | 0.3126     | 0.8242      | 0.8254     |
| **LightGBM**                  | 88.99%     | 87.09%       | 88.00%     | 88.90%      | 0.3143     | 0.8215      | 0.8227     |
| **Random Forest**             | 88.69%     | 86.78%       | 87.62%     | 88.60%      | 0.3329     | 0.8167      | 0.8178     |
| **Logistic Regression**       | 85.79%     | 83.25%       | 84.18%     | 85.62%      | 0.3935     | 0.7691      | 0.7706     |
| **Bayes Classifier (LDA)**    | 84.50%     | 81.21%       | 82.59%     | 84.21%      | 0.4251     | 0.7460      | 0.7497     |
| **Quadratic Bayes (QDA)**     | 83.88%     | 81.06%       | 81.97%     | 83.69%      | 0.8523     | 0.7380      | 0.7402     |
| **Naive Bayes (Gaussian NB)** | 79.44%     | 76.67%       | 76.93%     | 79.34%      | 1.9622     | 0.6692      | 0.6694     |

---

## 7. Best Model & Leaderboard Performance

The winning predictive architecture is the **Calibrated Multi-Model Ensemble with Test-Time Augmentation (TTA)**:

- **Optimal Blending Weights**:
  - **CatBoost**: 35.31%
  - **XGBoost**: 32.73%
  - **LightGBM**: 31.96%
  - _Baselines & Random Forest_: 0.00% (excluded to prevent score dilution).
- **Normal-Protection Calibration**: Multipliers ($M_{\text{Normal}}=1.000, M_{\text{Thermal}}=0.9127, M_{\text{Mechanical}}=1.0200$) curtailed false mechanical alarms while ensuring healthy machine retention.
- **Test-Time Augmentation**: 5 micro-jitter passes smoothed predictions in high-density transition regions.
- **Public Leaderboard Score**: Top competitive score on the public leaderboard (~0.893).
- **Kaggle Submission (`submission.csv`)**: 4,000 rows, 0 nulls, formatted as `unit_id,condition` (Normal: 53.15%, Thermal: 23.18%, Mechanical: 23.68%).

---
