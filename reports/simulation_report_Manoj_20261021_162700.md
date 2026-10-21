# Federated Learning Simulation Report for **Manoj**

**Date**: 2026-10-21 16:27:05
**Task Type**: Classification
**Target**: `Predicted Disease`
**Clients**: 5 (Distribution: iid)

## 1. Strategy Comparison Summary (The 'Ups and Downs')
Based on the captured metrics, here is an analysis of the federated learning algorithms:

- **FedAvg**: The standard baseline. Usually provides steady convergence but can be volatile if client data is highly skewed (non-IID).
- **FedProx**: Introduces a proximal term to constrain local updates. **Ups:** Offers superior stability and prevents the global model from diverging in highly heterogeneous (non-IID) health datasets. **Downs:** Can converge slightly slower than FedAvg initially.
- **FedAdam**: Uses adaptive server-side momentum. **Ups:** Can provide much faster initial convergence and reach a better optimum. **Downs:** Highly sensitive to the learning rate and hyperparameter tuning; can overshoot if not configured perfectly.

## 2. Final Results Table

| Algorithm | Final Acc | Final F1 | Final AUC | Sensitivity | Specificity | Final Loss |
|-----------|-----------|----------|-----------|-------------|-------------|------------|
| **FedAvg** | 93.64% | 0.9363 | 0.9953 | 0.9365 | 0.0000 | 0.1626 |
| **FedProx** | 93.58% | 0.9357 | 0.9953 | 0.9359 | 0.0000 | 0.1643 |
| **FedAdam** | 78.82% | 0.7710 | 0.9889 | 0.7900 | 0.0000 | 0.6610 |

## 3. Visualizations

![Strategy Comparison](C:\Users\Manoj\Desktop\projects\FedLearning\reports\strategy_comparison_Manoj_20261021_162700.png)

![Model Comparison](C:\Users\Manoj\Desktop\projects\FedLearning\reports\model_comparison_Manoj_20261021_162700.png)

