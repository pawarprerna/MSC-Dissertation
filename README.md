# Interpretable Anomaly Detection in High-Frequency Trading

This repository contains the implementation and dissertation for a research project on **anomaly detection in high-frequency trading (HFT)** using a **hybrid LSTM–GNN deep learning model** with **SHAP-based interpretability**.

## 📄 Overview
High-frequency trading generates thousands of trades per second, creating both opportunities and systemic risks. Detecting anomalies in such environments requires capturing:
- **Temporal patterns** (using LSTMs)  
- **Structural dependencies** (using GNNs)  

This project integrates both modalities into a **hybrid LSTM–GNN architecture**, enhancing anomaly detection accuracy while ensuring interpretability through SHAP.

---

## 📂 Repository Contents
- `Dissertation.pdf` – Full dissertation report describing the research, methodology, results, and analysis.  
- `Dissertation Code.ipynb` – Jupyter Notebook containing the complete implementation:  
  - Data preprocessing (Nifty50 stock dataset from Kaggle)  
  - LSTM, GNN, and Hybrid architectures (PyTorch + PyTorch Geometric)  
  - Training and evaluation pipeline (accuracy, F1, AUROC, AP)  
  - SHAP-based interpretability analysis  
- `requirements.txt` – Python dependencies (PyTorch, PyTorch Geometric, scikit-learn, SHAP, etc.)  

---


