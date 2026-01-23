# KI-Fraud-Detection

A Machine Learning project for detecting fraudulent transactions on an e-commerce and payment platform.
Developed as part of the course **Foundations of Artificial Intelligence** at **TU Clausthal**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Dataset](#dataset)
* [Project Pipeline](#project-pipeline)
* [Technologies Used](#technologies-used)
* [Project Structure](#project-structure)
* [Current Status](#current-status)
* [Installation](#installation)
* [License](#license)

---

## Project Overview

Fraud detection is a critical task in modern financial and e-commerce systems.
This project aims to develop a **Machine Learning-based fraud detection system** capable of identifying fraudulent transactions using historical transaction data.

The focus of this project is on applying **classical machine learning techniques**, combined with proper data preprocessing and evaluation strategies, to address the challenges of fraud detection such as **class imbalance** and **feature complexity**.

---

## Objectives

The main objectives of this project are:

* Understand and preprocess real-world transaction data
* Perform exploratory data analysis (EDA)
* Engineer relevant features for fraud detection
* Train and evaluate multiple machine learning models
* Compare models using fraud-relevant evaluation metrics
* Interpret results and discuss limitations and future improvements

---

## Dataset

The dataset used in this project comes from the **Xente Fraud Detection Challenge** hosted on **Zindi**.

**Dataset characteristics:**

* Approximately 140,000 transactions
* Combination of numerical and categorical features
* Highly imbalanced target variable (fraud vs. non-fraud)

⚠️ **Note:**
Due to licensing restrictions, the dataset is **not included** in this repository.
Only code, documentation, and results are provided.

---

## Project Pipeline

The project follows a standard Machine Learning workflow:

1. Load raw data
2. Data cleaning and preprocessing
3. Feature engineering
4. Exploratory data analysis (EDA)
5. Model training
6. Hyperparameter tuning
7. Model evaluation
8. Results interpretation
9. Conclusion and future work

---

## Technologies Used

### Programming Language

* Python

### Libraries & Tools

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* imbalanced-learn
* jupyter

---

## Project Structure

```text
KI-Fraud-Detection/
│
├── assets/                   # Static and non-experimental resources
│   └── logos/
│
├── data/                     # Data directory (not tracked)
│   ├── raw/                  # Original dataset (not included)
│   └── processed/            # Cleaned and preprocessed data
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb      # Model training & tuning
│   └── 04_evaluation.ipynb    # Model evaluation & metrics
│
├── results/
│   ├── figures/               # Visual results
│   └── metrics/               # Numerical evaluation results
│
├── src/                      # Reusable Python modules
│   ├── data/                 # Data loading and preprocessing logic
│   │   └── preprocessing.py
│   ├── features/             # Feature engineering functions
│   │   └── features.py
│   ├── models/               # Model training and inference
│   │   └── train_model.py
│   └── evaluation/           # Evaluation metrics and plots
│       └── evaluation.py
├── .gitignore                # Files and folders to ignore
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```


---

## Current Status

🚧 **Project in early development stage**

* Repository structure initialized
* Dataset description and documentation completed
* Data preprocessing, modeling, and evaluation to be implemented

This repository will be updated progressively as the project advances.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Bernardtouck/KI-Fraud-Detection.git
cd KI-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## License

This project is developed for **academic purposes only**.
The dataset is the property of **Zindi** and must not be redistributed.
Only source code and results are publicly available.
