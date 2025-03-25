# 524ML-Proj
# Diabetes Detection with Ensemble Voting

This repository contains a comprehensive system for predicting diabetes risk based on clinical features and an ensemble voting strategy. It showcases the entire workflow from data preprocessing and model training (including multiple algorithms) to real-time training visualization and a user-facing web interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Realtime Training](#realtime-training)
- [Web Interface](#web-interface)
- [API Endpoints](#api-endpoints)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The objective of this project is to detect the risk of diabetes using an **Ensemble Voting** approach, combining multiple models such as Logistic Regression, Random Forest, XGBoost, and MLP. By leveraging their collective predictions, the system enhances overall robustness and accuracy compared to any single algorithm.

---

## Features

1. **Multiple ML Algorithms**  
   - Logistic Regression, Random Forest, XGBoost, MLP  
   - Hyperparameter search using GridSearchCV or RandomizedSearchCV

2. **Ensemble Voting**  
   - Soft-voting classifier that aggregates model outputs  
   - Automatically selects optimized sub-models from the training phase

3. **Realtime Training (Optional)**  
   - Demonstrates a partial-fit approach (e.g., MLP)  
   - Epoch-by-epoch progress logged and available for frontend visualization

4. **RESTful API**  
   - Powered by Flask for model training, evaluation, and predictions  
   - Easily integrable with any front-end or third-party service

5. **User-Friendly Web Interface**  
   - React-based client for inputting patient data  
   - Model selection, training configuration, and real-time training monitor

---

## Project Structure

```
/data
  └── diabetes.csv           # (Example or placeholder dataset)
  
/src
  ├── train_model.py         # ModelTrainer and AdvancedEnsembleTrainer classes
  ├── evaluate_model.py      # ModelEvaluator for metrics and reports
  ├── infer.py               # InferenceService for single/batch predictions
  ├── main_app.py            # Flask server defining all routes
  └── config.py              # Settings & environment configuration

/frontend
  └── src
      ├── Home.tsx           # Main React component with UI for model training, real-time progress, etc.
      ├── ...
      └── index.tsx          # Entry point for the React app

requirements.txt             # Python dependencies
package.json                # Front-end dependencies
README.md                   # Project documentation (this file)
```

---

## Installation

1. **Clone the repository**  
   ```bash
   https://github.com/White0000/524ML-Proj.git
   cd diabetes-ensemble
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Install front-end dependencies** (if applicable)  
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables**  
   - Modify or create `.env` for Python settings  
   - Adjust any front-end `.env` if needed

---

## Usage

1. **Run the Flask backend**  
   ```bash
   cd src
   python main_app.py
   ```
   By default, it listens on `http://localhost:5000`.

2. **Run the React front-end**  
   ```bash
   cd frontend
   npm start
   ```
   By default, it opens `http://localhost:3000` in your browser.

---

## Model Training

- **Single-model training:**  
  The system supports training one algorithm at a time by specifying `model_type` (`logistic`, `rf`, `xgb`, `mlp`) and search settings (`search_method`, `search_iters`).
- **Ensemble Voting:**  
  If `model_type` is set to something like `"voting"` or `"stacking"`, the `AdvancedEnsembleTrainer` merges multiple optimized models into a single ensemble, typically improving performance.

**Example**:  
```bash
POST /train
{
  "model_type": "voting",
  "search_method": "random",
  "search_iters": 30,
  "do_param_search": true
}
```

---

## Realtime Training

- **Partial-fit approach**  
  An optional demonstration using MLP partial fitting.  
  - Allows epoch-by-epoch updates  
  - Tracks metrics and writes them to `train_progress.json`  

**Usage**:  
```bash
POST /train-realtime
{
  "epochs": 10,
  "batch_size": 32
}
```
Then poll `/train-progress` to retrieve progress data for visualization.

---

## Web Interface

- **Data input**:  
  Users can enter or adjust patient features (e.g., Glucose, BMI) and click **Predict**.
- **Model selection**:  
  A dropdown for `model_type` to specify single model vs. ensemble.
- **Search method**:  
  Choose between `grid` or `random` for hyperparameter tuning.
- **Realtime chart**:  
  Displays partial-fit training curves for accuracy or F1 over epochs.

---

## API Endpoints

- **`POST /train`**  
  Triggers single or ensemble training.  
  - Body parameters: `model_type`, `search_method`, `search_iters`, etc.
- **`POST /evaluate`**  
  Evaluates the current model, can save a JSON report.
- **`POST /predict`**  
  Predicts for single or multiple patient data.
- **`POST /train-realtime`**  
  Starts partial-fit training for real-time updates.
- **`GET /train-progress`**  
  Fetches the current training progress (epoch-wise metrics).
- **`GET /metrics`**  
  Retrieves final saved metrics from `.metrics.json`.

---

## Future Work

- **Deploy to Cloud**  
  Containerize with Docker, set up on a cloud provider.  
- **Advanced Visualization**  
  Possibly integrate Plotly or ECharts for more interactive real-time charts.
- **Additional Models**  
  Could experiment with LightGBM or TabNet for further variety in the ensemble.
- **Data Augmentation or Feature Engineering**  
  Explore domain-specific transformations, polynomial features, or synthetic data generation.

---

## License

MIT License
