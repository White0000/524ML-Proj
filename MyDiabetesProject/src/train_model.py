import logging
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import settings

logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

class ModelTrainer:
    def __init__(self, model_type="logistic", search_method="grid", search_iters=20, scoring="accuracy", folds=3):
        self.model_type = model_type
        self.search_method = search_method
        self.search_iters = search_iters
        self.scoring = scoring
        self.folds = folds
        self.random_state = settings.RANDOM_STATE
        self.model_path = settings.MODEL_PATH
        self.param_grids = {
            "logistic": {
                "model": LogisticRegression(max_iter=500, random_state=self.random_state),
                "params": {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["lbfgs", "liblinear"]
                }
            },
            "rf": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "xgb": {
                "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 9],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            },
            "mlp": {
                "model": MLPClassifier(random_state=self.random_state),
                "params": {
                    "hidden_layer_sizes": [(64,), (128,), (128, 64)],
                    "learning_rate_init": [0.001, 0.0001],
                    "max_iter": [200, 300, 500]
                }
            }
        }

    def load_data(self):
        try:
            p = Path(settings.DATA_PATH)
            if not p.exists():
                logger.error(str(p) + " not found")
                return None, None, None, None
            df = pd.read_csv(p)
            if "Outcome" not in df.columns:
                logger.error("No 'Outcome' column found in dataset")
                return None, None, None, None
            x = df.drop("Outcome", axis=1)
            y = df["Outcome"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=settings.TEST_RATIO, random_state=self.random_state, stratify=y)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logger.error(str(e))
            return None, None, None, None

    def run(self):
        start = time.time()
        x_train, x_test, y_train, y_test = self.load_data()
        if x_train is None:
            logger.error("Failed to load training data")
            return
        info = self.param_grids.get(self.model_type, self.param_grids["logistic"])
        if self.search_method == "grid":
            search = GridSearchCV(estimator=info["model"], param_grid=info["params"], scoring=self.scoring, cv=self.folds, n_jobs=-1)
        else:
            search = RandomizedSearchCV(estimator=info["model"], param_distributions=info["params"], scoring=self.scoring, cv=self.folds, n_iter=self.search_iters, n_jobs=-1, random_state=self.random_state)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        train_preds = best.predict(x_train)
        test_preds = best.predict(x_test)
        train_acc = accuracy_score(y_train, train_preds)
        train_prec = precision_score(y_train, train_preds)
        train_rec = recall_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds)
        test_rec = recall_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        data = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "train": {
                "accuracy": float(train_acc),
                "precision": float(train_prec),
                "recall": float(train_rec),
                "f1_score": float(train_f1)
            },
            "test": {
                "accuracy": float(test_acc),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "f1_score": float(test_f1)
            }
        }
        self.save_model(best)
        self.save_metrics(data)
        logger.info(str(data))
        logger.info("Single-model training finished in " + str(round(time.time() - start, 2)) + "s")

    def save_model(self, model):
        p = Path(self.model_path).resolve()
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(p))

    def save_metrics(self, data):
        mp = Path(self.model_path).with_suffix(".metrics.json")
        try:
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(str(e))

class AdvancedEnsembleTrainer:
    def __init__(self, ensemble_type="voting", do_param_search=False, search_method="grid", search_iters=10, random_state=None):
        self.ensemble_type = ensemble_type
        self.do_param_search = do_param_search
        self.search_method = search_method
        self.search_iters = search_iters
        self.random_state = random_state if random_state is not None else settings.RANDOM_STATE
        self.model_path = settings.MODEL_PATH

    def load_data(self):
        p = Path(settings.DATA_PATH)
        if not p.exists():
            logger.error(str(p) + " not found")
            return None, None, None, None
        df = pd.read_csv(p)
        if "Outcome" not in df.columns:
            logger.error("No 'Outcome' column found in dataset")
            return None, None, None, None
        x = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=settings.TEST_RATIO, random_state=self.random_state, stratify=y)
        return x_train, x_test, y_train, y_test

    def small_param_search(self, estimator, param_grid: Dict[str, Any], x_train, y_train):
        if self.search_method == "grid":
            s = GridSearchCV(estimator, param_grid, scoring="f1", cv=3, n_jobs=-1)
        else:
            s = RandomizedSearchCV(estimator, param_grid, scoring="f1", cv=3, n_iter=self.search_iters, n_jobs=-1, random_state=self.random_state)
        s.fit(x_train, y_train)
        return s.best_estimator_

    def train_ensemble(self):
        start = time.time()
        x_train, x_test, y_train, y_test = self.load_data()
        if x_train is None:
            logger.error("Failed to load training data")
            return
        lr = LogisticRegression(max_iter=500, random_state=self.random_state)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=self.random_state)
        mlp = MLPClassifier(random_state=self.random_state, max_iter=300)
        if self.do_param_search:
            lr_params = {
                "C": [0.01, 0.1, 1.0],
                "solver": ["lbfgs", "liblinear"]
            }
            rf_params = {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None]
            }
            xgb_params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6]
            }
            mlp_params = {
                "hidden_layer_sizes": [(64,), (128,)],
                "learning_rate_init": [0.001, 0.0001],
                "max_iter": [200, 300]
            }
            lr = self.small_param_search(lr, lr_params, x_train, y_train)
            rf = self.small_param_search(rf, rf_params, x_train, y_train)
            xgb = self.small_param_search(xgb, xgb_params, x_train, y_train)
            mlp = self.small_param_search(mlp, mlp_params, x_train, y_train)
        if self.ensemble_type.lower() == "voting":
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(
                estimators=[
                    ("lr", lr),
                    ("rf", rf),
                    ("xgb", xgb),
                    ("mlp", mlp)
                ],
                voting="soft"
            )
        elif self.ensemble_type.lower() == "stacking":
            from sklearn.ensemble import StackingClassifier
            ensemble = StackingClassifier(
                estimators=[
                    ("lr", lr),
                    ("rf", rf),
                    ("xgb", xgb)
                ],
                final_estimator=mlp,
                passthrough=False
            )
        else:
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(
                estimators=[
                    ("lr", lr),
                    ("rf", rf),
                    ("xgb", xgb),
                    ("mlp", mlp)
                ],
                voting="soft"
            )
        ensemble.fit(x_train, y_train)
        train_preds = ensemble.predict(x_train)
        test_preds = ensemble.predict(x_test)
        train_acc = accuracy_score(y_train, train_preds)
        train_prec = precision_score(y_train, train_preds)
        train_rec = recall_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds)
        test_rec = recall_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        data = {
            "ensemble_type": self.ensemble_type,
            "param_search": self.do_param_search,
            "train": {
                "accuracy": float(train_acc),
                "precision": float(train_prec),
                "recall": float(train_rec),
                "f1_score": float(train_f1)
            },
            "test": {
                "accuracy": float(test_acc),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "f1_score": float(test_f1)
            }
        }
        self.save_model(ensemble)
        self.save_metrics(data)
        logger.info(str(data))
        logger.info("Ensemble training finished in " + str(round(time.time() - start, 2)) + "s")

    def save_model(self, model):
        p = Path(self.model_path).resolve()
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(p))

    def save_metrics(self, data):
        mp = Path(self.model_path).with_suffix(".metrics.json")
        try:
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(str(e))
