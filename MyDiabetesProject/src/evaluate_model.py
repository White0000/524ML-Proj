import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from config import settings

logger = logging.getLogger("ModelEvaluation")
logger.setLevel(logging.INFO)

class ModelEvaluator:
    def __init__(self, model_path=None, enable_calibration=False, cv_folds=5):
        self.model_path = model_path if model_path else settings.MODEL_PATH
        self.enable_calibration = enable_calibration
        self.cv_folds = cv_folds

    def run(self, save_report=False, report_path="evaluation_report.json"):
        x_train, x_test, y_train, y_test = self.load_data()
        if x_train is None:
            logger.error("Failed to load evaluation data")
            return
        model = self.load_model()
        if not model:
            logger.error("Failed to load model")
            return
        if self.enable_calibration and hasattr(model, "predict_proba"):
            model = self.calibrate_model(model, x_train, y_train)
        preds_train = model.predict(x_train)
        preds_test = model.predict(x_test)
        results = {
            "train": self.evaluate(x_train, y_train, preds_train, model),
            "test": self.evaluate(x_test, y_test, preds_test, model)
        }
        if save_report:
            self.save_report(results, report_path)
        return results

    def load_data(self):
        try:
            path = Path(settings.DATA_PATH)
            if not path.exists():
                logger.error(str(path) + " not found")
                return None, None, None, None
            df = pd.read_csv(path)
            if "Outcome" not in df.columns:
                logger.error("No 'Outcome' column found in dataset")
                return None, None, None, None
            x = df.drop("Outcome", axis=1)
            y = df["Outcome"]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=settings.TEST_RATIO,
                random_state=settings.RANDOM_STATE,
                stratify=y
            )
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logger.error(str(e))
            return None, None, None, None

    def load_model(self):
        p = Path(self.model_path)
        if not p.exists():
            logger.error(str(p) + " not found")
            return None
        try:
            return joblib.load(str(p))
        except Exception as e:
            logger.error(str(e))
            return None

    def calibrate_model(self, model, x_train, y_train):
        c = CalibratedClassifierCV(base_estimator=model, cv=self.cv_folds)
        c.fit(x_train, y_train)
        return c

    def evaluate(self, x, y_true, y_pred, model):
        d = {}
        d["accuracy"] = float(accuracy_score(y_true, y_pred))
        d["precision"] = float(precision_score(y_true, y_pred))
        d["recall"] = float(recall_score(y_true, y_pred))
        d["f1_score"] = float(f1_score(y_true, y_pred))
        try:
            d["auc"] = float(roc_auc_score(y_true, y_pred))
        except:
            d["auc"] = None
        d["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        d["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        d["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        d["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
                d["roc_auc_score_proba"] = float(roc_auc_score(y_true, proba[:, 1]))
                d["average_precision"] = float(average_precision_score(y_true, proba[:, 1]))
                d["brier_score"] = float(brier_score_loss(y_true, proba[:, 1]))
            else:
                d["roc_auc_score_proba"] = None
                d["average_precision"] = None
                d["brier_score"] = None
        except:
            d["roc_auc_score_proba"] = None
            d["average_precision"] = None
            d["brier_score"] = None
        return d

    def save_report(self, data, path):
        p = Path(path).resolve()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(str(e))

if __name__ == "__main__":
    ModelEvaluator(enable_calibration=True, cv_folds=3).run(save_report=True)
