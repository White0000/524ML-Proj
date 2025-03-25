import logging
import os
import json
import random
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from typing import List
from train_model import AdvancedEnsembleTrainer, ModelTrainer
from evaluate_model import ModelEvaluator
from infer import InferenceService
from config import settings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger("MainApp")
logger.setLevel(logging.INFO)
app = Flask(__name__)
CORS(app)

class PredictData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class RealtimeTrainer:
    def __init__(self, epochs=10, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = settings.RANDOM_STATE
        self.model_path = settings.MODEL_PATH
        self.progress_file = "train_progress.json"

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
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=settings.TEST_RATIO,
            random_state=self.random_state,
            stratify=y
        )
        return x_train, x_test, y_train, y_test

    def train_with_partial_fit(self):
        start = time.time()
        x_train, x_test, y_train, y_test = self.load_data()
        if x_train is None:
            logger.error("Failed to load data")
            return

        classes = np.unique(y_train)
        if len(classes) != 2:
            logger.error("Partial-fit example only for binary classification.")
            return

        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),
            learning_rate_init=0.001,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state
        )

        x_train_arr = x_train.values
        y_train_arr = y_train.values
        n_samples = x_train_arr.shape[0]
        steps_per_epoch = n_samples // self.batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1

        self.clear_progress_file()

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train_arr = x_train_arr[indices]
            y_train_arr = y_train_arr[indices]
            for step in range(steps_per_epoch):
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size
                x_batch = x_train_arr[start_idx:end_idx]
                y_batch = y_train_arr[start_idx:end_idx]
                mlp.partial_fit(x_batch, y_batch, classes=classes)

            train_preds = mlp.predict(x_train)
            test_preds = mlp.predict(x_test)
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            train_f1 = f1_score(y_train, train_preds)
            test_f1 = f1_score(y_test, test_preds)

            info = {
                "epoch": epoch + 1,
                "train_accuracy": float(train_acc),
                "train_f1": float(train_f1),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1)
            }
            self.update_progress_file(info)
            logger.info(f"Epoch {epoch+1}/{self.epochs} -> train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

        joblib.dump(mlp, str(Path(self.model_path).resolve()))
        logger.info("Realtime training finished in " + str(round(time.time() - start, 2)) + "s")

    def clear_progress_file(self):
        f = Path(self.progress_file)
        if f.exists():
            f.unlink()

    def update_progress_file(self, data):
        f = Path(self.progress_file)
        rows = []
        if f.exists():
            with open(f, "r", encoding="utf-8") as fd:
                rows = json.load(fd)
        rows.append(data)
        with open(f, "w", encoding="utf-8") as fd:
            json.dump(rows, fd, indent=2)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(str(e))
    return jsonify({"code": 500, "message": "Server Error", "data": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"code": 200, "message": "OK", "data": {"status": "healthy"}}), 200

@app.route("/train", methods=["POST"])
def train_endpoint():
    try:
        body = request.get_json(force=True) if request.data else {}
        model_type = body.get("model_type", "logistic")
        search_method = body.get("search_method", "grid")
        search_iters = body.get("search_iters", 20)
        do_param_search = bool(body.get("do_param_search", False))
        if model_type.lower() in ["voting", "stacking", "ensemble"]:
            t = AdvancedEnsembleTrainer(
                ensemble_type=model_type.lower(),
                do_param_search=do_param_search,
                search_method=search_method,
                search_iters=search_iters,
                random_state=settings.RANDOM_STATE
            )
            t.train_ensemble()
            return jsonify({
                "code": 200,
                "message": "Ensemble Training Completed",
                "data": {
                    "ensemble_type": model_type.lower(),
                    "do_param_search": do_param_search,
                    "search_method": search_method,
                    "search_iters": search_iters
                }
            }), 200
        else:
            t = ModelTrainer(model_type=model_type, search_method=search_method, search_iters=search_iters)
            t.run()
            return jsonify({
                "code": 200,
                "message": "Training Completed",
                "data": {
                    "model_type": model_type,
                    "search_method": search_method,
                    "search_iters": search_iters
                }
            }), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Training Failed", "data": str(e)}), 500

@app.route("/train-realtime", methods=["POST"])
def train_realtime_endpoint():
    try:
        body = request.get_json(force=True) if request.data else {}
        epochs = body.get("epochs", 10)
        batch_size = body.get("batch_size", 32)
        r = RealtimeTrainer(epochs=epochs, batch_size=batch_size)
        r.train_with_partial_fit()
        return jsonify({"code": 200, "message": "Realtime Training Completed"}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Realtime Training Failed", "data": str(e)}), 500

@app.route("/train-progress", methods=["GET"])
def train_progress_endpoint():
    try:
        p = Path("train_progress.json")
        if not p.exists():
            return jsonify({"progress": []}), 200
        with open(p, "r", encoding="utf-8") as fd:
            data = json.load(fd)
        return jsonify({"progress": data}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Progress Failed", "data": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    try:
        body = request.get_json(force=True) if request.data else {}
        save_report = bool(body.get("save_report", False))
        report_path = body.get("report_path", "evaluation_report.json")
        evaluator = ModelEvaluator()
        result = evaluator.run(save_report=save_report, report_path=report_path)
        return jsonify({"code": 200, "message": "Evaluation Completed", "data": result}), 200
    except Exception as ex:
        logger.error(str(ex))
        return jsonify({"code": 500, "message": "Evaluation Failed", "data": str(ex)}), 500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"code": 400, "message": "Invalid request body", "data": {}}), 400
        srv = InferenceService()
        if isinstance(data, dict):
            parsed = PredictData(**data)
            result = srv.predict_single(parsed.dict())
            return jsonify({"code": 200, "message": "Success", "data": result}), 200
        if isinstance(data, list):
            validated = []
            for item in data:
                validated.append(PredictData(**item).dict())
            result = srv.predict_batch(validated)
            return jsonify({"code": 200, "message": "Success", "data": result}), 200
        return jsonify({"code": 400, "message": "Input must be a dict or list of dicts", "data": {}}), 400
    except ValidationError as ve:
        logger.error(str(ve))
        return jsonify({"code": 422, "message": "Validation Error", "data": ve.errors()}), 422
    except Exception as err:
        logger.error(str(err))
        return jsonify({"code": 500, "message": "Prediction Failed", "data": str(err)}), 500

@app.route("/explain", methods=["POST"])
def explain_endpoint():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"code": 400, "message": "Invalid request body", "data": {}}), 400
        srv = InferenceService()
        if isinstance(data, dict):
            parsed = PredictData(**data)
            explanation = srv.explain_single(parsed.dict())
            return jsonify({"code": 200, "message": "Success", "data": explanation}), 200
        if isinstance(data, list):
            validated = []
            for item in data:
                validated.append(PredictData(**item).dict())
            explanation = srv.explain_batch(validated)
            return jsonify({"code": 200, "message": "Success", "data": explanation}), 200
        return jsonify({"code": 400, "message": "Input must be a dict or list of dicts", "data": {}}), 400
    except ValidationError as ve:
        logger.error(str(ve))
        return jsonify({"code": 422, "message": "Validation Error", "data": ve.errors()}), 422
    except Exception as ex:
        logger.error(str(ex))
        return jsonify({"code": 500, "message": "Explain Failed", "data": str(ex)}), 500

@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    try:
        p = Path(settings.MODEL_PATH).with_suffix(".metrics.json")
        if not p.exists():
            return jsonify({"code": 404, "message": "Metrics Not Found", "data": {}}), 404
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"code": 200, "message": "Metrics Retrieved", "data": data}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Metrics Failed", "data": str(e)}), 500

@app.route("/config", methods=["GET"])
def config_endpoint():
    return jsonify({
        "code": 200,
        "message": "Config",
        "data": {
            "PROJECT_NAME": settings.PROJECT_NAME,
            "BASE_DIR": settings.BASE_DIR,
            "DATA_PATH": settings.DATA_PATH,
            "MODEL_PATH": settings.MODEL_PATH,
            "TEST_RATIO": settings.TEST_RATIO,
            "RANDOM_STATE": settings.RANDOM_STATE,
            "BATCH_SIZE": settings.BATCH_SIZE,
            "EPOCHS": settings.EPOCHS,
            "LEARNING_RATE": settings.LEARNING_RATE,
            "ENV_NAME": getattr(settings, "ENV_NAME", "development"),
            "LOG_LEVEL": settings.LOG_LEVEL,
            "MODEL_VERSION": settings.MODEL_VERSION,
            "SERVICE_URL": settings.SERVICE_URL
        }
    }), 200

@app.route("/models", methods=["GET"])
def list_models_endpoint():
    try:
        folder = Path(settings.MODEL_PATH).parent
        files = list(folder.glob("*.pkl"))
        models = [f.name for f in files]
        return jsonify({"code": 200, "message": "Models Listed", "data": models}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "List Models Failed", "data": str(e)}), 500

@app.route("/download-model", methods=["GET"])
def download_model_endpoint():
    try:
        model_file = request.args.get("file")
        if not model_file:
            return jsonify({"code": 400, "message": "No model file specified", "data": {}}), 400
        folder = Path(settings.MODEL_PATH).parent
        path = folder.joinpath(model_file).resolve()
        if not path.exists():
            return jsonify({"code": 404, "message": "Model File Not Found", "data": {}}), 404
        return send_file(str(path), as_attachment=True)
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Download Failed", "data": str(e)}), 500

@app.route("/switch-model", methods=["POST"])
def switch_model_endpoint():
    try:
        body = request.get_json(force=True) if request.data else {}
        file = body.get("model_file")
        if not file:
            return jsonify({"code": 400, "message": "No model file specified", "data": {}}), 400
        folder = Path(settings.MODEL_PATH).parent
        path = folder.joinpath(file).resolve()
        if not path.exists():
            return jsonify({"code": 404, "message": "Specified model file not found", "data": {}}), 404
        settings.MODEL_PATH = str(path)
        return jsonify({
            "code": 200,
            "message": "Model Switched",
            "data": {"model_path": settings.MODEL_PATH}
        }), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Switch Model Failed", "data": str(e)}), 500

@app.route("/dashboard-data", methods=["GET"])
def dashboard_data_endpoint():
    try:
        p = Path("dashboard_metrics.json").resolve()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data), 200
        return jsonify({
            "glucoseTrends": [110, 120, 130, 140, 125],
            "outcomeDistribution": [40, 60],
            "barMetrics": [45, 52, 61, 37, 72]
        }), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/monitor", methods=["GET"])
def monitor_endpoint():
    try:
        data = {
            "timestamp": int(time.time() * 1000),
            "current_value": random.randint(80, 150)
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = os.getenv("PORT", "5000")
    debug_mode = str(os.getenv("FLASK_DEBUG", "true")).lower() == "true"
    app.run(host="0.0.0.0", port=int(port), debug=debug_mode)
