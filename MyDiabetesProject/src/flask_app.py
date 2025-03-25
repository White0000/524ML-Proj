import logging
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from typing import List
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
from infer import InferenceService
from config import settings

logger = logging.getLogger("FlaskApp")
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
        b = request.get_json(force=True) if request.data else {}
        t = b.get("model_type", "logistic")
        sm = b.get("search_method", "grid")
        it = b.get("search_iters", 20)
        m = ModelTrainer(model_type=t, search_method=sm, search_iters=it)
        m.run()
        return jsonify({"code": 200, "message": "Training Completed", "data": {"model_type": t, "search_method": sm}}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Training Failed", "data": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    try:
        b = request.get_json(force=True) if request.data else {}
        s = bool(b.get("save_report", False))
        r = b.get("report_path", "evaluation_report.json")
        e = ModelEvaluator()
        d = e.run(save_report=s, report_path=r)
        return jsonify({"code": 200, "message": "Evaluation Completed", "data": d}), 200
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
            p = PredictData(**data)
            r = srv.predict_single(p.dict())
            return jsonify({"code": 200, "message": "Success", "data": r}), 200
        elif isinstance(data, list):
            v = []
            for i in data:
                v.append(PredictData(**i).dict())
            r = srv.predict_batch(v)
            return jsonify({"code": 200, "message": "Success", "data": r}), 200
        else:
            return jsonify({"code": 400, "message": "Input must be a dict or list of dicts", "data": {}}), 400
    except ValidationError as ve:
        logger.error(str(ve))
        return jsonify({"code": 422, "message": "Validation Error", "data": ve.errors()}), 422
    except Exception as ex:
        logger.error(str(ex))
        return jsonify({"code": 500, "message": "Prediction Failed", "data": str(ex)}), 500

@app.route("/explain", methods=["POST"])
def explain_endpoint():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"code": 400, "message": "Invalid request body", "data": {}}), 400
        srv = InferenceService()
        if isinstance(data, dict):
            p = PredictData(**data)
            r = srv.explain_single(p.dict())
            return jsonify({"code": 200, "message": "Success", "data": r}), 200
        elif isinstance(data, list):
            v = []
            for i in data:
                v.append(PredictData(**i).dict())
            r = srv.explain_batch(v)
            return jsonify({"code": 200, "message": "Success", "data": r}), 200
        else:
            return jsonify({"code": 400, "message": "Input must be a dict or list of dicts", "data": {}}), 400
    except ValidationError as ve:
        logger.error(str(ve))
        return jsonify({"code": 422, "message": "Validation Error", "data": ve.errors()}), 422
    except Exception as ex:
        logger.error(str(ex))
        return jsonify({"code": 500, "message": "Explain Failed", "data": str(ex)}), 500

@app.route("/config", methods=["GET"])
def config_endpoint():
    return jsonify({"code": 200, "message": "Config", "data": {
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
    }}), 200

@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    try:
        p = Path(settings.MODEL_PATH).with_suffix(".metrics.json")
        if not p.exists():
            return jsonify({"code": 404, "message": "Metrics Not Found", "data": {}}), 404
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return jsonify({"code": 200, "message": "Metrics Retrieved", "data": d}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Metrics Failed", "data": str(e)}), 500

@app.route("/models", methods=["GET"])
def list_models_endpoint():
    try:
        folder = Path(settings.MODEL_PATH).parent
        files = list(folder.glob("*.pkl"))
        m = []
        for f in files:
            m.append(f.name)
        return jsonify({"code": 200, "message": "Models Listed", "data": m}), 200
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
        p = folder.joinpath(model_file).resolve()
        if not p.exists():
            return jsonify({"code": 404, "message": "Model File Not Found", "data": {}}), 404
        return send_file(str(p), as_attachment=True)
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Download Failed", "data": str(e)}), 500

@app.route("/switch-model", methods=["POST"])
def switch_model_endpoint():
    try:
        b = request.get_json(force=True) if request.data else {}
        file = b.get("model_file")
        if not file:
            return jsonify({"code": 400, "message": "No model file specified", "data": {}}), 400
        folder = Path(settings.MODEL_PATH).parent
        p = folder.joinpath(file).resolve()
        if not p.exists():
            return jsonify({"code": 404, "message": "Specified model file not found", "data": {}}), 404
        settings.MODEL_PATH = str(p)
        return jsonify({"code": 200, "message": "Model Switched", "data": {"model_path": settings.MODEL_PATH}}), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"code": 500, "message": "Switch Model Failed", "data": str(e)}), 500

@app.route("/dashboard-data", methods=["GET"])
def dashboard_data_endpoint():
    try:
        # 这里示例从本地文件中读取, 或者返回默认值
        p = Path("dashboard_metrics.json").resolve()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data), 200
        return jsonify({
            "glucoseTrends": [120, 130, 110, 140, 135],
            "outcomeDistribution": [35, 65],
            "barMetrics": [40, 55, 60, 35, 70]
        }), 200
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    pt = os.getenv("PORT", 5000)
    db = str(os.getenv("FLASK_DEBUG", "true")).lower() == "true"
    app.run(host="0.0.0.0", port=pt, debug=db)
