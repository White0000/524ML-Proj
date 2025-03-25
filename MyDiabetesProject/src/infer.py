import logging
import pathlib
import joblib
import pandas as pd
import numpy as np
from typing import Union, List
from pydantic import BaseModel
from config import settings

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger("ModelInference")
logger.setLevel(logging.INFO)

class DiabetesFeatures(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class InferenceService:
    def __init__(self, model_path: str = None, shap_method: str = "auto"):
        self.model_path = model_path if model_path else settings.MODEL_PATH
        self.shap_method = shap_method
        self.load_model()
        self.explainer = None
        if shap:
            try:
                self.prepare_explainer()
            except:
                pass

    def load_model(self):
        p = pathlib.Path(self.model_path).resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        self.model = joblib.load(str(p))

    def prepare_explainer(self):
        if hasattr(self.model, "predict_proba"):
            if self.shap_method == "auto":
                self.explainer = shap.Explainer(self.model)
            elif self.shap_method == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.sample_background())
            else:
                self.explainer = shap.Explainer(self.model)

    def sample_background(self, n=50):
        p = pathlib.Path(settings.DATA_PATH)
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p).dropna()
            if len(df) > n:
                df = df.sample(n, random_state=settings.RANDOM_STATE)
            if "Outcome" in df.columns:
                df = df.drop("Outcome", axis=1)
            return df
        except:
            return None

    def validate_input(self, data: Union[dict, List[dict]]) -> pd.DataFrame:
        if isinstance(data, dict):
            d = DiabetesFeatures(**data).dict()
            return pd.DataFrame([d])
        if isinstance(data, list):
            r = []
            for i in data:
                r.append(DiabetesFeatures(**i).dict())
            return pd.DataFrame(r)
        raise TypeError("Invalid input type")

    def predict_single(self, data: dict):
        df = self.validate_input(data)
        p = self.model.predict(df)
        prob = self.model.predict_proba(df)
        i = int(p[0])
        v = float(prob[0][i])
        return {"prediction": i, "probability": v}

    def predict_batch(self, data_list: List[dict]):
        df = self.validate_input(data_list)
        p = self.model.predict(df)
        prob = self.model.predict_proba(df)
        res = []
        for i in range(len(df)):
            lbl = int(p[i])
            pv = float(prob[i][lbl])
            res.append({"prediction": lbl, "probability": pv})
        return res

    def explain_single(self, data: dict):
        if not self.explainer:
            return None
        df = self.validate_input(data)
        v = self.explainer(df)
        if hasattr(v, "values"):
            return v.values[0].tolist()
        return None

    def explain_batch(self, data_list: List[dict]):
        if not self.explainer:
            return None
        df = self.validate_input(data_list)
        v = self.explainer(df)
        r = []
        if hasattr(v, "values"):
            for i in range(len(df)):
                r.append(v.values[i].tolist())
        return r

if __name__ == "__main__":
    s = {
        "Pregnancies": 3,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 23,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 32
    }
    svc = InferenceService()
    r1 = svc.predict_single(s)
    logger.info(str(r1))
    r2 = svc.predict_batch([s, s])
    logger.info(str(r2))
    e1 = svc.explain_single(s)
    logger.info(str(e1))
    e2 = svc.explain_batch([s, s])
    logger.info(str(e2))
