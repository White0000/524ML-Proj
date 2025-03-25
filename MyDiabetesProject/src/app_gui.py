import sys
import logging
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QComboBox, QSpinBox,
    QTabWidget, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
from infer import InferenceService
from config import settings

logger = logging.getLogger("PyQtApp")
logger.setLevel(logging.INFO)

class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.info_label = QLabel("Welcome to the Diabetes Detection Dashboard")
        self.metrics_box = QTextEdit()
        self.metrics_box.setReadOnly(True)
        self.refresh_button = QPushButton("Refresh Metrics")
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.refresh_button)
        self.layout.addWidget(self.metrics_box)
        self.setLayout(self.layout)
        self.refresh_button.clicked.connect(self.load_metrics)

    def load_metrics(self):
        p = os.path.splitext(settings.MODEL_PATH)[0] + ".metrics.json"
        if not os.path.exists(p):
            self.metrics_box.setPlainText("No metrics file found. Train a model first.")
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                self.metrics_box.setPlainText(f.read())
        except Exception as e:
            self.metrics_box.setPlainText(str(e))

class DataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load CSV")
        self.clean_button = QPushButton("Clean & Preview")
        self.table = QTableWidget()
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.clean_button)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)
        self.file_path = None
        self.load_button.clicked.connect(self.select_file)
        self.clean_button.clicked.connect(self.clean_preview)

    def select_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv);;All Files (*)")
        if f:
            self.file_path = f
            self.preview_file()

    def preview_file(self):
        if not self.file_path:
            return
        try:
            df = pd.read_csv(self.file_path)
            self.show_table(df)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def clean_preview(self):
        if not self.file_path:
            QMessageBox.information(self, "Info", "No file selected")
            return
        try:
            from data_preprocessing import DataPreprocessor
            dp = DataPreprocessor(data_path=self.file_path)
            x_train, x_test, y_train, y_test = dp.run()
            if x_train is None or y_train is None:
                QMessageBox.warning(self, "Warning", "Data preprocessing failed.")
                return
            df_train = x_train.copy()
            df_train["Outcome"] = y_train.values
            self.show_table(df_train.head(50))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_table(self, df):
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = str(df.iloc[i, j])
                item = QTableWidgetItem(val)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

class ModelTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.form_layout = QGridLayout()
        self.buttons_layout = QHBoxLayout()
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.input_fields = {}
        self.model_type_box = QComboBox()
        self.search_method_box = QComboBox()
        self.search_iters_box = QSpinBox()
        self.data_labels = [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ]
        self.init_ui()
        self.setLayout(self.layout)

    def init_ui(self):
        for i, text in enumerate(self.data_labels):
            lbl = QLabel(text)
            ln = QLineEdit()
            self.input_fields[text] = ln
            self.form_layout.addWidget(lbl, i, 0)
            self.form_layout.addWidget(ln, i, 1)
        self.model_type_box.addItems(["logistic", "rf", "xgb", "mlp"])
        self.search_method_box.addItems(["grid", "random"])
        self.search_iters_box.setRange(1, 1000)
        self.search_iters_box.setValue(20)
        self.form_layout.addWidget(QLabel("Model Type"), len(self.data_labels), 0)
        self.form_layout.addWidget(self.model_type_box, len(self.data_labels), 1)
        self.form_layout.addWidget(QLabel("Search Method"), len(self.data_labels) + 1, 0)
        self.form_layout.addWidget(self.search_method_box, len(self.data_labels) + 1, 1)
        self.form_layout.addWidget(QLabel("Search Iters"), len(self.data_labels) + 2, 0)
        self.form_layout.addWidget(self.search_iters_box, len(self.data_labels) + 2, 1)
        b_train = QPushButton("Train")
        b_eval = QPushButton("Evaluate")
        b_pred = QPushButton("Predict")
        b_explain = QPushButton("Explain")
        b_train.clicked.connect(self.train_model)
        b_eval.clicked.connect(self.evaluate_model)
        b_pred.clicked.connect(self.predict)
        b_explain.clicked.connect(self.explain)
        self.buttons_layout.addWidget(b_train)
        self.buttons_layout.addWidget(b_eval)
        self.buttons_layout.addWidget(b_pred)
        self.buttons_layout.addWidget(b_explain)
        self.layout.addLayout(self.form_layout)
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.result_box)

    def train_model(self):
        t = self.model_type_box.currentText()
        s = self.search_method_box.currentText()
        i = self.search_iters_box.value()
        ModelTrainer(model_type=t, search_method=s, search_iters=i).run()
        self.result_box.append("Training completed")

    def evaluate_model(self):
        ModelEvaluator().run()
        self.result_box.append("Evaluation completed, check logs or saved report")

    def predict(self):
        d = {}
        for k, f in self.input_fields.items():
            v = f.text().strip()
            try:
                if k in ["BMI", "DiabetesPedigreeFunction"]:
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except:
                d[k] = 0
        r = InferenceService().predict_single(d)
        self.result_box.append("Prediction: " + str(r.get("prediction")) + ", Probability: " + str(r.get("probability")))

    def explain(self):
        d = {}
        for k, f in self.input_fields.items():
            v = f.text().strip()
            try:
                if k in ["BMI", "DiabetesPedigreeFunction"]:
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except:
                d[k] = 0
        r = InferenceService().explain_single(d)
        self.result_box.append("Explanation: " + str(r))

class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.info_label = QLabel("Settings and Environment")
        self.config_box = QTextEdit()
        self.config_box.setReadOnly(True)
        self.refresh_button = QPushButton("Show Config")
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.refresh_button)
        self.layout.addWidget(self.config_box)
        self.setLayout(self.layout)
        self.refresh_button.clicked.connect(self.load_config)

    def load_config(self):
        d = {
            "PROJECT_NAME": settings.PROJECT_NAME,
            "BASE_DIR": settings.BASE_DIR,
            "DATA_PATH": settings.DATA_PATH,
            "MODEL_PATH": settings.MODEL_PATH,
            "TEST_RATIO": settings.TEST_RATIO,
            "RANDOM_STATE": settings.RANDOM_STATE,
            "BATCH_SIZE": settings.BATCH_SIZE,
            "EPOCHS": settings.EPOCHS,
            "LEARNING_RATE": settings.LEARNING_RATE
        }
        self.config_box.setPlainText(str(d))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Detection System")
        self.tabs = QTabWidget()
        self.dashboard_tab = DashboardTab()
        self.data_tab = DataTab()
        self.model_tab = ModelTab()
        self.settings_tab = SettingsTab()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(self.tabs)
        self.resize(1000, 600)

def main():
    a = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(a.exec_())

if __name__ == "__main__":
    main()
