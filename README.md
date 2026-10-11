# Diabetes Prediction Model 🩺

An interactive machine learning dashboard that estimates diabetes risk from patient health measurements. Built with **Python**, **Scikit-learn**, and **Streamlit**, the app combines a prediction interface with model evaluation visualizations.

**Live app:** https://diabetes-prediction-model-xv.streamlit.app/  
**Source code:** https://github.com/Ajay9760/Diabetes-Prediction-Model

> **Disclaimer:** This project is for educational and informational purposes only. Its predictions are not a medical diagnosis and should not be used as a substitute for professional medical advice.

## Features

- **Interactive risk prediction:** Adjust patient measurements in the sidebar and run a prediction.
- **Probability output:** Displays the model's estimated probability for the positive class.
- **Model analytics dashboard:** Shows reported evaluation metrics, Logistic Regression coefficients, and a confusion-matrix visualization.
- **Pre-trained assets:** Loads a saved model and fitted scaler from local files.
- **Streamlit interface:** Provides a browser-based app with two tabs for predictions and analytics.

## Tech Stack

- **Language:** Python
- **Web app:** Streamlit
- **Machine learning:** Scikit-learn (Logistic Regression)
- **Data processing:** Pandas, NumPy
- **Model serialization:** Joblib
- **Visualizations:** Matplotlib, Seaborn

## Project Structure

```text
Diabetes-Prediction-Model/
├── app.py                 # Streamlit application
├── diabetes (1).csv       # Dataset used by the project
├── diabetes_model.pkl     # Serialized trained model
├── scaler.pkl             # Fitted feature scaler
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Input Features

The model accepts eight measurements:

| Feature | Description | Unit / Format |
|---|---|---|
| Pregnancies | Number of pregnancies | Count |
| Glucose | Glucose concentration | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skinfold thickness | mm |
| Insulin | Serum insulin level | μU/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Diabetes pedigree function score | Score |
| Age | Patient age | Years |

The application builds the input row using this feature order, applies the saved scaler, and passes the transformed values to the saved model. Keep the training and inference feature order and preprocessing consistent.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Ajay9760/Diabetes-Prediction-Model.git
cd Diabetes-Prediction-Model
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal (typically http://localhost:8501). Open it in your browser.

Make sure the following files remain in the project directory, since the app loads them using relative paths:

- `diabetes_model.pkl`
- `scaler.pkl`
- `diabetes (1).csv`

## Using the Dashboard

1. Open the app and adjust the patient parameters in the sidebar.
2. Select **Run Risk Prediction** in the **Real-Time Prediction** tab.
3. Review the predicted class and the model's estimated probability.
4. Open **Model Evaluation & Insights** to view the displayed metrics, Logistic Regression coefficients, and confusion matrix.

## Model Evaluation

The current dashboard displays the following values:

| Metric | Value shown in the app |
|---|---:|
| Test accuracy | 73.38% |
| ROC-AUC | 0.79 |
| Precision (positive class) | 62% |
| Recall (positive class) | 57% |

The app also displays a confusion matrix and Logistic Regression coefficients. These values are documented as currently shown in the code; they have not been independently re-evaluated as part of this README. For a rigorous evaluation, calculate all metrics from a held-out test set and ensure the confusion matrix matches that same test set.

## Repository Files

- `app.py` — dashboard interface, input handling, prediction flow, metrics, and plots.
- `diabetes (1).csv` — project dataset.
- `diabetes_model.pkl` — serialized model loaded for prediction.
- `scaler.pkl` — fitted scaler applied to input measurements.
- `requirements.txt` — Python package dependencies.

## Limitations

- This is a demonstration project, not a clinically validated diagnostic system.
- Model performance depends on the original training data, preprocessing, decision threshold, and evaluation methodology.
- A prediction probability is a model output and should not automatically be interpreted as an individual's medically calibrated risk.
- Pickle/joblib model files should only be loaded from trusted sources.

## Future Improvements

- Add reproducible training and evaluation scripts.
- Generate evaluation metrics and the confusion matrix dynamically from a documented test set rather than hard-coding them.
- Add cross-validation, calibration analysis, and threshold selection focused on sensitivity and specificity.
- Include input validation and clearer guidance for missing or physiologically implausible measurements.
- Track model/data versions and document dataset provenance.

## Author

**A. Ajay Kumar**  
GitHub: [@Ajay9760](https://github.com/Ajay9760)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
