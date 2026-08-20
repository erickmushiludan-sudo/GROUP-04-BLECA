"""
Group 4 - Student Dropout Prediction
Minimal Flask deployment app.

Run with:
    python app/app.py
Then open http://127.0.0.1:5000 in a browser.

IMPORTANT: the model only accepts the 4 raw, leakage-free features
(attendance, study_hours, assignment_score, lms_activity).
`academic_index` is intentionally NOT an input — it was identified as
target leakage during the investigation (see notebooks/group04_project.ipynb).
"""
from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURES = ["attendance", "study_hours", "assignment_score", "lms_activity"]

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
    <title>Student Dropout Risk Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 480px; margin: 40px auto; }
        label { display: block; margin-top: 12px; font-weight: bold; }
        input { width: 100%; padding: 6px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px 16px; }
        .result { margin-top: 20px; padding: 12px; border-radius: 6px; }
        .risk { background: #fdecea; color: #b71c1c; }
        .safe { background: #e8f5e9; color: #1b5e20; }
    </style>
</head>
<body>
    <h2>Student Dropout Risk Predictor</h2>
    <p>Group 4 &mdash; Machine Learning Weekend Task</p>
    <form method="POST">
        <label>Attendance (%)</label>
        <input type="number" step="any" name="attendance" value="{{ vals.attendance }}" required>
        <label>Study hours / week</label>
        <input type="number" step="any" name="study_hours" value="{{ vals.study_hours }}" required>
        <label>Assignment score (%)</label>
        <input type="number" step="any" name="assignment_score" value="{{ vals.assignment_score }}" required>
        <label>LMS activity (login/interaction count)</label>
        <input type="number" step="any" name="lms_activity" value="{{ vals.lms_activity }}" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result {{ 'risk' if prediction == 1 else 'safe' }}">
        <strong>Prediction:</strong> {{ 'At risk of dropout' if prediction == 1 else 'Not at risk' }}<br>
        <strong>Probability of dropout:</strong> {{ '%.2f'|format(proba*100) }}%
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    vals = {"attendance": 75, "study_hours": 8, "assignment_score": 70, "lms_activity": 300}
    prediction, proba = None, None
    if request.method == "POST":
        vals = {f: float(request.form[f]) for f in FEATURES}
        X = np.array([[vals[f] for f in FEATURES]])
        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
    return render_template_string(PAGE, vals=vals, prediction=prediction, proba=proba)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    X = np.array([[data[f] for f in FEATURES]])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return jsonify({"dropout_prediction": pred, "dropout_probability": proba})


if __name__ == "__main__":
    app.run(debug=True)
