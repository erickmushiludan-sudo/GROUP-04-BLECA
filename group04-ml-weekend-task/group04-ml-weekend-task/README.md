# Group 4 — Student Dropout Prediction
ML Weekend Task — Synthetic Data Investigation

## 1. Business Problem
A university wants to identify students who are at risk of dropping out, using
routinely-collected data: attendance, study hours, assignment scores, and LMS
(learning management system) activity.

- **Task type:** Binary classification
- **Target:** `dropout` (1 = at risk, 0 = not at risk) — perfectly balanced (2500 / 2500)
- **n = 5000** synthetic students, generated with `np.random.seed(88)`

## 2. Key Finding: Target Leakage in `academic_index`
The raw dataset ships a fifth column, `academic_index`, which looks like a normal
feature but is actually the exact linear formula used to build the label:

```
academic_index = 0.04*attendance + 0.10*study_hours + 0.03*assignment_score + 0.0005*lms_activity
dropout = 1 if academic_index < median(academic_index) else 0
```

**Proof:** a single threshold rule on `academic_index` alone predicts `dropout`
with **100% accuracy**. Including it in modeling means the model is reading the
answer key, not learning a pattern — the textbook definition of target leakage.

**Action taken:** `academic_index` was **excluded** from all final models. Only
`attendance`, `study_hours`, `assignment_score`, and `lms_activity` are used.

## 3. A Second, More Subtle Finding
Even after removing `academic_index`, models still reach **92–100% accuracy**
(Logistic Regression tuned, `C=100`, reaches 100% on the held-out test set with
zero misclassifications). This is *not* a leakage bug — it's because the
synthetic generator adds **no random noise** to the label formula, so the four
raw features are, by construction, an almost perfectly deterministic predictor
of the label. Real student data would contain unmeasured factors (health,
finances, motivation, etc.) that add noise a model can never fully explain,
so this level of accuracy should **not** be expected on real data. This is
documented as a core limitation, not hidden.

## 4. Modeling Summary

| Experiment | Best model | Test accuracy | Test F1 | ROC-AUC | 5-fold CV |
|---|---|---|---|---|---|
| A — with `academic_index` (leaky) | Random Forest / Gradient Boosting | 1.000 | 1.000 | 1.000 | 0.9998 ± 0.0004 |
| B — without `academic_index` (corrected) | Logistic Regression (tuned, C=100) | 1.000 | 1.000 | 1.000 | 0.998 (CV F1) |

Four models were trained and compared in both experiments: Logistic Regression,
Random Forest, Gradient Boosting, and KNN. Full training/validation/test
metrics, cross-validation, and error analysis are in the notebook.

**Final model chosen: Logistic Regression** (tuned via `GridSearchCV`, `C=100`).
It matches the ceiling performance of the tree ensembles on the corrected
feature set but does so with a fully interpretable, linear decision boundary
and no train/CV gap — appropriate given the (synthetically) linear nature of
the true relationship, and easier to audit/explain to university stakeholders
than a black-box ensemble.

## 5. Model Interpretation
All four coefficients are negative — higher attendance, study hours,
assignment scores, and LMS activity all reduce predicted dropout risk, which
matches domain intuition and how the labels were constructed.

## 6. Limitations
- Synthetic, noise-free labels make reported accuracy unrealistically high;
  real-world performance would be materially lower.
- Only four features — no demographic, financial, or prior-academic-history
  context that a real dropout model would typically include.
- Single random seed / single synthetic sample — not validated against any
  real student population.
- The deployment app is a development-only Flask server, not a production
  WSGI/HTTPS deployment.

## 7. Project Structure
```
group04-ml-weekend-task/
├── data/
│   ├── raw/group4_dataset.csv          # generated dataset (reproducible)
│   └── processed/                      # post-inspection copy
├── notebooks/
│   ├── group04_project.ipynb           # full investigation & modeling notebook
│   └── figures/                        # EDA and leakage-evidence plots
├── src/
│   └── data_generation.py              # unmodified generation code
├── models/
│   └── best_model.joblib               # final tuned Logistic Regression pipeline
├── app/
│   └── app.py                          # Flask deployment app (web form + /api/predict)
├── screenshots/
│   ├── app_home.png
│   └── app_prediction_result.png
├── requirements.txt
└── README.md
```

## 8. How to Reproduce
```bash
pip install -r requirements.txt
python src/data_generation.py          # regenerate data/raw/group4_dataset.csv
jupyter notebook notebooks/group04_project.ipynb   # full investigation
python app/app.py                      # run the deployment app at http://127.0.0.1:5000
```

## 9. API
`POST /api/predict` with JSON body:
```json
{"attendance": 45, "study_hours": 2, "assignment_score": 35, "lms_activity": 60}
```
Returns:
```json
{"dropout_prediction": 1, "dropout_probability": 0.999...}
```

## 10. Answers to the Final Submission Questions
1. **Initial observation:** every model, including a trivial threshold rule on
   one column, hit 97–100% accuracy immediately.
2. **Trustworthy?** No — accuracy that high on a "real-world-style" problem is
   a red flag, not a success.
3. **Investigation performed:** tested whether `academic_index` alone could
   predict the target (it could, perfectly), then compared full-feature vs.
   leak-removed experiments across four models with cross-validation.
4. **Actual cause:** `academic_index` is the exact pre-thresholded formula
   used to generate the label (target leakage).
5. **Proof:** a single-variable threshold rule on `academic_index` scores
   100% accuracy against `dropout`.
6. **What changed after fixing it:** accuracy dropped modestly (to 92–100%
   depending on model) but stayed high, because the label itself is
   noise-free by construction — a second, distinct finding documented above.
7. **Necessary preprocessing:** removal of `academic_index`; standard scaling
   of the remaining four features.
8. **Final model:** tuned Logistic Regression (`C=100`).
9. **Why:** matches the best achievable performance with full interpretability
   and no overfitting gap.
10. **Generalization testing:** stratified 5-fold cross-validation plus a
    held-out 20% test set, both consistent with training performance.
11. **Leakage investigation method:** single-feature threshold test on
    `academic_index`, then a controlled leaky-vs-corrected experiment.
12. **Main limitations:** noise-free synthetic labels, narrow feature set,
    single seed, dev-only deployment (see Section 6).
13. **Production improvements:** collect real, noisy student data; add more
    feature types (demographics, prior performance, financial aid status);
    monitor for concept drift each term; deploy behind a production WSGI
    server with authentication and logging; recalibrate thresholds based on
    the real cost of false negatives vs. false positives.
