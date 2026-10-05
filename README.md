# AI Contract Risk & Compliance Screening Dashboard

An AI-assisted contract screening system that classifies business/legal documents into **Low Risk, Medium Risk, or High Risk** and routes low-confidence predictions for **manual review**.

The project combines CUAD contract data, custom weighted contractual risk scoring, TF-IDF, Logistic Regression, confidence-based review routing, explainability, and a simple Streamlit dashboard.

> **Disclaimer:** This project is for AI-assisted risk screening and decision support. It is not legal advice and should not replace professional legal review.

## 1. Project Objective

The system is designed to:
1. Extract contract text.
2. Use CUAD clause categories to create a project-defined risk score.
3. Create Low / Medium / High Risk labels.
4. Train a TF-IDF + Logistic Regression classifier.
5. Predict contract risk and confidence.
6. Route low-confidence predictions to manual review.
7. Explain predictions using important text features and CUAD risk factors.
8. Provide a simple Streamlit dashboard for PDF analysis.

## 2. Dataset

The project uses the **CUAD (Contract Understanding Atticus Dataset)**.

Project statistics:
- **510 contracts**
- **41 CUAD categories**
- **28,031 flattened QA/category rows**

CUAD does **not** provide official Low / Medium / High Risk labels. Therefore, this project creates custom labels from CUAD category presence and project-defined risk weights.

The raw CUAD JSON files are intentionally excluded from GitHub and should be obtained separately.

## 3. Custom Risk Scoring

Selected CUAD categories are assigned project-defined weights from 0 to 5.

Examples:

| Category | Weight |
|---|---:|
| Liquidated Damages | 5 |
| Uncapped Liability | 5 |
| Non-Compete | 4 |
| IP Ownership Assignment | 4 |
| Joint IP Ownership | 4 |
| Irrevocable / Perpetual License | 4 |
| Exclusivity | 3 |
| Minimum Commitment | 3 |
| Cap On Liability | 3 |
| Audit Rights | 2 |
| License Grant | 2 |
| Governing Law | 1 |
| Agreement Date | 0 |
| Document Name | 0 |
| Parties | 0 |

Risk labels:
- **Low Risk:** 0–9
- **Medium Risk:** 10–31
- **High Risk:** 32+

Class distribution:
- Low Risk: **133 (26.08%)**
- Medium Risk: **252 (49.41%)**
- High Risk: **125 (24.51%)**

> These weights and thresholds are project-defined heuristics, not official CUAD risk ratings.

## 4. Text Processing

Conservative preprocessing:
- lowercase text
- normalize non-breaking spaces and tabs
- normalize repeated whitespace
- remove long separator patterns

Legal negations and potentially meaningful legal wording were not aggressively removed.

## 5. Machine Learning

### TF-IDF
- n-grams: `(1, 2)`
- max features: `20,000`
- `min_df = 2`
- `max_df = 0.95`
- `sublinear_tf = True`

### Logistic Regression
- `max_iter = 2000`
- `class_weight = "balanced"`
- `random_state = 42`

Classes:
- High Risk
- Low Risk
- Medium Risk

## 6. Evaluation

Data split:
- Training: **408 contracts**
- Test: **102 contracts**

A separate validation split was used for confidence-threshold selection so the final test set remained untouched for that decision.

### Final test performance

**Accuracy: 76.47%**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| High Risk | 0.76 | 0.88 | 0.81 |
| Low Risk | 0.84 | 0.59 | 0.70 |
| Medium Risk | 0.74 | 0.80 | 0.77 |
| **Macro F1** | | | **0.76** |

The main confusion was between Low Risk and Medium Risk.

## 7. Confidence-Based Manual Review

Selected confidence threshold:

**0.60**

Validation results at 0.60:
- Automatic decisions: **18 / 82**
- Manual reviews: **64 / 82**
- Automatic coverage: **21.95%**
- Automatic-subset accuracy: **94.44%**

Untouched final test set:
- Automatic decisions: **25 / 102**
- Manual reviews: **77 / 102**
- Automatic coverage: **24.51%**
- Manual review: **75.49%**
- Automatic-subset accuracy: **92.00%**

**Important:** 92% is the accuracy of the automatically accepted subset, not the overall model accuracy. Overall test accuracy remains **76.47%**.

## 8. Explainability

The project has two complementary explanation layers.

### ML explanation

For a predicted class, local feature contribution is calculated from:

`TF-IDF value × class coefficient`

Example High Risk signals included:
- `territory`
- `commercialization`
- `royalty`
- `net sales`
- `patent`
- `indemnitee`
- `clinical`
- `milestone`

### CUAD explanation

The system also reports detected weighted CUAD risk factors.

Example: **Array BioPharma Inc. - LICENSE, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT**
- ML prediction: **High Risk**
- Confidence: **86.52%**
- CUAD risk score: **49**
- CUAD label: **High Risk**

Important factors included:
- Liquidated Damages — 5
- Irrevocable / Perpetual License — 4
- Non-Compete — 4
- Joint IP Ownership — 4
- IP Ownership Assignment — 4
- Minimum Commitment — 3
- Revenue / Profit Sharing — 3
- Exclusivity — 3
- Cap On Liability — 3

The ML and CUAD explanations are separate layers and should not be interpreted as the same thing.

## 9. Streamlit Dashboard

`app.py` provides a deliberately simple dashboard with:
- PDF upload
- PDF text extraction
- Risk prediction
- Confidence score
- Automatic / Manual Review decision
- Class probabilities
- Top supporting ML features
- Flagged/manual-review document list
- Session history

Run:

```bash
streamlit run app.py
```

The current dashboard supports selectable/text PDFs. Scanned image-only PDFs require OCR, which is not included in this simple version.

## 10. Project Structure

```text
AI_DOCUMENT/
├── DC.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── CUADv1.json                    # excluded from Git
├── train.json                     # excluded from Git
├── test.json                      # excluded from Git
├── train_separate_questions.json  # excluded from Git
│
├── final_model.pkl                # excluded from Git
├── final_tfidf.pkl                # excluded from Git
├── risk_threshold.pkl             # excluded from Git
├── risk_weights.pkl               # excluded from Git
├── model_metadata.pkl             # excluded from Git
│
├── feature_importance.csv
├── final_test_predictions.csv
├── validation_threshold_results.csv
├── array_biopharma_local_explanation.csv
├── array_biopharma_cuad_risk_factors.csv
└── sample_high_risk_contract.pdf
```

## 11. Installation

Create/activate a Python environment and install:

```bash
pip install -r requirements.txt
```

If `pypdf` is missing:

```bash
pip install pypdf
```

Run the dashboard:

```bash
streamlit run app.py
```

## 12. Reproducing the Model

1. Obtain the CUAD dataset.
2. Place the required JSON file in the project directory.
3. Open `DC.ipynb`.
4. Run the notebook sections in order.
5. Generate the model artifacts.
6. Run `streamlit run app.py`.

The notebook contains dataset preparation, custom risk scoring, text cleaning, train/validation/test splitting, TF-IDF, model comparison, threshold selection, final evaluation, explainability, artifact saving, and sanity testing.

## 13. Model Artifacts

The following are generated locally and excluded from GitHub:

```text
final_model.pkl
final_tfidf.pkl
risk_threshold.pkl
risk_weights.pkl
model_metadata.pkl
```

They can be regenerated by running the notebook.

## 14. Limitations

1. CUAD does not provide official Low / Medium / High Risk labels.
2. Risk labels are based on project-defined heuristic weights.
3. The classifier learns statistical associations from the available contracts.
4. Some learned features can be dataset/document-type specific.
5. High confidence does not guarantee that a contract is legally safe.
6. The system should not replace qualified legal review.
7. The current Streamlit app does not perform OCR.
8. The system is intended for screening and prioritization, not autonomous legal decisions.

## 15. Key Results

| Metric | Result |
|---|---:|
| Contracts | 510 |
| CUAD categories | 41 |
| Training samples | 408 |
| Test samples | 102 |
| Test accuracy | **76.47%** |
| Manual-review threshold | **0.60** |
| Test automatic coverage | **24.51%** |
| Test manual review | **75.49%** |
| Automatic-subset accuracy | **92.00%** |

## 16. Future Improvements

- OCR support for scanned PDFs
- Better clause-level extraction
- More robust risk-label methodology
- Transformer/BERT-based classification
- Clause-specific explanations
- Persistent document database
- Human reviewer feedback loop
- Model calibration and threshold optimization
- Deployment to Streamlit Cloud or another hosting platform

## 17. Conclusion

This project demonstrates an end-to-end AI-assisted contract screening workflow combining machine learning, rule-based contractual risk scoring, confidence-based manual review, and explainability.

The design intentionally keeps a **human-in-the-loop** rather than treating model predictions as autonomous legal decisions.


## Done By :
Vishal Kirad and Satvik Aggarwal


Artificial Intelligence Project
