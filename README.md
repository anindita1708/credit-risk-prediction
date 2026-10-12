# Credit Risk Prediction & Explainable AI

An end-to-end machine learning project that predicts the likelihood of payment difficulties for loan applicants using the Home Credit Default Risk dataset. The project combines data analysis, feature engineering, model evaluation, SHAP-based explainability, threshold tuning, a FastAPI prediction service, and an interactive Streamlit dashboard.

The goal is to demonstrate how machine learning can support credit-risk assessment by providing risk estimates and interpretable model explanations for human review.

## Project Overview

Financial institutions need reliable ways to identify applicants who may experience difficulties repaying loans. This project explores supervised machine learning techniques to estimate payment-difficulty risk from applicant information.

The solution includes:

* Exploratory Data Analysis (EDA) to understand applicant characteristics and target imbalance.
* Data preprocessing using imputation, scaling, and categorical encoding.
* Feature engineering to create financial and demographic ratios.
* Model comparison using Logistic Regression, Random Forest, and XGBoost.
* SHAP explainability to examine factors contributing to individual predictions.
* Decision-threshold tuning to balance precision and recall.
* A FastAPI service for model inference.
* A Streamlit dashboard for interactive predictions and explanations.
* Automated tests using pytest.

## Business Problem

Loan repayment difficulties can expose financial institutions to credit losses. A predictive model can help identify applications that may require additional review.

This project addresses three questions:

1. Can applicant information help distinguish between applicants with and without recorded payment difficulties?
2. Which machine learning model provides the strongest predictive performance on the validation data?
3. How can model predictions be explained to help a human reviewer understand the factors influencing an estimate?

**Important:** The model produces estimates, not definitive judgments about an applicant's ability or willingness to repay. It is an educational portfolio project and is not validated or approved for real-world lending decisions.

## Tech Stack

| Area                  | Technologies                           |
| --------------------- | -------------------------------------- |
| Programming           | Python                                 |
| Data analysis         | Pandas, NumPy                          |
| Visualization         | Matplotlib, Seaborn, Plotly            |
| Machine learning      | Scikit-learn, XGBoost                  |
| Explainable AI        | SHAP                                   |
| API development       | FastAPI, Pydantic, Uvicorn             |
| Interactive dashboard | Streamlit                              |
| Model persistence     | Joblib, Pickle                         |
| Testing               | Pytest                                 |
| Development tools     | Jupyter Notebook, VS Code, Git, GitHub |

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

The project currently uses the `application_train.csv` dataset, which contains approximately 307,511 records and 122 columns.

The target variable is `TARGET`:

* `0` — No recorded payment difficulties.
* `1` — Payment difficulties recorded.

The target is imbalanced, so evaluation considers more than accuracy alone.

The current implementation uses the application training table. Other Home Credit tables, such as previous applications, bureau records, and payment histories, are potential extensions.

**Dataset access:** The raw dataset is not included in this repository. Download it from Kaggle and place `application_train.csv` in `data/raw/`.

## Project Architecture

```text
Home Credit Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Data Preprocessing
(Imputation, Scaling, Encoding)
        |
        v
Feature Engineering
        |
        v
Model Training & Evaluation
        |
        v
XGBoost Model Selection
        |
        v
SHAP Explainability
        |
        v
Validation-Based Threshold Tuning
        |
        v
FastAPI Prediction Service
        |
        v
Streamlit Interactive Dashboard
        |
        v
Risk Estimate & Explanations
```

### Prediction workflow

1. Applicant information is collected through the dashboard.
2. The dashboard sends the applicant data to the FastAPI endpoint.
3. The inference pipeline creates the engineered features and applies the saved preprocessor.
4. The trained XGBoost model estimates the probability of payment difficulties.
5. The configured threshold is applied to produce a binary prediction.
6. SHAP values identify features that contribute positively or negatively to the model's risk estimate.
7. The dashboard presents the result and explanatory factors.

## Repository Structure

```text
credit-risk-prediction/
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost.pkl
│   ├── feature_preprocessor.pkl
│   └── threshold_config.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The repository structure represents the project organization. Some scripts may serve as reusable implementations of steps explored in notebooks. Confirm that every listed file exists in your final repository before publishing.

## Exploratory Data Analysis

EDA was performed to understand the data distribution, missing values, class imbalance, and relationships between applicant characteristics and payment difficulties.

Areas examined include:

* Target distribution and class imbalance.
* Missing values and duplicate records.
* Numerical and categorical feature distributions.
* Applicant age and employment duration.
* Income, credit amount, and annuity distributions.
* Education and income categories.
* External credit scores.
* Correlations and outliers.

One important data-quality observation was the anomalous `DAYS_EMPLOYED` value of `365243`. This value was treated as a placeholder: an anomaly indicator was created and the anomalous value was replaced with a missing value before deriving employment duration.

## Feature Engineering

Additional features were created to represent applicant characteristics and financial ratios:

| Feature                    | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `AGE_YEARS`                | Approximate age in years                              |
| `EMPLOYMENT_YEARS`         | Approximate employment duration in years              |
| `EMPLOYED_ANOMALY`         | Indicator for the anomalous employment-duration value |
| `CREDIT_INCOME_RATIO`      | Credit amount divided by total income                 |
| `ANNUITY_INCOME_RATIO`     | Loan annuity divided by total income                  |
| `CREDIT_GOODS_RATIO`       | Credit amount divided by goods price                  |
| `INCOME_PER_FAMILY_MEMBER` | Total income divided by family size                   |
| `CHILDREN_RATIO`           | Number of children divided by family size             |

Infinite ratio values are replaced with missing values so that preprocessing can handle them.

## Data Preprocessing

The preprocessing pipeline includes:

* Median imputation for numerical features.
* Standard scaling for numerical features.
* Most-frequent imputation for categorical features.
* One-hot encoding of categorical variables.
* Stratified train-validation splitting.

The preprocessor is fitted on the training data and then applied to the validation data. This helps prevent information leakage from the validation set into preprocessing.

The target variable and applicant identifier `SK_ID_CURR` are excluded from the model features.

## Model Training & Evaluation

Three classification models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

The reported metrics were obtained on the validation split. Precision, recall, and F1 use the classification threshold applied during the initial model comparison.

### Validation results

| Model               |      ROC-AUC |       PR-AUC | Precision |   Recall |     F1-score |
| ------------------- | -----------: | -----------: | --------: | -------: | -----------: |
| Logistic Regression |     0.749202 |     0.228309 |  0.161692 | 0.677543 |     0.261079 |
| Random Forest       |     0.735885 |     0.216059 |  0.184012 | 0.520645 |     0.271919 |
| **XGBoost**         | **0.761247** | **0.252586** |  0.171736 | 0.672105 | **0.273569** |

### Why XGBoost?

XGBoost achieved the highest ROC-AUC, PR-AUC, and F1-score among these three models under the reported evaluation setup.

* **ROC-AUC** measures how well the model ranks positive cases above negative cases across thresholds.
* **PR-AUC** summarizes precision-recall performance and is especially informative for an imbalanced target.
* **Precision** measures the proportion of predicted positive cases that are actually positive.
* **Recall** measures the proportion of actual positive cases identified by the model.
* **F1-score** is the harmonic mean of precision and recall.

The results show that identifying payment difficulties involves a trade-off between finding more positive cases and limiting false positives. Model selection should ultimately depend on validated business requirements, costs, fairness, and operational constraints—not a single metric.

## Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) was used to interpret the trained XGBoost model.

The global feature-importance analysis highlighted features including:

* `EXT_SOURCE_3`
* `EXT_SOURCE_2`
* `EXT_SOURCE_1`
* `CREDIT_GOODS_RATIO`
* `AMT_GOODS_PRICE`
* `AMT_ANNUITY`
* `DAYS_EMPLOYED`
* `DAYS_ID_PUBLISH`

The analysis also examined individual predictions to identify factors that increased or decreased the model's estimated risk.

SHAP explanations are model attributions, not proof of causation. A feature's importance does not establish that changing the feature would change an applicant's real repayment outcome. Sensitive and demographic-related features require particular care in credit-risk applications.

## Decision-Threshold Tuning

The default classification threshold of 0.50 is not always the most appropriate operating point, especially when the target is imbalanced.

The project evaluated thresholds from 0.10 to 0.70 in increments of 0.05 and selected **0.65** because it achieved the highest F1-score among the tested thresholds on the validation set.

### Results at the selected threshold

| Metric          |   Result |
| --------------- | -------: |
| Threshold       |     0.65 |
| Precision       | 0.242998 |
| Recall          | 0.431621 |
| F1-score        | 0.310940 |
| True negatives  |   50,000 |
| False positives |    6,676 |
| False negatives |    2,822 |
| True positives  |    2,143 |

The threshold was saved in `models/threshold_config.json`.

This threshold is a demonstration setting, not an approved lending policy. Because it was selected using validation results, an independent test set and further validation would be needed for a stronger estimate of generalization performance. Business costs, calibration, and fairness should also inform any real operating threshold.

## FastAPI Prediction API

FastAPI exposes the model through HTTP endpoints.

| Method | Endpoint   | Purpose                                                 |
| ------ | ---------- | ------------------------------------------------------- |
| GET    | `/`        | API status                                              |
| GET    | `/health`  | Health check                                            |
| POST   | `/predict` | Predict payment-difficulty risk and return explanations |

### Run the API locally

From the project root, install the dependencies and start the server:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

The prediction endpoint accepts applicant data in JSON format. A valid request must contain the required raw input features expected by the inference pipeline.

## Streamlit Dashboard

The Streamlit dashboard provides a graphical interface for interacting with the prediction API.

Its current features include:

* Input controls for selected applicant and financial attributes.
* Submission of applicant data to the API.
* Display of predicted probability and configured threshold.
* Classification result.
* SHAP-based risk and protective factor tables.
* A chart of explanatory factors.

### Run the dashboard locally

Keep the FastAPI server running in one terminal. Open a second terminal in the project root and run:

```bash
python -m streamlit run app/dashboard.py
```

Streamlit will display the local dashboard URL in the terminal.

**Prototype limitation:** The current dashboard uses a baseline applicant record and overrides selected fields entered through the form. Unedited fields therefore remain inherited from that baseline record. This is useful for demonstrating the interface, but it is not equivalent to collecting and validating a complete real applicant profile.

## Automated Testing

The project includes a pytest test suite at `tests/test_predict.py`.

The five tests verify:

1. The prediction response contains the expected fields.
2. The estimated probability is between 0 and 1.
3. The configured threshold is between 0 and 1.
4. The binary prediction matches the configured threshold.
5. Risk and protective explanations are returned in the expected format.

Run the tests with:

```bash
python -m pytest tests/test_predict.py -v
```

**Current test status: 5 passed.**

These are basic functional checks. They do not establish predictive reliability, model fairness, API security, or production readiness.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/anindita1708/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes the libraries needed for inference, API serving, dashboard execution, and testing.

### 4. Prepare the dataset and model artifacts

Download the Home Credit dataset from Kaggle and place `application_train.csv` under `data/raw/`.

The raw data and large model artifacts are excluded by `.gitignore`. Consequently, cloning the repository alone may not be sufficient to run predictions: the required trained model, fitted preprocessor, and threshold configuration must also be available locally.

Generate these artifacts by running the appropriate training and preprocessing workflow, or provide a documented artifact-download procedure if you later host them separately. Use compatible library versions for training and inference.

### 5. Start the API and dashboard

Start the API:

```bash
python -m uvicorn app.api:app --reload
```

In another terminal, start the dashboard:

```bash
python -m streamlit run app/dashboard.py
```

## Limitations & Future Improvements

Potential next steps include:

* Evaluate performance on a separate, untouched test set.
* Improve model calibration and assess alternative threshold-selection criteria.
* Add additional Home Credit tables through carefully designed feature aggregation.
* Strengthen input validation and error handling in the API.
* Collect complete applicant inputs rather than relying on baseline-record values.
* Add unit and integration tests for preprocessing, API responses, and malformed inputs.
* Evaluate subgroup performance, potential discrimination, and fairness concerns.
* Add model-version tracking and prediction monitoring.
* Containerize the API and dashboard with Docker.
* Deploy the application using a suitable hosting platform.
* Document model artifact versioning and reproducible setup procedures.

Dockerization, online deployment, production monitoring, and database integration should be treated as future work unless their implementation is added to the repository.

## Key Learnings

Through this project, I practiced:

* Working with a large, imbalanced tabular dataset.
* Building preprocessing pipelines and engineered features.
* Comparing machine learning classifiers using multiple metrics.
* Applying SHAP to interpret model behavior.
* Selecting a decision threshold based on validation performance.
* Serving predictions through FastAPI.
* Building an interactive Streamlit interface.
* Writing automated tests with pytest.
* Organizing an end-to-end machine learning project for version control and portfolio presentation.

## Author

**Anindita**

B.Tech in Information Technology | MBA Candidate

Interests: Machine Learning, Data Science, Explainable AI, Python Development, and Data Analytics.

* GitHub: [anindita1708](https://github.com/anindita1708)

## Disclaimer

This project is intended for educational and portfolio purposes only. It is not a financial service, a credit approval system, or a substitute for a properly validated credit-risk assessment process. Predictions may be inaccurate or biased and should not be used to make consequential lending decisions without appropriate independent validation, governance, and regulatory review.

