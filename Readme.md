# EasyVisa — Visa Approval Prediction Using Ensemble Machine Learning

## Project Overview

The U.S. Office of Foreign Labor Certification (OFLC) processes a large and growing number of labor certification applications submitted by employers seeking to hire foreign workers.

In FY 2016 alone, the OFLC processed more than **775,000 employer applications covering approximately 1.7 million positions**, representing a significant increase from the previous year.

Manually reviewing a growing volume of applications can be time-consuming. This project explores how machine learning can help identify application profiles associated with higher or lower likelihood of certification and support more efficient case prioritization.

The project develops and compares multiple **ensemble classification models**, evaluates strategies for handling class imbalance, performs hyperparameter tuning, and analyzes the key factors associated with visa certification outcomes.

---

## Business Objective

The objective of this project is to build a classification model that can help EasyVisa and immigration-processing teams:

- Predict whether a visa application is likely to be **Certified** or **Denied**
- Identify application characteristics associated with certification outcomes
- Compare multiple machine learning algorithms
- Handle class imbalance using oversampling and undersampling
- Improve model performance through hyperparameter tuning
- Identify important predictive features
- Support efficient prioritization of applications for further review

The model is intended as a **decision-support tool**, not as a replacement for legal or regulatory review.

---

## Machine Learning Problem

This is a **binary classification problem**.

The target variable is:

`case_status`

The two possible outcomes are:

- **Certified**
- **Denied**

The model uses applicant qualifications, employment characteristics, employer information, wage information, and geographic variables to predict the historical certification outcome.

---

## Dataset

The dataset contains information about visa applicants and sponsoring employers.

| Feature | Description |
|---|---|
| `case_id` | Unique identifier for each visa application |
| `continent` | Continent of the applicant |
| `education_of_employee` | Applicant's education level |
| `has_job_experience` | Whether the applicant has previous job experience |
| `requires_job_training` | Whether the applicant requires job training |
| `no_of_employees` | Number of employees in the sponsoring company |
| `yr_of_estab` | Year in which the employer was established |
| `region_of_employment` | Intended U.S. region of employment |
| `prevailing_wage` | Wage paid to similarly employed workers for the occupation and location |
| `unit_of_wage` | Unit of prevailing wage: Hourly, Weekly, Monthly, or Yearly |
| `full_time_position` | Whether the position is full-time |
| `case_status` | Whether the application was Certified or Denied |

---

## Project Workflow

```text
Business Problem
      ↓
Data Understanding
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Original Training Data
      ↓
Oversampling using SMOTE
      ↓
Undersampling using RandomUnderSampler
      ↓
Multiple Classification Models
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Feature Importance
      ↓
Business Insights & Recommendations
```

---

# Exploratory Data Analysis

A detailed exploratory analysis was performed to understand the distributions of individual variables and their relationships with visa certification outcomes.

## Univariate Analysis

Individual variables were analyzed using appropriate visualizations to understand:

- Distribution patterns
- Frequency distributions
- Skewness
- Potential outliers
- Applicant characteristics
- Employer characteristics
- Wage characteristics

## Bivariate Analysis

Relationships between individual features and `case_status` were examined to understand which characteristics showed meaningful differences between Certified and Denied applications.

The analysis particularly examined relationships involving:

- Education
- Work experience
- Prevailing wage
- Continent
- Region of employment
- Employer characteristics
- Employment type

Each visualization was accompanied by observations explaining the relevant patterns and their potential business implications.

---

# Data Preprocessing

The dataset was cleaned and transformed before model training.

## Missing Values

The dataset was examined for missing values and handled appropriately where necessary.

## Outlier Detection

Numerical variables were analyzed for unusual observations and potential outliers.

Outliers were reviewed in the context of the underlying business variables before determining whether treatment was required.

## Wage Standardization

`unit_of_wage` contained different wage frequencies such as:

- Hourly
- Weekly
- Monthly
- Yearly

Wage information was transformed appropriately so that compensation values could be compared more consistently across applications.

## Categorical Encoding

Categorical variables were transformed into numerical representations suitable for machine learning models.

## Data Preparation

The final dataset was separated into:

```text
Features → X
Target   → case_status
```

Training, validation, and test datasets were then prepared for model development and evaluation.

---

# Handling Class Imbalance

Visa certification outcomes were not perfectly balanced, making it important to examine whether class imbalance affected model performance.

Three different training strategies were evaluated.

## 1. Original Data

Models were first trained using the original class distribution.

## 2. Oversampled Data

**SMOTE — Synthetic Minority Oversampling Technique** was applied to the training data.

SMOTE generates synthetic observations for the minority class to provide the model with a more balanced training dataset.

```text
Original Training Data
        ↓
      SMOTE
        ↓
Balanced Training Data
        ↓
Model Training
```

## 3. Undersampled Data

**RandomUnderSampler** was also applied.

This technique reduces observations from the majority class to create a more balanced training dataset.

```text
Original Training Data
        ↓
Random Undersampling
        ↓
Balanced Training Data
        ↓
Model Training
```

The performance of models trained using original, oversampled, and undersampled datasets was compared.

---

# Model Development

At least five classification models were developed and evaluated across the different sampling strategies.

Models included:

- Decision Tree
- Random Forest
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- XGBoost

This allowed comparison between individual tree models and multiple ensemble-learning approaches.

---

# Ensemble Learning

The project evaluates several ensemble techniques.

## Bagging

Bagging trains multiple models independently on different samples of the training data and combines their predictions.

Random Forest is a common example of this approach.

```text
Training Data
   ↙   ↓   ↘
Tree Tree Tree
   ↘   ↓   ↙
Combined Prediction
```

Bagging primarily helps reduce variance and improve stability.

---

## Boosting

Boosting trains models sequentially, with later models focusing more heavily on errors made by earlier models.

Models evaluated include:

- AdaBoost
- Gradient Boosting
- XGBoost

```text
Model 1
   ↓
Identify Errors
   ↓
Model 2
   ↓
Focus on Remaining Errors
   ↓
Model 3
   ↓
Combined Prediction
```

Boosting can capture complex relationships and often performs strongly on structured/tabular datasets.

---

# Model Evaluation

Models were compared using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Accuracy alone was not used for model selection because it can be misleading when class distributions are unequal.

## Precision

Precision measures how many applications predicted as a particular class were actually in that class.

Higher precision reduces false-positive predictions.

## Recall

Recall measures how many actual cases of a class were successfully identified.

For this project, recall is important because missing applications belonging to the relevant class may have operational consequences.

## F1 Score

F1 Score combines precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 Score was used as an important model-selection metric because it captures the balance between precision and recall.

---

# Model Building Across Sampling Strategies

Each major classification algorithm was evaluated using:

### Original Training Data

Models were trained on the original class distribution.

### Oversampled Training Data

Models were trained using SMOTE-balanced data.

### Undersampled Training Data

Models were trained using RandomUnderSampler-balanced data.

This created a broad comparison of model behavior across different class-balancing strategies.

---

# Hyperparameter Tuning

The strongest-performing candidate models were selected for additional tuning.

The top three ensemble models were:

- Gradient Boosting
- AdaBoost
- XGBoost

Hyperparameters were systematically tuned to improve predictive performance and model generalization.

The tuned models were then compared using training, validation, and test metrics.

---

# Comparative Model Performance

Multiple ensemble models were evaluated using Accuracy, Recall, Precision, and F1 Score across training, validation, and test datasets.

## Key Observations

- All shortlisted models demonstrated strong generalization.
- Train–validation performance gaps remained below approximately `0.01`, indicating limited evidence of overfitting.
- Performance differences between the strongest models were small.
- Accuracy was similar across models, reinforcing the importance of considering Precision, Recall, and F1 Score rather than accuracy alone.

### Metric-Level Comparison

**Recall**

AdaBoost demonstrated strong recall performance, helping reduce missed positive-class cases.

**Precision**

Gradient Boosting produced slightly stronger precision in the model comparison.

**F1 Score**

XGBoost demonstrated strong and consistent F1 performance during model selection, providing a balanced precision–recall trade-off.

---

# Final Test Performance

| Model | Test F1 Score |
|---|---:|
| Gradient Boosting | 0.8181 |
| XGBoost | 0.8182 |
| AdaBoost | 0.8208 |

The absolute differences between the three models were very small, showing that all three approaches produced competitive performance on the test dataset.

---

# Final Model Selection

## XGBoost

**XGBoost was selected as the final model for this project.**

The decision was based on:

- Strong validation F1 Score during model selection
- Stable performance across training, validation, and test datasets
- Balanced Precision and Recall
- Limited train–validation performance gap
- Ability to capture nonlinear relationships and feature interactions

Although AdaBoost achieved a slightly higher F1 Score on the final test set, the final model selection was based on the validation-stage model-selection process rather than choosing a model retrospectively based on test-set performance.

This preserves the role of the test dataset as an unbiased final evaluation set.

---

# Feature Importance

Feature importance analysis was performed on the final ensemble model to understand which variables contributed most strongly to its predictions.

## Strong Predictive Features

### Education of Employee

Education emerged as one of the strongest predictive signals.

Applicants with different educational qualifications showed meaningful differences in historical certification outcomes.

### Job Experience

Previous job experience was another highly influential feature.

Applications involving candidates with professional experience showed different prediction patterns from those without prior experience.

### Continent

Applicant continent contributed predictive information based on historical application patterns.

This should be interpreted carefully: geographic variables may reflect historical differences in applicant populations, job types, policy environments, or other correlated factors. They should not automatically be treated as causal drivers of eligibility.

### Region of Employment

The intended U.S. region of employment also contributed predictive information.

This may reflect differences in labor markets, occupational demand, wages, and application composition across regions.

---

## Moderate Predictive Influence

Features with moderate importance included:

- Prevailing wage
- Unit of wage

Compensation contributed useful predictive information but was less influential than several applicant qualification features in the fitted model.

---

## Lower Predictive Influence

Features with comparatively lower importance included:

- Number of employees
- Full-time position
- Employer age

Within this dataset and model, employer size and maturity contributed less predictive information than several applicant-related variables.

---

# Business Insights

## 1. Applicant Qualifications Are Important Predictive Signals

Education and prior professional experience were among the strongest features used by the model.

### Business Implication

Qualification-related information may be useful when prioritizing applications for further review or estimating historical certification likelihood.

These predictions should complement rather than replace the formal eligibility criteria used by immigration authorities.

---

## 2. Work Experience Contributes Meaningful Predictive Information

Previous professional experience had substantial importance in the model.

### Business Implication

Employers and application-support teams can use experience information when estimating the historical profile of an application and identifying cases requiring additional review.

---

## 3. Geographic Variables Require Careful Interpretation

Continent and region of employment contributed predictive information.

However, these relationships should **not be interpreted as evidence that geography itself causes an approval or denial**.

They may capture underlying differences involving:

- Occupational composition
- Labor demand
- Wage distributions
- Applicant populations
- Regional employment patterns
- Historical administrative patterns

### Business Implication

Geographic features should be monitored carefully for fairness, legal, and compliance concerns before being incorporated into any real-world decision-support workflow.

---

## 4. Wage Is Useful but Not Dominant

Prevailing wage contributes to predictions, but its importance was lower than several qualification-related variables.

### Business Implication

Compensation should be considered alongside the broader applicant and employment profile rather than used as a standalone indicator.

---

## 5. Employer Characteristics Contribute Less Predictive Information

Employer size and company age showed relatively low feature importance in this model.

### Business Implication

Within this dataset, applicant and employment characteristics provided more predictive information than company scale alone.

---

# Business Recommendations

## Recommendation 1 — Use the Model for Application Prioritization

XGBoost can be used as a decision-support model to estimate historical certification likelihood and help prioritize applications requiring review.

The model should not independently approve or deny applications.

---

## Recommendation 2 — Focus on High-Value Predictive Features

The strongest predictive signals included:

- Education
- Work experience
- Geographic and employment characteristics
- Wage information

These variables can help analysts understand why the model produces a particular prediction.

---

## Recommendation 3 — Monitor Fairness and Geographic Effects

Features such as continent can encode historical patterns that may introduce unwanted bias.

Before production use:

- Perform subgroup performance analysis
- Measure false-positive and false-negative rates across relevant groups
- Review whether sensitive or proxy variables are appropriate
- Apply legal and compliance review
- Maintain human oversight

---

## Recommendation 4 — Avoid Using a Single Metric

Model monitoring should consider:

- Precision
- Recall
- F1 Score
- Class-specific errors
- Performance across applicant groups
- Performance drift over time

Accuracy alone is not sufficient for evaluating this system.

---

## Recommendation 5 — Retrain and Monitor the Model

Immigration application patterns, labor markets, regulations, and applicant populations can change over time.

A production implementation should include:

```text
New Applications
      ↓
Model Predictions
      ↓
Human Review
      ↓
Actual Outcomes
      ↓
Performance Monitoring
      ↓
Periodic Model Retraining
```

---

# Executive Summary

An ensemble-based machine learning framework was developed to predict historical visa certification outcomes using applicant, employer, wage, and employment-related attributes.

Multiple models were trained across **original, oversampled, and undersampled datasets**, followed by hyperparameter tuning of the strongest candidates.

Gradient Boosting, AdaBoost, and XGBoost demonstrated strong and consistent performance. XGBoost was selected as the final model based on its validation-stage F1 performance, balanced Precision–Recall behavior, and stable generalization.

Feature importance analysis showed that **education and prior job experience were among the strongest predictive signals**, while employer size and company age contributed comparatively less information.

The project demonstrates how ensemble machine learning can support application prioritization and operational efficiency while emphasizing the importance of model interpretability, fairness monitoring, human review, and responsible use in high-impact decision-support systems.

---

# Key Concepts Demonstrated

This project demonstrates practical application of:

- Binary Classification
- Exploratory Data Analysis
- Univariate Analysis
- Bivariate Analysis
- Feature Engineering
- Categorical Encoding
- Wage Normalization
- Class Imbalance
- SMOTE
- Random Undersampling
- Decision Trees
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- XGBoost
- Ensemble Learning
- Hyperparameter Tuning
- Accuracy
- Precision
- Recall
- F1 Score
- Train / Validation / Test Evaluation
- Overfitting Analysis
- Feature Importance
- Model Comparison
- Model Selection
- Business Interpretation
- Responsible ML Considerations

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Google Colab
- Jupyter Notebook

---

# Repository Structure

```text
easyvisa-visa-approval-prediction/
│
├── README.md
├── EasyVisa_Notebook.ipynb
├── EasyVisa_Notebook.html
│
└── data/
    └── EasyVisa.csv
```

## Files

- `README.md` — Project overview, methodology, modeling approach, results, and business insights
- `EasyVisa_Notebook.ipynb` — Complete Google Colab notebook containing EDA, preprocessing, sampling strategies, model development, tuning, evaluation, and recommendations
- `EasyVisa_Notebook.html` — Rendered notebook for convenient viewing without running the code
- `data/` — Dataset used for the analysis

---

# Conclusion

This project demonstrates an end-to-end **Advanced Machine Learning classification workflow** for predicting historical visa certification outcomes.

The project goes beyond training a single classifier by comparing multiple algorithms across original, oversampled, and undersampled datasets, applying hyperparameter tuning, evaluating generalization, and analyzing feature importance.

The final XGBoost model provides a strong balance between Precision and Recall and demonstrates stable performance across dataset splits.

More importantly, the project illustrates how machine learning results can be translated into operational insights while recognizing that high-impact domains such as immigration require **human oversight, fairness evaluation, transparency, and careful interpretation of predictive features**.