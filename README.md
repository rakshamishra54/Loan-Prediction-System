
# Loan Prediction System 🏦

Predicts whether a loan application will be approved or rejected
based on applicant details using a Support Vector Machine (SVM) classifier.

## The Problem
Banks and financial institutions receive thousands of loan applications
daily. Manually evaluating each application is time-consuming and prone
to human bias. In India, where credit penetration is still growing,
automating loan eligibility decisions helps banks reduce risk and serve
customers faster. This project builds an ML model that predicts loan
approval based on applicant profile and financial history.

## Dataset
[Loan Prediction Dataset — Analytics Vidhya](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- 614 loan applications
- 12 features including income, credit history, education, property area
- Binary target: Y = Approved | N = Rejected

## Input Features
| Feature | Description |
|---|---|
| Gender | Male or Female |
| Married | Applicant married or not |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate or Not Graduate |
| Self_Employed | Self employed or not |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Term of loan in months |
| Credit_History | Credit history meets guidelines (1) or not (0) |
| Property_Area | Rural, Semiurban, or Urban |

## Model
| Model | Test Accuracy |
|---|---|
| SVM — Linear Kernel | ~79% |

## Project Workflow
- Loaded and explored loan applicant dataset
- Handled missing values using mode and median imputation
- Replaced string value 3+ in Dependents column with numeric 4
- Converted all categorical columns to numerical using label encoding
- Visualized patterns using countplots (Education, Married vs Loan Status)
- Standardized features using StandardScaler
- Trained Support Vector Machine with linear kernel
- Built prediction system for new applicant data

## How To Run
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
jupyter notebook LoanPrediction.ipynb
```

## Sample Prediction
```python
input_data = (1, 1, 0, 0, 0, 4583, 1508, 128, 360, 1, 0)
# Output → Loan will be Approved
```

## Tech Stack
Python | NumPy | Pandas | Scikit-learn | SVM | Seaborn | Matplotlib

## Future Improvements
- Compare with Random Forest and XGBoost
- Handle class imbalance using SMOTE
- Deploy as Streamlit web application
- Add SHAP explainability to show why loan was approved or rejected