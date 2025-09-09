# Employee Attrition Project Overview

## Project Objective

The Employee Attrition project aims to understand and predict patterns of staff turnover using structured HR data. By analyzing both current and exited employees, the project seeks to uncover key drivers of attrition and build predictive models to support HR decision-making.  

We seek to predict the total number of employees likely to exit over the next six months and calculate the probability of exit for new hires. 

---

## Data Sources

- **Existing Staff.csv**: Contains records of employees currently at the company.
- **Exited Staff.csv**: Contains records of employees who have left the company.

Both datasets include demographic, job-related, and performance features. See [data.md](data.md) for details.

---

## Workflow Summary

1. **Data Loading & Inspection**
   - Data is loaded from CSV files into pandas DataFrames.
   - Initial inspection includes viewing column names, data types, and missing values.

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics and distributions are explored for both datasets.
   - Visualizations include histograms, boxplots, and correlation heatmaps.

3. **Data Cleaning & Preprocessing**
   - Cleaning is performed using the custom `DataCleaner` class (`model_training_src/cleaner.py`).
   - Supports duplicate removal, column dropping, missing value handling, text standardization, and outlier management.
   - Cleaning options are configurable for flexible workflows.

4. **Feature Engineering**
   - Preparation of features for modeling, including encoding categorical variables and scaling numeric features.

5. **Model Training**
   - Modular pipelines support multiple algorithms (e.g., Logistic Regression, Random Forest, XGBoost).
   - Training scripts are organized in `model_training_src/`.

6. **Deployment**
   - Models can be deployed using Streamlit (`app.py`) for interactive HR analytics.

---

## Tools & Technologies

- **Python** (pandas, numpy, scikit-learn, seaborn, matplotlib)
- **Jupyter Notebooks** for EDA and prototyping
- **Custom OOP DataCleaner** for robust preprocessing
- **Streamlit** for deployment

---

## Documentation

- [data.md](data.md): Data structure and sources
- [EDA-cleaning.md](EDA-cleaning.md): EDA and cleaning workflow
- [training.md](training.md): Model training pipeline
- [deployment.md](deployment.md): Deployment guide

---

## Status

- Data loading, inspection, and cleaning are complete.
- EDA and initial visualizations are implemented.
- Modular cleaning pipeline is operational.
- Ready for feature engineering and model training.

---

## Next Steps

- Expand feature engineering and selection.
- Train and evaluate predictive models.
- Deploy models for HR analytics