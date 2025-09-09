# Employee Attrition Project Data Overview

## Data Sources

This project uses two main datasets located in the `data/` directory:

- **Existing Staff.csv**: Contains records of current employees.
- **Exited Staff.csv**: Contains records of employees who have left the organization.

## Data Structure

Both datasets are structured as tabular CSV files, with each row representing an individual employee and columns representing various attributes such as:

- Employee ID / S/N :- unique identifier of each employee
- Job Title
- Department
- Age
- Gender 
- Marital Status
- Years of Service
- Salary
- Mode of Exit
- Date of Exit
- Reasons for Exit

## Data Loading

Data is loaded into pandas DataFrames for analysis:

```python
import pandas as pd

df_1 = pd.read_csv("data/Existing Staff.csv")  # Current employees
df_2 = pd.read_csv("data/Exited Staff.csv")    # Exited employees
```

## Initial Inspection

df_1 (Existing Staff): Used to analyze retention, demographics, and current workforce characteristics.
df_2 (Exited Staff): Used to study attrition patterns, reasons for exit, and compare with retained staff  

### Typical inspection steps include:

- Viewing the first few rows with .head()
- Checking column names and data types with .info()
- Generating summary statistics with .describe(include='all').T
- Identifying missing values and duplicates
- ***Data cleaning and preprocessing are performed using the custom DataCleaner class.***

# Purpose

**These datasets form the foundation for all analysis, feature engineering, and modeling in the Employee Attrition project. They enable exploration of factors influencing employee retention and turnover.**