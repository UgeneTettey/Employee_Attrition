# Employee Attrition

## Overview
This project explores employee attrition using structured HR data. The goal is to analyze patterns of staff exits and build machine learning models that can help predict attrition. The workflow follows a modular approach, covering data cleaning, exploratory data analysis (EDA), model training, and deployment.

## Data
The dataset comes from two CSV files:
- **Existing Staff.csv** — data on employees currently at the company  
- **Exited Staff.csv** — data on employees who have left the company  

### Key Features
- **Job Title** — role of the employee  
- **Department** — business unit where the employee worked  
- **Age** — age of the employee  
- *(more features described in [docs/data.md](docs/data.md))*  


## Workflow
1. **Exploratory Analysis** — done in Jupyter notebooks  
2. **Data Cleaning & Encoding** — handled using `model_training/cleaner.py` 
3. **Model Training** — modular pipeline in `model_training/trainer.py` with support for multiple algorithms (Logistic Regression, Random Forest, XGBoost, etc.)  
4. **Deployment** — `app.py` to serve models via Streamlit  

## Documentation
Detailed docs for each part of the workflow are available in the [`docs/`](docs/) folder:
- [overview.md](docs/overview.md) — project goals and summary  
- [EDA-cleaning.md](docs/EDA-cleaning.md) — dataset structure, EDA and cleaning  
- [training.md](docs/training.md) — model training pipeline  
- [deployment.md](docs/deployment.md) — deployment guide  




