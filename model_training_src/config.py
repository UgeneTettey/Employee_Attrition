from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union   #this is used for type-hinting. This indicates what a function parameter or variable is expected to be.

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

@dataclass
class Config:
    """This is a generalized configuration class for model training
     designed to be project-agnostic, it has both classification and regression settings."""
    
    # ------Data paths------------------------------------------------------------
    data_dir: Path = Path("data")                   # Directory for data files
    train_data_path: Optional[Path] = None          # Path to training data file
    test_data_path: Optional[Path] = None           # Path to test data file
    raw_data_path: Optional[Path] = None            # Path to raw data file
    target_column: str = "target"                   # Target variable name
    


    # ----Training parameters--------------------------------------------------------
    test_size: float = 0.2                # Proportion of data to use for testing   
    random_state: int = 42          # Random seed for reproducibility
    cv_folds: int = 5             # Number of cross-validation folds
    scoring_metrics: Union[str, List[str]] = "accuracy"            # Scoring metric(s) for model evaluation



    # -----task type--------------------------------------------------------------
    task_type: str = "classification"     # "classification" or "regression"
    
    # ----------Model parameters (mutable, can be overridden)------------------------------
    classification_models: Optional[Dict[str, Dict[str, Any]]] = None  # Dictionary of classification models and their parameters
    regression_models: Optional[Dict[str, Dict[str, Any]]] = None      # Dictionary of regression models and their parameters



    # -----Output paths--------------------------------------------------------------
    artifacts_dir: Path = Path("artifacts")     # Directory to store all artifacts/outputs
    model_dir: Path = field(init=False)        # Models will be stored here
    results_dir: Path = field(init=False)      # Evaluation results (metrics, confusion matrices, etc.)
    reports_dir: Path = field(init=False)      # Detailed reports (classification reports, etc.)



    # -----Preprocessing options----------------------------------------------
    scale_features: bool = True
    encode_categorical: bool = True
    

    def __post_init__(self):
        # Default classification models
        if self.models is None:
            self.models = {
                "logistic_regression": {
                    "class": LogisticRegression,
                    "params": {
                        "random_state": self.random_state,
                        "max_iter": 1000
                    }
                },
                "random_forest": {
                    "class": RandomForestClassifier,
                    "params": {
                        "random_state": self.random_state,
                        "n_estimators": 100
                    }
                },
                "xgboost": {
                    "class": XGBClassifier,
                    "params": {
                        "random_state": self.random_state,
                        "n_estimators": 100
                    }
                },
            }

        # Default regression models
        if self.regression_models is None:
            self.regression_models = {
                "LinearRegression": {
                    "class": LinearRegression,
                    "params": {}
                },
                "RandomForestRegressor": {
                    "class": RandomForestRegressor,
                    "params": {
                        "random_state": self.random_state,
                        "n_estimators": 100
                    }
                },
                "XGBRegressor": {
                    "class": XGBRegressor,
                    "params": {
                        "random_state": self.random_state,
                        "n_estimators": 100
                    }
                },
            }
        
        # Create sub-directories under artifacts_dir
        self.model_dir = self.artifacts_dir / "models"
        self.results_dir = self.artifacts_dir / "results"
        self.reports_dir = self.artifacts_dir / "reports"

        # self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        # self.model_dir.mkdir(exist_ok=True, parents=True)
        # self.results_dir.mkdir(exist_ok=True, parents=True)
        # self.reports_dir.mkdir(exist_ok=True, parents=True)