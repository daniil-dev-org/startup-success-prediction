import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json

from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, REPORTS_DIR

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=5, sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n).replace("num__", "").replace("cat__", "") for n in names]
    except Exception:
        return []

def save_used_features(features_list):
    path = REPORTS_DIR / "used_features.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"used_features": features_list}, f, ensure_ascii=False, indent=4)
