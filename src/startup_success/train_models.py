import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import numpy as np

from .config import MODELS_DIR, ALL_FEATURES
from .features import build_preprocessor

def get_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    except (ImportError, ModuleNotFoundError):
        logging.warning("XGBoost не знайдено, пропускаємо цю модель.")
        
    return models

def train_and_evaluate_cv(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    preprocessor = build_preprocessor()
    models = get_models()
    
    cv_results = []
    trained_pipelines = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy'
    }
    
    for name, model in models.items():
        logging.info(f"Тренування та крос-валідація моделі: {name}")
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        
        res = {
            "model": name,
            "cv_auc_mean": np.mean(scores['test_auc']),
            "cv_auc_std": np.std(scores['test_auc']),
            "cv_f1_mean": np.mean(scores['test_f1']),
            "cv_f1_std": np.std(scores['test_f1']),
            "cv_precision_mean": np.mean(scores['test_precision']),
            "cv_recall_mean": np.mean(scores['test_recall']),
            "cv_accuracy_mean": np.mean(scores['test_accuracy'])
        }
        cv_results.append(res)
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")
        
    comparison_df = pd.DataFrame(cv_results)
    
    best_model_name = comparison_df.sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    logging.info(f"Найкраща модель: {best_model_name}")
    
    return trained_pipelines, best_pipeline, X_test, y_test, comparison_df
