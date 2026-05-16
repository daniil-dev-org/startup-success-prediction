import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import numpy as np

from .config import REPORTS_DIR, FIGURES_DIR

def evaluate_models(trained_pipelines, X_test, y_test, cv_comparison_df):
    test_metrics = []
    
    for name, pipeline in trained_pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['model'], "predict_proba") else None
        
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        
        test_metrics.append(metrics)
        
    test_metrics_df = pd.DataFrame(test_metrics)
    
    full_comparison = pd.merge(cv_comparison_df, test_metrics_df, on="model")
    full_comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    
    best_model_name = cv_comparison_df.sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
    best_metrics = next(m for m in test_metrics if m["model"] == best_model_name)
    
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)
        
    logging.info(f"Метрики збережено у {REPORTS_DIR}")
    
    return full_comparison

def plot_evaluation_results(full_comparison, best_pipeline, X_test, y_test):
    fig, ax = plt.subplots(figsize=(12, 6))
    melted = full_comparison.melt(id_vars="model", value_vars=["cv_auc_mean", "cv_f1_mean", "roc_auc", "f1"])
    sns.barplot(data=melted, x="model", y="value", hue="variable", ax=ax)
    ax.set_title("Порівняння метрик моделей (CV vs Test)")
    ax.set_ylim(0, 1.0)
    fig.savefig(FIGURES_DIR / "13_model_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"Best Model (AUC = {auc_val:.2f})", color="blue", lw=2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(FIGURES_DIR / "14_roc_curve.png", bbox_inches="tight")
    plt.close(fig)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label="Precision-Recall", color="green", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(FIGURES_DIR / "15_precision_recall_curve.png", bbox_inches="tight")
    plt.close(fig)
    
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Best Model)")
    fig.savefig(FIGURES_DIR / "16_confusion_matrix_best_model.png", bbox_inches="tight")
    plt.close(fig)
    
    logging.info("Графіки оцінки моделей згенеровано.")
