import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import numpy as np

from .config import REPORTS_DIR, FIGURES_DIR
from .features import get_feature_names

def run_explanation(best_pipeline, X_test, y_test):
    logging.info("Пояснення моделі: важливість ознак та SHAP...")
    
    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    
    importance_list = []
    
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance_value": importances,
                "importance_type": "model_builtin",
                "model": type(model).__name__
            }).sort_values("importance_value", ascending=False)
            importance_list.append(fi_df)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi_df.head(20), x="importance_value", y="feature", ax=ax, palette="viridis", hue="feature", legend=False)
            ax.set_title(f"Feature Importance ({type(model).__name__})")
            fig.savefig(FIGURES_DIR / "17_feature_importance.png", bbox_inches="tight")
            plt.close(fig)

    logging.info("Розрахунок permutation importance...")
    perm_res = permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_value": perm_res.importances_mean,
        "importance_type": "permutation",
        "model": type(model).__name__
    }).sort_values("importance_value", ascending=False)
    importance_list.append(perm_df)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=perm_df.head(20), x="importance_value", y="feature", ax=ax, palette="plasma", hue="feature", legend=False)
    ax.set_title("Permutation Importance (on Raw Features)")
    fig.savefig(FIGURES_DIR / "18_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)
    
    logging.info("Розрахунок SHAP values (sample)...")
    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        sample_size = min(1000, len(X_test))
        X_test_sample = X_test.sample(sample_size, random_state=42)
        X_test_transformed = preprocessor.transform(X_test_sample)
        
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
            
        explainer = shap.Explainer(model, X_test_transformed, feature_names=feature_names)
        shap_values = explainer(X_test_transformed, check_additivity=False)
        shap_values.feature_names = list(feature_names)
        
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.title("SHAP Feature Importance")
        plt.savefig(FIGURES_DIR / "19_shap_bar.png", bbox_inches="tight")
        plt.close()
        
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title("SHAP Summary Plot")
        plt.savefig(FIGURES_DIR / "20_shap_summary.png", bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        logging.warning(f"SHAP розрахунок не вдався: {e}")

    if importance_list:
        all_imp = pd.concat(importance_list)
        all_imp["rank"] = all_imp.groupby("importance_type")["importance_value"].rank(ascending=False)
        all_imp.to_csv(REPORTS_DIR / "top_features.csv", index=False)
        
    logging.info("Пояснення моделі завершено.")
