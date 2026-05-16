import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from startup_success.clean_data import clean_data
from startup_success.eda import run_eda
from startup_success.train_models import train_and_evaluate_cv
from startup_success.evaluate import evaluate_models, plot_evaluation_results
from startup_success.explain import run_explanation
from startup_success.features import save_used_features
from startup_success.config import ALL_FEATURES

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", encoding="utf-8")
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Startup Success Prediction Pipeline")
    parser.add_argument("--target", type=str, default="clean_exit", choices=["clean_exit", "proxy"], help="Target logic to use")
    args = parser.parse_args()

    setup_logging()
    logging.info("Початок виконання pipeline...")

    try:
        logging.info(f"Крок 1: Очищення даних (target={args.target})...")
        df_modeling = clean_data(target_mode=args.target)
        
        logging.info("Крок 2: Розвідувальний аналіз (EDA)...")
        run_eda(df_modeling)
        
        logging.info("Крок 3: Тренування моделей та крос-валідація...")
        trained_pipelines, best_pipeline, X_test, y_test, comparison_df = train_and_evaluate_cv(df_modeling)
        
        save_used_features(ALL_FEATURES)
        
        logging.info("Крок 4: Оцінка моделей на тестовому наборі...")
        full_comparison = evaluate_models(trained_pipelines, X_test, y_test, comparison_df)
        plot_evaluation_results(full_comparison, best_pipeline, X_test, y_test)
        
        logging.info("Крок 5: Інтерпретація результатів...")
        run_explanation(best_pipeline, X_test, y_test)
        
        logging.info("Pipeline успішно завершено!")
        
    except Exception as e:
        logging.critical(f"Помилка під час виконання pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
