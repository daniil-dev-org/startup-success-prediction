import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

DATA_RAW_PATH = BASE_DIR / "data" / "external" / "big_startup_success_fail" / "big_startup_secsees_dataset.csv"
DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "clean_startups.csv"
DATA_MODELING_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"

FIGURES_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

NUMERICAL_FEATURES = [
    "log_funding_total_usd",
    "funding_rounds_num",
    "company_age_years",
    "years_to_first_funding",
    "funding_duration_years",
]

CATEGORICAL_FEATURES = [
    "category_code",
    "country_code",
    "state_code",
    "region",
    "city",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
