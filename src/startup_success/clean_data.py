import json
import logging
import numpy as np
import pandas as pd
import pathlib

from .config import DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_MODELING_PATH

def safe_json_loads(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, TypeError):
        pass
    
    if isinstance(value, str):
        if '|' in value:
            return value.split('|')
        if ',' in value:
            return [x.strip() for x in value.split(',')]
        return [value]
    return []

def parse_money(value):
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    val_str = str(value).replace(' ', '').replace(',', '')
    if val_str.startswith('$'):
        val_str = val_str[1:]
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors='coerce')

def clean_data(input_path: str = None, output_path: str = None, target_mode: str = 'clean_exit'):
    input_path = input_path or str(DATA_RAW_PATH)
    output_path = output_path or str(DATA_PROCESSED_PATH)
    modeling_path = str(DATA_MODELING_PATH)
    
    logging.info(f"Зчитування даних з {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    
    status_col = 'status'
    if status_col not in df.columns:
        raise ValueError(f"Колонка {status_col} відсутня у даних.")
        
    df['status_lower'] = df[status_col].fillna('').str.lower()
    
    df['success'] = np.nan
    success_mask = df['status_lower'].isin(['acquired', 'ipo', 'public'])
    failure_mask = df['status_lower'].isin(['closed'])
    
    df.loc[success_mask, 'success'] = 1
    df.loc[failure_mask, 'success'] = 0
    
    df['proxy_success'] = df['success']
    
    logging.info("Парсинг фінансових та інших ознак...")
    
    if 'funding_total_usd' in df.columns:
        df['funding_total_usd'] = df['funding_total_usd'].apply(parse_money)
    elif ' funds_total_usd ' in df.columns:
        df['funding_total_usd'] = df[' funds_total_usd '].apply(parse_money)
    else:
        df['funding_total_usd'] = np.nan
        
    if 'funding_rounds' in df.columns:
        df['funding_rounds_num'] = pd.to_numeric(df['funding_rounds'], errors='coerce')
    else:
        df['funding_rounds_num'] = np.nan

    df['log_funding_total_usd'] = np.log1p(df['funding_total_usd'].fillna(0))

    q75 = df['funding_total_usd'].quantile(0.75) if not df['funding_total_usd'].isna().all() else np.inf
    proxy_cond = (df['funding_total_usd'] > q75) | (df['funding_rounds_num'] >= 3)
    df.loc[proxy_cond & (df['success'].isna()), 'proxy_success'] = 1
    
    date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
    for c in date_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_date)
        else:
            df[c] = pd.NaT
            
    reference_date = pd.to_datetime('2023-01-01')
    
    df['founded_year'] = df['founded_at'].dt.year
    df['first_funding_year'] = df['first_funding_at'].dt.year
    df['last_funding_year'] = df['last_funding_at'].dt.year
    df['company_age_years'] = (df['last_funding_at'].fillna(reference_date) - df['founded_at']).dt.days / 365.25
    df['years_to_first_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days / 365.25
    df['funding_duration_years'] = (df['last_funding_at'] - df['first_funding_at']).dt.days / 365.25
    invalid_logic = (df['founded_at'].notna()) & (
        ((df['last_funding_at'].notna()) & (df['founded_at'] > df['last_funding_at'])) |
        ((df['first_funding_at'].notna()) & (df['founded_at'] > df['first_funding_at']))
    )
    date_features = ['company_age_years', 'years_to_first_funding', 'funding_duration_years']
    df.loc[invalid_logic, date_features] = np.nan
    
    df = df[~(df['company_age_years'] < 0)].copy()
    
    df['company_age_years'] = df['company_age_years'].apply(lambda x: min(x, 50) if pd.notna(x) else x)
    
    df = df[~(df['funding_duration_years'] < 0)].copy()
    df = df[~(df['years_to_first_funding'] < 0)].copy()

    
    if 'category_list' in df.columns:
        df['category_code'] = df['category_list'].apply(lambda x: str(x).split('|')[0] if pd.notna(x) else 'unknown')
    else:
        df['category_code'] = 'unknown'
        
    for c in ['country_code', 'state_code', 'region', 'city']:
        if c not in df.columns:
            df[c] = 'unknown'

    non_null_funds = df['funding_total_usd'].notna().sum()
    if non_null_funds == 0:
        logging.critical("Всі funding_total_usd дорівнюють нулю або NaN. Парсинг не вдався.")
    else:
        logging.info(f"Median funding: {df['funding_total_usd'].median()}, Max: {df['funding_total_usd'].max()}")
        logging.info(f"Non-null funds: {non_null_funds}, Non-null rounds: {df['funding_rounds_num'].notna().sum()}")

    target_col = 'success' if target_mode == 'clean_exit' else 'proxy_success'
    df['target'] = df[target_col]
    
    df.to_csv(output_path, index=False)
    logging.info(f"Повний очищений датасет збережено у {output_path}")
    
    model_df = df[df['target'].notna()].copy()
    model_df['target'] = model_df['target'].astype(int)
    model_df.to_csv(modeling_path, index=False)
    
    logging.info(f"Датасет для моделювання (n={len(model_df)}) збережено у {modeling_path}")
    logging.info(f"Розподіл класів:\n{model_df['target'].value_counts(normalize=True)}")
    
    return model_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    clean_data()
