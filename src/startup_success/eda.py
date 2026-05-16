import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .config import FIGURES_DIR, NUMERICAL_FEATURES

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def _save_figure(fig, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_target_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["target"].value_counts().sort_index()
    sns.barplot(x=[str(int(k)) for k in counts.index], y=counts.values, ax=ax, palette="viridis")
    ax.set_title("Розподіл цільової змінної (Success vs Failure)", fontsize=14)
    ax.set_xlabel("Успіх (1) чи невдача (0)", fontsize=12)
    ax.set_ylabel("Кількість компаній", fontsize=12)
    _save_figure(fig, "01_target_distribution.png")

def plot_funding_total_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df[df["funding_total_usd"] > 0]["funding_total_usd"].dropna()
    if len(data) > 0:
        p99 = data.quantile(0.99)
        data_plot = data[data <= p99]
        sns.histplot(data_plot, bins=50, kde=True, ax=ax, color="blue")
        ax.set_title(f"Розподіл фінансування (до 99-го перцентиля, < ${p99/1e6:.1f}M)", fontsize=14)
        ax.set_xlabel("Сума в USD", fontsize=12)
        ax.set_ylabel("Частота", fontsize=12)
        _save_figure(fig, "02_funding_total_distribution.png")
    else:
        logging.warning("Пропущено 02_funding_total_distribution.png: немає даних")

def plot_log_funding_total_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df["log_funding_total_usd"].dropna()
    sns.histplot(data, bins=50, kde=True, ax=ax, color="green")
    ax.set_title("Розподіл логарифмованого фінансування", fontsize=14)
    ax.set_xlabel("log(1 + funding_total_usd)", fontsize=12)
    ax.set_ylabel("Частота", fontsize=12)
    _save_figure(fig, "03_log_funding_total_distribution.png")

def plot_funding_by_success(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="target", y="log_funding_total_usd", data=df, ax=ax, palette="Set2")
    ax.set_title("Фінансування в розрізі успішності", fontsize=14)
    ax.set_xlabel("Успіх (1) чи невдача (0)", fontsize=12)
    ax.set_ylabel("log(1 + funding_total_usd)", fontsize=12)
    _save_figure(fig, "04_funding_by_success.png")

def plot_funding_rounds_by_success(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if "funding_rounds_num" in df.columns:
        sns.boxplot(x="target", y="funding_rounds_num", data=df, ax=ax, palette="Set1")
        ax.set_title("Кількість раундів за успішністю", fontsize=14)
        ax.set_xlabel("Успіх (1) чи невдача (0)", fontsize=12)
        ax.set_ylabel("Кількість раундів", fontsize=12)
        _save_figure(fig, "06_funding_rounds_by_success.png")

def plot_top_markets(df, top_n=15) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df["category_code"].value_counts().head(top_n)
    sns.barplot(y=counts.index, x=counts.values, ax=ax, palette="magma")
    ax.set_title(f"Топ-{top_n} індустрій/ринків", fontsize=14)
    ax.set_xlabel("Кількість компаній", fontsize=12)
    ax.set_ylabel("Індустрія", fontsize=12)
    _save_figure(fig, "07_top_markets_or_industries.png")

def plot_success_rate_by_market(df, min_count=100) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    stats = df.groupby("category_code")["target"].agg(['mean', 'count'])
    stats = stats[stats['count'] >= min_count].sort_values("mean", ascending=False).head(15)
    if not stats.empty:
        new_index = [f"{idx} (n={int(count)})" for idx, count in zip(stats.index, stats['count'])]
        stats.index = new_index
        sns.barplot(y=stats.index, x=stats['mean'], ax=ax, palette="coolwarm")
        ax.set_title("Рівень успіху за індустріями (n>=100)", fontsize=14)
        ax.set_xlabel("Частка успішних", fontsize=12)
        ax.set_ylabel("Індустрія", fontsize=12)
        _save_figure(fig, "08_success_rate_by_market_or_industry.png")
    else:
        logging.warning("Пропущено 08_success_rate_by_market_or_industry.png: недостатньо даних")

def plot_success_rate_by_country(df, min_count=100) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    stats = df.groupby("country_code")["target"].agg(['mean', 'count'])
    stats = stats[stats['count'] >= min_count].sort_values("mean", ascending=False).head(15)
    if not stats.empty:
        new_index = [f"{idx} (n={int(count)})" for idx, count in zip(stats.index, stats['count'])]
        stats.index = new_index
        sns.barplot(y=stats.index, x=stats['mean'], ax=ax, palette="viridis")
        ax.set_title("Рівень успіху за країнами (n>=100)", fontsize=14)
        ax.set_xlabel("Частка успішних", fontsize=12)
        ax.set_ylabel("Країна", fontsize=12)
        _save_figure(fig, "09_success_rate_by_country.png")
    else:
        logging.warning("Пропущено 09_success_rate_by_country.png: недостатньо даних")

def plot_company_age_by_success(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if "company_age_years" in df.columns:
        sns.boxplot(x="target", y="company_age_years", data=df, ax=ax, palette="muted")
        ax.set_title("Вік компанії за успішністю", fontsize=14)
        ax.set_xlabel("Успіх (1) чи невдача (0)", fontsize=12)
        ax.set_ylabel("Вік (роки)", fontsize=12)
        _save_figure(fig, "10_company_age_by_success.png")

def plot_correlation_heatmap(df) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    cols_to_corr = [c for c in NUMERICAL_FEATURES if c in df.columns] + ["target"]
    corr = df[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Кореляційна матриця ознак", fontsize=14)
    _save_figure(fig, "11_correlation_heatmap.png")

def plot_missing_values(df) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="Reds_r")
        ax.set_title("Відсоток пропущених значень", fontsize=14)
        ax.set_xlabel("% пропусків", fontsize=12)
        _save_figure(fig, "12_missing_values.png")
    else:
        logging.info("Пропущено 12_missing_values.png: немає пропущених значень")

def run_eda(df: pd.DataFrame) -> None:
    logging.info("Запуск EDA та генерація графіків...")
    plot_target_distribution(df)
    plot_funding_total_distribution(df)
    plot_log_funding_total_distribution(df)
    plot_funding_by_success(df)
    plot_funding_rounds_by_success(df)
    plot_top_markets(df)
    plot_success_rate_by_market(df)
    plot_success_rate_by_country(df)
    plot_company_age_by_success(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)
    logging.info("EDA завершено.")
