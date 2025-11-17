# advanced_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import streamlit as st
import plotly.express as px

# -----------------------------
# Core analysis utilities
# -----------------------------

def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats plus skewness & kurtosis for numeric columns."""
    num_df = df.select_dtypes(include=["int64", "float64"])
    desc = num_df.describe().T
    desc["skewness"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()
    return desc

def normality_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run Shapiro-Wilk test for numeric columns (returns statistic and p-value)."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    rows = []
    for c in numeric_cols:
        series = df[c].dropna()
        if len(series) < 3:
            rows.append([c, None, None])
            continue
        try:
            stat, p = shapiro(series)
            rows.append([c, float(stat), float(p)])
        except Exception:
            rows.append([c, None, None])
    return pd.DataFrame(rows, columns=["Column", "Shapiro Statistic", "p-value"])

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric correlation matrix (Pearson)."""
    return df.corr(numeric_only=True)

def pairplot(df: pd.DataFrame, cols: list):
    """Generate seaborn pairplot and return the figure."""
    # seaborn PairGrid returns a FacetGrid object; we convert to fig via plt.gcf()
    sns.set(style="whitegrid")
    g = sns.pairplot(df[cols].dropna())
    return plt.gcf()

def numerical_vs_categorical(df: pd.DataFrame, cat_col: str, num_col: str):
    """Return a seaborn boxplot figure comparing numeric across categories."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with missing count and percentage."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    report = pd.DataFrame({"Missing Count": missing, "Missing %": percent})
    report = report[report["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    return report

def missing_value_heatmap_fig(df: pd.DataFrame):
    """Return a matplotlib figure showing missing value heatmap."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    plt.xlabel("Columns")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    return fig

def duplicate_rows_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicated rows (full rows)."""
    return df[df.duplicated(keep=False)].copy()

# -----------------------------
# Streamlit "show_*" wrappers
# -----------------------------
# these are the functions main.py expects to call

def show_statistical_summary(df: pd.DataFrame):
    """Render statistical summary and normality tests in Streamlit."""
    if df is None:
        st.warning("No dataset loaded.")
        return

    with st.expander("Descriptive statistics (numeric columns)"):
        st.dataframe(statistical_summary(df))

    with st.expander("Normality test (Shapiro-Wilk)"):
        st.write("Note: Shapiro-Wilk requires at least 3 non-null observations per column.")
        st.dataframe(normality_test(df))

def show_correlation_analysis(df: pd.DataFrame):
    """Render correlation heatmap, pairplot selection, and categorical-vs-numerical tool."""
    if df is None:
        st.warning("No dataset loaded.")
        return

    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4, ax=ax)
    st.pyplot(fig)

    # Pairplot selection
    st.markdown("**Pairplot (select numeric columns)**")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    chosen = st.multiselect("Select numeric columns (>=2)", numeric_cols, default=numeric_cols[:3])
    if len(chosen) >= 2:
        try:
            fig_pp = pairplot(df, chosen)
            st.pyplot(fig_pp)
        except Exception as e:
            st.error(f"Could not generate pairplot: {e}")

    # Categorical vs numerical
    st.markdown("**Categorical vs Numerical**")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols and numeric_cols:
        cat = st.selectbox("Categorical column", cat_cols)
        num = st.selectbox("Numeric column", numeric_cols)
        fig_box = numerical_vs_categorical(df, cat, num)
        st.pyplot(fig_box)
    else:
        st.info("Not enough categorical or numerical columns for categorical-numerical analysis.")

def show_missing_value_report(df: pd.DataFrame):
    """Render missing value report, heatmap, and duplicates in Streamlit."""
    if df is None:
        st.warning("No dataset loaded.")
        return

    st.markdown("**Missing Value Summary**")
    report = missing_value_report(df)
    if report.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(report)

    st.markdown("**Missing Value Heatmap**")
    try:
        fig = missing_value_heatmap_fig(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render missing value heatmap: {e}")

    st.markdown("**Duplicate Rows**")
    dup = duplicate_rows_report(df)
    if dup.empty:
        st.info("No duplicate rows found.")
    else:
        st.write(f"Found {len(dup)} duplicated rows (showing sample):")
        st.dataframe(dup.head(200))
