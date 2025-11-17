import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Local imports
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page
import advanced_analysis

# -------------------------
# Page config & global CSS
# -------------------------
st.set_page_config(page_icon="✨", page_title="AutoEDA", layout="wide")

GLOBAL_CSS = """
<style>
:root{
  --bg:#f5f7fb;
  --card:#ffffff;
  --accent1:#5A5DF0;
  --accent2:#8A8CFF;
  --muted:#6b7280;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
.section-header { font-size:20px; font-weight:700; color:#1f2937; margin: 8px 0; }
.app-card { background: var(--card); border-radius:12px; padding:18px; box-shadow:0 6px 18px rgba(13,38,61,0.06); }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -------------------------
# Sidebar: file upload
# -------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#5A5DF0;text-align:center;'>✨ AutoEDA</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload CSV / Excel", type=["csv", "xls", "xlsx"])
    
    # --- FIX: Changed checkbox to button ---
    use_example = st.button("Load Example Titanic Dataset")

# -------------------------
# Top navigation
# -------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Exploration", "Data Preprocessing", "Advanced EDA"],
    icons=["house", "bar-chart", "wrench", "layers"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0px", "background": "transparent"},
        "nav-link": {"font-size": "16px", "color": "#111827"},
        "nav-link-selected": {
            "background": "linear-gradient(90deg,#5A5DF0,#8A8CFF)",
            "color": "white"
        },
    },
)

# -------------------------
# Data loading
# -------------------------

# --- FIX: Changed 'elif' to 'if' ---
# This ensures a button click or file upload
# sets the session state only *once*.

if uploaded_file:
    df = function.load_data(uploaded_file)
    st.session_state["new_df"] = df.copy()

if use_example:
    df = function.load_data("example_dataset/titanic.csv")
    st.session_state["new_df"] = df.copy()

# HOME PAGE
if selected == "Home":
    home_page.show_home_page()

# If no data and not Home -> stop execution
if selected != "Home" and ("new_df" not in st.session_state):
    st.warning("Please upload a dataset first.")
    st.stop()

# -------------------------
# DATA EXPLORATION
# -------------------------
if selected == "Data Exploration":
    df = st.session_state["new_df"]
    num_cols, cat_cols = function.categorical_numerical(df)

    tab1, tab2 = st.tabs(["📊 Overview", "🔍 Visualization"])

    with tab1:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("📁 Dataset Overview")
        function.display_dataset_overview(df, cat_cols, num_cols)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("❌ Missing Values")
        function.display_missing_values(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("📊 Statistics & Types")
        function.display_statistics_visualization(df, cat_cols, num_cols)
        function.display_data_types(df)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("📈 Feature Distributions")
        function.display_individual_feature_distribution(df, num_cols)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("🔗 Categorical Variable Analysis")
        if cat_cols:
            function.categorical_variable_analysis(df, cat_cols)
        else:
            st.info("No categorical columns available.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# DATA PREPROCESSING
# -------------------------
if selected == "Data Preprocessing":

    new_df = st.session_state["new_df"]
    st.header("🛠 Data Preprocessing")

    tabs = st.tabs(["🧩 Missing Values", "🧠 Encoding", "📏 Scaling", "📈 Outliers", "🧾 Column Ops"])

    # MISSING VALUES TAB
    with tabs[0]:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("Missing Values")

        function.display_missing_values(new_df)

        col_list = new_df.columns.tolist()

        cols_fill = st.multiselect("Select columns to fill:", col_list)
        fill_method = st.selectbox("Method:", ["mean", "median", "mode"])

        if st.button("Apply Fill"):
            st.session_state["new_df"] = preprocessing_function.fill_missing_data(new_df.copy(), cols_fill, fill_method)
            st.success(f"Missing values filled using {fill_method}")
            st.rerun() # Added rerun for immediate update

        remove_cols = st.multiselect("Drop rows where selected columns have missing:", col_list)

        if st.button("Drop Rows"):
            st.session_state["new_df"] = preprocessing_function.remove_rows_with_missing_data(new_df.copy(), remove_cols)
            st.success("Rows dropped.")
            st.rerun() # Added rerun for immediate update

        st.markdown("</div>", unsafe_allow_html=True)

    # ENCODING TAB
    with tabs[1]:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.subheader("Encoding")

        cat_cols = new_df.select_dtypes(include=['object', 'category']).columns.tolist()

        if cat_cols:
            enc_choice = st.radio("Encoding method:", ["Label Encoding", "One Hot Encoding"])
            sel_cols = st.multiselect("Select columns", cat_cols)

            if enc_choice == "Label Encoding" and st.button("Apply Label Encoding"):
                st.session_state["new_df"] = preprocessing_function.label_encode(new_df.copy(), sel_cols)
                st.success("Label Encoding applied.")
                st.rerun() # Added rerun for immediate update

            if enc_choice == "One Hot Encoding" and st.button("Apply One Hot Encoding"):
                st.session_state["new_df"] = preprocessing_function.one_hot_encode(new_df.copy(), sel_cols)
                st.success("One Hot Encoding applied.")
                st.rerun() # Added rerun for immediate update
        else:
            st.info("No categorical columns found.")

        st.markdown("</div>", unsafe_allow_html=True)

    # SCALING TAB
    with tabs[2]:
        st.subheader("Scaling")
        st.markdown("<div class'app-card'>", unsafe_allow_html=True)

        numeric_cols = new_df.select_dtypes(include=['number']).columns.tolist()

        cols_scale = st.multiselect("Select columns:", numeric_cols)
        scale_method = st.selectbox("Method:", ["Standardization", "Min-Max"])

        if st.button("Apply Scaling"):
            if scale_method == "Standardization":
                st.session_state["new_df"] = preprocessing_function.standard_scale(new_df.copy(), cols_scale)
            else:
                st.session_state["new_df"] = preprocessing_function.min_max_scale(new_df.copy(), cols_scale)
            st.success(f"{scale_method} applied.")
            st.rerun() # Added rerun for immediate update

        st.markdown("</div>", unsafe_allow_html=True)

    # OUTLIERS TAB
    with tabs[3]:
        st.subheader("Outliers")
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)

        numeric_cols = new_df.select_dtypes(include=['number']).columns.tolist()

        out_col = st.selectbox("Select column", numeric_cols)
        detect_method = st.radio("Detection Method", ["IQR", "Z-Score"])

        if st.button("Detect"):
            if detect_method == "IQR":
                outliers = preprocessing_function.detect_outliers_iqr(new_df.copy(), out_col)
            else:
                outliers = preprocessing_function.detect_outliers_zscore(new_df.copy(), out_col)
            st.write(outliers[:200])

        handle = st.selectbox("Handle outliers:", ["None", "Remove", "Replace with Median"])

        if handle != "None" and st.button("Apply Handling"):
            outliers = preprocessing_function.detect_outliers_iqr(new_df.copy(), out_col)
            if handle == "Remove":
                st.session_state["new_df"] = preprocessing_function.remove_outliers(new_df.copy(), out_col, outliers)
                st.success("Outliers removed.")
            else:
                st.session_state["new_df"] = preprocessing_function.transform_outliers(new_df.copy(), out_col, outliers)
                st.success("Outliers replaced with median.")
            st.rerun() # Added rerun for immediate update

        st.markdown("</div>", unsafe_allow_html=True)

    # COLUMN OPS TAB
    with tabs[4]:
        st.subheader("Column Operations")
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)

        cols = new_df.columns.tolist()

        with st.expander("Rename Column"):
            sel = st.selectbox("Select column", cols, key="rename_sel")
            new = st.text_input("New name", key="rename_new")
            if st.button("Rename"):
                tmp = new_df.copy()
                tmp.rename(columns={sel: new}, inplace=True)
                st.session_state["new_df"] = tmp
                st.success("Renamed.")
                st.rerun() # Added rerun for immediate update

        with st.expander("Change Type"):
            sel = st.selectbox("Column", cols, key="type_sel")
            dtype = st.selectbox("New Type", ["int", "float", "string"], key="type_new")
            if st.button("Convert"):
                tmp = new_df.copy()
                tmp[sel] = tmp[sel].astype(dtype)
                st.session_state["new_df"] = tmp
                st.success("Converted.")
                st.rerun() # Added rerun for immediate update

        with st.expander("Drop Duplicates"):
            if st.button("Drop"):
                tmp = new_df.copy().drop_duplicates()
                st.session_state["new_df"] = tmp
                st.success("Duplicates removed.")
                st.rerun() # Added rerun for immediate update

        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- DOWNLOAD SECTION (with key fix) ---
    
    st.subheader("Preview & Download")
    st.dataframe(st.session_state["new_df"].head(10))

    # First, get the final dataframe from session state
    df_to_download = st.session_state["new_df"]

    # Convert it to CSV
    csv = df_to_download.to_csv(index=False)

    # Add a 'key' to the button.
    # The key changes whenever the number of rows/columns changes,
    # forcing Streamlit to create a new button with the new data.
    st.download_button(
        "⬇️ Download Processed Data",
        csv,
        "processed_data.csv",
        key=f"download-csv-{df_to_download.shape}"
    )
    # --- END OF DOWNLOAD SECTION ---


# -------------------------
# ADVANCED EDA
# -------------------------
if selected == "Advanced EDA":
    df = st.session_state["new_df"]

    st.header("🧠 Advanced EDA")

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("📌 Enhanced Statistical Summary")
    advanced_analysis.show_statistical_summary(df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("📌 Correlation & Bivariate Analysis")
    advanced_analysis.show_correlation_analysis(df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1Date_Time_Conversion_Functions12px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("📌 Advanced Missing Value Report")
    advanced_analysis.show_missing_value_report(df)
    st.markdown("</div>", unsafe_allow_html=True)