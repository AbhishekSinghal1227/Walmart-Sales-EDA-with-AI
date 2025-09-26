# frontend.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
from backend import answer_query, fixed_questions, df

st.set_page_config(page_title="Walmart Sales EDA with AI", layout="wide")
st.title("🛒 Walmart Sales EDA with AI")

# ---------------------------
# Dataset Brief
# ---------------------------
st.subheader("Dataset Overview")
st.write("""
This dataset contains **Walmart sales transactions**.  
It includes sales from different branches, invoice information, product quantities, and unit prices.  
You can use this app to explore trends, correlations, missing data, and more using AI-assisted analysis.
""")
st.write("**Columns available in the dataset:**")
st.write(list(df.columns))

st.sidebar.header("Choose an Analysis Option")
choice = st.sidebar.selectbox(
    "Options",
    list(fixed_questions.keys()),
    format_func=lambda x: f"{x}. {fixed_questions[x]}"
)

# ---------------------------
# Custom/general questions (7 & 8)
# ---------------------------
if choice in ["7"]:
    user_q = st.text_area("✍️ Enter your question:", "")
    if st.button("Run Analysis"):
        response = answer_query(choice, user_input=user_q)
        st.subheader("📊 Output")
        st.text(response["output"])

# ---------------------------
# Full analysis (6)
# ---------------------------
elif choice == "6":
    if st.button("Run Full Analysis"):
        response = answer_query("6")
        try:
            parsed = json.loads(response["output"])

            # Summary
            if parsed.get("summary"):
                st.info("📄 **Dataset Summary & Insights:**\n" + parsed["summary"])

            # Top sales
            if parsed.get("top_sales_invoice"):
                st.success(f"💰 **Top Sales Invoice:** {parsed['top_sales_invoice']}")

            # Average sales
            if parsed.get("avg_sales"):
                st.metric("📊 Average Sales", f"{parsed['avg_sales']:.2f}")

            # Missing values
            if parsed.get("missing_values"):
                st.write("❓ **Missing Values per Column:**", parsed["missing_values"])

            # Correlation heatmap
            st.subheader("🔗 Correlation Heatmap")
            st.write("This heatmap shows how numerical columns are correlated. Values close to 1 or -1 indicate strong correlation, while values near 0 indicate little correlation.")
            corr = df.select_dtypes(include='number').corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Numeric plots
            st.subheader("📊 Numeric Columns")
            st.write("Histograms show the distribution of numeric columns. Peaks indicate the most common values.")
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

            # Categorical plots
            st.subheader("📊 Categorical Columns")
            st.write("Bar charts show the frequency of each category in a column.")
            cat_cols = df.select_dtypes(include='object').columns
            for col in cat_cols:
                fig, ax = plt.subplots(figsize=(5,3))
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Bar chart of {col}")
                st.pyplot(fig)

        except Exception:
            st.error("⚠️ Could not parse output")
            st.text(response)

# ---------------------------
# Plots only (5)
# ---------------------------
elif choice == "5":
    if st.button("Generate Plots"):
        st.subheader("📊 Numeric & Categorical Plots")
        st.write("Histograms for numeric columns and bar charts for categorical columns are shown below to help understand distributions.")

        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            st.write(f"**Histogram of {col}:**")
            fig, ax = plt.subplots(figsize=(5,3))
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            st.write(f"**Bar chart of {col}:**")
            fig, ax = plt.subplots(figsize=(5,3))
            df[col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

# ---------------------------
# Correlation only (3)
# ---------------------------
elif choice == "3":
    if st.button("Show Correlation Heatmap"):
        st.subheader("🔗 Correlation Heatmap")
        st.write("Heatmap shows correlations between numerical columns. Helps identify relationships between variables.")
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.write("Correlation Matrix:", corr)

# ---------------------------
# Other options (1,2,4)
# ---------------------------
else:
    if st.button("Run Analysis"):
        response = answer_query(choice)
        st.subheader("📌 Walmart EDA Report")
        st.text(response["output"])

