
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#f0f5fc;}
.block-container {max-width:1150px;padding-top:2rem;}
h1,h2,h3 {color:#17365d;}
div[data-testid="stMetric"] {
    background:white;
    padding:15px;
    border-radius:12px;
    border:1px solid #dce6f2;
}
div[data-testid="stSidebar"] {background:#e1ecfa;}
.stButton>button {
    background:#2563eb;
    color:white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/amazon_ecommerce_1M.csv")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df = df.dropna(subset=["purchase_date", "final_price"]).copy()
    df = df[df["is_returned"] == False].copy()
    df["month"] = df["purchase_date"].apply(lambda x: str(x)[:7])
    monthly = df.groupby("month")["final_price"].sum().reset_index()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["year"] = monthly["month"].apply(lambda x: x.year)
    monthly["month_number"] = monthly["month"].apply(lambda x: x.month)
    monthly["month_label"] = monthly["month"].apply(lambda x: x.strftime("%b %Y"))
    return df, monthly

@st.cache_resource
def load_model():
    saved = joblib.load("sales_model.pkl")
    return saved["model"], saved["features"]

try:
    df, monthly = load_data()
    model, features = load_model()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

st.title("📈 Retail Sales Forecasting Using Machine Learning")
st.caption("Monthly sales analysis and revenue")

page = st.sidebar.radio("Navigation", ["Sales Dashboard", "Sales Prediction", "Model Evaluation"])

if page == "Sales Dashboard":
    years = sorted(monthly["year"].unique().tolist())
    selected_year = st.selectbox("Select Year", ["All Years"] + years)
    if selected_year == "All Years":
        data = monthly.copy()
        category_data = df.copy()
    else:
        data = monthly[monthly["year"] == selected_year].copy()
        category_data = df[df["purchase_date"].apply(lambda x: x.year) == selected_year].copy()
    col1, col2 = st.columns(2)
    col1.metric("Total Sales", f"{data['final_price'].sum():,.2f}")
    col2.metric("Average Monthly Sales", f"{data['final_price'].mean():,.2f}")
    st.subheader("Monthly Sales Trend")
    fig = px.line(data, x="month_label", y="final_price", markers=True, labels={"month_label": "Month", "final_price": "Monthly Sales"})
    fig.update_traces(line_color="#2563eb")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Category-wise Sales")
    category = category_data.groupby("category")["final_price"].sum().reset_index().sort_values("final_price", ascending=False)
    fig = px.bar(category, x="category", y="final_price", labels={"category": "Category", "final_price": "Total Sales"})
    fig.update_traces(marker_color="#14b8a6")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Sales Prediction":
    st.subheader("Monthly Sales Prediction")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Select Year", min_value=1900, max_value=2100, value=int(monthly["year"].max()) + 1)
    with col2:
        month = st.selectbox("Select Month", months)
    if st.button("Predict Sales", use_container_width=True):
        month_number = months.index(month) + 1
        input_data = pd.DataFrame([{"year": int(year), "month_number": month_number}])[features]
        prediction = float(model.predict(input_data)[0])
        st.markdown("---")
        st.success(f"Predicted Sales for {month} {year}")
        st.metric("Predicted Monthly Sales", f"{prediction:,.2f}")
        st.metric("Historical Monthly Average", f"{monthly['final_price'].mean():,.2f}")

elif page == "Model Evaluation":
    st.subheader("Linear Regression Model Evaluation")
    X = monthly[["year", "month_number"]]
    y = monthly["final_price"]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    if len(X_test) < 2:
        st.warning("Not enough test records for evaluation.")
    else:
        predictions = model.predict(X_test[features])
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("R² Score", f"{r2:.3f}")
        comparison = pd.DataFrame({"Month": monthly.iloc[split:]["month_label"].values, "Actual Sales": y_test.values, "Predicted Sales": predictions})
        st.subheader("Actual vs Predicted Sales")
        chart = comparison.melt(id_vars="Month", var_name="Sales Type", value_name="Sales")
        fig = px.line(chart, x="Month", y="Sales", color="Sales Type", markers=True)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Retail Sales Forecasting Using Machine Learning")
