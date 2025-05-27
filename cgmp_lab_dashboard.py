import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from datetime import datetime

# ========== Cached File Reader ==========
@st.cache_data(show_spinner=False)
def load_file(file):
    if file.name.endswith("xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    return pd.read_csv(file)

# ========== CSV Download Button ==========
def generate_download_link(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ========== Dashboard Header ==========
st.set_page_config(page_title="📊 Lab Dashboard", layout="wide")
st.title("📊 Houston KPI Dashboard")

# ========== File Upload UI ==========
uploaded_files = st.file_uploader(
    "📁 Upload Lab File(s)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True, 
    key="lab_upload"
)

# ========== STOP HERE IF NO FILE ==========
if not uploaded_files:
    st.info("📂 Please upload one or more lab files to begin.")
    st.stop()  # ✅ Prevents any further code from running

# ========== File Processing ==========
df = pd.DataFrame()
df_raw = pd.DataFrame()
canceled_df = pd.DataFrame()

df_list = []
for file in uploaded_files:
    temp_df = load_file(file)

    # Safe parsing
    if "Submission_Release_Date" in temp_df.columns:
        temp_df["Submission_Release_Date"] = pd.to_datetime(temp_df["Submission_Release_Date"], errors="coerce")
    if "Price" in temp_df.columns:
        temp_df["Price"] = pd.to_numeric(temp_df["Price"], errors="coerce")

    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df_raw = df.copy()

if "Status" in df.columns:
    canceled_df = df[df["Status"] == "Canceled"].copy()
    df = df[df["Status"] != "Canceled"].copy()


# ========== Preprocessing ==========

# Parse date columns safely
for col in ["Completion_Date", "Estimated_Completion_Date", "Submission_Release_Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Add derived date parts only if Submission_Release_Date is valid
if "Submission_Release_Date" in df.columns:
    df["Week"] = df["Submission_Release_Date"].dt.isocalendar().week
    df["Month"] = df["Submission_Release_Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Submission_Release_Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Submission_Release_Date"].dt.year

# Compute Delay only for Completed tests
if all(col in df.columns for col in ["Status", "Completion_Date", "Estimated_Completion_Date"]):
    completed_mask = df["Status"] == "Completed"
    df.loc[completed_mask, "Delay_Days"] = (
        df.loc[completed_mask, "Completion_Date"] - df.loc[completed_mask, "Estimated_Completion_Date"]
    ).dt.days

# ========== Sidebar Filters ==========

st.sidebar.header("Filter Panel")

# Multi-select filters with safe access
def safe_multiselect(label, col):
    if col in df.columns:
        return st.sidebar.multiselect(label, sorted(df[col].dropna().unique()))
    return []

test_filter = safe_multiselect("Test Name", "Test_Name")
client_filter = safe_multiselect("Client Facility", "Client_Facility")
analyst_filter = safe_multiselect("Analyst", "Analyst")
facility_type_filter = safe_multiselect("Facility Type", "Facility_Type")  # fixed typo from "Facilty_Type"
rush_filter = safe_multiselect("Rush", "Rush")
control_filter = safe_multiselect("Controlled Substance Class", "Controlled_Substance_Class")
hazardous_filter = safe_multiselect("Hazardous Drug", "Hazardous_Drug")
status_filter = safe_multiselect("Test Status", "Status")

sample_search = st.sidebar.text_input("Search Sample Name")

# ========== Date Range Filter ==========

if "Submission_Release_Date" in df.columns:
    min_date = df["Submission_Release_Date"].min()
    max_date = df["Submission_Release_Date"].max()
    st.sidebar.markdown("### Release Date Filter")
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    df = df[
        (df["Submission_Release_Date"] >= pd.to_datetime(start_date)) &
        (df["Submission_Release_Date"] <= pd.to_datetime(end_date))
    ]

# ========== Apply Filters ==========

if test_filter:
    df = df[df["Test_Name"].isin(test_filter)]
if client_filter:
    df = df[df["Client_Facility"].isin(client_filter)]
if analyst_filter:
    df = df[df["Analyst"].isin(analyst_filter)]
if facility_type_filter:
    df = df[df["Facility_Type"].isin(facility_type_filter)]
if rush_filter:
    df = df[df["Rush"].isin(rush_filter)]
if control_filter:
    df = df[df["Controlled_Substance_Class"].isin(control_filter)]
if hazardous_filter:
    df = df[df["Hazardous_Drug"].isin(hazardous_filter)]
if status_filter:
    df = df[df["Status"].isin(status_filter)]
if sample_search and "Sample_Name" in df.columns:
    df = df[df["Sample_Name"].astype(str).str.contains(sample_search, case=False, na=False)]

# ========== Tabs ==========

tabs = st.tabs([
    "Executive Overview", 
    "Client Analysis", 
    "Test Performance", 
    "Delays & TAT", 
    "Forecasting", 
    "Statistical Insights"])

# ========== Executive Overview Tab ==========

with tabs[0]:
    st.header("📊 Executive Overview")

    # Define safe test_df and lost revenue
    test_df = canceled_df.copy() if not canceled_df.empty else pd.DataFrame(columns=["Price"])
    test_df["Price"] = pd.to_numeric(test_df.get("Price", pd.NA), errors="coerce")
    test_revenue = test_df["Price"].sum()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Tests", f"{df.shape[0]:,}")
    if "Price" in df.columns:
        col2.metric("Total Revenue", f"${df['Price'].sum():,.2f}")
    else:
        col2.metric("Total Revenue", "N/A")
    col3.metric("Avg Price/Test", f"${df['Price'].mean():.2f}")
    col4.metric("Lost Revenue (Canceled)", f"${test_revenue:,.2f}", delta_color="inverse")

    # Top 5 Tests by Volume
    st.subheader("📊 Top 5 Tests by Volume")
    if "Test_Name" in df.columns:
        top_tests = df["Test_Name"].value_counts().nlargest(5).reset_index(name="Count").rename(columns={"index": "Test_Name"})
        fig1 = px.bar(
            top_tests, x="Test_Name", y="Count", text="Count",
            title="Top Tests by Volume"
        )
        fig1.update_traces(marker_color="#4c78a8", textposition="outside")
        fig1.update_layout(yaxis_title="Test Count", xaxis_title="Test Name")
        st.plotly_chart(fig1, use_container_width=True, key="fig1_volume")
        generate_download_link(top_tests, "top_tests_volume.csv")

    # Top 5 Tests by Revenue
    st.subheader("💰 Top 5 Tests by Revenue")
    if "Test_Name" in df.columns and "Price" in df.columns:
        top_revenue = df.groupby("Test_Name")["Price"].sum().nlargest(5).reset_index()
        fig2 = px.bar(
            top_revenue, x="Test_Name", y="Price", text="Price",
            labels={"Price": "Revenue"}, title="Top Tests by Revenue"
        )
        fig2.update_traces(marker_color="#59a14f", texttemplate="$%{text:.2f}", textposition="outside")
        fig2.update_layout(yaxis_title="Total Revenue", xaxis_title="Test Name")
        st.plotly_chart(fig2, use_container_width=True, key="fig2_top_revenue")
        generate_download_link(top_revenue, "top_tests_revenue.csv")

    # Revenue by Facility Type
    st.subheader("🏭 Revenue by Facility Type")
    if "Facility_Type" in df.columns and "Price" in df.columns:
        rev_by_facility = df.groupby("Facility_Type")["Price"].sum().reset_index()
        fig3 = px.pie(
            rev_by_facility, names="Facility_Type", values="Price", hole=0.4,
            title="Revenue Contribution by Facility Type"
        )
        st.plotly_chart(fig3, use_container_width=True, key="fig3_facility_pie")
        generate_download_link(rev_by_facility, "revenue_by_facility_type.csv")


        

# === Client Analysis === #
with tabs[1]:
    st.header("🏥 Client-Level Performance Analysis")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clients", df["Client_Facility"].nunique() if "Client_Facility" in df.columns else 0)

    if not df.empty and "Client_Facility" in df.columns and "Price" in df.columns:
        top_client = df.groupby("Client_Facility")["Price"].sum().idxmax()
    else:
        top_client = "N/A"
    col2.metric("Top Client by Revenue", top_client)

    top_canceled_test = (
        canceled_df["Test_Name"].value_counts().idxmax()
        if not canceled_df.empty and "Test_Name" in canceled_df.columns else "N/A"
    )
    canceled_count = (
        canceled_df["Test_Name"].value_counts().max()
        if not canceled_df.empty and "Test_Name" in canceled_df.columns else 0
    )
    col3.metric("Top Canceled Test", f"{top_canceled_test} ({canceled_count})")

    # --- Volume by Client ---
    st.subheader("🧪 Test Volume by Client")
    if "Client_Facility" in df.columns:
        volume_by_client = df["Client_Facility"].value_counts().reset_index(name="Test_Count").rename(columns={"index": "Client_Facility"})
        fig4 = px.bar(volume_by_client.head(10), x="Client_Facility", y="Test_Count", text="Test_Count", title="Top Clients by Test Volume")
        fig4.update_layout(yaxis_title="Number of Tests", xaxis_title="Client")
        fig4.update_traces(marker_color="#4c78a8", textposition="outside")
        st.plotly_chart(fig4, use_container_width=True, key="fig4_volume_client")
        generate_download_link(volume_by_client, "volume_by_client.csv")

    # --- Revenue by Client & Facility Type ---
    st.subheader("💰 Revenue by Client & Facility Type")
    if all(col in df.columns for col in ["Client_Facility", "Facility_Type", "Price"]):
        client_facility_rev = df.groupby(["Client_Facility", "Facility_Type"])["Price"].sum().reset_index()
        client_facility_rev = client_facility_rev.sort_values("Price", ascending=False).head(10)
        fig_fac = px.bar(
            client_facility_rev,
            x="Client_Facility", y="Price", color="Facility_Type", text="Price",
            title="Top Clients by Revenue (Grouped by Facility Type)"
        )
        fig_fac.update_layout(yaxis_title="Revenue ($)", xaxis_title="Client")
        st.plotly_chart(fig_fac, use_container_width=True, key="fig5_revenue_client_facility")
        generate_download_link(client_facility_rev, "revenue_by_client_facility.csv")

    # --- Cancellation Rate ---
    st.subheader("📉 Cancellation Rate by Client")
    if "Client_Facility" in canceled_df.columns and "Client_Facility" in df_raw.columns:
        canceled_counts = canceled_df["Client_Facility"].value_counts()
        total_counts_all = df_raw["Client_Facility"].value_counts()
        cancel_rate = (canceled_counts / total_counts_all * 100).fillna(0).round(2).reset_index()
        cancel_rate.columns = ["Client_Facility", "Canceled %"]
        fig6 = px.bar(
            cancel_rate.sort_values("Canceled %", ascending=False).head(10),
            x="Client_Facility", y="Canceled %", text="Canceled %",
            title="Top Clients by Cancellation Rate"
        )
        fig6.update_traces(marker_color="#e45756", textposition="outside")
        st.plotly_chart(fig6, use_container_width=True, key="fig6_cancel_client")
        generate_download_link(cancel_rate, "cancellation_rate_by_client.csv")

    # --- Top Tests per Client & Facility Type ---
    st.subheader("🔍 Top Tests per Client & Facility Type")
    if all(col in df.columns for col in ["Client_Facility", "Facility_Type", "Test_Name"]):
        test_client_group = df.groupby(["Client_Facility", "Facility_Type", "Test_Name"]).size().reset_index(name="Count")
        top_tests_per_client = test_client_group.sort_values(["Client_Facility", "Count"], ascending=[True, False])
        st.dataframe(top_tests_per_client)
        generate_download_link(top_tests_per_client, "top_tests_by_client_facility.csv")



# === Test Performance === #
with tabs[2]:
    st.header("🧪 Test-Level Performance")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tests", df["Test_Name"].nunique() if "Test_Name" in df.columns else 0)

    if not df.empty and "Test_Name" in df.columns and "Price" in df.columns:
        top_test = df.groupby("Test_Name")["Price"].sum().idxmax()
    else:
        top_test = "N/A"
    col2.metric("Top Revenue-Generating Test", top_test)

    if not df.empty and all(col in df.columns for col in ["Status", "Delay_Days", "Test_Name"]):
        completed_df = df[df["Status"] == "Completed"]
        if not completed_df.empty:
            most_delayed_test = (
                completed_df.groupby("Test_Name")["Delay_Days"]
                .mean()
                .sort_values(ascending=False)
                .idxmax()
            )
        else:
            most_delayed_test = "N/A"
    else:
        most_delayed_test = "N/A"
    col3.metric("Most Delayed Test (Avg)", most_delayed_test)

    # --- Volume by Test ---
    st.subheader("📊 Volume by Test")
    if "Test_Name" in df.columns:
        test_volume = df["Test_Name"].value_counts().reset_index(name="Volume").rename(columns={"index": "Test_Name"})
        fig7 = px.bar(test_volume.head(10), x="Test_Name", y="Volume", text="Volume", title="Top 10 Tests by Volume")
        fig7.update_traces(marker_color="#4c78a8", textposition="outside")
        st.plotly_chart(fig7, use_container_width=True, key="fig7_volume")
        generate_download_link(test_volume, "test_volume.csv")

    # --- Revenue by Test ---
    st.subheader("💵 Revenue by Test")
    if all(col in df.columns for col in ["Test_Name", "Price"]):
        test_revenue_data = df.groupby("Test_Name")["Price"].sum().nlargest(10).reset_index()
        fig8 = px.bar(test_revenue_data, x="Test_Name", y="Price", text="Price", title="Top 10 Revenue-Generating Tests")
        fig8.update_traces(marker_color="#59a14f", texttemplate="$%{text:.2f}", textposition="outside")
        st.plotly_chart(fig8, use_container_width=True, key="fig8_revenue")
        generate_download_link(test_revenue_data, "test_revenue.csv")

    # --- Delay by Test (Completed Only) ---
    st.subheader("⏳ Average Delay by Test")
    if all(col in df.columns for col in ["Status", "Test_Name", "Delay_Days"]):
        completed_only = df[df["Status"] == "Completed"]
        delay_by_test = completed_only.groupby("Test_Name")["Delay_Days"].mean().dropna().reset_index()
        fig9 = px.bar(delay_by_test.sort_values("Delay_Days", ascending=False).head(10),
                      x="Test_Name", y="Delay_Days", text="Delay_Days", title="Most Delayed Tests")
        fig9.update_traces(marker_color="#f28e2b", textposition="outside")
        st.plotly_chart(fig9, use_container_width=True, key="fig9_delay")
        generate_download_link(delay_by_test, "average_delay_by_test.csv")

    # --- Analyst Breakdown ---
    st.subheader("🧑‍🔬 Analyst Breakdown by Test")
    if all(col in df.columns for col in ["Test_Name", "Analyst", "Price", "Delay_Days"]):
        analyst_test = df.groupby(["Test_Name", "Analyst"]).agg(
            Total_Revenue=("Price", "sum"),
            Avg_Delay=("Delay_Days", "mean"),
            Volume=("Test_Name", "count")
        ).reset_index()
        st.dataframe(analyst_test.sort_values("Total_Revenue", ascending=False).round(2))
        generate_download_link(analyst_test, "analyst_breakdown_by_test.csv")

    # --- Volume by Test & Facility Type ---
    st.subheader("🏷 Volume by Test & Facility Type")
    if all(col in df.columns for col in ["Test_Name", "Facility_Type"]):
        volume_by_test_facility = df.groupby(["Test_Name", "Facility_Type"]).size().reset_index(name="Volume")
        fig10 = px.bar(volume_by_test_facility.sort_values("Volume", ascending=False).head(15),
                       x="Test_Name", y="Volume", color="Facility_Type", text="Volume",
                       barmode="group", title="Top Test Volumes by Facility Type")
        fig10.update_layout(xaxis_title="Test Name", yaxis_title="Volume", bargap=0.3)
        st.plotly_chart(fig10, use_container_width=True, key="fig10_volume_fac")
        generate_download_link(volume_by_test_facility, "volume_by_test_facility.csv")

    # --- Revenue by Test & Facility Type ---
    st.subheader("💰 Revenue by Test & Facility Type")
    if all(col in df.columns for col in ["Test_Name", "Facility_Type", "Price"]):
        rev_by_test_facility = df.groupby(["Test_Name", "Facility_Type"])["Price"].sum().reset_index()
        fig11 = px.bar(rev_by_test_facility.sort_values("Price", ascending=False).head(15),
                       x="Test_Name", y="Price", color="Facility_Type", text="Price",
                       barmode="group", title="Top Test Revenues by Facility Type")
        fig11.update_layout(xaxis_title="Test Name", yaxis_title="Revenue", bargap=0.3)
        st.plotly_chart(fig11, use_container_width=True, key="fig11_revenue_fac")
        generate_download_link(rev_by_test_facility, "revenue_by_test_facility.csv")


# === Delays & TAT === #
with tabs[3]:
    st.header("⏱ Delays and Turnaround Time")

    # Filter only valid completed rows with usable delay values
    if all(col in df.columns for col in ["Status", "Delay_Days"]):
        completed_df = df[df["Status"] == "Completed"].copy()
        completed_df = completed_df[completed_df["Delay_Days"].notna()]
        completed_df = completed_df[
            (completed_df["Delay_Days"] > -30) & 
            (completed_df["Delay_Days"] < 60)
        ]

        # --- Delay Distribution ---
        st.subheader("📊 Distribution of Completion Delays")
        fig10 = px.histogram(completed_df, x="Delay_Days", nbins=40, title="Delay Distribution (Days)")
        fig10.update_traces(marker_color="#4c78a8", marker_line_color="black", marker_line_width=0.5)
        fig10.update_layout(xaxis_title="Delay (Days)", yaxis_title="Test Count")
        st.plotly_chart(fig10, use_container_width=True, key="fig10_delay_dist")

        # --- On-Time vs Delayed ---
        st.subheader("✅ On-Time vs Delayed Completion")
        completed_df["Status_Eval"] = np.where(completed_df["Delay_Days"] > 0, "Delayed", "On-Time")
        status_count = completed_df["Status_Eval"].value_counts().reset_index(name="Count").rename(columns={"index": "Status_Eval"})
        fig11 = px.pie(status_count, names="Status_Eval", values="Count", hole=0.4, title="Proportion of On-Time vs Delayed Tests")
        st.plotly_chart(fig11, use_container_width=True, key="fig11_ontime_pie")

        # --- Rush vs Non-Rush Comparison ---
        st.subheader("⚡ Rush vs Non-Rush Turnaround Comparison")
        if "Rush" in completed_df.columns:
            rush_df = completed_df[completed_df["Rush"].isin(["Y", "N"])]
            rush_delay = rush_df.groupby("Rush")["Delay_Days"].agg(["mean", "count"]).reset_index().round(2)
            fig12 = px.bar(rush_delay, x="Rush", y="mean", text="mean", title="Avg Delay by Rush Category")
            fig12.update_traces(marker_color=["#59a14f", "#f28e2b"], textposition="outside")
            fig12.update_layout(xaxis_title="Rush Category", yaxis_title="Average Delay (Days)")
            st.plotly_chart(fig12, use_container_width=True, key="fig12_rush")
            generate_download_link(rush_delay, "rush_vs_nonrush_delay.csv")

        # --- Delay by Test & Facility Type ---
        st.subheader("🔍 Avg Delay by Test & Facility Type")
        if all(col in completed_df.columns for col in ["Test_Name", "Facility_Type"]):
            delay_fac = completed_df.groupby(["Test_Name", "Facility_Type"])["Delay_Days"].mean().dropna().reset_index()
            fig14 = px.bar(
                delay_fac.sort_values("Delay_Days", ascending=False).head(20),
                x="Test_Name", y="Delay_Days", color="Facility_Type", barmode="group",
                text="Delay_Days", title="Top Delays by Test and Facility Type"
            )
            fig14.update_layout(xaxis_title="Test Name", yaxis_title="Avg Delay (Days)", bargap=0.3)
            st.plotly_chart(fig14, use_container_width=True, key="fig14_delay_fac")

        # --- Delay Table Preview ---
        st.subheader("📋 Delay Data Preview")
        preview_cols = ["Submission_Release_Date", "Test_Name", "Client_Facility", "Analyst", "Delay_Days", "Rush"]
        preview_cols = [col for col in preview_cols if col in completed_df.columns]
        st.dataframe(completed_df[preview_cols].head(100))

        # --- Download Buttons ---
        generate_download_link(completed_df, "completed_tests_delay_data.csv")
        generate_download_link(status_count, "ontime_vs_delayed.csv")
    else:
        st.warning("Delay and completion status columns not found in data.")

# === Forecasting === #

with tabs[4]:
    st.header("📊 Forecasting with Prophet (Selected Tests Only)")

    confidence_level = st.slider("Confidence Level (%)", 50, 99, 95)
    forecast_periods = st.slider("Periods to Predict", 4, 20, 8)
    forecast_freq = st.radio("🔁 Forecast Granularity", ["Daily", "Monthly"], horizontal=True)
    freq = "D" if forecast_freq == "Daily" else "M"

    if not test_filter:
        st.info("📌 Please select one or more tests from the sidebar to view forecast.")
        st.stop()

    selected_tests = test_filter
    base = df[(df["Status"] == "Completed") & df["Submission_Release_Date"].notna()].copy()
    base["ds"] = pd.to_datetime(base["Submission_Release_Date"], errors="coerce")

    for test in selected_tests:
        st.subheader(f"🔍 Forecast for: {test}")
        test_df = base[base["Test_Name"] == test].copy().dropna(subset=["ds", "Price"])

        if test_df.empty or test_df["ds"].nunique() < 4:
            st.warning(f"Not enough valid data to forecast '{test}'")
            continue

        try:
            agg_df = test_df.groupby(pd.Grouper(key="ds", freq=freq)).agg(
                y=("Test_Name", "count"),
                price_avg=("Price", "mean")
            ).fillna(0).reset_index()

            # Remove periods with zero tests (e.g., weekends)
            agg_df = agg_df[agg_df["y"] > 0]

            from prophet import Prophet
            m = Prophet(interval_width=confidence_level / 100)
            m.fit(agg_df[["ds", "y"]])
            future = m.make_future_dataframe(periods=forecast_periods, freq=freq)
            forecast = m.predict(future).merge(agg_df, on="ds", how="left")

            forecast["price_avg"] = forecast["price_avg"].ffill().replace(0, np.nan).fillna(method="bfill")
            forecast["Predicted Volume"] = forecast["yhat"].clip(lower=0)
            forecast["Actual Volume"] = forecast["y"]
            forecast["Predicted Revenue"] = forecast["Predicted Volume"] * forecast["price_avg"]
            forecast["Actual Revenue"] = forecast["Actual Volume"] * forecast["price_avg"]
            forecast["Predicted Revenue Upper"] = forecast["yhat_upper"] * forecast["price_avg"]
            forecast["Predicted Revenue Lower"] = forecast["yhat_lower"] * forecast["price_avg"]

            # === Metrics & Trend Lines ===
            import plotly.graph_objects as go
            from sklearn.linear_model import LinearRegression
            import numpy as np

            def add_stats(title, y_true, y_pred, model=None):
                r2 = np.corrcoef(y_true, y_pred)[0, 1]**2 if len(y_true) > 1 else 0
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mase = mae / np.mean(np.abs(y_true - y_true.shift(1)).dropna()) if len(y_true) > 1 else 0
                smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
                alpha = round(model.params["m"].mean(), 4) if model else np.nan
                beta = round(model.params["k"].mean(), 4) if model else np.nan
                gamma = round(np.std(model.params["delta"]) if "delta" in model.params else 0.0, 4) if model else np.nan

                metrics = pd.DataFrame({
                    "Metric": ["R²", "MAE", "RMSE", "MASE", "SMAPE", "Alpha", "Beta", "Gamma"],
                    "Value": [r2, mae, rmse, mase, smape, alpha, beta, gamma]
                }).round(3)
                st.markdown(f"### 📊 {title} Metrics")
                st.dataframe(metrics, use_container_width=True)

            eval_df = forecast.dropna(subset=["Actual Volume"])
            X = np.arange(len(forecast)).reshape(-1, 1)

            # === Volume Stats + Graph ===
            add_stats("Volume Forecast", eval_df["Actual Volume"], eval_df["Predicted Volume"], m)
            lr = LinearRegression().fit(X[:len(eval_df)], eval_df["Actual Volume"])
            forecast["Vol_Linear"] = lr.predict(X)
            y_log = np.log1p(eval_df["Actual Volume"])
            exp_model = LinearRegression().fit(X[:len(y_log)], y_log)
            forecast["Vol_Exp"] = np.expm1(exp_model.predict(X))

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", fillcolor="rgba(76, 120, 168, 0.2)", line=dict(width=0), name="Confidence Band"))
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Actual Volume"], mode="lines+markers", name="Actual", line=dict(color="white")))
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Predicted Volume"], mode="lines", name="Forecast", line=dict(color="lightblue")))
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Vol_Linear"], mode="lines", name="Linear Trend", line=dict(color="orange", dash="dot")))
            fig_vol.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Vol_Exp"], mode="lines", name="Exponential Trend", line=dict(color="teal", dash="dash")))
            fig_vol.update_layout(title=f"{test} - Test Volume Forecast", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True)

            # === Volume Summary ===
            forecast["Month"] = forecast["ds"].dt.to_period("M")
            forecast["Quarter"] = forecast["ds"].dt.to_period("Q")
            future_only = forecast[forecast["ds"] > test_df["ds"].max()]
            daily_avg = pd.DataFrame([{
                "Group_Type": f"{forecast_freq} (avg)",
                "Group": f"{future_only['ds'].min().date()} to {future_only['ds'].max().date()}",
                "Predicted Volume": round(future_only["Predicted Volume"].mean(), 2),
                "Actual Volume": np.nan
            }]) if not future_only.empty else pd.DataFrame()

            month_summary = forecast.groupby("Month").agg({"Predicted Volume": "sum", "Actual Volume": "sum"}).reset_index().rename(columns={"Month": "Group"}).assign(Group_Type="Month")
            quarter_summary = forecast.groupby("Quarter").agg({"Predicted Volume": "sum", "Actual Volume": "sum"}).reset_index().rename(columns={"Quarter": "Group"}).assign(Group_Type="Quarter")

            st.subheader("📋 Volume Forecast Summary")
            st.dataframe(pd.concat([daily_avg, month_summary, quarter_summary], ignore_index=True), use_container_width=True)

            # === Revenue Stats + Graph ===
            eval_df_r = forecast.dropna(subset=["Actual Revenue"])
            add_stats("Revenue Forecast", eval_df_r["Actual Revenue"], eval_df_r["Predicted Revenue"], m)

            lr_r = LinearRegression().fit(X[:len(eval_df_r)], eval_df_r["Actual Revenue"])
            forecast["Rev_Linear"] = lr_r.predict(X)
            y_log_r = np.log1p(eval_df_r["Actual Revenue"])
            exp_model_r = LinearRegression().fit(X[:len(y_log_r)], y_log_r)
            forecast["Rev_Exp"] = np.expm1(exp_model_r.predict(X))

            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue Upper"], line=dict(width=0), showlegend=False))
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue Lower"], fill="tonexty", fillcolor="rgba(89, 161, 79, 0.2)", line=dict(width=0), name="Confidence Band"))
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Actual Revenue"], mode="lines+markers", name="Actual", line=dict(color="white")))
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue"], mode="lines", name="Forecast", line=dict(color="green")))
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Rev_Linear"], mode="lines", name="Linear Trend", line=dict(color="purple", dash="dot")))
            fig_rev.add_trace(go.Scatter(x=forecast["ds"], y=forecast["Rev_Exp"], mode="lines", name="Exponential Trend", line=dict(color="gold", dash="dash")))
            fig_rev.update_layout(title=f"{test} - Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue")
            st.plotly_chart(fig_rev, use_container_width=True)

            daily_avg_r = pd.DataFrame([{
                "Group_Type": f"{forecast_freq} (avg)",
                "Group": f"{future_only['ds'].min().date()} to {future_only['ds'].max().date()}",
                "Predicted Revenue": round(future_only["Predicted Revenue"].mean(), 2),
                "Actual Revenue": np.nan
            }]) if not future_only.empty else pd.DataFrame()

            month_summary_r = forecast.groupby("Month").agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"}).reset_index().rename(columns={"Month": "Group"}).assign(Group_Type="Month")
            quarter_summary_r = forecast.groupby("Quarter").agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"}).reset_index().rename(columns={"Quarter": "Group"}).assign(Group_Type="Quarter")

            st.subheader("📋 Revenue Forecast Summary")
            st.dataframe(pd.concat([daily_avg_r, month_summary_r, quarter_summary_r], ignore_index=True), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Forecasting failed for {test}: {e}")


# === Statistical Insights === #
with tabs[5]:
    st.header("📐 Statistical Insights by Test")

    # 1. Revenue Summary by Test
    st.subheader("📊 Revenue Summary by Test")
    if all(col in df.columns for col in ["Test_Name", "Price"]):
        test_summary = df.groupby("Test_Name")["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean",
            Std_Dev="std"
        ).reset_index().round(2)
        st.dataframe(test_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(test_summary, "test_summary.csv")
    else:
        st.warning("Missing columns: 'Test_Name' or 'Price'")

    # 2. Revenue Summary by Client
    st.subheader("🏢 Revenue Summary by Client")
    if all(col in df.columns for col in ["Client_Facility", "Price"]):
        client_summary = df.groupby("Client_Facility")["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean",
            Std_Dev="std"
        ).reset_index().round(2)
        st.dataframe(client_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(client_summary, "client_summary.csv")
    else:
        st.warning("Missing columns: 'Client_Facility' or 'Price'")

    # 3. Revenue Summary by Test & Client
    st.subheader("🔗 Revenue Summary by Test & Client")
    if all(col in df.columns for col in ["Test_Name", "Client_Facility", "Price"]):
        combo_summary = df.groupby(["Test_Name", "Client_Facility"])["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean"
        ).reset_index().round(2)
        st.dataframe(combo_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(combo_summary, "test_client_summary.csv")
    else:
        st.warning("Missing columns: 'Test_Name', 'Client_Facility', or 'Price'")

    # 4. Revenue Summary by Facility Type
    st.subheader("🏬 Revenue Summary by Facility Type")
    if all(col in df.columns for col in ["Facility_Type", "Price"]):
        facility_summary = df.groupby("Facility_Type")["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean",
            Std_Dev="std"
        ).reset_index().round(2)
        st.dataframe(facility_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(facility_summary, "facility_summary.csv")
    else:
        st.warning("Missing columns: 'Facility_Type' or 'Price'")

    # 5. Revenue Histogram
    st.subheader("📈 Revenue Distribution per Test")
    if "Price" in df.columns:
        fig = px.histogram(df, x="Price", nbins=50, title="Distribution of Test Prices", marginal="box")
        fig.update_layout(bargap=0.1, height=400)
        fig.update_traces(marker_line_width=0.6, marker_line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing column: 'Price'")

