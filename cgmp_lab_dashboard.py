# cgmp_lab_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import io

# --- Global CSV Export Helper ---
def generate_download_link(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# --- Page Config ---
st.set_page_config(page_title="Houston KPI Dashboard", page_icon=":microscope:", layout="wide")
st.title("Houston KPI Dashboard")

# --- File Upload ---
uploaded_files = st.file_uploader("📁 Upload One or More Lab Files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        temp_df = pd.read_excel(file, engine="openpyxl") if file.name.endswith("xlsx") else pd.read_csv(file)
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
else:
    st.stop()

# --- Preprocessing ---
df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
df["Completion_Date"] = pd.to_datetime(df["Completion_Date"], errors="coerce")
df["Estimated_Completion_Date"] = pd.to_datetime(df["Estimated_Completion_Date"], errors="coerce")
df = df[df["Status"].notna() & df["Status"] != "Canceled"]

# Derived Time Fields
df["Week"] = df["Submission_Date"].dt.isocalendar().week
df["Month"] = df["Submission_Date"].dt.to_period("M").astype(str)
df["Quarter"] = df["Submission_Date"].dt.to_period("Q").astype(str)
df["Year"] = df["Submission_Date"].dt.year

# Delay Calculation
mask = df["Status"] == "Completed"
df.loc[mask, "Delay_Days"] = (df.loc[mask, "Completion_Date"] - df.loc[mask, "Estimated_Completion_Date"]).dt.days

# --- Sidebar Filters ---
st.sidebar.header("Filter Panel")
with st.sidebar:
    test_options = df["Test_Name"].dropna().astype(str).unique()
    test_filter = st.multiselect("Test Name", sorted(test_options))
    selected_tests = test_filter 

    client_filter = st.multiselect("Client Facility", sorted(df["Client_Facility"].dropna().astype(str).unique()))
    analyst_filter = st.multiselect("Analyst", sorted(df["Analyst"].dropna().astype(str).unique()))
    facility_type_filter = st.multiselect("Facility Type", sorted(df["Facilty_Type"].dropna().astype(str).unique()))
    rush_filter = st.multiselect("Rush", sorted(df["Rush"].dropna().astype(str).unique()))
    control_filter = st.multiselect("Controlled Substance Class", sorted(df["Controlled_Substance_Class"].dropna().astype(str).unique()))
    hazardous_filter = st.multiselect("Hazardous Drug", sorted(df["Hazardous_Drug"].dropna().astype(str).unique()))
    status_filter = st.multiselect("Test Status", sorted(df["Status"].dropna().astype(str).unique()))
    sample_search = st.text_input("Search Sample Name")
# Date range picker
    min_date = df["Submission_Date"].min()
    max_date = df["Submission_Date"].max()
    st.markdown("### Submission Date Filter")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)


# Filter by date range
df = df[(df["Submission_Date"] >= pd.to_datetime(start_date)) & (df["Submission_Date"] <= pd.to_datetime(end_date))]


# --- Apply Filters ---
if test_filter:
    df = df[df["Test_Name"].astype(str).isin(test_filter)]
if client_filter:
    df = df[df["Client_Facility"].astype(str).isin(client_filter)]
if analyst_filter:
    df = df[df["Analyst"].astype(str).isin(analyst_filter)]
if facility_type_filter:
    df = df[df["Facilty_Type"].astype(str).isin(facility_type_filter)]
if rush_filter:
    df = df[df["Rush"].astype(str).isin(rush_filter)]
if control_filter:
    df = df[df["Controlled_Substance_Class"].astype(str).isin(control_filter)]
if hazardous_filter:
    df = df[df["Hazardous_Drug"].astype(str).isin(hazardous_filter)]
if status_filter:
    df = df[df["Status"].astype(str).isin(status_filter)]
if sample_search:
    df = df[df["Sample_Name"].astype(str).str.contains(sample_search, case=False, na=False)]

# --- Tabs ---
tabs = st.tabs(["Executive Overview", "Client Analysis", "Test Performance", "Delays & TAT", "Forecasting", "Statistical Insights"])

# === Executive Overview === #
with tabs[0]:
    st.header("Company-Wide Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tests", f"{df.shape[0]:,}")
    col2.metric("Total Revenue", f"${df['Price'].sum():,.2f}")
    col3.metric("Avg Price/Test", f"${df['Price'].mean():.2f}")

    st.subheader("Top 5 Tests by Volume")
    top_tests = df["Test_Name"].value_counts().nlargest(5).reset_index(name="count").rename(columns={"index": "Test_Name"})
    fig1 = px.bar(top_tests, x="Test_Name", y="count", text="count")
    st.plotly_chart(fig1, use_container_width=True, key="fig1_volume")

    st.subheader("Top 5 Tests by Revenue")
    top_revenue = df.groupby("Test_Name")["Price"].sum().nlargest(5).reset_index()
    fig2 = px.bar(top_revenue, x="Test_Name", y="Price", text="Price", labels={"Price": "Revenue"})
    st.plotly_chart(fig2, use_container_width=True, key="fig2_top_revenue")

    st.subheader("Revenue by Facility Type")
    rev_by_facility = df.groupby("Facilty_Type")["Price"].sum().reset_index()
    fig3 = px.pie(rev_by_facility, names="Facilty_Type", values="Price", hole=0.4)
    st.plotly_chart(fig3, use_container_width=True,key="fig3_facility_pie" )

    generate_download_link(top_tests, "top_tests_volume.csv")
    generate_download_link(top_revenue, "top_tests_revenue.csv")
    generate_download_link(rev_by_facility, "revenue_by_facility_type.csv")


# === Client Analysis === #
with tabs[1]:
    st.header("Client-Level Performance Analysis")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clients", df["Client_Facility"].nunique())
    top_client = df.groupby("Client_Facility")["Price"].sum().idxmax()
    col2.metric("Top Client by Revenue", top_client)
    col3.metric("Canceled Tests", df[df["Status"] == "Canceled"].shape[0])

    st.subheader("Test Volume by Client")
    volume_by_client = df["Client_Facility"].value_counts().reset_index(name="Test_Count").rename(columns={"index": "Client_Facility"})
    fig4 = px.bar(volume_by_client.head(10), x="Client_Facility", y="Test_Count", text="Test_Count")
    st.plotly_chart(fig4, use_container_width=True, key="fig4_volume_client")

    st.subheader("Revenue by Client")
    revenue_by_client = df.groupby("Client_Facility")["Price"].sum().nlargest(10).reset_index()
    fig5 = px.bar(revenue_by_client, x="Client_Facility", y="Price", text="Price", labels={"Price": "Revenue"})
    st.plotly_chart(fig5, use_container_width=True, key="fig5_revenue_client")

    st.subheader("Cancellation Rate by Client")
    cancel_df = df.copy()
    cancel_df["Canceled"] = cancel_df["Status"] == "Canceled"
    cancel_rate = cancel_df.groupby("Client_Facility")[["Canceled"]].mean().reset_index()
    cancel_rate["Canceled"] = (cancel_rate["Canceled"] * 100).round(2)
    fig6 = px.bar(cancel_rate.sort_values("Canceled", ascending=False).head(10), x="Client_Facility", y="Canceled",
                  text="Canceled", labels={"Canceled": "% Canceled"})
    st.plotly_chart(fig6, use_container_width=True, key="fig6_cancel_client")

    st.subheader("Top Tests per Client")
    test_client_group = df.groupby(["Client_Facility", "Test_Name"]).size().reset_index(name="Count")
    top_tests_per_client = test_client_group.sort_values(["Client_Facility", "Count"], ascending=[True, False])
    st.dataframe(top_tests_per_client)
    
    generate_download_link(volume_by_client, "volume_by_client.csv")
    generate_download_link(revenue_by_client, "revenue_by_client.csv")
    generate_download_link(cancel_rate, "cancellation_rate_by_client.csv")
    generate_download_link(top_tests_per_client, "top_tests_per_client.csv")

# === Test Performance === #
with tabs[2]:
    st.header("Test-Level Performance")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tests", df["Test_Name"].nunique())
    top_test = df.groupby("Test_Name")["Price"].sum().idxmax()
    col2.metric("Top Revenue-Generating Test", top_test)
    most_delayed_test = df.groupby("Test_Name")["Delay_Days"].mean().sort_values(ascending=False).idxmax()
    col3.metric("Most Delayed Test (Avg)", most_delayed_test)

 # Volume by Test
    st.subheader("Volume by Test")
    test_volume = df["Test_Name"].value_counts().reset_index(name="Volume").rename(columns={"index": "Test_Name"})
    fig7 = px.bar(test_volume.head(10), x="Test_Name", y="Volume", text="Volume")
    st.plotly_chart(fig7, use_container_width=True, key="fig7_volume")

    # Revenue by Test
    st.subheader("Revenue by Test")
    test_revenue = df.groupby("Test_Name")["Price"].sum().nlargest(10).reset_index()
    fig8 = px.bar(test_revenue, x="Test_Name", y="Price", text="Price", labels={"Price": "Revenue"})
    st.plotly_chart(fig8, use_container_width=True, key="fig8_revenue")

    # Delay by Test
    st.subheader("Average Delay by Test (Completed Only)")
    completed_only = df[df["Status"] == "Completed"]
    delay_by_test = completed_only.groupby("Test_Name")["Delay_Days"].mean().dropna().reset_index()
    fig9 = px.bar(delay_by_test.sort_values("Delay_Days", ascending=False).head(10),
                  x="Test_Name", y="Delay_Days", text="Delay_Days", labels={"Delay_Days": "Avg Delay (Days)"})
    st.plotly_chart(fig9, use_container_width=True, key="fig9_delay")

    # Analyst-Test Breakdown
    st.subheader("Analyst Breakdown by Test")
    analyst_test = df.groupby(["Test_Name", "Analyst"]).agg({
        "Price": "sum",
        "Delay_Days": "mean",
        "Test_Name": "count"
    }).rename(columns={"Price": "Total_Revenue", "Delay_Days": "Avg_Delay", "Test_Name": "Volume"}).reset_index()
    st.dataframe(analyst_test.sort_values("Total_Revenue", ascending=False).round(2))

    generate_download_link(test_volume, "test_volume.csv")
    generate_download_link(test_revenue, "test_revenue.csv")
    generate_download_link(delay_by_test, "average_delay_by_test.csv")
    generate_download_link(analyst_test, "analyst_breakdown_by_test.csv")

# === Delays & TAT === #
with tabs[3]:
    st.header(" Delays and Turnaround Time")

    completed_df = df[df["Status"] == "Completed"].copy()
    completed_df = completed_df[completed_df["Delay_Days"].notna()]
    completed_df = completed_df[(completed_df["Delay_Days"] > -30) & (completed_df["Delay_Days"] < 60)]

    st.subheader("Distribution of Completion Delays")
    fig10 = px.histogram(completed_df, x="Delay_Days", nbins=40, title="Delay Distribution (Days)")
    fig10.update_traces(marker_line_width=0.5, marker_line_color="black")
    st.plotly_chart(fig10, use_container_width=True, key="fig10_delay_dist")

    st.subheader("On-Time vs Delayed Completion")
    completed_df["Status_Eval"] = np.where(completed_df["Delay_Days"] > 0, "Delayed", "On-Time")
    status_count = completed_df["Status_Eval"].value_counts().reset_index(name="count").rename(columns={"index": "Status_Eval"})
    fig11 = px.pie(status_count, names="Status_Eval", values="count", hole=0.4,
                   labels={"Status_Eval": "Completion Status", "count": "Count"})
    st.plotly_chart(fig11, use_container_width=True, key="fig11_ontime_pie")

    st.subheader("Rush vs Non-Rush Turnaround Comparison")
    rush_df = completed_df[completed_df["Rush"].isin(["Y", "N"])]
    rush_delay = rush_df.groupby("Rush")["Delay_Days"].agg(["mean", "count"]).reset_index()
    fig13 = px.bar(rush_delay, x="Rush", y="mean", text="mean", labels={"mean": "Avg Delay (Days)"})
    st.plotly_chart(fig13, use_container_width=True, key="fig13_rush_bar")

    st.subheader("Delay Data Table")
    st.dataframe(completed_df[["Submission_Date", "Test_Name", "Client_Facility", "Analyst", "Delay_Days", "Rush"]].head(100))

    generate_download_link(completed_df, "completed_tests_delay_data.csv")
    generate_download_link(status_count, "ontime_vs_delayed.csv")
    generate_download_link(rush_delay, "rush_vs_nonrush_delay.csv")

# === Forecasting Tab === #
with tabs[4]:
    st.header("🔍 Test Forecasting & Diagnostics")

    confidence_level = st.slider("Forecast Confidence Level", min_value=50, max_value=99, value=95)

    if test_filter:
        selected_tests = test_filter
        base = df[(df["Test_Name"].isin(selected_tests)) & (df["Status"] == "Completed")].copy()

        for test in selected_tests:
           
            test_df = base[base["Test_Name"] == test]
            test_df = test_df.groupby("Submission_Date").agg({"Price": "sum", "Test_Name": "count"}).reset_index()
            test_df = test_df.rename(columns={"Submission_Date": "ds", "Test_Name": "y", "Price": "Actual_Revenue"})

            if test_df.shape[0] < 2 or test_df["ds"].nunique() < 2:
                st.warning(f"Not enough data for forecasting '{test}'")
                continue

            avg_price = test_df["Actual_Revenue"].sum() / test_df["y"].sum()

            m = Prophet(interval_width=confidence_level / 100)
            m.fit(test_df[["ds", "y"]])

            future = m.make_future_dataframe(periods=60)
            forecast = m.predict(future)

            forecast = forecast.merge(test_df[["ds", "y", "Actual_Revenue"]], on="ds", how="left")
            forecast["Predicted Volume"] = forecast["yhat"]
            forecast["Predicted Revenue"] = forecast["yhat"] * avg_price

            eval_df = forecast.dropna(subset=["y"])
            y_true = eval_df["y"]
            y_pred = eval_df["yhat"]
            r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            mase = mae / np.mean(np.abs(y_true - y_true.shift(1)).dropna())
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) if len(y_true) > 0 else 0

            try:
                beta = round(m.params['k'].mean(), 4)
                alpha = round(m.params['m'].mean(), 4)
                gamma = round(np.std(m.params['delta']), 4) if "delta" in m.params else 0.0
            except Exception:
                alpha, beta, gamma = np.nan, np.nan, np.nan

            metrics = pd.DataFrame({
                "Statistic": ["R²", "MAE", "RMSE", "MASE", "SMAPE", "Alpha (level)", "Beta (trend)", "Gamma (seasonality)"],
                "Value": [r2, mae, rmse, mase, smape, alpha, beta, gamma]
            }).round(3)

            # Revenue Forecast Evaluation
            eval_df["Pred_Revenue"] = eval_df["yhat"] * avg_price
            rev_true = eval_df["Actual_Revenue"]
            rev_pred = eval_df["Pred_Revenue"]
            r2_rev = np.corrcoef(rev_true, rev_pred)[0, 1]**2
            mae_rev = np.mean(np.abs(rev_true - rev_pred))
            rmse_rev = np.sqrt(np.mean((rev_true - rev_pred)**2))
            mase_rev = mae_rev / np.mean(np.abs(rev_true - rev_true.shift(1)).dropna())
            smape_rev = np.mean(2 * np.abs(rev_true - rev_pred) / (np.abs(rev_true) + np.abs(rev_pred)))

            rev_metrics = pd.DataFrame({
                "Statistic": ["R²", "MAE", "RMSE", "MASE", "SMAPE"],
                "Value": [r2_rev, mae_rev, rmse_rev, mase_rev, smape_rev]
            }).round(3)

            st.subheader(f"Forecast & Stats for: {test}")
            st.dataframe(metrics)
            st.subheader(f"Revenue Forecast Stats - {test}")
            st.dataframe(rev_metrics)

            fig1 = px.line(forecast, x="ds", y="yhat", title=f"{test} - 60 Day Volume Forecast",
                           labels={"ds": "Date", "yhat": "Predicted Volume"})
            st.plotly_chart(fig1, use_container_width=True, key=f"{test}_vol_forecast")

            fig2 = px.line(forecast, x="ds", y=["yhat", "y"], title=f"{test} - Actual vs Predicted Volume",
                           labels={"ds": "Date", "value": "Volume", "variable": "Legend"})
            st.plotly_chart(fig2, use_container_width=True, key=f"{test}_vol_overlay")

            fig3 = px.line(forecast, x="ds", y="Predicted Revenue", title=f"{test} - 60 Day Revenue Forecast",
                           labels={"ds": "Date", "Predicted Revenue": "Predicted Revenue"})
            st.plotly_chart(fig3, use_container_width=True, key=f"{test}_rev_forecast")

            fig4 = px.line(forecast, x="ds", y=["Predicted Revenue", "Actual_Revenue"], title=f"{test} - Actual vs Predicted Revenue",
                           labels={"ds": "Date", "value": "Revenue", "variable": "Legend"})
            st.plotly_chart(fig4, use_container_width=True, key=f"{test}_rev_overlay")

            forecast["Month"] = forecast["ds"].dt.to_period("M")
            forecast["Quarter"] = forecast["ds"].dt.to_period("Q")

            summary_rows = []

            # Monthly
            month_summary = forecast.groupby("Month").agg({
                "Predicted Volume": "sum",
                "y": "sum",
                "Predicted Revenue": "sum",
                "Actual_Revenue": "sum"
            }).reset_index().rename(columns={"Month": "Group"})
            month_summary.insert(0, "Group_Type", "Month")
            summary_rows.append(month_summary)

            # Quarterly
            quarter_summary = forecast.groupby("Quarter").agg({
                "Predicted Volume": "sum",
                "y": "sum",
                "Predicted Revenue": "sum",
                "Actual_Revenue": "sum"
            }).reset_index().rename(columns={"Quarter": "Group"})
            quarter_summary.insert(0, "Group_Type", "Quarter")
            summary_rows.append(quarter_summary)

            # Day Avg for forecast period
            forecast_range = forecast[forecast["ds"] > test_df["ds"].max()]
            if not forecast_range.empty:
                row = pd.DataFrame([{
                    "Group_Type": "Day (avg)",
                    "Group": f"{forecast_range['ds'].dt.to_period('M').min()} to {forecast_range['ds'].dt.to_period('M').max()}",
                    "Predicted Volume": round(forecast_range["Predicted Volume"].mean(), 2),
                    "y": np.nan,
                    "Predicted Revenue": round(forecast_range["Predicted Revenue"].mean(), 2),
                    "Actual_Revenue": np.nan
                }])
                summary_rows.insert(0, row)

            final_summary = pd.concat(summary_rows, ignore_index=True)
            final_summary = final_summary.rename(columns={"y": "Actual Volume"})

            st.subheader(f"Combined Forecast Summary - {test}")
            st.dataframe(final_summary)

            generate_download_link(metrics, f"{test}_forecast_metrics.csv")
            generate_download_link(final_summary, f"{test}_forecast_summary.csv")
            generate_download_link(forecast, f"{test}_full_forecast_data.csv")

# === Statistical Insights === #
with tabs[5]:
    st.header("📐 Statistical Insights by Test")

    # 1. Revenue Summary by Test
    st.subheader("📊 Revenue Summary by Test")
    test_summary = df.groupby("Test_Name")["Price"].agg(
        Test_Count="count",
        Total_Revenue="sum",
        Avg_Revenue="mean",
        Std_Dev="std"
    ).reset_index().round(2)
    st.dataframe(test_summary.sort_values("Total_Revenue", ascending=False))
    generate_download_link(test_summary, "test_summary.csv")

    # 2. Revenue Summary by Client
    st.subheader("🏢 Revenue Summary by Client")
    client_summary = df.groupby("Client_Facility")["Price"].agg(
        Test_Count="count",
        Total_Revenue="sum",
        Avg_Revenue="mean",
        Std_Dev="std"
    ).reset_index().round(2)
    st.dataframe(client_summary.sort_values("Total_Revenue", ascending=False))
    generate_download_link(client_summary, "client_summary.csv")

    # 3. Revenue Summary by Test & Client
    st.subheader("🧩 Revenue Summary by Test & Client")
    combo_summary = df.groupby(["Test_Name", "Client_Facility"])["Price"].agg(
        Test_Count="count",
        Total_Revenue="sum",
        Avg_Revenue="mean"
    ).reset_index().round(2)
    st.dataframe(combo_summary.sort_values("Total_Revenue", ascending=False))
    generate_download_link(combo_summary, "test_client_summary.csv")

    # 4. Revenue Distribution
    st.subheader("📈 Revenue Distribution Histogram")
    fig = px.histogram(df, x="Price", nbins=40, title="Revenue Distribution Across All Tests")
    st.plotly_chart(fig, use_container_width=True)







