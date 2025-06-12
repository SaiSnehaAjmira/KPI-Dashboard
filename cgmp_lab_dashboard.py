import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from prophet import Prophet
from sklearn.linear_model import LinearRegression

# ========== Delay Helper ==========
def adjusted_calendar_delay(submit_date, est_date, skip_sundays=True):
    """Calculate adjusted delay, skipping Sundays if enabled."""
    if pd.isna(submit_date) or pd.isna(est_date):
        return pd.NaT
    if skip_sundays:
        days = pd.date_range(start=submit_date + timedelta(days=1), end=est_date)
        skipped = sum(1 for d in days if d.weekday() == 6)
        return est_date - timedelta(days=skipped)
    else:
        return est_date

# ========== CSV Download Button ==========
def generate_download_link(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ========== Dashboard Header ==========
st.set_page_config(page_title="Lab Dashboard", layout="wide")
st.title("Houston KPI Dashboard")

# ========== File Upload ==========
uploaded_files = st.file_uploader(
    "Upload Lab File(s)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True, 
    key="lab_upload"
)

if not uploaded_files:
    st.info("Please upload one or more lab files to begin.")
    st.stop()

# ========== Load + Merge Files (CACHED) ==========
@st.cache_data(show_spinner=True)
def load_and_merge(files):
    df_list = []
    for file in files:
        if file.name.endswith("xlsx"):
            temp_df = pd.read_excel(file, engine="openpyxl")
        else:
            temp_df = pd.read_csv(file)
        for col in ["Submission_Release_Date", "Completion_Date", "Estimated_Completion_Date"]:
            if col in temp_df.columns:
                temp_df[col] = pd.to_datetime(temp_df[col], errors="coerce")
        if "Price" in temp_df.columns:
            temp_df["Price"] = pd.to_numeric(temp_df["Price"], errors="coerce")
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    return df

df = load_and_merge(uploaded_files)
df_raw = df.copy()

# ========== Standardize Analyst Names (lowercase, strip spaces) ==========
for d in [df, df_raw]:
    if "Analyst" in d.columns:
        d["Analyst"] = d["Analyst"].astype(str).str.strip().str.lower()

# ========== Separate Canceled Data BEFORE Filtering ==========
canceled_df = df[df["Status"] == "Canceled"].copy()
df = df[df["Status"] != "Canceled"].copy()

# ========== Add Date Parts ==========
df["Week"] = df["Submission_Release_Date"].dt.isocalendar().week
df["Month"] = df["Submission_Release_Date"].dt.to_period("M").astype(str)
df["Quarter"] = df["Submission_Release_Date"].dt.to_period("Q").astype(str)
df["Year"] = df["Submission_Release_Date"].dt.year

# ========== Sidebar Filters ==========
st.sidebar.header("Filter Panel")

def safe_multiselect(label, col):
    return st.sidebar.multiselect(label, sorted(df[col].dropna().unique()))

test_filter = safe_multiselect("Test Name", "Test_Name")
client_filter = safe_multiselect("Client Facility", "Client_Facility")
analyst_filter = safe_multiselect("Analyst", "Analyst")  # Now only shows unique, lowercase names
facility_type_filter = safe_multiselect("Facility Type", "Facility_Type")
rush_filter = safe_multiselect("Rush", "Rush")
control_filter = safe_multiselect("Controlled Substance Class", "Controlled_Substance_Class")
hazardous_filter = safe_multiselect("Hazardous Drug", "Hazardous_Drug")
status_filter = safe_multiselect("Test Status", "Status")
sample_search = st.sidebar.text_input("Search Sample Name")

# ... (rest of your filter, date, and dashboard logic remains unchanged)


# ========== Delay Toggle ==========
delay_toggle = st.sidebar.radio(
    "Delay Calculation Method",
    ["Adjusted (skip Sundays only)", "System ETC"],
    horizontal=True,
    key="delay_method_toggle_main"
)
skip_sundays = delay_toggle == "Adjusted (skip Sundays only)"

# ========== Date Range Filter ==========
min_date = df["Submission_Release_Date"].min()
max_date = df["Submission_Release_Date"].max()
st.sidebar.markdown("### Release Date Filter")
start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# ========== Reset Filters Button ==========
if st.sidebar.button("Reset Filters"):
    st.rerun()

# ========== Apply Filters ==========
df = df[
    (df["Submission_Release_Date"] >= pd.to_datetime(start_date)) &
    (df["Submission_Release_Date"] <= pd.to_datetime(end_date))
]

filter_conditions = [
    ("Test_Name", test_filter),
    ("Client_Facility", client_filter),
    ("Analyst", analyst_filter),
    ("Facility_Type", facility_type_filter),
    ("Rush", rush_filter),
    ("Controlled_Substance_Class", control_filter),
    ("Hazardous_Drug", hazardous_filter),
    ("Status", status_filter)
]
for col, selected in filter_conditions:
    if selected:
        df = df[df[col].isin(selected)]

if sample_search:
    df = df[df["Sample_Name"].astype(str).str.contains(sample_search, case=False, na=False)]

# ========== Filter canceled data based on current filters ==========
filtered_canceled_df = df_raw[df_raw["Status"] == "Canceled"]
# Apply the same filters to canceled data
filtered_canceled_df = filtered_canceled_df[
    (filtered_canceled_df["Submission_Release_Date"] >= pd.to_datetime(start_date)) &
    (filtered_canceled_df["Submission_Release_Date"] <= pd.to_datetime(end_date))
]
for col, selected in filter_conditions:
    if selected:
        filtered_canceled_df = filtered_canceled_df[filtered_canceled_df[col].isin(selected)]
if sample_search:
    filtered_canceled_df = filtered_canceled_df[filtered_canceled_df["Sample_Name"].astype(str).str.contains(sample_search, case=False, na=False)]

# ========== Compute Delay (Consistent Everywhere) ==========
completed_mask = df["Status"] == "Completed"
df.loc[completed_mask, "Adj_Est_Completion"] = df.loc[completed_mask].apply(
    lambda row: adjusted_calendar_delay(
        row["Submission_Release_Date"], row["Estimated_Completion_Date"], skip_sundays=skip_sundays
    ), axis=1
)
df.loc[completed_mask, "Delay_Days"] = (
    df.loc[completed_mask, "Completion_Date"] - df.loc[completed_mask, "Adj_Est_Completion"]
).dt.days

# ========== Tabs ==========
tabs = st.tabs([
    "Executive Overview", 
    "Client Analysis", 
    "Test Performance", 
    "Delays & TAT", 
    "Forecasting", 
    "Statistical Insights"
])


#Executive Tab

with tabs[0]:
    st.markdown(
        """
        <div style="background-color:#f0f2f6;padding:1.2rem;border-radius:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.2rem; color:#4c78a8;">
        <b>ℹ️ Executive Overview:</b> All metrics and charts below are based on <b>all available data</b> and are <u>not affected by sidebar filters</u>.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## :bar_chart: Executive Overview")

    # --- KPI Metrics --- #
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🧪 Total Active Tests", f"{df_raw.shape[0]:,}")
    kpi2.metric("💰 Total Revenue", f"${df_raw['Price'].sum():,.2f}")
    kpi3.metric("📊 Avg Price/Test", f"${df_raw['Price'].mean():.2f}")
    lost_revenue = canceled_df["Price"].sum()
    kpi4.metric("❌ Lost Revenue (Canceled)", f"${lost_revenue:,.2f}", delta_color="inverse")

    st.markdown("---")

    # --- Top 5 Tests by Volume --- #
    st.markdown("#### 🏆 Top 5 Tests by Volume")
    top_tests = (
        df_raw[df_raw["Status"] != "Canceled"]["Test_Name"]
        .value_counts()
        .nlargest(5)
        .reset_index(name="Count")
        .rename(columns={"index": "Test_Name"})
    )
    fig1 = px.bar(
        top_tests, x="Test_Name", y="Count", text="Count",
        color="Test_Name", color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Top 5 Tests by Volume"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(yaxis_title="Test Count", xaxis_title="Test Name", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True, key="exec_top_tests_volume")
    generate_download_link(top_tests, "top_tests_volume.csv")

    # --- Top 5 Tests by Revenue --- #
    st.markdown("#### 💵 Top 5 Tests by Revenue")
    top_revenue = (
        df_raw[df_raw["Status"] != "Canceled"]
        .groupby("Test_Name")["Price"]
        .sum()
        .nlargest(5)
        .reset_index()
    )
    fig2 = px.bar(
        top_revenue, x="Test_Name", y="Price", text="Price",
        color="Test_Name", color_discrete_sequence=px.colors.qualitative.Set2,
        title="Top 5 Tests by Revenue"
    )
    fig2.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
    fig2.update_layout(yaxis_title="Total Revenue", xaxis_title="Test Name", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True, key="exec_top_tests_revenue")
    generate_download_link(top_revenue, "top_tests_revenue.csv")

    # --- Revenue by Facility Type --- #
    st.markdown("#### 🏭 Revenue by Facility Type")
    rev_by_facility = (
        df_raw[df_raw["Status"] != "Canceled"]
        .groupby("Facility_Type")["Price"].sum().reset_index()
    )
    fig3 = px.pie(
        rev_by_facility, names="Facility_Type", values="Price", hole=0.45,
        color_discrete_sequence=px.colors.sequential.RdBu,
        title="Revenue Contribution by Facility Type"
    )
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True, key="exec_revenue_by_facility")
    generate_download_link(rev_by_facility, "revenue_by_facility_type.csv")

    # --- Top 5 Cancelled Tests --- #
    st.markdown("#### ❌ Top 5 Cancelled Tests")
    top_cancelled = (
        canceled_df["Test_Name"]
        .value_counts()
        .nlargest(5)
        .reset_index(name="Canceled_Count")
        .rename(columns={"index": "Test_Name"})
    )
    fig4 = px.bar(
        top_cancelled, x="Test_Name", y="Canceled_Count", text="Canceled_Count",
        color="Test_Name", color_discrete_sequence=px.colors.qualitative.Safe,
        title="Top 5 Cancelled Tests"
    )
    fig4.update_traces(marker_color="#e45756", textposition="outside")
    fig4.update_layout(yaxis_title="Canceled Count", xaxis_title="Test Name", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True, key="exec_top_cancelled_tests")
    generate_download_link(top_cancelled, "top_cancelled_tests.csv")

    st.markdown(
        """
        <style>
        .stMetric {background-color: #f8fafc; border-radius: 10px; padding: 10px;}
        .stPlotlyChart {background-color: #fff; border-radius: 10px; padding: 10px;}
        </style>
        """,
        unsafe_allow_html=True
    )


# Client Analysis

with tabs[1]:
    st.markdown(
        """
        <div style="background-color:#f0f2f6;padding:1.2rem;border-radius:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.2rem; color:#4c78a8;">
        <b>ℹ️ Client Analysis:</b> All metrics and charts below reflect <b>current filters</b> and date range.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## :busts_in_silhouette: Client-Level Performance Analysis")

    # --- KPIs --- #
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("👥 Total Clients", df["Client_Facility"].nunique())
    if not df.empty:
        top_client_rev = df.groupby("Client_Facility")["Price"].sum().idxmax()
        top_client_vol = df["Client_Facility"].value_counts().idxmax()
    else:
        top_client_rev = "N/A"
        top_client_vol = "N/A"
    kpi2.metric("💰 Top Client by Revenue", top_client_rev)
    kpi3.metric("🧪 Top Client by Volume", top_client_vol)
    if not canceled_df.empty:
        top_client_cancel = canceled_df["Client_Facility"].value_counts().idxmax()
    else:
        top_client_cancel = "N/A"
    kpi4.metric("❌ Top Client by Cancelling", top_client_cancel)

    st.markdown("---")

    # --- Top 10 Clients by Revenue --- #
    st.markdown("#### 💵 Top 10 Clients by Revenue")
    top10_rev = (
        df.groupby("Client_Facility")["Price"].sum()
        .nlargest(10)
        .reset_index()
        .rename(columns={"Price": "Total Revenue"})
    )
    fig1 = px.bar(
        top10_rev, x="Client_Facility", y="Total Revenue", text="Total Revenue",
        color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Set2,
        title="Top 10 Clients by Revenue"
    )
    fig1.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
    fig1.update_layout(yaxis_title="Total Revenue", xaxis_title="Client", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True, key="client_top10_revenue")
    generate_download_link(top10_rev, "top10_clients_by_revenue.csv")

    # --- Top 10 Clients by Volume --- #
    st.markdown("#### 🧪 Top 10 Clients by Volume")
    top10_vol = (
        df["Client_Facility"].value_counts()
        .head(10)
        .reset_index(name="Test Volume")
    )
    fig2 = px.bar(
        top10_vol, x="Client_Facility", y="Test Volume", text="Test Volume",
        color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Top 10 Clients by Test Volume"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis_title="Test Volume", xaxis_title="Client", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True, key="client_top10_volume")
    generate_download_link(top10_vol, "top10_clients_by_volume.csv")

    # --- Top 10 Clients by Cancellations --- #
    st.markdown("#### ❌ Top 10 Clients by Cancellations")
    top10_cancel = (
        canceled_df["Client_Facility"].value_counts()
        .head(10)
        .reset_index(name="Canceled Count")
    )
    fig3 = px.bar(
        top10_cancel, x="Client_Facility", y="Canceled Count", text="Canceled Count",
        color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Safe,
        title="Top 10 Clients by Canceled Tests"
    )
    fig3.update_traces(marker_color="#e45756", textposition="outside")
    fig3.update_layout(yaxis_title="Canceled Count", xaxis_title="Client", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True, key="client_top10_cancellations")
    generate_download_link(top10_cancel, "top10_clients_by_cancellations.csv")

    st.markdown("---")

    # --- Growth Analysis Toggle --- #
    st.markdown("#### 📈 Top 10 Growing Clients (by Volume & Revenue)")
    growth_window = st.radio(
        "Select Growth Window:",
        options=[3, 6, 12],
        format_func=lambda x: f"Last {x} Months vs Previous {x} Months",
        horizontal=True,
        key="growth_window_toggle"
    )

    if df["Submission_Release_Date"].notna().any():
        most_recent_date = df["Submission_Release_Date"].max()
        cutoff_recent = most_recent_date - pd.DateOffset(months=growth_window)
        cutoff_prev = cutoff_recent - pd.DateOffset(months=growth_window)

        recent_period = df[(df["Submission_Release_Date"] > cutoff_recent)]
        prev_period = df[
            (df["Submission_Release_Date"] > cutoff_prev) &
            (df["Submission_Release_Date"] <= cutoff_recent)
        ]

        # --- By Volume ---
        recent_vol = recent_period.groupby("Client_Facility").size().rename("Recent Volume")
        prev_vol = prev_period.groupby("Client_Facility").size().rename("Prev Volume")
        growth_vol = pd.concat([recent_vol, prev_vol], axis=1).fillna(0)
        growth_vol["Abs Growth"] = growth_vol["Recent Volume"] - growth_vol["Prev Volume"]
        growth_vol["% Growth"] = np.where(
            growth_vol["Prev Volume"] == 0,
            np.nan,
            100 * growth_vol["Abs Growth"] / growth_vol["Prev Volume"]
        )
        growth_vol = growth_vol.reset_index()
        growth_vol = growth_vol.sort_values("Abs Growth", ascending=False).head(10)

        # --- By Revenue ---
        recent_rev = recent_period.groupby("Client_Facility")["Price"].sum().rename("Recent Revenue")
        prev_rev = prev_period.groupby("Client_Facility")["Price"].sum().rename("Prev Revenue")
        growth_rev = pd.concat([recent_rev, prev_rev], axis=1).fillna(0)
        growth_rev["Abs Growth"] = growth_rev["Recent Revenue"] - growth_rev["Prev Revenue"]
        growth_rev["% Growth"] = np.where(
            growth_rev["Prev Revenue"] == 0,
            np.nan,
            100 * growth_rev["Abs Growth"] / growth_rev["Prev Revenue"]
        )
        growth_rev = growth_rev.reset_index()
        growth_rev = growth_rev.sort_values("Abs Growth", ascending=False).head(10)

        # --- Growth Volume Chart ---
        st.markdown("##### 🚀 Top 10 Clients by Volume Growth")
        fig_grow_vol = px.bar(
            growth_vol, x="Client_Facility", y="Abs Growth", text="Abs Growth",
            color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={"% Growth": ":.2f"},
            title="Top 10 Clients by Absolute Volume Growth"
        )
        fig_grow_vol.update_traces(textposition="outside")
        fig_grow_vol.update_layout(yaxis_title="Absolute Growth", xaxis_title="Client", showlegend=False)
        st.plotly_chart(fig_grow_vol, use_container_width=True, key="client_top10_grow_volume")
        st.dataframe(
            growth_vol[["Client_Facility", "Recent Volume", "Prev Volume", "Abs Growth", "% Growth"]]
            .rename(columns={"Abs Growth": "Absolute Growth", "% Growth": "Percent Growth (%)"})
            .replace({np.nan: "N/A"})
            .round(2),
            use_container_width=True
        )
        generate_download_link(
            growth_vol[["Client_Facility", "Recent Volume", "Prev Volume", "Abs Growth", "% Growth"]],
            "top10_clients_by_volume_growth.csv"
        )

        # --- Growth Revenue Chart ---
        st.markdown("##### 💹 Top 10 Clients by Revenue Growth")
        fig_grow_rev = px.bar(
            growth_rev, x="Client_Facility", y="Abs Growth", text="Abs Growth",
            color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Prism,
            hover_data={"% Growth": ":.2f"},
            title="Top 10 Clients by Absolute Revenue Growth"
        )
        fig_grow_rev.update_traces(textposition="outside")
        fig_grow_rev.update_layout(yaxis_title="Absolute Growth ($)", xaxis_title="Client", showlegend=False)
        st.plotly_chart(fig_grow_rev, use_container_width=True, key="client_top10_grow_revenue")
        st.dataframe(
            growth_rev[["Client_Facility", "Recent Revenue", "Prev Revenue", "Abs Growth", "% Growth"]]
            .rename(columns={"Abs Growth": "Absolute Growth", "% Growth": "Percent Growth (%)"})
            .replace({np.nan: "N/A"})
            .round(2),
            use_container_width=True
        )
        generate_download_link(
            growth_rev[["Client_Facility", "Recent Revenue", "Prev Revenue", "Abs Growth", "% Growth"]],
            "top10_clients_by_revenue_growth.csv"
        )
    else:
        st.info("Not enough date data to calculate growth trends.")

    st.markdown(
        """
        <style>
        .stMetric {background-color: #f8fafc; border-radius: 10px; padding: 10px;}
        .stPlotlyChart {background-color: #fff; border-radius: 10px; padding: 10px;}
        </style>
        """,
        unsafe_allow_html=True
    )

# === Test Performance === #
with tabs[2]:
    st.markdown(
        """
        <div style="background-color:#f0f2f6;padding:1.2rem;border-radius:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.2rem; color:#4c78a8;">
        <b>ℹ️ Test Performance:</b> All metrics and charts below reflect <b>current filters</b> and date range.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## :microscope: Test Performance Dashboard")

    # --- KPIs --- #
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔬 Unique Tests", df["Test_Name"].nunique())
    k2.metric("🧪 Total Test Volume", df.shape[0])
    k3.metric("💰 Total Revenue by Test", f"${df['Price'].sum():,.2f}")
    lost_rev_by_test = canceled_df.groupby("Test_Name")["Price"].sum() if not canceled_df.empty else pd.Series(dtype=float)
    k4.metric("❌ Lost Revenue by Test (Total)", f"${lost_rev_by_test.sum():,.2f}")

    k5, k6, k7, k8 = st.columns(4)
    avg_rev = df.groupby("Test_Name")["Price"].mean().mean() if not df.empty else 0
    k5.metric("📊 Avg Revenue/Test", f"${avg_rev:,.2f}")

    most_performed = df["Test_Name"].value_counts().idxmax() if not df.empty else "N/A"
    k6.metric("🏆 Most Performed Test", most_performed)

    most_profitable = df.groupby("Test_Name")["Price"].sum().idxmax() if not df.empty else "N/A"
    k7.metric("💸 Most Profitable Test", most_profitable)

    if not lost_rev_by_test.empty:
        test_most_lost = lost_rev_by_test.idxmax()
    else:
        test_most_lost = "N/A"
    k8.metric("❗Test with Most Lost Revenue", test_most_lost)

    # --- Top 10 Tests by Volume --- #
    st.markdown("#### 🧪 Top 10 Tests by Volume")
    top10_vol = (
        df["Test_Name"].value_counts()
        .head(10)
        .reset_index(name="Volume")
        .rename(columns={"index": "Test_Name"})
    )
    fig_vol = px.bar(
        top10_vol, x="Test_Name", y="Volume", text="Volume",
        color="Test_Name", color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Top 10 Tests by Volume"
    )
    fig_vol.update_traces(textposition="outside")
    fig_vol.update_layout(yaxis_title="Test Volume", xaxis_title="Test Name", showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True, key="test_top10_volume")
    generate_download_link(top10_vol, "top10_tests_by_volume.csv")

    # --- Top 10 Tests by Revenue --- #
    st.markdown("#### 💵 Top 10 Tests by Revenue")
    top10_rev = (
        df.groupby("Test_Name")["Price"].sum().nlargest(10).reset_index()
        .rename(columns={"Price": "Total Revenue"})
    )
    fig_rev = px.bar(
        top10_rev, x="Test_Name", y="Total Revenue", text="Total Revenue",
        color="Test_Name", color_discrete_sequence=px.colors.qualitative.Set2,
        title="Top 10 Tests by Revenue"
    )
    fig_rev.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
    fig_rev.update_layout(yaxis_title="Total Revenue", xaxis_title="Test Name", showlegend=False)
    st.plotly_chart(fig_rev, use_container_width=True, key="test_top10_revenue")
    generate_download_link(top10_rev, "top10_tests_by_revenue.csv")

    # --- Top 10 Tests by Lost Revenue --- #
    st.markdown("#### ❌ Top 10 Tests by Lost Revenue (Canceled)")
    if not canceled_df.empty:
        lost_rev_by_test = (
            canceled_df.groupby("Test_Name")["Price"].sum().nlargest(10).reset_index()
            .rename(columns={"Price": "Lost Revenue"})
        )
        fig_lost = px.bar(
            lost_rev_by_test, x="Test_Name", y="Lost Revenue", text="Lost Revenue",
            color="Test_Name", color_discrete_sequence=px.colors.qualitative.Safe,
            title="Top 10 Tests by Lost Revenue"
        )
        fig_lost.update_traces(marker_color="#e45756", textposition="outside")
        fig_lost.update_layout(yaxis_title="Lost Revenue", xaxis_title="Test Name", showlegend=False)
        st.plotly_chart(fig_lost, use_container_width=True, key="test_top10_lost_revenue")
        generate_download_link(lost_rev_by_test, "top10_tests_by_lost_revenue.csv")
    else:
        st.info("No canceled tests in current filters.")

    # --- Rush vs Non-Rush Volume Comparison --- #
    st.markdown("#### ⚡ Rush vs Non-Rush Test Volume")
    if "Rush" in df.columns:
        rush_counts = (
            df.groupby(["Test_Name", "Rush"]).size().unstack(fill_value=0)
            .reset_index().rename_axis(None, axis=1)
        )
        # Ensure both columns 'Y' and 'N' exist for plotting
        for col in ['Y', 'N']:
            if col not in rush_counts.columns:
                rush_counts[col] = 0
        rush_counts = rush_counts.sort_values(by=["Y"], ascending=False).head(10)
        fig_rush = px.bar(
            rush_counts, x="Test_Name", y=["Y", "N"],
            title="Top 10 Tests: Rush vs Non-Rush Volume",
            labels={"value": "Test Volume", "variable": "Rush Status"},
            barmode="group",
            color_discrete_sequence=["#f28e2b", "#4c78a8"]
        )
        st.plotly_chart(fig_rush, use_container_width=True, key="test_rush_nonrush")
        generate_download_link(rush_counts, "rush_vs_nonrush_by_test.csv")
    else:
        st.info("No Rush data available.")

    # --- Fastest Growing Tests by Volume --- #
    st.markdown("#### 🚀 Fastest Growing Tests by Volume")
    growth_window = st.radio(
        "Select Growth Window:",
        options=[3, 6, 12],
        format_func=lambda x: f"Last {x} Months vs Previous {x} Months",
        horizontal=True,
        key="test_growth_window"
    )
    if df["Submission_Release_Date"].notna().any():
        most_recent_date = df["Submission_Release_Date"].max()
        cutoff_recent = most_recent_date - pd.DateOffset(months=growth_window)
        cutoff_prev = cutoff_recent - pd.DateOffset(months=growth_window)

        recent_period = df[df["Submission_Release_Date"] > cutoff_recent]
        prev_period = df[(df["Submission_Release_Date"] > cutoff_prev) & (df["Submission_Release_Date"] <= cutoff_recent)]
        recent_vol = recent_period.groupby("Test_Name").size().rename("Recent Volume")
        prev_vol = prev_period.groupby("Test_Name").size().rename("Prev Volume")
        growth = pd.concat([recent_vol, prev_vol], axis=1).fillna(0)
        growth["Abs Growth"] = growth["Recent Volume"] - growth["Prev Volume"]
        growth["% Growth"] = np.where(
            growth["Prev Volume"] == 0,
            np.nan,
            100 * growth["Abs Growth"] / growth["Prev Volume"]
        )
        growth = growth.reset_index()
        growth = growth.sort_values("Abs Growth", ascending=False).head(10)

        fig_growth = px.bar(
            growth, x="Test_Name", y="Abs Growth", text="Abs Growth",
            color="Test_Name", color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={"% Growth": ":.2f"},
            title="Top 10 Fastest Growing Tests by Volume"
        )
        fig_growth.update_traces(textposition="outside")
        fig_growth.update_layout(yaxis_title="Absolute Growth", xaxis_title="Test Name", showlegend=False)
        st.plotly_chart(fig_growth, use_container_width=True, key="test_top10_growing")
        st.dataframe(
            growth[["Test_Name", "Recent Volume", "Prev Volume", "Abs Growth", "% Growth"]]
            .rename(columns={"Abs Growth": "Absolute Growth", "% Growth": "Percent Growth (%)"})
            .replace({np.nan: "N/A"})
            .round(2),
            use_container_width=True
        )
        generate_download_link(
            growth[["Test_Name", "Recent Volume", "Prev Volume", "Abs Growth", "% Growth"]],
            "top10_growing_tests_by_volume.csv"
        )
    else:
        st.info("Not enough date data to calculate growth trends.")

    # --- Analyst Breakdown --- #
    st.markdown("#### 👩‍🔬 Analyst Breakdown")
    if all(col in df.columns for col in ["Test_Name", "Analyst"]):
        analyst_test = df.groupby("Analyst").agg(
            Test_Count=("Test_Name", "count"),
            Total_Revenue=("Price", "sum")
        ).reset_index().sort_values("Test_Count", ascending=False)
        fig_analyst = px.bar(
            analyst_test, x="Analyst", y="Test_Count", text="Test_Count",
            title="Total Test Volume by Analyst",
            color="Analyst", color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_analyst.update_traces(textposition="outside")
        fig_analyst.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_analyst, use_container_width=True, key="test_analyst_breakdown")
        st.dataframe(analyst_test, use_container_width=True)
        generate_download_link(analyst_test, "analyst_test_performance.csv")

    st.markdown(
        """
        <style>
        .stMetric {background-color: #f8fafc; border-radius: 10px; padding: 10px;}
        .stPlotlyChart {background-color: #fff; border-radius: 10px; padding: 10px;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Month-over-Month Analyst Completed Test Count (by Completion Date) ---
    st.markdown("#### 📈 Month-over-Month Analyst Completed Test Count (by Completion Date)")

    # 1. Filter for completed tests within the selected completion date range
    completed_df = df_raw[
        (df_raw["Status"] == "Completed") &
        (df_raw["Completion_Date"].notna()) &
        (df_raw["Completion_Date"] >= pd.to_datetime(start_date)) &
        (df_raw["Completion_Date"] <= pd.to_datetime(end_date))
    ].copy()
    completed_df["Month"] = completed_df["Completion_Date"].dt.to_period("M").dt.to_timestamp()

    # 2. Further filter by analyst if analyst_filter is used
    if analyst_filter:
        completed_df = completed_df[completed_df["Analyst"].isin(analyst_filter)]

    # 3. Group by month and analyst
    monthly_analyst_completed = (
        completed_df.groupby(['Month', 'Analyst'])
        .size()
        .reset_index(name='Completed_Count')
    )

    # 4. Get all relevant analysts for the legend
    if analyst_filter:
        analysts_to_show = analyst_filter
    else:
        analysts_to_show = monthly_analyst_completed["Analyst"].unique()

    # 5. Ensure all months are present for each analyst (fill missing with 0)
    if not monthly_analyst_completed.empty:
        all_months = pd.date_range(
            start=monthly_analyst_completed['Month'].min(),
            end=monthly_analyst_completed['Month'].max(),
            freq='MS'
        )
        full_index = pd.MultiIndex.from_product([all_months, analysts_to_show], names=['Month', 'Analyst'])
        monthly_analyst_completed = (
            monthly_analyst_completed
            .set_index(['Month', 'Analyst'])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

    # 6. Plot
    fig_line = px.line(
        monthly_analyst_completed,
        x="Month",
        y="Completed_Count",
        color="Analyst",
        markers=True,
        title="Month-over-Month Completed Test Count by Analyst",
        labels={"Completed_Count": "Completed Test Count", "Month": "Month"}
    )
    fig_line.update_xaxes(
        dtick="M1",
        tickformat="%b %Y",
        ticklabelmode="period",
        tickangle=-45,
        type='category'
    )
    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Completed Test Count",
        legend_title_text="Analyst",
        hovermode="x unified"
    )

    st.plotly_chart(fig_line, use_container_width=True, key="mo_monthly_analyst_completed")
    st.dataframe(monthly_analyst_completed, use_container_width=True)
    generate_download_link(monthly_analyst_completed, "monthly_analyst_completed_count.csv")

 


#Delay days and TAT
with tabs[3]:
    st.markdown(
        """
        <div style="background-color:#f0f2f6;padding:1.2rem;border-radius:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.2rem; color:#4c78a8;">
        <b>ℹ️ Delays & TAT:</b> All metrics and charts below reflect <b>current filters</b> and date range.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("## ⏱ Delays and Turnaround Time (TAT)")

    # --- Prepare Data ---
    df_delay = df.copy()
    df_delay["Delay_Days"] = (df_delay["Completion_Date"] - df_delay["Estimated_Completion_Date"]).dt.days
    df_delay["TAT_Days"] = (df_delay["Completion_Date"] - df_delay["Submission_Release_Date"]).dt.days
    df_delay["Estimated_TAT_Days"] = (df_delay["Estimated_Completion_Date"] - df_delay["Submission_Release_Date"]).dt.days
    df_delay["Delay_in_TAT_Days"] = df_delay["TAT_Days"] - df_delay["Estimated_TAT_Days"]

    def completion_status(row):
        if pd.isna(row["Completion_Date"]) or pd.isna(row["Estimated_Completion_Date"]):
            return "Unknown"
        elif row["Delay_Days"] < 0:
            return "Early"
        elif row["Delay_Days"] == 0:
            return "On Time"
        else:
            return "Delayed"
    df_delay["Completion_Status"] = df_delay.apply(completion_status, axis=1)
    df_delay["Rush_Label"] = df_delay["Rush"].map({"Y": "Rush", "N": "Non-Rush"}).fillna("Unknown")

    status_order = ["Early", "On Time", "Delayed"]
    color_map = {"Early": "#59a14f", "On Time": "#4c78a8", "Delayed": "#e45756"}

    # --- KPIs ---
    avg_actual_tat = df_delay["TAT_Days"].mean()
    avg_estimated_tat = df_delay["Estimated_TAT_Days"].mean()
    avg_early_days = df_delay.loc[df_delay["Completion_Status"] == "Early", "Delay_Days"].mean()
    avg_delay_days = df_delay.loc[df_delay["Completion_Status"] == "Delayed", "Delay_Days"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Actual TAT (days)", f"{avg_actual_tat:.2f}" if not pd.isna(avg_actual_tat) else "N/A")
    k2.metric("Avg Estimated TAT (days)", f"{avg_estimated_tat:.2f}" if not pd.isna(avg_estimated_tat) else "N/A")
    k3.metric("Avg Early Days", f"{avg_early_days:.2f}" if not pd.isna(avg_early_days) else "N/A")
    k4.metric("Avg Delay Days", f"{avg_delay_days:.2f}" if not pd.isna(avg_delay_days) else "N/A")

    st.markdown("---")

    # --- Donut: Completion Status ---
    st.markdown("### 🥯 Completion Status Overview")
    status_counts = (
        df_delay["Completion_Status"]
        .value_counts()
        .reindex(status_order + ["Unknown"], fill_value=0)
        .rename_axis("Status")
        .reset_index(name="Count")
    )
    fig_pie = px.pie(
        status_counts, names="Status", values="Count", hole=0.4,
        color="Status", color_discrete_map={**color_map, "Unknown": "#bab0ab"},
        category_orders={"Status": status_order + ["Unknown"]},
        title="Completion Status"
    )
    st.plotly_chart(fig_pie, use_container_width=True, key="status_pie")

    # --- Bar: Distribution of Delay Days (with outlier handling) ---
    st.markdown("### 📊 Distribution of Delay Days")
    delay_dist = df_delay.groupby("Delay_Days").size().reset_index(name="Test_Count")
    PLOT_MIN = -20
    PLOT_MAX = 90
    plot_delay_dist = delay_dist[(delay_dist["Delay_Days"] >= PLOT_MIN) & (delay_dist["Delay_Days"] <= PLOT_MAX)]
    outliers_count = delay_dist[(delay_dist["Delay_Days"] < PLOT_MIN) | (delay_dist["Delay_Days"] > PLOT_MAX)]["Test_Count"].sum()

    fig_delay_dist = px.bar(
        plot_delay_dist,
        x="Delay_Days",
        y="Test_Count",
        labels={
            "Delay_Days": "Delay Days (Negative=Early, 0=On Time, Positive=Delayed)",
            "Test_Count": "Number of Tests"
        },
        title="Number of Tests by Delay Days",
        color_discrete_sequence=["#4c78a8"]
    )
    fig_delay_dist.update_layout(
        xaxis=dict(dtick=1, range=[PLOT_MIN, PLOT_MAX]),
        bargap=0.1
    )
    st.plotly_chart(fig_delay_dist, use_container_width=True, key="delay_days_dist")

    if outliers_count > 0:
        st.info(f"{outliers_count} tests had delay days outside the -20 to +90 range and are not shown in this chart.")

    st.markdown("---")

    # --- Rush/Non-Rush Distribution (toggle for Volume/Avg Delay Days) ---
    st.markdown("## ⚡ Rush vs Non-Rush Completion Status Distribution")
    toggle_metric = st.toggle("Show charts by Avg Delay Days (toggle off for Volume)", value=False)
    metric_label = "Delay_Days" if toggle_metric else "Count"
    metric_title = "Avg Delay Days" if toggle_metric else "Volume"

    if toggle_metric:
        rush_status = (
            df_delay.groupby(["Rush_Label", "Completion_Status"])["Delay_Days"].mean()
            .unstack().reindex(columns=status_order, fill_value=0)
            .reset_index()
        )
        fig_rush = px.bar(
            rush_status, x="Rush_Label", y=status_order,
            barmode="group",
            color_discrete_map=color_map,
            labels={"value": "Avg Delay Days"},
            title="Rush vs Non-Rush: Avg Delay Days by Status"
        )
    else:
        rush_status = (
            df_delay.groupby(["Rush_Label", "Completion_Status"]).size()
            .unstack(fill_value=0).reindex(columns=status_order, fill_value=0)
            .reset_index()
        )
        fig_rush = px.bar(
            rush_status, x="Rush_Label", y=status_order,
            barmode="group",
            color_discrete_map=color_map,
            labels={"value": "Test Count"},
            title="Rush vs Non-Rush: Volume by Status"
        )
    st.plotly_chart(fig_rush, use_container_width=True, key="rush_status_dist")

    st.markdown("---")

    # --- Top 10 Tests by (toggle) for Early, On Time, Delayed ---
    st.markdown(f"## 🏆 Top 10 Tests by {metric_title} (by Status)")
    for status in status_order:
        st.markdown(f"#### Top 10 {status} Tests by {metric_title}")
        if toggle_metric:
            top_tests = (
                df_delay[df_delay["Completion_Status"] == status]
                .groupby("Test_Name")["Delay_Days"].mean()
                .nlargest(10).reset_index()
                .rename(columns={"Delay_Days": metric_label})
            )
        else:
            top_tests = (
                df_delay[df_delay["Completion_Status"] == status]
                .groupby("Test_Name").size()
                .nlargest(10).reset_index(name=metric_label)
            )
        fig = px.bar(
            top_tests, x="Test_Name", y=metric_label, text=metric_label,
            color="Test_Name", color_discrete_sequence=px.colors.qualitative.Pastel,
            title=f"Top 10 {status} Tests by {metric_title}"
        )
        if toggle_metric:
            fig.update_traces(texttemplate="%{y:.1f} days")
        st.plotly_chart(fig, use_container_width=True, key=f"top10_tests_{metric_label}_{status}")

    st.markdown("---")

    # --- Top 10 Delayed Clients: Toggle for Volume/Avg Delay Days, only Delayed ---
    st.markdown("## 🏢 Top 10 Delayed Clients")
    if toggle_metric:
        top_clients = (
            df_delay[df_delay["Completion_Status"] == "Delayed"]
            .groupby("Client_Facility")["Delay_Days"].mean()
            .nlargest(10).reset_index()
            .rename(columns={"Delay_Days": metric_label})
        )
    else:
        top_clients = (
            df_delay[df_delay["Completion_Status"] == "Delayed"]
            .groupby("Client_Facility").size()
            .nlargest(10).reset_index(name=metric_label)
        )
    fig = px.bar(
        top_clients, x="Client_Facility", y=metric_label, text=metric_label,
        color="Client_Facility", color_discrete_sequence=px.colors.qualitative.Prism,
        title=f"Top 10 Delayed Clients by {metric_title}"
    )
    if toggle_metric:
        fig.update_traces(texttemplate="%{y:.1f} days")
    st.plotly_chart(fig, use_container_width=True, key=f"top10_clients_{metric_label}_delayed")

    st.markdown("---")
    st.markdown("#### Download Raw Delay Data")
    generate_download_link(df_delay, "full_delay_data.csv")

#Forecast

with tabs[4]:
    st.header("Forecasting with Prophet (Selected Tests Only)")

    # Forecasting controls
    confidence_level = st.slider("Confidence Level (%)", 50, 99, 95, key="conf_slider")
    forecast_periods = st.slider("Periods to Predict", 4, 20, 8, key="period_slider")
    forecast_freq = st.radio("Forecast Granularity", ["Daily", "Monthly"], horizontal=True, key="freq_radio")
    freq = "D" if forecast_freq == "Daily" else "M"

    if not test_filter:
        st.info("Please select one or more tests from the sidebar to view forecast.")
        st.stop()

    # --- Filter completed tests by COMPLETION DATE in selected range ---
    completed_df = df_raw[
        (df_raw["Status"] == "Completed") &
        (df_raw["Completion_Date"].notna()) &
        (df_raw["Completion_Date"] >= pd.to_datetime(start_date)) &
        (df_raw["Completion_Date"] <= pd.to_datetime(end_date))
    ].copy()
    completed_df["ds"] = pd.to_datetime(completed_df["Completion_Date"], errors="coerce")

    for test in test_filter:
        st.subheader(f"Forecast for: {test}")
        test_df = completed_df[completed_df["Test_Name"] == test].copy().dropna(subset=["ds", "Price"])

        if test_df.empty or test_df["ds"].nunique() < 4:
            st.warning(f"Not enough valid data to forecast '{test}'")
            continue

        # Aggregate by selected frequency
        agg_df = test_df.groupby(pd.Grouper(key="ds", freq=freq)).agg(
            y=("Test_Name", "count"),
            price_avg=("Price", "mean")
        ).reset_index()
        agg_df = agg_df[agg_df["y"] > 0]

        # Prophet model
        m = Prophet(interval_width=confidence_level / 100)
        m.fit(agg_df[["ds", "y"]])
        future = m.make_future_dataframe(periods=forecast_periods, freq=freq)
        forecast = m.predict(future).merge(agg_df, on="ds", how="left")

        # Fill missing pricing
        forecast["price_avg"] = forecast["price_avg"].ffill().replace(0, np.nan).bfill()
        forecast["Predicted Volume"] = forecast["yhat"].clip(lower=0)
        forecast["Actual Volume"] = forecast["y"]
        forecast["Predicted Revenue"] = forecast["Predicted Volume"] * forecast["price_avg"]
        forecast["Actual Revenue"] = forecast["Actual Volume"] * forecast["price_avg"]
        forecast["Predicted Revenue Upper"] = forecast["yhat_upper"] * forecast["price_avg"]
        forecast["Predicted Revenue Lower"] = forecast["yhat_lower"] * forecast["price_avg"]

        # Linear regression for trend
        X = np.arange(len(forecast)).reshape(-1, 1)
        eval_df = forecast.dropna(subset=["Actual Volume"])
        lr = LinearRegression().fit(X[:len(eval_df)], eval_df["Actual Volume"])
        forecast["Vol_Linear"] = lr.predict(X)
        y_log = np.log1p(eval_df["Actual Volume"])
        exp_model = LinearRegression().fit(X[:len(y_log)], y_log)
        forecast["Vol_Exp"] = np.expm1(exp_model.predict(X))

        # --- Volume Forecast Plot ---
        fig_vol = go.Figure([
            go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False),
            go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", fillcolor="rgba(76, 120, 168, 0.2)", line=dict(width=0), name="Confidence Band"),
            go.Scatter(x=forecast["ds"], y=forecast["Actual Volume"], mode="lines+markers", name="Actual", line=dict(color="white")),
            go.Scatter(x=forecast["ds"], y=forecast["Predicted Volume"], mode="lines", name="Forecast", line=dict(color="lightblue")),
            go.Scatter(x=forecast["ds"], y=forecast["Vol_Linear"], mode="lines", name="Linear Trend", line=dict(color="orange", dash="dot")),
            go.Scatter(x=forecast["ds"], y=forecast["Vol_Exp"], mode="lines", name="Exponential Trend", line=dict(color="teal", dash="dash"))
        ])
        fig_vol.update_layout(title=f"{test} - Test Volume Forecast", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True, key=f"forecast_vol_{test}")

        # --- Volume Summary Table ---
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

        st.subheader("Volume Forecast Summary")
        st.dataframe(pd.concat([daily_avg, month_summary, quarter_summary], ignore_index=True), use_container_width=True)

        # --- Revenue Forecast ---
        lr_r = LinearRegression().fit(X[:len(eval_df)], eval_df["Actual Revenue"])
        forecast["Rev_Linear"] = lr_r.predict(X)
        y_log_r = np.log1p(eval_df["Actual Revenue"])
        exp_model_r = LinearRegression().fit(X[:len(y_log_r)], y_log_r)
        forecast["Rev_Exp"] = np.expm1(exp_model_r.predict(X))

        fig_rev = go.Figure([
            go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue Upper"], line=dict(width=0), showlegend=False),
            go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue Lower"], fill="tonexty", fillcolor="rgba(89, 161, 79, 0.2)", line=dict(width=0), name="Confidence Band"),
            go.Scatter(x=forecast["ds"], y=forecast["Actual Revenue"], mode="lines+markers", name="Actual", line=dict(color="white")),
            go.Scatter(x=forecast["ds"], y=forecast["Predicted Revenue"], mode="lines", name="Forecast", line=dict(color="green")),
            go.Scatter(x=forecast["ds"], y=forecast["Rev_Linear"], mode="lines", name="Linear Trend", line=dict(color="purple", dash="dot")),
            go.Scatter(x=forecast["ds"], y=forecast["Rev_Exp"], mode="lines", name="Exponential Trend", line=dict(color="gold", dash="dash"))
        ])
        fig_rev.update_layout(title=f"{test} - Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue")
        st.plotly_chart(fig_rev, use_container_width=True, key=f"forecast_rev_{test}")

        # --- Revenue Summary Table ---
        daily_avg_r = pd.DataFrame([{
            "Group_Type": f"{forecast_freq} (avg)",
            "Group": f"{future_only['ds'].min().date()} to {future_only['ds'].max().date()}",
            "Predicted Revenue": round(future_only["Predicted Revenue"].mean(), 2),
            "Actual Revenue": np.nan
        }]) if not future_only.empty else pd.DataFrame()

        month_summary_r = forecast.groupby("Month").agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"}).reset_index().rename(columns={"Month": "Group"}).assign(Group_Type="Month")
        quarter_summary_r = forecast.groupby("Quarter").agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"}).reset_index().rename(columns={"Quarter": "Group"}).assign(Group_Type="Quarter")

        st.subheader("Revenue Forecast Summary")
        st.dataframe(pd.concat([daily_avg_r, month_summary_r, quarter_summary_r], ignore_index=True), use_container_width=True)


# === Statistical Insights === #
with tabs[5]:
    st.markdown(
        """
        <div style="background-color:#f0f2f6;padding:1.2rem;border-radius:10px;margin-bottom:1.5rem;">
        <span style="font-size:1.2rem; color:#4c78a8;">
        <b>ℹ️ Statistical Insights by Test:</b> All tables below reflect <b>current filters</b> and date range.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("Statistical Insights by Test")

    # Apply filters
    filtered_df = df.copy()
    if test_filter:
        filtered_df = filtered_df[filtered_df["Test_Name"].isin(test_filter)]
    if client_filter:
        filtered_df = filtered_df[filtered_df["Client_Facility"].isin(client_filter)]
    if analyst_filter:
        filtered_df = filtered_df[filtered_df["Analyst"].isin(analyst_filter)]

    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters.")
        st.stop()

    # 1. Revenue Summary by Test
    st.subheader("Revenue Summary by Test")
    if all(col in filtered_df.columns for col in ["Test_Name", "Price"]):
        test_summary = filtered_df.groupby("Test_Name")["Price"].agg(
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
    st.subheader("Revenue Summary by Client")
    if all(col in filtered_df.columns for col in ["Client_Facility", "Price"]):
        client_summary = filtered_df.groupby("Client_Facility")["Price"].agg(
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
    st.subheader("Revenue Summary by Test & Client")
    if all(col in filtered_df.columns for col in ["Test_Name", "Client_Facility", "Price"]):
        combo_summary = filtered_df.groupby(["Test_Name", "Client_Facility"])["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean"
        ).reset_index().round(2)
        st.dataframe(combo_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(combo_summary, "test_client_summary.csv")
    else:
        st.warning("Missing columns: 'Test_Name', 'Client_Facility', or 'Price'")

    # 4. Revenue Summary by Facility Type
    st.subheader("Revenue Summary by Facility Type")
    if all(col in filtered_df.columns for col in ["Facility_Type", "Price"]):
        facility_summary = filtered_df.groupby("Facility_Type")["Price"].agg(
            Test_Count="count",
            Total_Revenue="sum",
            Avg_Revenue="mean",
            Std_Dev="std"
        ).reset_index().round(2)
        st.dataframe(facility_summary.sort_values("Total_Revenue", ascending=False), use_container_width=True)
        generate_download_link(facility_summary, "facility_summary.csv")
    else:
        st.warning("Missing columns: 'Facility_Type' or 'Price'")

    # 5. Correlation: Test Volume vs. Average Delay
    st.subheader("Correlation: Test Volume vs. Average Delay")
    if all(col in filtered_df.columns for col in ["Test_Name", "Delay_Days"]):
        delay_volume = filtered_df.groupby("Test_Name").agg(
            Volume=("Test_Name", "count"),
            Avg_Delay=("Delay_Days", "mean")
        ).reset_index()
        delay_volume = delay_volume.dropna(subset=["Avg_Delay"])
        st.dataframe(delay_volume.sort_values("Avg_Delay", ascending=False), use_container_width=True)
        generate_download_link(delay_volume, "test_volume_vs_avg_delay.csv")

        # Calculate and display correlation
        if not delay_volume.empty and delay_volume["Volume"].nunique() > 1 and delay_volume["Avg_Delay"].nunique() > 1:
            corr = delay_volume["Volume"].corr(delay_volume["Avg_Delay"])
            st.info(f"Correlation coefficient between test volume and average delay: **{corr:.2f}**")
            fig_corr = px.scatter(
                delay_volume, x="Volume", y="Avg_Delay", text="Test_Name",
                trendline="ols", title="Test Volume vs. Average Delay"
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="statinsights_vol_delay_scatter")
        else:
            st.info("Not enough data for correlation analysis.")
    else:
        st.warning("Missing columns: 'Test_Name' or 'Delay_Days'")

    st.markdown(
        """
        <style>
        .stDataFrame {background-color: #fff; border-radius: 10px; padding: 10px;}
        .stPlotlyChart {background-color: #fff; border-radius: 10px; padding: 10px;}
        </style>
        """,
        unsafe_allow_html=True
    )

    
