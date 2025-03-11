from main import parse_user_agent, configure_time, \
    configure_device_and_geography, preprocess, \
    split_by_metric, split_by_dimension, \
    prophet_model, determine_anomaly_weight, anomaly_contribution, \
    find_eps, find_maximum_contributors, create_contributor_col, \
    total_metric_anomaly_contributors, get_dimensions, save_contributor_processing, \
    get_anomaly_dates, merge, \
    plot_anomaly_chart_with_hover
import pandas as pd
import streamlit as st
from datetime import datetime
from tqdm import tqdm
import gdown
import io
from drive import download_parquet_from_drive_link

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stTextInput input {
        color: #bfbbbb;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Get Anomaly Contributions for Time Series Customer Data")
link_input = st.text_input("Google Drive Link: ")
# https://drive.google.com/file/d/1T31Y3ch6tESLpwZ4yPWAaUJaI9W95q7T/view?usp=sharing

def load_data_from_drive(drive_link):
    output = io.BytesIO()
    gdown.download(drive_link, output, quiet=False, fuzzy=True)
    output.seek(0)
    df = pd.read_parquet(output)
    return df

@st.cache_data
def run_data_prep(file):
    tqdm.pandas()

    drive_link = link_input
    print("Starting data download")
    # df = load_data_from_drive(drive_link)
    # Download and read the Parquet file
    client_secrets = st.secrets["CLIENT_SECRETS_JSON"]
    client_secrets = {"web": dict(client_secrets)}
    df = download_parquet_from_drive_link(drive_link, client_secrets)
    print("Downloaded data")

    df = preprocess(df)
    print("Preprocessed data")

    all_events, visitors, buyers, orders = split_by_metric(df)
    print("Split data")

    all_events_top = prophet_model(all_events.groupby("time_hour").size().reset_index(name='Total'), 'Total')
    visitors_top = prophet_model(visitors.groupby("time_hour").size().reset_index(name='Total'), 'Total')
    buyers_top = prophet_model(buyers.groupby("time_hour").size().reset_index(name='Total'), 'Total')
    orders_top = prophet_model(orders.groupby("time_hour").size().reset_index(name='Total'), 'Total')

    visitors_top = visitors_top[['ds', 'y', 'is_anomaly']]
    visitors_top = visitors_top.rename(columns={'ds': 'ds', 'y': 'visitors_y', 'is_anomaly': 'visitors_is_anomaly'})
    visitors_top['ds'] = pd.to_datetime(visitors_top['ds'])

    all_events_top = all_events_top[['ds', 'y', 'is_anomaly']]
    all_events_top = all_events_top.rename(columns={'ds': 'ds', 'y': 'all_events_y', 'is_anomaly': 'all_events_is_anomaly'})
    all_events_top['ds'] = pd.to_datetime(all_events_top['ds'])

    buyers_top = buyers_top[['ds', 'y', 'is_anomaly']]
    buyers_top = buyers_top.rename(columns={'ds': 'ds', 'y': 'buyers_y', 'is_anomaly': 'buyers_is_anomaly'})
    buyers_top['ds'] = pd.to_datetime(buyers_top['ds'])

    orders_top = orders_top[['ds', 'y', 'is_anomaly']]
    orders_top = orders_top.rename(columns={'ds': 'ds', 'y': 'orders_y', 'is_anomaly': 'orders_is_anomaly'})
    orders_top['ds'] = pd.to_datetime(orders_top['ds'])

    all_top_levels = visitors_top.merge(all_events_top, on='ds').merge(buyers_top, on='ds').merge(orders_top, on='ds')
    all_top_levels['visitors_is_anomaly'] = all_top_levels['visitors_is_anomaly'].apply(lambda x: 'Yes' if x == 1 else 'No')
    all_top_levels['all_events_is_anomaly'] = all_top_levels['all_events_is_anomaly'].apply(lambda x: 'Yes' if x == 1 else 'No')
    all_top_levels['buyers_is_anomaly'] = all_top_levels['buyers_is_anomaly'].apply(lambda x: 'Yes' if x == 1 else 'No')
    all_top_levels['orders_is_anomaly'] = all_top_levels['orders_is_anomaly'].apply(lambda x: 'Yes' if x == 1 else 'No')

    all_events_devices = split_by_dimension(all_events, "device")
    all_events_geography = split_by_dimension(all_events, "geography")

    visitors_devices = split_by_dimension(visitors, "device")
    visitors_geography = split_by_dimension(visitors, "geography")

    buyers_devices = split_by_dimension(buyers, "device")
    buyers_geography = split_by_dimension(buyers, "geography")

    orders_devices = split_by_dimension(orders, "device")
    orders_geography = split_by_dimension(orders, "geography")

    all_events_devices_contributions = save_contributor_processing(all_events_devices, "all_events_devices")
    all_events_geography_contributions = save_contributor_processing(all_events_geography, "all_events_geography")

    visitors_devices_contributions = save_contributor_processing(visitors_devices, "visitors_devices")
    visitors_geography_contributions = save_contributor_processing(visitors_geography, "visitors_geography")

    buyers_devices_contributions = save_contributor_processing(buyers_devices, "buyers_devices")
    buyers_geography_contributions = save_contributor_processing(buyers_geography, "buyers_geography")

    orders_devices_contributions = save_contributor_processing(orders_devices, "orders_devices")
    orders_geography_contributions = save_contributor_processing(orders_geography, "orders_geography")
    
    return all_events_geography_contributions, all_events_devices_contributions, buyers_geography_contributions, buyers_devices_contributions, visitors_geography_contributions, visitors_devices_contributions, orders_geography_contributions, orders_devices_contributions, all_top_levels

# all_events_geography = pd.read_csv("all_events_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# all_events_devices = pd.read_csv("all_events_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# buyers_geography = pd.read_csv("buyers_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# buyers_devices = pd.read_csv("buyers_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# visitors_geography = pd.read_csv("visitors_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# visitors_devices = pd.read_csv("visitors_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# orders_geography = pd.read_csv("orders_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
# orders_devices = pd.read_csv("orders_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()

inputs_valid = False

if link_input:
    try:
        all_events_geography_contributions, all_events_devices_contributions, buyers_geography_contributions, buyers_devices_contributions, visitors_geography_contributions, visitors_devices_contributions, orders_geography_contributions, orders_devices_contributions, all_top_levels = run_data_prep("al_15day_export.parquet")
        inputs_valid = True
    except Exception as e:
        st.markdown(f"{e}")
        inputs_valid = False

if inputs_valid:
    st.header("Get contributions")
    input_metric = st.selectbox(
        "Which metric would you like to view?",
        ("Visitors", "Orders", "Buyers", "All Events"),
        index=None,
        placeholder="Select your metric...",
    )
    date_range = st.date_input(
    "Select date range to view",
    value=(pd.to_datetime("2025-01-06"), pd.to_datetime("2025-01-08")),  # Default start and end dates
    min_value=min(all_events_geography_contributions['ds']).date(),  # Convert to date
    max_value=max(all_events_geography_contributions['ds']).date()    # Convert to date
    )

    if len(date_range) == 2:  # When a range is fully selected
        start_date, end_date = date_range
        st.write(f"Selected range: {start_date} to {end_date}")
        inputs_valid = True
    else:  # When only one date is selected (incomplete range)
        start_date = date_range[0]
        end_date = start_date  # Default to same day if range not fully selected
        st.write(f"Only start date selected: {start_date}. Using it as end date too.")
        inputs_valid = True
    # if date.replace("-", "").isdigit() and input:
    #     date_format = "%Y-%m-%d"
    #     date_obj = datetime.strptime(date, date_format)
    #     start_date = datetime(2025, 1, 6)
    #     end_date = datetime(2025, 1, 21)
    #     if date_obj < start_date or date_obj > end_date:
    #         st.text("Please enter a valid date")
    #     else:
    #         inputs_valid = True

    if input_metric:
        if input_metric.lower() == "buyers":
            st.header("Buyers")
            with st.expander("Buyers - geography"):
                st.dataframe(buyers_geography_contributions, use_container_width=True)
            with st.expander("Buyers - devices"):
                st.dataframe(buyers_devices_contributions, use_container_width=True)
                
            figure = plot_anomaly_chart_with_hover('buyers', merge(buyers_devices_contributions, buyers_geography_contributions), \
                all_top_levels, start_date, end_date)
        elif input_metric.lower() == "visitors":
            st.header("Visitors")
            with st.expander("Visitors - geography"):
                st.dataframe(visitors_geography_contributions, use_container_width=True)
            with st.expander("Visitors - devices"):
                st.dataframe(visitors_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover('visitors', merge(visitors_devices_contributions, visitors_geography_contributions), \
                all_top_levels, start_date, end_date)
        elif input_metric.lower() == "orders":
            st.header("Orders")
            with st.expander("Orders - geography"):
                st.dataframe(orders_geography_contributions, use_container_width=True)
            with st.expander("Orders - devices"):
                st.dataframe(orders_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover('orders', merge(orders_devices_contributions, orders_geography_contributions), \
                all_top_levels, start_date, end_date)
        elif input_metric.lower() == "all events":
            st.header("All Events")
            with st.expander("All Events - geography"):
                st.dataframe(all_events_geography_contributions, use_container_width=True)
            with st.expander("All Events - devices"):
                st.dataframe(all_events_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover('all_events', merge(all_events_devices_contributions, all_events_geography_contributions), \
                all_top_levels, start_date, end_date)



