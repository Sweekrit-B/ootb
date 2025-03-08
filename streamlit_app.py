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
from drive import download_parquet_from_drive_link

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

@st.cache_data
def run_data_prep(file):
    tqdm.pandas()

    drive_link = link_input

    # Download and read the Parquet file
    client_secrets = st.secrets["CLIENT_SECRETS_JSON"]
    client_secrets = {"web": dict(client_secrets)}
    df = download_parquet_from_drive_link(drive_link, client_secrets)
    df = preprocess(df)

    all_events, visitors, buyers, orders = split_by_metric(df)

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

    return all_events_geography_contributions, all_events_devices_contributions, buyers_geography_contributions, buyers_devices_contributions, visitors_geography_contributions, visitors_devices_contributions, orders_geography_contributions, orders_devices_contributions

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
        all_events_geography_contributions, all_events_devices_contributions, buyers_geography_contributions, buyers_devices_contributions, visitors_geography_contributions, visitors_devices_contributions, orders_geography_contributions, orders_devices_contributions = run_data_prep("al_15day_export.parquet")
        inputs_valid = True
    except Exception as e:
        st.markdown(f"{e}")
        inputs_valid = False

if inputs_valid:
    st.header("Get contributions")
    input = st.selectbox(
        "Which metric would you like to view?",
        ("Visitors", "Orders", "Buyers", "All Events"),
        index=None,
        placeholder="Select your metric...",
    )
    date = st.date_input("What date do you want to view?", value="2025-01-06", min_value=min(all_events_geography_contributions['ds']), max_value=max(all_events_geography_contributions['ds']))

    if not input or date == "YYYY-MM-DD":
        st.text("Please select all options")
        inputs_valid = False
    # if date.replace("-", "").isdigit() and input:
    #     date_format = "%Y-%m-%d"
    #     date_obj = datetime.strptime(date, date_format)
    #     start_date = datetime(2025, 1, 6)
    #     end_date = datetime(2025, 1, 21)
    #     if date_obj < start_date or date_obj > end_date:
    #         st.text("Please enter a valid date")
    #     else:
    #         inputs_valid = True

    if inputs_valid:
        if input.lower() == "buyers":
            st.header("Buyers")
            with st.expander("Buyers - geography"):
                st.dataframe(buyers_geography_contributions, use_container_width=True)
            with st.expander("Buyers - devices"):
                st.dataframe(buyers_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover(get_anomaly_dates(buyers_devices_contributions), \
                merge(buyers_devices_contributions, buyers_geography_contributions), date, title="Anomaly Contributions to Buyers")
            st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
        elif input.lower() == "visitors":
            st.header("Visitors")
            with st.expander("Visitors - geography"):
                st.dataframe(visitors_geography_contributions, use_container_width=True)
            with st.expander("Visitors - devices"):
                st.dataframe(visitors_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover(get_anomaly_dates(visitors_devices_contributions), \
                merge(visitors_devices_contributions, visitors_geography_contributions), date, title="Anomaly Contributions to Visitors")
            st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
        elif input.lower() == "orders":
            st.header("Orders")
            with st.expander("Orders - geography"):
                st.dataframe(orders_geography_contributions, use_container_width=True)
            with st.expander("Orders - devices"):
                st.dataframe(orders_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover(get_anomaly_dates(orders_devices_contributions), \
                merge(orders_devices_contributions, orders_geography_contributions), date, title="Anomaly Contributions to Orders")
            st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
        elif input.lower() == "all events":
            st.header("All Events")
            with st.expander("All Events - geography"):
                st.dataframe(all_events_geography_contributions, use_container_width=True)
            with st.expander("All Events - devices"):
                st.dataframe(all_events_devices_contributions, use_container_width=True)
            figure = plot_anomaly_chart_with_hover(get_anomaly_dates(all_events_devices_contributions), \
                merge(all_events_devices_contributions, all_events_geography_contributions), date, title="Anomaly Contributions to All Events")
            st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)



