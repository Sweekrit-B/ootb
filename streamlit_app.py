from main import get_anomaly_dates, merge, \
    plot_anomaly_chart_with_hover
import pandas as pd
import streamlit as st
from datetime import datetime

st.markdown("""
    <style>
    .stTextInput input {
        color: #bfbbbb;
    }
    </style>
    """, unsafe_allow_html=True)

all_events_geography = pd.read_csv("all_events_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
all_events_devices = pd.read_csv("all_events_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
buyers_geography = pd.read_csv("buyers_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
buyers_devices = pd.read_csv("buyers_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
visitors_geography = pd.read_csv("visitors_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
visitors_devices = pd.read_csv("visitors_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
orders_geography = pd.read_csv("orders_geography_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
orders_devices = pd.read_csv("orders_devices_contributors.csv").drop(columns="Unnamed: 0").set_index('ds').reset_index()
inputs_valid = True

st.title("📈 Get Anomaly Contributions for Time Series Customer Data")
st.header("Get contributions")
input = st.selectbox(
    "Which metric would you like to view?",
    ("Visitors", "Orders", "Buyers", "All Events"),
    index=None,
    placeholder="Select your metric...",
)
date = st.date_input("What date do you want to view?", value="2025-01-06", min_value=min(all_events_geography['ds']), max_value=max(all_events_geography['ds']))

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
            st.dataframe(buyers_geography, use_container_width=True)
        with st.expander("Buyers - devices"):
            st.dataframe(buyers_devices, use_container_width=True)
        figure = plot_anomaly_chart_with_hover(get_anomaly_dates(buyers_devices), \
            merge(buyers_devices, buyers_geography), date, title="Anomaly Contributions to Buyers")
        st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
    elif input.lower() == "visitors":
        st.header("Visitors")
        with st.expander("Visitors - geography"):
            st.dataframe(visitors_geography, use_container_width=True)
        with st.expander("Visitors - devices"):
            st.dataframe(visitors_devices, use_container_width=True)
        figure = plot_anomaly_chart_with_hover(get_anomaly_dates(visitors_devices), \
            merge(visitors_devices, visitors_geography), date, title="Anomaly Contributions to Visitors")
        st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
    elif input.lower() == "orders":
        st.header("Orders")
        with st.expander("Orders - geography"):
            st.dataframe(orders_geography, use_container_width=True)
        with st.expander("Orders - devices"):
            st.dataframe(orders_devices, use_container_width=True)
        figure = plot_anomaly_chart_with_hover(get_anomaly_dates(orders_devices), \
            merge(orders_devices, orders_geography), date, title="Anomaly Contributions to Orders")
        st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)
    elif input.lower() == "all events":
        st.header("All Events")
        with st.expander("All Events - geography"):
            st.dataframe(orders_geography, use_container_width=True)
        with st.expander("All Events - devices"):
            st.dataframe(orders_devices, use_container_width=True)
        figure = plot_anomaly_chart_with_hover(get_anomaly_dates(all_events_devices), \
            merge(all_events_devices, all_events_geography), date, title="Anomaly Contributions to All Events")
        st.plotly_chart(figure, use_container_width=True, theme="streamlit", key=None)



