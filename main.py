# %% Import statements
import pandas as pd
import pyarrow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from user_agents import parse
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from drive import download_parquet_from_drive_link
# %% Preprocessing Functions

tqdm.pandas()

def parse_user_agent(user_agent):
    ua = parse(user_agent)
    return f"{ua.os.family}"

def configure_time(df):
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='mixed')
    df['event_timestamp'] = df['event_timestamp'].dt.tz_localize(None)
    df["time_hour"] = df["event_timestamp"].dt.floor("h")
    return df

def configure_device_and_geography(df):
    df['device'] = df['user_agent'].progress_apply(parse_user_agent)
    df['device'] = df['device'].apply(lambda x: 'Linux' if x == 'Linux Mint' else x)
    df['device'] = df['device'].apply(lambda x: 'Other' if (x == 'Tizen') or (x == 'Ubuntu') or (x == 'OpenBSD') or (x == 'FreeBSD') or (x == "BlackBerry OS") else x)
    df['geography'] = df.get('shop').apply(lambda shop: 'Global' if shop == 'abbott-lyon-global.myshopify.com' else 'US')

def preprocess(df):
    configure_time(df)
    print("Finished configuring time")
    configure_device_and_geography(df)
    print("Finished configuring device and geography")
    return df

#%% Data Splitting

def split_by_metric(df):
    all_events = df
    visitors = df.drop_duplicates(subset='event_details_clientId')
    buyers = df[df['event_name'] == 'checkout_completed'].drop_duplicates(subset='event_details_clientId')
    orders = df[df['event_name'] == 'checkout_completed']
    return all_events, visitors, buyers, orders

def split_by_dimension(df, dimension):
    metric_by_dim = df.groupby(["time_hour", dimension]).size().reset_index(name='y')
    metric_by_dim = metric_by_dim.pivot(index='time_hour', columns=dimension, values='y')
    metric_by_dim = metric_by_dim.fillna(0)
    metric_by_dim['Total'] = metric_by_dim.sum(axis=1)
    metric_by_dim = metric_by_dim.reset_index()
    return metric_by_dim

# %% Prophet model and contribution processing functions

def prophet_model(df, dimension):

    """
    Fits a Prophet model to the given dataframe and dimension, and identifies anomalies.
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data with a 'time_hour' column and the specified dimension column.
    dimension (str): The column name of the dimension to be modeled and analyzed for anomalies.
    Returns:
    pd.DataFrame: A DataFrame containing the forecasted values, original values, anomaly indicators, and additional metrics.
    """

    subset = df[['time_hour', dimension]].reset_index()
    subset = subset.assign(y=subset[dimension]).assign(ds=subset['time_hour']).drop(columns=['time_hour', dimension])

    #Initializing and fitting the prophet model
    m = Prophet()
    m.fit(subset)

    #Creating prophet predictions on historical data
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)

    #Assigning forecast dataframe the initial y values (no. of events using this device), is_anomaly (if it is an anomaly or not), and anomaly_weight (how far it is from either yhat_upper or lower)
    forecast['y'] = subset['y']
    forecast['is_anomaly'] = forecast.apply(lambda row: 1 if not (row['yhat_lower'] <= row['y'] <= row['yhat_upper']) else 0, axis=1)
    forecast['diff'] = abs(forecast['yhat'] - forecast['y'])
    forecast['diff sign'] = forecast['yhat'] - forecast['y']
    forecast['percent diff'] = (abs(forecast['yhat'] - forecast['y'])/forecast['y'].where(forecast['y'] != 0))
    forecast['percent diff sign'] = ((forecast['yhat'] - forecast['y'])/forecast['y'].where(forecast['y'] != 0))
    forecast['anomaly_weight'] = forecast.apply(lambda row: determine_anomaly_weight(forecast, row), axis=1)

    #Plotting forecast, anomaly, and forecast components
    fig1 = m.plot(forecast)
    anomalies = forecast[forecast['is_anomaly'] == 1]
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=20, label='Anomalies')
    plt.title(f"{dimension}")

    return forecast

def determine_anomaly_weight(df, row):
    """
    Determine the weight of an anomaly in a given row of a DataFrame.
    This function calculates the anomaly weight based on the difference between
    the actual value and the predicted upper and lower bounds within a specified
    time window around the anomaly.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data with columns 'ds', 'y', 'yhat_upper', and 'yhat_lower'.
    row (pandas.Series): A row from the DataFrame that includes 'ds', 'y', 'yhat_upper', 'yhat_lower', and 'is_anomaly' columns.
    Returns:
    float: The calculated anomaly weight. If the row is not an anomaly, returns 0.
    """
  
    if row['is_anomaly'] == 1:
        #print(f"Processing anomaly at {df['ds']}")
        subset = df[(df['ds'] >= row['ds'] - pd.Timedelta(hours=8)) & (df['ds'] <= row['ds'] + pd.Timedelta(hours=8))]
        #print(len(subset))
        if not subset.empty:
        #print("Subset is not empty")
            result = min(abs(subset['yhat_upper'] - row['y']).min(),
                        abs(subset['yhat_lower'] - row['y']).min())
        else:
            #print("Subset is empty")
            result = abs(row['yhat_upper'] - row['y'])
    else:
        result = 0
    return result

def anomaly_contribution(site_visits_top_level, site_visits_bottom_level, dimension_list):
    """
    Calculate the contribution of anomalies for each dimension and aggregate the results.
    Parameters:
    site_visits_top_level (DataFrame): DataFrame containing top-level site visit data.
    site_visits_bottom_level (DataFrame): DataFrame containing bottom-level site visit data.
    dimension_list (list): List of dimensions to analyze for anomaly contributions.
    Returns:
    DataFrame: A new DataFrame with added columns for predicted values, differences, 
               and percentage differences for each dimension, as well as aggregated values.
    """
  
    site_visits_top_level_new = site_visits_top_level.copy()
    site_visits_top_level_new["pred added"] = 0
    site_visits_top_level_new["diff added"] = 0
    site_visits_top_level_new["diff sign added"] = 0

    for dimension in dimension_list:
        data_curr = prophet_model(site_visits_bottom_level, dimension)
        site_visits_top_level_new["pred "+dimension] = data_curr['yhat']
        site_visits_top_level_new["diff "+dimension] = data_curr['diff']
        site_visits_top_level_new["diff sign "+dimension] = data_curr['diff sign']
        site_visits_top_level_new['percent diff ' + dimension] = data_curr['percent diff']
        site_visits_top_level_new['percent diff sign ' + dimension] = data_curr['percent diff sign']


        #Aggregated anomaly weights by dimension for verification with total anomaly weight found at top level
        site_visits_top_level_new["pred added"] += site_visits_top_level_new["pred "+dimension]
        site_visits_top_level_new["diff added"] += site_visits_top_level_new["diff "+dimension]
        site_visits_top_level_new["diff sign added"] += site_visits_top_level_new["diff sign "+dimension]


    for dimension in dimension_list:
        site_visits_top_level_new["perc diff "+dimension] = site_visits_top_level_new["diff "+dimension]/site_visits_top_level_new["diff added"]
        print(dimension + " added!")
    return site_visits_top_level_new

def find_eps(df):
    """
    Calculate the optimal epsilon value for DBSCAN clustering using the k-nearest neighbors method.
    Parameters:
    df (DataFrame): The input data frame containing the features for clustering.
    Returns:
    float: The optimal epsilon value determined by the knee point in the k-distance graph.
    This function performs the following steps:
    1. Fits a NearestNeighbors model to the input data with 8 neighbors.
    2. Computes the distances to the nearest neighbors.
    3. Sorts the distances and extracts the distance to the first nearest neighbor.
    4. Uses the KneeLocator to find the knee point in the sorted distance graph.
    5. Plots the k-distance graph with the knee point.
    6. Returns the distance value at the knee point, which is the optimal epsilon value.
    """
    
    nn = NearestNeighbors(n_neighbors=8).fit(df)
    distances, indices = nn.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    kneedle.plot_knee()
    return distances[kneedle.elbow]

def find_maximum_contributors(df, index, eps):
    """
    Identifies the columns in a DataFrame that contribute the most to a specific row's value.
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    index (int): The index of the row to analyze.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    Returns:
    list: A list of column names that are the maximum contributors to the specified row's value.
    """

    reshaped_row = np.array(df.iloc[index].to_list()).reshape(-1, 1)
    db = DBSCAN(eps=eps*0.5, min_samples=1).fit(reshaped_row)
    labels = db.labels_
    clusters = {}
    for label in set(labels):
        clusters[label] = reshaped_row[labels == label]
    sorted_clusters = {k: clusters[k] for k in sorted(clusters, key=lambda k: max(clusters[k]), reverse=True)}
    max_contributors = sorted_clusters[list(sorted_clusters.keys())[0]]
    if len(sorted_clusters.keys()) == 1:
        return
    contributing_columns = []
    for val in max_contributors:
        col_index = list(reshaped_row).index(val)
        contributing_columns.append(df.columns[col_index])
    return contributing_columns

def create_contributor_col(df):
    """
    Creates a new column 'contributors' in the original DataFrame indicating the maximum contributors for each row.
    Parameters:
    df (pandas.DataFrame): The input DataFrame with a 'ds' column to be set as the index.
    Returns:
    pandas.DataFrame: The original DataFrame with an added 'contributors' column.
    Notes:
    - The function sets the 'ds' column as the index of the DataFrame.
    - It fills any NaN values in the DataFrame with 0.
    - It calculates an epsilon value using the find_eps function.
    - For each row in the DataFrame, it appends the maximum contributors to the 'contributors' column using the find_maximum_contributors function.
    """
  
    df = df.set_index("ds")
    df = df.fillna(0)
    eps = find_eps(df)
    contribution_list = []
    for i in range(df.shape[0]):
        contribution_list.append(find_maximum_contributors(df, i, eps))

    df['contributors'] = contribution_list
    return df

def total_metric_anomaly_contributors(site_visits_bottom_level, dimension_list):
    """
    Calculate the total metric anomaly contributors for a given set of site visits and dimensions.
    This function uses a Prophet model to predict the top-level site visits and then calculates the 
    anomaly contributions for each dimension. It identifies the dates with anomalies and aggregates 
    the anomaly contributions over time.
    Args:
        site_visits_bottom_level (pd.DataFrame): DataFrame containing the bottom-level site visits data.
        dimension_list (list): List of dimensions to consider for anomaly contributions.
    Returns:
        pd.DataFrame: DataFrame containing the anomaly contributions and a column indicating if the 
                      row is an anomaly.
    """

    site_visits_top_level = prophet_model(site_visits_bottom_level, 'Total')

    site_visits_contributions = anomaly_contribution(site_visits_top_level, site_visits_bottom_level, dimension_list)
    site_visits_top_level_new_anoms = site_visits_contributions[site_visits_contributions["is_anomaly"] == 1]

    # Assuming 'anomaly_dates' is a list of timestamps with anomalies
    anomaly_dates = site_visits_contributions[site_visits_contributions["is_anomaly"]==1]["ds"]   # Example dates

    # Convert to datetime format if not already
    anomaly_dates = pd.to_datetime(anomaly_dates)

    # Aggregate anomaly contributions over time
    anomaly_contributions_grouped = site_visits_contributions.set_index('ds')[["perc diff " + dim for dim in dimension_list]]
    anomaly_percentages_grouped = site_visits_contributions.set_index('ds')[ ["percent diff " + dim for dim in dimension_list]]
    return_df = create_contributor_col(anomaly_percentages_grouped.reset_index())
    # print(site_visits_contributions['is_anomaly'])

    # Reset indices to ensure proper alignment
    return_df = return_df.reset_index(drop=True)
    site_visits_contributions = site_visits_contributions.reset_index(drop=True)

    # Ensure is_anomaly is set
    return_df["is_anomaly"] = site_visits_contributions["is_anomaly"]
    return_df['ds'] = site_visits_contributions['ds']
    return return_df

def get_dimensions(df):
    dimensions = list(df.columns)
    dimensions.remove('time_hour')
    dimensions.remove('Total')
    return dimensions

def save_contributor_processing(df, name):
    df_contributors = total_metric_anomaly_contributors(df, get_dimensions(df))
    df_contributors.to_csv(f"{name}_contributors.csv")
    return df_contributors

# %% Graphing

def merge(df1, df2):
    merged = df1.merge(df2, on='ds')
    return merged

def get_anomaly_dates(df):
    return pd.to_datetime(df[df['is_anomaly'] == 1]['ds'])

def create_hover_text(row):
    contributors_x = row['contributors_x']
    contributors_y = row['contributors_y']
    hover_text = '<b>%{x}</b><br>'
    print(contributors_x)
    print(type(contributors_x))
    print(contributors_y)
    
    if type(contributors_x) == str:
        contributors_x = contributors_x.replace("percent diff ", "").replace("[", "").replace("]", "")
        hover_text += f"Contributors Devices: {contributors_x}<br>"
    else:
        hover_text += f"Contributors Devices: None<br>"
    
    if type(contributors_y) == str:
        contributors_y = contributors_y.replace("percent diff ", "").replace("[", "").replace("]", "")
        hover_text += f"Contributors Geography: {contributors_y}<br>"
    else:
        hover_text += f"Contributors Geography: None<br>"
    
    hover_text += "Contribution: %{y}<br>"
    return hover_text

def plot_anomaly_chart_with_hover(anomaly_dates, merged, target_day, title):

    target_day = pd.to_datetime(target_day)
    merged['ds'] = pd.to_datetime(merged['ds'])
    filtered_data = merged[merged['ds'].dt.date == target_day.date()]

    # Rename columns
    filtered_data.columns = [col.replace("percent diff ", "") for col in filtered_data.columns]

    # Aggregate anomaly contributions over time for the filtered data
    # Extract unique contributors from Contributors_x and Contributors_y
    contributors_x = list(item for sublist in filtered_data['contributors_x'].dropna() for item in sublist)
    contributors_x = set([col.replace("percent diff ", "") for col in contributors_x])
    contributors_y = list(item for sublist in filtered_data['contributors_y'].dropna() for item in sublist)
    contributors_y = set([col.replace("percent diff ", "") for col in contributors_y])

    # Combine the contributors
    selected_columns = list(contributors_x.union(contributors_y))

    # Filter the DataFrame to include only the selected columns
    filtered_df = filtered_data[['ds', 'contributors_x', 'contributors_y'] + selected_columns].set_index('ds')

    # Create a Plotly figure
    fig = px.area(filtered_df, x=filtered_df.index, y=selected_columns,
                  title=f"{title}",
                  labels={"value": "Contribution Percentage", "variable": "Contributors"})

    for anomaly_time in anomaly_dates:
        if anomaly_time.date() == target_day.date():
            fig.add_vrect(x0=anomaly_time - pd.Timedelta(hours=1), x1=anomaly_time + pd.Timedelta(hours=1),
                      fillcolor="red", opacity=0.3, line_width=0)

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Contribution Percentage",
        legend_title="Contributors",
        hovermode="x unified"
    )

    hover_text = filtered_df.apply(create_hover_text, axis=1)

    fig.update_layout(
        hovermode="x unified"
    )

    fig.update_traces(
        hovertemplate=hover_text
    )

    # Show the figure
    return fig
    # fig.show()

# %% Run all functions and create intermediary CSVs

if __name__ == "__main__":
    drive_link = "https://drive.google.com/file/d/1T31Y3ch6tESLpwZ4yPWAaUJaI9W95q7T/view?usp=sharing"

    # Download and read the Parquet file
    df = download_parquet_from_drive_link(drive_link)
    # df = pd.read_parquet("al_15day_export.parquet")
    print("Finished processing Parquet file")

    df = preprocess(df)
    df.to_csv("preprocessed.csv", index=False)

    df = pd.read_csv("preprocessed.csv")

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

    # plot_anomaly_chart_with_hover(get_anomaly_dates(visitors_devices_contributions), merge(visitors_devices_contributions, visitors_geography_contributions), '2025-01-06')
