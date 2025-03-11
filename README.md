# Run Instructions

python -m streamlit run streamlit_app.py

# Documentation

## drive.py

This script provides functionality to authenticate with Google Drive via OAuth 2.0, extract file IDs from shared Google Drive URLs, and download Parquet files directly into a Pandas DataFrame.

## main.py

This script performs preprocessing, detects anomolies across metrics using Prophet, finds anomaly contribution of different dimensions (eg. device type, geography) by determining percentage difference between actual and predicted values, and identifies contributors using DBSCAN and KNN to cluster contributors into groups.

## streamlit_app.py

This Streamlit application creates an interactive web dashboard for analyzing and visualizing anomalies in e-commerce time series data using drive.py and main.py.

## setup.py

This file contains the build script for the streamlit app. To run, make sure to first install all packages with:

```
pip install -r requirements.txt
```

Then, run the build script:

```
setup.py build
```

Make sure to use the py / python3 extension before these commands if necessary.
