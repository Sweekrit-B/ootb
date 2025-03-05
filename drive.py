# Install required packages:
# pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas pyarrow

import re
import os
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json


# Define scopes - we need drive.readonly access to download files
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def extract_file_id_from_url(url):
    """
    Extract file ID from Google Drive URL.
    Supports various Google Drive URL formats.
    """
    # Pattern for /file/d/{fileId}/view format
    file_id_match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if file_id_match:
        return file_id_match.group(1)
    
    # Pattern for id={fileId} format
    file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if file_id_match:
        return file_id_match.group(1)
    
    # If no patterns match, assume the URL might be just the ID
    if re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
    
    raise ValueError("Could not extract file ID from the provided URL")

def get_credentials():
    """
    Get and refresh OAuth credentials.
    Creates a token.json file to store credentials for future use.
    """
    creds = None
    
    # Check if token.json file exists with saved credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(
            json.loads(open('token.json').read()), 
            SCOPES
        )
    
    # If credentials don't exist or are invalid, go through auth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This requires client_secrets.json to be in the working directory
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=61899)
        
        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

def download_parquet_from_drive_link(drive_link, output_path=None):
    """
    Download a Parquet file from Google Drive using a sharing URL and OAuth.
    
    Args:
        drive_link (str): Google Drive sharing URL or file ID
        output_path (str, optional): Path where to save the file. If None, uses a temporary file.
        
    Returns:
        pandas.DataFrame: Data from the Parquet file
    """
    # Extract file ID from the link
    file_id = extract_file_id_from_url(drive_link)
    
    # Set default output path if not provided
    if output_path is None:
        output_path = f"temp_{file_id}.parquet"
    
    # Get OAuth credentials
    creds = get_credentials()
    
    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)
    
    # Create a request to download the file
    request = service.files().get_media(fileId=file_id)
    
    # Download the file to a bytes buffer
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    
    # Download the file in chunks
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download progress: {int(status.progress() * 100)}%")
    
    # Reset buffer position to beginning
    file_buffer.seek(0)
    
    # Read the Parquet file directly from the buffer
    try:
        df = pd.read_parquet(file_buffer)
    except Exception as e:
        # If reading directly from buffer fails, save to file first
        with open(output_path, 'wb') as f:
            f.write(file_buffer.getbuffer())
        df = pd.read_parquet(output_path)
    
    print(f"Successfully downloaded and loaded Parquet file with {len(df)} rows")
    
    return df

 
# # Replace with your Google Drive sharing URL
# drive_link = "https://drive.google.com/file/d/1T31Y3ch6tESLpwZ4yPWAaUJaI9W95q7T/view?usp=sharing"

# # Download and read the Parquet file
# df = download_parquet_from_drive_link(drive_link)

# # Display the first few rows of the DataFrame
# print(df.head())

# # Optional: Save the DataFrame to a local Parquet file
# # df.to_parquet("local_copy.parquet")