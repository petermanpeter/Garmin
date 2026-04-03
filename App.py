import streamlit as st
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
#from __future__ import print_function
import os
from io import BytesIO
import base64
import datetime
import time
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from garminconnect import Garmin, GarminConnectAuthenticationError
import json
from math import radians, sin, cos, asin, sqrt
import math

#0 Target file to be imported
df_G = pd.read_csv('Activities_Run_20251202.csv')
garmin_file = 'Weight_20260111.xlsx'

def pace_to_float(t):
    if pd.isna(t) or str(t).strip() in ('', '--'):
        return np.nan
    parts = str(t).strip().split(':')
    try:
        if len(parts) == 2:  # mm:ss
            return int(parts[0]) + int(parts[1])/60
        elif len(parts) == 3:  # hh:mm:ss
            return int(parts[0])*60 + int(parts[1]) + int(parts[2])/60
    except:
        return np.nan
    return np.nan

# ---------------- CONFIG ----------------
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SEARCH_SENDER = "PICOOC"
SEARCH_SUBJECT = "Health Data file"
SAVE_DIR = "."
# ----------------------------------------


def gmail_authenticate():
    """Authenticate Gmail using OAuth (requires credentials.json)."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            #creds = flow.run_local_server(port=0)
            #creds = flow.run_console()
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("Please go to this URL: ", auth_url)
            code = input("Enter the authorization code here: ")
            creds = flow.fetch_token(code=code)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def search_latest_message(service, query):
    """Find the most recent Gmail message matching a query."""
    results = service.users().messages().list(userId="me", q=query, maxResults=1).execute()
    messages = results.get("messages", [])
    if not messages:
        return None
    msg_id = messages[0]["id"]
    return service.users().messages().get(userId="me", id=msg_id).execute()


def get_attachment_as_bytes(service, message):
    """Return (filename, bytes, received_date_str) for the Excel attachment."""
    headers = message.get("payload", {}).get("headers", [])
    date_header = next((h["value"] for h in headers if h["name"] == "Date"), None)
    if date_header:
        try:
            received_dt = datetime.datetime.strptime(date_header[:25], "%a, %d %b %Y %H:%M:%S")
        except Exception:
            received_dt = datetime.datetime.utcnow()
    else:
        received_dt = datetime.datetime.utcnow()

    date_str = received_dt.strftime("%Y%m%d")

    parts = message.get("payload", {}).get("parts", [])
    if not parts:
        parts = [message.get("payload", {})]

    for part in parts:
        filename = part.get("filename")
        if not filename:
            continue
        if filename.lower().endswith(".xlsx"):
            att_id = part["body"].get("attachmentId")
            if not att_id:
                continue
            att = service.users().messages().attachments().get(
                userId="me", messageId=message["id"], id=att_id
            ).execute()
            file_data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))
            return filename, file_data, date_str
    return None, None, date_str

def haversine_km(lat1, lon1, lat2, lon2):
    # Haversine distance between two lat/lon points in km [web:163][web:165]
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def add_km_markers(gps_df):
    """Return a DataFrame with one row per full km (1 km, 2 km, ...) along the route."""
    if len(gps_df) < 2:
        return pd.DataFrame(columns=["lat", "lon", "km"])

    cum_dist = [0.0]
    for i in range(1, len(gps_df)):
        d = haversine_km(
            gps_df.loc[i-1, "lat"], gps_df.loc[i-1, "lon"],
            gps_df.loc[i, "lat"], gps_df.loc[i, "lon"],
        )
        cum_dist.append(cum_dist[-1] + d)

    gps_df = gps_df.copy()
    gps_df["cum_km"] = cum_dist

    total_km = int(gps_df["cum_km"].iloc[-1])  # floor
    if total_km < 1:
        return pd.DataFrame(columns=["lat", "lon", "km"])

    marker_rows = []
    for k in range(1, total_km + 1):
        # index of point closest to k km
        idx = (gps_df["cum_km"] - k).abs().idxmin()
        marker_rows.append({
            "lat": gps_df.loc[idx, "lat"],
            "lon": gps_df.loc[idx, "lon"],
            "km": k,
        })
    return pd.DataFrame(marker_rows)

def guess_zoom_from_bounds(gps_df, viewport_width=800, viewport_height=500):
    """
    Calculate optimal Mapbox zoom for given lat/lon bounds.
    viewport_width/height approximate Streamlit container pixels.
    """
    if len(gps_df) < 2:
        return 14
    
    lat_min, lat_max = gps_df["lat"].min(), gps_df["lat"].max()
    lon_min, lon_max = gps_df["lon"].min(), gps_df["lon"].max()
    
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    
    if lat_span == 0 or lon_span == 0:
        return 14
    
    # Earth circumference in meters at equator
    C = 40075016.686
    LOG2 = math.log(2)
    
    # Center latitude for cos adjustment
    lat_center = (lat_max + lat_min) / 2
    
    # Lat/lon fractions of world
    lat_fraction = lat_span / 180.0
    lon_fraction = lon_span / 360.0
    
    # Meters per pixel needed
    lat_mpp = (C * lat_fraction) / viewport_height
    lon_mpp = (C * math.cos(math.radians(lat_center)) * lon_fraction) / viewport_width
    
    mpp = max(lat_mpp, lon_mpp)
    
    # Zoom formula: log2(C / mpp) - 8 (for 256px tiles)
    zoom = math.floor((math.log(C / mpp) / LOG2) - 8)
    
    # Clamp to valid Mapbox range (0-22)
    return max(0, min(22, zoom))

def extract_polyline_with_metrics(poly):
    """Extract lat, lon, elevation, and timestamp from polyline data."""
    pts = []
    elevations = []
    times = []
    
    for p in poly:
        # Extract lat/lon
        if isinstance(p, dict) and "lat" in p and "lon" in p:
            pts.append((p["lat"], p["lon"]))
            elevations.append(p.get("elevation", p.get("elev", None)))
            times.append(p.get("timestamp", None))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((p[0], p[1]))
            elevations.append(p[2] if len(p) > 2 else None)
            times.append(p[3] if len(p) > 3 else None)
    
    return pts, elevations, times

def calculate_segment_metrics(gps_df, start_idx, end_idx):
    """Calculate distance, time, and slope data for a segment."""
    if start_idx >= end_idx or start_idx >= len(gps_df) or end_idx > len(gps_df):
        return None
    
    segment = gps_df.iloc[start_idx:end_idx].copy()
    
    # Calculate distance
    cum_dist = [0.0]
    for i in range(1, len(segment)):
        d = haversine_km(
            segment.iloc[i-1]["lat"], segment.iloc[i-1]["lon"],
            segment.iloc[i]["lat"], segment.iloc[i]["lon"],
        )
        cum_dist.append(cum_dist[-1] + d)
    
    segment["cum_dist_km"] = cum_dist
    total_distance = cum_dist[-1]
    
    # Calculate elevation change and slope with proper None checks
    slopes = []
    if "elevation" in segment.columns and segment["elevation"].notna().any():
        for i in range(1, len(segment)):
            elev_curr = segment.iloc[i]["elevation"]
            elev_prev = segment.iloc[i-1]["elevation"]
            # Only calculate slope if both elevation values exist
            if pd.notna(elev_curr) and pd.notna(elev_prev):
                try:
                    elev_diff = float(elev_curr) - float(elev_prev)
                    dist_m = cum_dist[i] * 1000  # Convert km to m
                    if dist_m > 0:
                        slope = (elev_diff / dist_m) * 100  # Percentage slope
                        slopes.append(slope)
                    else:
                        slopes.append(0)
                except (TypeError, ValueError):
                    slopes.append(0)
            else:
                slopes.append(0)
        if slopes:
            segment["slope"] = [0] + slopes
        else:
            segment["slope"] = 0
    else:
        segment["slope"] = 0
    
    # Calculate time segment if available
    time_display = "N/A"
    if "timestamp" in segment.columns and segment["timestamp"].notna().any():
        # Try to calculate time between points
        pass
    
    return {
        "distance_km": total_distance,
        "segment": segment,
        "slopes": [s for s in slopes if s is not None] if slopes else None,
        "time_display": time_display
    }

def to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:  # pip install xlsxwriter
        df.to_excel(writer, index=False, sheet_name="Activities")
    buffer.seek(0)
    return buffer

def garmin_login_with_retry(email, password, max_retries=1, initial_delay=30):
    """
    Login to Garmin with long delays between retries.
    We use fewer retries but longer delays to avoid hitting rate limits.
    
    Args:
        email: Garmin email
        password: Garmin password
        max_retries: Maximum number of retry attempts (default 1 = just 2 attempts total)
        initial_delay: Initial delay in seconds before retry (default 30)
        
    Returns:
        Garmin API object if successful
        
    Raises:
        GarminConnectAuthenticationError: For auth failures
        Exception: For other errors after all retries exhausted
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            st.write(f"🔄 Attempt {attempt + 1}/{max_retries + 1}...")
            api = Garmin(email, password)
            api.login()
            return api
            
        except requests.exceptions.HTTPError as e:
            # Handle 429 (Too Many Requests) with exponential backoff
            if e.response.status_code == 429:
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)  # 30s, 60s, 120s, etc.
                    st.warning(
                        f"⏳ **HTTP 429: Rate Limited**\n\n"
                        f"Waiting **{delay} seconds** before retry {attempt + 1}/{max_retries}...\n\n"
                        f"*Garmin's servers are protecting against too many login attempts.*"
                    )
                    time.sleep(delay)
                    last_error = e
                    continue
                else:
                    raise Exception(
                        f"❌ Rate limited by Garmin after {max_retries} retries.\n\n"
                        f"Please wait **5-10 minutes** before trying again."
                    )
            else:
                # Other HTTP errors - don't retry
                raise
                
        except GarminConnectAuthenticationError as e:
            # Don't retry auth errors
            raise
            
        except Exception as e:
            # For non-HTTP errors, check if it's a rate limit wrapped in another exception
            error_str = str(e)
            if "429" in error_str or "Too Many Requests" in error_str or "Rate limited" in error_str:
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)
                    st.warning(
                        f"⏳ **Rate Limit Detected**\n\n"
                        f"Waiting **{delay} seconds** before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(delay)
                    last_error = e
                    continue
                else:
                    raise
            else:
                # Other exceptions - re-raise immediately
                raise
    
    if last_error:
        raise last_error


 
#1 Load the data
#service = gmail_authenticate()
#query = f'from:{SEARCH_SENDER} subject:"{SEARCH_SUBJECT}"'
#message = search_latest_message(service, query)
#if not message:
#    print("❌ No matching PICOOC email found.")
#filename, file_bytes, date_str = get_attachment_as_bytes(service, message)
#if not file_bytes:
#    print("❌ No .xlsx attachment found.")

# Save locally
#save_name = f"Weight_{date_str}.xlsx"
#with open(save_name, "wb") as f:
#    f.write(file_bytes)
#print(f"✅ Saved attachment as {save_name}")

# ---- NEW PART: Load into Pandas ----
try:
    df_weight = pd.read_excel(garmin_file)
    #df_weight = pd.read_excel(BytesIO(file_bytes))
    #print(f"✅ Loaded Excel into DataFrame ({len(df)} rows, {len(df.columns)} columns)")
    #print(df.head())  # preview first few rows
except Exception as e:
    print("⚠️ Failed to read Excel:", e)

# Find relative max/min indexes
n = 3 # window width for extrema
rel_max = argrelextrema(df_weight['Body weight(kg)'].values, np.greater, order=n)
rel_min = argrelextrema(df_weight['Body weight(kg)'].values, np.less, order=n)


#2 main program
(tab4,)  = st.tabs(['GarminConnect login'])
    
with tab4:  #GarminConnect login
    st.title('GarminConnect login')
    email = st.text_input("Garmin email")
    password = st.text_input("Garmin password", type="password")
    activity_type = 'run'
    activity_type = st.radio(
        "Choose activity type:",
        ['run', 'walk'],
        index=0,
        key="activity_type",
    )            
    # Initialize session state
    if "gc_api" not in st.session_state:
        st.session_state.gc_api = None
    if "gc_df" not in st.session_state:
        st.session_state.gc_df = None
    if "last_activity_type" not in st.session_state:
        st.session_state.last_activity_type = activity_type

    # trigger fetch when login OR activity_type changes
    fetch_needed = False
    if st.button("Login"):
        fetch_needed = True
    elif activity_type != st.session_state.last_activity_type:
        fetch_needed = True
        
    if fetch_needed:
        if not email or not password:
            st.error("Please enter email and password")
        else:
            try:
                st.info("🔐 Logging in to Garmin Connect...")
                api = garmin_login_with_retry(email, password)
                st.session_state.gc_api = api  # save API
                st.success("✅ Logged in successfully!")
                today = datetime.date.today()
                start_date = today - datetime.timedelta(days=365*5)

                if activity_type == 'run':
                    activity_keys = ["running", "trail_running", "treadmill_running"]
                else:
                    activity_keys = ["walking"]
                raw_acts = []
                start = 0
                page_size = 1000
                while True:
                    batch = api.get_activities(start, page_size)
                    if not batch:
                        break
                    raw_acts.extend(batch)
                    start += page_size
                acts = []
                for a in raw_acts:
                    act_date = datetime.datetime.strptime(a["startTimeLocal"][:10], "%Y-%m-%d").date()
                    if act_date < start_date:
                        continue
                    if a.get("activityType", {}).get("typeKey") not in activity_keys:
                        continue
                    acts.append(a)

                if not acts:
                    st.info("No run activities found in the last 5 years.")
                else:
                    df = pd.DataFrame(acts)
                    st.write("Found runs:", len(df))
                    df["distance_km"] = df["distance"] / 1000.0
                    df["distance_km"] = df["distance_km"].round(1)
                    st.session_state.gc_df = df  # save dataframe
                    st.session_state.last_activity_type = activity_type

    
            except GarminConnectAuthenticationError as e:
                st.error("❌ **Authentication Failed**\n\nCheck that your email and password are correct.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    st.error(
                        "❌ **HTTP 429 - Rate Limited by Garmin**\n\n"
                        "Too many requests detected. Please wait **5-10 minutes** before trying again."
                    )
                else:
                    st.error(f"❌ **HTTP Error {e.response.status_code}**\n\n{str(e)}")
            except Exception as e:
                error_str = str(e)
                st.error(
                    f"❌ **Login Error**\n\n"
                    f"```\n{error_str}\n```\n\n"
                    "**Troubleshooting:**\n"
                    "- Try logging in with a browser first to verify credentials\n"
                    "- Check if Garmin is down: https://status.garmin.com\n"
                    "- Wait 5-10 minutes if you had recent failed attempts\n"
                    "- Check your internet connection"
                )



# Always render UI if we have data
runs_df = st.session_state.get("gc_df")  # None if not set
api = st.session_state.get("gc_api")     # None if not set

# Early return if no data yet
if runs_df is None:
    st.info("👆 Please log in first to see your Garmin runs")
    st.stop()  # Stop script execution here

# Prepare Excel for download (all activities currently in runs_df)
excel_buffer = to_excel_bytes(runs_df)
today_str = datetime.date.today().strftime("%Y%m%d")  # yyyymmdd
current_activity_type = st.session_state.get("activity_type", "run")  # or just use activity_type if still in scope
st.download_button(
    label="Download Garmin activities as Excel",
    data=excel_buffer,
    file_name=f"garmin_activity_{current_activity_type}_{today_str}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="download_gc_runs",
)

GC_tab1, GC_tab2, GC_tab3, GC_tab4 = st.tabs(['Weight', 'Distance per month', 'Pacing vs Cadence', 'Map'])

with GC_tab1:  #Weight
    st.title('Weight vs Date')
    unit = 'lb'
    unit = st.radio("Choose unit:", ['lb', 'kg'], index=0, key="weight_unit")  # default kg
    #save_attachment(service, message)
    #df_weight = pd.read_excel('Weight_20250928.xlsx')
    df_weight['time'] = pd.to_datetime(df_weight['time'])
    if unit == 'kg':
        weight = df_weight['Body weight(kg)']
        yaxis_title = "Weight (kg)"
    else:
        weight = df_weight['Body weight(kg)'] * 2.20462  # convert to lb
        yaxis_title = "Weight (lb)"
    fig = go.Figure()
    # Main weight line
    fig.add_trace(go.Scatter(
        x=df_weight['time'], y=weight, mode='lines+markers', name='Weight',
        marker=dict(size=1, color='gray')
    ))
    # Relative maxima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_max], y=weight.iloc[rel_max], mode='markers', name='Rel max',
        marker=dict(size=2, color='red', symbol='diamond'),
        hovertemplate='<b>Max</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'+unit
    ))

    # Relative minima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_min], y=weight.iloc[rel_min], mode='markers', name='Rel min',
        marker=dict(size=2, color='blue', symbol='star'),
        hovertemplate='<b>Min</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'+unit
    ))

    fig.update_layout(
        title="Weight vs Date with Range Slider and Clickable Max/Min",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=12, label="12m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis_title=yaxis_title,
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

with GC_tab2: #Distance per month
    st.title('Distance per Month')
    df_G = runs_df.copy()
    df_G['Date'] = pd.to_datetime(df_G['startTimeLocal'], errors='coerce')
    df_G['Distance'] = df_G['distance'] / 1000.0
    
    df_G['month'] = df_G['Date'].dt.to_period('M')
    df_monthly = df_G.groupby('month')['Distance'].sum().reset_index()
    df_monthly['month'] = df_monthly['month'].dt.to_timestamp()

    df_G['Avg Pace'] = df_G['duration'] / 60 / df_G['Distance']
    df_G['Avg Pace ori'] = df_G['Avg Pace']
    df_G['Avg Pace'] = df_G['Avg Pace'].apply(pace_to_float)

    df_avg_pacing = df_G.groupby(['month'])['Avg Pace'].mean().reset_index()
    df_avg_pacing['month'] = df_avg_pacing['month'].dt.to_timestamp()

    df_G['run_type'] = pd.cut(df_G['Distance'], bins=[0, 7.5, 12.5, 17.5, np.inf], labels=['5km', '10km', '15km', '20km+'])

    # Get min and max dates
    min_date = df_monthly['month'].min().date()
    max_date = df_monthly['month'].max().date()

    # Range slider for selecting period
    start_date, end_date = st.slider(
        "Drag to select period:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )

    # Filter by slider range
    mask = (df_monthly['month'].dt.to_period('M') >= pd.Period(start_date, freq='M')) & (df_monthly['month'].dt.to_period('M') <= pd.Period(end_date, freq='M'))
    #mask = (df_monthly['month'].dt.date >= start_date) & (df_monthly['month'].dt.date <= end_date)
    df_filtered = df_monthly.loc[mask]

    # Calculate stats
    total_distance = df_filtered['Distance'].sum()
    filtered_yyyymm_min = pd.Period(start_date, freq='M')
    filtered_yyyymm_max = pd.Period(end_date, freq='M')
    total_months = (filtered_yyyymm_max.year - filtered_yyyymm_min.year) * 12 + (filtered_yyyymm_max.month - filtered_yyyymm_min.month) + 1
    #total_months = df_filtered['month'].nunique()
    df_filtered['Distance_rounded'] = df_filtered['Distance'].round(0)

    # Show dynamic metrics
    st.metric("Total Distance (km)", f"{total_distance:,.1f}")
    years = total_months // 12
    months = total_months % 12
    if years > 0:
        st.metric("Period", f"{years} year{'s' if years > 1 else ''} {months} month{'s' if months!= 1 else ''}")
    else:
        st.metric("Period", f"{months} month{'s' if months!= 1 else ''}")


    # Plot filtered chart
    fig = px.bar(
        df_filtered, x='month', y='Distance',
        title='Distance per Month',
        labels={'month': 'Month', 'Distance_rounded': 'Distance (km)'},
        text_auto='.0f',
    )
    st.plotly_chart(fig, use_container_width=True)
    # Prepare Excel for download (all activities currently in runs_df)
    excel_buffer = to_excel_bytes(df_filtered)
    today_str = datetime.date.today().strftime("%Y%m%d")  # yyyymmdd
    current_activity_type = st.session_state.get("activity_type", "run")  # or just use activity_type if still in scope
    
    st.download_button(
        label="Download distance by month summary as Excel",
        data=excel_buffer,
        file_name=f"garmin_{current_activity_type}_distance by month_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_df_filtered",
    )
with GC_tab3: #Pacing vs Cadence
    st.title('Pacing vs Cadence')
    df_G['Avg Pace'] = df_G['duration'] / 60 / df_G['Distance']
    st.write(df_G[['run_type', 'Avg Pace', 'averageRunningCadenceInStepsPerMinute']])
    if runs_df is None:
        st.info("No run data available")
        st.stop()
    #df_G = runs_df.copy()
    #
    run_types = ['5km', '10km', '15km', '20km+']
    selected = st.multiselect('Select Run Types:', run_types, default=run_types)
    color_map = {'5km': 'blue', '10km': 'red', '15km': 'green', '20km+': 'purple'}
    fig = go.Figure()
    df_filtered = df_G[
        (df_G['run_type'].isin(selected)) & 
        (df_G['Avg Pace'] <= 30)
    ].copy()
    #for run in color_map.keys():
    for run in selected:
        #df_sub = df_G[df_G['run_type'] == run]
        df_sub = df_filtered[df_filtered['run_type'] == run]
        #st.write(f"Run type {run} has {len(df_sub)} rows")
        #st.write(df_sub[['Avg Pace', 'Avg Run Cadence']])
        #Title = df_sub[['Title']].values
        #Time = df_sub[['Time']].values
        #Distance = df_sub[['Distance']].values
        fig.add_trace(go.Scatter(
            x=df_sub['Avg Pace'], y=df_sub['averageRunningCadenceInStepsPerMinute'], mode='markers',
            marker=dict(color=color_map[run]), name=run,
            customdata=df_sub[['activityName', 'startTimeLocal', 'Distance','Avg Pace ori']].values,
            hovertemplate=(
            '%{customdata[0]}<br>Time: %{customdata[1]}<br>Distance: ('+run+') %{customdata[2]} km<br>'
            'Pacing: %{customdata[3]} min/km<br>Cadence: %{y:.0f} spm'
            )
            #hovertemplate='Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm<br>Distance: '+run
            #hovertemplate=Title_val +'<br>Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm<br>Distance: '+run
            #hovertemplate=Title +'<br>Time: ' + Time +'<br>Distance: ('+run+') <br>Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm'
            #hovertemplate=Title + '<br>Time: ' +Time+'<br>Distance: ('+run+')'+str(Distance)+' <br>Pacing: ' + str(Avg_Pace) + ' min/km<br>Cadence: %{y:.0f} spm'
        ))
    fig.update_layout(title='Pacing vs Cadence by Run Distance',
                        xaxis_title='Pacing (min/km)', yaxis_title='Cadence (steps/min)')
    st.plotly_chart(fig, use_container_width=True)

with GC_tab4: #Map - Enhanced with Route Selection
    st.dataframe(df_G[["activityId", "activityName", "startTimeLocal", "distance_km"]])

    sel_id = st.selectbox(
        "Select a run to show map",
        df_G["activityId"].tolist(),
        format_func=lambda x: f" {runs_df.loc[runs_df['activityId'] == x, 'distance_km'].iloc[0]:.1f}km/ " +
                              f"{runs_df.loc[runs_df['activityId'] == x, 'duration'].iloc[0]/60:.1f}min/ " +
                              runs_df.loc[runs_df['activityId'] == x, "activityName"].iloc[0],
        key="run_select",
    )

    if sel_id and api is not None:
        try:
            details = api.get_activity_details(sel_id)
            geo = details.get("geoPolylineDTO")
            
            if not geo:
                st.warning("No GPS polyline available for this activity.")
            else:
                poly = geo.get("polyline", [])
                if not isinstance(poly, list):
                    st.error(f"Unexpected polyline format: {type(poly)}")
                else:
                    # Extract polyline data with metrics
                    pts, elevations, times = extract_polyline_with_metrics(poly)

                    if not pts:
                        st.warning("No GPS track points found.")
                    else:
                        # Create GPS DataFrame
                        gps_df = pd.DataFrame(pts, columns=["lat", "lon"])
                        gps_df["elevation"] = elevations
                        gps_df["timestamp"] = times
                        
                        # Initialize session state for point selection
                        if "start_idx" not in st.session_state:
                            st.session_state.start_idx = 0
                        if "end_idx" not in st.session_state:
                            st.session_state.end_idx = len(gps_df) - 1
                        
                        # Point selection controls
                        st.subheader("📍 Route Segment Selection")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.session_state.start_idx = st.slider(
                                "Start point index",
                                0,
                                len(gps_df) - 2,
                                st.session_state.start_idx,
                                key="start_slider"
                            )
                        
                        with col2:
                            st.session_state.end_idx = st.slider(
                                "End point index",
                                st.session_state.start_idx + 1,
                                len(gps_df),
                                st.session_state.end_idx,
                                key="end_slider"
                            )
                        
                        # Create base map with full route in red
                        fig = px.line_mapbox(
                            gps_df,
                            lat="lat",
                            lon="lon",
                            zoom=guess_zoom_from_bounds(gps_df),
                            height=600,
                        )
                        fig.update_traces(line_color="red", line_width=2, name="Full Route")
                        
                        # Add selected segment in green
                        segment_gps = gps_df.iloc[st.session_state.start_idx:st.session_state.end_idx]
                        fig.add_scattermapbox(
                            lat=segment_gps["lat"],
                            lon=segment_gps["lon"],
                            mode="lines",
                            line=dict(color="green", width=4),
                            name="Selected Segment",
                            hovertext=[f"Point {i}" for i in range(len(segment_gps))],
                            hoverinfo="text",
                        )
                        
                        # Add start point marker (blue)
                        fig.add_scattermapbox(
                            lat=[gps_df.iloc[st.session_state.start_idx]["lat"]],
                            lon=[gps_df.iloc[st.session_state.start_idx]["lon"]],
                            mode="markers",
                            marker=dict(size=12, color="blue"),
                            name="Start Point",
                            hovertext=["START"],
                            showlegend=True,
                        )
                        
                        # Add end point marker (red)
                        fig.add_scattermapbox(
                            lat=[gps_df.iloc[st.session_state.end_idx - 1]["lat"]],
                            lon=[gps_df.iloc[st.session_state.end_idx - 1]["lon"]],
                            mode="markers",
                            marker=dict(size=12, color="red"),
                            name="End Point",
                            hovertext=["END"],
                            showlegend=True,
                        )
                        
                        # Add km markers
                        km_df = add_km_markers(gps_df)
                        if not km_df.empty:
                            fig.add_scattermapbox(
                                lat=km_df["lat"],
                                lon=km_df["lon"],
                                mode="markers+text",
                                marker=dict(size=12, color="white", opacity=0.7),
                                text=km_df["km"].astype(str),
                                textposition="middle center",
                                textfont=dict(color="black", size=9),
                                name="km markers",
                                showlegend=False,
                            )
                        
                        fig.update_layout(
                            mapbox_style="carto-positron",
                            mapbox=dict(
                                zoom=guess_zoom_from_bounds(gps_df),
                                center=dict(
                                    lat=gps_df["lat"].mean(),
                                    lon=gps_df["lon"].mean(),
                                ),
                            ),
                            hovermode="closest",
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=600,
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display segment metrics
                        st.subheader("📊 Segment Metrics")
                        metrics = calculate_segment_metrics(gps_df, st.session_state.start_idx, st.session_state.end_idx)
                        
                        if metrics:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Distance", f"{metrics['distance_km']:.2f} km")
                            
                            with col2:
                                # Calculate time if possible
                                duration_text = metrics['time_display']
                                st.metric("Time", duration_text)
                            
                            with col3:
                                # Calculate average slope
                                if metrics['slopes'] and len(metrics['slopes']) > 0:
                                    try:
                                        slope_vals = [s for s in metrics['slopes'] if pd.notna(s)]
                                        if slope_vals:
                                            avg_slope = np.mean(slope_vals)
                                            st.metric("Avg Slope", f"{avg_slope:.1f}%")
                                        else:
                                            st.metric("Avg Slope", "N/A")
                                    except (TypeError, ValueError):
                                        st.metric("Avg Slope", "N/A")
                                else:
                                    st.metric("Avg Slope", "N/A")
                            
                            # Display elevation profile (slope graph)
                            if "elevation" in metrics['segment'].columns and metrics['segment']["elevation"].notna().any():
                                st.subheader("📈 Elevation Profile")
                                segment_df = metrics['segment'].copy()
                                segment_df['distance_progress'] = segment_df['cum_dist_km']
                                
                                # Create slope graph
                                fig_slope = go.Figure()
                                
                                # Add elevation line (only include non-null elevation data)
                                elev_mask = segment_df["elevation"].notna()
                                if elev_mask.any():
                                    fig_slope.add_trace(go.Scatter(
                                        x=segment_df.loc[elev_mask, 'cum_dist_km'],
                                        y=segment_df.loc[elev_mask, 'elevation'],
                                        mode='lines',
                                        name='Elevation',
                                        fill='tozeroy',
                                        line=dict(color='steelblue'),
                                        hovertemplate='<b>Distance: %{x:.2f} km</b><br>Elevation: %{y:.0f} m<extra></extra>'
                                    ))
                                    
                                    fig_slope.update_layout(
                                        title="Elevation Profile Along Route",
                                        xaxis_title="Distance (km)",
                                        yaxis_title="Elevation (m)",
                                        hovermode='x unified',
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_slope, use_container_width=True)
                        
                        # Hover box with detailed information
                        with st.expander("ℹ️ Route Details", expanded=False):
                            st.write(f"**Total points in this segment:** {len(segment_gps)}")
                            st.write(f"**Start point:** Index {st.session_state.start_idx}")
                            st.write(f"**End point:** Index {st.session_state.end_idx - 1}")
                            
                            if metrics:
                                st.write(f"**Segment distance:** {metrics['distance_km']:.2f} km")
                                if "elevation" in segment_gps.columns:
                                    elev_data = segment_gps["elevation"].dropna()
                                    if len(elev_data) > 0:
                                        st.write(f"**Elevation gain:** {(elev_data.max() - elev_data.min()):.0f} m")

        except Exception as e:
            st.error(f"Error while talking to Garmin: {e!r} ({type(e)})")
        